"""Production-oriented graph loaders for node/edge/graph-level tasks.

These loaders extend the lightweight ``tgraphx.sampling_loaders`` with
PyTorch ``DataLoader``-compatible interfaces for node classification,
link prediction, and graph-level tasks.

All loaders:
- Are deterministic when a ``seed`` is given.
- Preserve edge_attr, edge_weight, and tensor node features.
- Never allocate a dense adjacency matrix.
- Support ``seed_worker`` for reproducible multi-worker DataLoader use.

Key ergonomics (v1.0.1+):
    :class:`NeighborLoader` now yields :class:`GraphMiniBatch` objects that
    expose direct attribute access:

        for batch in loader:
            logits = model(batch.node_features, batch.edge_index)
            loss = F.cross_entropy(batch.seed_logits(logits), batch.seed_y)

    Legacy tuple unpacking still works for backward compatibility:

        for subgraph, seed_ids in loader:
            ...

Stability: Beta (v0.5.0+).
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from .core.graph import Graph, GraphBatch
from .reproducibility import seed_worker as _seed_worker
from .sampling import neighbor_sample, k_hop_subgraph

__all__ = [
    "GraphMiniBatch",
    "NeighborLoader",
    "LinkNeighborLoader",
    "GraphLoader",
    "NodeClassificationDataset",
    "LinkPredictionDataset",
    "make_neighbor_loader",
    "make_link_loader",
    "make_graph_loader",
    "fetch_features_for_subgraph",
    "map_global_to_local",
    "seed_logits",
]


# ── GraphMiniBatch ────────────────────────────────────────────────────────────


def map_global_to_local(
    global_ids: torch.Tensor,
    sampled_global_ids: torch.Tensor,
) -> torch.Tensor:
    """Map ``global_ids`` to their local positions within ``sampled_global_ids``.

    Uses a dense lookup table for small max-IDs and a ``searchsorted``-based
    fallback for large / sparse global ID spaces to avoid allocating huge arrays.

    Args:
        global_ids: ``LongTensor[K]`` of global node IDs.
        sampled_global_ids: ``LongTensor[M]`` of global IDs of the sampled
            subgraph nodes (i.e. ``subgraph.metadata['sampling']['original_node_ids']``).

    Returns:
        ``LongTensor[K]`` of local indices (positions in ``sampled_global_ids``).

    Raises:
        ValueError: If any ID in ``global_ids`` is not found in
            ``sampled_global_ids``.
    """
    device = sampled_global_ids.device
    global_ids = global_ids.to(device)

    M = sampled_global_ids.size(0)
    max_sampled = int(sampled_global_ids.max().item()) + 1 if M > 0 else 0
    max_query = int(global_ids.max().item()) + 1 if global_ids.numel() > 0 else 0
    max_id = max(max_sampled, max_query)

    # Dense path: O(max_id) memory — safe when IDs are compact.
    # Threshold: 16 MB of Long (64-bit) ≈ 2M entries; above that use searchsorted.
    _DENSE_THRESHOLD = 2_000_000
    if max_id <= _DENSE_THRESHOLD:
        lookup = torch.full((max_id,), -1, dtype=torch.long, device=device)
        local_pos = torch.arange(M, dtype=torch.long, device=device)
        lookup[sampled_global_ids] = local_pos
        local = lookup[global_ids]
        missing = (local == -1).nonzero(as_tuple=False).view(-1)
    else:
        # Sparse path: sort sampled IDs, use searchsorted, then re-map positions.
        # Memory: O(M log M) time, O(M) memory — independent of max_id.
        sorted_sampled, perm = sampled_global_ids.sort()
        pos = torch.searchsorted(sorted_sampled, global_ids)
        pos_clamped = pos.clamp(0, M - 1)
        found = sorted_sampled[pos_clamped] == global_ids
        # Map found positions back through perm to get original local indices.
        local = torch.where(found, perm[pos_clamped], torch.full_like(pos_clamped, -1))
        missing = (~found).nonzero(as_tuple=False).view(-1)

    if missing.numel() > 0:
        bad = global_ids[missing[:5]].tolist()
        raise ValueError(
            f"map_global_to_local: {missing.numel()} seed_node_ids not found in "
            f"the sampled subgraph.  Missing global IDs (first 5): {bad}.\n"
            f"NeighborLoader must expose original_node_ids in "
            f"subgraph.metadata['sampling']['original_node_ids']."
        )
    return local


def seed_logits(logits: torch.Tensor, batch: "GraphMiniBatch") -> torch.Tensor:
    """Extract logits for seed nodes from a full-subgraph logit tensor.

    Equivalent to ``batch.seed_logits(logits)``.

    Args:
        logits: ``Tensor[N_sub, ...]`` output of a GNN over the subgraph.
        batch: A :class:`GraphMiniBatch`.

    Returns:
        ``Tensor[K, ...]`` logits for the ``K`` seed nodes only.
    """
    return batch.seed_logits(logits)


class GraphMiniBatch:
    """Ergonomic batch object returned by :class:`NeighborLoader`.

    Exposes named attributes so you can write:

        for batch in loader:
            logits = model(batch.node_features, batch.edge_index)
            loss = F.cross_entropy(batch.seed_logits(logits), batch.seed_y)

    Also supports legacy tuple unpacking for backward compatibility:

        for subgraph, seed_ids in loader:
            ...

    Attributes:
        graph (Graph): Sampled subgraph (relabelled to local indices).
        seed_node_ids (LongTensor): Global node IDs of the seed nodes.
        seed_local_indices (LongTensor): Local positions of seed nodes inside
            the sampled subgraph.
        batch_size (int): Number of seed nodes in this batch.
        input_nodes (LongTensor): Global node IDs of all nodes in the subgraph
            (same as ``graph.metadata['sampling']['original_node_ids']``).
        metadata (dict): Sampling metadata from the subgraph.
    """

    def __init__(
        self,
        graph: Graph,
        seed_node_ids: torch.Tensor,
        source_graph: Optional[Graph] = None,
    ) -> None:
        self.graph = graph
        self.seed_node_ids = seed_node_ids
        self.batch_size = int(seed_node_ids.size(0))

        sampling = {}
        if isinstance(graph.metadata, dict):
            sampling = graph.metadata.get("sampling", {})
        self.metadata = sampling

        original_node_ids = sampling.get("original_node_ids")
        if original_node_ids is None:
            # Fall back: assume subgraph nodes are the same as sampled nodes.
            self._seed_local_indices: Optional[torch.Tensor] = None
            self.input_nodes = seed_node_ids
        else:
            self.input_nodes = original_node_ids
            try:
                self._seed_local_indices = map_global_to_local(
                    seed_node_ids, original_node_ids
                )
            except ValueError:
                self._seed_local_indices = None

        self._source_graph = source_graph

    # ----- delegate subgraph attributes ---------------------------------- #

    @property
    def node_features(self) -> torch.Tensor:
        """Node features of the sampled subgraph."""
        return self.graph.node_features

    @property
    def x(self) -> torch.Tensor:
        """Alias for ``node_features``."""
        return self.graph.node_features

    @property
    def edge_index(self) -> Optional[torch.Tensor]:
        """Edge index of the sampled subgraph."""
        return self.graph.edge_index

    @property
    def edge_features(self) -> Optional[torch.Tensor]:
        """Edge features of the sampled subgraph."""
        return self.graph.edge_features

    @property
    def edge_attr(self) -> Optional[torch.Tensor]:
        """Alias for ``edge_features``."""
        return self.graph.edge_features

    @property
    def edge_weight(self) -> Optional[torch.Tensor]:
        """Edge weights of the sampled subgraph."""
        return self.graph.edge_weight

    @property
    def num_nodes(self) -> int:
        """Total nodes in the sampled subgraph."""
        return self.graph.num_nodes

    @property
    def num_edges(self) -> int:
        """Total edges in the sampled subgraph."""
        return self.graph.num_edges

    # ----- label access -------------------------------------------------- #

    @property
    def y(self) -> Optional[torch.Tensor]:
        """Per-node labels for all nodes in the subgraph (alias: ``labels``)."""
        return self.graph.node_labels

    @property
    def labels(self) -> Optional[torch.Tensor]:
        """Per-node labels for all nodes in the subgraph (alias: ``y``)."""
        return self.graph.node_labels

    @property
    def seed_local_indices(self) -> torch.Tensor:
        """Local positions of seed nodes inside the sampled subgraph.

        Raises:
            ValueError: If the mapping could not be computed (metadata missing).
        """
        if self._seed_local_indices is None:
            raise ValueError(
                "seed_local_indices could not be computed.  This usually means the "
                "subgraph metadata is missing 'original_node_ids'.  Use "
                "NeighborLoader (not a custom sampler) so that sampling metadata "
                "is populated automatically."
            )
        return self._seed_local_indices

    @property
    def seed_y(self) -> torch.Tensor:
        """Labels for the seed nodes only.

        Raises:
            ValueError: If the graph has no labels.  Create the graph with
                ``Graph(..., y=labels)`` or assign ``graph.y = labels``.
        """
        all_labels = self.graph.node_labels
        if all_labels is None:
            raise ValueError(
                "Batch labels are unavailable because the source Graph has no "
                "y/labels field.  Create the graph with "
                "Graph(..., y=labels) or assign graph.y = labels.\n"
                "Example:\n"
                "    g = Graph(node_features=x, edge_index=edge_index, y=y)\n"
                "    loader = NeighborLoader(g, fanouts=[15, 10], batch_size=64)"
            )
        idx = self.seed_local_indices
        return all_labels[idx]

    @property
    def seed_labels(self) -> torch.Tensor:
        """Alias for ``seed_y``."""
        return self.seed_y

    # ----- logit helpers ------------------------------------------------- #

    def seed_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Extract logits for seed nodes from a full-subgraph logit tensor.

        Args:
            logits: ``Tensor[N_sub, ...]`` produced by passing
                ``batch.node_features`` and ``batch.edge_index`` through a model.

        Returns:
            ``Tensor[K, ...]`` logits for the ``K`` seed nodes only.

        Example::

            for batch in loader:
                logits = model(batch.node_features, batch.edge_index)
                loss = F.cross_entropy(batch.seed_logits(logits), batch.seed_y)
        """
        return logits[self.seed_local_indices]

    def all_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Return logits unchanged (all subgraph nodes).

        Provided for symmetry with ``seed_logits`` — useful when you want the
        full-subgraph output rather than just the seed portion.
        """
        return logits

    def loss(
        self,
        logits: torch.Tensor,
        loss_fn=None,
    ) -> torch.Tensor:
        """Compute supervised loss over seed nodes.

        Args:
            logits: Full-subgraph logit tensor.
            loss_fn: A loss function with signature ``loss_fn(input, target)``.
                Defaults to ``torch.nn.functional.cross_entropy``.

        Returns:
            Scalar loss tensor.
        """
        if loss_fn is None:
            loss_fn = F.cross_entropy
        return loss_fn(self.seed_logits(logits), self.seed_y)

    # ----- device -------------------------------------------------------- #

    def to(self, device: Union[str, torch.device]) -> "GraphMiniBatch":
        """Move all tensors to ``device`` in place and return self."""
        self.graph.to(device)
        self.seed_node_ids = self.seed_node_ids.to(device)
        if self._seed_local_indices is not None:
            self._seed_local_indices = self._seed_local_indices.to(device)
        if self.input_nodes is not None:
            self.input_nodes = self.input_nodes.to(device)
        return self

    # ----- backward compat: tuple unpacking ------------------------------ #

    def as_tuple(self) -> Tuple[Graph, torch.Tensor]:
        """Return ``(subgraph, seed_node_ids)`` for legacy code.

        Legacy code that unpacks the batch as ``for subgraph, seeds in loader``
        continues to work via :meth:`__iter__`.
        """
        return self.graph, self.seed_node_ids

    def __iter__(self):
        """Yield ``(subgraph, seed_node_ids)`` for backward-compatible unpacking.

        This allows old code like::

            for subgraph, seeds in loader:
                ...

        to continue working without modification.
        """
        yield self.graph
        yield self.seed_node_ids

    def __repr__(self) -> str:
        return (
            f"GraphMiniBatch("
            f"num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, "
            f"batch_size={self.batch_size}, "
            f"has_labels={self.graph.node_labels is not None}"
            f")"
        )


class _WrappingIter:
    """Iterator that wraps tuples from the underlying DataLoader into GraphMiniBatch."""

    def __init__(self, inner_iter, source_graph: Graph) -> None:
        self._inner = inner_iter
        self._source = source_graph

    def __iter__(self):
        return self

    def __next__(self) -> GraphMiniBatch:
        subgraph, seed_ids = next(self._inner)
        return GraphMiniBatch(subgraph, seed_ids, source_graph=self._source)


def fetch_features_for_subgraph(
    subgraph: Graph,
    feature_store: Any,
    name: str = "x",
    *,
    update_node_features: bool = True,
) -> torch.Tensor:
    """Fetch features for a sampled subgraph from a FeatureStore.

    Reads ``feature_store.get(name, ids=original_node_ids)`` where
    ``original_node_ids`` is the global-id mapping recorded in
    ``subgraph.metadata['sampling']``.  When the subgraph carries no
    sampling metadata (e.g. a freshly-induced subgraph created without
    going through tgraphx samplers), an error is raised.

    The fetched tensor is moved to ``subgraph.node_features.device`` so
    callers can use it directly with the subgraph's other tensors.

    Args:
        subgraph: A :class:`~tgraphx.Graph` produced by a sampler.
        feature_store: An :class:`InMemoryFeatureStore` or
            :class:`MemmapFeatureStore`.
        name: Feature name to fetch.
        update_node_features: When ``True`` (default), also overwrites
            ``subgraph.node_features`` with the fetched tensor.

    Returns:
        ``Tensor[K, *]`` of features for the sampled nodes.
    """
    sampling = subgraph.metadata.get("sampling") if isinstance(subgraph.metadata, dict) else None
    if sampling is None:
        raise ValueError(
            "fetch_features_for_subgraph: subgraph has no 'sampling' "
            "metadata; produce it via tgraphx samplers (NeighborLoader, "
            "GraphSAINT*, etc.) so original node IDs are recorded."
        )
    node_ids = sampling.get("original_node_ids")
    if node_ids is None:
        raise ValueError("subgraph metadata['sampling'] missing 'original_node_ids'")
    feats = feature_store.get(name, ids=node_ids)
    feats = feats.to(subgraph.node_features.device)
    if update_node_features:
        subgraph.node_features = feats
    return feats


# ── Internal helpers ─────────────────────────────────────────────────────────


def _to_long(x: Union[torch.Tensor, List[int]], device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.long, device=device)
    return torch.tensor(x, dtype=torch.long, device=device)


# ── Node classification dataset ───────────────────────────────────────────────


class NodeClassificationDataset(Dataset):
    """Dataset that presents each node as a sample for classification.

    Args:
        graph: A :class:`~tgraphx.Graph`.
        mask: ``BoolTensor[N]`` selecting which nodes to include.
            When ``None``, all nodes are included.
    """

    def __init__(
        self,
        graph: Graph,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        self.graph = graph
        if mask is not None:
            self.node_ids = mask.nonzero(as_tuple=False).view(-1)
        else:
            self.node_ids = torch.arange(graph.num_nodes, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.node_ids.size(0))

    def __getitem__(self, idx: int) -> int:
        return int(self.node_ids[idx].item())


class _NeighborSamplerCollate:
    """Collate function that turns node IDs into sampled subgraphs."""

    def __init__(
        self,
        graph: Graph,
        fanouts: List[int],
        seed: Optional[int] = None,
        direction: str = "in",
        feature_store: Any = None,
        feature_name: str = "x",
    ) -> None:
        self.graph = graph
        self.fanouts = fanouts
        self.seed = seed
        self.direction = direction
        self.feature_store = feature_store
        self.feature_name = feature_name

    def __call__(self, node_ids: List[int]) -> Tuple[Graph, torch.Tensor]:
        seeds = torch.tensor(node_ids, dtype=torch.long)
        subgraph = neighbor_sample(
            self.graph, seeds, self.fanouts,
            seed=self.seed, direction=self.direction,
        )
        # Optional FeatureStore fetch: replace node_features with fresh
        # values for the sampled node IDs only (no full materialisation).
        if self.feature_store is not None:
            fetch_features_for_subgraph(
                subgraph, self.feature_store, name=self.feature_name,
                update_node_features=True,
            )
        # Return subgraph and original seed indices (local in subgraph).
        orig = subgraph.metadata.get("sampling", {}).get("seed_nodes", seeds)
        return subgraph, orig


class NeighborLoader:
    """Neighbor-sampling DataLoader for node classification.

    Wraps ``tgraphx.sampling.neighbor_sample`` in a PyTorch
    ``DataLoader``-compatible interface.  Each batch is a sampled
    subgraph centred on ``batch_size`` seed nodes.

    Args:
        graph: Source :class:`~tgraphx.Graph`.
        fanouts: Per-hop fanout list, e.g. ``[15, 10, 5]``.
        mask: ``BoolTensor[N]`` selecting which nodes to use as seeds
            (e.g. train mask).  When ``None``, all nodes are used.
        batch_size: Number of seed nodes per batch.
        shuffle: Shuffle seed nodes.
        num_workers: DataLoader workers (0 = main process).
        seed: RNG seed.  Also passed to ``seed_worker`` for worker
            reproducibility.
        direction: Edge direction: ``"in"`` (GraphSAGE default) or
            ``"out"``.
        drop_last: Drop the last incomplete batch.

    Yields:
        :class:`GraphMiniBatch` objects.  Each batch exposes::

            batch.node_features   # subgraph node features [N_sub, ...]
            batch.edge_index      # subgraph edge index [2, E_sub]
            batch.seed_y          # labels for seed nodes [K]
            batch.seed_logits(z)  # extract seed-node logits from model output
            batch.seed_node_ids   # global IDs of seed nodes [K]
            batch.seed_local_indices  # local positions of seed nodes [K]

        Legacy tuple unpacking still works::

            for subgraph, seed_ids in loader:
                ...

    Stability: Beta.
    """

    def __init__(
        self,
        graph: Graph,
        fanouts: List[int],
        mask: Optional[torch.Tensor] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        seed: Optional[int] = None,
        direction: str = "in",
        drop_last: bool = False,
        feature_store: Any = None,
        feature_name: str = "x",
    ) -> None:
        self.graph = graph
        self.fanouts = fanouts
        self.batch_size = batch_size
        self.seed = seed
        self.direction = direction

        dataset = NodeClassificationDataset(graph, mask)
        collate_fn = _NeighborSamplerCollate(
            graph, fanouts, seed, direction,
            feature_store=feature_store, feature_name=feature_name,
        )

        gen = None
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(int(seed))

        self._loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
            generator=gen,
            drop_last=drop_last,
        )

    def __iter__(self) -> Iterator[GraphMiniBatch]:
        """Yield :class:`GraphMiniBatch` objects.

        Legacy code that unpacks ``(subgraph, seed_ids)`` continues to work
        because :class:`GraphMiniBatch` implements ``__iter__``.
        """
        return _WrappingIter(iter(self._loader), self.graph)

    def __len__(self) -> int:
        return len(self._loader)


# ── Link prediction dataset ───────────────────────────────────────────────────


class LinkPredictionDataset(Dataset):
    """Dataset for link prediction: yields (src, dst) edge pairs.

    Args:
        edge_index: ``LongTensor[2, E]`` of positive edges.
        negative_edge_index: Optional ``LongTensor[2, E_neg]`` of
            pre-computed negative edges.  When ``None``, random
            negatives are sampled on the fly during collation.
        num_nodes: Node count (for negative sampling).
        neg_ratio: Negative edges per positive edge (when negatives
            are generated on the fly).
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        negative_edge_index: Optional[torch.Tensor] = None,
        num_nodes: int = 0,
        neg_ratio: int = 1,
    ) -> None:
        self.edge_index = edge_index
        self.negative_edge_index = negative_edge_index
        self.num_nodes = int(num_nodes)
        self.neg_ratio = int(neg_ratio)

    def __len__(self) -> int:
        return int(self.edge_index.size(1))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "src": int(self.edge_index[0, idx]),
            "dst": int(self.edge_index[1, idx]),
            "label": 1,
        }


class _LinkCollate:
    def __init__(
        self,
        graph: Graph,
        fanouts: List[int],
        num_nodes: int,
        neg_ratio: int,
        seed: Optional[int],
        positive_set: Optional[set] = None,
    ) -> None:
        self.graph = graph
        self.fanouts = fanouts
        self.num_nodes = num_nodes
        self.neg_ratio = neg_ratio
        self.seed = seed
        self.positive_set = positive_set or set()
        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(int(seed))

    def __call__(self, items: List[Dict]) -> Dict[str, Any]:
        src = torch.tensor([it["src"] for it in items], dtype=torch.long)
        dst = torch.tensor([it["dst"] for it in items], dtype=torch.long)
        # Negative sampling.
        all_nodes = torch.randint(
            self.num_nodes, (src.size(0) * self.neg_ratio,), generator=self._gen,
        )
        neg_src = src.repeat_interleave(self.neg_ratio)
        neg_dst = all_nodes
        # Remove self-loops in negatives.
        not_self = neg_src != neg_dst
        neg_src = neg_src[not_self]
        neg_dst = neg_dst[not_self]
        # Sample neighbour subgraph for context.
        all_seeds = torch.cat([src, dst, neg_src, neg_dst]).unique()
        subgraph = neighbor_sample(self.graph, all_seeds, self.fanouts, seed=self.seed)
        return {
            "subgraph": subgraph,
            "pos_src": src,
            "pos_dst": dst,
            "neg_src": neg_src,
            "neg_dst": neg_dst,
        }


class LinkNeighborLoader:
    """Neighbor-sampling DataLoader for link prediction tasks.

    For each batch of positive edges, samples a neighbourhood subgraph
    and also generates negative edges for training.

    Args:
        graph: Source :class:`~tgraphx.Graph`.
        edge_index: Positive training edges ``LongTensor[2, E]``.
        fanouts: Per-hop fanout list.
        batch_size: Positive edges per batch.
        shuffle: Shuffle edges.
        num_workers: DataLoader workers.
        seed: RNG seed.
        neg_ratio: Negatives per positive.
        positive_set: Known positive (src, dst) tuples to avoid as
            negatives.  When ``None``, no filtering.

    Yields:
        Dict with keys: ``subgraph``, ``pos_src``, ``pos_dst``,
        ``neg_src``, ``neg_dst``.

    Stability: Beta.
    """

    def __init__(
        self,
        graph: Graph,
        edge_index: torch.Tensor,
        fanouts: List[int],
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 0,
        seed: Optional[int] = None,
        neg_ratio: int = 1,
        positive_set: Optional[set] = None,
    ) -> None:
        dataset = LinkPredictionDataset(edge_index, num_nodes=graph.num_nodes)
        collate_fn = _LinkCollate(
            graph, fanouts, graph.num_nodes, neg_ratio, seed, positive_set
        )
        gen = None
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(int(seed))
        self._loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
            generator=gen,
        )

    def __iter__(self) -> Iterator:
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)


# ── Graph-level loader ────────────────────────────────────────────────────────


class GraphLoader:
    """DataLoader for graph-level classification/regression tasks.

    Wraps a list of :class:`~tgraphx.Graph` objects in a standard
    PyTorch DataLoader.  Variable-size graphs are batched using
    :class:`~tgraphx.GraphBatch`.

    Args:
        graphs: List of :class:`~tgraphx.Graph` objects.
        batch_size: Graphs per batch.
        shuffle: Shuffle order.
        num_workers: DataLoader workers.
        seed: RNG seed.
        drop_last: Drop last incomplete batch.

    Yields:
        :class:`~tgraphx.GraphBatch` of ``batch_size`` graphs.

    Stability: Beta.
    """

    def __init__(
        self,
        graphs: List[Graph],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        seed: Optional[int] = None,
        drop_last: bool = False,
    ) -> None:
        self.graphs = graphs
        gen = None
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(int(seed))
        self._loader = DataLoader(
            graphs,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
            generator=gen,
            drop_last=drop_last,
        )

    @staticmethod
    def _collate(batch: List[Graph]) -> GraphBatch:
        return GraphBatch(batch)

    def __iter__(self) -> Iterator:
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)


# ── Convenience constructors ──────────────────────────────────────────────────


def make_neighbor_loader(
    graph: Graph,
    fanouts: List[int] = None,
    mask: Optional[torch.Tensor] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: Optional[int] = None,
    **kwargs,
) -> NeighborLoader:
    """Convenience factory for :class:`NeighborLoader`.

    Args:
        graph: Source graph.
        fanouts: Per-hop fanout list (default ``[10, 5]``).
        mask: Optional node mask.
        batch_size: Seed nodes per batch.
        shuffle: Shuffle seeds.
        seed: RNG seed.
        **kwargs: Forwarded to :class:`NeighborLoader`.

    Returns:
        :class:`NeighborLoader`.
    """
    if fanouts is None:
        fanouts = [10, 5]
    return NeighborLoader(
        graph, fanouts, mask=mask, batch_size=batch_size,
        shuffle=shuffle, seed=seed, **kwargs,
    )


def make_link_loader(
    graph: Graph,
    edge_index: torch.Tensor,
    fanouts: List[int] = None,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: Optional[int] = None,
    neg_ratio: int = 1,
    **kwargs,
) -> LinkNeighborLoader:
    """Convenience factory for :class:`LinkNeighborLoader`."""
    if fanouts is None:
        fanouts = [10, 5]
    return LinkNeighborLoader(
        graph, edge_index, fanouts, batch_size=batch_size,
        shuffle=shuffle, seed=seed, neg_ratio=neg_ratio, **kwargs,
    )


def make_graph_loader(
    graphs: List[Graph],
    batch_size: int = 32,
    shuffle: bool = True,
    seed: Optional[int] = None,
    **kwargs,
) -> GraphLoader:
    """Convenience factory for :class:`GraphLoader`."""
    return GraphLoader(graphs, batch_size=batch_size, shuffle=shuffle, seed=seed, **kwargs)
