"""Graph and GraphBatch data structures.

These are the user-facing containers for tensor-aware GNN inputs. They keep
multi-dimensional node and edge features intact (e.g. ``[N, C, H, W]`` for
nodes, ``[E, C_e, H, W]`` for edges) and offer a uniform device/dtype API
on top of clear, eager input validation.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from .graph_utils import (
    add_self_loops as _add_self_loops,
    coalesce_edges as _coalesce_edges,
    is_undirected as _is_undirected,
    make_undirected as _make_undirected,
    remove_self_loops as _remove_self_loops,
    validate_edge_features,
    validate_edge_index,
    validate_edge_weight,
)


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _check_label_tensor(
    name: str,
    tensor: torch.Tensor,
    expected_first_dim: int,
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor or None, got {type(tensor).__name__}"
        )
    if tensor.size(0) != expected_first_dim:
        raise ValueError(
            f"{name} has {tensor.size(0)} entries but expected {expected_first_dim}"
        )
    if tensor.device != device:
        raise ValueError(
            f"{name} device ({tensor.device}) must match "
            f"node_features device ({device})"
        )


def _move(
    tensor: Optional[torch.Tensor],
    *,
    device: Optional[torch.device],
    dtype: Optional[torch.dtype],
    allow_dtype: bool = True,
) -> Optional[torch.Tensor]:
    """Move ``tensor`` to ``device``/``dtype`` if provided.

    ``allow_dtype=False`` skips the dtype change (used for index/label tensors
    whose dtype must not be coerced to a float).
    """
    if tensor is None:
        return None
    kwargs: Dict[str, Any] = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None and allow_dtype and tensor.is_floating_point():
        kwargs["dtype"] = dtype
    if not kwargs:
        return tensor
    return tensor.to(**kwargs)


# --------------------------------------------------------------------------- #
# Graph                                                                        #
# --------------------------------------------------------------------------- #

class Graph:
    r"""A single graph with tensor-shaped node and edge features.

    Args:
        node_features (Tensor): ``[N, ...]`` tensor. Common shapes are
            ``[N, D]`` (vector features), ``[N, C, H, W]`` (image-like) and
            ``[N, C, D, H, W]`` (volumetric — storage only, layers may not
            consume it).
        edge_index (LongTensor, optional): ``[2, E]`` source/destination indices.
        edge_weight (Tensor, optional): ``[E]`` per-edge scalar weight.
        edge_features (Tensor, optional): ``[E, ...]`` per-edge feature tensor
            (vector ``[E, D_e]``, image-like ``[E, C_e, H, W]``, or volumetric
            ``[E, C_e, D, H, W]``).
        node_labels (Tensor, optional): ``[N, ...]`` per-node labels.
        edge_labels (Tensor, optional): ``[E, ...]`` per-edge labels.
        graph_label (Tensor, optional): a graph-level label tensor.
        metadata (dict, optional): arbitrary user data carried alongside the
            graph. Preserved verbatim by ``GraphBatch``.

    The constructor validates every input eagerly and raises ``TypeError`` /
    ``ValueError`` with descriptive messages on mismatched shape, dtype, or
    device. No silent coercion is performed.
    """

    def __init__(
        self,
        node_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
        node_labels: Optional[torch.Tensor] = None,
        edge_labels: Optional[torch.Tensor] = None,
        graph_label: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        # --- node_features ---
        if not isinstance(node_features, torch.Tensor):
            raise TypeError(
                f"node_features must be a torch.Tensor, "
                f"got {type(node_features).__name__}"
            )
        if node_features.dim() < 2:
            raise ValueError(
                f"node_features must have at least 2 dimensions [N, ...] "
                f"(vector or spatial per-node feature), "
                f"got shape {tuple(node_features.shape)}"
            )

        device = node_features.device
        num_nodes = node_features.size(0)

        # --- edge_index ---
        edge_index = validate_edge_index(edge_index, num_nodes, device=device)
        num_edges = 0 if edge_index is None else edge_index.size(1)

        # --- edge_weight ---
        if edge_weight is not None and edge_index is None:
            raise ValueError("edge_weight was provided but edge_index is None")
        edge_weight = validate_edge_weight(edge_weight, num_edges, device=device)

        # --- edge_features ---
        if edge_features is not None and edge_index is None:
            raise ValueError("edge_features were provided but edge_index is None")
        edge_features = validate_edge_features(edge_features, num_edges, device=device)

        # --- node_labels ---
        if node_labels is not None:
            _check_label_tensor("node_labels", node_labels, num_nodes, device)

        # --- edge_labels ---
        if edge_labels is not None:
            if edge_index is None:
                raise ValueError("edge_labels were provided but edge_index is None")
            _check_label_tensor("edge_labels", edge_labels, num_edges, device)

        # --- graph_label ---
        if graph_label is not None:
            if not isinstance(graph_label, torch.Tensor):
                raise TypeError(
                    f"graph_label must be a torch.Tensor or None, "
                    f"got {type(graph_label).__name__}"
                )
            if graph_label.device != device:
                raise ValueError(
                    f"graph_label device ({graph_label.device}) must match "
                    f"node_features device ({device})"
                )

        # --- metadata ---
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError(
                f"metadata must be a dict or None, got {type(metadata).__name__}"
            )

        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.edge_features = edge_features
        self.node_labels = node_labels
        self.edge_labels = edge_labels
        self.graph_label = graph_label
        self.metadata = metadata

    # ----- properties ----------------------------------------------------- #

    @property
    def num_nodes(self) -> int:
        return self.node_features.size(0)

    @property
    def num_edges(self) -> int:
        return 0 if self.edge_index is None else self.edge_index.size(1)

    @property
    def feature_shape(self) -> Tuple[int, ...]:
        return tuple(self.node_features.shape[1:])

    @property
    def edge_feature_shape(self) -> Optional[Tuple[int, ...]]:
        if self.edge_features is None:
            return None
        return tuple(self.edge_features.shape[1:])

    @property
    def has_edges(self) -> bool:
        return self.edge_index is not None and self.edge_index.size(1) > 0

    @property
    def has_edge_weight(self) -> bool:
        return self.edge_weight is not None

    @property
    def has_edge_features(self) -> bool:
        return self.edge_features is not None

    @property
    def device(self) -> torch.device:
        return self.node_features.device

    @property
    def dtype(self) -> torch.dtype:
        return self.node_features.dtype

    # ----- copy / device / dtype ----------------------------------------- #

    def clone(self) -> "Graph":
        """Deep-copy: every tensor is cloned and metadata is deep-copied."""
        return Graph(
            node_features=self.node_features.clone(),
            edge_index=self.edge_index.clone() if self.edge_index is not None else None,
            edge_weight=self.edge_weight.clone() if self.edge_weight is not None else None,
            edge_features=self.edge_features.clone() if self.edge_features is not None else None,
            node_labels=self.node_labels.clone() if self.node_labels is not None else None,
            edge_labels=self.edge_labels.clone() if self.edge_labels is not None else None,
            graph_label=self.graph_label.clone() if self.graph_label is not None else None,
            metadata=copy.deepcopy(self.metadata) if self.metadata is not None else None,
        )

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Graph":
        """Move all tensor fields. ``dtype`` only applies to floating tensors."""
        if isinstance(device, str):
            device = torch.device(device)
        self.node_features = _move(self.node_features, device=device, dtype=dtype)
        self.edge_index = _move(self.edge_index, device=device, dtype=dtype, allow_dtype=False)
        self.edge_weight = _move(self.edge_weight, device=device, dtype=dtype)
        self.edge_features = _move(self.edge_features, device=device, dtype=dtype)
        self.node_labels = _move(self.node_labels, device=device, dtype=dtype, allow_dtype=False)
        self.edge_labels = _move(self.edge_labels, device=device, dtype=dtype, allow_dtype=False)
        self.graph_label = _move(self.graph_label, device=device, dtype=dtype, allow_dtype=False)
        return self

    def cpu(self) -> "Graph":
        return self.to(device=torch.device("cpu"))

    def cuda(self, device: Optional[Union[int, str, torch.device]] = None) -> "Graph":
        if device is None:
            return self.to(device=torch.device("cuda"))
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        return self.to(device=device)

    # ----- validation re-run --------------------------------------------- #

    def validate(self) -> "Graph":
        """Re-run input validation against the current state.

        Useful after manual mutation of attributes. Raises with a descriptive
        message on the first inconsistency found.
        """
        if not isinstance(self.node_features, torch.Tensor):
            raise TypeError("node_features is no longer a torch.Tensor")
        if self.node_features.dim() < 2:
            raise ValueError(
                f"node_features must have at least 2 dimensions, "
                f"got shape {tuple(self.node_features.shape)}"
            )
        device = self.node_features.device
        validate_edge_index(self.edge_index, self.num_nodes, device=device)
        validate_edge_weight(self.edge_weight, self.num_edges, device=device)
        validate_edge_features(self.edge_features, self.num_edges, device=device)
        if self.node_labels is not None:
            _check_label_tensor("node_labels", self.node_labels, self.num_nodes, device)
        if self.edge_labels is not None:
            _check_label_tensor("edge_labels", self.edge_labels, self.num_edges, device)
        if self.graph_label is not None and self.graph_label.device != device:
            raise ValueError(
                f"graph_label device ({self.graph_label.device}) must match "
                f"node_features device ({device})"
            )
        return self

    # ----- topology ------------------------------------------------------ #

    def is_undirected(self) -> bool:
        return _is_undirected(self.edge_index, num_nodes=self.num_nodes)

    def is_directed(self) -> bool:
        return not self.is_undirected()

    def add_self_loops(self, fill_value: float = 1.0) -> "Graph":
        """Add ``i->i`` self-loops in place (skipping nodes that already have one).

        Raises if ``edge_labels`` is set, since label assignment for new
        loops is ambiguous — drop or detach ``edge_labels`` first.
        """
        if self.edge_labels is not None:
            raise ValueError(
                "Graph.add_self_loops() is unsupported when edge_labels is set. "
                "Clear edge_labels (set to None) and re-attach them after the "
                "operation if needed."
            )
        new_ei, new_w, new_ef = _add_self_loops(
            self.edge_index,
            self.edge_weight,
            self.edge_features,
            num_nodes=self.num_nodes,
            fill_value=fill_value,
            device=self.device,
        )
        self.edge_index = new_ei
        self.edge_weight = new_w
        self.edge_features = new_ef
        return self

    def remove_self_loops(self) -> "Graph":
        """Remove self-loops in place; per-edge tensors are filtered by the same mask."""
        new_ei, new_w, new_ef, new_el = _remove_self_loops(
            self.edge_index,
            self.edge_weight,
            self.edge_features,
            self.edge_labels,
        )
        self.edge_index = new_ei
        self.edge_weight = new_w
        self.edge_features = new_ef
        self.edge_labels = new_el
        return self

    def make_undirected(self, reduce: str = "mean") -> "Graph":
        """Symmetrize edges in place. Coalesces duplicates with ``reduce``.

        Raises if ``edge_labels`` is set — symmetric duplicates would have
        ambiguous labels under arithmetic reduction.
        """
        if self.edge_labels is not None:
            raise ValueError(
                "Graph.make_undirected() is unsupported when edge_labels is set "
                "because reverse-edge label assignment is ambiguous. "
                "Clear edge_labels (set to None) and re-attach them after the "
                "operation if needed."
            )
        new_ei, new_w, new_ef = _make_undirected(
            self.edge_index,
            self.edge_weight,
            self.edge_features,
            num_nodes=self.num_nodes,
            reduce=reduce,
        )
        self.edge_index = new_ei
        self.edge_weight = new_w
        self.edge_features = new_ef
        return self

    def coalesce(self, reduce: str = "mean") -> "Graph":
        """Sort edges by ``(src, dst)`` and merge duplicates in place."""
        new_ei, new_w, new_ef = _coalesce_edges(
            self.edge_index,
            self.edge_weight,
            self.edge_features,
            num_nodes=self.num_nodes,
            reduce=reduce,
        )
        self.edge_index = new_ei
        self.edge_weight = new_w
        self.edge_features = new_ef
        return self

    # ----- repr ---------------------------------------------------------- #

    def __repr__(self) -> str:
        parts = [
            f"num_nodes={self.num_nodes}",
            f"num_edges={self.num_edges}",
            f"feature_shape={self.feature_shape}",
        ]
        if self.has_edge_weight:
            parts.append("edge_weight=True")
        if self.has_edge_features:
            parts.append(f"edge_feature_shape={self.edge_feature_shape}")
        if self.node_labels is not None:
            parts.append("node_labels=True")
        if self.edge_labels is not None:
            parts.append("edge_labels=True")
        if self.graph_label is not None:
            parts.append("graph_label=True")
        return f"Graph({', '.join(parts)})"


# --------------------------------------------------------------------------- #
# GraphBatch                                                                   #
# --------------------------------------------------------------------------- #

class GraphBatch:
    r"""Batch of :class:`Graph` objects concatenated into a single super-graph.

    All graphs in the batch must share the same per-node feature shape
    (``node_features.shape[1:]``). If any graph has an optional per-edge or
    per-node tensor (``edge_weight``, ``edge_features``, ``edge_labels``,
    ``node_labels``, ``graph_label``), every graph that has edges (or nodes,
    respectively) must provide that tensor with a matching trailing shape —
    otherwise a ``ValueError`` is raised. Mixing-some-with-none is rejected
    rather than silently dropped, because silently dropping per-edge data
    is a footgun.

    Attributes:
        graphs: the list of source graphs (kept for traceability).
        node_features: ``[N_total, ...]``.
        edge_index: ``[2, E_total]`` with per-graph index offsets applied.
        edge_weight: ``[E_total]`` or ``None``.
        edge_features: ``[E_total, ...]`` or ``None``.
        node_labels: ``[N_total, ...]`` or ``None``.
        edge_labels: ``[E_total, ...]`` or ``None``.
        graph_labels: ``[B, ...]`` stacked from per-graph ``graph_label`` or ``None``.
        metadata: ``list[Any]`` of length ``B``, one entry per graph (the original
            ``Graph.metadata``, possibly ``None``).
        batch: ``[N_total]`` LongTensor mapping each node to its graph index.
    """

    def __init__(self, graphs: List[Graph]) -> None:
        if not graphs:
            raise ValueError("Cannot create GraphBatch from an empty list of graphs")
        self._validate_compatibility(graphs)
        self.graphs = graphs
        (
            self.node_features,
            self.edge_index,
            self.edge_weight,
            self.edge_features,
            self.node_labels,
            self.edge_labels,
            self.graph_labels,
            self.metadata,
            self.batch,
        ) = self._batch_graphs(graphs)

    # ----- validation ---------------------------------------------------- #

    @staticmethod
    def _validate_compatibility(graphs: List[Graph]) -> None:
        ref_shape = graphs[0].node_features.shape[1:]
        for i, g in enumerate(graphs):
            actual = g.node_features.shape[1:]
            if actual != ref_shape:
                raise ValueError(
                    f"Cannot batch graph {i} (per-node feature shape {tuple(actual)}) "
                    f"with graph 0 (per-node feature shape {tuple(ref_shape)}). "
                    f"All graphs in a batch must share the same per-node feature shape. "
                    f"Consider resizing or padding node features before batching."
                )

        ef_ref: Optional[Tuple[int, ...]] = None
        for g in graphs:
            if g.edge_features is not None:
                ef_ref = tuple(g.edge_features.shape[1:])
                break
        if ef_ref is not None:
            for i, g in enumerate(graphs):
                if g.edge_features is not None and tuple(g.edge_features.shape[1:]) != ef_ref:
                    raise ValueError(
                        f"Cannot batch graph {i} (per-edge feature shape "
                        f"{tuple(g.edge_features.shape[1:])}) with reference "
                        f"per-edge feature shape {ef_ref}."
                    )

    # ----- batching ------------------------------------------------------ #

    def _batch_graphs(
        self, graphs: List[Graph]
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[Any],
        torch.Tensor,
    ]:
        needs_w = any(g.edge_weight is not None for g in graphs)
        needs_ef = any(g.edge_features is not None for g in graphs)
        needs_el = any(g.edge_labels is not None for g in graphs)
        needs_nl = any(g.node_labels is not None for g in graphs)
        needs_gl = any(g.graph_label is not None for g in graphs)

        node_features_list: List[torch.Tensor] = []
        edge_index_list: List[torch.Tensor] = []
        edge_weight_list: List[torch.Tensor] = []
        edge_features_list: List[torch.Tensor] = []
        node_labels_list: List[torch.Tensor] = []
        edge_labels_list: List[torch.Tensor] = []
        graph_labels_list: List[torch.Tensor] = []
        metadata_list: List[Any] = []
        batch_list: List[torch.Tensor] = []
        node_offset = 0

        for i, g in enumerate(graphs):
            N = g.node_features.size(0)
            node_features_list.append(g.node_features)
            batch_list.append(
                torch.full((N,), i, dtype=torch.long, device=g.node_features.device)
            )

            if needs_nl:
                if g.node_labels is None:
                    raise ValueError(
                        f"Graph {i} is missing node_labels but other graphs in "
                        f"the batch have them. Provide node_labels on every "
                        f"graph or none."
                    )
                node_labels_list.append(g.node_labels)

            if needs_gl:
                if g.graph_label is None:
                    raise ValueError(
                        f"Graph {i} is missing graph_label but other graphs in "
                        f"the batch have one. Provide graph_label on every "
                        f"graph or none."
                    )
                graph_labels_list.append(g.graph_label)

            metadata_list.append(g.metadata)

            if g.edge_index is None:
                if needs_w or needs_ef or needs_el:
                    # A graph with no edges contributes nothing per-edge — no
                    # check required against per-edge optional fields.
                    pass
                node_offset += N
                continue

            edge_index_list.append(g.edge_index + node_offset)
            E_g = g.edge_index.size(1)

            if needs_w:
                if g.edge_weight is None:
                    raise ValueError(
                        f"Graph {i} has edges but no edge_weight, while other "
                        f"graphs in the batch provide edge_weight. Provide "
                        f"edge_weight on every graph with edges or none."
                    )
                edge_weight_list.append(g.edge_weight)
            if needs_ef:
                if g.edge_features is None:
                    raise ValueError(
                        f"Graph {i} has edges but no edge_features, while other "
                        f"graphs in the batch provide edge_features."
                    )
                edge_features_list.append(g.edge_features)
            if needs_el:
                if g.edge_labels is None:
                    raise ValueError(
                        f"Graph {i} has edges but no edge_labels, while other "
                        f"graphs in the batch provide edge_labels."
                    )
                edge_labels_list.append(g.edge_labels)

            # Consistency: optional per-edge tensors must match E_g (caught
            # already by Graph.__init__, but assert here for defensive batching).
            if g.edge_weight is not None and g.edge_weight.size(0) != E_g:
                raise ValueError(
                    f"Graph {i}: edge_weight length {g.edge_weight.size(0)} != "
                    f"num_edges {E_g}"
                )

            node_offset += N

        node_features = torch.cat(node_features_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else None
        edge_weight = torch.cat(edge_weight_list, dim=0) if edge_weight_list else None
        edge_features = torch.cat(edge_features_list, dim=0) if edge_features_list else None
        edge_labels = torch.cat(edge_labels_list, dim=0) if edge_labels_list else None
        node_labels = torch.cat(node_labels_list, dim=0) if node_labels_list else None

        graph_labels: Optional[torch.Tensor] = None
        if graph_labels_list:
            try:
                graph_labels = torch.stack(graph_labels_list, dim=0)
            except RuntimeError as e:
                raise ValueError(
                    "Failed to stack graph_label across the batch — every graph "
                    f"must have the same graph_label shape. Underlying error: {e}"
                ) from e

        batch = torch.cat(batch_list, dim=0)
        return (
            node_features,
            edge_index,
            edge_weight,
            edge_features,
            node_labels,
            edge_labels,
            graph_labels,
            metadata_list,
            batch,
        )

    # Keep old name as alias for backward compatibility.
    batch_graphs = _batch_graphs

    # ----- properties ---------------------------------------------------- #

    @property
    def num_graphs(self) -> int:
        return len(self.graphs)

    @property
    def num_nodes(self) -> int:
        return self.node_features.size(0)

    @property
    def num_edges(self) -> int:
        return 0 if self.edge_index is None else self.edge_index.size(1)

    @property
    def feature_shape(self) -> Tuple[int, ...]:
        return tuple(self.node_features.shape[1:])

    @property
    def edge_feature_shape(self) -> Optional[Tuple[int, ...]]:
        if self.edge_features is None:
            return None
        return tuple(self.edge_features.shape[1:])

    @property
    def has_edges(self) -> bool:
        return self.edge_index is not None and self.edge_index.size(1) > 0

    @property
    def has_edge_weight(self) -> bool:
        return self.edge_weight is not None

    @property
    def has_edge_features(self) -> bool:
        return self.edge_features is not None

    @property
    def device(self) -> torch.device:
        return self.node_features.device

    # ----- device / dtype ------------------------------------------------ #

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "GraphBatch":
        if isinstance(device, str):
            device = torch.device(device)
        self.node_features = _move(self.node_features, device=device, dtype=dtype)
        self.edge_index = _move(self.edge_index, device=device, dtype=dtype, allow_dtype=False)
        self.edge_weight = _move(self.edge_weight, device=device, dtype=dtype)
        self.edge_features = _move(self.edge_features, device=device, dtype=dtype)
        self.node_labels = _move(self.node_labels, device=device, dtype=dtype, allow_dtype=False)
        self.edge_labels = _move(self.edge_labels, device=device, dtype=dtype, allow_dtype=False)
        self.graph_labels = _move(self.graph_labels, device=device, dtype=dtype, allow_dtype=False)
        self.batch = _move(self.batch, device=device, dtype=dtype, allow_dtype=False)
        return self

    def cpu(self) -> "GraphBatch":
        return self.to(device=torch.device("cpu"))

    def cuda(self, device: Optional[Union[int, str, torch.device]] = None) -> "GraphBatch":
        if device is None:
            return self.to(device=torch.device("cuda"))
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        return self.to(device=device)

    # ----- repr ---------------------------------------------------------- #

    def __repr__(self) -> str:
        parts = [
            f"num_graphs={self.num_graphs}",
            f"num_nodes={self.num_nodes}",
            f"num_edges={self.num_edges}",
            f"feature_shape={self.feature_shape}",
        ]
        if self.has_edge_weight:
            parts.append("edge_weight=True")
        if self.has_edge_features:
            parts.append(f"edge_feature_shape={self.edge_feature_shape}")
        return f"GraphBatch({', '.join(parts)})"
