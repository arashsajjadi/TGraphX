"""Subgraph and neighbor sampling utilities for TGraphX.

All samplers return :class:`~tgraphx.Graph` instances and preserve as much
node/edge metadata as possible (features, weights, edge features, labels,
metadata).  CPU and CUDA tensors are both supported; samplers honour the
device of the input ``Graph``.

API summary
-----------
``induced_subgraph(graph, node_ids, relabel_nodes=True)``
    Keep only nodes in ``node_ids`` and the edges between them.

``edge_subgraph(graph, edge_ids, relabel_nodes=True)``
    Keep only the listed edges (and the nodes they touch).

``k_hop_subgraph(graph, seed_nodes, num_hops, relabel_nodes=True, direction)``
    Expand ``num_hops`` hops from ``seed_nodes``; ``direction`` is
    ``"both"`` (default), ``"in"``, or ``"out"``.

``sample_nodes(graph, num_nodes, seed=None, relabel_nodes=True)``
    Uniform random node subset.

``sample_edges(graph, num_edges, seed=None, relabel_nodes=True)``
    Uniform random edge subset (induces the touched-node subgraph).

``neighbor_sample(graph, seed_nodes, fanouts, seed=None, direction="in",
                  relabel_nodes=True)``
    Multi-layer neighbour sampling with per-layer fanout (one int per
    hop).  Direction controls whether to follow incoming or outgoing
    edges (``"in"`` is the GraphSAGE-style default).

Determinism
-----------
All samplers accept an optional ``seed`` and use a per-call
``torch.Generator`` so they do not affect global RNG state.

Returned graph metadata
-----------------------
The returned :class:`~tgraphx.Graph` has its ``metadata`` extended with a
``sampling`` dict:

.. code-block:: python

    g_sampled.metadata["sampling"] = {
        "kind": "induced_subgraph" | "edge_subgraph" | "k_hop_subgraph"
                 | "sample_nodes" | "sample_edges" | "neighbor_sample",
        "original_node_ids": LongTensor,   # global → local mapping
        "original_edge_ids": LongTensor,   # only when meaningful
        "seed_nodes": LongTensor,          # for k_hop / neighbor_sample
        "fanouts": list[int],              # for neighbor_sample
        "direction": str,                  # for k_hop / neighbor_sample
    }

When ``relabel_nodes=False``, edge_index keeps the original (global) node
ids and the returned graph's ``num_nodes`` is unchanged.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

from .core.graph import Graph

__all__ = [
    "induced_subgraph",
    "edge_subgraph",
    "k_hop_subgraph",
    "sample_nodes",
    "sample_edges",
    "neighbor_sample",
    "random_walk_sample",
]


# ── Internal helpers ─────────────────────────────────────────────────────────

def _to_long_1d(name: str, t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Validate and convert a node/edge id tensor to 1-D LongTensor on device."""
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor; got {type(t)}")
    if t.dim() != 1:
        raise ValueError(f"{name} must be 1-D; got shape {tuple(t.shape)}")
    return t.to(device=device, dtype=torch.long)


def _build_subgraph_from_node_mask(
    graph: Graph,
    node_mask: torch.Tensor,
    relabel_nodes: bool,
    sampling_kind: str,
    extra_metadata: Optional[dict] = None,
    edge_keep_mask: Optional[torch.Tensor] = None,
) -> Graph:
    """Construct a Graph by keeping nodes where ``node_mask`` is True.

    ``edge_keep_mask`` (optional) lets callers further restrict which
    edges to keep among those touching surviving nodes (useful for
    ``edge_subgraph``).  When ``None``, all edges with both endpoints in
    the mask are kept.
    """
    device = graph.node_features.device
    if node_mask.device != device:
        node_mask = node_mask.to(device)

    keep_node_ids = torch.where(node_mask)[0]  # global ids of surviving nodes

    new_features = graph.node_features.index_select(0, keep_node_ids)
    new_node_labels = (
        graph.node_labels.index_select(0, keep_node_ids)
        if graph.node_labels is not None else None
    )

    if graph.edge_index is not None and graph.edge_index.numel() > 0:
        ei = graph.edge_index
        endpoints_kept = node_mask[ei[0]] & node_mask[ei[1]]
        if edge_keep_mask is not None:
            endpoints_kept = endpoints_kept & edge_keep_mask
        keep_edge_ids = torch.where(endpoints_kept)[0]
        new_edge_index = ei.index_select(1, keep_edge_ids)

        new_edge_weight = (
            graph.edge_weight.index_select(0, keep_edge_ids)
            if graph.edge_weight is not None else None
        )
        new_edge_features = (
            graph.edge_features.index_select(0, keep_edge_ids)
            if graph.edge_features is not None else None
        )
        new_edge_labels = (
            graph.edge_labels.index_select(0, keep_edge_ids)
            if graph.edge_labels is not None else None
        )
    else:
        keep_edge_ids = torch.empty(0, dtype=torch.long, device=device)
        new_edge_index = None
        new_edge_weight = None
        new_edge_features = None
        new_edge_labels = None

    if relabel_nodes and new_edge_index is not None and new_edge_index.numel() > 0:
        # Build dense relabel map: global id → local id (or -1 if dropped).
        N = graph.num_nodes
        relabel = torch.full((N,), -1, dtype=torch.long, device=device)
        relabel[keep_node_ids] = torch.arange(
            keep_node_ids.numel(), device=device, dtype=torch.long
        )
        new_edge_index = relabel[new_edge_index]
    elif not relabel_nodes:
        # Keep global ids; effective num_nodes stays at original.
        # Pad node_features back to original size to keep edge_index valid.
        # This is a "view-with-mask" semantics; we materialise dense zeros
        # for non-kept rows.
        N = graph.num_nodes
        full_feat = graph.node_features.new_zeros((N, *graph.node_features.shape[1:]))
        full_feat[keep_node_ids] = new_features
        new_features = full_feat
        if new_node_labels is not None:
            full_lbl = graph.node_labels.new_zeros(
                (N, *graph.node_labels.shape[1:])
            )
            full_lbl[keep_node_ids] = new_node_labels
            new_node_labels = full_lbl

    # Compose metadata.
    base_meta = dict(graph.metadata) if isinstance(graph.metadata, dict) else None
    sampling_meta = {
        "kind": sampling_kind,
        "original_node_ids": keep_node_ids.detach().cpu(),
        "original_edge_ids": keep_edge_ids.detach().cpu(),
    }
    if extra_metadata:
        sampling_meta.update(extra_metadata)
    if base_meta is None:
        base_meta = {}
    base_meta = dict(base_meta)  # don't mutate caller's metadata
    base_meta["sampling"] = sampling_meta

    return Graph(
        node_features=new_features,
        edge_index=new_edge_index,
        edge_weight=new_edge_weight,
        edge_features=new_edge_features,
        node_labels=new_node_labels,
        edge_labels=new_edge_labels,
        graph_label=graph.graph_label,
        metadata=base_meta,
    )


# ── Public API ───────────────────────────────────────────────────────────────

def induced_subgraph(
    graph: Graph,
    node_ids: torch.Tensor,
    relabel_nodes: bool = True,
) -> Graph:
    """Return the subgraph induced by ``node_ids``.

    Args:
        graph: Source :class:`~tgraphx.Graph`.
        node_ids: 1-D LongTensor of unique node indices to keep.
        relabel_nodes: If ``True`` (default), edge_index is remapped to
            consecutive ``[0, K)`` ids matching the surviving nodes; the
            returned graph's ``num_nodes`` equals ``len(node_ids)``.
            If ``False``, edge_index keeps original (global) ids and the
            graph's node feature buffer is right-sized to original ``N``
            with zeros for dropped rows (legal for downstream layers).

    Returns:
        A new :class:`~tgraphx.Graph`.

    Raises:
        ValueError: ``node_ids`` is not 1-D, contains duplicates, or has
            out-of-range entries.
    """
    device = graph.node_features.device
    node_ids = _to_long_1d("node_ids", node_ids, device)
    if node_ids.numel() == 0:
        raise ValueError("node_ids must contain at least one id")
    if (node_ids < 0).any() or (node_ids >= graph.num_nodes).any():
        raise ValueError(
            f"node_ids out of range [0, {graph.num_nodes})"
        )
    if node_ids.unique().numel() != node_ids.numel():
        raise ValueError("node_ids contains duplicates")

    mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=device)
    mask[node_ids] = True
    return _build_subgraph_from_node_mask(
        graph, mask, relabel_nodes, "induced_subgraph",
    )


def edge_subgraph(
    graph: Graph,
    edge_ids: torch.Tensor,
    relabel_nodes: bool = True,
) -> Graph:
    """Return the subgraph that keeps only the listed edges.

    Touched nodes are kept; orphan nodes are dropped (when
    ``relabel_nodes=True``) or left as zero-row placeholders (when
    ``relabel_nodes=False``).

    Args:
        graph: Source graph.
        edge_ids: 1-D LongTensor of unique edge indices to keep.
        relabel_nodes: See :func:`induced_subgraph`.
    """
    device = graph.node_features.device
    if graph.edge_index is None:
        raise ValueError("edge_subgraph requires graph.edge_index to be present")
    edge_ids = _to_long_1d("edge_ids", edge_ids, device)
    if edge_ids.numel() == 0:
        raise ValueError("edge_ids must contain at least one id")
    if (edge_ids < 0).any() or (edge_ids >= graph.num_edges).any():
        raise ValueError(f"edge_ids out of range [0, {graph.num_edges})")
    if edge_ids.unique().numel() != edge_ids.numel():
        raise ValueError("edge_ids contains duplicates")

    edge_mask = torch.zeros(graph.num_edges, dtype=torch.bool, device=device)
    edge_mask[edge_ids] = True
    # Nodes touched by selected edges.
    src = graph.edge_index[0, edge_ids]
    dst = graph.edge_index[1, edge_ids]
    touched = torch.cat([src, dst]).unique()
    node_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=device)
    node_mask[touched] = True

    return _build_subgraph_from_node_mask(
        graph, node_mask, relabel_nodes, "edge_subgraph",
        edge_keep_mask=edge_mask,
    )


def k_hop_subgraph(
    graph: Graph,
    seed_nodes: torch.Tensor,
    num_hops: int,
    relabel_nodes: bool = True,
    direction: str = "both",
) -> Graph:
    """Expand ``num_hops`` hops from ``seed_nodes`` and return the induced subgraph.

    Args:
        graph: Source graph.
        seed_nodes: 1-D LongTensor of seed node ids.
        num_hops: Number of hops (>= 0).  ``num_hops=0`` keeps only seeds.
        relabel_nodes: See :func:`induced_subgraph`.
        direction: ``"both"`` (default), ``"in"`` (follow only incoming
            edges → expand reachable predecessors), or ``"out"``.
    """
    if direction not in ("both", "in", "out"):
        raise ValueError(f"direction must be 'both', 'in', or 'out'; got {direction!r}")
    if num_hops < 0:
        raise ValueError(f"num_hops must be >= 0; got {num_hops}")

    device = graph.node_features.device
    seed_nodes = _to_long_1d("seed_nodes", seed_nodes, device)
    if seed_nodes.numel() == 0:
        raise ValueError("seed_nodes must contain at least one id")
    if (seed_nodes < 0).any() or (seed_nodes >= graph.num_nodes).any():
        raise ValueError(f"seed_nodes out of range [0, {graph.num_nodes})")

    visited = torch.zeros(graph.num_nodes, dtype=torch.bool, device=device)
    visited[seed_nodes] = True

    if graph.edge_index is not None and num_hops > 0:
        ei = graph.edge_index
        for _ in range(num_hops):
            frontier = visited.clone()
            if direction in ("out", "both"):
                # Edges out of frontier: src ∈ frontier ⇒ add dst
                src_in = frontier[ei[0]]
                visited[ei[1, src_in]] = True
            if direction in ("in", "both"):
                # Edges into frontier: dst ∈ frontier ⇒ add src
                dst_in = frontier[ei[1]]
                visited[ei[0, dst_in]] = True

    return _build_subgraph_from_node_mask(
        graph, visited, relabel_nodes, "k_hop_subgraph",
        extra_metadata={
            "seed_nodes": seed_nodes.detach().cpu(),
            "num_hops": num_hops,
            "direction": direction,
        },
    )


def sample_nodes(
    graph: Graph,
    num_nodes: int,
    seed: Optional[int] = None,
    relabel_nodes: bool = True,
) -> Graph:
    """Uniformly sample ``num_nodes`` nodes (without replacement) and induce."""
    if num_nodes < 1:
        raise ValueError(f"num_nodes must be >= 1; got {num_nodes}")
    if num_nodes > graph.num_nodes:
        raise ValueError(
            f"num_nodes={num_nodes} > graph.num_nodes={graph.num_nodes}"
        )
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    perm = torch.randperm(graph.num_nodes, generator=gen)[:num_nodes]
    perm = perm.to(graph.node_features.device)
    return induced_subgraph(graph, perm, relabel_nodes=relabel_nodes)


def sample_edges(
    graph: Graph,
    num_edges: int,
    seed: Optional[int] = None,
    relabel_nodes: bool = True,
) -> Graph:
    """Uniformly sample ``num_edges`` edges and return the touched-node subgraph."""
    if graph.edge_index is None:
        raise ValueError("sample_edges requires graph.edge_index")
    if num_edges < 1:
        raise ValueError(f"num_edges must be >= 1; got {num_edges}")
    if num_edges > graph.num_edges:
        raise ValueError(
            f"num_edges={num_edges} > graph.num_edges={graph.num_edges}"
        )
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    perm = torch.randperm(graph.num_edges, generator=gen)[:num_edges]
    perm = perm.to(graph.node_features.device)
    return edge_subgraph(graph, perm, relabel_nodes=relabel_nodes)


def neighbor_sample(
    graph: Graph,
    seed_nodes: torch.Tensor,
    fanouts: Sequence[int],
    seed: Optional[int] = None,
    direction: str = "in",
    relabel_nodes: bool = True,
) -> Graph:
    """Multi-layer neighbour sampling with per-layer fanout.

    For ``direction="in"`` (GraphSAGE-style): at layer ``l``, every node
    in the current frontier samples up to ``fanouts[l]`` predecessors
    (via incoming edges).  ``direction="out"`` follows outgoing edges.

    Args:
        graph: Source graph.
        seed_nodes: 1-D LongTensor of seed node ids (the inference targets).
        fanouts: Sequence of positive ints; one entry per hop.  An entry
            of ``-1`` means "keep all neighbours" at that layer.
        seed: Optional RNG seed.
        direction: ``"in"`` (default) or ``"out"``.
        relabel_nodes: See :func:`induced_subgraph`.

    Returns:
        :class:`~tgraphx.Graph` containing seeds + sampled neighbourhood.
        ``metadata["sampling"]["seed_nodes"]`` holds the original seed ids
        and ``metadata["sampling"]["fanouts"]`` records the configuration.
    """
    if direction not in ("in", "out"):
        raise ValueError(f"direction must be 'in' or 'out'; got {direction!r}")
    if not fanouts:
        raise ValueError("fanouts must be a non-empty sequence")
    for i, f in enumerate(fanouts):
        if not isinstance(f, int):
            raise TypeError(f"fanouts[{i}] must be int; got {type(f)}")
        if f == 0 or f < -1:
            raise ValueError(f"fanouts[{i}] must be >= 1 or -1 (all); got {f}")
    if graph.edge_index is None:
        raise ValueError("neighbor_sample requires graph.edge_index")

    device = graph.node_features.device
    seed_nodes = _to_long_1d("seed_nodes", seed_nodes, device)
    if seed_nodes.numel() == 0:
        raise ValueError("seed_nodes must contain at least one id")
    if (seed_nodes < 0).any() or (seed_nodes >= graph.num_nodes).any():
        raise ValueError(f"seed_nodes out of range [0, {graph.num_nodes})")

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))

    visited = torch.zeros(graph.num_nodes, dtype=torch.bool, device=device)
    visited[seed_nodes] = True
    frontier = seed_nodes
    ei = graph.edge_index
    # Pre-build per-node neighbour list (CPU-friendly).
    if direction == "in":
        # Incoming: predecessors of dst.
        neighbour_dim_for = 1   # row to filter on (dst)
        return_dim = 0          # row to read (src)
    else:
        neighbour_dim_for = 0
        return_dim = 1

    for layer_idx, fanout in enumerate(fanouts):
        if frontier.numel() == 0:
            break
        # Find edges where the relevant endpoint is in the frontier.
        in_frontier = torch.zeros(graph.num_nodes, dtype=torch.bool, device=device)
        in_frontier[frontier] = True
        edge_mask = in_frontier[ei[neighbour_dim_for]]
        candidate_edges = torch.where(edge_mask)[0]
        if candidate_edges.numel() == 0:
            break
        # Group candidate edges by frontier endpoint.
        endpoint_per_edge = ei[neighbour_dim_for, candidate_edges]
        # For each unique frontier node, sample up to `fanout` candidates.
        new_neighbours_set = set()
        # Iterate per frontier node — small graphs are typical.
        for u in frontier.tolist():
            edges_u = candidate_edges[endpoint_per_edge == u]
            if edges_u.numel() == 0:
                continue
            if fanout == -1 or edges_u.numel() <= fanout:
                picks = edges_u
            else:
                idx = torch.randperm(edges_u.numel(), generator=gen)[:fanout]
                picks = edges_u[idx]
            new_nbr = ei[return_dim, picks].tolist()
            new_neighbours_set.update(new_nbr)
        if not new_neighbours_set:
            break
        new_neighbours = torch.tensor(
            sorted(new_neighbours_set), dtype=torch.long, device=device
        )
        # Update frontier and visited mask.
        novel_mask = ~visited[new_neighbours]
        frontier = new_neighbours[novel_mask]
        visited[new_neighbours] = True

    return _build_subgraph_from_node_mask(
        graph, visited, relabel_nodes, "neighbor_sample",
        extra_metadata={
            "seed_nodes": seed_nodes.detach().cpu(),
            "fanouts": list(fanouts),
            "direction": direction,
        },
    )


def random_walk_sample(
    graph: Graph,
    seed_nodes: torch.Tensor,
    walk_length: int,
    num_walks_per_seed: int = 1,
    direction: str = "out",
    restart_prob: float = 0.0,
    seed: Optional[int] = None,
    relabel_nodes: bool = True,
) -> Graph:
    """Random-walk sampling rooted at ``seed_nodes``.

    From each seed, perform ``num_walks_per_seed`` random walks of length
    ``walk_length``.  At every step the walker transitions uniformly at
    random to one out- or in-neighbour of the current node.  With
    probability ``restart_prob`` the walker resets to its original seed
    (DeepWalk / node2vec-style restart).  The induced subgraph over all
    visited nodes is returned.

    Args:
        graph: Source graph (must contain edges).
        seed_nodes: 1-D LongTensor of seed node ids (one walk root per id).
        walk_length: Number of transition steps per walk (>= 0).
            ``walk_length=0`` keeps only the seeds.
        num_walks_per_seed: Number of independent walks rooted at each
            seed (>= 1).
        direction: ``"out"`` (default) follows outgoing edges,
            ``"in"`` follows incoming edges.
        restart_prob: Probability of returning to the original seed at
            each step (in ``[0, 1)``).
        seed: Optional RNG seed; uses a per-call ``torch.Generator``.
        relabel_nodes: See :func:`induced_subgraph`.

    Returns:
        :class:`~tgraphx.Graph` containing the induced subgraph over the
        union of visited nodes.  ``metadata["sampling"]`` records walk
        configuration and the seed list.

    Notes:
        * Nodes with no outgoing (resp. incoming) neighbours absorb the
          walk for that step — the walker stays in place.  This is a
          standard random-walk convention.
        * Visited node ordering in the returned subgraph follows the
          ``induced_subgraph`` convention (sorted by global id).
    """
    if direction not in ("out", "in"):
        raise ValueError(f"direction must be 'out' or 'in'; got {direction!r}")
    if walk_length < 0:
        raise ValueError(f"walk_length must be >= 0; got {walk_length}")
    if num_walks_per_seed < 1:
        raise ValueError(f"num_walks_per_seed must be >= 1; got {num_walks_per_seed}")
    if not (0.0 <= restart_prob < 1.0):
        raise ValueError(
            f"restart_prob must be in [0, 1); got {restart_prob}"
        )
    if graph.edge_index is None:
        raise ValueError("random_walk_sample requires graph.edge_index")

    device = graph.node_features.device
    seed_nodes = _to_long_1d("seed_nodes", seed_nodes, device)
    if seed_nodes.numel() == 0:
        raise ValueError("seed_nodes must contain at least one id")
    if (seed_nodes < 0).any() or (seed_nodes >= graph.num_nodes).any():
        raise ValueError(f"seed_nodes out of range [0, {graph.num_nodes})")

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))

    ei = graph.edge_index
    if direction == "out":
        # Out-neighbours: for each src in row 0, list its dst (row 1).
        from_row, to_row = 0, 1
    else:
        from_row, to_row = 1, 0

    # Build per-node neighbour lists once (CPU is fine; sampling is sequential).
    from_cpu = ei[from_row].detach().cpu()
    to_cpu = ei[to_row].detach().cpu()
    nbrs: list[list[int]] = [[] for _ in range(graph.num_nodes)]
    for u, v in zip(from_cpu.tolist(), to_cpu.tolist()):
        nbrs[u].append(v)

    visited = torch.zeros(graph.num_nodes, dtype=torch.bool, device=device)
    visited[seed_nodes] = True

    total_walks = int(seed_nodes.numel()) * int(num_walks_per_seed)
    if walk_length > 0 and total_walks > 0:
        # Walk one (seed, walk_idx) at a time.  The walks themselves are
        # short (typical walk_length <= a few hundred); per-walk Python
        # is acceptable and keeps the implementation tractable.
        for seed_id in seed_nodes.tolist():
            for _ in range(num_walks_per_seed):
                cur = seed_id
                for _step in range(walk_length):
                    if (
                        restart_prob > 0.0
                        and torch.rand((), generator=gen).item() < restart_prob
                    ):
                        cur = seed_id
                        continue
                    candidates = nbrs[cur]
                    if not candidates:
                        # Absorb: walker stays put.
                        continue
                    pick = torch.randint(
                        len(candidates), (), generator=gen,
                    ).item()
                    cur = candidates[pick]
                    visited[cur] = True

    return _build_subgraph_from_node_mask(
        graph, visited, relabel_nodes, "random_walk_sample",
        extra_metadata={
            "seed_nodes": seed_nodes.detach().cpu(),
            "walk_length": int(walk_length),
            "num_walks_per_seed": int(num_walks_per_seed),
            "direction": direction,
            "restart_prob": float(restart_prob),
        },
    )
