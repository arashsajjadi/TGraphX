"""Triangle, wedge, and clustering coefficient mining in pure PyTorch.

These functions compute small-motif counts that are foundational for
graph pattern recognition.  All implementations are exact for small
graphs.  A density guard limits accidental O(N²) allocations.

Stability: Beta (v0.3.2+).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

__all__ = [
    "triangle_count",
    "wedge_count",
    "local_clustering_coefficient",
    "motif_counts",
    "motif_features",
]


_DENSE_GUARD = 10_000  # warn above this node count


def _validate(edge_index: torch.Tensor, num_nodes: int) -> None:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E]; got {tuple(edge_index.shape)}")
    if num_nodes < 0:
        raise ValueError(f"num_nodes must be non-negative; got {num_nodes}")
    if edge_index.numel() and int(edge_index.max().item()) >= num_nodes:
        raise ValueError(f"edge_index max id {int(edge_index.max())} >= num_nodes={num_nodes}")


def _to_adj_sets(edge_index: torch.Tensor, num_nodes: int, directed: bool = False) -> list:
    """Build adjacency as Python sets (out-neighbours)."""
    adj: list = [set() for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return adj
    src = edge_index[0].cpu().tolist()
    dst = edge_index[1].cpu().tolist()
    for u, v in zip(src, dst):
        if u != v:  # ignore self-loops for motif counts
            adj[u].add(v)
            if not directed:
                adj[v].add(u)
    return adj


def triangle_count(
    edge_index: torch.Tensor,
    num_nodes: int,
    directed: bool = False,
    node_level: bool = False,
) -> Any:
    """Count triangles in the (optionally undirected) graph.

    For an undirected graph, each triangle is counted **once** at graph level.
    For the directed graph, only cycles of length 3 (3-cycles) are counted.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        directed: When ``False`` (default), treat the graph as undirected.
        node_level: When ``True``, return per-node triangle counts
            (``LongTensor[N]``); otherwise return a scalar integer.

    Returns:
        Scalar ``int`` (graph-level) or ``LongTensor[num_nodes]``
        (node-level).

    Complexity: O(N * d² / 2) for undirected where d is average degree.
    For dense graphs with N > 10 000, a warning is raised.
    """
    _validate(edge_index, num_nodes)
    if num_nodes > _DENSE_GUARD:
        import warnings
        warnings.warn(
            f"triangle_count called with num_nodes={num_nodes}; "
            "this is O(N * d²) and may be slow for dense graphs.",
            stacklevel=2,
        )
    adj = _to_adj_sets(edge_index, num_nodes, directed=directed)
    node_tri = [0] * num_nodes
    for u in range(num_nodes):
        nbrs_u = adj[u]
        for v in nbrs_u:
            if not directed and v <= u:
                continue
            shared = len(nbrs_u & adj[v])
            node_tri[u] += shared
            node_tri[v] += shared

    if node_level:
        # Each node-triangle is over-counted by 2 in the loop above
        # (the triangle {a,b,c} contributes 1 to node_tri[a] from edge (a,b)
        # and 1 from edge (a,c)).  Divide by 2 to get per-node triangle count.
        return torch.tensor(
            [c // 2 for c in node_tri], dtype=torch.long,
        )
    # Graph-level: the loop above adds 2 to sum(node_tri) per triangle edge,
    # i.e. +6 per triangle in total.  Divide by 6.
    return sum(node_tri) // 6


def wedge_count(
    edge_index: torch.Tensor,
    num_nodes: int,
    directed: bool = False,
) -> int:
    """Count open wedges (paths of length 2) in the undirected graph.

    A wedge centred at ``v`` is a pair of distinct edges ``(u, v)`` and
    ``(v, w)`` where ``u ≠ w``.  The total wedge count is
    ``Σ_v C(deg(v), 2)``.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        directed: When ``False`` (default), treat as undirected.

    Returns:
        Non-negative integer.
    """
    _validate(edge_index, num_nodes)
    adj = _to_adj_sets(edge_index, num_nodes, directed=directed)
    total = 0
    for u in range(num_nodes):
        d = len(adj[u])
        total += d * (d - 1) // 2
    return total


def local_clustering_coefficient(
    edge_index: torch.Tensor,
    num_nodes: int,
    directed: bool = False,
) -> torch.Tensor:
    """Local (undirected) clustering coefficient for each node.

    ``C(v) = 2 * triangles(v) / (deg(v) * (deg(v) - 1))``.
    Nodes with degree < 2 have C(v) = 0.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        directed: When ``False`` (default), treat as undirected.

    Returns:
        ``FloatTensor[num_nodes]`` in ``[0.0, 1.0]``.
    """
    _validate(edge_index, num_nodes)
    adj = _to_adj_sets(edge_index, num_nodes, directed=directed)
    coeffs = torch.zeros(num_nodes, dtype=torch.float)
    for v in range(num_nodes):
        nbrs = adj[v]
        d = len(nbrs)
        if d < 2:
            continue
        t = 0
        nbrs_list = list(nbrs)
        for i in range(len(nbrs_list)):
            for j in range(i + 1, len(nbrs_list)):
                if nbrs_list[j] in adj[nbrs_list[i]]:
                    t += 1
        coeffs[v] = 2.0 * float(t) / float(d * (d - 1))
    return coeffs


def motif_counts(
    edge_index: torch.Tensor,
    num_nodes: int,
    directed: bool = False,
) -> Dict[str, Any]:
    """Return a JSON-serializable dict of small motif counts.

    Keys: ``edges``, ``self_loops``, ``triangles``, ``wedges``,
    ``mean_clustering_coefficient``.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        directed: When ``False`` (default), treat as undirected.

    Returns:
        Plain Python dict (JSON-serializable).
    """
    _validate(edge_index, num_nodes)
    # Edges excluding self-loops.
    if edge_index.numel():
        no_loops = edge_index[:, edge_index[0] != edge_index[1]]
    else:
        no_loops = edge_index
    n_self_loops = int(edge_index.size(1)) - int(no_loops.size(1))

    t = triangle_count(no_loops, num_nodes, directed=directed)
    w = wedge_count(no_loops, num_nodes, directed=directed)
    cc = local_clustering_coefficient(no_loops, num_nodes, directed=directed)
    mean_cc = float(cc.mean().item()) if num_nodes > 0 else 0.0

    return {
        "num_nodes": num_nodes,
        "edges": int(no_loops.size(1)),
        "self_loops": n_self_loops,
        "triangles": int(t),
        "wedges": int(w),
        "mean_clustering_coefficient": round(mean_cc, 6),
    }


def motif_features(
    edge_index: torch.Tensor,
    num_nodes: int,
    directed: bool = False,
) -> torch.Tensor:
    """Return a per-node motif feature matrix ``[num_nodes, 3]``.

    Columns: ``[degree, triangle_count, clustering_coefficient]``.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        directed: When ``False`` (default), treat as undirected.

    Returns:
        ``FloatTensor[num_nodes, 3]``.
    """
    _validate(edge_index, num_nodes)
    if edge_index.numel():
        no_loops = edge_index[:, edge_index[0] != edge_index[1]]
    else:
        no_loops = edge_index
    adj = _to_adj_sets(no_loops, num_nodes, directed=directed)
    deg = torch.tensor([float(len(adj[v])) for v in range(num_nodes)])
    tri = triangle_count(no_loops, num_nodes, directed=directed, node_level=True).float()
    cc = local_clustering_coefficient(no_loops, num_nodes, directed=directed)
    return torch.stack([deg, tri, cc], dim=1)
