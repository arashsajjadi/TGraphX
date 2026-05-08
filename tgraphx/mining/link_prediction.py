"""Classical link prediction scoring functions in pure PyTorch.

These are **structural** link prediction baselines.  They score candidate
edges based on the topology of the graph, with no learned model.  They
are useful for:
- link prediction baselines,
- hard-negative candidate re-scoring,
- graph membership confusion analysis.

All functions work on directed ``edge_index`` by default.  For undirected
graphs the caller should symmetrise the graph first (e.g. with
``tgraphx.transforms.ToUndirected``).

Stability: Beta (v0.3.2+).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

__all__ = [
    "common_neighbors_score",
    "jaccard_score",
    "adamic_adar_score",
    "resource_allocation_score",
    "preferential_attachment_score",
]


# ── Internal helpers ─────────────────────────────────────────────────────────


def _validate_pairs(
    pairs: torch.Tensor, num_nodes: Optional[int],
) -> Tuple[torch.Tensor, int]:
    if pairs.dim() != 2 or pairs.size(0) != 2:
        raise ValueError(f"pairs must have shape [2, P]; got {tuple(pairs.shape)}")
    pairs = pairs.to(torch.long)
    if num_nodes is None:
        num_nodes = int(pairs.max().item()) + 1 if pairs.numel() else 0
    return pairs, int(num_nodes)


def _build_neighbor_sets(
    edge_index: torch.Tensor, num_nodes: int,
) -> list:
    """Return adjacency as a list of Python sets (one per node)."""
    adj: list = [set() for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return adj
    src = edge_index[0].cpu().tolist()
    dst = edge_index[1].cpu().tolist()
    for u, v in zip(src, dst):
        adj[u].add(v)
    return adj


def _degrees_vec(
    edge_index: torch.Tensor, num_nodes: int,
) -> torch.Tensor:
    """Out-degree vector."""
    device = edge_index.device
    deg = torch.zeros(num_nodes, dtype=torch.float, device=device)
    if edge_index.numel():
        ones = torch.ones(edge_index.size(1), dtype=torch.float, device=device)
        deg.scatter_add_(0, edge_index[0], ones)
    return deg


# We use a Python-based neighbor-set approach because the number of pairs is
# typically small in practice and the set-intersection is straightforward.
# For very large queries a sparse matrix approach would be faster; this is
# documented in the complexity notes.


def common_neighbors_score(
    edge_index: torch.Tensor,
    pairs: torch.Tensor,
    num_nodes: Optional[int] = None,
    directed: bool = False,
) -> torch.Tensor:
    """Number of common neighbours for each candidate pair.

    For an undirected graph with symmetrised ``edge_index``, this counts
    ``|N(u) ∩ N(v)|``.

    Args:
        edge_index: ``LongTensor[2, E]`` of existing edges.
        pairs: ``LongTensor[2, P]`` of candidate pairs ``(u, v)`` to score.
        num_nodes: Optional node count; inferred when ``None``.
        directed: When ``True``, uses out-neighbours only.

    Returns:
        ``FloatTensor[P]`` of scores.  Higher = more common neighbours.

    Complexity: O(P * d_max) where d_max is maximum degree.
    """
    pairs, num_nodes = _validate_pairs(pairs, num_nodes)
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_index.numel() and int(edge_index.max().item()) >= num_nodes:
        raise ValueError("edge_index node ids exceed num_nodes")

    adj = _build_neighbor_sets(edge_index, num_nodes)
    src = pairs[0].cpu().tolist()
    dst = pairs[1].cpu().tolist()
    scores = torch.zeros(len(src), dtype=torch.float, device=pairs.device)
    for k, (u, v) in enumerate(zip(src, dst)):
        scores[k] = float(len(adj[u] & adj[v]))
    return scores


def jaccard_score(
    edge_index: torch.Tensor,
    pairs: torch.Tensor,
    num_nodes: Optional[int] = None,
    directed: bool = False,
) -> torch.Tensor:
    """Jaccard coefficient for each candidate pair.

    ``|N(u) ∩ N(v)| / |N(u) ∪ N(v)|``.  Returns 0 when the union is empty.

    Args:
        edge_index: ``LongTensor[2, E]``.
        pairs: ``LongTensor[2, P]``.
        num_nodes: Optional.
        directed: When ``True``, uses out-neighbours only.

    Returns:
        ``FloatTensor[P]`` in ``[0, 1]``.
    """
    pairs, num_nodes = _validate_pairs(pairs, num_nodes)
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")

    adj = _build_neighbor_sets(edge_index, num_nodes)
    src = pairs[0].cpu().tolist()
    dst = pairs[1].cpu().tolist()
    scores = torch.zeros(len(src), dtype=torch.float, device=pairs.device)
    for k, (u, v) in enumerate(zip(src, dst)):
        inter = len(adj[u] & adj[v])
        union = len(adj[u] | adj[v])
        scores[k] = float(inter) / float(union) if union > 0 else 0.0
    return scores


def adamic_adar_score(
    edge_index: torch.Tensor,
    pairs: torch.Tensor,
    num_nodes: Optional[int] = None,
    directed: bool = False,
) -> torch.Tensor:
    """Adamic-Adar index for each candidate pair.

    ``Σ_{w ∈ N(u) ∩ N(v)} 1 / log(deg(w))``.  Nodes with degree ≤ 1
    are skipped (log(1) = 0, log(0) undefined).

    Args:
        edge_index: ``LongTensor[2, E]``.
        pairs: ``LongTensor[2, P]``.
        num_nodes: Optional.
        directed: When ``True``, uses out-neighbours only.

    Returns:
        ``FloatTensor[P]``, non-negative.
    """
    pairs, num_nodes = _validate_pairs(pairs, num_nodes)
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")

    adj = _build_neighbor_sets(edge_index, num_nodes)
    src = pairs[0].cpu().tolist()
    dst = pairs[1].cpu().tolist()
    # Degree of each node (out-degree from the adj sets).
    deg = [len(nb) for nb in adj]
    scores = torch.zeros(len(src), dtype=torch.float, device=pairs.device)
    import math
    for k, (u, v) in enumerate(zip(src, dst)):
        s = 0.0
        for w in adj[u] & adj[v]:
            dw = deg[w]
            if dw > 1:
                s += 1.0 / math.log(float(dw))
        scores[k] = s
    return scores


def resource_allocation_score(
    edge_index: torch.Tensor,
    pairs: torch.Tensor,
    num_nodes: Optional[int] = None,
    directed: bool = False,
) -> torch.Tensor:
    """Resource allocation index for each candidate pair.

    ``Σ_{w ∈ N(u) ∩ N(v)} 1 / deg(w)``.  Nodes with degree 0 are skipped.

    Args:
        edge_index: ``LongTensor[2, E]``.
        pairs: ``LongTensor[2, P]``.
        num_nodes: Optional.
        directed: When ``True``, uses out-neighbours only.

    Returns:
        ``FloatTensor[P]``, non-negative.
    """
    pairs, num_nodes = _validate_pairs(pairs, num_nodes)
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")

    adj = _build_neighbor_sets(edge_index, num_nodes)
    src = pairs[0].cpu().tolist()
    dst = pairs[1].cpu().tolist()
    deg = [len(nb) for nb in adj]
    scores = torch.zeros(len(src), dtype=torch.float, device=pairs.device)
    for k, (u, v) in enumerate(zip(src, dst)):
        s = 0.0
        for w in adj[u] & adj[v]:
            dw = deg[w]
            if dw > 0:
                s += 1.0 / float(dw)
        scores[k] = s
    return scores


def preferential_attachment_score(
    edge_index: torch.Tensor,
    pairs: torch.Tensor,
    num_nodes: Optional[int] = None,
    directed: bool = False,
) -> torch.Tensor:
    """Preferential attachment score for each candidate pair.

    ``deg(u) * deg(v)``.  Does not depend on common neighbours.

    Args:
        edge_index: ``LongTensor[2, E]``.
        pairs: ``LongTensor[2, P]``.
        num_nodes: Optional.
        directed: When ``True``, uses out-degrees; otherwise total degree.

    Returns:
        ``FloatTensor[P]``, non-negative.
    """
    pairs, num_nodes = _validate_pairs(pairs, num_nodes)
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")

    deg = _degrees_vec(edge_index, num_nodes)
    src_deg = deg[pairs[0]]
    dst_deg = deg[pairs[1]]
    return (src_deg * dst_deg).to(torch.float)
