"""Community detection foundation in pure PyTorch.

Implements label-propagation community detection and modularity scoring
as simple, deterministic baselines for exploratory graph mining.

This is **not** a Louvain / Leiden implementation.  For production
community detection use dedicated community-detection libraries.

Stability: Beta (v0.3.2+).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

__all__ = [
    "label_propagation_communities",
    "modularity",
    "community_summary",
]


def _validate(edge_index: torch.Tensor, num_nodes: int) -> None:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E]; got {tuple(edge_index.shape)}")
    if num_nodes < 0:
        raise ValueError(f"num_nodes must be non-negative; got {num_nodes}")
    if edge_index.numel() and int(edge_index.max().item()) >= num_nodes:
        raise ValueError(f"edge_index max node id exceeds num_nodes={num_nodes}")


def label_propagation_communities(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_iter: int = 50,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Simple synchronous label propagation community detection.

    Each node adopts the most common label among its neighbours.
    Isolated nodes keep their own label.  Ties are broken by choosing
    the smallest label.

    Args:
        edge_index: ``LongTensor[2, E]``.  Treated as undirected.
        num_nodes: Node count.
        max_iter: Maximum propagation iterations.
        seed: Optional seed for reproducibility (used to shuffle
            tie-breaking order in future extensions; currently no-op
            because tie-breaking is by min-label which is deterministic).

    Returns:
        ``LongTensor[num_nodes]`` of community labels in ``[0, K)``.
    """
    _validate(edge_index, num_nodes)
    if num_nodes == 0:
        return torch.zeros(0, dtype=torch.long)

    # Build symmetric adjacency lists.
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    if edge_index.numel():
        src = edge_index[0].cpu().tolist()
        dst = edge_index[1].cpu().tolist()
        for u, v in zip(src, dst):
            if u != v:
                adj[u].append(v)
                adj[v].append(u)

    labels = list(range(num_nodes))  # start: each node is its own community

    for _ in range(max_iter):
        old = list(labels)
        changed = False
        for v in range(num_nodes):
            nbrs = adj[v]
            if not nbrs:
                continue
            # Count labels of neighbours.
            counts: Dict[int, int] = {}
            for u in nbrs:
                lab = labels[u]
                counts[lab] = counts.get(lab, 0) + 1
            max_count = max(counts.values())
            # Choose the smallest label with max count (deterministic tie-break).
            best = min(k for k, c in counts.items() if c == max_count)
            if best != labels[v]:
                labels[v] = best
                changed = True
        if not changed:
            break

    # Compact labels to [0, K).
    unique = sorted(set(labels))
    remap = {lab: i for i, lab in enumerate(unique)}
    labels = [remap[l] for l in labels]
    return torch.tensor(labels, dtype=torch.long)


def modularity(
    edge_index: torch.Tensor,
    communities: torch.Tensor,
    num_nodes: Optional[int] = None,
    directed: bool = False,
) -> float:
    """Compute Newman-Girvan modularity for a community assignment.

    ``Q = (1/2m) * Σ_{ij} [A_ij - k_i*k_j / (2m)] * delta(c_i, c_j)``

    where m = number of edges (undirected), k_i = degree of node i.
    Returns 0 if there are no edges.

    Args:
        edge_index: ``LongTensor[2, E]``.
        communities: ``LongTensor[N]`` of community ids.
        num_nodes: Optional node count; inferred when ``None``.
        directed: Currently only undirected modularity is implemented.

    Returns:
        Float Q in ``(-1, 1]``.
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    N = int(communities.shape[0])
    if num_nodes is None:
        num_nodes = N
    _validate(edge_index, num_nodes)

    if edge_index.numel() == 0:
        return 0.0

    # Treat as undirected: combine edge_index with its reverse.
    ei_sym = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    m = float(ei_sym.size(1)) / 2.0  # number of undirected edges
    if m == 0:
        return 0.0

    deg = torch.zeros(num_nodes, dtype=torch.float)
    ones = torch.ones(ei_sym.size(1), dtype=torch.float)
    deg.scatter_add_(0, ei_sym[0], ones)

    comm = communities.cpu()
    src = ei_sym[0].cpu()
    dst = ei_sym[1].cpu()

    Q = 0.0
    for e in range(ei_sym.size(1)):
        i, j = int(src[e].item()), int(dst[e].item())
        if int(comm[i].item()) == int(comm[j].item()):
            A = 1.0
            expected = float(deg[i].item()) * float(deg[j].item()) / (2.0 * m)
            Q += A - expected
    Q /= 2.0 * m
    return float(Q)


def community_summary(
    edge_index: torch.Tensor,
    communities: torch.Tensor,
    num_nodes: Optional[int] = None,
) -> Dict[str, Any]:
    """Return a JSON-serializable dict summarising the community structure.

    Keys: ``num_communities``, ``community_sizes``, ``largest_community_size``,
    ``smallest_community_size``, ``modularity``.

    Args:
        edge_index: ``LongTensor[2, E]``.
        communities: ``LongTensor[N]`` of community labels.
        num_nodes: Optional.

    Returns:
        Plain Python dict (JSON-serializable).
    """
    N = int(communities.shape[0])
    if num_nodes is None:
        num_nodes = N
    comm = communities.cpu().tolist()
    unique = sorted(set(comm))
    sizes = {c: comm.count(c) for c in unique}
    size_list = sorted(sizes.values(), reverse=True)
    Q = modularity(edge_index, communities, num_nodes)
    return {
        "num_communities": len(unique),
        "community_sizes": size_list,
        "largest_community_size": max(size_list) if size_list else 0,
        "smallest_community_size": min(size_list) if size_list else 0,
        "modularity": round(Q, 6),
    }
