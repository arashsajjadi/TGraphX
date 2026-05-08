"""BFS traversal and unweighted shortest paths.

These helpers operate on a directed ``edge_index``.  Pass the
``ToUndirected`` transform first if you want undirected behaviour.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

__all__ = [
    "bfs_edges",
    "bfs_layers",
    "shortest_path_length",
]


def _check(edge_index: torch.Tensor, num_nodes: Optional[int]) -> Tuple[torch.Tensor, int]:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"edge_index must have shape [2, E]; got {tuple(edge_index.shape)}"
        )
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() else 0
    if num_nodes < 0:
        raise ValueError(f"num_nodes must be non-negative; got {num_nodes}")
    # Validate node ids only when edges are present.
    if edge_index.numel() and (
        edge_index.min().item() < 0 or edge_index.max().item() >= num_nodes
    ):
        raise ValueError(
            f"edge_index entries out of range for num_nodes={num_nodes}: "
            f"min={int(edge_index.min())}, max={int(edge_index.max())}"
        )
    return edge_index.to(torch.long), int(num_nodes)


def _adjacency_lists(
    edge_index: torch.Tensor, num_nodes: int,
) -> List[List[int]]:
    """CSR-style adjacency as Python lists (CPU only)."""
    out: List[List[int]] = [[] for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return out
    src = edge_index[0].cpu().tolist()
    dst = edge_index[1].cpu().tolist()
    for u, v in zip(src, dst):
        out[u].append(v)
    return out


def bfs_layers(
    edge_index: torch.Tensor,
    source: int,
    num_nodes: Optional[int] = None,
    max_hops: Optional[int] = None,
) -> List[torch.Tensor]:
    """Return BFS frontiers as a list of tensors.

    Layer 0 is ``[source]``; layer ``k`` contains all nodes whose
    shortest unweighted distance from ``source`` is exactly ``k``.

    Args:
        edge_index: ``LongTensor[2, E]`` (directed).  Use
            :class:`~tgraphx.transforms.ToUndirected` first for
            undirected BFS.
        source: Starting node id.
        num_nodes: Optional node count; inferred when ``None``.
        max_hops: When set, stops after this many hops.

    Returns:
        ``list[LongTensor]`` of length ``≤ num_layers``.  Each tensor
        contains node ids on the corresponding layer; the union over
        layers is the set of nodes reachable from ``source``.
    """
    edge_index, num_nodes = _check(edge_index, num_nodes)
    if num_nodes == 0:
        return []
    if not (0 <= source < num_nodes):
        raise ValueError(f"source={source} out of range [0, {num_nodes})")

    device = edge_index.device
    visited = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    visited[source] = True
    layers = [torch.tensor([source], dtype=torch.long, device=device)]

    if edge_index.numel() == 0:
        return layers

    src = edge_index[0]
    dst = edge_index[1]
    cap = max_hops if max_hops is not None else num_nodes

    for _ in range(cap):
        frontier = layers[-1]
        if frontier.numel() == 0:
            break
        # Edges leaving the current frontier.
        in_front = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        in_front[frontier] = True
        edge_mask = in_front[src]
        if not bool(edge_mask.any()):
            break
        next_dst = dst[edge_mask]
        # Drop already-visited destinations.
        new_mask = ~visited[next_dst]
        new_nodes = torch.unique(next_dst[new_mask])
        if new_nodes.numel() == 0:
            break
        visited[new_nodes] = True
        layers.append(new_nodes)
    return layers


def bfs_edges(
    edge_index: torch.Tensor,
    source: int,
    num_nodes: Optional[int] = None,
    max_hops: Optional[int] = None,
) -> torch.Tensor:
    """Return the BFS spanning-tree edges from ``source`` in visit order.

    Each non-source node ``v`` reachable from ``source`` is paired with
    its BFS predecessor ``u``.  The returned tensor has shape
    ``[2, num_visited_non_source]`` with row 0 = predecessors and
    row 1 = visited children.

    Args:
        edge_index: ``LongTensor[2, E]`` (directed).
        source: Starting node id.
        num_nodes: Optional node count.
        max_hops: When set, stops after this many hops.

    Returns:
        ``LongTensor[2, M]`` of (predecessor, child) pairs.  ``M`` is
        ``len(reachable) - 1`` when the entire reachable set is explored.
    """
    edge_index, num_nodes = _check(edge_index, num_nodes)
    if num_nodes == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device)
    if not (0 <= source < num_nodes):
        raise ValueError(f"source={source} out of range [0, {num_nodes})")

    device = edge_index.device
    if edge_index.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    adj = _adjacency_lists(edge_index, num_nodes)
    visited = [False] * num_nodes
    visited[source] = True
    pred: List[int] = []
    child: List[int] = []
    queue: List[int] = [source]
    cap = max_hops if max_hops is not None else num_nodes
    hop = 0
    while queue and hop < cap:
        next_queue: List[int] = []
        for u in queue:
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    pred.append(u)
                    child.append(v)
                    next_queue.append(v)
        queue = next_queue
        hop += 1
    if not pred:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    return torch.tensor(
        [pred, child], dtype=torch.long, device=device,
    )


def shortest_path_length(
    edge_index: torch.Tensor,
    source: int,
    num_nodes: Optional[int] = None,
    max_hops: Optional[int] = None,
) -> torch.Tensor:
    """Unweighted single-source shortest-path distances.

    Args:
        edge_index: ``LongTensor[2, E]`` (directed).
        source: Source node id.
        num_nodes: Optional node count.
        max_hops: When set, distances above this are reported as ``-1``.

    Returns:
        ``LongTensor[num_nodes]`` where entry ``v`` is the unweighted
        distance from ``source`` to ``v``.  Unreachable nodes are
        ``-1``.  Entry at ``source`` is ``0``.
    """
    edge_index, num_nodes = _check(edge_index, num_nodes)
    if num_nodes == 0:
        return torch.zeros((0,), dtype=torch.long, device=edge_index.device)
    if not (0 <= source < num_nodes):
        raise ValueError(f"source={source} out of range [0, {num_nodes})")

    device = edge_index.device
    distances = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    distances[source] = 0
    layers = bfs_layers(edge_index, source, num_nodes, max_hops=max_hops)
    for hop, layer in enumerate(layers):
        if hop == 0:
            continue  # source already set to 0
        distances[layer] = hop
    return distances
