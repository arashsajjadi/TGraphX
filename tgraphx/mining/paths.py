"""Graph path algorithms: shortest paths, spanning trees, and cuts.

All algorithms are implemented in pure Python/PyTorch without dense
adjacency matrix allocation (except where explicitly guarded).

Stability: Beta (v0.4.3+).
"""
from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Set, Tuple

import torch

__all__ = [
    # Traversal
    "bfs_order",
    "dfs_order",
    "multi_source_bfs",
    "reachable_nodes",
    # Shortest paths
    "dijkstra_shortest_path",
    "all_pairs_shortest_path_length",
    "batched_shortest_path_length",
    "reconstruct_path",
    # Spanning trees
    "minimum_spanning_tree",
    "maximum_spanning_tree",
    # Cuts and metrics
    "cut_size",
    "normalized_cut",
    "conductance",
    "volume",
    "boundary_edges",
    # Utilities
    "write_path_summary",
]

_LARGE_GRAPH_APSP = 1_000  # guard for all-pairs shortest path


def _validate(edge_index: torch.Tensor, num_nodes: int) -> None:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E]; got {tuple(edge_index.shape)}")
    if num_nodes < 0:
        raise ValueError(f"num_nodes must be non-negative; got {num_nodes}")
    if edge_index.numel() and int(edge_index.max().item()) >= num_nodes:
        raise ValueError(f"edge_index max node id >= num_nodes={num_nodes}")


def _adj_lists(
    edge_index: torch.Tensor, num_nodes: int, directed: bool = True,
    weight: Optional[torch.Tensor] = None,
) -> List[List[Tuple[int, float]]]:
    """Build adjacency lists as list of (neighbour, weight) pairs."""
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return adj
    src = edge_index[0].cpu().tolist()
    dst = edge_index[1].cpu().tolist()
    wts = weight.cpu().tolist() if weight is not None else [1.0] * len(src)
    for u, v, w in zip(src, dst, wts):
        adj[u].append((v, w))
        if not directed:
            adj[v].append((u, w))
    return adj


# ── Traversal ────────────────────────────────────────────────────────────────


def bfs_order(
    edge_index: torch.Tensor,
    start: int,
    num_nodes: int,
    directed: bool = False,
) -> torch.Tensor:
    """BFS traversal order starting from ``start``.

    Args:
        edge_index: ``LongTensor[2, E]``.
        start: Source node.
        num_nodes: Node count.
        directed: When ``False`` (default), treats edges as undirected.

    Returns:
        ``LongTensor[K]`` of visited node IDs in BFS order.
    """
    _validate(edge_index, num_nodes)
    if not (0 <= start < num_nodes):
        raise ValueError(f"start={start} out of range [0, {num_nodes})")
    adj = _adj_lists(edge_index, num_nodes, directed=directed)
    visited = [False] * num_nodes
    visited[start] = True
    order = [start]
    queue = [start]
    qi = 0
    while qi < len(queue):
        u = queue[qi]; qi += 1
        for v, _ in adj[u]:
            if not visited[v]:
                visited[v] = True
                order.append(v)
                queue.append(v)
    return torch.tensor(order, dtype=torch.long)


def dfs_order(
    edge_index: torch.Tensor,
    start: int,
    num_nodes: int,
    directed: bool = False,
) -> torch.Tensor:
    """DFS traversal order starting from ``start`` (iterative).

    Args:
        edge_index: ``LongTensor[2, E]``.
        start: Source node.
        num_nodes: Node count.
        directed: When ``False`` (default), treats edges as undirected.

    Returns:
        ``LongTensor[K]`` of visited node IDs in DFS order.
    """
    _validate(edge_index, num_nodes)
    if not (0 <= start < num_nodes):
        raise ValueError(f"start={start} out of range [0, {num_nodes})")
    adj = _adj_lists(edge_index, num_nodes, directed=directed)
    visited = [False] * num_nodes
    order = []
    stack = [start]
    while stack:
        u = stack.pop()
        if visited[u]:
            continue
        visited[u] = True
        order.append(u)
        for v, _ in reversed(adj[u]):  # reversed for deterministic order
            if not visited[v]:
                stack.append(v)
    return torch.tensor(order, dtype=torch.long)


def multi_source_bfs(
    edge_index: torch.Tensor,
    sources: torch.Tensor,
    num_nodes: int,
    directed: bool = False,
) -> torch.Tensor:
    """Multi-source BFS: nearest distances from any source.

    Args:
        edge_index: ``LongTensor[2, E]``.
        sources: ``LongTensor[S]`` of source node IDs.
        num_nodes: Node count.
        directed: When ``False``, treats edges as undirected.

    Returns:
        ``LongTensor[N]`` distances; -1 for unreachable nodes.
    """
    _validate(edge_index, num_nodes)
    adj = _adj_lists(edge_index, num_nodes, directed=directed)
    dist = [-1] * num_nodes
    queue = []
    for s in sources.cpu().tolist():
        if 0 <= s < num_nodes and dist[s] < 0:
            dist[s] = 0
            queue.append(s)
    qi = 0
    while qi < len(queue):
        u = queue[qi]; qi += 1
        for v, _ in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                queue.append(v)
    return torch.tensor(dist, dtype=torch.long)


def reachable_nodes(
    edge_index: torch.Tensor,
    sources: torch.Tensor,
    num_nodes: int,
    directed: bool = False,
) -> torch.Tensor:
    """Return all node IDs reachable from any source.

    Returns:
        ``LongTensor[K]`` of reachable node IDs (including sources).
    """
    dist = multi_source_bfs(edge_index, sources, num_nodes, directed=directed)
    return (dist >= 0).nonzero(as_tuple=False).view(-1)


# ── Shortest paths ────────────────────────────────────────────────────────────


def dijkstra_shortest_path(
    edge_index: torch.Tensor,
    source: int,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
    directed: bool = False,
) -> Tuple[torch.Tensor, Dict[int, int]]:
    """Dijkstra's algorithm for non-negative weighted shortest paths.

    Args:
        edge_index: ``LongTensor[2, E]``.
        source: Source node.
        num_nodes: Node count.
        edge_weight: Optional ``FloatTensor[E]`` non-negative weights.
            When ``None``, uses unit weights (equivalent to BFS).
        directed: When ``False``, treats edges as undirected.

    Returns:
        ``(distances, predecessors)`` where ``distances`` is
        ``FloatTensor[N]`` (inf for unreachable nodes) and
        ``predecessors`` is a dict ``{node: predecessor_node}``
        for path reconstruction.

    Raises:
        ValueError: If any edge weight is negative.
    """
    _validate(edge_index, num_nodes)
    if not (0 <= source < num_nodes):
        raise ValueError(f"source={source} out of range [0, {num_nodes})")
    if edge_weight is not None and float(edge_weight.min().item()) < 0:
        raise ValueError(
            "dijkstra_shortest_path requires non-negative edge weights. "
            "For negative weights use Bellman-Ford (not yet implemented)."
        )

    adj = _adj_lists(edge_index, num_nodes, directed=directed, weight=edge_weight)
    dist = [math.inf] * num_nodes
    dist[source] = 0.0
    pred: Dict[int, int] = {}
    heap = [(0.0, source)]
    visited: Set[int] = set()

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                pred[v] = u
                heapq.heappush(heap, (nd, v))

    dist_t = torch.tensor(dist, dtype=torch.float)
    return dist_t, pred


def batched_shortest_path_length(
    edge_index: torch.Tensor,
    sources: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
    directed: bool = False,
) -> torch.Tensor:
    """Shortest path lengths from multiple sources.

    Args:
        edge_index: ``LongTensor[2, E]``.
        sources: ``LongTensor[S]`` of source node IDs.
        num_nodes: Node count.
        edge_weight: Optional ``FloatTensor[E]`` non-negative weights.
        directed: When ``False``, treats edges as undirected.

    Returns:
        ``FloatTensor[S, N]`` — row i = distances from sources[i].
    """
    _validate(edge_index, num_nodes)
    rows = []
    for s in sources.cpu().tolist():
        d, _ = dijkstra_shortest_path(
            edge_index, int(s), num_nodes, edge_weight, directed
        )
        rows.append(d)
    if not rows:
        return torch.zeros((0, num_nodes), dtype=torch.float)
    return torch.stack(rows, dim=0)


def all_pairs_shortest_path_length(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
    max_nodes: int = _LARGE_GRAPH_APSP,
) -> torch.Tensor:
    """All-pairs shortest path length matrix.

    Only feasible for small graphs.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        edge_weight: Optional ``FloatTensor[E]``.
        max_nodes: Size guard (default 1 000).

    Returns:
        ``FloatTensor[N, N]``.  Unreachable pairs have ``inf``.

    Raises:
        ValueError: If ``num_nodes > max_nodes``.
    """
    _validate(edge_index, num_nodes)
    if num_nodes > max_nodes:
        raise ValueError(
            f"all_pairs_shortest_path_length: num_nodes={num_nodes} > "
            f"max_nodes={max_nodes}.  Use batched_shortest_path_length "
            f"for subsets."
        )
    sources = torch.arange(num_nodes, dtype=torch.long)
    return batched_shortest_path_length(
        edge_index, sources, num_nodes, edge_weight, directed=False
    )


def reconstruct_path(
    source: int,
    target: int,
    predecessors: Dict[int, int],
) -> List[int]:
    """Reconstruct a shortest path from predecessor dict.

    Args:
        source: Source node.
        target: Target node.
        predecessors: Dict returned by :func:`dijkstra_shortest_path`.

    Returns:
        List of node IDs from source to target (inclusive).
        Empty list if target is unreachable from source.
    """
    if target == source:
        return [source]
    path = [target]
    cur = target
    while cur in predecessors:
        cur = predecessors[cur]
        path.append(cur)
        if cur == source:
            return list(reversed(path))
    return []  # unreachable


# ── Spanning trees ────────────────────────────────────────────────────────────


def _kruskal(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor],
    maximize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Kruskal's algorithm returning MST/MaxST edge_index + weights."""
    _validate(edge_index, num_nodes)
    E = edge_index.size(1)
    if E == 0:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0), 0.0

    wts = edge_weight.float().cpu() if edge_weight is not None else torch.ones(E)
    # Sort edges by weight (ascending for MST, descending for MaxST).
    order = torch.argsort(wts, descending=maximize, stable=True)

    # Union-Find.
    parent = list(range(num_nodes))
    rank = [0] * num_nodes

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    src_all = edge_index[0].cpu().tolist()
    dst_all = edge_index[1].cpu().tolist()
    mst_src, mst_dst, mst_w = [], [], []
    total_weight = 0.0
    for idx in order.tolist():
        u, v, w = src_all[idx], dst_all[idx], float(wts[idx])
        if u != v and union(u, v):
            mst_src.extend([u, v])
            mst_dst.extend([v, u])
            mst_w.extend([w, w])
            total_weight += w

    if not mst_src:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0), 0.0
    mst_ei = torch.tensor([mst_src, mst_dst], dtype=torch.long)
    mst_wt = torch.tensor(mst_w, dtype=torch.float)
    return mst_ei, mst_wt, total_weight


def minimum_spanning_tree(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Kruskal's minimum spanning tree (or forest for disconnected graphs).

    Args:
        edge_index: ``LongTensor[2, E]`` (treated as undirected).
        num_nodes: Node count.
        edge_weight: Optional ``FloatTensor[E]`` edge weights.
            When ``None``, uses unit weights.

    Returns:
        ``(mst_edge_index, mst_weights, total_weight)`` tuple.
        ``mst_edge_index`` is ``LongTensor[2, 2*(N-1)]`` or smaller for forests.
    """
    return _kruskal(edge_index, num_nodes, edge_weight, maximize=False)


def maximum_spanning_tree(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Kruskal's maximum spanning tree (or forest).

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        edge_weight: Optional ``FloatTensor[E]``.

    Returns:
        ``(mst_edge_index, mst_weights, total_weight)`` tuple.
    """
    return _kruskal(edge_index, num_nodes, edge_weight, maximize=True)


# ── Cuts and graph metrics ────────────────────────────────────────────────────


def _partition_edges(
    edge_index: torch.Tensor,
    subset: Set[int],
) -> Tuple[int, int]:
    """Return (cut_edges, internal_edges) counts."""
    if edge_index.numel() == 0:
        return 0, 0
    src = edge_index[0].cpu().tolist()
    dst = edge_index[1].cpu().tolist()
    cut = 0
    internal = 0
    for u, v in zip(src, dst):
        if (u in subset) != (v in subset):
            cut += 1
        elif u in subset and v in subset:
            internal += 1
    return cut, internal


def cut_size(
    edge_index: torch.Tensor,
    num_nodes: int,
    subset: torch.Tensor,
) -> int:
    """Number of edges crossing the (subset, complement) partition.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        subset: ``LongTensor`` or ``BoolTensor[N]`` of node IDs in set S.

    Returns:
        Non-negative integer.
    """
    _validate(edge_index, num_nodes)
    if subset.dtype == torch.bool:
        nodes = set(subset.nonzero(as_tuple=False).view(-1).cpu().tolist())
    else:
        nodes = set(subset.cpu().tolist())
    cut, _ = _partition_edges(edge_index, nodes)
    return cut


def volume(
    edge_index: torch.Tensor,
    num_nodes: int,
    subset: torch.Tensor,
) -> int:
    """Volume of a subset: sum of degrees of nodes in subset.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        subset: ``LongTensor`` or ``BoolTensor[N]``.

    Returns:
        Non-negative integer.
    """
    _validate(edge_index, num_nodes)
    if subset.dtype == torch.bool:
        nodes = set(subset.nonzero(as_tuple=False).view(-1).cpu().tolist())
    else:
        nodes = set(subset.cpu().tolist())
    if not edge_index.numel():
        return 0
    src = edge_index[0].cpu().tolist()
    return sum(1 for u in src if u in nodes)


def conductance(
    edge_index: torch.Tensor,
    num_nodes: int,
    subset: torch.Tensor,
) -> float:
    """Conductance of a subset: cut_size / min(vol(S), vol(complement(S))).

    Returns 0 for empty set; 1 for impossible partition.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        subset: ``LongTensor`` or ``BoolTensor[N]``.

    Returns:
        Float in ``[0, 1]``.
    """
    _validate(edge_index, num_nodes)
    if subset.dtype == torch.bool:
        nodes_s = set(subset.nonzero(as_tuple=False).view(-1).cpu().tolist())
    else:
        nodes_s = set(subset.cpu().tolist())
    nodes_comp = set(range(num_nodes)) - nodes_s
    if not nodes_s or not nodes_comp:
        return 0.0
    vol_s = volume(edge_index, num_nodes, torch.tensor(sorted(nodes_s), dtype=torch.long))
    vol_c = volume(edge_index, num_nodes, torch.tensor(sorted(nodes_comp), dtype=torch.long))
    cut = cut_size(edge_index, num_nodes, torch.tensor(sorted(nodes_s), dtype=torch.long))
    denom = min(vol_s, vol_c)
    if denom == 0:
        return 0.0
    return float(cut) / float(denom)


def normalized_cut(
    edge_index: torch.Tensor,
    num_nodes: int,
    labels: torch.Tensor,
) -> float:
    """Normalized cut for a K-way partition.

    NCut = Σ_k cut(k) / vol(k)

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        labels: ``LongTensor[N]`` community labels in ``[0, K)``.

    Returns:
        Non-negative float.  Lower = better partition.
    """
    _validate(edge_index, num_nodes)
    classes = set(labels.cpu().tolist())
    total = 0.0
    for c in classes:
        subset = (labels == c).nonzero(as_tuple=False).view(-1)
        vol_c = volume(edge_index, num_nodes, subset)
        cut_c = cut_size(edge_index, num_nodes, subset)
        if vol_c > 0:
            total += float(cut_c) / float(vol_c)
    return total


def boundary_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    subset: torch.Tensor,
) -> torch.Tensor:
    """Return edge indices crossing the (subset, complement) boundary.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        subset: ``LongTensor`` or ``BoolTensor[N]``.

    Returns:
        ``LongTensor[2, K]`` of boundary edges.
    """
    _validate(edge_index, num_nodes)
    if subset.dtype == torch.bool:
        nodes = set(subset.nonzero(as_tuple=False).view(-1).cpu().tolist())
    else:
        nodes = set(subset.cpu().tolist())
    if not edge_index.numel():
        return torch.zeros((2, 0), dtype=torch.long)
    src = edge_index[0].cpu().tolist()
    dst = edge_index[1].cpu().tolist()
    mask = [(u in nodes) != (v in nodes) for u, v in zip(src, dst)]
    mask_t = torch.tensor(mask, dtype=torch.bool)
    return edge_index[:, mask_t]


# ── Report writer ─────────────────────────────────────────────────────────────


def write_path_summary(
    path: str,
    *,
    source: Optional[int] = None,
    num_reachable: Optional[int] = None,
    mean_distance: Optional[float] = None,
    max_distance: Optional[float] = None,
    mst_total_weight: Optional[float] = None,
    mst_num_edges: Optional[int] = None,
    conductance: Optional[float] = None,
    **extra,
) -> str:
    """Write a ``graph_algorithm_summary.json`` dashboard artifact.

    Args:
        path: Output file path.
        source: Source node for shortest path analysis.
        num_reachable: Reachable node count.
        mean_distance: Mean shortest path distance.
        max_distance: Eccentricity (max distance from source).
        mst_total_weight: Total MST weight.
        mst_num_edges: MST edge count.
        conductance: Conductance of a partition.
        **extra: Additional key-value pairs.

    Returns:
        Resolved path string.
    """
    import json, os, tempfile
    from pathlib import Path
    payload = {}
    if source is not None:
        payload["source"] = int(source)
    if num_reachable is not None:
        payload["num_reachable"] = int(num_reachable)
    if mean_distance is not None:
        payload["mean_distance"] = round(float(mean_distance), 6)
    if max_distance is not None:
        payload["max_distance"] = round(float(max_distance), 6)
    if mst_total_weight is not None:
        payload["mst_total_weight"] = round(float(mst_total_weight), 6)
    if mst_num_edges is not None:
        payload["mst_num_edges"] = int(mst_num_edges)
    if conductance is not None:
        payload["conductance"] = round(float(conductance), 6)
    payload.update({k: v for k, v in extra.items()})
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, default=str)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, str(p))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return str(p)
