"""Graph matching, coloring, clique, and flow algorithms.

All algorithms are size-guarded, exact for small graphs, and clearly
documented regarding complexity.  Heavy optional dependencies are never
required.

Stability: Beta (v0.4.4+) for matching/coloring;
           Experimental (v0.4.4+) for max-flow.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import torch

__all__ = [
    # Matching
    "greedy_maximal_matching",
    "bipartite_greedy_matching",
    # Coloring
    "greedy_coloring",
    "welsh_powell_coloring",
    # Clique / independent set
    "greedy_maximal_independent_set",
    "enumerate_maximal_cliques",
    # Max-flow / min-cut
    "edmonds_karp_max_flow",
    "min_cut_from_max_flow",
    # Isomorphism heuristics
    "wl_isomorphism_test",
    # Report writer
    "write_algorithm_report",
]

_MAXFLOW_SIZE = 500  # guard for Edmonds-Karp
_CLIQUE_SIZE = 50    # guard for clique enumeration


def _validate(edge_index: torch.Tensor, num_nodes: int) -> None:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must be [2, E]; got {tuple(edge_index.shape)}")
    if num_nodes < 0:
        raise ValueError(f"num_nodes must be >= 0; got {num_nodes}")
    if edge_index.numel() and int(edge_index.max().item()) >= num_nodes:
        raise ValueError(f"edge_index max id >= num_nodes={num_nodes}")


def _build_adj(edge_index: torch.Tensor, num_nodes: int, directed: bool = False) -> list:
    adj: list = [set() for _ in range(num_nodes)]
    if not edge_index.numel():
        return adj
    src = edge_index[0].cpu().tolist()
    dst = edge_index[1].cpu().tolist()
    for u, v in zip(src, dst):
        if u != v:
            adj[u].add(v)
            if not directed:
                adj[v].add(u)
    return adj


# ── Matching ─────────────────────────────────────────────────────────────────


def greedy_maximal_matching(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Greedy maximal matching.

    Iterates edges in ``edge_index`` order and greedily adds each edge
    if neither endpoint is already matched.  Result is maximal (cannot
    add another edge) but not necessarily maximum-cardinality.

    Args:
        edge_index: ``LongTensor[2, E]`` (treated as undirected).
        num_nodes: Node count.

    Returns:
        ``LongTensor[2, M]`` of matched (src, dst) pairs.  Each
        unordered pair appears once.

    Complexity: O(E).
    """
    _validate(edge_index, num_nodes)
    matched = set()
    pairs_src, pairs_dst = [], []
    if not edge_index.numel():
        return torch.zeros((2, 0), dtype=torch.long)
    src = edge_index[0].cpu().tolist()
    dst = edge_index[1].cpu().tolist()
    seen_pairs: set = set()
    for u, v in zip(src, dst):
        if u == v:
            continue
        pair = (min(u, v), max(u, v))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        if u not in matched and v not in matched:
            matched.add(u)
            matched.add(v)
            pairs_src.append(u)
            pairs_dst.append(v)
    if not pairs_src:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([pairs_src, pairs_dst], dtype=torch.long)


def bipartite_greedy_matching(
    edge_index: torch.Tensor,
    num_nodes_left: int,
    num_nodes_right: int,
) -> torch.Tensor:
    """Greedy matching for a bipartite graph.

    Left nodes have IDs ``[0, num_nodes_left)``; right nodes have IDs
    ``[num_nodes_left, num_nodes_left + num_nodes_right)``.

    Args:
        edge_index: ``LongTensor[2, E]`` — only edges from left to right.
        num_nodes_left: Number of left-side nodes.
        num_nodes_right: Number of right-side nodes.

    Returns:
        ``LongTensor[2, M]`` of matched pairs.

    Complexity: O(E).
    """
    N = num_nodes_left + num_nodes_right
    _validate(edge_index, N)
    matched_left: set = set()
    matched_right: set = set()
    pairs_src, pairs_dst = [], []
    if not edge_index.numel():
        return torch.zeros((2, 0), dtype=torch.long)
    src = edge_index[0].cpu().tolist()
    dst = edge_index[1].cpu().tolist()
    for u, v in zip(src, dst):
        if u >= num_nodes_left or v < num_nodes_left:
            continue
        if u not in matched_left and v not in matched_right:
            matched_left.add(u)
            matched_right.add(v)
            pairs_src.append(u)
            pairs_dst.append(v)
    if not pairs_src:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([pairs_src, pairs_dst], dtype=torch.long)


# ── Coloring ─────────────────────────────────────────────────────────────────


def greedy_coloring(
    edge_index: torch.Tensor,
    num_nodes: int,
    strategy: str = "natural",
) -> Tuple[torch.Tensor, int]:
    """Greedy graph coloring.

    Assigns colors to nodes greedily so that no two adjacent nodes share
    a color.  The result is a valid coloring but not necessarily
    minimum-chromatic.

    Args:
        edge_index: ``LongTensor[2, E]`` (treated as undirected).
        num_nodes: Node count.
        strategy: Node ordering strategy: ``"natural"`` (ID order) or
            ``"largest_first"`` (highest degree first).

    Returns:
        ``(colors, num_colors)`` — ``LongTensor[N]`` of color IDs in
        ``[0, num_colors)`` and the total number of colors used.

    Complexity: O(N + E).
    """
    _validate(edge_index, num_nodes)
    adj = _build_adj(edge_index, num_nodes, directed=False)
    if strategy == "largest_first":
        order = sorted(range(num_nodes), key=lambda v: -len(adj[v]))
    else:
        order = list(range(num_nodes))
    colors = [-1] * num_nodes
    for v in order:
        used = {colors[u] for u in adj[v] if colors[u] >= 0}
        c = 0
        while c in used:
            c += 1
        colors[v] = c
    num_colors = max(colors) + 1 if colors else 0
    return torch.tensor(colors, dtype=torch.long), int(num_colors)


def welsh_powell_coloring(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> Tuple[torch.Tensor, int]:
    """Welsh-Powell greedy graph coloring (largest-first order).

    Equivalent to ``greedy_coloring(..., strategy='largest_first')``.

    Returns:
        ``(colors, num_colors)`` — same convention as :func:`greedy_coloring`.
    """
    return greedy_coloring(edge_index, num_nodes, strategy="largest_first")


# ── Clique / independent set ──────────────────────────────────────────────────


def greedy_maximal_independent_set(
    edge_index: torch.Tensor,
    num_nodes: int,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Greedy maximal independent set.

    Iterates nodes (in a seed-controlled order) and greedily adds each
    node if none of its neighbours are already in the set.  Result is
    maximal but not necessarily maximum-cardinality.

    Args:
        edge_index: ``LongTensor[2, E]`` (treated as undirected).
        num_nodes: Node count.
        seed: Optional order seed (None = natural order 0..N-1).

    Returns:
        ``LongTensor[K]`` of independent node IDs.

    Complexity: O(N + E).
    """
    _validate(edge_index, num_nodes)
    adj = _build_adj(edge_index, num_nodes, directed=False)
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    order = (
        torch.randperm(num_nodes, generator=gen).tolist()
        if seed is not None else list(range(num_nodes))
    )
    in_set = [False] * num_nodes
    excluded = [False] * num_nodes
    for v in order:
        if not excluded[v]:
            in_set[v] = True
            for u in adj[v]:
                excluded[u] = True
    selected = [v for v in range(num_nodes) if in_set[v]]
    if not selected:
        return torch.zeros(0, dtype=torch.long)
    return torch.tensor(selected, dtype=torch.long)


def enumerate_maximal_cliques(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_nodes: int = _CLIQUE_SIZE,
) -> List[FrozenSet[int]]:
    """Enumerate all maximal cliques (Bron-Kerbosch with pivot).

    Args:
        edge_index: ``LongTensor[2, E]`` (treated as undirected).
        num_nodes: Node count.
        max_nodes: Size guard (default 50).

    Returns:
        List of ``frozenset``s of node IDs, each forming a maximal clique.

    Complexity: O(3^{N/3}) worst-case; practical runtime is much lower for
    sparse graphs.

    Raises:
        ValueError: If ``num_nodes > max_nodes``.
    """
    _validate(edge_index, num_nodes)
    if num_nodes > max_nodes:
        raise ValueError(
            f"enumerate_maximal_cliques: num_nodes={num_nodes} > max_nodes={max_nodes}. "
            "Clique enumeration is exponential; increase max_nodes with caution."
        )
    adj = _build_adj(edge_index, num_nodes, directed=False)
    cliques: List[FrozenSet[int]] = []

    def bron_kerbosch(R: set, P: set, X: set) -> None:
        if not P and not X:
            cliques.append(frozenset(R))
            return
        # Choose pivot with maximum |P ∩ N(pivot)|.
        pivot = max(P | X, key=lambda v: len(adj[v] & P))
        for v in list(P - adj[pivot]):
            Nv = adj[v]
            bron_kerbosch(R | {v}, P & Nv, X & Nv)
            P.remove(v)
            X.add(v)

    bron_kerbosch(set(), set(range(num_nodes)), set())
    return cliques


# ── Max-flow / min-cut ────────────────────────────────────────────────────────


def edmonds_karp_max_flow(
    edge_index: torch.Tensor,
    num_nodes: int,
    capacity: torch.Tensor,
    source: int,
    sink: int,
) -> Tuple[float, Dict[Tuple[int, int], float]]:
    """Edmonds-Karp max-flow algorithm (BFS-based Ford-Fulkerson).

    Args:
        edge_index: ``LongTensor[2, E]`` directed edges.
        num_nodes: Node count.
        capacity: ``FloatTensor[E]`` non-negative edge capacities.
        source: Source node ID.
        sink: Sink node ID.

    Returns:
        ``(max_flow_value, flow_dict)`` where ``flow_dict`` maps
        ``(u, v)`` → flow value for each edge.

    Raises:
        ValueError: If negative capacities, or source/sink out of range.

    Complexity: O(V × E²).  Guarded to num_nodes <= 500.
    """
    _validate(edge_index, num_nodes)
    if num_nodes > _MAXFLOW_SIZE:
        raise ValueError(
            f"edmonds_karp_max_flow: num_nodes={num_nodes} > {_MAXFLOW_SIZE}. "
            "Use a dedicated flow library for large graphs."
        )
    if not (0 <= source < num_nodes) or not (0 <= sink < num_nodes):
        raise ValueError(f"source={source} or sink={sink} out of range [0, {num_nodes})")
    if source == sink:
        return float("inf"), {}
    if capacity.numel() > 0 and float(capacity.min().item()) < 0:
        raise ValueError("edmonds_karp_max_flow requires non-negative capacities.")

    # Build capacity matrix (dict of dicts for sparse representation).
    cap: Dict[int, Dict[int, float]] = {i: {} for i in range(num_nodes)}
    src_list = edge_index[0].cpu().tolist()
    dst_list = edge_index[1].cpu().tolist()
    cap_list = capacity.float().cpu().tolist()
    for u, v, c in zip(src_list, dst_list, cap_list):
        cap[u][v] = cap[u].get(v, 0.0) + c
        cap[v].setdefault(u, 0.0)  # reverse edge

    flow: Dict[int, Dict[int, float]] = {i: {} for i in range(num_nodes)}

    def bfs_path(s: int, t: int) -> Optional[List[int]]:
        visited = {s}
        queue = deque([[s]])
        while queue:
            path = queue.popleft()
            u = path[-1]
            for v in cap[u]:
                residual = cap[u].get(v, 0.0) - flow[u].get(v, 0.0)
                if v not in visited and residual > 1e-12:
                    new_path = path + [v]
                    if v == t:
                        return new_path
                    visited.add(v)
                    queue.append(new_path)
        return None

    max_flow = 0.0
    while (path := bfs_path(source, sink)) is not None:
        # Bottleneck.
        bottleneck = min(
            cap[path[i]].get(path[i + 1], 0.0) - flow[path[i]].get(path[i + 1], 0.0)
            for i in range(len(path) - 1)
        )
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            flow[u][v] = flow[u].get(v, 0.0) + bottleneck
            flow[v][u] = flow[v].get(u, 0.0) - bottleneck
        max_flow += bottleneck

    # Build flat flow dict.
    flat_flow = {(u, v): f for u, d in flow.items() for v, f in d.items() if f > 1e-12}
    return max_flow, flat_flow


def min_cut_from_max_flow(
    edge_index: torch.Tensor,
    num_nodes: int,
    capacity: torch.Tensor,
    source: int,
    sink: int,
) -> Tuple[float, Set[int], Set[int]]:
    """Minimum cut via max-flow (max-flow min-cut theorem).

    Args:
        edge_index: ``LongTensor[2, E]`` directed edges.
        num_nodes: Node count.
        capacity: ``FloatTensor[E]`` non-negative capacities.
        source: Source node.
        sink: Sink node.

    Returns:
        ``(cut_value, S_set, T_set)`` where ``S_set`` contains the source
        side and ``T_set`` the sink side of the min cut.
    """
    max_flow, flow = edmonds_karp_max_flow(
        edge_index, num_nodes, capacity, source, sink
    )
    # BFS on residual graph from source.
    cap: Dict[int, Dict[int, float]] = {i: {} for i in range(num_nodes)}
    src_list = edge_index[0].cpu().tolist()
    dst_list = edge_index[1].cpu().tolist()
    cap_list = capacity.float().cpu().tolist()
    for u, v, c in zip(src_list, dst_list, cap_list):
        cap[u][v] = cap[u].get(v, 0.0) + c
        cap[v].setdefault(u, 0.0)

    visited: Set[int] = {source}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in cap[u]:
            residual = cap[u].get(v, 0.0) - flow.get((u, v), 0.0)
            if v not in visited and residual > 1e-12:
                visited.add(v)
                queue.append(v)
    S = visited
    T = set(range(num_nodes)) - S
    return max_flow, S, T


# ── WL isomorphism heuristic ──────────────────────────────────────────────────


def wl_isomorphism_test(
    edge_index_1: torch.Tensor,
    num_nodes_1: int,
    edge_index_2: torch.Tensor,
    num_nodes_2: int,
    node_labels_1: Optional[List[int]] = None,
    node_labels_2: Optional[List[int]] = None,
    num_iterations: int = 5,
) -> bool:
    """Weisfeiler-Lehman graph isomorphism test (necessary but not sufficient).

    Returns ``False`` if graphs are definitely NOT isomorphic; returns
    ``True`` if they *might* be isomorphic (no false negatives, possible
    false positives).

    Args:
        edge_index_1, num_nodes_1: First graph.
        edge_index_2, num_nodes_2: Second graph.
        node_labels_1, node_labels_2: Optional initial integer labels.
        num_iterations: WL refinement rounds.

    Returns:
        ``True`` if WL certificates match (graphs *might* be isomorphic).
        ``False`` if certificates differ (graphs are definitely not isomorphic).
    """
    from .kernels import weisfeiler_lehman_labels
    h1 = weisfeiler_lehman_labels(edge_index_1, num_nodes_1, node_labels_1, num_iterations)
    h2 = weisfeiler_lehman_labels(edge_index_2, num_nodes_2, node_labels_2, num_iterations)
    # Compare multisets of labels at each iteration.
    for labels_a, labels_b in zip(h1, h2):
        if sorted(labels_a) != sorted(labels_b):
            return False
    return True


# ── Report writer ─────────────────────────────────────────────────────────────


def write_algorithm_report(
    path: str,
    algorithm: str,
    **fields,
) -> str:
    """Write a graph algorithm result as a JSON dashboard artifact.

    Args:
        path: Output file path.
        algorithm: Algorithm name (e.g. ``"max_flow"``).
        **fields: Additional key-value result fields.

    Returns:
        Resolved path string.
    """
    import json
    import os
    import tempfile
    from pathlib import Path
    payload = {"algorithm": algorithm}
    payload.update({k: v for k, v in fields.items()})
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
