"""Synthetic graph generators for graph mining and learning.

All generators are deterministic when a seed is provided.
They return ``(edge_index, num_nodes)`` pairs compatible with TGraphX
``Graph`` objects.

Stability: Beta (v0.4.2+).

No large graph is created silently — size guards raise clear errors.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

__all__ = [
    "erdos_renyi_graph",
    "barabasi_albert_graph",
    "stochastic_block_model_graph",
    "watts_strogatz_graph",
    "random_geometric_graph",
    "planted_partition_graph",
    "grid_2d_graph",
    "complete_graph",
    "cycle_graph",
    "path_graph",
    "star_graph",
    "karate_club_graph",
    "synthetic_anomaly_graph",
    "motif_injected_graph",
]

_MAX_NODES = 10_000


def _rng(seed: Optional[int] = None) -> torch.Generator:
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(int(seed))
    return g


def _undirected(src: List[int], dst: List[int]) -> torch.Tensor:
    """Build undirected edge_index from directed edge lists."""
    all_src = src + dst
    all_dst = dst + src
    if not all_src:
        return torch.zeros((2, 0), dtype=torch.long)
    ei = torch.tensor([all_src, all_dst], dtype=torch.long)
    return torch.unique(ei, dim=1)


# ── Classical generators ─────────────────────────────────────────────────────


def erdos_renyi_graph(
    num_nodes: int,
    p: float,
    directed: bool = False,
    self_loops: bool = False,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """Erdős-Rényi G(n, p) random graph.

    Args:
        num_nodes: Number of nodes.
        p: Edge probability in (0, 1).
        directed: When ``True``, generate directed edges.
        self_loops: When ``True``, allow self-loops.
        seed: Optional RNG seed.

    Returns:
        ``(edge_index, num_nodes)`` tuple.
    """
    if num_nodes > _MAX_NODES:
        raise ValueError(f"num_nodes={num_nodes} > {_MAX_NODES}")
    N = int(num_nodes)
    rng = _rng(seed)
    src_list: List[int] = []
    dst_list: List[int] = []
    for i in range(N):
        start = 0 if directed else i + 1
        for j in range(start, N):
            if not self_loops and i == j:
                continue
            if torch.rand(1, generator=rng).item() < p:
                src_list.append(i)
                dst_list.append(j)
                if not directed:
                    src_list.append(j)
                    dst_list.append(i)
    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long), N
    return torch.tensor([src_list, dst_list], dtype=torch.long), N


def barabasi_albert_graph(
    num_nodes: int,
    m: int,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """Barabási-Albert preferential attachment graph.

    Grows a network by adding ``m`` edges per new node using
    preferential attachment.

    Args:
        num_nodes: Final number of nodes.
        m: Number of edges each new node attaches to.
        seed: Optional RNG seed.

    Returns:
        Undirected ``(edge_index, num_nodes)`` tuple.
    """
    if num_nodes > _MAX_NODES:
        raise ValueError(f"num_nodes={num_nodes} > {_MAX_NODES}")
    N = int(num_nodes)
    m = min(int(m), N - 1)
    if m <= 0 or N <= 1:
        return torch.zeros((2, 0), dtype=torch.long), N
    rng = _rng(seed)
    src_list: List[int] = []
    dst_list: List[int] = []
    # Start with a clique of size m+1.
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            src_list.extend([i, j])
            dst_list.extend([j, i])
    degree = [0] * N
    for i in range(m + 1):
        degree[i] = m
    for new_node in range(m + 1, N):
        # Sample m targets proportional to degree.
        total = sum(degree[:new_node]) or 1
        targets: set = set()
        attempts = 0
        while len(targets) < m and attempts < 10 * m * N:
            r = torch.rand(1, generator=rng).item() * total
            cumsum = 0.0
            for cand in range(new_node):
                cumsum += degree[cand]
                if r < cumsum:
                    if cand not in targets:
                        targets.add(cand)
                    break
            attempts += 1
        for t in targets:
            src_list.extend([new_node, t])
            dst_list.extend([t, new_node])
            degree[new_node] += 1
            degree[t] += 1
    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long), N
    ei = torch.tensor([src_list, dst_list], dtype=torch.long)
    return torch.unique(ei, dim=1), N


def stochastic_block_model_graph(
    block_sizes: List[int],
    p_in: float,
    p_out: float,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """Stochastic Block Model (SBM) graph.

    Args:
        block_sizes: List of community sizes.
        p_in: Intra-community edge probability.
        p_out: Inter-community edge probability.
        seed: Optional RNG seed.

    Returns:
        ``(edge_index, num_nodes, community_labels)`` where
        ``community_labels`` is ``LongTensor[N]``.
    """
    N = sum(block_sizes)
    if N > _MAX_NODES:
        raise ValueError(f"total num_nodes={N} > {_MAX_NODES}")
    rng = _rng(seed)
    # Build community assignments.
    labels: List[int] = []
    for c, size in enumerate(block_sizes):
        labels.extend([c] * size)
    labels_t = torch.tensor(labels, dtype=torch.long)
    src_list: List[int] = []
    dst_list: List[int] = []
    for i in range(N):
        for j in range(i + 1, N):
            p = p_in if labels[i] == labels[j] else p_out
            if torch.rand(1, generator=rng).item() < p:
                src_list.extend([i, j])
                dst_list.extend([j, i])
    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long), N, labels_t
    ei = torch.tensor([src_list, dst_list], dtype=torch.long)
    return torch.unique(ei, dim=1), N, labels_t


def watts_strogatz_graph(
    num_nodes: int,
    k: int,
    p: float,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """Watts-Strogatz small-world graph.

    Args:
        num_nodes: Number of nodes.
        k: Each node is connected to k nearest neighbours in a ring.
        p: Probability of rewiring each edge.
        seed: Optional RNG seed.

    Returns:
        Undirected ``(edge_index, num_nodes)`` tuple.
    """
    if num_nodes > _MAX_NODES:
        raise ValueError(f"num_nodes={num_nodes} > {_MAX_NODES}")
    N = int(num_nodes)
    k = min(int(k), N - 1)
    rng = _rng(seed)
    src_list: List[int] = []
    dst_list: List[int] = []
    # Start with ring lattice.
    for i in range(N):
        for j in range(1, k // 2 + 1):
            src_list.extend([i, (i + j) % N])
            dst_list.extend([(i + j) % N, i])
    # Rewire edges.
    rewired_src, rewired_dst = list(src_list), list(dst_list)
    existing = set(zip(rewired_src, rewired_dst))
    for idx in range(0, len(src_list), 2):  # iterate unique directed pairs
        if torch.rand(1, generator=rng).item() < p:
            u = src_list[idx]
            for _ in range(N):
                w = int(torch.randint(N, (1,), generator=rng).item())
                if w != u and (u, w) not in existing:
                    existing.discard((u, src_list[idx + 1]))
                    existing.discard((src_list[idx + 1], u))
                    rewired_src[idx] = u
                    rewired_dst[idx] = w
                    rewired_src[idx + 1] = w
                    rewired_dst[idx + 1] = u
                    existing.add((u, w))
                    existing.add((w, u))
                    break
    if not rewired_src:
        return torch.zeros((2, 0), dtype=torch.long), N
    ei = torch.tensor([rewired_src, rewired_dst], dtype=torch.long)
    return torch.unique(ei, dim=1), N


def random_geometric_graph(
    num_nodes: int,
    radius: float,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """2-D random geometric graph: connect nodes within ``radius``.

    Args:
        num_nodes: Number of nodes (placed uniformly in [0,1]²).
        radius: Connectivity radius.
        seed: Optional RNG seed.

    Returns:
        Undirected ``(edge_index, num_nodes)`` tuple.
    """
    if num_nodes > _MAX_NODES:
        raise ValueError(f"num_nodes={num_nodes} > {_MAX_NODES}")
    N = int(num_nodes)
    rng = _rng(seed)
    pos = torch.rand(N, 2, generator=rng)
    src_list, dst_list = [], []
    r2 = radius ** 2
    for i in range(N):
        for j in range(i + 1, N):
            if float(((pos[i] - pos[j]) ** 2).sum().item()) < r2:
                src_list.extend([i, j])
                dst_list.extend([j, i])
    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long), N
    return torch.tensor([src_list, dst_list], dtype=torch.long), N


def planted_partition_graph(
    num_communities: int,
    community_size: int,
    p_in: float,
    p_out: float,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """Planted partition graph (equal-size SBM).

    Returns:
        ``(edge_index, num_nodes, community_labels)``
    """
    return stochastic_block_model_graph(
        [community_size] * num_communities, p_in, p_out, seed=seed
    )


# ── Structural generators ─────────────────────────────────────────────────────


def grid_2d_graph(rows: int, cols: int) -> Tuple[torch.Tensor, int]:
    """2-D grid graph with 4-connectivity."""
    N = rows * cols
    src_list, dst_list = [], []
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            if c + 1 < cols:
                u = v + 1
                src_list.extend([v, u])
                dst_list.extend([u, v])
            if r + 1 < rows:
                u = v + cols
                src_list.extend([v, u])
                dst_list.extend([u, v])
    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long), N
    return torch.tensor([src_list, dst_list], dtype=torch.long), N


def complete_graph(num_nodes: int) -> Tuple[torch.Tensor, int]:
    """Complete undirected graph K_n."""
    N = int(num_nodes)
    src_list, dst_list = [], []
    for i in range(N):
        for j in range(i + 1, N):
            src_list.extend([i, j])
            dst_list.extend([j, i])
    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long), N
    return torch.tensor([src_list, dst_list], dtype=torch.long), N


def cycle_graph(num_nodes: int) -> Tuple[torch.Tensor, int]:
    """Undirected cycle graph C_n."""
    N = int(num_nodes)
    src_list, dst_list = [], []
    for i in range(N):
        j = (i + 1) % N
        src_list.extend([i, j])
        dst_list.extend([j, i])
    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long), N
    ei = torch.tensor([src_list, dst_list], dtype=torch.long)
    return torch.unique(ei, dim=1), N


def path_graph(num_nodes: int) -> Tuple[torch.Tensor, int]:
    """Undirected path graph P_n."""
    N = int(num_nodes)
    src_list, dst_list = [], []
    for i in range(N - 1):
        src_list.extend([i, i + 1])
        dst_list.extend([i + 1, i])
    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long), N
    return torch.tensor([src_list, dst_list], dtype=torch.long), N


def star_graph(num_nodes: int) -> Tuple[torch.Tensor, int]:
    """Undirected star graph with node 0 as hub."""
    N = int(num_nodes)
    src_list, dst_list = [], []
    for i in range(1, N):
        src_list.extend([0, i])
        dst_list.extend([i, 0])
    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long), N
    return torch.tensor([src_list, dst_list], dtype=torch.long), N


def karate_club_graph() -> Tuple[torch.Tensor, int]:
    """Zachary's karate club graph (34 nodes, 78 undirected edges).

    A classic social network benchmark.  Community labels:
    0-based faction membership.
    """
    # Hard-coded edge list (original Zachary 1977 data).
    edges = [
        (0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,10),(0,11),(0,12),
        (0,13),(0,17),(0,19),(0,21),(0,31),(1,2),(1,3),(1,7),(1,13),(1,17),
        (1,19),(1,21),(1,30),(2,3),(2,7),(2,8),(2,9),(2,13),(2,27),(2,28),
        (2,32),(3,7),(3,12),(3,13),(4,6),(4,10),(5,6),(5,10),(5,16),(6,16),
        (8,30),(8,32),(8,33),(9,33),(13,33),(14,32),(14,33),(15,32),(15,33),
        (18,32),(18,33),(19,33),(20,32),(20,33),(22,32),(22,33),(23,25),(23,27),
        (23,29),(23,32),(23,33),(24,25),(24,27),(24,31),(25,31),(26,29),(26,33),
        (27,33),(28,31),(28,33),(29,32),(29,33),(30,32),(30,33),(31,32),(31,33),
        (32,33),
    ]
    src_list, dst_list = [], []
    for u, v in edges:
        src_list.extend([u, v])
        dst_list.extend([v, u])
    return torch.tensor([src_list, dst_list], dtype=torch.long), 34


# ── Mining-specific generators ────────────────────────────────────────────────


def synthetic_anomaly_graph(
    num_nodes: int,
    p_normal: float = 0.1,
    num_anomalous: int = 3,
    anomaly_degree_boost: int = 5,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """Graph with injected high-degree anomalous nodes.

    Returns:
        ``(edge_index, num_nodes, anomaly_mask)`` where
        ``anomaly_mask`` is ``BoolTensor[N]``.
    """
    ei_base, N = erdos_renyi_graph(num_nodes, p_normal, seed=seed)
    rng = _rng(None if seed is None else seed + 1)
    anomaly_nodes = torch.randperm(N, generator=rng)[:num_anomalous].tolist()
    anomaly_mask = torch.zeros(N, dtype=torch.bool)
    for a in anomaly_nodes:
        anomaly_mask[a] = True
    extra_src, extra_dst = [], []
    for a in anomaly_nodes:
        targets = [v for v in range(N) if v != a][:anomaly_degree_boost]
        for t in targets:
            extra_src.extend([a, t])
            extra_dst.extend([t, a])
    if extra_src:
        extra_ei = torch.tensor([extra_src, extra_dst], dtype=torch.long)
        ei = torch.cat([ei_base, extra_ei], dim=1)
        ei = torch.unique(ei, dim=1)
    else:
        ei = ei_base
    return ei, N, anomaly_mask


def motif_injected_graph(
    num_nodes: int,
    p_base: float = 0.05,
    num_triangles: int = 5,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """Random graph with injected triangle motifs.

    Returns:
        ``(edge_index, num_nodes, triangle_nodes)`` where
        ``triangle_nodes`` is ``LongTensor[num_triangles*3]``.
    """
    ei_base, N = erdos_renyi_graph(num_nodes, p_base, seed=seed)
    rng = _rng(None if seed is None else seed + 7)
    extra_src, extra_dst = [], []
    tri_nodes = []
    for _ in range(num_triangles):
        perm = torch.randperm(N, generator=rng)[:3].tolist()
        a, b, c = perm
        tri_nodes.extend([a, b, c])
        extra_src.extend([a, b, b, c, a, c])
        extra_dst.extend([b, a, c, b, c, a])
    if extra_src:
        extra_ei = torch.tensor([extra_src, extra_dst], dtype=torch.long)
        ei = torch.cat([ei_base, extra_ei], dim=1)
        ei = torch.unique(ei, dim=1)
    else:
        ei = ei_base
    tri_t = torch.tensor(tri_nodes, dtype=torch.long) if tri_nodes else torch.zeros(0, dtype=torch.long)
    return ei, N, tri_t
