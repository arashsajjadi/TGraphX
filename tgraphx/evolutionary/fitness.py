"""Fitness functions for evolutionary graph optimization.

Each function takes a GraphGenome and returns a float (higher = better).

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from .genome import GraphGenome

__all__ = [
    "connectivity_fitness",
    "density_fitness",
    "clustering_fitness",
    "motif_count_fitness",
    "constraint_penalty",
    "composite_fitness",
]


def connectivity_fitness(genome: GraphGenome) -> float:
    """Fraction of reachable node pairs (weakly connected reachability).

    f = |{(i,j) : j reachable from i}| / (N * (N-1))

    For N <= 1, returns 1.0.

    Args:
        genome: Input genome.

    Returns:
        Float in [0, 1].
    """
    n = genome.num_nodes
    if n <= 1:
        return 1.0

    # BFS from each node
    adj = {i: [] for i in range(n)}
    for s, d in zip(genome.edge_index[0].tolist(), genome.edge_index[1].tolist()):
        adj[s].append(d)
        adj[d].append(s)  # undirected reachability

    total_reachable = 0
    for start in range(n):
        visited = {start}
        queue = [start]
        while queue:
            v = queue.pop()
            for u in adj[v]:
                if u not in visited:
                    visited.add(u)
                    queue.append(u)
        total_reachable += len(visited) - 1  # exclude self

    return total_reachable / (n * (n - 1))


def density_fitness(genome: GraphGenome, target_density: float = 0.5) -> float:
    """Closeness to target edge density.

    f = 1 - |density(G) - target_density|

    density = |E| / (N * (N-1))

    Args:
        genome: Input genome.
        target_density: Target edge density in [0, 1].

    Returns:
        Float in [0, 1].
    """
    n = genome.num_nodes
    if n <= 1:
        actual = 0.0
    else:
        actual = genome.num_edges / (n * (n - 1))
    return 1.0 - abs(actual - target_density)


def clustering_fitness(genome: GraphGenome) -> float:
    """Average clustering coefficient.

    For node v with degree k_v:
        C(v) = |{(u,w) in E : u,w in N(v)}| / (k_v * (k_v - 1) / 2)

    f = mean_v C(v)

    Args:
        genome: Input genome.

    Returns:
        Float in [0, 1].
    """
    n = genome.num_nodes
    if n == 0:
        return 0.0

    adj = {i: set() for i in range(n)}
    for s, d in zip(genome.edge_index[0].tolist(), genome.edge_index[1].tolist()):
        adj[s].add(d)
        adj[d].add(s)

    coeffs = []
    for v in range(n):
        nb = list(adj[v])
        k = len(nb)
        if k < 2:
            coeffs.append(0.0)
            continue
        tri = sum(
            1 for i in range(len(nb))
            for j in range(i + 1, len(nb))
            if nb[j] in adj[nb[i]]
        )
        coeffs.append(tri / (k * (k - 1) / 2))

    return sum(coeffs) / len(coeffs) if coeffs else 0.0


def motif_count_fitness(genome: GraphGenome, motif_type: str = "triangle") -> float:
    """Count of specified motifs.

    Normalized by the maximum possible count.

    Args:
        genome: Input genome.
        motif_type: 'triangle' or 'wedge'.

    Returns:
        Non-negative float.
    """
    n = genome.num_nodes
    if n == 0:
        return 0.0

    adj = {i: set() for i in range(n)}
    for s, d in zip(genome.edge_index[0].tolist(), genome.edge_index[1].tolist()):
        adj[s].add(d)
        adj[d].add(s)

    if motif_type == "triangle":
        count = 0
        for v in range(n):
            nb = list(adj[v])
            for i in range(len(nb)):
                for j in range(i + 1, len(nb)):
                    if nb[j] in adj[nb[i]]:
                        count += 1
        return count / 3.0

    elif motif_type == "wedge":
        count = 0
        for v in range(n):
            k = len(adj[v])
            count += k * (k - 1) // 2
        return float(count)

    else:
        raise ValueError(f"Unknown motif_type={motif_type!r}. Choose 'triangle' or 'wedge'.")


def constraint_penalty(
    genome: GraphGenome,
    max_nodes: Optional[int] = None,
    max_edges: Optional[int] = None,
    no_self_loops: bool = True,
    connected: bool = False,
    acyclic: bool = False,
) -> float:
    """Compute constraint violation penalty (negative contribution to fitness).

    Returns a non-positive value: 0.0 if all constraints are satisfied,
    negative otherwise.

    Args:
        genome: Input genome.
        max_nodes: Maximum allowed nodes.
        max_edges: Maximum allowed edges.
        no_self_loops: Penalize self-loops.
        connected: Penalize disconnected graphs.
        acyclic: Penalize cycles.

    Returns:
        Non-positive float penalty.
    """
    penalty = 0.0
    n = genome.num_nodes
    e = genome.num_edges

    if max_nodes is not None and n > max_nodes:
        penalty -= float(n - max_nodes)
    if max_edges is not None and e > max_edges:
        penalty -= float(e - max_edges)

    if no_self_loops and e > 0:
        src_arr = genome.edge_index[0]
        dst_arr = genome.edge_index[1]
        self_loop_count = int((src_arr == dst_arr).sum().item())
        penalty -= float(self_loop_count)

    if connected and n > 1:
        # BFS
        adj = {i: [] for i in range(n)}
        for s, d in zip(genome.edge_index[0].tolist(), genome.edge_index[1].tolist()):
            adj[s].append(d)
            adj[d].append(s)
        visited = {0}
        queue = [0]
        while queue:
            v = queue.pop()
            for u in adj[v]:
                if u not in visited:
                    visited.add(u)
                    queue.append(u)
        unreachable = n - len(visited)
        penalty -= float(unreachable)

    return penalty


def composite_fitness(
    genome: GraphGenome,
    components: List[Tuple[Callable, float]],
) -> float:
    """Weighted sum of multiple fitness components.

    f = sum_i w_i * f_i(G)

    Args:
        genome: Input genome.
        components: List of (fitness_fn, weight) pairs.

    Returns:
        Weighted sum of component scores.
    """
    total = 0.0
    for fn, weight in components:
        total += weight * fn(genome)
    return total
