"""Multi-objective optimization utilities (NSGA-II).

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch

from .genome import GraphGenome

__all__ = [
    "pareto_dominates",
    "non_dominated_sort",
    "crowding_distance",
    "nsga2_select",
    "ParetoFront",
    "compute_hypervolume_2d",
]


def pareto_dominates(f_a: List[float], f_b: List[float]) -> bool:
    """Check if fitness vector f_a dominates f_b.

    a dominates b iff:
        - For all i: f_a[i] >= f_b[i]  (a is at least as good in every objective)
        - For some j: f_a[j] > f_b[j]  (a is strictly better in at least one)

    Args:
        f_a: Fitness vector for individual a.
        f_b: Fitness vector for individual b.

    Returns:
        True if a dominates b.
    """
    if len(f_a) != len(f_b):
        raise ValueError(f"Fitness vectors must have equal length, got {len(f_a)} vs {len(f_b)}")
    at_least_as_good = all(a >= b for a, b in zip(f_a, f_b))
    strictly_better = any(a > b for a, b in zip(f_a, f_b))
    return at_least_as_good and strictly_better


def non_dominated_sort(
    population: List[GraphGenome],
    fitness_vectors: List[List[float]],
) -> List[List[int]]:
    """NSGA-II non-dominated sort.

    Returns a list of Pareto fronts, where front 0 contains indices of
    non-dominated individuals, front 1 contains individuals dominated only
    by front 0, etc.

    Complexity: O(M * N^2) where M = number of objectives, N = population size.

    Args:
        population: List of genomes (indices correspond to fitness_vectors).
        fitness_vectors: List of fitness vectors (one per individual).

    Returns:
        List of fronts, each front is a list of indices.
    """
    n = len(population)
    if n == 0:
        return []

    dominated_by: List[int] = [0] * n  # how many individuals dominate this one
    dominates: List[List[int]] = [[] for _ in range(n)]  # who this individual dominates

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if pareto_dominates(fitness_vectors[i], fitness_vectors[j]):
                dominates[i].append(j)
            elif pareto_dominates(fitness_vectors[j], fitness_vectors[i]):
                dominated_by[i] += 1

    fronts: List[List[int]] = []
    current_front = [i for i in range(n) if dominated_by[i] == 0]
    fronts.append(current_front)

    while True:
        next_front = []
        for i in fronts[-1]:
            for j in dominates[i]:
                dominated_by[j] -= 1
                if dominated_by[j] == 0:
                    next_front.append(j)
        if not next_front:
            break
        fronts.append(next_front)

    return fronts


def crowding_distance(
    fitness_vectors: List[List[float]],
    front_indices: List[int],
) -> List[float]:
    """Compute NSGA-II crowding distance for a Pareto front.

    Crowding distance for individual i:
        d_i = sum_m (f_m_next - f_m_prev) / (f_m_max - f_m_min)

    Boundary individuals get infinite crowding distance.

    Args:
        fitness_vectors: All fitness vectors.
        front_indices: Indices of individuals in this front.

    Returns:
        List of crowding distances (same length as front_indices).
    """
    if not front_indices:
        return []

    n_front = len(front_indices)
    distances = [0.0] * n_front

    if n_front <= 2:
        return [float("inf")] * n_front

    n_obj = len(fitness_vectors[0])

    for m in range(n_obj):
        # Sort by objective m
        sorted_front = sorted(range(n_front), key=lambda k: fitness_vectors[front_indices[k]][m])
        distances[sorted_front[0]] = float("inf")
        distances[sorted_front[-1]] = float("inf")

        f_min = fitness_vectors[front_indices[sorted_front[0]]][m]
        f_max = fitness_vectors[front_indices[sorted_front[-1]]][m]

        if f_max <= f_min:
            continue

        for k in range(1, n_front - 1):
            prev_val = fitness_vectors[front_indices[sorted_front[k - 1]]][m]
            next_val = fitness_vectors[front_indices[sorted_front[k + 1]]][m]
            distances[sorted_front[k]] += (next_val - prev_val) / (f_max - f_min)

    return distances


def nsga2_select(
    population: List[GraphGenome],
    offspring: List[GraphGenome],
    fitness_fn_list: List[Callable[[GraphGenome], float]],
    n_select: int,
) -> List[GraphGenome]:
    """NSGA-II selection: combine population + offspring, sort, select top n.

    Args:
        population: Current population.
        offspring: Offspring genomes.
        fitness_fn_list: List of fitness functions (one per objective).
        n_select: Number of individuals to select.

    Returns:
        List of n_select selected genomes.
    """
    combined = list(population) + list(offspring)
    fitness_vectors = [
        [fn(g) for fn in fitness_fn_list]
        for g in combined
    ]

    fronts = non_dominated_sort(combined, fitness_vectors)
    selected: List[GraphGenome] = []

    for front in fronts:
        if len(selected) + len(front) <= n_select:
            selected.extend([combined[i].clone() for i in front])
        else:
            # Fill remaining slots using crowding distance
            needed = n_select - len(selected)
            distances = crowding_distance(fitness_vectors, front)
            sorted_by_dist = sorted(
                range(len(front)),
                key=lambda k: distances[k],
                reverse=True,
            )
            for k in sorted_by_dist[:needed]:
                selected.append(combined[front[k]].clone())
            break

        if len(selected) >= n_select:
            break

    return selected[:n_select]


@dataclass
class ParetoFront:
    """Stores the Pareto-optimal front from multi-objective optimization.

    Args:
        genomes: List of Pareto-optimal genomes.
        fitness_vectors: Corresponding fitness vectors.
        crowding_distances: Crowding distances for diversity preservation.
    """

    genomes: List[GraphGenome] = field(default_factory=list)
    fitness_vectors: List[List[float]] = field(default_factory=list)
    crowding_distances: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.genomes)

    def best_by_objective(self, obj_idx: int) -> Tuple[GraphGenome, List[float]]:
        """Return the genome with highest value for objective obj_idx."""
        if not self.genomes:
            raise ValueError("ParetoFront is empty")
        best_idx = max(range(len(self.genomes)), key=lambda i: self.fitness_vectors[i][obj_idx])
        return self.genomes[best_idx], self.fitness_vectors[best_idx]


def compute_hypervolume_2d(
    pareto_front: "ParetoFront",
    reference_point: Tuple[float, float],
) -> float:
    """Compute 2D hypervolume indicator.

    Only valid for 2-objective problems.

    The hypervolume is the area of the space dominated by the Pareto front
    and bounded above by reference_point.

    HV = area dominated by P and bounded by (r1, r2)

    Args:
        pareto_front: ParetoFront with 2-objective fitness vectors.
        reference_point: (ref_f1, ref_f2) — must be dominated by all Pareto points.

    Returns:
        Hypervolume indicator (float, higher = better front).
    """
    if not pareto_front.genomes:
        return 0.0

    vecs = pareto_front.fitness_vectors
    for v in vecs:
        if len(v) != 2:
            raise ValueError(
                f"compute_hypervolume_2d only supports 2-objective problems, "
                f"got vector of length {len(v)}"
            )

    r1, r2 = reference_point

    # Sort by first objective descending
    sorted_vecs = sorted(vecs, key=lambda v: v[0], reverse=True)

    hv = 0.0
    prev_f2 = r2  # start from reference

    for f1, f2 in sorted_vecs:
        if f1 <= r1 or f2 <= r2:
            # This point is dominated by reference (reference should be a nadir)
            # Skip if below reference
            if f2 > prev_f2:
                continue
        width = f1 - r1
        height = prev_f2 - f2
        if width > 0 and height > 0:
            hv += width * height
        if f2 < prev_f2:
            prev_f2 = f2

    return max(0.0, hv)
