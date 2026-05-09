"""Selection operators for evolutionary graph optimization.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .genome import GraphGenome

__all__ = [
    "tournament_selection",
    "roulette_wheel_selection",
    "rank_selection",
    "elitism_selection",
    "diversity_preserving_selection",
]


def tournament_selection(
    population: List[GraphGenome],
    fitness_scores: List[float],
    k: int = 3,
    generator: Optional[torch.Generator] = None,
) -> List[GraphGenome]:
    """Tournament selection.

    Runs len(population) tournaments of size k; winner is individual with highest fitness.

    Args:
        population: List of genomes.
        fitness_scores: Fitness value for each genome (higher = better).
        k: Tournament size.
        generator: Optional RNG.

    Returns:
        List of selected genomes (same length as population).
    """
    n = len(population)
    selected = []
    for _ in range(n):
        contestants = torch.randperm(n, generator=generator)[:k].tolist()
        winner = max(contestants, key=lambda i: fitness_scores[i])
        selected.append(population[winner].clone())
    return selected


def roulette_wheel_selection(
    population: List[GraphGenome],
    fitness_scores: List[float],
    n: int,
    generator: Optional[torch.Generator] = None,
) -> List[GraphGenome]:
    """Roulette-wheel (fitness-proportional) selection.

    Probability of selecting individual i: p_i = f_i / sum(f)

    Args:
        population: List of genomes.
        fitness_scores: Fitness values (must be non-negative).
        n: Number of individuals to select.
        generator: Optional RNG.

    Returns:
        List of n selected genomes.
    """
    scores = torch.tensor(fitness_scores, dtype=torch.float)
    min_score = scores.min().item()
    if min_score < 0:
        scores = scores - min_score + 1e-9
    total = scores.sum().item()
    if total <= 0:
        probs = torch.ones(len(population)) / len(population)
    else:
        probs = scores / total

    indices = torch.multinomial(probs, n, replacement=True, generator=generator).tolist()
    return [population[i].clone() for i in indices]


def rank_selection(
    population: List[GraphGenome],
    fitness_scores: List[float],
    n: int,
    generator: Optional[torch.Generator] = None,
) -> List[GraphGenome]:
    """Rank-proportional selection.

    Assigns selection probability proportional to rank (1 = worst, N = best).

    Args:
        population: List of genomes.
        fitness_scores: Fitness values.
        n: Number to select.
        generator: Optional RNG.

    Returns:
        List of n selected genomes.
    """
    N = len(population)
    order = sorted(range(N), key=lambda i: fitness_scores[i])
    ranks = [0.0] * N
    for rank, idx in enumerate(order):
        ranks[idx] = float(rank + 1)

    ranks_t = torch.tensor(ranks)
    probs = ranks_t / ranks_t.sum()

    indices = torch.multinomial(probs, n, replacement=True, generator=generator).tolist()
    return [population[i].clone() for i in indices]


def elitism_selection(
    population: List[GraphGenome],
    fitness_scores: List[float],
    n_elite: int,
) -> List[GraphGenome]:
    """Return the top-n_elite individuals by fitness.

    Args:
        population: List of genomes.
        fitness_scores: Fitness values.
        n_elite: Number of elites to select.

    Returns:
        List of n_elite genomes (best first).
    """
    n = min(n_elite, len(population))
    order = sorted(range(len(population)), key=lambda i: fitness_scores[i], reverse=True)
    return [population[i].clone() for i in order[:n]]


def diversity_preserving_selection(
    population: List[GraphGenome],
    fitness_scores: List[float],
    n: int,
    wl_cache: Optional[Dict[int, str]] = None,
    generator: Optional[torch.Generator] = None,
) -> List[GraphGenome]:
    """Select individuals balancing fitness and diversity.

    Computes a combined score = 0.5 * norm_fitness + 0.5 * norm_uniqueness.
    Uniqueness is measured by WL hash frequency (rarer = more diverse).

    Args:
        population: List of genomes.
        fitness_scores: Fitness values.
        n: Number to select.
        wl_cache: Optional dict for caching WL hashes {genome_id: hash_str}.
        generator: Optional RNG.

    Returns:
        List of n selected genomes.
    """
    from tgraphx.generation.metrics import graph_wl_hash

    # Compute WL hashes
    hashes = []
    for genome in population:
        h = graph_wl_hash(genome.edge_index, genome.num_nodes, genome.node_types)
        hashes.append(h)

    from collections import Counter
    hash_counts = Counter(hashes)
    uniqueness = [1.0 / hash_counts[h] for h in hashes]

    # Normalize
    fit_t = torch.tensor(fitness_scores, dtype=torch.float)
    fit_min, fit_max = fit_t.min(), fit_t.max()
    if fit_max > fit_min:
        norm_fit = (fit_t - fit_min) / (fit_max - fit_min)
    else:
        norm_fit = torch.zeros_like(fit_t)

    uniq_t = torch.tensor(uniqueness, dtype=torch.float)
    uniq_min, uniq_max = uniq_t.min(), uniq_t.max()
    if uniq_max > uniq_min:
        norm_uniq = (uniq_t - uniq_min) / (uniq_max - uniq_min)
    else:
        norm_uniq = torch.zeros_like(uniq_t)

    combined = 0.5 * norm_fit + 0.5 * norm_uniq
    combined = combined + 1e-9  # Avoid zero probs

    indices = torch.multinomial(combined, n, replacement=True, generator=generator).tolist()
    return [population[i].clone() for i in indices]
