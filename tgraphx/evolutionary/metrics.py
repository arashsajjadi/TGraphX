"""Metrics for evolutionary optimization results.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

from typing import Optional, Tuple

from .algorithms import EvolutionResult
from .multi_objective import ParetoFront, compute_hypervolume_2d

__all__ = [
    "best_fitness_curve",
    "mean_fitness_curve",
    "diversity_curve",
    "mutation_acceptance_rate",
    "constraint_violation_rate",
    "pareto_front_size",
    "hypervolume_2d",
]


def best_fitness_curve(result: EvolutionResult) -> list:
    """Return the best fitness per generation.

    Args:
        result: EvolutionResult.

    Returns:
        List of float.
    """
    return list(result.fitness_history)


def mean_fitness_curve(result: EvolutionResult) -> list:
    """Return smoothed mean fitness (running average).

    Args:
        result: EvolutionResult.

    Returns:
        List of float (same length as fitness_history).
    """
    if not result.fitness_history:
        return []
    means = []
    total = 0.0
    for i, f in enumerate(result.fitness_history):
        total += f
        means.append(total / (i + 1))
    return means


def diversity_curve(result: EvolutionResult) -> list:
    """Return the diversity (WL uniqueness fraction) per generation.

    Args:
        result: EvolutionResult.

    Returns:
        List of float.
    """
    return list(result.diversity_history)


def mutation_acceptance_rate(result: EvolutionResult) -> float:
    """Fraction of generations where fitness improved.

    Args:
        result: EvolutionResult.

    Returns:
        Float in [0, 1].
    """
    history = result.fitness_history
    if len(history) <= 1:
        return 0.0
    improvements = sum(
        1 for i in range(1, len(history)) if history[i] > history[i - 1]
    )
    return improvements / (len(history) - 1)


def constraint_violation_rate(result: EvolutionResult) -> float:
    """Fraction of states where fitness is negative (proxy for violations).

    Args:
        result: EvolutionResult.

    Returns:
        Float in [0, 1].
    """
    history = result.fitness_history
    if not history:
        return 0.0
    violations = sum(1 for f in history if f < 0)
    return violations / len(history)


def pareto_front_size(result: EvolutionResult) -> int:
    """Number of individuals on the Pareto front.

    Args:
        result: EvolutionResult.

    Returns:
        Int (0 if no Pareto front stored).
    """
    if result.pareto_front is None:
        return 0
    return len(result.pareto_front)


def hypervolume_2d(
    pareto_front: ParetoFront,
    reference_point: Tuple[float, float],
) -> float:
    """2D hypervolume indicator.

    Args:
        pareto_front: ParetoFront.
        reference_point: (ref_f1, ref_f2).

    Returns:
        Float hypervolume.
    """
    return compute_hypervolume_2d(pareto_front, reference_point)
