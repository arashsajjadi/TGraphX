"""Configuration dataclass for evolutionary graph optimization.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

__all__ = ["EvolutionConfig"]


@dataclass
class EvolutionConfig:
    """All hyperparameters for evolutionary graph optimization.

    Args:
        population_size: Number of individuals per generation.
        mutation_rate: Probability of applying each mutation operator.
        crossover_rate: Probability of applying crossover.
        n_generations: Number of generations.
        elitism_k: Number of elite individuals preserved each generation.
        tournament_k: Tournament size for tournament selection.
        seed: Random seed.
        max_nodes: Maximum number of nodes per genome.
        max_edges: Maximum number of edges per genome.
        T_init: Initial temperature (for SA).
        T_min: Minimum temperature (for SA).
        cooling_rate: Cooling rate alpha in T <- T * alpha (for SA).
        num_restarts: Number of random restarts (for hill climbing).
        num_samples: Number of random samples (for random search).
    """

    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    n_generations: int = 100
    elitism_k: int = 2
    tournament_k: int = 3
    seed: Optional[int] = None
    max_nodes: int = 50
    max_edges: int = 500
    T_init: float = 1.0
    T_min: float = 0.01
    cooling_rate: float = 0.95
    num_restarts: int = 10
    num_samples: int = 100
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvolutionConfig":
        text = json.dumps(d)
        parsed = json.loads(text)
        known = {
            "population_size", "mutation_rate", "crossover_rate", "n_generations",
            "elitism_k", "tournament_k", "seed", "max_nodes", "max_edges",
            "T_init", "T_min", "cooling_rate", "num_restarts", "num_samples",
        }
        kwargs = {k: v for k, v in parsed.items() if k in known}
        extra = {k: v for k, v in parsed.items() if k not in known}
        return cls(**kwargs, extra=extra)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return json.loads(json.dumps(d, default=str))
