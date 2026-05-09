"""High-level API for evolutionary graph optimization.

Stability: Beta (v0.7.0+).

Usage:
    from tgraphx.evolutionary import run_evolutionary_optimization, list_evolutionary_optimizers
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import torch

__all__ = [
    "list_evolutionary_optimizers",
    "make_evolutionary_optimizer",
    "run_evolutionary_optimization",
    "OptimizationResult",
]


_EVOLUTIONARY_ALGORITHMS: Dict[str, Dict[str, str]] = {
    "ga": {
        "stability": "Experimental",
        "description": "Genetic Algorithm with crossover, mutation, elitism.",
        "multi_objective": "no",
    },
    "sa": {
        "stability": "Experimental",
        "description": "Simulated Annealing with Metropolis criterion.",
        "multi_objective": "no",
    },
    "nsga2": {
        "stability": "Experimental",
        "description": "NSGA-II multi-objective evolutionary algorithm.",
        "multi_objective": "yes",
    },
    "hill_climbing": {
        "stability": "Beta",
        "description": "Random restart hill climbing.",
        "multi_objective": "no",
    },
    "random_search": {
        "stability": "Beta",
        "description": "Baseline random search.",
        "multi_objective": "no",
    },
}

_OBJECTIVE_FNS: Dict[str, str] = {
    "connectivity": "connectivity_fitness",
    "density": "density_fitness",
    "clustering": "clustering_fitness",
    "motif_count": "motif_count_fitness",
}


def list_evolutionary_optimizers() -> Dict[str, Dict[str, str]]:
    """Return dict mapping algorithm name -> info dict.

    Returns:
        Dict with keys: algorithm name -> {stability, description, multi_objective}.
    """
    return dict(_EVOLUTIONARY_ALGORITHMS)


def _make_fitness_fn(objective: str) -> Callable:
    """Create a fitness function from objective name.

    Args:
        objective: Objective name from _OBJECTIVE_FNS.

    Returns:
        Callable(GraphGenome) -> float.

    Raises:
        ValueError: If objective not recognized.
    """
    from tgraphx.evolutionary.fitness import (
        connectivity_fitness, density_fitness,
        clustering_fitness, motif_count_fitness,
    )

    fns: Dict[str, Callable] = {
        "connectivity": connectivity_fitness,
        "density": density_fitness,
        "clustering": clustering_fitness,
        "motif_count": motif_count_fitness,
    }

    if objective not in fns:
        known = sorted(fns.keys())
        raise ValueError(
            f"Unknown objective '{objective}'. Choose from: {known}"
        )

    return fns[objective]


def make_evolutionary_optimizer(algorithm: str, fitness_fn: Any, **kwargs) -> Any:
    """Create an evolutionary optimizer by name.

    Args:
        algorithm: Algorithm name from list_evolutionary_optimizers().
        fitness_fn: Fitness function or list of fitness functions (for nsga2).
        **kwargs: Extra kwargs forwarded to optimizer config.

    Returns:
        Optimizer instance.

    Raises:
        ValueError: If algorithm not recognized.
    """
    if algorithm not in _EVOLUTIONARY_ALGORITHMS:
        known = sorted(_EVOLUTIONARY_ALGORITHMS.keys())
        raise ValueError(
            f"Unknown evolutionary algorithm '{algorithm}'. Choose from: {known}"
        )

    from tgraphx.evolutionary.algorithms import (
        GeneticAlgorithmConfig, GeneticAlgorithmOptimizer,
        SimulatedAnnealingOptimizer, NSGAIIOptimizer,
        HillClimbingOptimizer, RandomSearchOptimizer,
    )
    from tgraphx.evolutionary.config import EvolutionConfig

    seed = kwargs.pop("seed", None)
    population_size = kwargs.pop("population_size", 20)
    generations = kwargs.pop("generations", 10)
    max_nodes = kwargs.pop("max_nodes", 20)
    max_edges = kwargs.pop("max_edges", 100)

    if algorithm == "ga":
        cfg = GeneticAlgorithmConfig(
            population_size=population_size,
            n_generations=generations,
            max_nodes=max_nodes,
            max_edges=max_edges,
            seed=seed,
        )
        return GeneticAlgorithmOptimizer(cfg, fitness_fn)

    elif algorithm == "sa":
        cfg = EvolutionConfig(
            max_nodes=max_nodes, max_edges=max_edges,
            seed=seed, n_generations=generations,
        )
        return SimulatedAnnealingOptimizer(cfg, fitness_fn)

    elif algorithm == "nsga2":
        cfg = EvolutionConfig(
            population_size=population_size,
            n_generations=generations,
            max_nodes=max_nodes, max_edges=max_edges,
            seed=seed,
        )
        fn_list = fitness_fn if isinstance(fitness_fn, list) else [fitness_fn]
        return NSGAIIOptimizer(cfg, fn_list)

    elif algorithm == "hill_climbing":
        cfg = EvolutionConfig(
            max_nodes=max_nodes, max_edges=max_edges,
            seed=seed, n_generations=generations,
        )
        return HillClimbingOptimizer(cfg, fitness_fn)

    elif algorithm == "random_search":
        cfg = EvolutionConfig(
            max_nodes=max_nodes, max_edges=max_edges,
            seed=seed, n_generations=generations,
        )
        return RandomSearchOptimizer(cfg, fitness_fn, num_graphs=generations)

    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'")


@dataclass
class OptimizationResult:
    """Result of a run_evolutionary_optimization call.

    Attributes:
        best_genome: Best genome found.
        best_fitness: Best fitness value.
        fitness_history: Best fitness per generation.
        metrics: Dict summary of optimization results (best_fitness, n_generations, algorithm).
        pareto_front: Pareto front (multi-objective only).
        config: Serializable config dict.
        report_path: Path to JSON report if dashboard_dir set.
    """
    best_genome: Any  # GraphGenome
    best_fitness: float
    fitness_history: List[float]
    metrics: Dict[str, Any] = field(default_factory=dict)
    pareto_front: Optional[Any] = None  # ParetoFront
    config: Dict[str, Any] = field(default_factory=dict)
    report_path: Optional[str] = None


def _make_initial_genome(num_nodes: int, max_edges: int, seed: int) -> Any:
    """Create a simple initial GraphGenome."""
    from tgraphx.evolutionary.genome import GraphGenome

    rng = torch.Generator()
    rng.manual_seed(seed)

    # Path graph as starter
    if num_nodes >= 2:
        src = list(range(num_nodes - 1))
        dst = list(range(1, num_nodes))
        ei = torch.tensor([src + dst, dst + src], dtype=torch.long)
    else:
        ei = torch.zeros((2, 0), dtype=torch.long)

    return GraphGenome(edge_index=ei, num_nodes=num_nodes)


def run_evolutionary_optimization(
    algorithm: str = "ga",
    objective: Union[str, List[str]] = "connectivity",
    population_size: int = 20,
    generations: int = 10,
    n_generations: Optional[int] = None,
    num_nodes: int = 10,
    max_edges: int = 20,
    seed: int = 42,
    dashboard_dir: Optional[str] = None,
    **algo_kwargs,
) -> OptimizationResult:
    """Run evolutionary graph optimization.

    Args:
        algorithm: Algorithm name from list_evolutionary_optimizers().
        objective: Single objective name or list (for nsga2).
        population_size: Population size.
        generations: Number of generations (alias: n_generations).
        n_generations: Alias for generations (takes precedence if set).
        num_nodes: Number of nodes in graphs.
        max_edges: Maximum edges.
        seed: Random seed.
        dashboard_dir: If set, writes report JSON here.
        **algo_kwargs: Extra kwargs forwarded to the optimizer.

    Returns:
        OptimizationResult.

    Raises:
        ValueError: If algorithm or objective not recognized.
    """
    if algorithm not in _EVOLUTIONARY_ALGORITHMS:
        known = sorted(_EVOLUTIONARY_ALGORITHMS.keys())
        raise ValueError(
            f"Unknown evolutionary algorithm '{algorithm}'. Choose from: {known}"
        )

    torch.manual_seed(seed)

    # n_generations is an alias for generations
    if n_generations is not None:
        generations = n_generations

    # Build fitness function(s)
    if isinstance(objective, list):
        fitness_fn_list = [_make_fitness_fn(o) for o in objective]
        fitness_fn = fitness_fn_list
        obj_str = str(objective)
    else:
        fitness_fn = _make_fitness_fn(objective)
        fitness_fn_list = [fitness_fn]
        obj_str = objective

    # Build optimizer
    optimizer = make_evolutionary_optimizer(
        algorithm,
        fitness_fn,
        seed=seed,
        population_size=population_size,
        generations=generations,
        max_nodes=num_nodes,
        max_edges=max_edges,
        **algo_kwargs,
    )

    # Build initial genome / population
    initial_genome = _make_initial_genome(num_nodes, max_edges, seed)

    # Run optimization
    if algorithm in ("ga", "nsga2"):
        # Population-based
        from tgraphx.evolutionary.genome import GraphGenome
        rng = torch.Generator()
        rng.manual_seed(seed + 1)

        population = []
        for i in range(population_size):
            n_extra = int(torch.randint(0, max(2, max_edges // 4), (1,), generator=rng).item())
            g = initial_genome.clone()
            # Add random edges
            for _ in range(n_extra):
                if g.num_nodes >= 2 and int(g.edge_index.shape[1]) < max_edges:
                    u = int(torch.randint(g.num_nodes, (1,), generator=rng).item())
                    v = int(torch.randint(g.num_nodes, (1,), generator=rng).item())
                    if u != v:
                        new_ei = torch.cat([g.edge_index, torch.tensor([[u, v], [v, u]], dtype=torch.long)], dim=1)
                        g = GraphGenome(edge_index=new_ei, num_nodes=g.num_nodes)
            population.append(g)

        ev_result = optimizer.optimize(population)

    else:
        # Single-solution based (SA, hill_climbing, random_search)
        ev_result = optimizer.optimize(initial_genome)

    best_genome = ev_result.best_genome
    best_fitness = float(ev_result.best_fitness)
    fitness_history = [float(f) for f in ev_result.fitness_history]
    pareto_front = ev_result.pareto_front

    config: Dict[str, Any] = {
        "algorithm": algorithm,
        "objective": obj_str,
        "population_size": population_size,
        "generations": generations,
        "num_nodes": num_nodes,
        "max_edges": max_edges,
        "seed": seed,
        **{k: v for k, v in algo_kwargs.items() if isinstance(v, (int, float, str, bool))},
    }

    report_path = None
    if dashboard_dir:
        os.makedirs(dashboard_dir, exist_ok=True)
        report_path = os.path.join(dashboard_dir, f"evo_{algorithm}.json")

        report_data: Dict[str, Any] = {
            "config": config,
            "best_fitness": best_fitness,
            "fitness_history": fitness_history[:50],  # truncate for JSON
        }
        if pareto_front is not None:
            report_data["pareto_front_size"] = len(pareto_front.genomes) if pareto_front.genomes else 0

        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

    metrics_summary: Dict[str, Any] = {
        "best_fitness": best_fitness,
        "n_generations": len(fitness_history),
        "algorithm": algorithm,
        "objective": obj_str,
    }
    if pareto_front is not None:
        metrics_summary["pareto_front_size"] = (
            len(pareto_front.genomes) if pareto_front.genomes else 0
        )

    return OptimizationResult(
        best_genome=best_genome,
        best_fitness=best_fitness,
        fitness_history=fitness_history,
        metrics=metrics_summary,
        pareto_front=pareto_front,
        config=config,
        report_path=report_path,
    )
