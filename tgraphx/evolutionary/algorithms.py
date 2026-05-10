"""Evolutionary optimization algorithms for graphs.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from .genome import GraphGenome
from .config import EvolutionConfig
from .operators import (
    mutate_add_node, mutate_remove_node, mutate_add_edge,
    mutate_remove_edge, mutate_rewire_edge, edge_set_crossover,
)
from .selection import tournament_selection, elitism_selection
from .multi_objective import ParetoFront, nsga2_select, non_dominated_sort, crowding_distance

__all__ = [
    "GeneticAlgorithmConfig",
    "EvolutionResult",
    "GeneticAlgorithmOptimizer",
    "SimulatedAnnealingOptimizer",
    "NSGAIIOptimizer",
    "HillClimbingOptimizer",
    "RandomSearchOptimizer",
]


@dataclass
class GeneticAlgorithmConfig:
    """Configuration for the genetic algorithm."""

    population_size: int = 30
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    n_generations: int = 50
    elitism_k: int = 2
    tournament_k: int = 3
    seed: Optional[int] = None
    max_nodes: int = 20
    max_edges: int = 100


@dataclass
class EvolutionResult:
    """Result of an evolutionary optimization run.

    Attributes:
        best_genome: Best genome found.
        best_fitness: Best fitness value (scalar).
        fitness_history: Best fitness value per generation ``[float, ...]``.
        diversity_history: WL uniqueness fraction per generation.
        pareto_front: Pareto front (multi-objective only; ``None`` otherwise).
        config: Configuration used.
        extra: Additional metadata.

    Derived properties:
        history: Per-generation list of dicts::

            [{"generation": 0, "best_fitness": ..., "diversity": ...,
              "pareto_front_size": ..., "population_size": ...}, ...]

    Methods:
        summary(): Return a human-readable string.
        to_dict(): Return a JSON-serialisable dict.
    """

    best_genome: Optional[GraphGenome]
    best_fitness: float
    fitness_history: List[float] = field(default_factory=list)
    diversity_history: List[float] = field(default_factory=list)
    pareto_front: Optional[ParetoFront] = None
    config: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Per-generation evolution history.

        Returns a list of dicts, one entry per completed generation::

            [
              {"generation": 0, "best_fitness": 0.82, "diversity": 0.9,
               "pareto_front_size": None, "population_size": None},
              ...
            ]

        Fields present in every entry:
            generation (int): 0-based generation index.
            best_fitness (float or None): Best fitness seen at that generation.
            diversity (float or None): Population diversity (WL uniqueness).
            pareto_front_size (int or None): Pareto-front size for
                multi-objective runs; ``None`` for single-objective.
            population_size (int or None): Population size if stored in
                ``extra["population_sizes"]``; else ``None``.
        """
        n = max(len(self.fitness_history), len(self.diversity_history))
        # Pareto-front sizes per generation if available.
        pf_sizes: List[Optional[int]] = self.extra.get("pareto_front_sizes", [])
        pop_sizes: List[Optional[int]] = self.extra.get("population_sizes", [])
        result = []
        for i in range(n):
            entry: Dict[str, Any] = {
                "generation": i,
                "best_fitness": self.fitness_history[i] if i < len(self.fitness_history) else None,
                "diversity": self.diversity_history[i] if i < len(self.diversity_history) else None,
                "pareto_front_size": pf_sizes[i] if i < len(pf_sizes) else None,
                "population_size": pop_sizes[i] if i < len(pop_sizes) else None,
            }
            result.append(entry)
        return result

    def summary(self) -> str:
        """Return a human-readable summary string (also prints it)."""
        lines = [
            "=" * 50,
            "TGraphX Evolutionary Optimization Result",
            "=" * 50,
            f"Best fitness:  {self.best_fitness:.6f}",
            f"Generations:   {len(self.fitness_history)}",
        ]
        if self.pareto_front:
            lines.append(f"Pareto front:  {len(self.pareto_front)} solutions")
        if self.fitness_history:
            lines.append(f"Initial best:  {self.fitness_history[0]:.6f}")
            lines.append(f"Final best:    {self.fitness_history[-1]:.6f}")
        text = "\n".join(lines)
        print(text)
        return text

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict (excludes non-serialisable genome)."""
        return {
            "best_fitness": float(self.best_fitness),
            "n_generations": len(self.fitness_history),
            "fitness_history": [float(f) for f in self.fitness_history],
            "diversity_history": [float(d) for d in self.diversity_history],
            "pareto_front_size": len(self.pareto_front) if self.pareto_front else None,
            "extra": {
                k: v for k, v in self.extra.items()
                if isinstance(v, (str, int, float, bool, list, dict, type(None)))
            },
        }


def _compute_diversity(population: List[GraphGenome]) -> float:
    """Compute population diversity as WL uniqueness fraction."""
    from tgraphx.generation.metrics import graph_wl_hash
    if not population:
        return 0.0
    hashes = [graph_wl_hash(g.edge_index, g.num_nodes, g.node_types) for g in population]
    return len(set(hashes)) / len(hashes)


def _mutate_random(
    genome: GraphGenome,
    mutation_rate: float,
    generator: Optional[torch.Generator],
    max_nodes: int,
    max_edges: int,
) -> GraphGenome:
    """Apply a random mutation to a genome."""
    ops = []
    if genome.num_nodes < max_nodes:
        ops.append("add_node")
    if genome.num_nodes > 1:
        ops.append("remove_node")
        if genome.num_edges < max_edges:
            ops.append("add_edge")
        if genome.num_edges > 0:
            ops.append("remove_edge")
            ops.append("rewire_edge")

    if not ops:
        return genome.clone()

    if torch.rand(1, generator=generator).item() > mutation_rate:
        return genome.clone()

    op_idx = int(torch.randint(len(ops), (1,), generator=generator).item())
    op = ops[op_idx]

    try:
        if op == "add_node":
            return mutate_add_node(genome, generator=generator)
        elif op == "remove_node":
            return mutate_remove_node(genome, generator=generator)
        elif op == "add_edge":
            return mutate_add_edge(genome, generator=generator)
        elif op == "remove_edge":
            return mutate_remove_edge(genome, generator=generator)
        elif op == "rewire_edge":
            return mutate_rewire_edge(genome, generator=generator)
    except ValueError:
        pass
    return genome.clone()


class GeneticAlgorithmOptimizer:
    """Genetic algorithm for graph optimization.

    Args:
        config: GeneticAlgorithmConfig or EvolutionConfig.
        fitness_fn: Function mapping GraphGenome -> float (higher = better).
    """

    def __init__(
        self,
        config: GeneticAlgorithmConfig,
        fitness_fn: Callable[[GraphGenome], float],
    ) -> None:
        self.config = config
        self.fitness_fn = fitness_fn
        self._generator: Optional[torch.Generator] = None
        if config.seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(config.seed)

        self._population: List[GraphGenome] = []
        self._fitness: List[float] = []
        self._generation: int = 0
        self._result = EvolutionResult(best_genome=None, best_fitness=float("-inf"), config=config)

    def optimize(
        self,
        initial_population: List[GraphGenome],
        progress_callback: Optional[Callable] = None,
    ) -> EvolutionResult:
        """Run the GA for n_generations.

        Args:
            initial_population: Starting population.
            progress_callback: Optional callable(gen, best_fitness, diversity).

        Returns:
            EvolutionResult.
        """
        self._population = [g.clone() for g in initial_population]
        self._fitness = [self.fitness_fn(g) for g in self._population]
        self._generation = 0

        for gen in range(self.config.n_generations):
            self.step()
            if progress_callback:
                div = _compute_diversity(self._population)
                progress_callback(gen, self._result.best_fitness, div)

        return self._result

    def step(self) -> None:
        """Run one generation of GA."""
        cfg = self.config
        gen = self._generator

        # Elites
        elites = elitism_selection(self._population, self._fitness, cfg.elitism_k)

        # Selection
        parents = tournament_selection(
            self._population, self._fitness, k=cfg.tournament_k, generator=gen
        )

        # Crossover + mutation
        offspring: List[GraphGenome] = list(elites)
        i = 0
        while len(offspring) < cfg.population_size:
            pa = parents[i % len(parents)]
            pb = parents[(i + 1) % len(parents)]

            if torch.rand(1, generator=gen).item() < cfg.crossover_rate:
                try:
                    child_a, child_b = edge_set_crossover(pa, pb, generator=gen)
                except Exception:
                    child_a, child_b = pa.clone(), pb.clone()
            else:
                child_a, child_b = pa.clone(), pb.clone()

            child_a = _mutate_random(child_a, cfg.mutation_rate, gen, cfg.max_nodes, cfg.max_edges)
            child_b = _mutate_random(child_b, cfg.mutation_rate, gen, cfg.max_nodes, cfg.max_edges)

            offspring.append(child_a)
            if len(offspring) < cfg.population_size:
                offspring.append(child_b)
            i += 1

        self._population = offspring[:cfg.population_size]
        self._fitness = [self.fitness_fn(g) for g in self._population]

        best_idx = max(range(len(self._fitness)), key=lambda i: self._fitness[i])
        best_fit = self._fitness[best_idx]

        if best_fit > self._result.best_fitness:
            self._result.best_fitness = best_fit
            self._result.best_genome = self._population[best_idx].clone()

        self._result.fitness_history.append(best_fit)
        self._result.diversity_history.append(_compute_diversity(self._population))
        self._generation += 1


class SimulatedAnnealingOptimizer:
    """Simulated Annealing for graph optimization.

    Uses Metropolis criterion:
        Accept if f(new) > f(old)
        Accept with probability exp((f(new) - f(old)) / T) otherwise

    Args:
        config: EvolutionConfig (uses T_init, T_min, cooling_rate, seed).
        fitness_fn: Function mapping GraphGenome -> float.
        T_init: Initial temperature (overrides config.T_init if provided).
        T_min: Minimum temperature.
        cooling_rate: Multiplicative cooling rate.
    """

    def __init__(
        self,
        config: EvolutionConfig,
        fitness_fn: Callable[[GraphGenome], float],
        T_init: Optional[float] = None,
        T_min: Optional[float] = None,
        cooling_rate: Optional[float] = None,
    ) -> None:
        self.config = config
        self.fitness_fn = fitness_fn
        self.T = T_init if T_init is not None else config.T_init
        self.T_min = T_min if T_min is not None else config.T_min
        self.cooling_rate = cooling_rate if cooling_rate is not None else config.cooling_rate
        self._generator: Optional[torch.Generator] = None
        if config.seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(config.seed)

    def optimize(
        self,
        initial_genome: GraphGenome,
        progress_callback: Optional[Callable] = None,
    ) -> EvolutionResult:
        """Run SA from initial_genome.

        Args:
            initial_genome: Starting genome.
            progress_callback: Optional callable(step, T, current_fitness).

        Returns:
            EvolutionResult.
        """
        current = initial_genome.clone()
        current_fitness = self.fitness_fn(current)
        best = current.clone()
        best_fitness = current_fitness

        result = EvolutionResult(
            best_genome=best,
            best_fitness=best_fitness,
            config=self.config,
        )

        T = self.T
        step = 0

        while T > self.T_min:
            # Random mutation
            candidate = _mutate_random(
                current, mutation_rate=1.0,  # always mutate in SA
                generator=self._generator,
                max_nodes=self.config.max_nodes,
                max_edges=self.config.max_edges,
            )
            candidate_fitness = self.fitness_fn(candidate)
            delta = candidate_fitness - current_fitness

            if delta > 0:
                accept = True
            else:
                prob = math.exp(delta / max(T, 1e-10))
                accept = torch.rand(1, generator=self._generator).item() < prob

            if accept:
                current = candidate
                current_fitness = candidate_fitness

            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best = current.clone()
                result.best_genome = best
                result.best_fitness = best_fitness

            result.fitness_history.append(current_fitness)

            T *= self.cooling_rate
            step += 1

            if progress_callback:
                progress_callback(step, T, current_fitness)

        return result


class NSGAIIOptimizer:
    """NSGA-II multi-objective optimizer.

    Args:
        config: EvolutionConfig.
        fitness_fn_list: List of fitness functions (one per objective).
    """

    def __init__(
        self,
        config: EvolutionConfig,
        fitness_fn_list: Union[Callable[[GraphGenome], float], List[Callable[[GraphGenome], float]]],
    ) -> None:
        self.config = config
        if callable(fitness_fn_list) and not isinstance(fitness_fn_list, list):
            import inspect
            try:
                sig = inspect.signature(fitness_fn_list)
                params = [
                    p for p in sig.parameters.values()
                    if p.default is inspect.Parameter.empty
                ]
                if len(params) > 1:
                    name = getattr(fitness_fn_list, "__name__", repr(fitness_fn_list))
                    raise TypeError(
                        f"NSGAIIOptimizer expects a sequence/list of objective functions "
                        f"(each taking a single GraphGenome argument), but received "
                        f"{name!r} which requires {len(params)} positional "
                        f"arguments {[p.name for p in params]}. "
                        f"For scalar composite fitness, use GeneticAlgorithmOptimizer or wrap "
                        f"components explicitly. "
                        f"For multi-objective NSGA-II use a list: "
                        f"NSGAIIOptimizer(config, [connectivity_fitness, sparsity_fitness])"
                    )
            except ValueError:
                pass
            self.fitness_fn_list = [fitness_fn_list]
        else:
            self.fitness_fn_list = fitness_fn_list
        self._generator: Optional[torch.Generator] = None
        if config.seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(config.seed)

    def optimize(
        self,
        initial_population: List[GraphGenome],
        progress_callback: Optional[Callable] = None,
    ) -> EvolutionResult:
        """Run NSGA-II.

        Args:
            initial_population: Starting population.
            progress_callback: Optional callable(gen, pareto_size).

        Returns:
            EvolutionResult with ParetoFront.
        """
        population = [g.clone() for g in initial_population]

        fitness_history: List[float] = []
        diversity_history: List[float] = []
        pareto_front_sizes: List[int] = []

        for gen in range(self.config.n_generations):
            # Generate offspring
            offspring = []
            gen_rng = self._generator
            n = len(population)
            for i in range(0, n, 2):
                pa = population[i % n]
                pb = population[(i + 1) % n]
                try:
                    ca, cb = edge_set_crossover(pa, pb, generator=gen_rng)
                except Exception:
                    ca, cb = pa.clone(), pb.clone()
                ca = _mutate_random(ca, self.config.mutation_rate, gen_rng,
                                    self.config.max_nodes, self.config.max_edges)
                cb = _mutate_random(cb, self.config.mutation_rate, gen_rng,
                                    self.config.max_nodes, self.config.max_edges)
                offspring.extend([ca, cb])

            population = nsga2_select(
                population, offspring, self.fitness_fn_list,
                n_select=self.config.population_size,
            )

            # Track per-generation history.
            gen_fitness = [fn(g) for fn in self.fitness_fn_list for g in population]
            best_f1 = max(self.fitness_fn_list[0](g) for g in population)
            fitness_history.append(best_f1)
            diversity_history.append(_compute_diversity(population))
            pareto_front_sizes.append(len(population))

            if progress_callback:
                progress_callback(gen, len(population))

        # Build final Pareto front
        fitness_vectors = [
            [fn(g) for fn in self.fitness_fn_list]
            for g in population
        ]
        fronts = non_dominated_sort(population, fitness_vectors)
        front0 = fronts[0] if fronts else []
        distances = crowding_distance(fitness_vectors, front0)

        pareto_front = ParetoFront(
            genomes=[population[i].clone() for i in front0],
            fitness_vectors=[fitness_vectors[i] for i in front0],
            crowding_distances=distances,
        )

        # Best by first objective
        if front0:
            best_idx = max(front0, key=lambda i: fitness_vectors[i][0])
            best_genome = population[best_idx].clone()
            best_fitness = fitness_vectors[best_idx][0]
        else:
            best_genome = population[0].clone() if population else None
            best_fitness = float("-inf")

        return EvolutionResult(
            best_genome=best_genome,
            best_fitness=best_fitness,
            fitness_history=fitness_history,
            diversity_history=diversity_history,
            pareto_front=pareto_front,
            config=self.config,
            extra={"pareto_front_sizes": pareto_front_sizes},
        )


class HillClimbingOptimizer:
    """Random restart hill climbing.

    Args:
        config: EvolutionConfig (uses n_generations, num_restarts, seed).
        fitness_fn: Fitness function.
    """

    def __init__(
        self,
        config: EvolutionConfig,
        fitness_fn: Callable[[GraphGenome], float],
    ) -> None:
        self.config = config
        self.fitness_fn = fitness_fn
        self._generator: Optional[torch.Generator] = None
        if config.seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(config.seed)

    def optimize(
        self,
        initial_genome: GraphGenome,
        progress_callback: Optional[Callable] = None,
    ) -> EvolutionResult:
        """Run hill climbing with random restarts.

        Args:
            initial_genome: Starting genome.
            progress_callback: Optional callback.

        Returns:
            EvolutionResult.
        """
        best_genome = initial_genome.clone()
        best_fitness = self.fitness_fn(best_genome)
        fitness_history = [best_fitness]

        for restart in range(self.config.num_restarts):
            current = initial_genome.clone() if restart == 0 else _mutate_random(
                initial_genome, 1.0, self._generator,
                self.config.max_nodes, self.config.max_edges
            )
            current_fitness = self.fitness_fn(current)

            for _ in range(self.config.n_generations // max(1, self.config.num_restarts)):
                candidate = _mutate_random(current, 1.0, self._generator,
                                           self.config.max_nodes, self.config.max_edges)
                candidate_fitness = self.fitness_fn(candidate)
                if candidate_fitness >= current_fitness:
                    current = candidate
                    current_fitness = candidate_fitness

            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_genome = current.clone()

            fitness_history.append(best_fitness)

        return EvolutionResult(
            best_genome=best_genome,
            best_fitness=best_fitness,
            fitness_history=fitness_history,
            config=self.config,
        )


class RandomSearchOptimizer:
    """Baseline random search.

    Samples random mutations from the initial genome.

    Args:
        config: EvolutionConfig (uses num_samples, seed).
        fitness_fn: Fitness function.
        num_graphs: Number of random graphs to try.
    """

    def __init__(
        self,
        config: EvolutionConfig,
        fitness_fn: Callable[[GraphGenome], float],
        num_graphs: int = 100,
    ) -> None:
        self.config = config
        self.fitness_fn = fitness_fn
        self.num_graphs = num_graphs
        self._generator: Optional[torch.Generator] = None
        if config.seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(config.seed)

    def optimize(
        self,
        initial_genome: GraphGenome,
        progress_callback: Optional[Callable] = None,
    ) -> EvolutionResult:
        """Random search.

        Args:
            initial_genome: Starting genome.
            progress_callback: Optional callback.

        Returns:
            EvolutionResult.
        """
        best_genome = initial_genome.clone()
        best_fitness = self.fitness_fn(best_genome)
        fitness_history = [best_fitness]

        for i in range(self.num_graphs):
            n_mutations = int(torch.randint(1, 4, (1,), generator=self._generator).item())
            current = initial_genome.clone()
            for _ in range(n_mutations):
                current = _mutate_random(
                    current, 1.0, self._generator,
                    self.config.max_nodes, self.config.max_edges
                )
            f = self.fitness_fn(current)
            if f > best_fitness:
                best_fitness = f
                best_genome = current.clone()
            fitness_history.append(best_fitness)

        return EvolutionResult(
            best_genome=best_genome,
            best_fitness=best_fitness,
            fitness_history=fitness_history,
            config=self.config,
        )
