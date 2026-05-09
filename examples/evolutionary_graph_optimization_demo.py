"""Demonstrate evolutionary graph optimization.

Shows: GA optimizing connectivity, SA optimizing density, NSGA-II on two objectives.
"""
import tempfile
import os
import torch

from tgraphx.evolutionary.genome import GraphGenome
from tgraphx.evolutionary.fitness import connectivity_fitness, density_fitness
from tgraphx.evolutionary.algorithms import (
    GeneticAlgorithmConfig,
    GeneticAlgorithmOptimizer,
    SimulatedAnnealingOptimizer,
    NSGAIIOptimizer,
    EvolutionConfig,
)
from tgraphx.evolutionary.reports import write_evolution_report


def _make_random_genome(n=8, seed=None) -> GraphGenome:
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    num_edges = int(torch.randint(1, n, (1,), generator=rng).item())
    src = torch.randint(n, (num_edges,), generator=rng).tolist()
    dst = [s + 1 if s < n - 1 else 0 for s in src]
    ei = torch.tensor([src, dst], dtype=torch.long)
    nf = torch.randn(n, 4, generator=rng)
    return GraphGenome(edge_index=ei, num_nodes=n, node_features=nf)


def main():
    print("=== Evolutionary Graph Optimization Demo ===\n")

    # Genetic Algorithm — maximize connectivity
    print("--- Genetic Algorithm (connectivity objective) ---")
    config = GeneticAlgorithmConfig(
        population_size=10,
        n_generations=20,
        seed=42,
        max_nodes=10,
        max_edges=50,
        mutation_rate=0.3,
    )
    pop = [_make_random_genome(n=6, seed=i) for i in range(10)]
    ga = GeneticAlgorithmOptimizer(config, connectivity_fitness)
    result = ga.optimize(pop)
    print(f"  Initial best fitness: {result.fitness_history[0]:.4f}")
    print(f"  Final best fitness:   {result.best_fitness:.4f}")
    print(f"  Best genome: {result.best_genome.num_nodes} nodes, {result.best_genome.num_edges} edges")
    print(f"  Generations: {len(result.fitness_history)}")

    # Simulated Annealing — density objective
    print("\n--- Simulated Annealing (density objective, target=0.5) ---")
    evo_cfg = EvolutionConfig(seed=7, max_nodes=8, max_edges=40)
    initial = _make_random_genome(n=6, seed=42)
    fn = lambda g: density_fitness(g, target_density=0.5)
    sa = SimulatedAnnealingOptimizer(evo_cfg, fn, T_init=2.0, T_min=0.01, cooling_rate=0.9)
    sa_result = sa.optimize(initial)
    print(f"  Best fitness (density closeness): {sa_result.best_fitness:.4f}")
    print(f"  Best genome: {sa_result.best_genome.num_nodes} nodes, {sa_result.best_genome.num_edges} edges")

    # NSGA-II — two objectives
    print("\n--- NSGA-II (connectivity + density objectives) ---")
    evo_cfg2 = EvolutionConfig(
        seed=99, max_nodes=8, max_edges=50, population_size=8, n_generations=15
    )
    pop2 = [_make_random_genome(n=6, seed=i + 100) for i in range(8)]
    fn_list = [
        connectivity_fitness,
        lambda g: density_fitness(g, target_density=0.3),
    ]
    nsga = NSGAIIOptimizer(evo_cfg2, fn_list)
    nsga_result = nsga.optimize(pop2)

    pf = nsga_result.pareto_front
    print(f"  Pareto front size: {len(pf)}")
    if len(pf) > 0:
        print("  Pareto front fitness vectors (first 5):")
        for v in pf.fitness_vectors[:5]:
            print(f"    connectivity={v[0]:.3f}, density_fitness={v[1]:.3f}")

    # Write report
    out = os.path.join(tempfile.gettempdir(), "evolution_report.json")
    write_evolution_report(out, result, EvolutionConfig.from_dict(config.__dict__))
    print(f"\nEvolution report written to: {out}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
