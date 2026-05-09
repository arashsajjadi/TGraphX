"""Benchmark evolutionary graph optimization.

Usage:
    python benchmarks/evolution/benchmark_evolutionary_graph_optimization.py
    python benchmarks/evolution/benchmark_evolutionary_graph_optimization.py --small --json
"""
import argparse
import json
import time

import torch

from tgraphx.evolutionary.genome import GraphGenome
from tgraphx.evolutionary.fitness import connectivity_fitness, density_fitness
from tgraphx.evolutionary.algorithms import (
    GeneticAlgorithmConfig,
    GeneticAlgorithmOptimizer,
    SimulatedAnnealingOptimizer,
    EvolutionConfig,
)


def _random_genome(n=6, seed=0) -> GraphGenome:
    rng = torch.Generator()
    rng.manual_seed(seed)
    e = int(torch.randint(1, n, (1,), generator=rng).item())
    src = torch.randint(n, (e,), generator=rng).tolist()
    dst = [(s + 1) % n for s in src]
    ei = torch.tensor([src, dst], dtype=torch.long)
    return GraphGenome(edge_index=ei, num_nodes=n, node_features=torch.randn(n, 4, generator=rng))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    pop_size = 5 if args.small else 15
    n_gen = 5 if args.small else 20
    n = 5

    results = []

    # GA
    t0 = time.perf_counter()
    config = GeneticAlgorithmConfig(
        population_size=pop_size,
        n_generations=n_gen,
        seed=42,
        max_nodes=n + 2,
        max_edges=30,
    )
    pop = [_random_genome(n=n, seed=i) for i in range(pop_size)]
    result = GeneticAlgorithmOptimizer(config, connectivity_fitness).optimize(pop)
    ga_time = time.perf_counter() - t0
    gen_per_sec = n_gen / ga_time if ga_time > 0 else 0

    results.append({
        "algorithm": "GA",
        "n_generations": n_gen,
        "pop_size": pop_size,
        "best_fitness": result.best_fitness,
        "generations_per_sec": gen_per_sec,
        "time_s": ga_time,
    })

    if not args.json:
        print(f"GA: best_fitness={result.best_fitness:.4f}, "
              f"{gen_per_sec:.1f} gen/s, {ga_time:.3f}s")

    # SA
    t0 = time.perf_counter()
    evo_cfg = EvolutionConfig(seed=7, max_nodes=n + 2, max_edges=30, n_generations=n_gen * 5)
    sa_result = SimulatedAnnealingOptimizer(
        evo_cfg, lambda g: density_fitness(g, 0.5), T_init=2.0, cooling_rate=0.9
    ).optimize(_random_genome(n=n, seed=0))
    sa_time = time.perf_counter() - t0

    results.append({
        "algorithm": "SA",
        "n_steps": len(sa_result.fitness_history),
        "best_fitness": sa_result.best_fitness,
        "time_s": sa_time,
    })

    if not args.json:
        print(f"SA: best_fitness={sa_result.best_fitness:.4f}, {sa_time:.3f}s")

    if args.json:
        import sys, tgraphx
        device = "cuda" if torch.cuda.is_available() else "cpu"
        output = {
            "package_version": tgraphx.__version__,
            "benchmark": "evolutionary_optimization",
            "seed": 42,
            "device": device,
            "status": "ok",
            "limitations": "CPU-only small-scale; Experimental stability",
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "torch": torch.__version__,
            "results": results,
        }
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
