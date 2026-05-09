"""Benchmark NSGA-II on two-objective toy problem.

Flags: --small --json --seed
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark NSGA-II")
    p.add_argument("--small", action="store_true", help="Use small configuration")
    p.add_argument("--json", action="store_true", help="Output JSON")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_benchmark(small: bool, seed: int) -> Dict[str, Any]:
    import torch
    from tgraphx.evolutionary.algorithms import NSGAIIOptimizer
    from tgraphx.evolutionary.config import EvolutionConfig
    from tgraphx.evolutionary.fitness import connectivity_fitness, density_fitness
    from tgraphx.evolutionary.genome import GraphGenome

    pop_size = 10 if small else 30
    generations = 3 if small else 10
    n_nodes = 8 if small else 20

    cfg = EvolutionConfig(
        population_size=pop_size,
        n_generations=generations,
        max_nodes=n_nodes,
        max_edges=n_nodes * 4,
        seed=seed,
    )

    # Initial population: path graphs
    torch.manual_seed(seed)
    src = list(range(n_nodes - 1))
    dst = list(range(1, n_nodes))
    ei = torch.tensor([src + dst, dst + src], dtype=torch.long)
    initial = [GraphGenome(edge_index=ei.clone(), num_nodes=n_nodes) for _ in range(pop_size)]

    optimizer = NSGAIIOptimizer(cfg, [connectivity_fitness, density_fitness])

    t0 = time.perf_counter()
    result = optimizer.optimize(initial)
    elapsed = time.perf_counter() - t0

    pareto_size = 0
    if result.pareto_front is not None and result.pareto_front.genomes:
        pareto_size = len(result.pareto_front.genomes)

    return {
        "seed": seed,
        "population_size": pop_size,
        "generations": generations,
        "metrics": {
            "best_fitness": float(result.best_fitness),
            "pareto_front_size": pareto_size,
            "time_s": elapsed,
            "generations_per_sec": generations / max(elapsed, 1e-9),
        },
    }


def main():
    args = parse_args()
    result = run_benchmark(small=args.small, seed=args.seed)
    if args.json:
        import sys as _sys, tgraphx as _tgx
        import torch as _torch
        result.setdefault('package_version', _tgx.__version__)
        result.setdefault('status', 'ok')
        result.setdefault('limitations', 'CPU-only small-scale; Experimental stability')
        result.setdefault('device', 'cuda' if _torch.cuda.is_available() else 'cpu')
        print(json.dumps(result, indent=2))
    else:
        print(f"NSGA-II benchmark")
        print(f"  pop={result['population_size']}, gens={result['generations']}")
        for k, v in result["metrics"].items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
