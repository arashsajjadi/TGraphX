"""Benchmark mutation and crossover operator throughput.

Flags: --small --json --seed
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark mutation/crossover operators")
    p.add_argument("--small", action="store_true", help="Use small configuration")
    p.add_argument("--json", action="store_true", help="Output JSON")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_benchmark(small: bool, seed: int) -> Dict[str, Any]:
    import torch
    from tgraphx.evolutionary.genome import GraphGenome
    from tgraphx.evolutionary.operators import (
        mutate_add_edge, mutate_remove_edge, mutate_add_node,
        edge_set_crossover,
    )

    n_iters = 50 if small else 500
    n_nodes = 8 if small else 20
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    src = list(range(n_nodes - 1))
    dst = list(range(1, n_nodes))
    ei = torch.tensor([src + dst, dst + src], dtype=torch.long)
    genome = GraphGenome(edge_index=ei.clone(), num_nodes=n_nodes)

    # Benchmark add_edge
    t0 = time.perf_counter()
    for i in range(n_iters):
        try:
            mutate_add_edge(genome, generator=gen)
        except Exception:
            pass
    add_edge_time = (time.perf_counter() - t0) / n_iters

    # Benchmark remove_edge
    t1 = time.perf_counter()
    for i in range(n_iters):
        try:
            mutate_remove_edge(genome, generator=gen)
        except Exception:
            pass
    rm_edge_time = (time.perf_counter() - t1) / n_iters

    # Benchmark add_node
    t2 = time.perf_counter()
    for i in range(n_iters):
        try:
            mutate_add_node(genome, generator=gen)
        except Exception:
            pass
    add_node_time = (time.perf_counter() - t2) / n_iters

    # Benchmark crossover
    genome2 = GraphGenome(edge_index=ei.clone(), num_nodes=n_nodes)
    t3 = time.perf_counter()
    for i in range(n_iters):
        try:
            edge_set_crossover(genome, genome2, generator=gen)
        except Exception:
            pass
    cross_time = (time.perf_counter() - t3) / n_iters

    return {
        "seed": seed,
        "n_iters": n_iters,
        "n_nodes": n_nodes,
        "metrics": {
            "add_edge_us": add_edge_time * 1e6,
            "remove_edge_us": rm_edge_time * 1e6,
            "add_node_us": add_node_time * 1e6,
            "crossover_us": cross_time * 1e6,
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
        print(f"Mutation/crossover benchmark ({result['n_iters']} iters)")
        for k, v in result["metrics"].items():
            print(f"  {k}: {v:.2f} us/op")


if __name__ == "__main__":
    main()
