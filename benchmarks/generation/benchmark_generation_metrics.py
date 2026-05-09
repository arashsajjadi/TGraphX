"""Benchmark graph generation metrics (validity, uniqueness, MMD, etc.).

Flags: --small --json --seed --device
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark graph generation metrics")
    p.add_argument("--small", action="store_true", help="Use small configuration")
    p.add_argument("--json", action="store_true", help="Output JSON")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def run_benchmark(small: bool, seed: int, device: str) -> Dict[str, Any]:
    import torch
    from tgraphx.generation.classical import FeatureAwareERGraph
    from tgraphx.generation.metrics import (
        uniqueness_score, diversity_score,
        degree_distribution_distance, mmd_degree,
    )

    torch.manual_seed(seed)
    num_graphs = 8 if small else 32
    n = 10 if small else 30

    # Generate graphs
    t0 = time.perf_counter()
    graphs = [FeatureAwareERGraph(n=n, p=0.3, node_feature_dim=4, seed=seed + i) for i in range(num_graphs)]
    gen_time = time.perf_counter() - t0

    # Validity (fraction with at least 1 edge)
    t1 = time.perf_counter()
    val = sum(1 for g in graphs if int(g.edge_index.shape[1]) > 0) / max(len(graphs), 1)
    val_time = time.perf_counter() - t1

    # Uniqueness
    t2 = time.perf_counter()
    uniq = uniqueness_score(graphs)
    uniq_time = time.perf_counter() - t2

    # Diversity
    t3 = time.perf_counter()
    div = diversity_score(graphs)
    div_time = time.perf_counter() - t3

    # MMD (degree)
    ref = graphs[:num_graphs // 2]
    gen = graphs[num_graphs // 2:]
    t4 = time.perf_counter()
    mmd_val = mmd_degree(ref, gen)
    mmd_time = time.perf_counter() - t4

    return {
        "seed": seed,
        "device": device,
        "num_graphs": num_graphs,
        "num_nodes": n,
        "metrics": {
            "validity": float(val),
            "uniqueness": float(uniq),
            "diversity": float(div),
            "mmd_degree": float(mmd_val),
        },
        "timing_seconds": {
            "generation": gen_time,
            "validity": val_time,
            "uniqueness": uniq_time,
            "diversity": div_time,
            "mmd_degree": mmd_time,
        },
    }


def main():
    args = parse_args()
    result = run_benchmark(small=args.small, seed=args.seed, device=args.device)
    if args.json:
        import sys as _sys, tgraphx as _tgx
        import torch as _torch
        result.setdefault('package_version', _tgx.__version__)
        result.setdefault('status', 'ok')
        result.setdefault('limitations', 'CPU-only small-scale; Experimental stability')
        result.setdefault('device', 'cuda' if _torch.cuda.is_available() else 'cpu')
        print(json.dumps(result, indent=2))
    else:
        print(f"Generation metrics benchmark")
        print(f"  Graphs: {result['num_graphs']} x {result['num_nodes']} nodes")
        for k, v in result["metrics"].items():
            print(f"  {k}: {v:.4f}")
        for k, v in result["timing_seconds"].items():
            print(f"  {k} time: {v:.4f}s")


if __name__ == "__main__":
    main()
