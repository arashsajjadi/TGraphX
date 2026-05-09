"""Benchmark classical graph generators.

Usage:
    python benchmarks/generation/benchmark_classical_generators.py
    python benchmarks/generation/benchmark_classical_generators.py --small --json
"""
import argparse
import json
import time

import torch

from tgraphx.generation.classical import FeatureAwareERGraph, FeatureAwareBAGraph


def main():
    parser = argparse.ArgumentParser(description="Benchmark classical graph generators")
    parser.add_argument("--small", action="store_true", help="Use small graphs for quick run")
    parser.add_argument("--json", action="store_true", help="Output JSON results")
    args = parser.parse_args()

    n = 50 if args.small else 200
    results = []

    for gen_name, fn, kwargs in [
        ("FeatureAwareERGraph", FeatureAwareERGraph, {"n": n, "p": 0.2, "node_feature_dim": 16, "edge_feature_dim": 8}),
        ("FeatureAwareBAGraph", FeatureAwareBAGraph, {"n": n, "m": 3, "node_feature_dim": 16}),
    ]:
        times = []
        for trial in range(3):
            t0 = time.perf_counter()
            g = fn(seed=trial, **kwargs)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_time = sum(times) / len(times)
        record = {
            "generator": gen_name,
            "n": n,
            "num_edges": g.num_edges,
            "avg_time_s": avg_time,
            "node_features_shape": list(g.node_features.shape) if g.node_features is not None else None,
        }
        results.append(record)

        if not args.json:
            print(f"{gen_name}: n={n}, e={g.num_edges}, time={avg_time*1000:.1f}ms")

    if args.json:
        import sys, tgraphx
        device = "cuda" if torch.cuda.is_available() else "cpu"
        output = {
            "package_version": tgraphx.__version__,
            "benchmark": "classical_generators",
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
