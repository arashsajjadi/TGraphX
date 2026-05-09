"""GraphSAINT sampler benchmark.

Usage:
    python benchmarks/sampling/benchmark_graphsaint.py --small --json
    python benchmarks/sampling/benchmark_graphsaint.py --sampler all \\
        --num-nodes 5000 --output /tmp/graphsaint_bench.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import tgraphx
from tgraphx import Graph
from tgraphx.graphsaint import (
    GraphSAINTNodeSampler,
    GraphSAINTEdgeSampler,
    GraphSAINTRandomWalkSampler,
    estimate_norm_coefficients,
)
from tgraphx.mining.reports import write_graphsaint_sampler_report


_SMALL_N = 200
_SMALL_E_FACTOR = 4
_DEFAULT_N = 2000
_DEFAULT_E_FACTOR = 6


def _make_graph(n: int, edge_factor: int, seed: int) -> Graph:
    torch.manual_seed(seed)
    src = torch.randint(n, (n * edge_factor,))
    dst = torch.randint(n, (n * edge_factor,))
    keep = src != dst
    src, dst = src[keep], dst[keep]
    ei = torch.unique(torch.stack([src, dst], dim=0), dim=1)
    x = torch.randn(n, 8)
    ew = torch.rand(ei.size(1))
    ef = torch.randn(ei.size(1), 4)
    return Graph(node_features=x, edge_index=ei, edge_weight=ew, edge_features=ef)


def _bench_sampler(
    name: str,
    sampler,
    num_samples: int,
    num_norm_samples: int,
) -> dict:
    node_counts, edge_counts, times = [], [], []
    for step in range(num_samples):
        t0 = time.perf_counter()
        sub = sampler.sample(step)
        dt = time.perf_counter() - t0
        node_counts.append(int(sub.num_nodes))
        edge_counts.append(int(sub.num_edges))
        times.append(dt * 1000)

    node_p, edge_p = estimate_norm_coefficients(sampler, num_samples=num_norm_samples)
    norm_finite = bool(torch.isfinite(node_p).all().item() and
                       torch.isfinite(edge_p).all().item())
    norm_positive = bool((node_p > 0).all().item() and (edge_p >= 0).all().item())

    return {
        "sampler": name,
        "num_samples": int(num_samples),
        "avg_sampled_nodes": round(sum(node_counts) / len(node_counts), 2),
        "avg_sampled_edges": round(sum(edge_counts) / len(edge_counts), 2),
        "avg_runtime_ms": round(sum(times) / len(times), 4),
        "min_nodes": min(node_counts),
        "max_nodes": max(node_counts),
        "normalization_finite": norm_finite,
        "normalization_positive": norm_positive,
        "node_norm_min": round(float(node_p.min().item()), 6),
        "node_norm_max": round(float(node_p.max().item()), 6),
        "node_norm_mean": round(float(node_p.mean().item()), 6),
        "edge_norm_min": round(float(edge_p.min().item()), 6) if edge_p.numel() else 0.0,
        "edge_norm_max": round(float(edge_p.max().item()), 6) if edge_p.numel() else 0.0,
        "edge_norm_mean": round(float(edge_p.mean().item()), 6) if edge_p.numel() else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphSAINT sampler benchmark")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-nodes", type=int, default=None)
    parser.add_argument("--sampler", default="all",
                        choices=["node", "edge", "random_walk", "all"])
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--node-budget", type=int, default=None)
    parser.add_argument("--walk-length", type=int, default=5)
    parser.add_argument("--num-roots", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    n = args.num_nodes or (_SMALL_N if args.small else _DEFAULT_N)
    num_samples = args.num_samples or (5 if args.small else 20)
    node_budget = args.node_budget or max(16, n // 4)
    num_norm_samples = 5 if args.small else 20

    g = _make_graph(n, _SMALL_E_FACTOR if args.small else _DEFAULT_E_FACTOR, args.seed)

    samplers_to_run = {
        "node": lambda: GraphSAINTNodeSampler(g, node_budget,
                                              num_steps=num_samples, seed=args.seed),
        "edge": lambda: GraphSAINTEdgeSampler(g, node_budget,
                                              num_steps=num_samples, seed=args.seed),
        "random_walk": lambda: GraphSAINTRandomWalkSampler(
            g, args.num_roots, args.walk_length,
            num_steps=num_samples, seed=args.seed,
        ),
    }
    if args.sampler != "all":
        samplers_to_run = {args.sampler: samplers_to_run[args.sampler]}

    results = []
    for name, factory in samplers_to_run.items():
        sampler = factory()
        res = _bench_sampler(name, sampler, num_samples, num_norm_samples)
        results.append(res)

    report = {
        "package_version": tgraphx.__version__,
        "seed": int(args.seed),
        "num_nodes": int(g.num_nodes),
        "num_edges": int(g.num_edges),
        "node_budget": int(node_budget),
        "sampler_results": results,
        "limitation_notes": [
            "Normalization estimates are Monte-Carlo approximations; "
            "increase --num-samples for production use.",
            "No real dataset used; this is a synthetic benchmark.",
        ],
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        write_graphsaint_sampler_report(args.output, report)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for r in results:
            print(f"[{r['sampler']:>12}] nodes={r['avg_sampled_nodes']:.0f} "
                  f"edges={r['avg_sampled_edges']:.0f} "
                  f"rt={r['avg_runtime_ms']:.2f}ms "
                  f"norm_finite={r['normalization_finite']}")


if __name__ == "__main__":
    main()
