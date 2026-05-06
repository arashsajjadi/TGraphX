"""benchmarks/benchmark_graph_builders.py — graph builder timing.

Usage
-----
python benchmarks/benchmark_graph_builders.py              # default config
python benchmarks/benchmark_graph_builders.py --small      # CI-safe small run
python benchmarks/benchmark_graph_builders.py --output results/builders.json

O(N²) warning is printed for builders that compute pairwise distances or
all-pair connections (kNN, radius, IoU, fully-connected).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))

import torch

from tgraphx.graph_builders import (
    build_fully_connected_graph,
    build_grid_graph,
    build_grid_graph_3d,
    build_iou_graph,
    build_knn_graph,
    build_radius_graph,
    build_random_graph,
)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

_O2_WARNING = "O(N²)"

def _time_ms(fn, iters: int = 5) -> float:
    """Return mean wall-clock time in milliseconds."""
    # One warmup
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000


def _edge_bytes(ei: torch.Tensor) -> int:
    return ei.numel() * ei.element_size()


def _build_configs(small: bool) -> List[Dict[str, Any]]:
    if small:
        N_small, N_fc, iters = 16, 8, 3
    else:
        N_small, N_fc, iters = 64, 32, 5

    return [
        {
            "name": "build_grid_graph",
            "fn": lambda: build_grid_graph(8 if small else 16, 8 if small else 16,
                                           directed=False, self_loops=True),
            "nodes": (8 if small else 16) ** 2,
            "iters": iters,
            "o2": False,
        },
        {
            "name": "build_grid_graph_3d",
            "fn": lambda: build_grid_graph_3d(4 if small else 8, 4 if small else 8,
                                              4 if small else 8,
                                              directed=False, self_loops=True),
            "nodes": (4 if small else 8) ** 3,
            "iters": iters,
            "o2": False,
        },
        {
            "name": "build_fully_connected_graph",
            "fn": lambda: build_fully_connected_graph(N_fc, self_loops=False),
            "nodes": N_fc,
            "iters": iters,
            "o2": True,
        },
        {
            "name": "build_knn_graph",
            "fn": (lambda coords=torch.randn(N_small, 3):
                   build_knn_graph(coords, k=min(4, N_small - 1))),
            "nodes": N_small,
            "iters": iters,
            "o2": True,
        },
        {
            "name": "build_radius_graph",
            "fn": (lambda coords=torch.randn(N_small, 2):
                   build_radius_graph(coords, radius=1.0)),
            "nodes": N_small,
            "iters": iters,
            "o2": True,
        },
        {
            "name": "build_iou_graph",
            "fn": (lambda boxes=torch.rand(N_small, 4).sort(dim=-1).values:
                   build_iou_graph(
                       boxes * torch.tensor([1.0, 1.0, 2.0, 2.0]),
                       threshold=0.3,
                   )),
            "nodes": N_small,
            "iters": iters,
            "o2": True,
        },
        {
            "name": "build_random_graph",
            "fn": lambda: build_random_graph(N_small,
                                             min(N_small * 3, N_small * (N_small - 1) - 1),
                                             directed=True, self_loops=False, seed=42),
            "nodes": N_small,
            "iters": iters,
            "o2": False,
        },
    ]


def run_builder_benchmark(small: bool = False) -> List[Dict[str, Any]]:
    configs = _build_configs(small)
    results = []
    for cfg in configs:
        ei = cfg["fn"]()
        ms = _time_ms(cfg["fn"], iters=cfg["iters"])
        results.append({
            "builder":  cfg["name"],
            "nodes":    cfg["nodes"],
            "edges":    int(ei.shape[1]),
            "time_ms":  round(ms, 4),
            "edge_kb":  round(_edge_bytes(ei) / 1024, 1),
            "o2_warn":  cfg["o2"],
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def _print_report(results: List[Dict]) -> None:
    sep = "-" * 70
    print(sep)
    print("  TGraphX Graph Builder Benchmark")
    print(sep)
    hdr = f"  {'Builder':<34} {'Nodes':>6} {'Edges':>8} {'Time ms':>9} {'KB':>6}"
    print(hdr)
    print(sep)
    for r in results:
        warn = "  [O(N²)]" if r["o2_warn"] else ""
        print(
            f"  {r['builder']:<34} {r['nodes']:>6} {r['edges']:>8} "
            f"{r['time_ms']:>9.4f} {r['edge_kb']:>6.1f}{warn}"
        )
    print(sep)
    print("  [O(N²)] builders use pairwise ops — scale with caution for large N.")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description="TGraphX graph builder benchmark")
    p.add_argument("--small",  action="store_true",
                   help="CI-safe small configuration")
    p.add_argument("--output", default=None,
                   help="Optional path to write JSON results")
    args = p.parse_args(argv)

    results = run_builder_benchmark(small=args.small)
    _print_report(results)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results written to: {args.output}")


if __name__ == "__main__":
    main()
