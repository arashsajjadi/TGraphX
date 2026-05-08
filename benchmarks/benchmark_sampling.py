"""benchmarks/benchmark_sampling.py — sampling utility timing.

CPU-safe.  Reports time per call and approximate sampled-node/edge counts.
No claim of universal speedup; this is a smoke / regression utility.
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

from tgraphx import (
    Graph,
    induced_subgraph,
    k_hop_subgraph,
    neighbor_sample,
    NeighborSamplerLoader,
    sample_nodes,
)


def _time_ms(fn, iters=5):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000


def _make_graph(N, E, seed=0):
    g = torch.Generator(); g.manual_seed(seed)
    x = torch.randn(N, 8)
    src = torch.randint(0, N, (E,), generator=g)
    dst = torch.randint(0, N, (E,), generator=g)
    ei = torch.stack([src, dst], dim=0).long()
    return Graph(x, ei)


def run(small: bool = False) -> List[Dict[str, Any]]:
    if small:
        N, E, iters = 200, 800, 3
    else:
        N, E, iters = 5000, 25000, 5
    graph = _make_graph(N, E)
    results: List[Dict[str, Any]] = []

    # induced
    keep = torch.arange(0, N, 4)  # ~25% of nodes
    sub = induced_subgraph(graph, keep)
    ms = _time_ms(lambda: induced_subgraph(graph, keep), iters=iters)
    results.append({
        "op": "induced_subgraph",
        "input_nodes": int(keep.numel()),
        "result_nodes": sub.num_nodes,
        "result_edges": sub.num_edges,
        "ms": round(ms, 4),
    })

    # k-hop
    seeds = torch.tensor([0, 1, 2])
    sub = k_hop_subgraph(graph, seeds, num_hops=2)
    ms = _time_ms(lambda: k_hop_subgraph(graph, seeds, num_hops=2), iters=iters)
    results.append({
        "op": "k_hop_subgraph(2)",
        "seeds": seeds.numel(),
        "result_nodes": sub.num_nodes,
        "result_edges": sub.num_edges,
        "ms": round(ms, 4),
    })

    # uniform sample_nodes
    ms = _time_ms(lambda: sample_nodes(graph, num_nodes=64, seed=0), iters=iters)
    results.append({
        "op": "sample_nodes(64)",
        "ms": round(ms, 4),
    })

    # neighbor_sample
    seeds = torch.arange(0, min(16, N))
    sub = neighbor_sample(graph, seeds, fanouts=[10, 5], seed=0)
    ms = _time_ms(
        lambda: neighbor_sample(graph, seeds, fanouts=[10, 5], seed=0),
        iters=iters,
    )
    results.append({
        "op": "neighbor_sample([10,5])",
        "seeds": seeds.numel(),
        "result_nodes": sub.num_nodes,
        "result_edges": sub.num_edges,
        "ms": round(ms, 4),
    })

    # NeighborSamplerLoader full sweep
    loader = NeighborSamplerLoader(
        graph, batch_size=32, fanouts=[10, 5], shuffle=False, seed=0,
    )
    t0 = time.perf_counter()
    n_batches = 0
    for _ in loader:
        n_batches += 1
    elapsed_ms = (time.perf_counter() - t0) * 1000
    results.append({
        "op": f"NeighborSamplerLoader(bs=32, fanouts=[10,5])",
        "batches": n_batches,
        "total_ms": round(elapsed_ms, 4),
        "ms_per_batch": round(elapsed_ms / max(n_batches, 1), 4),
    })

    return results


def _print(results):
    sep = "-" * 72
    print(sep)
    print(f"  TGraphX sampling benchmark (N=5000, E=25000 unless --small)")
    print(sep)
    for r in results:
        print(f"  {r['op']:<36}  " + "  ".join(
            f"{k}={v}" for k, v in r.items() if k != "op"
        ))
    print(sep)


def main(argv=None):
    p = argparse.ArgumentParser(description="TGraphX sampling benchmark")
    p.add_argument("--small", action="store_true", help="CI-safe small mode")
    p.add_argument("--output", default=None, help="optional JSON output path")
    args = p.parse_args(argv)

    res = run(small=args.small)
    _print(res)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n  Results written to: {args.output}")


if __name__ == "__main__":
    main()
