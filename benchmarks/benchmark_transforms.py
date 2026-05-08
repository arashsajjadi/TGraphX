"""Benchmark transform throughput on synthetic graphs."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

import tgraphx
from tgraphx import Graph
from tgraphx.transforms import (
    AddDegreeEncoding,
    AddDegreeFeatures,
    AddSelfLoops,
    Compose,
    NormalizeFeatures,
    RandomNodeSplit,
    StandardizeFeatures,
    ToUndirected,
)


def _make_graph(N: int, D: int, E: int, seed: int) -> Graph:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, D, generator=g)
    src = torch.randint(0, N, (E,), generator=g)
    dst = torch.randint(0, N, (E,), generator=g)
    return Graph(x, torch.stack([src, dst], dim=0).long())


def _bench_one(label: str, transform, graph: Graph, repeats: int) -> Dict[str, Any]:
    t0 = time.perf_counter()
    for _ in range(repeats):
        transform(graph)
    elapsed = time.perf_counter() - t0
    return {
        "transform": label,
        "elapsed_s": elapsed,
        "per_call_ms": elapsed / repeats * 1000.0,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--small", action="store_true")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args(argv)

    N = 64 if args.small else 512
    D = 16 if args.small else 64
    E = 256 if args.small else 4096
    repeats = 3 if args.small else 50
    g = _make_graph(N, D, E, seed=0)

    transforms = [
        ("NormalizeFeatures", NormalizeFeatures()),
        ("StandardizeFeatures", StandardizeFeatures()),
        ("AddSelfLoops", AddSelfLoops()),
        ("ToUndirected", ToUndirected()),
        ("AddDegreeFeatures(both)", AddDegreeFeatures("both", normalize=True)),
        ("AddDegreeEncoding(8)", AddDegreeEncoding(dim=8)),
        ("RandomNodeSplit", RandomNodeSplit(0.6, 0.2, seed=0)),
        ("Compose(5)", Compose([
            NormalizeFeatures(), AddSelfLoops(), ToUndirected(),
            AddDegreeFeatures("both", normalize=True),
            RandomNodeSplit(0.6, 0.2, seed=0),
        ])),
    ]
    results: List[Dict[str, Any]] = [_bench_one(lbl, t, g, repeats) for lbl, t in transforms]

    print(f"\nTGraphX transform benchmark "
          f"(version={tgraphx.__version__}, N={N}, D={D}, E={E}, repeats={repeats})\n")
    print(f"  {'Transform':<28} {'Total (s)':>10} {'Per-call (ms)':>14}")
    print("  " + "-" * 56)
    for r in results:
        print(f"  {r['transform']:<28} {r['elapsed_s']:>10.4f} {r['per_call_ms']:>14.3f}")
    if args.output:
        Path(args.output).write_text(json.dumps({
            "version": tgraphx.__version__,
            "small": args.small,
            "N": N, "D": D, "E": E, "repeats": repeats,
            "results": results,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
