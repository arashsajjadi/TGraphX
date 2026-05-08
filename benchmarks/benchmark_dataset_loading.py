"""Benchmark dataset loading time (synthetic + folder + optional adapters).

CI-safe by default: ``--small`` keeps every dataset tiny and never
downloads anything.  ``--include-optional`` runs PyG/DGL/OGB / image
folder benchmarks **only when the upstream package is installed**.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

import tgraphx
from tgraphx.datasets import get_dataset, list_datasets


def _measure(label: str, fn) -> Dict[str, Any]:
    t0 = time.perf_counter()
    out = fn()
    elapsed = time.perf_counter() - t0
    n = len(out) if out is not None and hasattr(out, "__len__") else None
    return {"label": label, "elapsed_s": elapsed, "n": n}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--small", action="store_true",
                   help="Use the smallest sane configuration for CI.")
    p.add_argument("--include-optional", action="store_true",
                   help="Also benchmark PyG/DGL/OGB adapters when installed.")
    p.add_argument("--output", type=str, default=None,
                   help="Optional JSON output path.")
    args = p.parse_args(argv)

    n_graphs = 4 if args.small else 32
    results: List[Dict[str, Any]] = []

    results.append(_measure(
        "synthetic:patch_graph",
        lambda: get_dataset("synthetic:patch_graph", num_graphs=n_graphs, seed=0),
    ))
    results.append(_measure(
        "synthetic:volume_graph",
        lambda: get_dataset("synthetic:volume_graph", num_graphs=max(2, n_graphs // 4), seed=0),
    ))
    results.append(_measure(
        "synthetic:node_classification",
        lambda: get_dataset("synthetic:node_classification",
                            num_nodes=40 if args.small else 200, seed=0),
    ))
    results.append(_measure(
        "synthetic:edge_prediction",
        lambda: get_dataset("synthetic:edge_prediction",
                            num_nodes=20 if args.small else 80,
                            num_pos=10, num_neg=10, seed=0),
    ))
    results.append(_measure(
        "synthetic:graph_regression",
        lambda: get_dataset("synthetic:graph_regression",
                            num_graphs=n_graphs, seed=0),
    ))
    results.append(_measure(
        "synthetic:hetero",
        lambda: get_dataset("synthetic:hetero",
                            num_papers=10 if args.small else 30, seed=0),
    ))
    results.append(_measure(
        "synthetic:temporal",
        lambda: get_dataset("synthetic:temporal",
                            num_sequences=4 if args.small else 16,
                            sequence_length=4, seed=0),
    ))

    if args.include_optional:
        # Each block is wrapped so a missing optional dep / network failure
        # does NOT crash the benchmark.
        try:
            from tgraphx.datasets import FakeDataPatchGraphDataset
            results.append(_measure(
                "torchvision:fake_patch",
                lambda: FakeDataPatchGraphDataset(
                    upstream_kwargs={"size": 4, "image_size": (3, 16, 16),
                                     "num_classes": 2},
                    patch_size=4,
                ),
            ))
        except Exception as exc:
            results.append({"label": "torchvision:fake_patch", "error": str(exc)})

    print(f"\nTGraphX dataset loading benchmark "
          f"(version={tgraphx.__version__}, small={args.small})\n")
    print(f"  {'Dataset':<36} {'Time (s)':>10} {'len':>10}")
    print("  " + "-" * 60)
    for r in results:
        if "error" in r:
            print(f"  {r['label']:<36} ERROR: {r['error']}")
        else:
            n_str = "-" if r["n"] is None else str(r["n"])
            print(f"  {r['label']:<36} {r['elapsed_s']:>10.4f} {n_str:>10}")

    if args.output:
        Path(args.output).write_text(
            json.dumps({
                "version": tgraphx.__version__,
                "small": args.small,
                "results": results,
            }, indent=2)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
