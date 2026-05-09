"""Filtered ranking evaluation benchmark.

Usage:
    python benchmarks/kg/benchmark_kg_filtered_eval.py --small --json
"""
from __future__ import annotations

import argparse
import json
import time

import torch

import tgraphx
from tgraphx.kg import (
    generate_synthetic_kg,
    DistMultModel,
    evaluate_filtered_ranking,
)
from tgraphx.kg.reports import write_kg_benchmark_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Filtered ranking evaluation benchmark")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-entities", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    n_e = args.num_entities or (20 if args.small else 100)
    n_r = 4
    n_t = 50 if args.small else 200
    chunk = n_e

    kg = generate_synthetic_kg(n_e, n_r, n_t, seed=args.seed)
    tr, va, te = kg.train_valid_test_split(0.7, 0.15, 0.15, seed=args.seed)
    pos_set = kg.positive_triple_set()

    model = DistMultModel(n_e, n_r, embedding_dim=16)
    t0 = time.perf_counter()
    result = evaluate_filtered_ranking(
        model, te.triples, pos_set, n_e,
        filtered=True, chunk_size=chunk,
        hits_at=(1, 3, 10), device=args.device,
    )
    runtime = time.perf_counter() - t0

    report = {
        "task": "filtered_ranking",
        "model": "DistMult (random, untrained)",
        "package_version": tgraphx.__version__,
        "seed": args.seed,
        "num_entities": n_e,
        "num_test_triples": te.num_triples,
        "chunk_size": chunk,
        "runtime_s": round(runtime, 3),
        "results": result.to_dict(),
        "limitation_notes": ["Model is randomly initialised (not trained)."],
    }
    if args.output:
        write_kg_benchmark_report(args.output, report)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Filtered MRR={result.filt_mrr:.4f} MR={result.filt_mr:.1f} rt={runtime:.3f}s")


if __name__ == "__main__":
    main()
