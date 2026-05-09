"""Negative sampling benchmark.

Usage:
    python benchmarks/kg/benchmark_kg_negative_sampling.py --small --json
"""
from __future__ import annotations

import argparse
import json
import time

import torch

import tgraphx
from tgraphx.kg import (
    generate_synthetic_kg,
    UniformNegativeSampler,
    BernoulliNegativeSampler,
    FilteredNegativeSampler,
)
from tgraphx.kg.reports import write_kg_benchmark_report


def _bench_sampler(name, sampler, pos, n_rounds, generator):
    neg_sizes = []
    t0 = time.perf_counter()
    for _ in range(n_rounds):
        neg = sampler.sample(pos, generator=generator)
        neg_sizes.append(int(neg.numel() / 3))
    dt = (time.perf_counter() - t0) * 1000 / n_rounds
    return {
        "sampler": name,
        "n_rounds": n_rounds,
        "avg_neg_triples": sum(neg_sizes) / len(neg_sizes),
        "avg_runtime_ms": round(dt, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="KG negative sampling benchmark")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-entities", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    n_e = args.num_entities or (30 if args.small else 200)
    n_r, n_t = 4, 80 if args.small else 500
    n_rounds = 5 if args.small else 20

    kg = generate_synthetic_kg(n_e, n_r, n_t, seed=args.seed)
    pos = kg.triples[:20 if args.small else 64]
    gen = torch.Generator().manual_seed(args.seed)
    pos_set = kg.positive_triple_set()

    uniform = UniformNegativeSampler(n_e, 2)
    bernoulli = BernoulliNegativeSampler(n_e, 2, train_triples=kg.triples)
    filtered = FilteredNegativeSampler(n_e, 2, positive_set=pos_set,
                                       base_sampler=UniformNegativeSampler(n_e, 1))

    results = [
        _bench_sampler("uniform", uniform, pos, n_rounds, gen),
        _bench_sampler("bernoulli", bernoulli, pos, n_rounds, gen),
        _bench_sampler("filtered", filtered, pos, n_rounds, gen),
    ]
    report = {
        "task": "negative_sampling",
        "package_version": tgraphx.__version__,
        "seed": args.seed,
        "num_entities": n_e, "num_triples": n_t,
        "sampler_results": results,
    }
    if args.output:
        write_kg_benchmark_report(args.output, report)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for r in results:
            print(f"[{r['sampler']:>10}] rt={r['avg_runtime_ms']:.3f}ms")


if __name__ == "__main__":
    main()
