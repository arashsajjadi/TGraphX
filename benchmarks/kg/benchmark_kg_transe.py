"""TransE KG benchmark.

Usage:
    python benchmarks/kg/benchmark_kg_transe.py --small --json
    python benchmarks/kg/benchmark_kg_transe.py --num-entities 200 --num-triples 1000 --epochs 50
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import tgraphx
from tgraphx.kg import (
    generate_synthetic_kg,
    TransEModel,
    UniformNegativeSampler,
    FilteredNegativeSampler,
    KGTrainer,
    KGTrainingConfig,
    KGEvaluator,
)
from tgraphx.kg.reports import write_kg_benchmark_report


def main() -> None:
    parser = argparse.ArgumentParser(description="TransE KG benchmark")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-entities", type=int, default=None)
    parser.add_argument("--num-relations", type=int, default=None)
    parser.add_argument("--num-triples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    n_e = args.num_entities or (30 if args.small else 200)
    n_r = args.num_relations or (4 if args.small else 10)
    n_t = args.num_triples or (80 if args.small else 500)
    epochs = args.epochs or (5 if args.small else 50)

    kg = generate_synthetic_kg(n_e, n_r, n_t, seed=args.seed)
    tr, va, te = kg.train_valid_test_split(0.7, 0.15, 0.15, seed=args.seed)

    model = TransEModel(n_e, n_r, embedding_dim=32)
    sampler = UniformNegativeSampler(n_e, num_negatives=2)
    evaluator = KGEvaluator(tr.triples, va.triples, te.triples, n_e, chunk_size=n_e)
    cfg = KGTrainingConfig(
        num_epochs=epochs, batch_size=32 if args.small else 128,
        loss_type="margin", lr=0.01, seed=args.seed, device=args.device,
        valid_every=max(1, epochs // 2),
    )
    trainer = KGTrainer(model, cfg, tr.triples, sampler=sampler, evaluator=evaluator)
    t0 = time.perf_counter()
    result = trainer.train()
    runtime = time.perf_counter() - t0

    # Final evaluation.
    if te.num_triples > 0:
        eval_result = evaluator.evaluate(model, te.triples, device=args.device)
        eval_dict = eval_result.to_dict()
    else:
        eval_dict = {}

    report = {
        "task": "link_prediction",
        "model": "TransE",
        "package_version": tgraphx.__version__,
        "seed": args.seed,
        "device": args.device,
        "num_entities": n_e,
        "num_relations": n_r,
        "num_train_triples": tr.num_triples,
        "num_valid_triples": va.num_triples,
        "num_test_triples": te.num_triples,
        "epochs": epochs,
        "runtime_s": round(runtime, 3),
        "final_loss": result["final_loss"],
        "evaluation": eval_dict,
        "limitation_notes": [
            "Synthetic dataset only — no real-dataset claims.",
            "TransE is Experimental; not benchmarked against reference implementations.",
        ],
    }
    if args.output:
        write_kg_benchmark_report(args.output, report)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        filt_mrr = eval_dict.get("filtered", {}).get("combined", {}).get("MRR", "N/A")
        print(f"TransE: loss={result['final_loss']:.4f} filtered_MRR={filt_mrr} rt={runtime:.2f}s")


if __name__ == "__main__":
    main()
