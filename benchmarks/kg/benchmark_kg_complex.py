"""ComplEx KG benchmark.

Usage:
    python benchmarks/kg/benchmark_kg_complex.py --small --json
"""
from __future__ import annotations

import argparse
import json
import time

import tgraphx
from tgraphx.kg import (
    generate_synthetic_kg, ComplExModel,
    UniformNegativeSampler, KGTrainer, KGTrainingConfig, KGEvaluator,
)
from tgraphx.kg.reports import write_kg_benchmark_report


def main() -> None:
    parser = argparse.ArgumentParser(description="ComplEx KG benchmark (Experimental)")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-entities", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    n_e = args.num_entities or (30 if args.small else 200)
    n_r, n_t = 4, (80 if args.small else 500)
    epochs = args.epochs or (5 if args.small else 50)

    kg = generate_synthetic_kg(n_e, n_r, n_t, seed=args.seed)
    tr, va, te = kg.train_valid_test_split(0.7, 0.15, 0.15, seed=args.seed)
    model = ComplExModel(n_e, n_r, embedding_dim=32)
    sampler = UniformNegativeSampler(n_e, 2)
    evaluator = KGEvaluator(tr.triples, va.triples, te.triples, n_e, chunk_size=n_e)
    cfg = KGTrainingConfig(num_epochs=epochs, batch_size=32 if args.small else 128,
                           loss_type="softplus", lr=0.01, seed=args.seed, device=args.device)
    trainer = KGTrainer(model, cfg, tr.triples, sampler=sampler, evaluator=evaluator)
    t0 = time.perf_counter()
    result = trainer.train()
    runtime = time.perf_counter() - t0
    eval_dict = evaluator.evaluate(model, te.triples, device=args.device).to_dict() if te.num_triples else {}
    report = {
        "task": "link_prediction", "model": "ComplEx",
        "package_version": tgraphx.__version__, "seed": args.seed, "device": args.device,
        "num_entities": n_e, "num_relations": n_r,
        "num_train_triples": tr.num_triples, "num_test_triples": te.num_triples,
        "epochs": epochs, "runtime_s": round(runtime, 3),
        "final_loss": result["final_loss"], "evaluation": eval_dict,
        "limitation_notes": ["Experimental; not benchmarked against reference implementations."],
    }
    if args.output:
        write_kg_benchmark_report(args.output, report)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        mrr = eval_dict.get("filtered", {}).get("combined", {}).get("MRR", "N/A")
        print(f"ComplEx: loss={result['final_loss']:.4f} MRR={mrr} rt={runtime:.2f}s")


if __name__ == "__main__":
    main()
