"""KG training pipeline demonstration.

Uses the FamilyKG synthetic dataset to train TransE and DistMult
with the KGTrainer pipeline and evaluates with filtered ranking.

Usage:
    python examples/kg_training_pipeline_demo.py
    python examples/kg_training_pipeline_demo.py --model distmult --epochs 50
"""
from __future__ import annotations

import argparse
import os

import torch

from tgraphx.kg import (
    FamilyKG,
    TransEModel,
    DistMultModel,
    UniformNegativeSampler,
    FilteredNegativeSampler,
    KGTrainer,
    KGTrainingConfig,
    KGEvaluator,
)
from tgraphx.kg.reports import write_kg_training_report, write_kg_evaluation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="KG training pipeline demo")
    parser.add_argument("--model", choices=("transe", "distmult"), default="distmult")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-run-dir", default="logs/kg_training_demo")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    ds = FamilyKG(num_persons=40, seed=args.seed)
    kg, tr, va, te = ds.kg, ds.train, ds.valid, ds.test
    print(f"KG: {kg}")
    print(f"Train: {tr.num_triples}, Valid: {va.num_triples}, Test: {te.num_triples}")

    N_e, N_r = kg.num_entities, kg.num_relations
    if args.model == "transe":
        model = TransEModel(N_e, N_r, embedding_dim=args.embedding_dim)
        loss_type = "margin"
    else:
        model = DistMultModel(N_e, N_r, embedding_dim=args.embedding_dim)
        loss_type = "softplus"

    sampler = FilteredNegativeSampler(
        N_e, 2, positive_set=kg.positive_triple_set(),
        base_sampler=UniformNegativeSampler(N_e, 1),
    )
    evaluator = KGEvaluator(tr.triples, va.triples, te.triples, N_e)
    cfg = KGTrainingConfig(
        num_epochs=args.epochs, batch_size=32, loss_type=loss_type,
        lr=0.01, seed=args.seed, valid_every=max(1, args.epochs // 4),
    )
    trainer = KGTrainer(model, cfg, tr.triples, sampler=sampler, evaluator=evaluator)
    result = trainer.train()

    # Final test evaluation.
    if te.num_triples > 0:
        test_result = evaluator.evaluate(model, te.triples)
        print(f"Test filtered MRR: {test_result.filt_mrr:.4f}")
        eval_report = test_result.to_dict()
    else:
        eval_report = {}

    os.makedirs(args.output_run_dir, exist_ok=True)
    write_kg_training_report(
        os.path.join(args.output_run_dir, "kg_training_report.json"),
        {**result, "model": args.model, "embedding_dim": args.embedding_dim},
    )
    if eval_report:
        write_kg_evaluation_report(
            os.path.join(args.output_run_dir, "kg_evaluation_report.json"), eval_report
        )
    print(f"Training complete. Final loss: {result['final_loss']:.4f}")
    print(f"Reports written to: {args.output_run_dir}")


if __name__ == "__main__":
    main()
