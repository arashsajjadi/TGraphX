"""sklearn-style estimator/pipeline demonstration."""
from __future__ import annotations

import argparse
import os

import torch

from tgraphx import Graph
from tgraphx.estimators import (
    LabelPropagationEstimator, Node2VecEstimator, GraphPipeline,
    node_train_val_test_split,
)
from tgraphx.mining.reports import write_pipeline_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-nodes", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-run-dir", default="logs/sklearn_pipeline_demo")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    N = args.num_nodes
    src = torch.arange(N).repeat_interleave(2)
    dst = torch.cat([(torch.arange(N) + 1) % N, (torch.arange(N) + 2) % N])
    ei = torch.stack([src, dst], dim=0)
    y = torch.randint(0, 3, (N,))
    g = Graph(node_features=torch.randn(N, 8), edge_index=ei, node_labels=y)

    train, val, test = node_train_val_test_split(N, 0.7, 0.15, 0.15, seed=args.seed)
    y_seed = y.clone()
    y_seed[~train] = -1

    est = LabelPropagationEstimator(num_iters=20, alpha=0.5, seed=args.seed)
    est.fit(g, y_seed)
    preds = est.predict(g)
    test_acc = float((preds[test] == y[test]).float().mean().item())

    os.makedirs(args.output_run_dir, exist_ok=True)
    report = {
        "estimator": type(est).__name__,
        "params": est.get_params(),
        "metrics": {"test_accuracy": test_acc},
        "split": {"train": int(train.sum()), "val": int(val.sum()), "test": int(test.sum())},
    }
    out = os.path.join(args.output_run_dir, "pipeline_report.json")
    write_pipeline_report(out, report)
    print(f"wrote {out}: test_accuracy={test_acc:.3f}")


if __name__ == "__main__":
    main()
