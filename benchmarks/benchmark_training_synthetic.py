"""Tiny synthetic-dataset training-loop benchmark.

For every key task family, fit a small TGraphX model for a few epochs
and report:

* time per epoch,
* final loss,
* loss decrease (start vs end).

This is a *trainability sanity benchmark*, not a real-world score.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

import tgraphx
from tgraphx import GraphBatch, build_model, fit, set_seed
from tgraphx.datasets import (
    SyntheticEdgePredictionDataset,
    SyntheticGraphRegressionDataset,
    SyntheticNodeClassificationDataset,
    SyntheticPatchGraphDataset,
)


def _patch_graph_classification(small: bool, seed: int) -> Dict[str, Any]:
    set_seed(seed)
    n = 8 if small else 32
    ds = SyntheticPatchGraphDataset(
        num_graphs=n, image_size=16, channels=1, patch_size=4,
        seed=seed,
    )
    batch = GraphBatch(list(ds))
    in_shape = tuple(batch.node_features.shape[1:])
    model = build_model(
        task="graph_classification", layer="conv",
        in_shape=in_shape, hidden_shape=(8, 4, 4),
        num_layers=2, num_classes=ds.metadata.num_classes,
        pooling="mean",
    )
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    losses: List[float] = []
    t0 = time.perf_counter()
    for _ in range(3 if small else 10):
        model.train()
        opt.zero_grad()
        logits = model(batch.node_features, batch.edge_index, batch=batch.batch)
        loss = nn.functional.cross_entropy(logits, batch.graph_labels.long())
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    elapsed = time.perf_counter() - t0
    return {
        "task": "patch_graph_classification",
        "elapsed_s": elapsed,
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "improved": losses[-1] < losses[0],
    }


def _node_classification(small: bool, seed: int) -> Dict[str, Any]:
    set_seed(seed)
    ds = SyntheticNodeClassificationDataset(
        num_nodes=40 if small else 200,
        feature_dim=8, num_classes=3, seed=seed,
    )
    g = ds[0]
    masks = g.metadata["masks"]
    model = build_model(
        task="node_classification", layer="linear",
        in_shape=(g.node_features.size(1),),
        hidden_shape=(16,), num_layers=2,
        num_classes=ds.metadata.num_classes,
    )
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    losses: List[float] = []
    t0 = time.perf_counter()
    for _ in range(3 if small else 12):
        model.train()
        opt.zero_grad()
        logits = model(g.node_features, g.edge_index)
        loss = nn.functional.cross_entropy(
            logits[masks["train_mask"]],
            g.node_labels[masks["train_mask"]].long(),
        )
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    elapsed = time.perf_counter() - t0
    return {
        "task": "node_classification",
        "elapsed_s": elapsed,
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "improved": losses[-1] < losses[0],
    }


def _graph_regression(small: bool, seed: int) -> Dict[str, Any]:
    set_seed(seed)
    n = 8 if small else 32
    ds = SyntheticGraphRegressionDataset(
        num_graphs=n, image_size=16, patch_size=4, seed=seed,
    )
    batch = GraphBatch(list(ds))
    in_shape = tuple(batch.node_features.shape[1:])
    model = build_model(
        task="graph_regression", layer="conv",
        in_shape=in_shape, hidden_shape=(8, 4, 4),
        num_layers=2, out_dim=1, pooling="mean",
    )
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    losses: List[float] = []
    t0 = time.perf_counter()
    for _ in range(3 if small else 10):
        model.train()
        opt.zero_grad()
        out = model(batch.node_features, batch.edge_index, batch=batch.batch)
        loss = nn.functional.mse_loss(out.squeeze(-1), batch.graph_labels.squeeze(-1).float())
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    elapsed = time.perf_counter() - t0
    return {
        "task": "graph_regression",
        "elapsed_s": elapsed,
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "improved": losses[-1] < losses[0],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--small", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args(argv)

    results = [
        _patch_graph_classification(args.small, args.seed),
        _node_classification(args.small, args.seed),
        _graph_regression(args.small, args.seed),
    ]
    print(f"\nTGraphX synthetic-training benchmark "
          f"(version={tgraphx.__version__}, small={args.small})\n")
    print(f"  {'Task':<32} {'Time (s)':>10} {'Loss start':>12} {'Loss end':>12} {'Improved':>10}")
    print("  " + "-" * 78)
    for r in results:
        print(f"  {r['task']:<32} {r['elapsed_s']:>10.3f} "
              f"{r['loss_start']:>12.4f} {r['loss_end']:>12.4f} "
              f"{str(r['improved']):>10}")

    if args.output:
        Path(args.output).write_text(json.dumps({
            "version": tgraphx.__version__,
            "small": args.small,
            "results": results,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
