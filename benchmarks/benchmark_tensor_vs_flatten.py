"""Compare tensor-aware GNN with a flatten-then-vector baseline.

The dataset is the synthetic patch-graph classification task.  Both
models train on identical inputs; the tensor-aware model uses
``ConvMessagePassing`` directly on ``[C, ph, pw]`` patches; the
baseline flattens patches to a vector before applying
``LinearMessagePassing``.

This benchmark is a *trainability/throughput sanity check*.  It does
**not** justify performance claims for either approach.
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
from tgraphx import (
    GraphBatch,
    LinearMessagePassing,
    NodeClassifier,
    build_model,
    set_seed,
)
from tgraphx.datasets import SyntheticPatchGraphDataset


class _FlattenBaseline(nn.Module):
    """LinearMessagePassing-on-flattened-patches GNN classifier."""

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.gnn1 = LinearMessagePassing(
            in_shape=(in_dim,), out_shape=(hidden_dim,), aggr="mean",
        )
        self.gnn2 = LinearMessagePassing(
            in_shape=(hidden_dim,), out_shape=(hidden_dim,), aggr="mean",
        )
        self.act = nn.ReLU(inplace=False)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch=None):
        x = x.flatten(start_dim=1)  # [N, C*ph*pw]
        x = self.act(self.gnn1(x, edge_index))
        x = self.act(self.gnn2(x, edge_index))
        # Mean pool per graph.
        if batch is None:
            pooled = x.mean(dim=0, keepdim=True)
        else:
            G = int(batch.max().item()) + 1
            pooled = torch.zeros(G, x.size(1), device=x.device)
            pooled.index_add_(0, batch, x)
            counts = torch.bincount(batch, minlength=G).clamp_min(1).to(x.dtype)
            pooled = pooled / counts.unsqueeze(-1)
        return self.head(pooled)


def _train(model, batch, epochs: int, lr: float = 5e-3) -> Dict[str, Any]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: List[float] = []
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(batch.node_features, batch.edge_index, batch=batch.batch)
        loss = nn.functional.cross_entropy(logits, batch.graph_labels.long())
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    elapsed = time.perf_counter() - t0
    return {
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

    set_seed(args.seed)
    n = 8 if args.small else 32
    ds = SyntheticPatchGraphDataset(
        num_graphs=n, image_size=16, channels=1, patch_size=4, seed=args.seed,
    )
    batch = GraphBatch(list(ds))

    C, ph, pw = batch.node_features.shape[1:]
    flat_dim = C * ph * pw

    tensor_model = build_model(
        task="graph_classification", layer="conv",
        in_shape=(C, ph, pw), hidden_shape=(8, ph, pw),
        num_layers=2, num_classes=ds.metadata.num_classes, pooling="mean",
    )
    flatten_model = _FlattenBaseline(flat_dim, 16, ds.metadata.num_classes)

    epochs = 3 if args.small else 10
    set_seed(args.seed)
    tensor_stats = _train(tensor_model, batch, epochs=epochs)
    set_seed(args.seed)
    flat_stats = _train(flatten_model, batch, epochs=epochs)

    print(f"\nTGraphX tensor-vs-flatten benchmark "
          f"(version={tgraphx.__version__}, small={args.small})\n")
    for label, stats in (("tensor (ConvMP)", tensor_stats),
                         ("flatten (LinearMP)", flat_stats)):
        print(
            f"  {label:<22}  time={stats['elapsed_s']:.3f}s "
            f"loss {stats['loss_start']:.4f} -> {stats['loss_end']:.4f}  "
            f"improved={stats['improved']}"
        )
    if args.output:
        Path(args.output).write_text(json.dumps({
            "version": tgraphx.__version__,
            "small": args.small,
            "tensor": tensor_stats,
            "flatten": flat_stats,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
