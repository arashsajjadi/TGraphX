"""mnist_patch_graph_demo.py — train a small TGraphX model on MNIST patches.

Defaults to ``--no-download`` so this script never touches the network
on its own.  Pass ``--download`` to let torchvision fetch MNIST under
``--root``.
"""
from __future__ import annotations

import argparse
import sys
import tempfile

import torch
import torch.nn as nn

from tgraphx import GraphBatch, build_model, set_seed
from tgraphx.datasets import (
    FakeDataPatchGraphDataset,
    MNISTPatchGraphDataset,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true",
                   help="Fetch MNIST via torchvision; default is offline (FakeData fallback).")
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--num-graphs", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    set_seed(args.seed)
    if args.download:
        print("Fetching MNIST via torchvision (will download on first run).")
        ds = MNISTPatchGraphDataset(
            root=args.root, download=True, patch_size=args.patch_size,
            train=True,
        )
    else:
        print("Offline mode: using torchvision.FakeData (no network).")
        ds = FakeDataPatchGraphDataset(
            root=args.root or tempfile.mkdtemp(),
            upstream_kwargs={
                "size": args.num_graphs,
                "image_size": (1, 28, 28),
                "num_classes": 10,
            },
            patch_size=args.patch_size,
        )

    items = [ds[i] for i in range(min(args.num_graphs, len(ds)))]
    batch = GraphBatch(items)
    in_shape = tuple(batch.node_features.shape[1:])
    model = build_model(
        task="graph_classification", layer="conv",
        in_shape=in_shape, hidden_shape=(8, in_shape[1], in_shape[2]),
        num_layers=2, num_classes=10, pooling="mean",
    )
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    losses = []
    for epoch in range(args.epochs):
        opt.zero_grad()
        logits = model(batch.node_features, batch.edge_index, batch=batch.batch)
        loss = nn.functional.cross_entropy(logits, batch.graph_labels.long())
        loss.backward()
        opt.step()
        losses.append(float(loss))
        print(f"  epoch {epoch}  loss={loss.item():.4f}")
    print(f"Loss decreased: {losses[-1] < losses[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
