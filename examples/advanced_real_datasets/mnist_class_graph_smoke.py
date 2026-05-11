"""Smoke test for notebook 31 — MNIST class-graph membership with tensor nodes.

Runs in FAST_MODE with a tiny synthetic fallback when torchvision/MNIST is
unavailable, so CI passes without network access.

Usage:
    python examples/advanced_real_datasets/mnist_class_graph_smoke.py
    python examples/advanced_real_datasets/mnist_class_graph_smoke.py --real
"""
from __future__ import annotations

import argparse
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def main(use_real: bool = False) -> None:
    from tgraphx.reproducibility import set_seed
    from tgraphx import Graph, ConvMessagePassing
    from tgraphx.loaders import NeighborLoader

    SEED, N, K, EPOCHS, BATCH = 42, 200, 5, 2, 16
    set_seed(SEED, deterministic=True)

    if use_real:
        try:
            from torchvision import datasets, transforms
            transform = transforms.Compose([transforms.ToTensor()])
            ds = datasets.MNIST(root="/tmp/mnist", train=True,
                                 download=True, transform=transform)
            idx = torch.randperm(len(ds))[:N]
            images = torch.stack([ds[i][0] for i in idx])
            labels = torch.tensor([ds[i][1] for i in idx])
            print(f"MNIST loaded: {images.shape}")
        except Exception as e:
            print(f"MNIST unavailable ({e}), falling back to synthetic.")
            use_real = False

    if not use_real:
        images = torch.randn(N, 1, 28, 28)
        labels = torch.randint(0, 10, (N,))
        print(f"Synthetic MNIST-shaped data: {images.shape}")

    # Build kNN graph
    flat = images.view(N, -1)
    flat_n = flat / flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    sim = flat_n @ flat_n.T
    sim.fill_diagonal_(-1.0)
    _, topk = sim.topk(K, dim=1)
    src = torch.arange(N).unsqueeze(1).expand(-1, K).reshape(-1)
    dst = topk.reshape(-1)
    edge_index = torch.unique(
        torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1), dim=1
    )
    print(f"edge_index: {edge_index.shape}")

    g = Graph(node_features=images, edge_index=edge_index, y=labels)

    class TinyMNISTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = ConvMessagePassing(in_shape=(1, 28, 28), out_shape=(4, 14, 14))
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten(),
                                      nn.Linear(4 * 4 * 4, 10))

        def forward(self, x, ei):
            return self.head(F.relu(self.conv(x, ei)))

    model = TinyMNISTModel()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Sanity forward: use only first 3 nodes and edges within those 3 nodes
    tiny_ei = edge_index[:, (edge_index[0] < 3) & (edge_index[1] < 3)]
    if tiny_ei.shape[1] == 0:
        tiny_ei = torch.zeros(2, 1, dtype=torch.long)  # at least self-loop edge
    with torch.no_grad():
        out = model(images[:3], tiny_ei)
    assert out.shape == (3, 10), f"Unexpected output shape: {out.shape}"

    # Train
    mask = torch.zeros(N, dtype=torch.bool)
    mask[:int(0.8 * N)] = True
    loader = NeighborLoader(g, fanouts=[5, 3], batch_size=BATCH, mask=mask,
                            shuffle=True, seed=SEED)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, n = 0.0, 0
        for batch in loader:
            opt.zero_grad()
            logits = model(batch.node_features, batch.edge_index)
            loss = F.cross_entropy(batch.seed_logits(logits), batch.seed_y)
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        avg = total / max(1, n)
        print(f"Epoch {epoch} | loss={avg:.4f}")
        assert math.isfinite(avg), "Training diverged"

    print("mnist_class_graph_smoke: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true",
                        help="Attempt to download real MNIST (requires network)")
    args = parser.parse_args()
    main(use_real=args.real)
