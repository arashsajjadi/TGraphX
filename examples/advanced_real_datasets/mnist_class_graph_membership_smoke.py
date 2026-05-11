"""Smoke test: notebook 31 — MNIST class-graph membership with tensor nodes.

Usage:
    python examples/advanced_real_datasets/mnist_class_graph_membership_smoke.py
    python examples/advanced_real_datasets/mnist_class_graph_membership_smoke.py --fast
    python examples/advanced_real_datasets/mnist_class_graph_membership_smoke.py --fast --no-download
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def main(fast: bool = True, no_download: bool = True) -> None:
    from tgraphx.reproducibility import set_seed
    from tgraphx import Graph, ConvMessagePassing, count_parameters
    from tgraphx.loaders import NeighborLoader
    from tgraphx.mining import graph_summary

    SEED, NUM_CLASSES = 42, 10
    N = 200 if fast else 1000
    K_VISUAL, K_PROTO = 3, 1
    HIDDEN = 16
    set_seed(SEED, deterministic=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    USING_REAL = False
    if not no_download:
        try:
            from torchvision import datasets, transforms
            ds = datasets.MNIST(root="/tmp/mnist", train=True,
                                download=True, transform=transforms.ToTensor())
            idx = torch.randperm(len(ds))[:N]
            images = torch.stack([ds[i][0] for i in idx])
            labels = torch.tensor([ds[i][1] for i in idx])
            USING_REAL = True
            print(f"[MNIST] Real MNIST: {images.shape}")
        except Exception as exc:
            print(f"[MNIST] MNIST unavailable ({exc}), using synthetic.")

    if not USING_REAL:
        gen = torch.Generator().manual_seed(SEED)
        images = torch.randn(N, 1, 28, 28, generator=gen)
        labels = torch.randint(0, NUM_CLASSES, (N,))
        print(f"[MNIST] Synthetic MNIST-shaped data: {images.shape}")

    # kNN edges
    flat = images.view(N, -1).float()
    flat_n = flat / flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    CHUNK = 64
    src_list, dst_list = [], []
    for i in range(0, N, CHUNK):
        chunk = flat_n[i:i + CHUNK]
        sims = chunk @ flat_n.T
        sims[:, i:i + CHUNK].fill_diagonal_(-1.0)
        _, topk = sims.topk(K_VISUAL, dim=1)
        base = torch.arange(i, min(i + CHUNK, N)).unsqueeze(1).expand_as(topk)
        src_list.append(base.reshape(-1))
        dst_list.append(topk.reshape(-1))
    vis_edges = torch.stack([torch.cat(src_list), torch.cat(dst_list)], 0)
    vis_edges = torch.cat([vis_edges, vis_edges.flip(0)], 1)
    vis_edges = torch.unique(vis_edges, dim=1)

    # Prototype edges
    perm = torch.randperm(N)
    n_train = int(0.7 * N)
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask = ~train_mask

    proto_flat = torch.zeros(NUM_CLASSES, flat_n.shape[1])
    for c in range(NUM_CLASSES):
        mc = train_mask & (labels == c)
        if mc.sum() > 0:
            proto_flat[c] = flat_n[mc].mean(0)
    proto_flat_n = proto_flat / proto_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    _, best_proto = (flat_n @ proto_flat_n.T).topk(K_PROTO, dim=1)
    proto_src = torch.arange(N).unsqueeze(1).expand(-1, K_PROTO).reshape(-1)
    proto_dst = best_proto.reshape(-1) + N
    proto_edges = torch.stack([proto_src, proto_dst], 0)

    TOTAL = N + NUM_CLASSES
    proto_feats = proto_flat.view(NUM_CLASSES, 1, 28, 28).clamp(-3, 3)
    all_images = torch.cat([images, proto_feats.to(images.dtype)], 0)
    proto_self = torch.stack([torch.arange(N, TOTAL), torch.arange(N, TOTAL)], 0)
    all_edges = torch.unique(torch.cat([vis_edges, proto_edges, proto_self], 1), dim=1)
    all_labels = torch.cat([labels, torch.full((NUM_CLASSES,), -1)])

    g = Graph(node_features=all_images, edge_index=all_edges, y=all_labels)
    print(f"[MNIST] Graph: {g}")

    # Verify summary
    s = graph_summary(vis_edges, num_nodes=N, directed=False)
    assert s["num_nodes"] == N

    class TinyGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = ConvMessagePassing(in_shape=(1, 28, 28), out_shape=(HIDDEN, 14, 14))
            self.pool = nn.AdaptiveAvgPool2d(2)
            self.head = nn.Linear(HIDDEN * 4, NUM_CLASSES)

        def forward(self, x, ei):
            h = F.relu(self.conv1(x, ei))
            return self.head(self.pool(h).flatten(1))

    model = TinyGNN().to(device)
    print(f"[MNIST] Model params: {count_parameters(model):,}")

    loader = NeighborLoader(
        g, fanouts=[5, 3], batch_size=16, mask=train_mask, shuffle=True, seed=SEED
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    for i, batch in enumerate(loader):
        if i >= 3:
            break
        x = batch.node_features.to(device)
        ei = batch.edge_index.to(device)
        logits = model(x, ei)
        sy = batch.seed_y.to(device)
        valid = sy >= 0
        if valid.sum() == 0:
            continue
        loss = F.cross_entropy(batch.seed_logits(logits)[valid], sy[valid])
        opt.zero_grad()
        loss.backward()
        opt.step()

    elapsed = time.time() - t0
    print(f"[MNIST] Smoke PASSED  loss={loss.item():.4f}  time={elapsed:.2f}s")
    assert loss.item() < 100, f"Loss too large: {loss.item()}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", default=True)
    parser.add_argument("--no-download", action="store_true", default=False)
    args = parser.parse_args()
    main(fast=args.fast, no_download=args.no_download)
