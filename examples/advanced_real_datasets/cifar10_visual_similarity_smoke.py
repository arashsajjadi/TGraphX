"""Smoke test: notebook 32 — CIFAR-10 patch-graph classification.

Uses TGraphX CIFAR10PatchGraphDataset (true patch graphs) with a synthetic
fallback when torchvision/CIFAR-10 is unavailable.

Usage:
    python examples/advanced_real_datasets/cifar10_visual_similarity_smoke.py
    python examples/advanced_real_datasets/cifar10_visual_similarity_smoke.py --fast
    python examples/advanced_real_datasets/cifar10_visual_similarity_smoke.py --fast --no-download
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def main(fast: bool = True, no_download: bool = True) -> None:
    from tgraphx.reproducibility import set_seed
    from tgraphx import (
        Graph, GraphBatch, GraphDataLoader, ConvMessagePassing,
        count_parameters, global_mean_pool, global_max_pool, build_grid_graph,
    )
    from tgraphx.mining import graph_summary

    SEED = 42
    PATCH_SIZE = 8
    HIDDEN = 16
    NUM_CLASSES = 10
    N_TRAIN = 30 if fast else 200
    set_seed(SEED, deterministic=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_rows = n_cols = 32 // PATCH_SIZE
    n_patches = n_rows * n_cols
    patch_shape = (3, PATCH_SIZE, PATCH_SIZE)
    grid_ei = build_grid_graph(n_rows, n_cols, directed=False, self_loops=True)

    USING_REAL = False
    graphs_train: list = []
    graphs_val: list = []

    if not no_download:
        try:
            from tgraphx.datasets import CIFAR10PatchGraphDataset
            ds = CIFAR10PatchGraphDataset(train=True, download=True,
                                          patch_size=PATCH_SIZE, graph_builder="grid")
            gen = torch.Generator().manual_seed(SEED)
            idx = torch.randperm(len(ds), generator=gen)[:N_TRAIN + 10]
            all_g = [ds.get(i) for i in idx]
            graphs_train = all_g[:N_TRAIN]
            graphs_val = all_g[N_TRAIN:]
            USING_REAL = True
            print(f"[CIFAR] Real CIFAR-10 patch graphs: {len(graphs_train)} train")
        except Exception as exc:
            print(f"[CIFAR] CIFAR-10 unavailable ({exc}), using synthetic.")

    if not USING_REAL:
        gen = torch.Generator().manual_seed(SEED)
        for i in range(N_TRAIN + 10):
            feats = torch.randn(n_patches, 3, PATCH_SIZE, PATCH_SIZE, generator=gen)
            graphs_train.append(
                Graph(node_features=feats, edge_index=grid_ei,
                      graph_label=torch.tensor(i % NUM_CLASSES))
            )
        graphs_val = graphs_train[N_TRAIN:]
        graphs_train = graphs_train[:N_TRAIN]
        print(f"[CIFAR] Synthetic patch graphs: {len(graphs_train)} train")

    s = graph_summary(graphs_train[0].edge_index, num_nodes=n_patches, directed=False)
    assert s["num_nodes"] == n_patches

    class PatchGNN(nn.Module):
        def __init__(self):
            super().__init__()
            C, pH, pW = patch_shape
            self.conv1 = ConvMessagePassing(in_shape=(C, pH, pW),
                                            out_shape=(HIDDEN, pH // 2, pW // 2))
            self.pool_sp = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(
                nn.Linear(HIDDEN * 2, HIDDEN), nn.ReLU(),
                nn.Linear(HIDDEN, NUM_CLASSES),
            )

        def forward(self, batch: GraphBatch) -> torch.Tensor:
            x = batch.node_features
            ei = batch.edge_index
            bi = batch.batch
            h = F.relu(self.conv1(x, ei))
            h = self.pool_sp(h).squeeze(-1).squeeze(-1)
            return self.head(torch.cat([global_mean_pool(h, bi),
                                        global_max_pool(h, bi)], dim=1))

    model = PatchGNN().to(device)
    print(f"[CIFAR] Model params: {count_parameters(model):,}")

    tb = GraphBatch(graphs_train[:2])
    out = model(tb.to(device))
    assert out.shape == (2, NUM_CLASSES), f"Shape error: {out.shape}"

    loader = GraphDataLoader(graphs_train, batch_size=8, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    for i, batch in enumerate(loader):
        if i >= 3:
            break
        batch = batch.to(device)
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.graph_labels.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()

    elapsed = time.time() - t0
    print(f"[CIFAR] Smoke PASSED  loss={loss.item():.4f}  time={elapsed:.2f}s")
    assert loss.item() < 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", default=True)
    parser.add_argument("--no-download", action="store_true", default=False)
    args = parser.parse_args()
    main(fast=args.fast, no_download=args.no_download)
