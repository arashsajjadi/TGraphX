"""Smoke test: notebook 33 — Cora citation network sampling and dashboard.

Uses a synthetic SBM fallback by default; pass --no-download=False to attempt
PyG Cora download.

Usage:
    python examples/advanced_real_datasets/cora_sampling_smoke.py
    python examples/advanced_real_datasets/cora_sampling_smoke.py --fast
    python examples/advanced_real_datasets/cora_sampling_smoke.py --fast --no-download
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def main(fast: bool = True, no_download: bool = True) -> None:
    from tgraphx.reproducibility import set_seed
    from tgraphx import Graph, GCNConv, count_parameters
    from tgraphx.loaders import NeighborLoader
    from tgraphx.mining import graph_summary

    SEED = 42
    NUM_CLASSES = 7
    set_seed(SEED, deterministic=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    USING_REAL = False
    g = None

    if not no_download:
        try:
            from tgraphx.datasets import PyGPlanetoidDataset
            pyg_ds = PyGPlanetoidDataset(name="Cora", download=True)
            g_raw = pyg_ds.get(0)
            x = g_raw.node_features
            edge_index = g_raw.edge_index
            y_raw = g_raw.node_labels if g_raw.node_labels is not None else g_raw.y
            pyg_data = pyg_ds._upstream[0]
            train_mask = pyg_data.train_mask.bool()
            val_mask = pyg_data.val_mask.bool()
            g = Graph(node_features=x, edge_index=edge_index, y=y_raw)
            NUM_CLASSES = int(y_raw.max().item()) + 1
            USING_REAL = True
            print(f"[Cora] Real Cora: {x.shape[0]} nodes, {edge_index.shape[1]} edges")
        except Exception as exc:
            print(f"[Cora] PyG/Cora unavailable ({exc}), using synthetic SBM.")

    if not USING_REAL:
        N = 300 if fast else 2000
        FEAT_DIM = 64
        gen = torch.Generator().manual_seed(SEED)
        x = torch.randn(N, FEAT_DIM, generator=gen)
        y_raw = torch.randint(0, NUM_CLASSES, (N,))
        edges_s, edges_d = [], []
        for i in range(N):
            for j in range(i + 1, min(i + 6, N)):
                edges_s.extend([i, j]); edges_d.extend([j, i])
        edge_index = torch.tensor([edges_s, edges_d], dtype=torch.long)
        g = Graph(node_features=x, edge_index=edge_index, y=y_raw)
        perm = torch.randperm(N)
        n_train = int(0.6 * N)
        train_mask = torch.zeros(N, dtype=torch.bool)
        train_mask[perm[:n_train]] = True
        val_mask = torch.zeros(N, dtype=torch.bool)
        val_mask[perm[n_train:int(0.8 * N)]] = True
        print(f"[Cora] Synthetic SBM: {N} nodes, {edge_index.shape[1]} edges")

    N = g.node_features.shape[0]
    feat_dim = g.node_features.shape[1]
    s = graph_summary(g.edge_index, num_nodes=N, directed=True)
    print(f"[Cora] Graph density: {s['density']:.5f}  mean_degree: {s['mean_degree']:.2f}")

    class CoraGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.gc1 = GCNConv(feat_dim, 32)
            self.gc2 = GCNConv(32, NUM_CLASSES)
        def forward(self, x, ei):
            return self.gc2(F.relu(F.dropout(self.gc1(x, ei), p=0.5, training=self.training)), ei)

    model = CoraGCN().to(device)
    print(f"[Cora] Model params: {count_parameters(model):,}")

    loader = NeighborLoader(g, fanouts=[15, 10], batch_size=32, mask=train_mask,
                            shuffle=True, seed=SEED)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    t0 = time.time()
    for epoch in range(1, 4):
        total, nb_ = 0.0, 0
        for batch in loader:
            logits = model(batch.node_features.to(device), batch.edge_index.to(device))
            loss = F.cross_entropy(batch.seed_logits(logits), batch.seed_y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb_ += 1
        avg = total / max(1, nb_)
        print(f"  Epoch {epoch} | loss={avg:.4f}")
        assert math.isfinite(avg)

    elapsed = time.time() - t0
    print(f"[Cora] Smoke PASSED  time={elapsed:.2f}s  "
          f"({'real Cora' if USING_REAL else 'synthetic fallback'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", default=True)
    parser.add_argument("--no-download", action="store_true", default=False)
    args = parser.parse_args()
    main(fast=args.fast, no_download=args.no_download)
