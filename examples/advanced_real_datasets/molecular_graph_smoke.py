"""Smoke test: notebook 35 — Molecular graph classification (MUTAG/synthetic).

Uses a synthetic molecule-like fallback by default; pass --no-download=False
to attempt PyG MUTAG download.

Usage:
    python examples/advanced_real_datasets/molecular_graph_smoke.py
    python examples/advanced_real_datasets/molecular_graph_smoke.py --fast
    python examples/advanced_real_datasets/molecular_graph_smoke.py --fast --no-download
"""
from __future__ import annotations

import argparse
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def main(fast: bool = True, no_download: bool = True) -> None:
    from tgraphx.reproducibility import set_seed
    from tgraphx import (
        Graph, GraphBatch, GraphDataLoader, LinearMessagePassing,
        count_parameters, global_mean_pool, global_max_pool,
    )
    from tgraphx.mining import graph_summary, motif_profile, triangle_count, degree_statistics

    SEED = 42
    set_seed(SEED, deterministic=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    graphs: list[Graph] = []
    USING_REAL = False

    if not no_download:
        try:
            from tgraphx.datasets import PyGTUDatasetAdapter
            tu_ds = PyGTUDatasetAdapter(name="MUTAG", download=True)
            graphs = [tu_ds.get(i) for i in range(len(tu_ds))]
            USING_REAL = True
            print(f"[MUTAG] Real MUTAG: {len(graphs)} graphs")
        except Exception as exc:
            print(f"[MUTAG] MUTAG unavailable ({exc}), using synthetic.")

    if not USING_REAL:
        rng = torch.Generator().manual_seed(SEED)
        random.seed(SEED)
        n_graphs = 40 if fast else 188
        for i in range(n_graphs):
            n_atoms = random.randint(10, 20)
            atom_feat = F.one_hot(
                torch.randint(0, 7, (n_atoms,), generator=rng), num_classes=7
            ).float()
            src = torch.randint(0, n_atoms, (n_atoms * 2,), generator=rng)
            dst = torch.randint(0, n_atoms, (n_atoms * 2,), generator=rng)
            mask = src != dst
            src, dst = src[mask], dst[mask]
            edge_index = torch.unique(
                torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], 0), dim=1
            )
            n_e = edge_index.shape[1]
            bond_feat = F.one_hot(
                torch.randint(0, 4, (n_e,), generator=rng), num_classes=4
            ).float()
            graphs.append(Graph(
                node_features=atom_feat,
                edge_index=edge_index,
                edge_features=bond_feat,
                graph_label=torch.tensor(i % 2),
            ))
        print(f"[MUTAG] Synthetic molecule graphs: {len(graphs)}")

    sample = graphs[0]
    NODE_DIM = sample.node_features.shape[1]
    EDGE_DIM = sample.edge_features.shape[1] if sample.edge_features is not None else 0
    print(f"[MUTAG] Sample: nodes={sample.node_features.shape[0]}  "
          f"edges={sample.edge_index.shape[1]}  "
          f"node_dim={NODE_DIM}  edge_dim={EDGE_DIM}")

    # Graph summary
    for i, g_mol in enumerate(graphs[:2]):
        s = graph_summary(g_mol.edge_index, num_nodes=g_mol.node_features.shape[0], directed=False)
        mp = motif_profile(g_mol.edge_index, num_nodes=g_mol.node_features.shape[0], directed=False)
        print(f"  Graph {i}: nodes={s['num_nodes']}  triangles={mp.get('triangles', 0)}")

    # Split
    gen = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(graphs), generator=gen).tolist()
    n_train = int(0.7 * len(graphs))
    graphs_train = [graphs[i] for i in perm[:n_train]]
    graphs_val = [graphs[i] for i in perm[n_train:]]

    # Model
    class MolGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.mp1 = LinearMessagePassing(in_shape=(NODE_DIM,), out_shape=(16,))
            self.mp2 = LinearMessagePassing(in_shape=(16,), out_shape=(16,))
            self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2))

        def forward(self, batch: GraphBatch) -> torch.Tensor:
            x = batch.node_features
            ei = batch.edge_index
            bi = batch.batch
            h = F.relu(self.mp1(x, ei))
            h = F.relu(self.mp2(h, ei))
            return self.head(torch.cat([global_mean_pool(h, bi),
                                        global_max_pool(h, bi)], 1))

    model = MolGNN().to(device)
    print(f"[MUTAG] Model params: {count_parameters(model):,}")

    # Degree baseline
    def degree_feat(g_list):
        feats = []
        for g_mol in g_list:
            n = g_mol.node_features.shape[0]
            ds = degree_statistics(g_mol.edge_index, num_nodes=n)
            tri = triangle_count(g_mol.edge_index, num_nodes=n)
            feats.append([float(n), float(ds["mean_degree"]), float(tri) / max(1, n)])
        return torch.tensor(feats, dtype=torch.float)

    train_df = degree_feat(graphs_train)
    val_df = degree_feat(graphs_val)
    train_labels = torch.tensor([g_mol.graph_label.item() for g_mol in graphs_train], dtype=torch.long)
    val_labels = torch.tensor([g_mol.graph_label.item() for g_mol in graphs_val], dtype=torch.long)

    loader = GraphDataLoader(graphs_train, batch_size=8, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    t0 = time.time()
    for epoch in range(1, 4):
        total, nb_ = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.graph_labels.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb_ += 1
        avg = total / max(1, nb_)
        print(f"  Epoch {epoch} | loss={avg:.4f}")
        assert math.isfinite(avg)

    elapsed = time.time() - t0
    print(f"[MUTAG] Smoke PASSED  time={elapsed:.2f}s  "
          f"({'real MUTAG' if USING_REAL else 'synthetic fallback'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", default=True)
    parser.add_argument("--no-download", action="store_true", default=False)
    args = parser.parse_args()
    main(fast=args.fast, no_download=args.no_download)
