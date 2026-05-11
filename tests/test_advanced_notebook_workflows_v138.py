"""Workflow regression tests for advanced notebooks (v1.3.8).

Runs the actual TGraphX-centered workflow of each notebook with no network
access, using synthetic fallbacks. Verifies the exact patterns we ship in
the notebooks still work.

Run:
    pytest tests/test_advanced_notebook_workflows_v138.py -q
"""
from __future__ import annotations

import json
import math
import pathlib
import random
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def _tmp_run_dir(name: str) -> pathlib.Path:
    d = pathlib.Path(tempfile.mkdtemp()) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── NB31 mini workflow ────────────────────────────────────────────────────


def test_nb31_workflow_with_edge_attr_and_prototype_edges() -> None:
    from tgraphx.reproducibility import set_seed
    from tgraphx import Graph, ConvMessagePassing, count_parameters
    from tgraphx.loaders import NeighborLoader
    from tgraphx.tracking import write_run_metadata, write_metrics_summary
    import tgraphx

    SEED, N, NUM_CLASSES, K_VISUAL, K_PROTO = 42, 60, 10, 3, 1
    set_seed(SEED, deterministic=True)
    run_dir = _tmp_run_dir("31_test")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gen = torch.Generator().manual_seed(SEED)
    images = torch.randn(N, 1, 28, 28, generator=gen)
    labels = torch.randint(0, NUM_CLASSES, (N,))

    flat = images.view(N, -1).float()
    flat_n = flat / flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    sims = flat_n @ flat_n.T
    sims.fill_diagonal_(-1.0)
    _, topk = sims.topk(K_VISUAL, dim=1)
    src = torch.arange(N).unsqueeze(1).expand(-1, K_VISUAL).reshape(-1)
    dst = topk.reshape(-1)
    vis_edges = torch.cat([torch.stack([src, dst], 0),
                            torch.stack([dst, src], 0)], dim=1)

    perm = torch.randperm(N)
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[perm[:int(0.7 * N)]] = True

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
    proto_feats = proto_flat.view(NUM_CLASSES, 1, 28, 28)
    all_images = torch.cat([images, proto_feats], 0)
    proto_self = torch.stack([torch.arange(N, TOTAL), torch.arange(N, TOTAL)], 0)

    ea_vis = torch.zeros(vis_edges.shape[1])
    ea_proto = torch.ones(proto_edges.shape[1])
    ea_self = torch.full((proto_self.shape[1],), 2.0)
    all_edges = torch.cat([vis_edges, proto_edges, proto_self], 1)
    all_edge_attr = torch.cat([ea_vis, ea_proto, ea_self]).unsqueeze(1)
    all_labels = torch.cat([labels, torch.full((NUM_CLASSES,), -1)])

    g = Graph(node_features=all_images, edge_index=all_edges,
              y=all_labels, edge_attr=all_edge_attr)
    assert g.edge_features is not None
    assert (all_edge_attr[:, 0] == 0).sum() > 0   # visual
    assert (all_edge_attr[:, 0] == 1).sum() > 0   # prototype
    assert (all_edge_attr[:, 0] == 2).sum() > 0   # self-loop

    # Test Graph.to(device)
    g_dev = g.to(device)
    assert g_dev.node_features.device.type == device.split(":")[0]
    assert g_dev.edge_features is not None

    class TinyGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = ConvMessagePassing(in_shape=(1, 28, 28), out_shape=(8, 14, 14))
            self.pool = nn.AdaptiveAvgPool2d(2)
            self.head = nn.Linear(8 * 4, NUM_CLASSES)
        def forward(self, x, ei):
            return self.head(self.pool(F.relu(self.conv(x, ei))).flatten(1))

    model = TinyGNN().to(device)
    loader = NeighborLoader(g, fanouts=[5, 3], batch_size=8,
                            mask=train_mask, shuffle=True, seed=SEED)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for i, batch in enumerate(loader):
        if i >= 2:
            break
        logits = model(batch.node_features.to(device), batch.edge_index.to(device))
        sy = batch.seed_y.to(device)
        valid = sy >= 0
        if valid.sum() == 0:
            continue
        loss = F.cross_entropy(batch.seed_logits(logits)[valid], sy[valid])
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 100

    write_run_metadata(str(run_dir / "run_metadata.json"),
                       tgraphx_version=tgraphx.__version__, seed=SEED)
    assert (run_dir / "run_metadata.json").exists()


# ── NB32 mini workflow ────────────────────────────────────────────────────


def test_nb32_patch_graph_workflow() -> None:
    from tgraphx.reproducibility import set_seed
    from tgraphx import (Graph, GraphBatch, GraphDataLoader, ConvMessagePassing,
                         global_mean_pool, global_max_pool, build_grid_graph)
    import tgraphx

    set_seed(42, deterministic=True)
    PATCH_SIZE, NC = 8, 10
    n_rows = 32 // PATCH_SIZE
    n_patches = n_rows * n_rows
    grid_ei = build_grid_graph(n_rows, n_rows, directed=False, self_loops=True)
    patch_shape = (3, PATCH_SIZE, PATCH_SIZE)

    gen = torch.Generator().manual_seed(42)
    graphs = [Graph(node_features=torch.randn(n_patches, 3, PATCH_SIZE, PATCH_SIZE, generator=gen),
                    edge_index=grid_ei,
                    graph_label=torch.tensor(i % NC))
              for i in range(20)]

    class PatchGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = ConvMessagePassing(in_shape=patch_shape,
                                          out_shape=(16, PATCH_SIZE // 2, PATCH_SIZE // 2))
            self.sp = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, NC))
        def forward(self, batch: GraphBatch) -> torch.Tensor:
            h = F.relu(self.conv(batch.node_features, batch.edge_index))
            h = self.sp(h).squeeze(-1).squeeze(-1)
            return self.head(torch.cat([global_mean_pool(h, batch.batch),
                                        global_max_pool(h, batch.batch)], 1))

    model = PatchGNN()
    loader = GraphDataLoader(graphs, batch_size=4, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for i, batch in enumerate(loader):
        if i >= 2:
            break
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.graph_labels)
        opt.zero_grad(); loss.backward(); opt.step()
    assert math.isfinite(loss.item())


# ── NB33 mini workflow ────────────────────────────────────────────────────


def test_nb33_cora_workflow_synthetic_fallback() -> None:
    from tgraphx.reproducibility import set_seed
    from tgraphx import Graph, GCNConv
    from tgraphx.loaders import NeighborLoader

    set_seed(42, deterministic=True)
    N, FEAT, NC = 80, 16, 7
    gen = torch.Generator().manual_seed(42)
    x = torch.randn(N, FEAT, generator=gen)
    y = torch.randint(0, NC, (N,))
    edges_s, edges_d = [], []
    for i in range(N):
        for j in range(i + 1, min(i + 4, N)):
            edges_s.extend([i, j]); edges_d.extend([j, i])
    edge_index = torch.tensor([edges_s, edges_d], dtype=torch.long)
    g = Graph(node_features=x, edge_index=edge_index, y=y)

    perm = torch.randperm(N)
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[perm[:int(0.6 * N)]] = True

    class CoraGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.gc1 = GCNConv(FEAT, 16)
            self.gc2 = GCNConv(16, NC)
        def forward(self, x, ei):
            return self.gc2(F.relu(self.gc1(x, ei)), ei)

    class FlattenMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(FEAT, 16), nn.ReLU(), nn.Linear(16, NC))
        def forward(self, x, ei):
            return self.net(x)

    model = CoraGCN()
    baseline = FlattenMLP()

    loader = NeighborLoader(g, fanouts=[6, 4], batch_size=16,
                            mask=train_mask, shuffle=True, seed=42)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for i, batch in enumerate(loader):
        if i >= 2:
            break
        logits = model(batch.node_features, batch.edge_index)
        loss = F.cross_entropy(batch.seed_logits(logits), batch.seed_y)
        opt.zero_grad(); loss.backward(); opt.step()
    assert math.isfinite(loss.item())


# ── NB34 mini workflow ────────────────────────────────────────────────────


def test_nb34_kg_multirelational_workflow() -> None:
    from tgraphx.reproducibility import set_seed
    from tgraphx import KnowledgeGraph, KGTrainer, KGTrainingConfig
    from tgraphx.kg import TransEModel, KGEvaluator
    from tgraphx.kg.hpo import run_kg_hpo

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    MAX_USERS, MAX_MOVIES, NUM_GENRES, NUM_OCC = 15, 30, 4, 2
    MOVIE_OFFSET = MAX_USERS
    GENRE_OFFSET = MOVIE_OFFSET + MAX_MOVIES
    OCC_OFFSET = GENRE_OFFSET + NUM_GENRES
    NUM_ENTITIES = OCC_OFFSET + NUM_OCC
    NUM_RELATIONS = 4

    random.seed(42)
    triples = []
    for uid in range(MAX_USERS):
        for _ in range(4):
            mid = random.randint(0, MAX_MOVIES - 1) + MOVIE_OFFSET
            triples.append([uid, random.choice([0, 1]), mid])  # rated_high / rated_low
    for mid in range(MAX_MOVIES):
        triples.append([mid + MOVIE_OFFSET, 2,
                        random.randint(0, NUM_GENRES - 1) + GENRE_OFFSET])  # has_genre
    for uid in range(MAX_USERS):
        triples.append([uid, 3, random.randint(0, NUM_OCC - 1) + OCC_OFFSET])  # has_occ

    triples_tensor = torch.tensor(triples, dtype=torch.long)
    entity_features = torch.zeros(NUM_ENTITIES, NUM_GENRES)
    for mid in range(MAX_MOVIES):
        entity_features[mid + MOVIE_OFFSET, mid % NUM_GENRES] = 1.0

    kg = KnowledgeGraph(triples_tensor, num_entities=NUM_ENTITIES,
                        num_relations=NUM_RELATIONS,
                        entity_features={"genre_vec": entity_features})

    # KGTrainer
    model = TransEModel(NUM_ENTITIES, NUM_RELATIONS, embedding_dim=8)
    config = KGTrainingConfig(num_epochs=2, batch_size=16, device=device, seed=42)
    trainer = KGTrainer(model, config, triples_tensor)
    history = trainer.fit()
    assert math.isfinite(history["final_loss"])

    # HPO
    hpo = run_kg_hpo(kg, model_names=["TransE"],
                     search_space={"embedding_dim": [8], "lr": [1e-3]},
                     max_trials=1, epochs=2, device=device)
    assert "mrr" in hpo.best_metrics

    # KG.to(device)
    kg_dev = kg.to(device)
    assert kg_dev.triples.device.type == device.split(":")[0]


# ── NB35 mini workflow ────────────────────────────────────────────────────


def test_nb35_mutag_workflow_with_edge_attr() -> None:
    from tgraphx.reproducibility import set_seed
    from tgraphx import (Graph, GraphBatch, GraphDataLoader, LinearMessagePassing,
                         global_mean_pool, global_max_pool)
    from tgraphx.mining import motif_profile, graph_summary

    set_seed(42, deterministic=True)
    rng = torch.Generator().manual_seed(42)
    random.seed(42)

    graphs = []
    for i in range(20):
        n = random.randint(8, 14)
        atom_feat = F.one_hot(torch.randint(0, 7, (n,), generator=rng),
                              num_classes=7).float()
        src = torch.randint(0, n, (n * 2,), generator=rng)
        dst = torch.randint(0, n, (n * 2,), generator=rng)
        mask = src != dst
        src, dst = src[mask], dst[mask]
        edge_index = torch.unique(
            torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], 0), dim=1
        )
        n_e = edge_index.shape[1]
        bond_feat = F.one_hot(torch.randint(0, 4, (n_e,), generator=rng),
                              num_classes=4).float()
        g = Graph(node_features=atom_feat, edge_index=edge_index,
                  edge_attr=bond_feat, graph_label=torch.tensor(i % 2))
        graphs.append(g)

    sample = graphs[0]
    assert sample.edge_features is not None
    mp = motif_profile(sample.edge_index, num_nodes=sample.node_features.shape[0],
                       directed=False)
    assert "triangles" in mp

    class MolGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.mp1 = LinearMessagePassing(in_shape=(7,), out_shape=(16,))
            self.mp2 = LinearMessagePassing(in_shape=(16,), out_shape=(16,))
            self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(),
                                      nn.Linear(16, 2))
        def forward(self, batch: GraphBatch) -> torch.Tensor:
            h = F.relu(self.mp1(batch.node_features, batch.edge_index))
            h = F.relu(self.mp2(h, batch.edge_index))
            return self.head(torch.cat([global_mean_pool(h, batch.batch),
                                        global_max_pool(h, batch.batch)], 1))

    model = MolGNN()
    loader = GraphDataLoader(graphs[:16], batch_size=4, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for i, batch in enumerate(loader):
        if i >= 2:
            break
        loss = F.cross_entropy(model(batch), batch.graph_labels)
        opt.zero_grad(); loss.backward(); opt.step()
    assert math.isfinite(loss.item())


# ── Package-level regression: PyG MUTAG label normalization ───────────────


def test_pyg_singleton_graph_label_normalized() -> None:
    """Regression: PyG MUTAG stores graph label as tensor([1]) (shape [1]).
    from_pyg_data should normalize this to scalar so cross_entropy works after batching."""
    import importlib
    pyg = importlib.util.find_spec("torch_geometric")
    if pyg is None:
        pytest.skip("torch_geometric not installed")
    from tgraphx.datasets import PyGTUDatasetAdapter
    from tgraphx import GraphDataLoader

    ds = PyGTUDatasetAdapter(name="MUTAG", download=True)
    g = ds.get(0)
    # graph_label must be scalar tensor (shape []), not [1]
    assert g.graph_label.dim() == 0, (
        f"PyG graph_label not normalized: shape={g.graph_label.shape}"
    )
    loader = GraphDataLoader([ds.get(i) for i in range(4)], batch_size=4)
    for batch in loader:
        # batched graph_labels must be 1D, not 2D
        assert batch.graph_labels.dim() == 1, (
            f"GraphBatch.graph_labels has wrong shape: {batch.graph_labels.shape}"
        )
        # cross_entropy must work
        logits = torch.randn(4, 2)
        loss = F.cross_entropy(logits, batch.graph_labels)
        assert math.isfinite(loss.item())
        break
