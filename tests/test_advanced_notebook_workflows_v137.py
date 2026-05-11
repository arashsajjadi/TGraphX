"""Workflow regression tests for advanced notebooks 31–35 (v1.3.7).

Tests the actual core workflow of each notebook: dataset loading (fallback),
graph/KG construction, one training step, artifact writing. Run in FAST_MODE
with no network access so CI can execute them.

Run:
    pytest tests/test_advanced_notebook_workflows_v137.py -q
"""
from __future__ import annotations

import json
import pathlib
import tempfile
import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ────────────────────────────────────────────────────────────────

def _tmp_run_dir(name: str) -> pathlib.Path:
    d = pathlib.Path(tempfile.mkdtemp()) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── NB31: MNIST class-graph membership ────────────────────────────────────

class TestNB31MNISTWorkflow:
    """Core NB31 workflow: synthetic MNIST-shaped data, kNN + prototype edges,
    ConvMessagePassing, NeighborLoader, artifact writing."""

    def test_end_to_end(self) -> None:
        from tgraphx.reproducibility import set_seed
        from tgraphx import Graph, ConvMessagePassing, count_parameters
        from tgraphx.loaders import NeighborLoader
        from tgraphx.tracking import write_run_metadata, write_metrics_summary
        from tgraphx.mining import graph_summary
        import tgraphx

        SEED, N, NUM_CLASSES, K_VISUAL, K_PROTO = 42, 80, 10, 3, 1
        set_seed(SEED, deterministic=True)
        run_dir = _tmp_run_dir("31_mnist_test")

        # Synthetic MNIST-shaped fallback
        gen = torch.Generator().manual_seed(SEED)
        images = torch.randn(N, 1, 28, 28, generator=gen)
        labels = torch.randint(0, NUM_CLASSES, (N,))

        # kNN visual edges
        flat = images.view(N, -1).float()
        flat_n = flat / flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        sims = flat_n @ flat_n.T
        sims.fill_diagonal_(-1.0)
        _, topk = sims.topk(K_VISUAL, dim=1)
        src = torch.arange(N).unsqueeze(1).expand(-1, K_VISUAL).reshape(-1)
        dst = topk.reshape(-1)
        vis_edges = torch.cat([torch.stack([src, dst], 0),
                                torch.stack([dst, src], 0)], dim=1)

        # Prototype edges (train-only labels)
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
        proto_feats = proto_flat.view(NUM_CLASSES, 1, 28, 28)
        all_images = torch.cat([images, proto_feats], 0)
        proto_self = torch.stack([torch.arange(N, TOTAL), torch.arange(N, TOTAL)], 0)

        # edge_attr: 0=visual, 1=prototype, 2=self
        ea_vis = torch.zeros(vis_edges.shape[1])
        ea_proto = torch.ones(proto_edges.shape[1])
        ea_self = torch.full((proto_self.shape[1],), 2.0)
        all_edges = torch.cat([vis_edges, proto_edges, proto_self], 1)
        all_edge_attr = torch.cat([ea_vis, ea_proto, ea_self]).unsqueeze(1)
        all_labels = torch.cat([labels, torch.full((NUM_CLASSES,), -1)])

        g = Graph(node_features=all_images, edge_index=all_edges,
                  y=all_labels, edge_attr=all_edge_attr)
        assert g.edge_features is not None, "edge_attr not stored"
        assert g.edge_features.shape[0] == all_edges.shape[1]

        # edge_type check
        assert (all_edge_attr[:, 0] == 0).sum() > 0
        assert (all_edge_attr[:, 0] == 1).sum() > 0

        # Graph summary
        s = graph_summary(vis_edges, num_nodes=N, directed=False)
        assert s["num_nodes"] == N

        # Model
        class TinyGNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = ConvMessagePassing(in_shape=(1, 28, 28), out_shape=(8, 14, 14))
                self.pool = nn.AdaptiveAvgPool2d(2)
                self.head = nn.Linear(8 * 4, NUM_CLASSES)
            def forward(self, x, ei):
                return self.head(self.pool(F.relu(self.conv(x, ei))).flatten(1))

        model = TinyGNN()
        loader = NeighborLoader(g, fanouts=[5, 3], batch_size=8,
                                mask=train_mask, shuffle=True, seed=SEED)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for i, batch in enumerate(loader):
            if i >= 2:
                break
            logits = model(batch.node_features, batch.edge_index)
            sy = batch.seed_y
            valid = sy >= 0
            if valid.sum() == 0:
                continue
            loss = F.cross_entropy(batch.seed_logits(logits)[valid], sy[valid])
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert loss.item() < 100

        # Artifacts
        write_run_metadata(str(run_dir / "run_metadata.json"),
                           tgraphx_version=tgraphx.__version__,
                           seed=SEED, fast_mode=True)
        write_metrics_summary(str(run_dir / "metrics_summary.json"),
                               gnn_val_acc=0.1, mlp_val_acc=0.1)
        assert (run_dir / "run_metadata.json").exists()
        data = json.loads((run_dir / "run_metadata.json").read_text())
        assert "tgraphx_version" in data


# ── NB32: CIFAR-10 patch graph ─────────────────────────────────────────────

class TestNB32CIFARWorkflow:
    """NB32: patch graphs, graph-level classification, GraphDataLoader."""

    def test_end_to_end(self) -> None:
        from tgraphx.reproducibility import set_seed
        from tgraphx import (Graph, GraphBatch, GraphDataLoader, ConvMessagePassing,
                              count_parameters, global_mean_pool, global_max_pool,
                              build_grid_graph)
        from tgraphx.tracking import write_run_metadata, write_metrics_summary
        from tgraphx.mining import graph_summary
        import tgraphx

        SEED, PATCH_SIZE, HIDDEN, NC = 42, 8, 16, 10
        set_seed(SEED, deterministic=True)
        run_dir = _tmp_run_dir("32_cifar_test")

        n_rows = n_cols = 32 // PATCH_SIZE
        n_patches = n_rows * n_cols
        grid_ei = build_grid_graph(n_rows, n_cols, directed=False, self_loops=True)
        patch_shape = (3, PATCH_SIZE, PATCH_SIZE)

        gen = torch.Generator().manual_seed(SEED)
        graphs = [Graph(node_features=torch.randn(n_patches, 3, PATCH_SIZE, PATCH_SIZE,
                                                   generator=gen),
                        edge_index=grid_ei,
                        graph_label=torch.tensor(i % NC))
                  for i in range(20)]

        # Inductive split
        graphs_train, graphs_val = graphs[:14], graphs[14:]
        assert len(graphs_train) > 0 and len(graphs_val) > 0

        class PatchGNN(nn.Module):
            def __init__(self):
                super().__init__()
                C, pH, pW = patch_shape
                self.conv = ConvMessagePassing(in_shape=(C, pH, pW),
                                              out_shape=(HIDDEN, pH // 2, pW // 2))
                self.sp = nn.AdaptiveAvgPool2d(1)
                self.head = nn.Sequential(nn.Linear(HIDDEN * 2, HIDDEN), nn.ReLU(),
                                          nn.Linear(HIDDEN, NC))
            def forward(self, batch: GraphBatch) -> torch.Tensor:
                h = F.relu(self.conv(batch.node_features, batch.edge_index))
                h = self.sp(h).squeeze(-1).squeeze(-1)
                return self.head(torch.cat([global_mean_pool(h, batch.batch),
                                            global_max_pool(h, batch.batch)], 1))

        class FlattenMLP(nn.Module):
            def __init__(self):
                super().__init__()
                C, pH, pW = patch_shape
                self.net = nn.Sequential(nn.Flatten(),
                                         nn.Linear(n_patches * C * pH * pW, 64),
                                         nn.ReLU(), nn.Linear(64, NC))
            def forward(self, batch: GraphBatch) -> torch.Tensor:
                return self.net(batch.node_features.view(batch.num_graphs, -1))

        model = PatchGNN()
        baseline = FlattenMLP()

        loader = GraphDataLoader(graphs_train, batch_size=4, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for i, batch in enumerate(loader):
            if i >= 2:
                break
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.graph_labels)
            opt.zero_grad(); loss.backward(); opt.step()
        assert loss.item() < 100

        # Baseline training
        opt_b = torch.optim.Adam(baseline.parameters(), lr=1e-3)
        for batch in GraphDataLoader(graphs_train, batch_size=4):
            logits_b = baseline(batch)
            loss_b = F.cross_entropy(logits_b, batch.graph_labels)
            opt_b.zero_grad(); loss_b.backward(); opt_b.step()
            break

        write_run_metadata(str(run_dir / "run_metadata.json"),
                           tgraphx_version=tgraphx.__version__, seed=SEED)
        write_metrics_summary(str(run_dir / "metrics_summary.json"), gnn_val_acc=0.1)
        with open(run_dir / "benchmark_summary.json", "w") as f:
            json.dump({"task": "graph_classification", "patch_size": PATCH_SIZE}, f)
        assert (run_dir / "benchmark_summary.json").exists()


# ── NB33: Cora citation network ────────────────────────────────────────────

class TestNB33CoraWorkflow:
    """NB33: synthetic SBM fallback, GCNConv, NeighborLoader, sampling metadata."""

    def test_end_to_end(self) -> None:
        from tgraphx.reproducibility import set_seed
        from tgraphx import Graph, GCNConv, count_parameters
        from tgraphx.loaders import NeighborLoader
        from tgraphx.tracking import write_run_metadata, write_metrics_summary, write_sampling_metadata
        from tgraphx.mining import graph_summary
        import tgraphx

        SEED, N, FEAT, NC = 42, 150, 32, 7
        set_seed(SEED, deterministic=True)
        run_dir = _tmp_run_dir("33_cora_test")

        gen = torch.Generator().manual_seed(SEED)
        x = torch.randn(N, FEAT, generator=gen)
        y = torch.randint(0, NC, (N,))
        edges_s, edges_d = [], []
        for i in range(N):
            for j in range(i + 1, min(i + 5, N)):
                edges_s.extend([i, j]); edges_d.extend([j, i])
        edge_index = torch.tensor([edges_s, edges_d], dtype=torch.long)
        g = Graph(node_features=x, edge_index=edge_index, y=y)

        perm = torch.randperm(N)
        train_mask = torch.zeros(N, dtype=torch.bool)
        train_mask[perm[:int(0.6 * N)]] = True
        val_mask = torch.zeros(N, dtype=torch.bool)
        val_mask[perm[int(0.6 * N):int(0.8 * N)]] = True

        # Transductive: all nodes in graph; only train labels used in loss
        s = graph_summary(edge_index, num_nodes=N, directed=True)
        assert s["num_nodes"] == N

        class CoraGCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.gc1 = GCNConv(FEAT, 32)
                self.gc2 = GCNConv(32, NC)
            def forward(self, x, ei):
                return self.gc2(F.relu(self.gc1(x, ei)), ei)

        class FlattenMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(FEAT, 32), nn.ReLU(), nn.Linear(32, NC))
            def forward(self, x, ei):
                return self.net(x)

        model = CoraGCN()
        baseline = FlattenMLP()

        loader = NeighborLoader(g, fanouts=[10, 5], batch_size=16,
                                mask=train_mask, shuffle=True, seed=SEED)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        for i, batch in enumerate(loader):
            if i >= 2:
                break
            logits = model(batch.node_features, batch.edge_index)
            loss = F.cross_entropy(batch.seed_logits(logits), batch.seed_y)
            opt.zero_grad(); loss.backward(); opt.step()
        assert loss.item() < 100

        write_run_metadata(str(run_dir / "run_metadata.json"),
                           tgraphx_version=tgraphx.__version__, seed=SEED)
        write_metrics_summary(str(run_dir / "metrics_summary.json"), gcn_val_acc=0.1)
        write_sampling_metadata(str(run_dir / "sampling_metadata.json"),
                                sampler="NeighborLoader", fanouts=[10, 5])
        assert (run_dir / "sampling_metadata.json").exists()


# ── NB34: MovieLens KG ─────────────────────────────────────────────────────

class TestNB34MovieLensWorkflow:
    """NB34: synthetic multi-relational KG, KGTrainer, run_kg_hpo, top-K."""

    def test_end_to_end(self) -> None:
        from tgraphx.reproducibility import set_seed
        from tgraphx import KnowledgeGraph, KGTrainer, KGTrainingConfig, count_parameters
        from tgraphx.kg import TransEModel, KGEvaluator, write_kg_summary
        from tgraphx.kg.hpo import run_kg_hpo
        from tgraphx.tracking import write_run_metadata, write_metrics_summary
        import tgraphx, random, math

        SEED = 42
        set_seed(SEED)
        run_dir = _tmp_run_dir("34_movielens_test")

        MAX_USERS, MAX_MOVIES, NUM_GENRES, NUM_OCC = 20, 40, 5, 3
        MOVIE_OFFSET = MAX_USERS
        GENRE_OFFSET = MOVIE_OFFSET + MAX_MOVIES
        OCC_OFFSET = GENRE_OFFSET + NUM_GENRES
        NUM_ENTITIES = OCC_OFFSET + NUM_OCC
        NUM_RELATIONS = 4

        random.seed(SEED)
        triples = []
        for uid in range(MAX_USERS):
            for _ in range(4):
                mid = random.randint(0, MAX_MOVIES - 1) + MOVIE_OFFSET
                triples.append([uid, random.choice([0, 1]), mid])
        for mid in range(MAX_MOVIES):
            gi = random.randint(0, NUM_GENRES - 1) + GENRE_OFFSET
            triples.append([mid + MOVIE_OFFSET, 2, gi])
        for uid in range(MAX_USERS):
            oi = random.randint(0, NUM_OCC - 1) + OCC_OFFSET
            triples.append([uid, 3, oi])

        triples_tensor = torch.tensor(triples, dtype=torch.long)
        perm = torch.randperm(len(triples_tensor))
        n_train = int(0.8 * len(triples_tensor))
        n_val = int(0.1 * len(triples_tensor))
        train_triples = triples_tensor[perm[:n_train]]
        val_triples = triples_tensor[perm[n_train:n_train + n_val]]
        test_triples = triples_tensor[perm[n_train + n_val:]]

        # Entity features (genre multi-hot)
        entity_features = torch.zeros(NUM_ENTITIES, NUM_GENRES)
        for mid in range(MAX_MOVIES):
            entity_features[mid + MOVIE_OFFSET, mid % NUM_GENRES] = 1.0

        kg = KnowledgeGraph(triples_tensor, num_entities=NUM_ENTITIES,
                            num_relations=NUM_RELATIONS,
                            entity_features={"genre_vec": entity_features})
        assert "genre_vec" in kg.entity_features
        assert kg.entity_features["genre_vec"].shape[0] == NUM_ENTITIES

        # Leakage policy: edge-wise split; entity IDs shared
        assert len(train_triples) > 0
        assert len(val_triples) > 0
        assert len(test_triples) > 0

        model = TransEModel(NUM_ENTITIES, NUM_RELATIONS, embedding_dim=16)
        config = KGTrainingConfig(num_epochs=2, batch_size=16, device="cpu", seed=SEED)
        trainer = KGTrainer(model, config, train_triples)
        history = trainer.fit()
        assert math.isfinite(history["final_loss"])

        # HPO smoke
        hpo = run_kg_hpo(kg, model_names=["TransE"],
                         search_space={"embedding_dim": [16], "lr": [1e-3]},
                         max_trials=1, epochs=2, device="cpu")
        assert "mrr" in hpo.best_metrics

        # Top-K recommendations
        model.eval()
        movie_ids = torch.arange(MOVIE_OFFSET, MOVIE_OFFSET + MAX_MOVIES)
        titles = {MOVIE_OFFSET + i: f"Movie_{i+1}" for i in range(MAX_MOVIES)}
        with torch.no_grad():
            queries = torch.stack([
                torch.zeros(len(movie_ids), dtype=torch.long),
                torch.zeros(len(movie_ids), dtype=torch.long),
                movie_ids,
            ], 1)
            scores = model.score_triples(queries)
        top5 = scores.argsort(descending=True)[:5]
        for j in top5.tolist():
            eid = movie_ids[j].item()
            assert eid in titles

        kg_summary_data = kg.summary() if hasattr(kg, "summary") else {"entities": NUM_ENTITIES}
        kg_summary_data["relations"] = ["rated_high", "has_genre"]
        write_kg_summary(str(run_dir / "kg_summary.json"), kg_summary_data)
        write_run_metadata(str(run_dir / "run_metadata.json"),
                           tgraphx_version=tgraphx.__version__, seed=SEED)
        with open(run_dir / "benchmark_summary.json", "w") as f:
            json.dump({"mrr": hpo.best_metrics["mrr"], "model": "TransE"}, f)
        assert (run_dir / "kg_summary.json").exists()
        assert (run_dir / "benchmark_summary.json").exists()


# ── NB35: MUTAG molecular graph ────────────────────────────────────────────

class TestNB35MUTAGWorkflow:
    """NB35: synthetic molecules, edge_attr (bond types), mean+max pool, baseline."""

    def test_end_to_end(self) -> None:
        from tgraphx.reproducibility import set_seed
        from tgraphx import (Graph, GraphBatch, GraphDataLoader, LinearMessagePassing,
                              count_parameters, global_mean_pool, global_max_pool)
        from tgraphx.tracking import write_run_metadata, write_metrics_summary
        from tgraphx.mining import graph_summary, motif_profile, triangle_count, degree_statistics
        import tgraphx, random, math

        SEED = 42
        set_seed(SEED, deterministic=True)
        run_dir = _tmp_run_dir("35_mutag_test")
        random.seed(SEED)
        rng = torch.Generator().manual_seed(SEED)

        graphs = []
        for i in range(30):
            n = random.randint(8, 16)
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
        assert sample.edge_features is not None, "edge_attr not on Graph"
        assert sample.edge_features.shape[1] == 4

        # Mining
        s = graph_summary(sample.edge_index, num_nodes=sample.node_features.shape[0],
                          directed=False)
        mp = motif_profile(sample.edge_index, num_nodes=sample.node_features.shape[0],
                           directed=False)
        assert "triangles" in mp

        # Split
        gen = torch.Generator().manual_seed(SEED)
        perm = torch.randperm(len(graphs), generator=gen).tolist()
        n_train = int(0.7 * len(graphs))
        graphs_train = [graphs[i] for i in perm[:n_train]]
        graphs_val = [graphs[i] for i in perm[n_train:]]

        NODE_DIM, EDGE_DIM = 7, 4

        class MolGNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.mp1 = LinearMessagePassing(in_shape=(NODE_DIM,), out_shape=(16,))
                self.mp2 = LinearMessagePassing(in_shape=(16,), out_shape=(16,))
                self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(),
                                          nn.Linear(16, 2))
            def forward(self, batch: GraphBatch) -> torch.Tensor:
                x = batch.node_features
                ei = batch.edge_index
                bi = batch.batch
                h = F.relu(self.mp1(x, ei))
                h = F.relu(self.mp2(h, ei))
                return self.head(torch.cat([global_mean_pool(h, bi),
                                            global_max_pool(h, bi)], 1))

        # Degree-feature baseline
        def deg_feats(gl):
            feats = []
            for gm in gl:
                n = gm.node_features.shape[0]
                ds = degree_statistics(gm.edge_index, num_nodes=n)
                tri = triangle_count(gm.edge_index, num_nodes=n)
                feats.append([float(n), float(ds["mean_degree"]),
                               float(tri) / max(1, n)])
            return torch.tensor(feats, dtype=torch.float)

        class DegreeBaseline(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 2))
            def forward(self, x):
                return self.net(x)

        model = MolGNN()
        baseline = DegreeBaseline()

        loader = GraphDataLoader(graphs_train, batch_size=4, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        for i, batch in enumerate(loader):
            if i >= 2:
                break
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.graph_labels)
            opt.zero_grad(); loss.backward(); opt.step()
        assert loss.item() < 100

        # Baseline
        train_df = deg_feats(graphs_train)
        train_labels = torch.tensor([g.graph_label.item() for g in graphs_train],
                                    dtype=torch.long)
        opt_b = torch.optim.Adam(baseline.parameters(), lr=1e-3)
        logits_b = baseline(train_df)
        loss_b = F.cross_entropy(logits_b, train_labels)
        opt_b.zero_grad(); loss_b.backward(); opt_b.step()

        write_run_metadata(str(run_dir / "run_metadata.json"),
                           tgraphx_version=tgraphx.__version__, seed=SEED)
        write_metrics_summary(str(run_dir / "metrics_summary.json"),
                               gnn_val_acc=0.5, degree_val_acc=0.5)
        with open(run_dir / "benchmark_summary.json", "w") as f:
            json.dump({"task": "graph_classification", "dataset": "synthetic_mutag"}, f)
        assert (run_dir / "run_metadata.json").exists()
        assert (run_dir / "benchmark_summary.json").exists()


# ── KGTrainer CUDA-generator regression ───────────────────────────────────

def test_kgtrainer_generator_device_compat() -> None:
    """Regression: randperm with CPU generator must work on any device."""
    from tgraphx.kg import TransEModel, KGTrainer, KGTrainingConfig
    from tgraphx import KnowledgeGraph
    import math

    triples = torch.zeros((60, 3), dtype=torch.long)
    for i in range(60):
        triples[i] = torch.tensor([i % 10, i % 3, (i + 1) % 10])
    model = TransEModel(10, 3, embedding_dim=8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    config = KGTrainingConfig(num_epochs=2, batch_size=16, device=dev, seed=42)
    trainer = KGTrainer(model, config, triples)
    history = trainer.fit()
    assert math.isfinite(history["final_loss"])


# ── Source-string checks (stale / forbidden text) ─────────────────────────

REPO = pathlib.Path(__file__).parent.parent
NB_DIR = REPO / "colab_drafts" / "advanced_real_datasets"


def _nb_text(filename: str) -> str:
    p = NB_DIR / filename
    if not p.exists():
        pytest.skip(f"Notebook {filename} not present (gitignored).")
    nb = json.loads(p.read_text(encoding="utf-8"))
    return "\n".join("".join(c.get("source", [])) for c in nb["cells"])


@pytest.mark.parametrize("filename,forbidden", [
    ("31_mnist_class_graph_membership_tensor_nodes.ipynb", "e[:120]"),
    ("32_cifar10_visual_similarity_patch_graph.ipynb", "e[:120]"),
    ("33_cora_citation_network_sampling_and_dashboard.ipynb", "graph RL training"),
    ("34_movielens_user_item_kg_recommendation.ipynb", "e[:120]"),
    ("35_molecular_graph_classification_mutag_or_qm9.ipynb", "No edge features are used"),
])
def test_forbidden_text_absent(filename: str, forbidden: str) -> None:
    assert forbidden not in _nb_text(filename), (
        f"{filename}: forbidden text {forbidden!r} found"
    )


@pytest.mark.parametrize("filename,required", [
    ("31_mnist_class_graph_membership_tensor_nodes.ipynb", "edge_type"),
    ("31_mnist_class_graph_membership_tensor_nodes.ipynb", "prototype"),
    ("32_cifar10_visual_similarity_patch_graph.ipynb", "CIFAR10PatchGraphDataset"),
    ("32_cifar10_visual_similarity_patch_graph.ipynb", "leakage"),
    ("33_cora_citation_network_sampling_and_dashboard.ipynb", "FlattenMLP"),
    ("33_cora_citation_network_sampling_and_dashboard.ipynb", "transductive"),
    ("34_movielens_user_item_kg_recommendation.ipynb", "rated_high"),
    ("34_movielens_user_item_kg_recommendation.ipynb", "has_genre"),
    ("34_movielens_user_item_kg_recommendation.ipynb", "entity_features"),
    ("34_movielens_user_item_kg_recommendation.ipynb", "Leakage policy"),
    ("35_molecular_graph_classification_mutag_or_qm9.ipynb", "edge_attr"),
    ("35_molecular_graph_classification_mutag_or_qm9.ipynb", "motif_profile"),
])
def test_required_text_present(filename: str, required: str) -> None:
    assert required in _nb_text(filename), (
        f"{filename}: required text {required!r} not found"
    )
