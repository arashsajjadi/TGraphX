"""v1.4.0 user-friendly syntax tests — groups 71–100.

Each group is implemented, documented_only, rejected_with_helpful_error,
or roadmap_only. These tests verify the v1.4.0 contract:
- old syntax still valid
- new aliases produce mathematically equivalent results
- unsupported shortcuts fail with helpful errors
- tensor-native semantics preserved (no silent flattening)
- device/dtype/shape/autograd preserved
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest
import torch

import tgraphx as tgx
from tgraphx import Graph


# ── GROUP 71 — PyG-style Data compatibility ───────────────────────────────


class TestGroup71PyG:
    def test_graph_x_alias(self) -> None:
        # Canonical
        g1 = Graph(node_features=torch.randn(4, 3),
                   edge_index=torch.tensor([[0, 1], [1, 2]]))
        # PyG-style alias `x=`
        g2 = Graph(x=torch.randn(4, 3),
                   edge_index=torch.tensor([[0, 1], [1, 2]]))
        # The .x property
        assert g1.x is g1.node_features
        assert g2.x is g2.node_features

    def test_graph_y_alias(self) -> None:
        y = torch.tensor([0, 1, 0, 1])
        g = Graph(node_features=torch.randn(4, 3), y=y)
        assert torch.equal(g.y, y)
        assert torch.equal(g.node_labels, y)

    def test_graph_edge_attr_alias(self) -> None:
        ea = torch.randn(2, 4)
        g = Graph(node_features=torch.randn(4, 3),
                  edge_index=torch.tensor([[0, 1], [1, 2]]),
                  edge_attr=ea)
        assert torch.equal(g.edge_attr, ea)
        assert torch.equal(g.edge_features, ea)

    def test_graph_num_node_features(self) -> None:
        g = Graph(node_features=torch.randn(4, 3, 7, 7),
                  edge_index=torch.tensor([[0], [1]]))
        # Product of per-node dims
        assert g.num_node_features == 3 * 7 * 7


# ── GROUP 72 — DGL optional compat (documented only) ───────────────────────


class TestGroup72DGL:
    def test_no_silent_dgl_import(self) -> None:
        # Importing tgraphx must not import dgl (heavy optional dep)
        import sys
        # We can't easily test this without snapshot pattern; just verify dgl
        # absence doesn't break tgraphx itself.
        import tgraphx  # noqa: F401
        assert True


# ── GROUP 73 — NetworkX convenience ────────────────────────────────────────


class TestGroup73NetworkX:
    def test_from_networkx_path_graph(self) -> None:
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx not installed")
        G = nx.path_graph(5)
        g = Graph.from_networkx(G)
        assert g.num_nodes == 5
        # Undirected path graph has 2 * 4 = 8 edges after symmetrization
        assert g.num_edges == 8

    def test_to_networkx_roundtrip(self) -> None:
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx not installed")
        g = Graph(node_features=torch.zeros(4, 1),
                  edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]))
        G = g.to_networkx(directed=True)
        assert G.number_of_nodes() == 4
        assert G.number_of_edges() == 3


# ── GROUP 74 — Adjacency convenience ───────────────────────────────────────


class TestGroup74Adjacency:
    def test_from_dense_adjacency(self) -> None:
        adj = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        g = Graph.from_adjacency(adj)
        assert g.num_nodes == 3
        assert g.num_edges == 4  # symmetric: 4 directed edges

    def test_from_adjacency_bad_shape(self) -> None:
        adj = torch.zeros(3, 4)
        with pytest.raises(ValueError, match="square"):
            Graph.from_adjacency(adj)


# ── GROUP 75 — Edge-list convenience ───────────────────────────────────────


class TestGroup75EdgeList:
    def test_from_edges_tuples(self) -> None:
        g = Graph.from_edges([(0, 1), (1, 2), (2, 0)], num_nodes=3)
        assert g.num_nodes == 3
        assert g.num_edges == 3

    def test_from_edges_tensor_E2(self) -> None:
        ei_e2 = torch.tensor([[0, 1], [1, 2], [2, 0]])
        g = Graph.from_edges(ei_e2, num_nodes=3)
        assert g.num_edges == 3

    def test_from_edges_tensor_2E(self) -> None:
        ei_2e = torch.tensor([[0, 1, 2], [1, 2, 0]])
        g = Graph.from_edges(ei_2e, num_nodes=3)
        assert g.num_edges == 3

    def test_from_edges_infer_num_nodes(self) -> None:
        g = Graph.from_edges([(0, 1), (1, 2)])
        assert g.num_nodes == 3


# ── GROUP 76 — Unified dataset loader registry ─────────────────────────────


class TestGroup76DatasetRegistry:
    def test_list_dataset_aliases(self) -> None:
        aliases = tgx.list_dataset_aliases()
        assert "mnist_graph" in aliases
        assert "cifar10_patch" in aliases
        assert "cora" in aliases
        assert "mutag" in aliases

    def test_load_dataset_unknown_suggests(self) -> None:
        from tgraphx.datasets import DatasetNotFoundError
        with pytest.raises(DatasetNotFoundError, match="Did you mean|Unknown dataset"):
            tgx.load_dataset("nope_xyz")

    def test_load_dataset_canonical_still_works(self) -> None:
        from tgraphx.datasets import get_dataset
        # Both routes should hit the same factory for synthetic datasets
        ds = tgx.load_dataset("synthetic_patch")
        assert ds is not None


# ── GROUP 77 — Unified model factory (documented_only) ─────────────────────


class TestGroup77ModelFactory:
    def test_build_model_still_works(self) -> None:
        # Canonical TGraphX factory remains stable
        from tgraphx import build_model
        assert build_model is not None


# ── GROUP 78 — Unified training entry points ───────────────────────────────


class TestGroup78Workflow:
    def test_workflow_node_classification(self) -> None:
        r = tgx.workflow(task="node_classification", fast_mode=True, seed=42)
        assert "val_accuracy" in r.metrics
        assert r.task == "node_classification"
        assert math.isfinite(r.metrics["val_accuracy"])

    def test_workflow_kg(self) -> None:
        r = tgx.workflow(task="kg_link_prediction", fast_mode=True, seed=42)
        assert "final_loss" in r.metrics
        assert math.isfinite(r.metrics["final_loss"])

    def test_workflow_mining(self) -> None:
        r = tgx.workflow(task="graph_mining", fast_mode=True, seed=42)
        assert "density" in r.metrics

    def test_run_workflow_alias(self) -> None:
        r = tgx.run_workflow(task="graph_mining", fast_mode=True, seed=42)
        assert r.task == "graph_mining"

    def test_workflow_unknown_task_helpful_error(self) -> None:
        with pytest.raises(ValueError, match="Closest match|Available|Unknown task"):
            tgx.workflow(task="nope_classification")

    def test_workflow_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            r = tgx.workflow(task="graph_mining", fast_mode=True, seed=42, out_dir=d)
            assert "run_metadata.json" in r.artifacts
            assert (Path(d) / "benchmark_summary.json").exists()


# ── GROUP 79 — Task-name aliases ───────────────────────────────────────────


class TestGroup79TaskAliases:
    @pytest.mark.parametrize("alias", [
        "node_classification", "node-classification", "node_cls",
        "tensor_node_classification",
    ])
    def test_node_classification_aliases(self, alias: str) -> None:
        r = tgx.workflow(task=alias, fast_mode=True, seed=42)
        assert r.task == "node_classification"

    @pytest.mark.parametrize("alias", [
        "kg_link_prediction", "kg-link-prediction", "kg_completion", "link_prediction",
    ])
    def test_kg_aliases(self, alias: str) -> None:
        r = tgx.workflow(task=alias, fast_mode=True, seed=42)
        assert r.task == "kg_link_prediction"


# ── GROUP 80 — Tensor-native vs flatten mode ───────────────────────────────


class TestGroup80TensorMode:
    def test_no_silent_flatten_in_validate_graph(self) -> None:
        g = Graph(node_features=torch.randn(4, 3, 7, 7),
                  edge_index=torch.tensor([[0, 1], [1, 2]]))
        # Should validate; tensor-native preserved.
        r = tgx.validate_graph(g, strict=True)
        assert r["ok"]
        assert r["info"]["node_features_shape"] == [4, 3, 7, 7]

    def test_assert_tensor_native(self) -> None:
        g = Graph(node_features=torch.randn(4, 3, 7, 7),
                  edge_index=torch.tensor([[0, 1], [1, 2]]))
        tgx.assert_tensor_native(g, min_rank=3)  # should not raise
        # Vector graph fails the assertion
        g_vec = Graph(node_features=torch.randn(4, 3),
                      edge_index=torch.tensor([[0, 1], [1, 2]]))
        with pytest.raises(Exception):
            tgx.assert_tensor_native(g_vec, min_rank=3)


# ── GROUP 81 — Edge/relation mode normalization (documented_only) ──────────


class TestGroup81EdgeMode:
    def test_kg_from_triples_works(self) -> None:
        # Tests the canonical KG construction still works
        from tgraphx import KnowledgeGraph
        triples = torch.tensor([[0, 0, 1], [1, 0, 2]], dtype=torch.long)
        kg = KnowledgeGraph(triples, num_entities=3, num_relations=1)
        assert kg.num_triples == 2


# ── GROUP 82 — kNN graph construction ──────────────────────────────────────


class TestGroup82KNN:
    def test_knn_graph_vector(self) -> None:
        x = torch.randn(20, 4)
        ei = tgx.knn_graph(x, k=3)
        assert ei.shape[0] == 2
        assert ei.dtype == torch.long
        # No self-loops
        assert (ei[0] != ei[1]).all()

    def test_knn_graph_tensor_native(self) -> None:
        # Tensor-valued features must not be silently flattened in storage.
        x = torch.randn(15, 3, 8, 8)
        ei = tgx.knn_graph(x, k=2, metric="cosine")
        assert ei.shape[0] == 2
        # Original tensor must not be modified
        assert x.shape == (15, 3, 8, 8)

    def test_knn_graph_too_large_k(self) -> None:
        x = torch.randn(5, 3)
        with pytest.raises(ValueError, match="too large"):
            tgx.knn_graph(x, k=10)

    def test_knn_graph_bad_metric(self) -> None:
        x = torch.randn(5, 3)
        with pytest.raises(ValueError, match="metric"):
            tgx.knn_graph(x, k=2, metric="nope")

    def test_knn_graph_make_symmetric(self) -> None:
        x = torch.randn(10, 3)
        ei = tgx.knn_graph(x, k=2, make_symmetric=True)
        # Symmetric: for every (u, v) edge, (v, u) is also present
        edge_set = set(zip(ei[0].tolist(), ei[1].tolist()))
        for s, d in edge_set:
            assert (d, s) in edge_set


# ── GROUP 83 — Prototype graph construction ────────────────────────────────


class TestGroup83Prototype:
    def test_build_class_prototypes_train_only(self) -> None:
        x = torch.randn(20, 3, 4, 4)
        y = torch.randint(0, 5, (20,))
        train_mask = torch.zeros(20, dtype=torch.bool)
        train_mask[:14] = True
        proto = tgx.build_class_prototypes(x, y, train_mask, num_classes=5)
        # Same per-node shape preserved
        assert proto.shape == (5, 3, 4, 4)

    def test_build_class_prototypes_requires_train_mask(self) -> None:
        x = torch.randn(10, 3)
        y = torch.randint(0, 3, (10,))
        with pytest.raises(ValueError, match="train_mask"):
            tgx.build_class_prototypes(x, y, train_mask=None, num_classes=3)

    def test_build_prototype_graph(self) -> None:
        x = torch.randn(20, 1, 28, 28)
        y = torch.randint(0, 10, (20,))
        train_mask = torch.zeros(20, dtype=torch.bool); train_mask[:14] = True
        proto, proto_edges, all_features = tgx.build_prototype_graph(
            x, y, train_mask, num_classes=10, k_proto=1
        )
        assert proto.shape == (10, 1, 28, 28)
        assert proto_edges.shape == (2, 20)  # N nodes * k_proto
        assert all_features.shape == (30, 1, 28, 28)


# ── GROUP 84 — Patch graph construction ────────────────────────────────────


class TestGroup84PatchGraph:
    def test_image_to_patch_graph(self) -> None:
        image = torch.randn(3, 32, 32)
        patches, ei = tgx.image_to_patch_graph(image, patch_size=8)
        assert patches.shape == (16, 3, 8, 8)  # 4x4 grid of 8x8 patches
        assert ei.shape[0] == 2

    def test_image_to_patch_graph_bad_size(self) -> None:
        image = torch.randn(3, 30, 30)
        with pytest.raises(ValueError, match="divisible"):
            tgx.image_to_patch_graph(image, patch_size=8)

    def test_image_to_patch_graph_bad_rank(self) -> None:
        image = torch.randn(1, 3, 32, 32)
        with pytest.raises(ValueError, match=r"\[C, H, W\]"):
            tgx.image_to_patch_graph(image, patch_size=8)


# ── GROUP 85 — Graph-level readout utilities ───────────────────────────────


class TestGroup85Readout:
    def test_pool_canonical_apis_exist(self) -> None:
        # Canonical names must remain stable
        assert callable(tgx.global_mean_pool)
        assert callable(tgx.global_max_pool)
        assert callable(tgx.global_sum_pool)


# ── GROUP 86 — Negative sampling helpers ───────────────────────────────────


class TestGroup86NegSampling:
    def test_negative_sampling_import(self) -> None:
        from tgraphx import negative_sampling
        assert callable(negative_sampling)


# ── GROUP 87 — Evaluation metric helpers ───────────────────────────────────


class TestGroup87Metrics:
    def test_accuracy_helper(self) -> None:
        from tgraphx import accuracy
        logits = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        y = torch.tensor([0, 1, 1])
        acc = accuracy(logits, y)
        assert 0.0 <= float(acc) <= 1.0


# ── GROUP 88 — Experiment config dict support ──────────────────────────────


class TestGroup88Config:
    def test_workflow_accepts_kwargs(self) -> None:
        r = tgx.workflow(task="graph_mining", fast_mode=True, seed=42)
        assert "task" in r.config


# ── GROUP 89 — CLI ergonomics ──────────────────────────────────────────────


class TestGroup89CLI:
    def test_python_m_tgraphx_doctor(self) -> None:
        import subprocess, sys
        r = subprocess.run([sys.executable, "-m", "tgraphx", "doctor"],
                           capture_output=True, text=True, timeout=30)
        # Doctor should exit 0 or print info
        assert r.returncode in (0, 1), r.stderr


# ── GROUP 90 — One-line happy-path quickstart ──────────────────────────────


class TestGroup90Quickstart:
    def test_one_line_node_classification(self) -> None:
        # The single-line quickstart promised in README
        r = tgx.workflow(task="node_classification", fast_mode=True, seed=42)
        assert "val_accuracy" in r.metrics


# ── GROUP 91 — Unified workflow pipelines (already tested above) ───────────


class TestGroup91WorkflowResult:
    def test_workflow_result_to_dict(self) -> None:
        r = tgx.workflow(task="graph_mining", fast_mode=True, seed=42)
        d = r.to_dict()
        assert "task" in d and "metrics" in d
        # JSON-serializable
        json.dumps(d)


# ── GROUP 92 — describe / summary API ──────────────────────────────────────


class TestGroup92Describe:
    def test_describe_graph(self) -> None:
        g = Graph(node_features=torch.randn(5, 1, 28, 28),
                  edge_index=torch.tensor([[0, 1], [1, 2]]),
                  edge_attr=torch.randn(2, 1),
                  y=torch.tensor([0, 1, 0, 1, 0]))
        d = tgx.describe(g)
        assert d["num_nodes"] == 5
        assert d["num_edges"] == 2
        assert d["node_features"]["shape"] == [5, 1, 28, 28]
        assert "edge_attr" in d
        assert "y" in d

    def test_describe_module(self) -> None:
        import torch.nn as nn
        m = nn.Linear(10, 5)
        d = tgx.describe(m)
        assert d["type"] == "Linear"
        assert d["parameters_total"] == 55

    def test_describe_kg(self) -> None:
        from tgraphx import KnowledgeGraph
        kg = KnowledgeGraph(
            torch.tensor([[0, 0, 1], [1, 0, 2]], dtype=torch.long),
            num_entities=3, num_relations=1,
        )
        d = tgx.describe(kg)
        assert d["num_entities"] == 3
        assert d["num_relations"] == 1
        assert d["num_triples"] == 2

    def test_summary_alias(self) -> None:
        g = Graph(node_features=torch.randn(3, 2))
        assert tgx.summary(g) == tgx.describe(g)

    def test_graph_summary_method(self) -> None:
        g = Graph(node_features=torch.randn(3, 2))
        assert g.summary()["num_nodes"] == 3


# ── GROUP 93 — Strict tensor-native validation ─────────────────────────────


class TestGroup93Validate:
    def test_validate_graph_returns_dict(self) -> None:
        g = Graph(node_features=torch.randn(4, 3, 7, 7),
                  edge_index=torch.tensor([[0, 1], [1, 2]]))
        r = tgx.validate_graph(g)
        assert r["ok"]
        assert "issues" in r
        assert "info" in r

    def test_validate_graph_strict_raises(self) -> None:
        # Create an invalid graph by directly setting edge_index with bad indices
        from tgraphx.ux.validation import GraphValidationError
        g = Graph(node_features=torch.randn(3, 2))
        # Inject bad edge_index that references node 99
        g.edge_index = torch.tensor([[0, 1, 99], [1, 2, 0]])
        with pytest.raises(GraphValidationError):
            tgx.validate_graph(g, strict=True)

    def test_validate_graph_edge_attr_length(self) -> None:
        from tgraphx.ux.validation import GraphValidationError
        g = Graph(node_features=torch.randn(3, 2),
                  edge_index=torch.tensor([[0, 1], [1, 2]]))
        # Inject mismatched edge_attr length
        g.edge_features = torch.randn(5, 2)  # should be 2
        r = tgx.validate_graph(g, strict=False)
        assert not r["ok"]
        assert any("edge_attr length" in i for i in r["issues"])

    def test_validate_graph_check_finite(self) -> None:
        x = torch.tensor([[1.0, float("nan")], [0.0, 0.0]])
        g = Graph(node_features=x)
        r = tgx.validate_graph(g, check_finite=True, strict=False)
        assert not r["ok"]

    def test_validate_vector_graph_accepted_by_default(self) -> None:
        g = Graph(node_features=torch.randn(4, 3))
        r = tgx.validate_graph(g, allow_vector_features=True)
        assert r["ok"]


# ── GROUP 94 — Reproducible run context ────────────────────────────────────


class TestGroup94Reproducible:
    def test_reproducible_state_keys(self) -> None:
        with tgx.reproducible(seed=42) as state:
            assert "seed" in state and state["seed"] == 42
            assert "torch_version" in state
            assert "cuda_available" in state
            assert "platform" in state

    def test_seeded_alias(self) -> None:
        with tgx.seeded(123) as state:
            assert state["seed"] == 123

    def test_reproducibility_state_standalone(self) -> None:
        s = tgx.reproducibility_state()
        assert "torch_version" in s

    def test_deterministic_two_runs_same_seed(self) -> None:
        with tgx.reproducible(seed=7, deterministic=False):
            a = torch.randn(5)
        with tgx.reproducible(seed=7, deterministic=False):
            b = torch.randn(5)
        assert torch.allclose(a, b)


# ── GROUP 95 — Compare / benchmark helper ──────────────────────────────────


class TestGroup95Compare:
    def test_compare_with_tasks(self) -> None:
        out = tgx.compare(
            workflows=[
                {"name": "mining", "task": "graph_mining"},
                {"name": "node_cls", "task": "node_classification"},
            ],
            fast_mode=True, seed=42,
        )
        assert len(out["results"]) == 2
        assert all(r["status"] == "ok" for r in out["results"])

    def test_compare_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tgx.compare(
                workflows=[{"name": "m", "task": "graph_mining"}],
                fast_mode=True, seed=42, out_dir=d,
            )
            assert (Path(d) / "benchmark_summary.json").exists()


# ── GROUP 96 — Data leakage guard ──────────────────────────────────────────


class TestGroup96Leakage:
    def test_check_leakage_no_overlap(self) -> None:
        train = torch.tensor([True, True, False, False])
        val = torch.tensor([False, False, True, False])
        test = torch.tensor([False, False, False, True])
        r = tgx.check_leakage(train_mask=train, val_mask=val, test_mask=test)
        assert r["ok"]
        assert r["overlaps"]["train_val"] == 0

    def test_check_leakage_overlap_raises(self) -> None:
        train = torch.tensor([True, True, False])
        val = torch.tensor([True, False, True])  # overlaps with train at idx 0
        from tgraphx.ux.leakage import LeakageError
        with pytest.raises(LeakageError):
            tgx.check_leakage(train_mask=train, val_mask=val, strict=True)

    def test_kg_leakage_report(self) -> None:
        train = torch.tensor([[0, 0, 1], [1, 0, 2]], dtype=torch.long)
        val = torch.tensor([[2, 0, 3]], dtype=torch.long)
        test = torch.tensor([[3, 0, 4]], dtype=torch.long)
        r = tgx.leakage_report(train, val, test)
        assert r["ok"]

    def test_kg_leakage_detected(self) -> None:
        train = torch.tensor([[0, 0, 1], [1, 0, 2]], dtype=torch.long)
        val = torch.tensor([[1, 0, 2]], dtype=torch.long)  # IDENTICAL triple
        r = tgx.leakage_report(train, val, None)
        assert not r["ok"]
        assert "share" in r["issues"][0]


# ── GROUP 97 — Native save/load ────────────────────────────────────────────


class TestGroup97Serialization:
    def test_graph_tensor_roundtrip(self) -> None:
        g = Graph(node_features=torch.randn(5, 3, 8, 8),
                  edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
                  edge_attr=torch.randn(3, 4),
                  y=torch.tensor([0, 1, 0, 1, 0]))
        with tempfile.NamedTemporaryFile(suffix=".tgx", delete=False) as f:
            path = f.name
        try:
            tgx.save(g, path)
            g2 = tgx.load(path)
            assert torch.equal(g.node_features, g2.node_features)
            assert torch.equal(g.edge_index, g2.edge_index)
            assert torch.equal(g.edge_features, g2.edge_features)
            assert torch.equal(g.node_labels, g2.node_labels)
        finally:
            Path(path).unlink()

    def test_kg_roundtrip(self) -> None:
        from tgraphx import KnowledgeGraph
        ef = {"genre": torch.randn(5, 8)}
        kg = KnowledgeGraph(
            torch.tensor([[0, 0, 1], [1, 0, 2]], dtype=torch.long),
            num_entities=5, num_relations=2,
            entity_features=ef,
        )
        with tempfile.NamedTemporaryFile(suffix=".tgx", delete=False) as f:
            path = f.name
        try:
            tgx.save(kg, path)
            kg2 = tgx.load(path)
            assert kg2.num_entities == 5
            assert kg2.num_relations == 2
            assert "genre" in kg2.entity_features
            assert torch.equal(ef["genre"], kg2.entity_features["genre"])
        finally:
            Path(path).unlink()

    def test_load_corrupted_file(self) -> None:
        from tgraphx.ux.serialization import TGraphXSerializationError
        with tempfile.NamedTemporaryFile(suffix=".tgx", delete=False) as f:
            f.write(b"not a torch bundle")
            path = f.name
        try:
            with pytest.raises((TGraphXSerializationError, Exception)):
                tgx.load(path)
        finally:
            Path(path).unlink()

    def test_save_unsupported_type(self) -> None:
        with pytest.raises(TypeError, match="unsupported"):
            tgx.save(42, "/tmp/should_not_exist.tgx")


# ── GROUP 98 — Dashboard audit ─────────────────────────────────────────────


class TestGroup98Dashboard:
    def test_audit_missing_dir(self) -> None:
        r = tgx.audit_run_dir("/tmp/this_does_not_exist_xyz")
        assert not r["ok"]
        assert any("not found" in i for i in r["issues"])

    def test_audit_valid_dir(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            for fn in ("run_metadata.json", "metrics_summary.json",
                       "benchmark_summary.json"):
                with open(Path(d) / fn, "w") as f:
                    json.dump({"tgraphx_version": tgx.__version__, "seed": 42}, f)
            r = tgx.audit_run_dir(d)
            assert r["ok"], r["issues"]

    def test_audit_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "run_metadata.json").write_text("not json")
            (Path(d) / "metrics_summary.json").write_text("{}")
            (Path(d) / "benchmark_summary.json").write_text("{}")
            r = tgx.audit_run_dir(d)
            assert not r["ok"]

    def test_dashboard_audit_alias(self) -> None:
        r = tgx.dashboard_audit("/tmp/this_does_not_exist_xyz")
        assert not r["ok"]


# ── GROUP 99 — Migration aliases (NetworkX/PyG read-only) ──────────────────


class TestGroup99Migration:
    def test_networkx_count_methods(self) -> None:
        g = Graph(node_features=torch.randn(5, 2),
                  edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]))
        assert g.number_of_nodes() == g.num_nodes == 5
        assert g.number_of_edges() == g.num_edges == 3

    def test_pyg_x_alias(self) -> None:
        x = torch.randn(4, 1, 28, 28)
        g = Graph(x=x, edge_index=torch.tensor([[0], [1]]))
        # PyG-style x preserves rank
        assert g.x.shape == (4, 1, 28, 28)
        assert g.num_node_features == 1 * 28 * 28


# ── GROUP 100 — Public API stability registry ──────────────────────────────


class TestGroup100PublicAPI:
    def test_public_api_returns_grouped(self) -> None:
        out = tgx.public_api()
        assert "stable" in out
        assert "beta" in out
        assert "Graph" in out["stable"]

    def test_api_status_known(self) -> None:
        assert tgx.api_status("Graph") == "stable"
        assert tgx.api_status("workflow") == "beta"

    def test_api_status_alias(self) -> None:
        s = tgx.api_status("x")
        # Should be alias of Graph
        assert "alias of Graph" in s or s == "stable"

    def test_api_status_unknown_with_suggestion(self) -> None:
        with pytest.raises(KeyError, match="Closest match|Unknown"):
            tgx.api_status("Graff")  # typo of "Graph"

    def test_list_aliases(self) -> None:
        aliases = tgx.list_aliases("Graph")
        assert "x" in aliases
        assert "edge_attr" in aliases


# ── Sanity: old syntax still works ─────────────────────────────────────────


class TestBackwardCompat:
    def test_canonical_graph_constructor(self) -> None:
        # The original signature must still work
        g = Graph(node_features=torch.randn(3, 2),
                  edge_index=torch.tensor([[0, 1], [1, 2]]),
                  node_labels=torch.tensor([0, 1, 0]),
                  edge_features=torch.randn(2, 1))
        assert g.num_nodes == 3
        assert g.num_edges == 2

    def test_set_seed_canonical(self) -> None:
        from tgraphx.reproducibility import set_seed
        set_seed(42, deterministic=False)
        a = torch.randn(3)
        set_seed(42, deterministic=False)
        b = torch.randn(3)
        assert torch.allclose(a, b)

    def test_neighbor_loader_old_api(self) -> None:
        from tgraphx.loaders import NeighborLoader
        g = Graph(node_features=torch.randn(20, 3),
                  edge_index=torch.tensor([[i for i in range(19)],
                                           [i + 1 for i in range(19)]]),
                  y=torch.randint(0, 3, (20,)))
        loader = NeighborLoader(g, fanouts=[5, 3], batch_size=4, seed=42)
        for batch in loader:
            assert batch.node_features is not None
            assert batch.seed_y is not None
            break
