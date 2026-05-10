"""Tests for common LLM-generated and first-user code patterns.

These tests verify that reasonable guesses by users and LLMs either:
- work correctly, or
- fail with a helpful, actionable error message.

They simulate the "first Colab" experience and are designed to catch the
specific failure modes documented in the v1.0.1 UX audit.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tgraphx import (
    Graph,
    ConvMessagePassing,
    GraphMiniBatch,
    NeighborLoader,
    map_global_to_local,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def small_tensor_graph():
    """Small synthetic tensor-feature graph for tests."""
    N, C, H, W = 64, 4, 6, 6
    torch.manual_seed(0)
    x = torch.randn(N, C, H, W)
    edge_index = torch.randint(0, N, (2, 256))
    y = torch.randint(0, 3, (N,))
    return Graph(node_features=x, edge_index=edge_index, y=y)


@pytest.fixture
def small_vector_graph():
    """Small synthetic vector-feature graph."""
    N, D = 64, 16
    torch.manual_seed(1)
    x = torch.randn(N, D)
    edge_index = torch.randint(0, N, (2, 256))
    y = torch.randint(0, 4, (N,))
    return Graph(node_features=x, edge_index=edge_index, y=y)


# ── Snippet 1: Graph accepts y ────────────────────────────────────────────────


class TestGraphAcceptsY:
    """Graph should accept y= (PyG-style) as an alias for node_labels."""

    def test_graph_with_y_keyword(self):
        x = torch.randn(20, 8)
        ei = torch.randint(0, 20, (2, 60))
        y = torch.randint(0, 3, (20,))
        g = Graph(node_features=x, edge_index=ei, y=y)
        assert g.y is not None
        assert g.y.shape == (20,)
        assert g.node_labels is g.y

    def test_graph_with_labels_keyword(self):
        x = torch.randn(20, 8)
        ei = torch.randint(0, 20, (2, 60))
        y = torch.randint(0, 3, (20,))
        g = Graph(node_features=x, edge_index=ei, labels=y)
        assert g.labels is not None
        assert g.y is g.labels

    def test_graph_with_node_labels_positional(self):
        """Old API: positional node_labels still works."""
        x = torch.randn(20, 8)
        ei = torch.randint(0, 20, (2, 60))
        y = torch.randint(0, 3, (20,))
        g = Graph(x, ei, node_labels=y)
        assert g.y is not None

    def test_graph_x_alias(self):
        x = torch.randn(20, 8)
        g = Graph(node_features=x)
        assert g.x is g.node_features

    def test_graph_edge_attr_alias(self):
        x = torch.randn(20, 8)
        ei = torch.randint(0, 20, (2, 40))
        ef = torch.randn(40, 4)
        g = Graph(node_features=x, edge_index=ei, edge_attr=ef)
        assert g.edge_attr is g.edge_features
        assert g.edge_attr.shape == (40, 4)

    def test_graph_num_classes(self):
        x = torch.randn(20, 8)
        # Construct deterministic labels covering all 5 classes so the test
        # is independent of global RNG state.
        y = torch.tensor([i % 5 for i in range(20)], dtype=torch.long)
        g = Graph(node_features=x, y=y)
        assert g.num_classes == 5

    def test_graph_has_labels_false(self):
        x = torch.randn(20, 8)
        g = Graph(node_features=x)
        assert not g.has_labels()

    def test_graph_get_labels_raises_helpful_error(self):
        x = torch.randn(20, 8)
        g = Graph(node_features=x)
        with pytest.raises(ValueError, match="Graph labels are missing"):
            g.get_labels()

    def test_graph_with_labels_method(self):
        x = torch.randn(20, 8)
        ei = torch.randint(0, 20, (2, 40))
        y = torch.randint(0, 3, (20,))
        g = Graph(node_features=x, edge_index=ei)
        g2 = g.with_labels(y)
        assert g2.y is not None
        assert g.y is None  # original unmodified

    def test_graph_repr_shows_y_shape(self):
        x = torch.randn(20, 8)
        y = torch.randint(0, 3, (20,))
        g = Graph(node_features=x, y=y)
        r = repr(g)
        assert "y_shape" in r
        assert "device" in r

    def test_graph_train_mask(self):
        x = torch.randn(50, 8)
        y = torch.randint(0, 3, (50,))
        mask = torch.zeros(50, dtype=torch.bool)
        mask[:30] = True
        g = Graph(node_features=x, y=y, train_mask=mask)
        assert g.train_mask is not None
        assert g.train_mask.sum() == 30

    def test_graph_y_setter(self):
        x = torch.randn(20, 8)
        y = torch.randint(0, 3, (20,))
        g = Graph(node_features=x)
        g.y = y
        assert g.y.shape == (20,)
        assert g.node_labels is y

    def test_duplicate_y_and_labels_same_tensor_ok(self):
        """Providing the same tensor via two aliases is accepted."""
        x = torch.randn(20, 8)
        y = torch.randint(0, 3, (20,))
        # Same tensor object → OK.
        g = Graph(node_features=x, y=y, labels=y)
        assert g.y is y

    def test_duplicate_y_and_labels_different_tensors_errors(self):
        """Two different label tensors raise an error."""
        x = torch.randn(20, 8)
        y1 = torch.randint(0, 3, (20,))
        y2 = torch.randint(0, 3, (20,))
        with pytest.raises(ValueError, match="at most one"):
            Graph(node_features=x, y=y1, labels=y2)


# ── Snippet 2: NeighborLoader returns GraphMiniBatch ──────────────────────────


class TestNeighborLoaderReturnsGraphMiniBatch:
    """NeighborLoader must return GraphMiniBatch with direct attribute access."""

    def test_batch_is_graphminibatch(self, small_tensor_graph):
        loader = NeighborLoader(small_tensor_graph, fanouts=[5, 3], batch_size=8, seed=0)
        batch = next(iter(loader))
        assert isinstance(batch, GraphMiniBatch)

    def test_batch_node_features(self, small_tensor_graph):
        loader = NeighborLoader(small_tensor_graph, fanouts=[5, 3], batch_size=8, seed=0)
        batch = next(iter(loader))
        assert batch.node_features.dim() == 4  # [N_sub, C, H, W]
        assert batch.node_features is batch.x

    def test_batch_edge_index(self, small_tensor_graph):
        loader = NeighborLoader(small_tensor_graph, fanouts=[5, 3], batch_size=8, seed=0)
        batch = next(iter(loader))
        assert batch.edge_index.shape[0] == 2

    def test_batch_seed_y(self, small_tensor_graph):
        loader = NeighborLoader(small_tensor_graph, fanouts=[5, 3], batch_size=8, seed=0)
        batch = next(iter(loader))
        assert batch.seed_y.shape == (batch.batch_size,)

    def test_batch_seed_y_labels_alias(self, small_tensor_graph):
        loader = NeighborLoader(small_tensor_graph, fanouts=[5, 3], batch_size=8, seed=0)
        batch = next(iter(loader))
        # seed_labels and seed_y are property aliases — values are equal.
        assert torch.equal(batch.seed_labels, batch.seed_y)

    def test_batch_seed_logits(self, small_tensor_graph):
        C, H, W = 4, 6, 6
        loader = NeighborLoader(small_tensor_graph, fanouts=[5, 3], batch_size=8, seed=0)
        batch = next(iter(loader))
        # Dummy model: output [N_sub, 3] logits.
        N_sub = batch.num_nodes
        dummy_logits = torch.randn(N_sub, 3)
        s = batch.seed_logits(dummy_logits)
        assert s.shape == (batch.batch_size, 3)

    def test_batch_seed_logits_loss_works(self, small_tensor_graph):
        loader = NeighborLoader(small_tensor_graph, fanouts=[5, 3], batch_size=8, seed=0)
        batch = next(iter(loader))
        dummy_logits = torch.randn(batch.num_nodes, 3)
        loss = F.cross_entropy(batch.seed_logits(dummy_logits), batch.seed_y)
        assert torch.isfinite(loss)

    def test_batch_seed_local_indices_valid(self, small_tensor_graph):
        loader = NeighborLoader(small_tensor_graph, fanouts=[5, 3], batch_size=8, seed=0)
        batch = next(iter(loader))
        idx = batch.seed_local_indices
        assert idx.shape == (batch.batch_size,)
        assert idx.min() >= 0
        assert idx.max() < batch.num_nodes

    def test_batch_no_labels_helpful_error(self):
        """seed_y should raise a helpful error when graph has no labels."""
        x = torch.randn(64, 4, 6, 6)
        ei = torch.randint(0, 64, (2, 256))
        g = Graph(node_features=x, edge_index=ei)  # no y
        loader = NeighborLoader(g, fanouts=[5, 3], batch_size=8, seed=0)
        batch = next(iter(loader))
        with pytest.raises(ValueError, match="Graph labels are missing|Batch labels are unavailable"):
            _ = batch.seed_y

    def test_batch_batch_size_attribute(self, small_tensor_graph):
        loader = NeighborLoader(small_tensor_graph, fanouts=[5, 3], batch_size=8, seed=0)
        batch = next(iter(loader))
        assert batch.batch_size == 8

    def test_batch_to_device(self, small_tensor_graph):
        loader = NeighborLoader(small_tensor_graph, fanouts=[5, 3], batch_size=8, seed=0)
        batch = next(iter(loader))
        batch.to("cpu")
        assert batch.node_features.device.type == "cpu"

    def test_batch_loss_method(self, small_tensor_graph):
        loader = NeighborLoader(small_tensor_graph, fanouts=[5, 3], batch_size=8, seed=0)
        batch = next(iter(loader))
        dummy_logits = torch.randn(batch.num_nodes, 3)
        loss = batch.loss(dummy_logits)
        assert torch.isfinite(loss)


# ── Snippet 3: Legacy tuple unpacking still works ─────────────────────────────


class TestLegacyTupleUnpacking:
    """Old code that unpacks (subgraph, seeds) must continue working."""

    def test_tuple_unpacking(self, small_tensor_graph):
        loader = NeighborLoader(small_tensor_graph, fanouts=[5, 3], batch_size=8, seed=0)
        for subgraph, seeds in loader:
            assert isinstance(subgraph, Graph)
            assert isinstance(seeds, torch.Tensor)
            assert seeds.dtype == torch.long
            break

    def test_as_tuple_method(self, small_tensor_graph):
        loader = NeighborLoader(small_tensor_graph, fanouts=[5, 3], batch_size=8, seed=0)
        batch = next(iter(loader))
        subgraph, seeds = batch.as_tuple()
        assert isinstance(subgraph, Graph)
        assert isinstance(seeds, torch.Tensor)


# ── Snippet 4: ConvMessagePassing shape contract ──────────────────────────────


class TestConvMessagePassingShapeContract:
    """ConvMessagePassing preserves spatial dims (H, W); only channels change."""

    def test_channel_change_preserved_spatial(self):
        conv = ConvMessagePassing(in_shape=(4, 8, 8), out_shape=(16, 8, 8))
        N = 20
        x = torch.randn(N, 4, 8, 8)
        ei = torch.randint(0, N, (2, 60))
        out = conv(x, ei)
        assert out.shape == (N, 16, 8, 8), f"Expected [N,16,8,8], got {out.shape}"

    def test_invalid_spatial_rank_mismatch_error(self):
        """Requesting out_shape with different spatial rank must error helpfully."""
        with pytest.raises(ValueError, match="must have the same rank"):
            ConvMessagePassing(in_shape=(4, 8, 8), out_shape=(16, 4))

    def test_3d_volume_shapes(self):
        conv = ConvMessagePassing(in_shape=(4, 4, 4, 4), out_shape=(8, 4, 4, 4))
        N = 10
        x = torch.randn(N, 4, 4, 4, 4)
        ei = torch.randint(0, N, (2, 30))
        out = conv(x, ei)
        assert out.shape == (N, 8, 4, 4, 4)

    def test_spatial_dims_preserved_across_layers(self):
        """Multiple layers should stack correctly."""
        N, C, H, W = 20, 4, 6, 6
        conv1 = ConvMessagePassing(in_shape=(C, H, W), out_shape=(8, H, W))
        conv2 = ConvMessagePassing(in_shape=(8, H, W), out_shape=(16, H, W))
        x = torch.randn(N, C, H, W)
        ei = torch.randint(0, N, (2, 60))
        z = conv1(x, ei).relu()
        out = conv2(z, ei)
        assert out.shape == (N, 16, H, W)

    def test_invalid_in_shape_raises(self):
        """Non-spatial in_shape (1-D feature) should raise helpful error."""
        with pytest.raises(ValueError):
            ConvMessagePassing(in_shape=(64,), out_shape=(128,))


# ── Snippet 5: map_global_to_local helper ────────────────────────────────────


class TestMapGlobalToLocal:
    def test_basic_mapping(self):
        sampled = torch.tensor([10, 20, 30, 40])
        seeds = torch.tensor([30, 10])
        local = map_global_to_local(seeds, sampled)
        assert local.tolist() == [2, 0]

    def test_missing_id_raises(self):
        sampled = torch.tensor([10, 20, 30])
        seeds = torch.tensor([99])
        with pytest.raises(ValueError, match="not found in"):
            map_global_to_local(seeds, sampled)

    def test_sparse_high_id_path(self):
        """Large/sparse global IDs must use the searchsorted path without
        allocating a huge dense lookup."""
        offset = 5_000_000  # Above the dense-path threshold (2M).
        sampled = torch.tensor([offset + 10, offset + 20, offset + 30, offset + 40])
        seeds = torch.tensor([offset + 30, offset + 10])
        local = map_global_to_local(seeds, sampled)
        assert local.tolist() == [2, 0]

    def test_sparse_high_id_missing_raises(self):
        """Sparse path must still produce a helpful error for missing IDs."""
        offset = 5_000_000
        sampled = torch.tensor([offset + 10, offset + 20])
        seeds = torch.tensor([offset + 99])
        with pytest.raises(ValueError, match="not found in"):
            map_global_to_local(seeds, sampled)

    def test_unsorted_sampled_ids(self):
        """Sampled IDs do not have to be sorted in the dense path."""
        sampled = torch.tensor([40, 10, 30, 20])
        seeds = torch.tensor([30, 10, 20, 40])
        local = map_global_to_local(seeds, sampled)
        # 30 → position 2, 10 → position 1, 20 → position 3, 40 → position 0
        assert local.tolist() == [2, 1, 3, 0]


# ── Snippet 5b: graph_features is INPUT features, NOT a label alias ──────────


class TestGraphFeaturesSemantics:
    """graph_features holds graph-level INPUT features, not a target label.

    This test pins the v1.0.2 semantic correction: in v1.0.1 the constructor
    aliased ``graph_features`` to ``graph_label`` (a target).  v1.0.2 stores
    ``graph_features`` as a distinct field for graph-level inputs.
    """

    def test_graph_features_separate_from_graph_label(self):
        x = torch.randn(8, 4)
        gf = torch.randn(16)
        gl = torch.tensor(2)
        g = Graph(node_features=x, graph_features=gf, graph_label=gl)
        assert g.graph_features is gf
        assert g.graph_label is gl

    def test_graph_features_only(self):
        x = torch.randn(8, 4)
        gf = torch.randn(16)
        g = Graph(node_features=x, graph_features=gf)
        assert g.graph_features is gf
        assert g.graph_label is None  # not aliased — separate fields

    def test_graph_features_clone_preserves(self):
        x = torch.randn(8, 4)
        gf = torch.randn(16)
        g = Graph(node_features=x, graph_features=gf)
        g2 = g.clone()
        assert g2.graph_features is not None
        assert g2.graph_features.shape == gf.shape
        # Clone is deep — different underlying tensor.
        assert g2.graph_features.data_ptr() != gf.data_ptr()

    def test_graph_features_to_device(self):
        x = torch.randn(8, 4)
        gf = torch.randn(16)
        g = Graph(node_features=x, graph_features=gf)
        g.to("cpu")
        assert g.graph_features.device.type == "cpu"

    def test_graph_features_repr_contains_shape(self):
        x = torch.randn(8, 4)
        gf = torch.randn(16)
        g = Graph(node_features=x, graph_features=gf)
        assert "graph_features_shape" in repr(g)

    def test_graph_features_device_mismatch_errors(self):
        x = torch.randn(8, 4)
        gf = torch.randn(16)
        # Move only graph_features; don't move x.
        with pytest.raises(ValueError, match="graph_features device"):
            # We deliberately pass mismatched-device tensors.
            # Skip the test on CUDA-only machines if CUDA is unavailable.
            if not torch.cuda.is_available():
                # Simulate a mismatch by putting graph_features on a different
                # device label via meta. We can't construct a real cross-device
                # graph without CUDA, so trigger via assignment + validate().
                g = Graph(node_features=x, graph_features=gf)
                # Manually create a mismatch using meta device
                g.graph_features = torch.randn(16, device="meta")
                g.validate()
            else:
                Graph(node_features=x.cpu(),
                      graph_features=gf.cuda())


# ── Snippet 6: run_graph_rl invalid algorithm ─────────────────────────────────


class TestRunGraphRLErrorMessages:
    def test_invalid_algorithm_name(self):
        from tgraphx import run_graph_rl
        with pytest.raises((ValueError, KeyError), match=r"[Uu]nknown|[Aa]vailable|abc"):
            run_graph_rl(algorithm="abc", env="navigation", episodes=1)

    def test_list_graph_rl_algorithms_nonempty(self):
        from tgraphx import list_graph_rl_algorithms
        algos = list_graph_rl_algorithms()
        assert isinstance(algos, dict)
        assert len(algos) >= 5
        assert "dqn" in algos or "DQN" in str(algos).lower()


# ── Snippet 7: run_graph_generation invalid method ────────────────────────────


class TestRunGraphGenerationErrorMessages:
    def test_invalid_method_name(self):
        from tgraphx import run_graph_generation
        with pytest.raises((ValueError, KeyError)):
            run_graph_generation(method="not_a_method", num_graphs=1)

    def test_list_graph_generation_methods_nonempty(self):
        from tgraphx import list_graph_generation_methods
        methods = list_graph_generation_methods()
        assert isinstance(methods, (dict, list))
        assert len(methods) >= 1


# ── Snippet 8: Easy mode workflows ───────────────────────────────────────────


class TestEasyModeWorkflows:
    def test_synthetic_tensor_node_classification(self):
        import tgraphx as tgx
        data = tgx.easy.synthetic_tensor_node_classification(
            num_nodes=64, node_shape=(4, 6, 6), num_classes=3, num_edges=200, seed=42,
        )
        assert isinstance(data, Graph)
        assert data.y is not None
        assert data.node_features.shape == (64, 4, 6, 6)

    def test_train_node_classifier_tensor(self):
        import tgraphx as tgx
        data = tgx.easy.synthetic_tensor_node_classification(
            num_nodes=64, node_shape=(4, 6, 6), num_classes=3, num_edges=200, seed=42,
        )
        result = tgx.easy.train_node_classifier(
            data, model="tensor_gcn", fanouts=[5, 3], batch_size=16,
            epochs=1, seed=42, verbose=False,
        )
        assert "loss" in result.metrics
        assert "accuracy" in result.metrics
        assert torch.isfinite(torch.tensor(result.metrics["loss"]))
        assert result.model is not None
        assert result.graph is not None
        assert "model" in result.config
        assert "device" in result.config

    def test_train_node_classifier_no_labels_error(self):
        import tgraphx as tgx
        data = tgx.easy.synthetic_tensor_node_classification(
            num_nodes=64, node_shape=(4, 6, 6), num_classes=3, seed=42,
        )
        data_no_labels = Graph(node_features=data.node_features, edge_index=data.edge_index)
        with pytest.raises(tgx.easy.TGraphXLabelError, match="Node labels are required"):
            tgx.easy.train_node_classifier(data_no_labels, epochs=1, verbose=False)

    def test_train_node_classifier_invalid_model_error(self):
        import tgraphx as tgx
        data = tgx.easy.synthetic_tensor_node_classification(
            num_nodes=64, node_shape=(4, 6, 6), num_classes=3, seed=42,
        )
        with pytest.raises(tgx.easy.TGraphXUnknownNameError, match="Unknown model"):
            tgx.easy.train_node_classifier(data, model="not_a_model", epochs=1, verbose=False)

    def test_list_tasks_nonempty(self):
        import tgraphx as tgx
        tasks = tgx.easy.list_tasks()
        assert len(tasks) >= 3
        assert "node_classification" in tasks

    def test_list_models_nonempty(self):
        import tgraphx as tgx
        models = tgx.easy.list_models()
        assert len(models) >= 2

    def test_list_models_by_task(self):
        import tgraphx as tgx
        models = tgx.easy.list_models("node_classification")
        assert "tensor_gcn" in models

    def test_list_samplers_nonempty(self):
        import tgraphx as tgx
        samplers = tgx.easy.list_samplers()
        assert "neighbor" in samplers

    def test_result_to_dict_json_serializable(self):
        import json
        import tgraphx as tgx
        data = tgx.easy.synthetic_tensor_node_classification(
            num_nodes=32, node_shape=(4, 4, 4), num_classes=2, num_edges=64, seed=0,
        )
        result = tgx.easy.train_node_classifier(
            data, epochs=1, batch_size=8, fanouts=[3, 2], verbose=False,
        )
        d = result.to_dict()
        # Should be JSON-serialisable (no tensors in dict).
        json.dumps(d)

    def test_result_to_markdown(self):
        import tgraphx as tgx
        data = tgx.easy.synthetic_tensor_node_classification(
            num_nodes=32, node_shape=(4, 4, 4), num_classes=2, num_edges=64, seed=0,
        )
        result = tgx.easy.train_node_classifier(
            data, epochs=1, batch_size=8, fanouts=[3, 2], verbose=False,
        )
        md = result.to_markdown()
        assert "loss" in md

    def test_doctor_runs(self):
        import tgraphx as tgx
        status = tgx.easy.doctor()
        assert "tgraphx_version" in status
        assert "torch_version" in status

    def test_check_install(self):
        import tgraphx as tgx
        status = tgx.easy.check_install()
        assert "torch_version" in status
        assert "cuda_available" in status

    def test_full_batch_sampler(self):
        import tgraphx as tgx
        data = tgx.easy.synthetic_tensor_node_classification(
            num_nodes=32, node_shape=(4, 4, 4), num_classes=2, num_edges=64, seed=0,
        )
        result = tgx.easy.train_node_classifier(
            data, sampler="full", epochs=1, verbose=False,
        )
        assert "loss" in result.metrics

    def test_invalid_sampler_error(self):
        import tgraphx as tgx
        data = tgx.easy.synthetic_tensor_node_classification(
            num_nodes=32, node_shape=(4, 4, 4), num_classes=2, num_edges=64, seed=0,
        )
        with pytest.raises(tgx.easy.TGraphXUnknownNameError):
            tgx.easy.train_node_classifier(
                data, sampler="not_a_sampler", epochs=1, verbose=False,
            )

    def test_gradients_nonzero(self):
        """Training must update parameters (non-zero gradients)."""
        import tgraphx as tgx
        data = tgx.easy.synthetic_tensor_node_classification(
            num_nodes=64, node_shape=(4, 4, 4), num_classes=3, num_edges=200, seed=7,
        )
        result = tgx.easy.train_node_classifier(
            data, epochs=2, batch_size=16, fanouts=[5, 3], verbose=False,
        )
        model = result.model
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "No gradients found — training may not be working"


# ── Snippet 9: KnowledgeGraph API ────────────────────────────────────────────


class TestKnowledgeGraphAPI:
    def test_kg_constructor(self):
        """KnowledgeGraph accepts a triples tensor [N_t, 3] of (head, rel, tail)."""
        from tgraphx.kg import KnowledgeGraph
        N_e = 10
        N_r = 3
        N_t = 50
        heads = torch.randint(0, N_e, (N_t,))
        rels = torch.randint(0, N_r, (N_t,))
        tails = torch.randint(0, N_e, (N_t,))
        triples = torch.stack([heads, rels, tails], dim=1)  # [N_t, 3]
        kg = KnowledgeGraph(triples, num_entities=N_e, num_relations=N_r)
        assert kg.num_entities == N_e
        assert kg.num_relations == N_r
        assert kg.num_triples == N_t

    def test_kg_from_hrt(self):
        """KnowledgeGraph.from_hrt creates a KG from separate h/r/t tensors."""
        from tgraphx.kg import KnowledgeGraph
        N_e, N_r, N_t = 10, 3, 50
        heads = torch.randint(0, N_e, (N_t,))
        rels  = torch.randint(0, N_r, (N_t,))
        tails = torch.randint(0, N_e, (N_t,))
        kg = KnowledgeGraph.from_hrt(heads, rels, tails,
                                     num_entities=N_e, num_relations=N_r)
        assert kg.num_entities == N_e
        assert kg.num_relations == N_r
        assert kg.num_triples == N_t
        assert kg.triples.shape == (N_t, 3)

    def test_kg_constructor_wrong_shape_error(self):
        """A wrong-shape triples tensor should give a helpful error mentioning from_hrt."""
        from tgraphx.kg import KnowledgeGraph
        heads = torch.randint(0, 5, (10,))
        with pytest.raises(ValueError, match="from_hrt"):
            KnowledgeGraph(heads)  # Wrong: 1-D tensor instead of [N_t, 3]

    def test_list_kg_models(self):
        from tgraphx.kg import list_kg_models
        models = list_kg_models()
        assert isinstance(models, (dict, list))
        assert len(models) >= 3


# ── Snippet 10: Error message quality ────────────────────────────────────────


class TestErrorMessageQuality:
    def test_graph_missing_labels_error_contains_fix(self):
        """The error message for missing labels must tell users how to fix it."""
        g = Graph(node_features=torch.randn(10, 8))
        try:
            g.get_labels()
            assert False, "Should have raised"
        except ValueError as e:
            msg = str(e)
            assert "Graph(..., y=labels)" in msg or "graph.y = labels" in msg

    def test_conv_wrong_rank_error_is_clear(self):
        """ConvMessagePassing should raise on wrong-rank in_shape."""
        try:
            ConvMessagePassing(in_shape=(32,), out_shape=(64,))
            assert False, "Should have raised"
        except ValueError as e:
            assert "2-D" in str(e) or "3-D" in str(e) or "spatial" in str(e).lower()

    def test_neighbor_loader_batch_missing_labels_error(self):
        """seed_y on unlabelled graph must give an actionable error."""
        x = torch.randn(40, 4, 4, 4)
        ei = torch.randint(0, 40, (2, 100))
        g = Graph(node_features=x, edge_index=ei)
        loader = NeighborLoader(g, fanouts=[5, 3], batch_size=8, seed=0)
        batch = next(iter(loader))
        try:
            _ = batch.seed_y
            assert False, "Should have raised"
        except ValueError as e:
            msg = str(e)
            # Must mention y= so users know how to fix it.
            assert "y" in msg.lower() or "label" in msg.lower()

    def test_easy_not_a_graph_error(self):
        """train_node_classifier must error if given a non-Graph argument."""
        import tgraphx as tgx
        with pytest.raises(tgx.easy.TGraphXConfigError, match="tgraphx.Graph"):
            tgx.easy.train_node_classifier(
                {"node_features": torch.randn(10, 8)}, epochs=1, verbose=False,
            )
