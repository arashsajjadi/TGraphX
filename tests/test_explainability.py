"""Tests for tgraphx.explain (v0.3.0)."""
from __future__ import annotations

import json
import warnings

import pytest
import torch

from tgraphx import Graph, build_model
from tgraphx.datasets import (
    SyntheticNodeClassificationDataset,
    SyntheticPatchGraphDataset,
)
from tgraphx.explain import (
    attention_to_edge_scores,
    edge_gradient_attribution,
    edge_perturbation_attribution,
    export_edge_scores_csv,
    export_explanation_metadata,
    export_patch_heatmap_json,
    integrated_gradients,
    node_feature_saliency,
    patch_saliency_to_image_grid,
)


@pytest.fixture
def patch_graph():
    torch.manual_seed(0)
    ds = SyntheticPatchGraphDataset(num_graphs=1, image_size=8, patch_size=4, seed=0)
    return ds[0]


@pytest.fixture
def patch_model():
    return build_model(
        task="graph_classification", layer="conv",
        in_shape=(1, 4, 4), hidden_shape=(4, 4, 4),
        num_layers=2, num_classes=6, pooling="mean",
    )


@pytest.fixture
def node_graph_and_model():
    ds = SyntheticNodeClassificationDataset(num_nodes=24, num_classes=3,
                                            feature_dim=8, seed=0)
    g = ds[0]
    model = build_model(
        task="node_classification", layer="linear",
        in_shape=(8,), hidden_shape=(8,), num_layers=2, num_classes=3,
    )
    return g, model


# ── Saliency ─────────────────────────────────────────────────────────────────


class TestSaliency:
    def test_shape_matches_features(self, patch_graph, patch_model):
        sal = node_feature_saliency(patch_model, patch_graph,
                                    target=int(patch_graph.graph_label))
        assert sal.shape == patch_graph.node_features.shape

    def test_finite_and_nonzero(self, patch_graph, patch_model):
        sal = node_feature_saliency(patch_model, patch_graph, target=0)
        assert torch.isfinite(sal).all()
        # At least one entry should be non-zero (the model is not constant).
        assert sal.abs().max() > 0

    def test_no_autograd_retention(self, patch_graph, patch_model):
        sal = node_feature_saliency(patch_model, patch_graph, target=0)
        assert not sal.requires_grad

    def test_vector_node_features(self, node_graph_and_model):
        g, model = node_graph_and_model
        sal = node_feature_saliency(model, g, target=0)
        assert sal.shape == g.node_features.shape


# ── Integrated gradients ─────────────────────────────────────────────────────


class TestIntegratedGradients:
    def test_finite(self, patch_graph, patch_model):
        ig = integrated_gradients(patch_model, patch_graph,
                                  target=int(patch_graph.graph_label), steps=4)
        assert ig.shape == patch_graph.node_features.shape
        assert torch.isfinite(ig).all()

    def test_invalid_steps(self, patch_graph, patch_model):
        with pytest.raises(ValueError, match="steps"):
            integrated_gradients(patch_model, patch_graph, steps=1)

    def test_baseline_shape_mismatch(self, patch_graph, patch_model):
        bad_baseline = torch.zeros(2, 2, 2, 2)
        with pytest.raises(ValueError, match="baseline shape"):
            integrated_gradients(patch_model, patch_graph,
                                 baseline=bad_baseline, steps=2)


# ── Edge attribution ─────────────────────────────────────────────────────────


class TestEdgeAttribution:
    def test_perturbation_shape(self, patch_graph, patch_model):
        scores = edge_perturbation_attribution(patch_model, patch_graph,
                                               target=0, max_edges=4)
        assert scores.numel() == 4

    def test_gradient_attribution_shape(self, patch_graph, patch_model):
        # Patch model uses ConvMessagePassing which respects edge_weight,
        # but we still capture the warning if gradient is zero.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = edge_gradient_attribution(patch_model, patch_graph, target=0)
        assert scores.numel() == patch_graph.num_edges


# ── Attention → edge scores ─────────────────────────────────────────────────


class TestAttentionScores:
    def test_basic_alignment(self):
        # Synthesise an attention tensor [E, K] and verify the shape.
        E, K = 6, 2
        attn = torch.softmax(torch.randn(E, K), dim=0)
        ei = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 2]], dtype=torch.long)
        scores = attention_to_edge_scores(attn, ei, head_reduce="mean")
        assert scores.shape == (E,)

    def test_channel_attention(self):
        E, K, C = 4, 2, 3
        attn = torch.rand(E, K, C)
        ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        scores = attention_to_edge_scores(attn, ei)
        assert scores.shape == (E,)


# ── Patch heatmap ────────────────────────────────────────────────────────────


class TestPatchHeatmap:
    def test_image_grid_shape(self, patch_graph, patch_model):
        sal = node_feature_saliency(patch_model, patch_graph, target=0)
        h = patch_saliency_to_image_grid(sal, grid_shape=patch_graph.metadata["grid_shape"])
        assert h.dim() == 2
        assert h.shape == (8, 8)


# ── Export ───────────────────────────────────────────────────────────────────


class TestExport:
    def test_explanation_metadata(self, tmp_path):
        out = export_explanation_metadata(
            tmp_path / "explanation_metadata.json",
            method="saliency", target=0, extra={"note": "demo"},
        )
        assert out.exists()
        meta = json.loads(out.read_text())
        assert meta["method"] == "saliency" and meta["target"] == 0

    def test_edge_scores_csv(self, tmp_path):
        ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        scores = torch.tensor([0.1, 0.5, 0.3])
        out = export_edge_scores_csv(
            tmp_path / "explanation_edges.csv",
            ei, scores, method="perturbation", top_k=2,
        )
        assert out.exists()
        text = out.read_text()
        assert text.startswith("edge_id,src,dst,score,method")

    def test_patch_heatmap_json(self, tmp_path):
        h = torch.rand(8, 8)
        out = export_patch_heatmap_json(
            tmp_path / "explanation_patch_heatmap.json",
            h, grid_shape=(2, 2), method="saliency",
        )
        payload = json.loads(out.read_text())
        assert payload["shape"] == [8, 8]
        assert payload["method"] == "saliency"
