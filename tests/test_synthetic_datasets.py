"""Synthetic dataset tests (v0.2.9)."""
from __future__ import annotations

import pytest
import torch

from tgraphx import Graph, GraphBatch, build_model
from tgraphx.datasets import (
    SyntheticEdgePredictionDataset,
    SyntheticGraphRegressionDataset,
    SyntheticHeteroGraphDataset,
    SyntheticNodeClassificationDataset,
    SyntheticPatchGraphDataset,
    SyntheticTemporalGraphDataset,
    SyntheticVolumeGraphDataset,
)


# ── Patch graph ──────────────────────────────────────────────────────────────


class TestPatchGraph:
    def test_basic_shapes(self):
        ds = SyntheticPatchGraphDataset(num_graphs=4, image_size=16,
                                        patch_size=4, seed=0)
        assert len(ds) == 4
        g = ds[0]
        assert g.node_features.shape[0] == 16  # 4x4 patches
        assert g.node_features.shape[1] == 1   # channels
        assert g.node_features.shape[2:] == (4, 4)
        assert g.graph_label.dtype == torch.long
        assert g.metadata["task"] == "graph_classification"
        assert g.metadata["grid_shape"] == (4, 4)
        assert ds.metadata.num_classes == 6

    def test_determinism_same_seed(self):
        a = SyntheticPatchGraphDataset(num_graphs=3, seed=42)
        b = SyntheticPatchGraphDataset(num_graphs=3, seed=42)
        for ga, gb in zip(a, b):
            assert torch.equal(ga.node_features, gb.node_features)
            assert int(ga.graph_label) == int(gb.graph_label)

    def test_different_seeds_change_data(self):
        a = SyntheticPatchGraphDataset(num_graphs=3, seed=1)
        b = SyntheticPatchGraphDataset(num_graphs=3, seed=2)
        # At least one feature tensor must differ.
        diff = any(not torch.equal(ga.node_features, gb.node_features)
                   for ga, gb in zip(a, b))
        assert diff

    def test_image_not_divisible_raises(self):
        with pytest.raises(ValueError):
            SyntheticPatchGraphDataset(image_size=15, patch_size=4)

    def test_regression_task(self):
        ds = SyntheticPatchGraphDataset(
            num_graphs=4, task="graph_regression", seed=0,
        )
        for g in ds:
            assert g.graph_label.dtype == torch.float

    def test_tiny_overfit_loss_decreases(self):
        torch.manual_seed(0)
        ds = SyntheticPatchGraphDataset(
            num_graphs=8, image_size=16, patch_size=4, seed=0,
        )
        batch = GraphBatch(list(ds))
        in_shape = tuple(batch.node_features.shape[1:])
        model = build_model(
            task="graph_classification", layer="conv",
            in_shape=in_shape, hidden_shape=(8, 4, 4),
            num_layers=2, num_classes=ds.metadata.num_classes, pooling="mean",
        )
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        losses = []
        for _ in range(8):
            opt.zero_grad()
            logits = model(batch.node_features, batch.edge_index, batch=batch.batch)
            loss = torch.nn.functional.cross_entropy(logits, batch.graph_labels.long())
            loss.backward()
            opt.step()
            losses.append(loss.detach().item())
        assert losses[-1] < losses[0]
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


# ── Volume graph ─────────────────────────────────────────────────────────────


class TestVolumeGraph:
    def test_shape(self):
        ds = SyntheticVolumeGraphDataset(num_graphs=2, volume_size=8,
                                         patch_size=4, seed=0)
        g = ds[0]
        assert g.node_features.shape[0] == 8  # 2x2x2 patches
        assert g.node_features.shape[1:] == (1, 4, 4, 4)


# ── Node classification ─────────────────────────────────────────────────────


class TestNodeClassification:
    def test_basic(self):
        ds = SyntheticNodeClassificationDataset(num_nodes=40, num_classes=3, seed=0)
        assert len(ds) == 1
        g = ds[0]
        assert g.num_nodes == 40
        assert g.node_labels.shape == (40,)
        masks = g.metadata["masks"]
        for k in ("train_mask", "val_mask", "test_mask"):
            assert masks[k].dtype == torch.bool
            assert masks[k].numel() == 40

    def test_split_disjoint(self):
        ds = SyntheticNodeClassificationDataset(num_nodes=40, seed=0)
        m = ds[0].metadata["masks"]
        union = m["train_mask"] | m["val_mask"] | m["test_mask"]
        assert union.all()
        # Pairwise disjoint:
        assert not (m["train_mask"] & m["val_mask"]).any()
        assert not (m["val_mask"] & m["test_mask"]).any()
        assert not (m["train_mask"] & m["test_mask"]).any()


# ── Edge prediction ──────────────────────────────────────────────────────────


class TestEdgePrediction:
    def test_basic(self):
        ds = SyntheticEdgePredictionDataset(num_nodes=20, num_pos=10, num_neg=10, seed=0)
        g = ds[0]
        assert g.edge_labels is not None
        assert set(g.edge_labels.unique().tolist()).issubset({0, 1})


# ── Graph regression ─────────────────────────────────────────────────────────


class TestGraphRegression:
    def test_labels_continuous(self):
        ds = SyntheticGraphRegressionDataset(num_graphs=4, seed=0)
        for g in ds:
            assert g.graph_label.dtype == torch.float


# ── Hetero ───────────────────────────────────────────────────────────────────


class TestHetero:
    def test_basic(self):
        ds = SyntheticHeteroGraphDataset(num_papers=8, num_authors=5, num_venues=3, seed=0)
        hg = ds[0]
        assert "paper" in hg.node_types
        assert "author" in hg.node_types
        assert "venue" in hg.node_types
        for et in hg.edge_types:
            ei = hg.edge_index(et)
            assert ei.dtype == torch.long
            assert ei.dim() == 2 and ei.size(0) == 2


# ── Temporal ─────────────────────────────────────────────────────────────────


class TestTemporal:
    def test_basic(self):
        ds = SyntheticTemporalGraphDataset(num_sequences=4, sequence_length=3, seed=0)
        seq = ds[0]
        assert seq.num_snapshots == 3
        # graph_label on each snapshot.
        for g in seq:
            assert g[1].graph_label is not None
