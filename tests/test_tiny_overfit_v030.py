"""Tiny-overfit trainability tests for v0.3.0 features."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tgraphx import APPNP, GATv2Conv, GCNConv, GraphBatch, build_model, set_seed
from tgraphx.datasets import (
    SyntheticGraphRegressionDataset,
    SyntheticNodeClassificationDataset,
    SyntheticPatchGraphDataset,
)


def _train_step(model, batch, optimizer, loss_fn):
    optimizer.zero_grad()
    logits = model(batch.node_features, batch.edge_index, batch=batch.batch)
    loss = loss_fn(logits, batch.graph_labels.long())
    loss.backward()
    optimizer.step()
    return float(loss.item())


# ── Model-zoo layers can be assembled into a working classifier ──────────────


class _GCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.act = nn.ReLU(inplace=False)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch=None):
        h = self.act(self.gcn1(x, edge_index))
        h = self.act(self.gcn2(h, edge_index))
        from tgraphx.layers.pooling import global_mean_pool
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        return self.head(global_mean_pool(h, batch))


class _GATv2Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.gat1 = GATv2Conv(in_dim, hidden_dim, num_heads=2)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, num_heads=2)
        self.act = nn.ELU(inplace=False)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch=None):
        h = self.act(self.gat1(x, edge_index))
        h = self.act(self.gat2(h, edge_index))
        from tgraphx.layers.pooling import global_mean_pool
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        return self.head(global_mean_pool(h, batch))


class TestTinyOverfit:
    def test_synthetic_patch_classification_loss_decreases(self):
        set_seed(0)
        ds = SyntheticPatchGraphDataset(num_graphs=8, image_size=16, patch_size=4, seed=0)
        batch = GraphBatch(list(ds))
        in_shape = tuple(batch.node_features.shape[1:])
        model = build_model(
            task="graph_classification", layer="conv",
            in_shape=in_shape, hidden_shape=(8, 4, 4),
            num_layers=2, num_classes=ds.metadata.num_classes,
            pooling="mean",
        )
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        loss_fn = torch.nn.functional.cross_entropy
        first = _train_step(model, batch, opt, loss_fn)
        for _ in range(7):
            last = _train_step(model, batch, opt, loss_fn)
        assert last < first
        # All trainable params received finite gradients.
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()

    def test_node_classification_loss_decreases(self):
        set_seed(0)
        ds = SyntheticNodeClassificationDataset(num_nodes=40, num_classes=3,
                                                feature_dim=8, seed=0)
        g = ds[0]
        masks = g.metadata["masks"]
        model = build_model(
            task="node_classification", layer="linear",
            in_shape=(8,), hidden_shape=(16,), num_layers=2, num_classes=3,
        )
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        losses = []
        for _ in range(8):
            opt.zero_grad()
            out = model(g.node_features, g.edge_index)
            loss = torch.nn.functional.cross_entropy(
                out[masks["train_mask"]], g.node_labels[masks["train_mask"]].long()
            )
            loss.backward()
            opt.step()
            losses.append(loss.detach().item())
        assert losses[-1] < losses[0]

    def test_graph_regression_loss_decreases(self):
        set_seed(0)
        ds = SyntheticGraphRegressionDataset(num_graphs=8, image_size=16,
                                             patch_size=4, seed=0)
        batch = GraphBatch(list(ds))
        in_shape = tuple(batch.node_features.shape[1:])
        model = build_model(
            task="graph_regression", layer="conv",
            in_shape=in_shape, hidden_shape=(8, 4, 4),
            num_layers=2, out_dim=1, pooling="mean",
        )
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        losses = []
        for _ in range(8):
            opt.zero_grad()
            out = model(batch.node_features, batch.edge_index, batch=batch.batch)
            loss = torch.nn.functional.mse_loss(
                out.squeeze(-1), batch.graph_labels.squeeze(-1).float()
            )
            loss.backward()
            opt.step()
            losses.append(loss.detach().item())
        assert losses[-1] < losses[0]

    def test_gcn_zoo_classifier_overfits(self):
        set_seed(0)
        ds = SyntheticNodeClassificationDataset(num_nodes=20, num_classes=3,
                                                feature_dim=4, seed=0)
        g = ds[0]
        # Treat each graph's nodes as a single graph_classification batch (toy).
        batch = torch.zeros(g.num_nodes, dtype=torch.long)
        labels = g.node_labels.long()
        model = _GCNClassifier(4, 16, 3)

        # Per-node classification: skip pooling by overriding forward usage.
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        # Drive node-level loss against per-node logits (skip pooling).
        losses = []
        for _ in range(8):
            opt.zero_grad()
            h = torch.relu(model.gcn1(g.node_features, g.edge_index))
            h = torch.relu(model.gcn2(h, g.edge_index))
            logits = model.head(h)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            opt.step()
            losses.append(loss.detach().item())
        assert losses[-1] < losses[0]

    def test_gatv2_classifier_finite_gradients(self):
        set_seed(0)
        x = torch.randn(8, 4, requires_grad=False)
        ei = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                           [1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long)
        layer = GATv2Conv(4, 8, num_heads=2)
        out = layer(x, ei)
        out.sum().backward()
        for p in layer.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()


# ── Gradient health on deeper stacks ─────────────────────────────────────────


class TestGradientHealth:
    def test_4layer_gcn_finite_grads(self):
        set_seed(0)
        N, D = 8, 4
        x = torch.randn(N, D)
        ei = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                           [1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long)
        layers = nn.ModuleList([GCNConv(D, D) for _ in range(4)])
        h = x
        for ly in layers:
            h = torch.relu(ly(h, ei))
        h.sum().backward()
        norms = []
        for ly in layers:
            g = ly.lin.weight.grad
            assert torch.isfinite(g).all()
            norms.append(g.norm().item())
        # No layer should have completely zero gradients.
        assert all(n > 0 for n in norms)
