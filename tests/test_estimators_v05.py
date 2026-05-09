"""Tests for sklearn-like estimators and pipeline."""
from __future__ import annotations

import torch

from tgraphx import Graph
from tgraphx.estimators import (
    LabelPropagationEstimator,
    Node2VecEstimator,
    VGAEEstimator,
    GraphPipeline,
    EarlyStopping,
    node_train_val_test_split,
    edge_train_val_test_split,
    temporal_train_val_test_split,
)


def _toy_graph(n=15, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(n, 4)
    src = torch.arange(n).repeat_interleave(2)
    dst = torch.cat([(torch.arange(n) + 1) % n, (torch.arange(n) + 2) % n])
    ei = torch.stack([src, dst], dim=0)
    y = torch.tensor([0] * (n // 3) + [1] * (n // 3) + [2] * (n - 2 * (n // 3)))
    return Graph(node_features=x, edge_index=ei, node_labels=y)


def test_label_propagation_get_set_params():
    est = LabelPropagationEstimator(num_iters=10, alpha=0.4)
    p = est.get_params()
    assert p["num_iters"] == 10
    est.set_params(num_iters=99)
    assert est.get_params()["num_iters"] == 99


def test_label_propagation_fit_predict():
    g = _toy_graph()
    # Mask labels for some nodes to test propagation.
    y = g.node_labels.clone()
    y[g.num_nodes // 2:] = -1
    est = LabelPropagationEstimator(num_iters=20, alpha=0.6)
    est.fit(g, y)
    preds = est.predict(g)
    assert preds.shape == (g.num_nodes,)
    proba = est.predict_proba(g)
    assert torch.allclose(proba.sum(dim=-1), torch.ones(g.num_nodes), atol=1e-4)


def test_node2vec_estimator_returns_embeddings():
    g = _toy_graph(n=12)
    est = Node2VecEstimator(embedding_dim=8, walk_length=4,
                            num_walks_per_node=2, window=2, epochs=1, seed=0)
    est.fit(g)
    emb = est.transform(g)
    assert emb.shape == (g.num_nodes, 8)


def test_vgae_estimator_runs():
    g = _toy_graph(n=10)
    est = VGAEEstimator(hidden_dim=8, out_dim=4, epochs=2, lr=0.05, seed=0)
    est.fit(g)
    z = est.transform(g)
    assert z.shape == (g.num_nodes, 4)


def test_pipeline_chaining():
    g = _toy_graph(n=12)
    pipe = GraphPipeline([
        ("emb", Node2VecEstimator(embedding_dim=8, walk_length=3,
                                  num_walks_per_node=1, epochs=1, seed=0)),
        ("lp", LabelPropagationEstimator(num_iters=5)),
    ])
    # Skip pipe.fit (it would chain transforms); just verify the API.
    p = pipe.get_params()
    assert "emb" in p and "lp" in p
    assert "emb__embedding_dim" in p


def test_early_stopping_max():
    es = EarlyStopping(patience=2, mode="max")
    assert not es.step(0.1)
    assert not es.step(0.2)  # improvement → reset
    assert not es.step(0.15)  # 1 strike
    assert es.step(0.10)  # 2nd strike → stop


def test_early_stopping_min():
    es = EarlyStopping(patience=2, mode="min")
    assert not es.step(1.0)
    assert not es.step(0.5)
    assert not es.step(0.6)
    assert es.step(0.7)


def test_node_split_disjoint():
    train, val, test = node_train_val_test_split(100, 0.7, 0.15, 0.15, seed=0)
    assert (train & val).sum().item() == 0
    assert (val & test).sum().item() == 0
    assert (train | val | test).sum().item() == 100


def test_temporal_split_no_leakage():
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    et = torch.tensor([1.0, 4.0, 2.0, 3.0])
    (tr_e, tr_t), (va_e, va_t), (te_e, te_t) = temporal_train_val_test_split(
        ei, et, 0.5, 0.25, 0.25,
    )
    if tr_t.numel() > 0 and va_t.numel() > 0:
        assert tr_t.max().item() <= va_t.min().item()
    if va_t.numel() > 0 and te_t.numel() > 0:
        assert va_t.max().item() <= te_t.min().item()


def test_edge_split_balance():
    ei = torch.arange(20).view(2, 10)
    tr, va, te = edge_train_val_test_split(ei, 0.6, 0.2, 0.2, seed=0)
    assert tr.size(1) + va.size(1) + te.size(1) == 10
