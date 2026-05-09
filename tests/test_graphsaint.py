"""Tests for GraphSAINT samplers and loader."""
from __future__ import annotations

import torch

from tgraphx import Graph
from tgraphx.graphsaint import (
    GraphSAINTNodeSampler,
    GraphSAINTEdgeSampler,
    GraphSAINTRandomWalkSampler,
    GraphSAINTLoader,
    estimate_norm_coefficients,
)


def _toy_graph(n=20, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(n, 4)
    src = torch.arange(n).repeat_interleave(2)
    dst = torch.cat([(torch.arange(n) + 1) % n, (torch.arange(n) + 2) % n])
    ei = torch.stack([src, dst], dim=0)
    ew = torch.rand(ei.size(1))
    ef = torch.randn(ei.size(1), 3)
    y = torch.randint(0, 3, (n,))
    return Graph(node_features=x, edge_index=ei, edge_weight=ew,
                 edge_features=ef, node_labels=y)


def test_node_sampler_deterministic():
    g = _toy_graph()
    s1 = GraphSAINTNodeSampler(g, budget=5, seed=42)
    s2 = GraphSAINTNodeSampler(g, budget=5, seed=42)
    a = list(iter(s1))
    b = list(iter(s2))
    assert len(a) == len(s1) > 0
    for sa, sb in zip(a, b):
        assert sa.num_nodes == sb.num_nodes
        nia = sa.metadata["sampling"]["original_node_ids"]
        nib = sb.metadata["sampling"]["original_node_ids"]
        assert torch.equal(nia, nib)


def test_node_sampler_features_preserved():
    g = _toy_graph()
    s = GraphSAINTNodeSampler(g, budget=5, seed=1, num_steps=2)
    sub = s.sample(0)
    orig_ids = sub.metadata["sampling"]["original_node_ids"]
    # Features should match original at the kept ids.
    assert torch.allclose(sub.node_features, g.node_features[orig_ids])
    # Labels preserved.
    assert torch.equal(sub.node_labels, g.node_labels[orig_ids])


def test_edge_sampler_features_preserved():
    g = _toy_graph()
    s = GraphSAINTEdgeSampler(g, budget=10, seed=7, num_steps=3)
    for sub in s:
        if sub.num_edges == 0:
            continue
        eid = sub.metadata["sampling"]["original_edge_ids"]
        assert torch.allclose(sub.edge_weight, g.edge_weight[eid])
        assert torch.allclose(sub.edge_features, g.edge_features[eid])


def test_random_walk_sampler():
    g = _toy_graph()
    s = GraphSAINTRandomWalkSampler(g, num_roots=3, walk_length=4, seed=0, num_steps=2)
    for sub in s:
        assert sub.num_nodes >= 1
        nid = sub.metadata["sampling"]["original_node_ids"]
        assert (nid >= 0).all() and (nid < g.num_nodes).all()


def test_norm_coefficients_finite():
    g = _toy_graph()
    s = GraphSAINTNodeSampler(g, budget=8, seed=2, num_steps=20)
    node_p, edge_p = estimate_norm_coefficients(s, num_samples=20)
    assert node_p.numel() == g.num_nodes
    assert edge_p.numel() == g.num_edges
    assert (node_p > 0).all()
    assert torch.isfinite(node_p).all() and torch.isfinite(edge_p).all()


def test_loader_attaches_norms():
    g = _toy_graph()
    s = GraphSAINTNodeSampler(g, budget=6, seed=3, num_steps=4)
    loader = GraphSAINTLoader(s, attach_norm=True, num_norm_samples=10)
    for sub in loader:
        saint = sub.metadata.get("graphsaint", {})
        assert "node_norm" in saint
        assert saint["node_norm"].numel() == sub.num_nodes


def test_node_sampler_validation_errors():
    g = _toy_graph()
    try:
        GraphSAINTNodeSampler(g, budget=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for budget=0")
    try:
        GraphSAINTNodeSampler(g, budget=g.num_nodes + 1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for budget > num_nodes")
