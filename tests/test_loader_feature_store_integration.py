"""Tests for FeatureStore integration with NeighborLoader."""
from __future__ import annotations

import torch

from tgraphx import (
    Graph, NeighborLoader, InMemoryFeatureStore, MemmapFeatureStore,
    fetch_features_for_subgraph,
)


def _toy_graph(n=20, seed=0):
    torch.manual_seed(seed)
    src = torch.arange(n).repeat_interleave(2)
    dst = torch.cat([(torch.arange(n) + 1) % n, (torch.arange(n) + 2) % n])
    ei = torch.stack([src, dst], dim=0)
    x_placeholder = torch.zeros(n, 4)
    return Graph(node_features=x_placeholder, edge_index=ei)


def test_neighbor_loader_with_in_memory_store():
    g = _toy_graph(n=12)
    real_x = torch.randn(g.num_nodes, 8)
    store = InMemoryFeatureStore()
    store.put("x", real_x)
    loader = NeighborLoader(
        g, fanouts=[2, 2], batch_size=4, shuffle=False, seed=0,
        feature_store=store, feature_name="x",
    )
    sub, seeds = next(iter(loader))
    # Sub features should be feature-store rows for sampled IDs.
    assert sub.node_features.shape[1:] == (8,)
    sampled_ids = sub.metadata["sampling"]["original_node_ids"]
    assert torch.allclose(sub.node_features, real_x[sampled_ids])


def test_memmap_equals_in_memory(tmp_path):
    g = _toy_graph(n=10)
    real_x = torch.randn(g.num_nodes, 6)
    in_mem = InMemoryFeatureStore()
    in_mem.put("x", real_x)
    mm = MemmapFeatureStore(root=str(tmp_path))
    mm.put("x", real_x)
    ids = torch.tensor([0, 3, 5])
    a = in_mem.get("x", ids)
    b = mm.get("x", ids)
    assert torch.allclose(a, b)


def test_fetch_features_for_subgraph_requires_metadata():
    g = _toy_graph(n=5)
    store = InMemoryFeatureStore()
    store.put("x", torch.randn(5, 4))
    # Plain Graph without sampling metadata -> error.
    try:
        fetch_features_for_subgraph(g, store, name="x")
    except ValueError as e:
        assert "sampling" in str(e)
    else:
        raise AssertionError("expected ValueError")
