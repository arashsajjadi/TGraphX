"""Sampling loader tests (v0.2.6)."""
from __future__ import annotations

import pytest
import torch

from tgraphx import Graph, NeighborSamplerLoader, SubgraphDataLoader


def _big_graph(N=30, E=80, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(N, 8)
    src = torch.randint(0, N, (E,))
    dst = torch.randint(0, N, (E,))
    ei = torch.stack([src, dst], dim=0).long()
    return Graph(x, ei, edge_weight=torch.rand(E))


class TestSubgraphDataLoader:
    def test_iteration_count(self):
        g = _big_graph()
        loader = SubgraphDataLoader(g, num_nodes=8, num_steps=5, seed=0)
        assert len(loader) == 5
        assert sum(1 for _ in loader) == 5

    def test_subgraph_shape(self):
        g = _big_graph()
        loader = SubgraphDataLoader(g, num_nodes=8, num_steps=3, seed=0)
        for sub in loader:
            assert sub.num_nodes == 8

    def test_deterministic(self):
        g = _big_graph()
        l1 = SubgraphDataLoader(g, num_nodes=8, num_steps=3, seed=42)
        l2 = SubgraphDataLoader(g, num_nodes=8, num_steps=3, seed=42)
        ids1 = [s.metadata["sampling"]["original_node_ids"].tolist() for s in l1]
        ids2 = [s.metadata["sampling"]["original_node_ids"].tolist() for s in l2]
        assert ids1 == ids2

    def test_too_many_nodes_raises(self):
        g = _big_graph(N=5)
        with pytest.raises(ValueError):
            SubgraphDataLoader(g, num_nodes=999, num_steps=1)


class TestNeighborSamplerLoader:
    def test_iteration_count_drop_last_false(self):
        g = _big_graph(N=20)
        loader = NeighborSamplerLoader(g, batch_size=7, fanouts=[3], seed=0)
        # 20 / 7 = 2 full + 1 partial = 3 batches
        assert len(loader) == 3
        assert sum(1 for _ in loader) == 3

    def test_iteration_count_drop_last_true(self):
        g = _big_graph(N=20)
        loader = NeighborSamplerLoader(g, batch_size=7, fanouts=[3],
                                        seed=0, drop_last=True)
        assert len(loader) == 2
        assert sum(1 for _ in loader) == 2

    def test_seed_nodes_in_metadata(self):
        g = _big_graph()
        loader = NeighborSamplerLoader(g, batch_size=4, fanouts=[3], seed=0)
        for sub in loader:
            seeds = sub.metadata["sampling"]["seed_nodes"]
            assert seeds.numel() <= 4

    def test_shuffle_changes_order(self):
        g = _big_graph()
        l1 = NeighborSamplerLoader(g, batch_size=5, fanouts=[3], shuffle=True, seed=1)
        l2 = NeighborSamplerLoader(g, batch_size=5, fanouts=[3], shuffle=True, seed=2)
        seeds1 = [s.metadata["sampling"]["seed_nodes"].tolist() for s in l1]
        seeds2 = [s.metadata["sampling"]["seed_nodes"].tolist() for s in l2]
        assert seeds1 != seeds2  # extremely likely to differ with N=30

    def test_input_nodes_subset(self):
        g = _big_graph(N=20)
        targets = torch.tensor([0, 1, 2, 3])
        loader = NeighborSamplerLoader(
            g, batch_size=2, fanouts=[3], input_nodes=targets, seed=0,
        )
        all_seeds = []
        for sub in loader:
            all_seeds.extend(sub.metadata["sampling"]["seed_nodes"].tolist())
        assert set(all_seeds) <= set(targets.tolist())

    def test_invalid_inputs(self):
        g = _big_graph()
        with pytest.raises(ValueError):
            NeighborSamplerLoader(g, batch_size=0, fanouts=[3])
        with pytest.raises(ValueError):
            NeighborSamplerLoader(g, batch_size=4, fanouts=[])


class TestLoaderIntegrationWithModel:
    def test_forward_through_loader(self):
        from tgraphx.layers import LinearMessagePassing
        g = _big_graph(N=20, E=60)
        layer = LinearMessagePassing(in_shape=(8,), out_shape=(8,)).eval()
        loader = NeighborSamplerLoader(g, batch_size=4, fanouts=[3], seed=0)
        outputs_seen = 0
        for sub in loader:
            with torch.no_grad():
                out = layer(sub.node_features, sub.edge_index)
            assert out.shape == (sub.num_nodes, 8)
            outputs_seen += 1
        assert outputs_seen > 0
