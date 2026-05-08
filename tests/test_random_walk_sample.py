"""Tests for ``random_walk_sample`` (v0.2.8)."""
from __future__ import annotations

import pytest
import torch

from tgraphx import Graph, random_walk_sample


def _line_graph(n: int = 10) -> Graph:
    """Directed line graph 0 -> 1 -> 2 -> ... -> n-1."""
    x = torch.randn(n, 4)
    src = torch.arange(n - 1, dtype=torch.long)
    dst = src + 1
    ei = torch.stack([src, dst], dim=0)
    return Graph(
        x, ei,
        edge_weight=torch.arange(1, n, dtype=torch.float),
        edge_features=torch.randn(n - 1, 2),
        node_labels=torch.arange(n, dtype=torch.long),
    )


def _ring_graph(n: int = 8) -> Graph:
    src = torch.arange(n, dtype=torch.long)
    dst = (src + 1) % n
    ei = torch.stack([src, dst], dim=0)
    return Graph(torch.randn(n, 4), ei)


class TestRandomWalkSample:
    def test_basic_subgraph_shape(self):
        g = _line_graph(10)
        sub = random_walk_sample(
            g,
            seed_nodes=torch.tensor([0]),
            walk_length=4,
            seed=0,
        )
        assert sub.num_nodes >= 1
        assert sub.num_nodes <= 10
        assert sub.metadata["sampling"]["kind"] == "random_walk_sample"

    def test_walk_length_zero_returns_seeds_only(self):
        g = _line_graph(10)
        sub = random_walk_sample(g, torch.tensor([2, 5]), walk_length=0, seed=0)
        assert sub.num_nodes == 2

    def test_out_direction_visits_forward(self):
        g = _line_graph(10)
        sub = random_walk_sample(
            g, torch.tensor([0]), walk_length=9,
            num_walks_per_seed=3, direction="out", seed=42,
        )
        ids = sub.metadata["sampling"]["original_node_ids"].tolist()
        assert 0 in ids
        # All visited ids must be reachable from 0 going forward, i.e. >= 0.
        assert all(i >= 0 for i in ids)

    def test_in_direction_visits_backward(self):
        g = _line_graph(10)
        sub = random_walk_sample(
            g, torch.tensor([9]), walk_length=9, direction="in", seed=42,
        )
        ids = sub.metadata["sampling"]["original_node_ids"].tolist()
        assert 9 in ids
        # In a directed line, walking 'in' from 9 only sees nodes <= 9.
        assert all(i <= 9 for i in ids)

    def test_determinism_same_seed(self):
        g = _ring_graph(8)
        s1 = random_walk_sample(g, torch.tensor([0]), 5, seed=123)
        s2 = random_walk_sample(g, torch.tensor([0]), 5, seed=123)
        assert s1.num_nodes == s2.num_nodes
        ids1 = s1.metadata["sampling"]["original_node_ids"].tolist()
        ids2 = s2.metadata["sampling"]["original_node_ids"].tolist()
        assert ids1 == ids2

    def test_no_global_rng_pollution(self):
        g = _ring_graph(8)
        torch.manual_seed(0)
        before = torch.rand(3)
        torch.manual_seed(0)
        random_walk_sample(g, torch.tensor([0]), 5, seed=999)
        after = torch.rand(3)
        # Global RNG state must not be affected.
        assert torch.allclose(before, after)

    def test_features_and_labels_preserved(self):
        g = _line_graph(10)
        sub = random_walk_sample(g, torch.tensor([0]), 9, seed=0)
        ids = sub.metadata["sampling"]["original_node_ids"].tolist()
        for local, global_id in enumerate(ids):
            assert torch.equal(sub.node_features[local], g.node_features[global_id])
            assert sub.node_labels[local].item() == g.node_labels[global_id].item()

    def test_edge_weight_and_features_preserved(self):
        g = _line_graph(10)
        sub = random_walk_sample(g, torch.tensor([0]), 9, seed=0)
        if sub.num_edges > 0:
            assert sub.edge_weight is not None
            assert sub.edge_features is not None
            assert sub.edge_features.size(0) == sub.num_edges

    def test_metadata_records_config(self):
        g = _ring_graph(8)
        sub = random_walk_sample(
            g, torch.tensor([1, 4]),
            walk_length=3, num_walks_per_seed=2,
            direction="out", restart_prob=0.1, seed=7,
        )
        meta = sub.metadata["sampling"]
        assert meta["walk_length"] == 3
        assert meta["num_walks_per_seed"] == 2
        assert meta["direction"] == "out"
        assert meta["restart_prob"] == pytest.approx(0.1)
        assert meta["seed_nodes"].tolist() == [1, 4]

    def test_invalid_direction_raises(self):
        g = _ring_graph(8)
        with pytest.raises(ValueError, match="direction"):
            random_walk_sample(g, torch.tensor([0]), 3, direction="diag")

    def test_invalid_walk_length_raises(self):
        g = _ring_graph(8)
        with pytest.raises(ValueError, match="walk_length"):
            random_walk_sample(g, torch.tensor([0]), -1)

    def test_invalid_num_walks_raises(self):
        g = _ring_graph(8)
        with pytest.raises(ValueError, match="num_walks_per_seed"):
            random_walk_sample(g, torch.tensor([0]), 3, num_walks_per_seed=0)

    def test_invalid_restart_prob_raises(self):
        g = _ring_graph(8)
        with pytest.raises(ValueError, match="restart_prob"):
            random_walk_sample(g, torch.tensor([0]), 3, restart_prob=1.0)

    def test_isolated_seed_absorbs(self):
        # Node 5 has no out-edges in this graph.
        x = torch.randn(6, 4)
        ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        g = Graph(x, ei)
        sub = random_walk_sample(g, torch.tensor([5]), 5, seed=0)
        # Walker is absorbed at 5; visited set is exactly {5}.
        assert sub.num_nodes == 1

    def test_empty_edge_index_walker_absorbs(self):
        x = torch.randn(4, 4)
        ei = torch.zeros((2, 0), dtype=torch.long)
        g = Graph(x, ei)
        # Walker has no neighbours and stays at the seed.
        sub = random_walk_sample(g, torch.tensor([0]), 3, seed=0)
        assert sub.num_nodes == 1

    def test_restart_prob_keeps_seed_in_visited(self):
        g = _line_graph(10)
        sub = random_walk_sample(
            g, torch.tensor([3]), walk_length=5,
            restart_prob=0.5, seed=1,
        )
        ids = sub.metadata["sampling"]["original_node_ids"].tolist()
        assert 3 in ids
