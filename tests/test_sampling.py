"""Sampling API tests (v0.2.6)."""
from __future__ import annotations

import pytest
import torch

from tgraphx import (
    Graph,
    induced_subgraph,
    edge_subgraph,
    k_hop_subgraph,
    neighbor_sample,
    sample_edges,
    sample_nodes,
)


def _g(N=10, E=20, seed=0, with_extras=True):
    torch.manual_seed(seed)
    x = torch.randn(N, 4)
    src = torch.randint(0, N, (E,))
    dst = torch.randint(0, N, (E,))
    ei = torch.stack([src, dst], dim=0).long()
    if with_extras:
        return Graph(
            x, ei,
            edge_weight=torch.rand(E),
            edge_features=torch.randn(E, 3),
            node_labels=torch.randint(0, 5, (N,)),
            edge_labels=torch.randint(0, 5, (E,)),
            graph_label=torch.tensor(0),
        )
    return Graph(x, ei)


# ── induced_subgraph ──────────────────────────────────────────────────────────

class TestInducedSubgraph:
    def test_basic(self):
        g = _g()
        sub = induced_subgraph(g, torch.tensor([0, 1, 2, 5]))
        assert sub.num_nodes == 4

    def test_features_preserved(self):
        g = _g()
        sub = induced_subgraph(g, torch.tensor([0, 3, 7]))
        # The original node 0 must be at local id 0 after relabeling.
        original_ids = sub.metadata["sampling"]["original_node_ids"]
        assert original_ids.tolist() == [0, 3, 7]
        # Feature vectors must match originals.
        for local, global_id in enumerate(original_ids.tolist()):
            assert torch.equal(sub.node_features[local], g.node_features[global_id])

    def test_edge_attrs_preserved(self):
        g = _g()
        sub = induced_subgraph(g, torch.tensor([0, 1, 2, 5, 7, 9]))
        if sub.num_edges > 0:
            assert sub.edge_weight is not None
            assert sub.edge_features is not None
            assert sub.edge_features.size(0) == sub.num_edges

    def test_labels_preserved(self):
        g = _g()
        sub = induced_subgraph(g, torch.tensor([0, 4, 9]))
        assert sub.node_labels is not None
        assert sub.node_labels.shape == (3,)

    def test_metadata_records_ids(self):
        g = _g()
        sub = induced_subgraph(g, torch.tensor([2, 5, 7]))
        meta = sub.metadata["sampling"]
        assert meta["kind"] == "induced_subgraph"
        assert meta["original_node_ids"].tolist() == [2, 5, 7]

    def test_no_relabel(self):
        g = _g()
        sub = induced_subgraph(g, torch.tensor([0, 1, 2, 9]), relabel_nodes=False)
        # num_nodes stays at original.
        assert sub.num_nodes == g.num_nodes

    def test_invalid_inputs(self):
        g = _g()
        with pytest.raises(ValueError):
            induced_subgraph(g, torch.tensor([0, 0, 1]))  # duplicates
        with pytest.raises(ValueError):
            induced_subgraph(g, torch.tensor([0, 999]))  # out of range
        with pytest.raises(ValueError):
            induced_subgraph(g, torch.empty(0, dtype=torch.long))  # empty


# ── edge_subgraph ─────────────────────────────────────────────────────────────

class TestEdgeSubgraph:
    def test_basic(self):
        g = _g()
        sub = edge_subgraph(g, torch.tensor([0, 1, 5]))
        assert sub.num_edges == 3 or sub.num_edges <= 3  # may include dups

    def test_metadata(self):
        g = _g()
        sub = edge_subgraph(g, torch.tensor([2, 4]))
        assert sub.metadata["sampling"]["kind"] == "edge_subgraph"
        assert sub.metadata["sampling"]["original_edge_ids"].tolist() == [2, 4]


# ── k_hop_subgraph ────────────────────────────────────────────────────────────

class TestKHopSubgraph:
    def test_zero_hop_keeps_seeds(self):
        g = _g()
        sub = k_hop_subgraph(g, torch.tensor([3, 5]), num_hops=0)
        assert sub.num_nodes == 2

    def test_one_hop_grows(self):
        g = _g(N=8, E=20, seed=1)
        sub = k_hop_subgraph(g, torch.tensor([0]), num_hops=1)
        # Should include node 0 and at least one neighbour (graph is dense).
        assert sub.num_nodes >= 1

    def test_direction_in_out_differ(self):
        # Construct a directed line: 0 -> 1 -> 2 -> 3
        x = torch.randn(4, 4)
        ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        g = Graph(x, ei)
        sub_in = k_hop_subgraph(g, torch.tensor([3]), num_hops=2, direction="in")
        sub_out = k_hop_subgraph(g, torch.tensor([0]), num_hops=2, direction="out")
        # in from 3 picks {3, 2, 1}; out from 0 picks {0, 1, 2}.
        in_ids = set(sub_in.metadata["sampling"]["original_node_ids"].tolist())
        out_ids = set(sub_out.metadata["sampling"]["original_node_ids"].tolist())
        assert in_ids == {1, 2, 3}
        assert out_ids == {0, 1, 2}

    def test_invalid_direction(self):
        g = _g()
        with pytest.raises(ValueError, match="direction"):
            k_hop_subgraph(g, torch.tensor([0]), num_hops=1, direction="bad")

    def test_negative_hops(self):
        g = _g()
        with pytest.raises(ValueError, match="num_hops"):
            k_hop_subgraph(g, torch.tensor([0]), num_hops=-1)


# ── sample_nodes / sample_edges ──────────────────────────────────────────────

class TestUniformSampling:
    def test_sample_nodes_count(self):
        g = _g()
        sub = sample_nodes(g, num_nodes=5, seed=0)
        assert sub.num_nodes == 5

    def test_sample_nodes_deterministic(self):
        g = _g()
        s1 = sample_nodes(g, num_nodes=5, seed=42)
        s2 = sample_nodes(g, num_nodes=5, seed=42)
        ids1 = s1.metadata["sampling"]["original_node_ids"]
        ids2 = s2.metadata["sampling"]["original_node_ids"]
        assert torch.equal(ids1, ids2)

    def test_sample_nodes_different_seeds(self):
        g = _g()
        s1 = sample_nodes(g, num_nodes=8, seed=1)
        s2 = sample_nodes(g, num_nodes=8, seed=2)
        ids1 = set(s1.metadata["sampling"]["original_node_ids"].tolist())
        ids2 = set(s2.metadata["sampling"]["original_node_ids"].tolist())
        assert ids1 != ids2  # extremely unlikely to coincide

    def test_sample_nodes_too_many(self):
        g = _g(N=5)
        with pytest.raises(ValueError, match="num_nodes"):
            sample_nodes(g, num_nodes=999)

    def test_sample_edges(self):
        g = _g()
        sub = sample_edges(g, num_edges=6, seed=0)
        assert sub.metadata["sampling"]["kind"] == "edge_subgraph"


# ── neighbor_sample ──────────────────────────────────────────────────────────

class TestNeighborSample:
    def test_basic(self):
        g = _g(N=20, E=80)
        sub = neighbor_sample(g, torch.tensor([0, 1]), fanouts=[5, 3], seed=0)
        assert sub.num_nodes >= 2
        assert sub.metadata["sampling"]["kind"] == "neighbor_sample"
        assert sub.metadata["sampling"]["fanouts"] == [5, 3]

    def test_deterministic(self):
        g = _g(N=20, E=80)
        s1 = neighbor_sample(g, torch.tensor([0, 1]), fanouts=[5, 3], seed=7)
        s2 = neighbor_sample(g, torch.tensor([0, 1]), fanouts=[5, 3], seed=7)
        ids1 = s1.metadata["sampling"]["original_node_ids"]
        ids2 = s2.metadata["sampling"]["original_node_ids"]
        assert torch.equal(ids1, ids2)

    def test_fanout_minus_one_keeps_all(self):
        # Build a star: node 0 has 5 in-neighbours.
        x = torch.randn(6, 4)
        ei = torch.tensor([[1, 2, 3, 4, 5], [0, 0, 0, 0, 0]], dtype=torch.long)
        g = Graph(x, ei)
        sub = neighbor_sample(g, torch.tensor([0]), fanouts=[-1], direction="in")
        # All 5 in-neighbours kept.
        assert sub.num_nodes == 6

    def test_fanout_caps(self):
        x = torch.randn(6, 4)
        ei = torch.tensor([[1, 2, 3, 4, 5], [0, 0, 0, 0, 0]], dtype=torch.long)
        g = Graph(x, ei)
        sub = neighbor_sample(g, torch.tensor([0]), fanouts=[2], direction="in", seed=0)
        assert sub.num_nodes == 3  # seed + 2 sampled neighbours

    def test_invalid_direction(self):
        g = _g()
        with pytest.raises(ValueError, match="direction"):
            neighbor_sample(g, torch.tensor([0]), fanouts=[3], direction="both")

    def test_invalid_fanouts(self):
        g = _g()
        with pytest.raises(ValueError):
            neighbor_sample(g, torch.tensor([0]), fanouts=[])
        with pytest.raises(ValueError):
            neighbor_sample(g, torch.tensor([0]), fanouts=[0])


# ── Device preservation ──────────────────────────────────────────────────────

class TestDevice:
    def test_induced_keeps_device(self):
        g = _g()
        sub = induced_subgraph(g, torch.tensor([0, 1]))
        assert sub.node_features.device == g.node_features.device

    def test_neighbor_sample_keeps_device(self):
        g = _g()
        sub = neighbor_sample(g, torch.tensor([0]), fanouts=[3])
        assert sub.node_features.device == g.node_features.device
