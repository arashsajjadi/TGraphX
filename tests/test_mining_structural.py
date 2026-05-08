"""Tests for tgraphx.mining.structural — graph density, degree stats, features."""
from __future__ import annotations

import json
import math

import pytest
import torch

from tgraphx.mining import (
    add_structural_features,
    degree_statistics,
    graph_density,
    graph_summary,
    structural_features,
)


class TestGraphDensity:
    def test_complete_directed(self):
        ei = torch.tensor([[0,0,1,1,2,2],[1,2,0,2,0,1]], dtype=torch.long)
        assert abs(graph_density(ei, 3, directed=True) - 1.0) < 1e-9

    def test_empty_graph(self):
        assert graph_density(torch.zeros((2,0),dtype=torch.long), 5) == 0.0

    def test_single_node(self):
        assert graph_density(torch.zeros((2,0),dtype=torch.long), 1) == 0.0

    def test_chain_directed(self):
        ei = torch.tensor([[0,1,2],[1,2,3]], dtype=torch.long)
        d = graph_density(ei, 4, directed=True)
        assert abs(d - 3.0 / 12.0) < 1e-9  # 3 / (4*3)

    def test_undirected(self):
        # Triangle (3 unique undirected edges) on 3 nodes: max=3.
        ei = torch.tensor([[0,1,2,1,2,0],[1,2,0,0,1,2]], dtype=torch.long)
        d = graph_density(ei, 3, directed=False)
        assert abs(d - 1.0) < 1e-9

    def test_self_loops_excluded(self):
        ei = torch.tensor([[0,1,0],[1,2,0]], dtype=torch.long)  # edge 0→0 is self-loop
        d_excl = graph_density(ei, 3, directed=True, exclude_self_loops=True)
        d_incl = graph_density(ei, 3, directed=True, exclude_self_loops=False)
        assert d_excl < d_incl or abs(d_excl - d_incl) < 1e-9

    def test_density_in_zero_one(self):
        ei = torch.tensor([[0,1],[1,2]], dtype=torch.long)
        d = graph_density(ei, 5)
        assert 0.0 <= d <= 1.0


class TestDegreeStatistics:
    def test_chain(self):
        ei = torch.tensor([[0,1,2],[1,2,3]], dtype=torch.long)
        stats = degree_statistics(ei, 4)
        assert stats["min_out_degree"] == 0  # node 3 has no outgoing
        assert stats["max_out_degree"] == 1
        assert stats["isolated_node_count"] == 0  # all have at least one edge
        assert "density" in stats

    def test_json_serializable(self):
        ei = torch.tensor([[0,1],[1,2]], dtype=torch.long)
        stats = degree_statistics(ei, 3)
        assert json.dumps(stats) is not None

    def test_empty_graph(self):
        stats = degree_statistics(torch.zeros((2,0),dtype=torch.long), 4)
        assert stats["isolated_node_count"] == 4
        assert stats["mean_total_degree"] == 0.0

    def test_zero_nodes(self):
        stats = degree_statistics(torch.zeros((2,0),dtype=torch.long), 0)
        assert stats["isolated_node_count"] == 0


class TestGraphSummary:
    def test_json_serializable(self):
        ei = torch.tensor([[0,1],[1,2]], dtype=torch.long)
        s = graph_summary(ei, 3)
        assert json.dumps(s) is not None

    def test_includes_components(self):
        ei = torch.tensor([[0,1],[1,2]], dtype=torch.long)
        s = graph_summary(ei, 5)
        assert "num_connected_components" in s
        assert s["num_connected_components"] >= 1

    def test_warnings_list(self):
        ei = torch.tensor([[0,1],[1,2]], dtype=torch.long)
        s = graph_summary(ei, 3)
        assert isinstance(s["warnings"], list)


class TestStructuralFeatures:
    def test_shape(self):
        ei = torch.tensor([[0,1,2],[1,2,3]], dtype=torch.long)
        sf = structural_features(ei, 4)
        assert sf.shape == (4, 4)  # degree, in_degree, out_degree, log_degree
        assert sf.dtype == torch.float32

    def test_invalid_feature(self):
        ei = torch.tensor([[0],[1]], dtype=torch.long)
        with pytest.raises(ValueError, match="Unknown feature"):
            structural_features(ei, 2, features=("bad_feature",))

    def test_log_degree_non_negative(self):
        ei = torch.tensor([[0,1],[1,2]], dtype=torch.long)
        sf = structural_features(ei, 3, features=("log_degree",))
        assert (sf >= 0).all()

    def test_empty_graph_zeros(self):
        sf = structural_features(torch.zeros((2,0),dtype=torch.long), 5)
        assert sf.shape[0] == 5
        assert (sf[:, 0] == 0).all()  # degree column = 0


class TestAddStructuralFeatures:
    def test_vector_features_concatenated(self):
        from tgraphx import Graph
        ei = torch.tensor([[0,1],[1,2]], dtype=torch.long)
        g = Graph(torch.randn(3, 8), ei)
        g2 = add_structural_features(g, features=("log_degree",))
        assert g2.node_features.shape == (3, 9)  # 8 + 1

    def test_spatial_features_stored_in_metadata(self):
        from tgraphx import Graph
        ei = torch.tensor([[0,1],[1,2]], dtype=torch.long)
        g = Graph(torch.randn(3, 4, 8, 8), ei)  # [N,C,H,W]
        g2 = add_structural_features(g, features=("degree",), key="sf")
        # Original spatial features unchanged.
        assert g2.node_features.shape == (3, 4, 8, 8)
        assert "sf" in g2.metadata
        assert g2.metadata["sf"].shape == (3, 1)
