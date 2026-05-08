"""Tests for tgraphx.algorithms.structural — degree utilities (v0.3.2 beta).

Invariants tested
-----------------
- out-degree, in-degree, total-degree all correct on directed graph.
- Isolated nodes have degree 0.
- Self-loops count correctly.
- Empty edge_index returns all-zero degrees.
- dtype and device preserved.
- degree_features shape and log_scale behavior.
"""
from __future__ import annotations

import pytest
import torch

from tgraphx.algorithms import degree, degree_features


class TestDegree:
    def test_out_degree_chain(self):
        # 0 → 1 → 2 → 3
        ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        d = degree(ei, num_nodes=4, mode="out")
        assert d.tolist() == [1, 1, 1, 0]

    def test_in_degree_chain(self):
        ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        d = degree(ei, num_nodes=4, mode="in")
        assert d.tolist() == [0, 1, 1, 1]

    def test_total_degree_chain(self):
        ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        d = degree(ei, num_nodes=4, mode="both")
        assert d.tolist() == [1, 2, 2, 1]

    def test_isolated_nodes_zero(self):
        ei = torch.zeros((2, 0), dtype=torch.long)
        d = degree(ei, num_nodes=5, mode="out")
        assert d.tolist() == [0, 0, 0, 0, 0]

    def test_empty_edge_index_all_zero(self):
        ei = torch.zeros((2, 0), dtype=torch.long)
        d = degree(ei, num_nodes=3)
        assert d.shape == (3,)
        assert d.sum().item() == 0

    def test_self_loop_counts_out_and_in(self):
        # Self-loop at node 0.
        ei = torch.tensor([[0], [0]], dtype=torch.long)
        assert degree(ei, num_nodes=2, mode="out").tolist() == [1, 0]
        assert degree(ei, num_nodes=2, mode="in").tolist() == [1, 0]
        assert degree(ei, num_nodes=2, mode="both").tolist() == [2, 0]

    def test_dtype_override(self):
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        d = degree(ei, num_nodes=3, dtype=torch.float32)
        assert d.dtype == torch.float32

    def test_inferred_num_nodes(self):
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        d = degree(ei)  # infers num_nodes=3
        assert d.shape == (3,)

    def test_out_degree_star(self):
        # Hub at node 0 connected to 1,2,3.
        ei = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)
        d = degree(ei, num_nodes=4, mode="out")
        assert d.tolist() == [3, 0, 0, 0]

    def test_invalid_mode(self):
        ei = torch.tensor([[0], [1]], dtype=torch.long)
        with pytest.raises(ValueError, match="mode"):
            degree(ei, num_nodes=2, mode="all")

    def test_invalid_edge_index_shape(self):
        with pytest.raises(ValueError, match="\\[2, E\\]"):
            degree(torch.zeros(3, 4, dtype=torch.long), num_nodes=4)

    def test_edge_id_out_of_range(self):
        ei = torch.tensor([[0, 5], [1, 2]], dtype=torch.long)
        with pytest.raises(ValueError):
            degree(ei, num_nodes=4, mode="out")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_preserved(self):
        ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long).cuda()
        d = degree(ei, num_nodes=4, mode="out")
        assert d.device.type == "cuda"
        assert d.tolist() == [1, 1, 1, 0]


class TestDegreeFeatures:
    def test_shape(self):
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        feats = degree_features(ei, num_nodes=3)
        assert feats.shape == (3, 3)

    def test_columns(self):
        # 0 → 1 → 2
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        feats = degree_features(ei, num_nodes=3)
        # out_deg: [1,1,0]; in_deg: [0,1,1]; total: [1,2,1]
        assert feats[:, 0].tolist() == [1, 1, 0]
        assert feats[:, 1].tolist() == [0, 1, 1]
        assert feats[:, 2].tolist() == [1, 2, 1]

    def test_log_scale(self):
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        feats = degree_features(ei, num_nodes=3, log_scale=True)
        assert feats.dtype == torch.float32
        # All values should be >= 0 (log1p of non-negative integers).
        assert (feats >= 0).all()

    def test_empty_graph(self):
        feats = degree_features(torch.zeros((2, 0), dtype=torch.long), num_nodes=4)
        assert feats.shape == (4, 3)
        assert feats.sum().item() == 0
