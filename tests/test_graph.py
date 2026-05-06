"""Tests for Graph and GraphBatch data structures.

Covers:
- Valid construction paths
- Input validation errors (M-07 fixes)
- GraphBatch node-count offsetting and batch vector correctness
- GraphBatch rejection of incompatible spatial sizes (H-09 fix)
- Device mobility via .to()
"""

import pytest
import torch

from tgraphx import Graph, GraphBatch


# ──────────────────────────────────────────────────────────────────── #
# Helpers                                                               #
# ──────────────────────────────────────────────────────────────────── #

def _x(N=4, C=3, H=8, W=8, device="cpu"):
    return torch.randn(N, C, H, W, device=device)


def _ei(N=4, device="cpu"):
    """Directed cycle: 0→1→…→(N-1)→0."""
    src = torch.arange(N, device=device)
    return torch.stack([src, (src + 1) % N])


# ──────────────────────────────────────────────────────────────────── #
# Graph — valid construction                                             #
# ──────────────────────────────────────────────────────────────────── #

class TestGraphConstruction:
    def test_spatial_features(self):
        g = Graph(_x(), _ei())
        assert g.node_features.shape == (4, 3, 8, 8)
        assert g.edge_index.shape == (2, 4)
        assert g.edge_features is None

    def test_vector_features(self):
        x = torch.randn(6, 32)
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        g = Graph(x, ei)
        assert g.node_features.shape == (6, 32)

    def test_none_edge_index(self):
        g = Graph(_x(), None)
        assert g.edge_index is None

    def test_edge_features_present(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        ef = torch.randn(2, 5)
        g = Graph(_x(), ei, edge_features=ef)
        assert g.edge_features.shape == (2, 5)

    def test_to_cpu_returns_self(self):
        g = Graph(_x(), _ei())
        g2 = g.to("cpu")
        assert g2 is g
        assert g.node_features.device.type == "cpu"

    def test_to_cpu_with_none_edge_index(self):
        """Graph.to() must handle edge_index=None without AttributeError."""
        g = Graph(_x(), None)
        g.to("cpu")  # must not crash


# ──────────────────────────────────────────────────────────────────── #
# Graph — input validation (M-07)                                       #
# ──────────────────────────────────────────────────────────────────── #

class TestGraphValidation:
    def test_non_tensor_node_features(self):
        with pytest.raises(TypeError, match="node_features must be a torch.Tensor"):
            Graph([[1.0, 2.0]], torch.zeros(2, 2, dtype=torch.long))

    def test_edge_index_wrong_shape_3_rows(self):
        with pytest.raises(ValueError, match=r"\[2, E\]"):
            Graph(_x(), torch.zeros(3, 4, dtype=torch.long))

    def test_edge_index_1d_tensor(self):
        with pytest.raises(ValueError, match=r"\[2, E\]"):
            Graph(_x(), torch.tensor([0, 1, 2, 3], dtype=torch.long))

    def test_edge_index_float_dtype(self):
        with pytest.raises(TypeError, match="torch.long"):
            Graph(_x(), torch.zeros(2, 2, dtype=torch.float32))

    def test_negative_edge_index(self):
        ei = torch.tensor([[0, -1], [1, 0]], dtype=torch.long)
        with pytest.raises(ValueError, match="out-of-range"):
            Graph(_x(), ei)

    def test_out_of_range_edge_index(self):
        ei = torch.tensor([[0, 99], [1, 0]], dtype=torch.long)
        with pytest.raises(ValueError, match="out-of-range"):
            Graph(_x(), ei)

    def test_edge_features_length_mismatch(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # 2 edges
        ef = torch.randn(5, 4)  # 5 ≠ 2
        with pytest.raises(ValueError, match="edge_features"):
            Graph(_x(), ei, edge_features=ef)

    def test_edge_features_without_edge_index(self):
        with pytest.raises(ValueError, match="edge_index is None"):
            Graph(_x(), None, edge_features=torch.randn(2, 4))

    def test_non_tensor_edge_features(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        with pytest.raises(TypeError, match="edge_features must be a torch.Tensor"):
            Graph(_x(), ei, edge_features=[[0.1, 0.2], [0.3, 0.4]])

    @pytest.mark.cuda
    def test_device_mismatch_edge_index_raises(self):
        """edge_index on CUDA with node_features on CPU must be rejected."""
        x_cpu = torch.randn(4, 3, 8, 8)
        ei_gpu = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).cuda()
        with pytest.raises(ValueError, match="device"):
            Graph(x_cpu, ei_gpu)


# ──────────────────────────────────────────────────────────────────── #
# GraphBatch                                                            #
# ──────────────────────────────────────────────────────────────────── #

def _graph(N, C=3, H=8, W=8, n_edges=1):
    x = torch.randn(N, C, H, W)
    # self-loop on node 0 as minimal valid edge
    ei = torch.zeros(2, n_edges, dtype=torch.long)
    return Graph(x, ei)


class TestGraphBatch:
    def test_node_features_concatenated(self):
        g1, g2 = _graph(3), _graph(2)
        b = GraphBatch([g1, g2])
        assert b.node_features.shape == (5, 3, 8, 8)

    def test_batch_vector_values(self):
        g1, g2 = _graph(3), _graph(2)
        b = GraphBatch([g1, g2])
        expected = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
        assert torch.equal(b.batch, expected)

    def test_edge_index_offset(self):
        """g2's edge indices must be shifted by the node count of g1."""
        x1 = torch.randn(3, 3, 8, 8)
        x2 = torch.randn(2, 3, 8, 8)
        ei1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)   # 2 edges in g1
        ei2 = torch.tensor([[0], [1]], dtype=torch.long)          # 1 edge in g2
        b = GraphBatch([Graph(x1, ei1), Graph(x2, ei2)])
        # g2's only edge: node 0+3=3  →  node 1+3=4
        assert b.edge_index[:, -1].tolist() == [3, 4]

    def test_no_edges_graph(self):
        """Graphs with edge_index=None should batch without error."""
        g1 = Graph(torch.randn(3, 3, 8, 8), None)
        g2 = Graph(torch.randn(2, 3, 8, 8), None)
        b = GraphBatch([g1, g2])
        assert b.node_features.shape == (5, 3, 8, 8)
        assert b.edge_index is None

    def test_incompatible_spatial_size_raises(self):
        """H-09: different H/W must give a descriptive ValueError, not a cryptic cat error."""
        g1 = _graph(3, H=8, W=8)
        g2 = _graph(2, H=16, W=16)
        with pytest.raises(ValueError, match="per-node feature shape"):
            GraphBatch([g1, g2])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            GraphBatch([])

    def test_to_cpu(self):
        b = GraphBatch([_graph(3), _graph(2)]).to("cpu")
        assert b.node_features.device.type == "cpu"
        assert b.batch.device.type == "cpu"

    def test_single_graph(self):
        g = _graph(4)
        b = GraphBatch([g])
        assert torch.equal(b.batch, torch.zeros(4, dtype=torch.long))
