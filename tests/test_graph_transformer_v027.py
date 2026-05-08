"""GraphTransformer v0.2.7 enhancements: encodings, edge bias, factory."""
from __future__ import annotations

import pytest
import torch

from tgraphx.layers.graph_transformer import GraphTransformerLayer
from tgraphx.layers.transformer_encodings import (
    build_adjacency_bias,
    degree_encoding,
    laplacian_eigvec_encoding,
)
from tgraphx.layers.factory import make_layer


class TestGraphTransformerCore:
    def test_basic_forward_unchanged(self):
        l = GraphTransformerLayer(16, 16, num_heads=4).eval()
        x = torch.randn(8, 16)
        out = l(x)
        assert out.shape == (8, 16)

    def test_backward(self):
        l = GraphTransformerLayer(8, 8, num_heads=2)
        x = torch.randn(6, 8, requires_grad=True)
        out = l(x)
        out.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


class TestPositionalEncoding:
    def test_degree_encoding_shape(self):
        ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        enc = degree_encoding(ei, num_nodes=3, dim=8, direction="both")
        assert enc.shape == (3, 16)  # both → dim*2

    def test_degree_encoding_in_only(self):
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        enc = degree_encoding(ei, num_nodes=3, dim=4, direction="in")
        assert enc.shape == (3, 4)

    def test_laplacian_encoding_shape(self):
        ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        enc = laplacian_eigvec_encoding(ei, num_nodes=4, dim=2)
        assert enc.shape == (4, 2)
        assert torch.isfinite(enc).all()

    def test_layer_with_degree_pe(self):
        ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        N = 3
        l = GraphTransformerLayer(
            in_dim=8, out_dim=8, num_heads=2,
            positional_encoding="degree", pe_dim=8,
        )
        x = torch.randn(N, 8)
        pe = degree_encoding(ei, num_nodes=N, dim=4, direction="both")  # → [N, 8]
        out = l(x, edge_index=ei, positional=pe)
        assert out.shape == (3, 8)
        assert torch.isfinite(out).all()

    def test_invalid_pe_dim(self):
        with pytest.raises(ValueError, match="pe_dim"):
            GraphTransformerLayer(in_dim=8, out_dim=8,
                                  positional_encoding="degree", pe_dim=0)

    def test_invalid_pe_kind(self):
        with pytest.raises(ValueError, match="positional_encoding"):
            GraphTransformerLayer(in_dim=8, out_dim=8,
                                  positional_encoding="bogus", pe_dim=4)


class TestEdgeBias:
    def test_adjacency_bias_shape(self):
        ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        bias = build_adjacency_bias(ei, num_nodes=3, neg_inf=True)
        assert bias.shape == (3, 3)
        # edges: (0,1), (1,2), (2,0) and self-loops on diagonal → 0
        assert bias[0, 1] == 0.0
        assert bias[1, 2] == 0.0
        # background = -1e4 at non-edges
        assert bias[0, 2] < -100

    def test_layer_with_edge_bias(self):
        ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        N = 4
        l = GraphTransformerLayer(in_dim=8, out_dim=8, num_heads=2, edge_bias=True)
        x = torch.randn(N, 8)
        bias = build_adjacency_bias(ei, num_nodes=N, neg_inf=True)
        out = l(x, edge_index=ei, edge_bias_dense=bias)
        assert out.shape == (N, 8)
        assert torch.isfinite(out).all()

    def test_per_head_edge_bias(self):
        N = 4
        K = 2
        l = GraphTransformerLayer(in_dim=8, out_dim=8, num_heads=K, edge_bias=True)
        x = torch.randn(N, 8)
        bias = torch.randn(K, N, N)
        out = l(x, edge_bias_dense=bias)
        assert out.shape == (N, 8)

    def test_wrong_bias_shape_raises(self):
        l = GraphTransformerLayer(in_dim=8, out_dim=8, num_heads=2, edge_bias=True)
        x = torch.randn(4, 8)
        with pytest.raises(ValueError):
            l(x, edge_bias_dense=torch.randn(3, 5))


class TestFactoryIntegration:
    def test_make_layer_graph_transformer_vector(self):
        l = make_layer("graph_transformer", in_shape=(16,), out_shape=(16,), heads=4)
        x = torch.randn(5, 16)
        out = l(x)
        assert out.shape == (5, 16)

    def test_make_layer_graph_transformer_kwargs(self):
        l = make_layer(
            "graph_transformer", in_shape=(8,), out_shape=(8,),
            heads=2, dropout=0.1, residual=True, ffn_dim=32,
        )
        out = l(torch.randn(4, 8))
        assert out.shape == (4, 8)

    def test_make_layer_spatial_raises(self):
        with pytest.raises(ValueError, match="vector"):
            make_layer("graph_transformer", in_shape=(8, 4, 4), out_shape=(8, 4, 4))

    def test_make_layer_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            make_layer("not_a_real_layer", in_shape=(8,), out_shape=(8,))
