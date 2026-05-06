"""Vector edge feature tests for tensor-aware GNN layers.

Verifies for each layer that supports vector edge features
(``[E, edge_dim]``):

* output shape is unchanged from the no-edge case;
* zeroing the edge features changes the output (so the layer actually
  uses them);
* edge projection parameters receive non-zero gradients during backward;
* ``edge_features.requires_grad=True`` produces finite, non-zero
  gradients on the edge tensor itself;
* shape and dtype validation gives clear errors.

Layers covered:

* ``TensorGATLayer``         — vector edge bias on attention logits
* ``TensorGINLayer``         — vector edge term added to source before ReLU
* ``TensorGraphSAGELayer``   — vector edge bias added after ``W_neigh``
"""

from __future__ import annotations

import pytest
import torch

from tgraphx.layers import (
    TensorGATLayer,
    TensorGraphSAGELayer,
    TensorGINLayer,
)


# ──────────────────────────────────────────────────────────────────── #
# Helpers                                                                #
# ──────────────────────────────────────────────────────────────────── #

# Use a graph where node 1 has multiple incoming edges so the GAT
# attention bias actually changes the softmax output.
def _graph(seed: int = 0):
    torch.manual_seed(seed)
    N, C, H, W = 6, 4, 4, 4
    x = torch.randn(N, C, H, W)
    ei = torch.tensor(
        [[0, 2, 3, 0, 1, 4, 5, 4],
         [1, 1, 1, 2, 3, 0, 4, 5]],
        dtype=torch.long,
    )
    return x, ei, N, C, H, W


# ──────────────────────────────────────────────────────────────────── #
# TensorGATLayer                                                         #
# ──────────────────────────────────────────────────────────────────── #

class TestTensorGATEdgeFeatures:
    EDGE_DIM = 3

    def _layer(self):
        return TensorGATLayer(
            in_channels=4, out_channels=8, num_heads=2,
            use_edge_features=True, edge_dim=self.EDGE_DIM,
            attn_dropout=0.0, residual=False, bias=False,
        )

    def test_forward_shape(self):
        x, ei, N, *_ = _graph()
        layer = self._layer()
        ef = torch.randn(ei.size(1), self.EDGE_DIM)
        out = layer(x, ei, edge_features=ef)
        assert out.shape == (N, 8, x.size(-2), x.size(-1))

    def test_output_changes_when_edges_zeroed(self):
        x, ei, *_ = _graph()
        layer = self._layer()
        layer.eval()
        ef = torch.randn(ei.size(1), self.EDGE_DIM)
        out_real = layer(x, ei, edge_features=ef)
        out_zero = layer(x, ei, edge_features=torch.zeros_like(ef))
        diff = (out_real - out_zero).abs().max().item()
        assert diff > 1e-4, f"edge bias had no effect; max abs diff = {diff:.2e}"

    def test_edge_projection_param_gradient(self):
        x, ei, *_ = _graph()
        layer = self._layer()
        ef = torch.randn(ei.size(1), self.EDGE_DIM)
        layer(x, ei, edge_features=ef).sum().backward()
        wgrad = layer.edge_bias_proj.weight.grad
        assert wgrad is not None
        assert torch.isfinite(wgrad).all()
        assert wgrad.norm().item() > 0.0

    def test_edge_features_input_gradient(self):
        x, ei, *_ = _graph()
        layer = self._layer()
        ef = torch.randn(ei.size(1), self.EDGE_DIM, requires_grad=True)
        layer(x, ei, edge_features=ef).sum().backward()
        assert ef.grad is not None
        assert torch.isfinite(ef.grad).all()
        assert ef.grad.norm().item() > 0.0

    def test_self_loop_padding(self):
        """Self-loops get zero edge bias; output must remain finite."""
        x, ei, *_ = _graph()
        layer = TensorGATLayer(
            in_channels=4, out_channels=8, num_heads=2,
            use_edge_features=True, edge_dim=self.EDGE_DIM,
            add_self_loops=True,
        )
        ef = torch.randn(ei.size(1), self.EDGE_DIM)
        out, attn = layer(x, ei, edge_features=ef, return_attention=True)
        assert torch.isfinite(out).all()
        # E_eff = E + N
        assert attn.shape == (ei.size(1) + x.size(0), layer.num_heads)

    def test_wrong_edge_dim_raises(self):
        layer = self._layer()
        x, ei, *_ = _graph()
        with pytest.raises(ValueError, match="edge_dim"):
            layer(x, ei, edge_features=torch.randn(ei.size(1), self.EDGE_DIM + 1))

    def test_wrong_edge_count_raises(self):
        layer = self._layer()
        x, ei, *_ = _graph()
        with pytest.raises(ValueError, match="edges"):
            layer(x, ei, edge_features=torch.randn(ei.size(1) + 5, self.EDGE_DIM))


# ──────────────────────────────────────────────────────────────────── #
# TensorGINLayer (vector kind)                                          #
# ──────────────────────────────────────────────────────────────────── #

class TestTensorGINEdgeFeaturesVector:
    EDGE_DIM = 3

    def _layer(self):
        return TensorGINLayer(
            in_channels=4, out_channels=8,
            use_edge_features=True, edge_dim=self.EDGE_DIM,
            edge_features_kind="vector",
        )

    def test_forward_shape(self):
        x, ei, N, *_ = _graph()
        layer = self._layer()
        ef = torch.randn(ei.size(1), self.EDGE_DIM)
        out = layer(x, ei, edge_features=ef)
        assert out.shape == (N, 8, x.size(-2), x.size(-1))

    def test_output_changes_when_edges_zeroed(self):
        x, ei, *_ = _graph()
        layer = self._layer()
        layer.eval()
        ef = torch.randn(ei.size(1), self.EDGE_DIM)
        out_real = layer(x, ei, edge_features=ef)
        out_zero = layer(x, ei, edge_features=torch.zeros_like(ef))
        diff = (out_real - out_zero).abs().max().item()
        assert diff > 1e-4, f"edge bias had no effect; max abs diff = {diff:.2e}"

    def test_edge_projection_param_gradient(self):
        x, ei, *_ = _graph()
        layer = self._layer()
        ef = torch.randn(ei.size(1), self.EDGE_DIM)
        layer(x, ei, edge_features=ef).sum().backward()
        wgrad = layer.edge_proj.weight.grad
        assert wgrad is not None and torch.isfinite(wgrad).all()
        assert wgrad.norm().item() > 0.0

    def test_edge_features_input_gradient(self):
        x, ei, *_ = _graph()
        layer = self._layer()
        ef = torch.randn(ei.size(1), self.EDGE_DIM, requires_grad=True)
        layer(x, ei, edge_features=ef).sum().backward()
        assert torch.isfinite(ef.grad).all()
        assert ef.grad.norm().item() > 0.0

    def test_wrong_kind_raises(self):
        with pytest.raises(ValueError, match="edge_features_kind"):
            TensorGINLayer(
                in_channels=4, out_channels=8,
                use_edge_features=True, edge_dim=3,
                edge_features_kind="bogus",
            )

    def test_passing_spatial_to_vector_layer_raises(self):
        layer = self._layer()
        x, ei, *_ = _graph()
        with pytest.raises(ValueError, match=r"\[E, edge_dim\]"):
            layer(x, ei, edge_features=torch.randn(ei.size(1), self.EDGE_DIM, 4, 4))


# ──────────────────────────────────────────────────────────────────── #
# TensorGraphSAGELayer (vector kind)                                    #
# ──────────────────────────────────────────────────────────────────── #

class TestTensorGraphSAGEEdgeFeaturesVector:
    EDGE_DIM = 3

    def _layer(self):
        return TensorGraphSAGELayer(
            in_channels=4, out_channels=8, aggr="mean",
            use_edge_features=True, edge_dim=self.EDGE_DIM,
            edge_features_kind="vector",
        )

    def test_forward_shape(self):
        x, ei, N, *_ = _graph()
        layer = self._layer()
        ef = torch.randn(ei.size(1), self.EDGE_DIM)
        out = layer(x, ei, edge_features=ef)
        assert out.shape == (N, 8, x.size(-2), x.size(-1))

    def test_output_changes_when_edges_zeroed(self):
        x, ei, *_ = _graph()
        layer = self._layer()
        layer.eval()
        ef = torch.randn(ei.size(1), self.EDGE_DIM)
        out_real = layer(x, ei, edge_features=ef)
        out_zero = layer(x, ei, edge_features=torch.zeros_like(ef))
        diff = (out_real - out_zero).abs().max().item()
        assert diff > 1e-4

    def test_edge_projection_param_gradient(self):
        x, ei, *_ = _graph()
        layer = self._layer()
        ef = torch.randn(ei.size(1), self.EDGE_DIM)
        layer(x, ei, edge_features=ef).sum().backward()
        wgrad = layer.edge_bias_proj.weight.grad
        assert wgrad is not None and torch.isfinite(wgrad).all()
        assert wgrad.norm().item() > 0.0

    def test_edge_features_input_gradient(self):
        x, ei, *_ = _graph()
        layer = self._layer()
        ef = torch.randn(ei.size(1), self.EDGE_DIM, requires_grad=True)
        layer(x, ei, edge_features=ef).sum().backward()
        assert torch.isfinite(ef.grad).all()
        assert ef.grad.norm().item() > 0.0

    def test_passing_spatial_to_vector_layer_raises(self):
        layer = self._layer()
        x, ei, *_ = _graph()
        with pytest.raises(ValueError, match=r"\[E, edge_dim\]"):
            layer(x, ei, edge_features=torch.randn(ei.size(1), self.EDGE_DIM, 4, 4))
