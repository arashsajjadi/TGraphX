"""Tests for message-passing layers.

Covers:
- ConvMessagePassing: shape, aggr variants, residual, backward (C-06, H-02)
- AttentionMessagePassing: vector branch (C-05), spatial branch, backward
- LinearMessagePassing: vector forward and backward (base layer)

Note on AttentionMessagePassing:
  The current implementation computes per-edge sigmoid gating, NOT per-destination
  softmax over the incoming neighbourhood.  It is therefore NOT mathematically
  equivalent to GAT (Velickovic et al. 2018).  The tests below verify the
  implemented behaviour, not a true attention mechanism.
"""

import pytest
import torch

from tgraphx.layers import (
    ConvMessagePassing,
    AttentionMessagePassing,
    LinearMessagePassing,
)


# ──────────────────────────────────────────────────────────────────── #
# Shared helpers                                                        #
# ──────────────────────────────────────────────────────────────────── #

N, C, H, W = 4, 3, 8, 8
D = 32  # vector dimension for LinearMessagePassing / AttentionMessagePassing


def _ei(n=N, device="cpu"):
    """Directed cycle edge index [2, n] with each node having exactly 1 in-edge."""
    src = torch.arange(n, device=device)
    return torch.stack([src, (src + 1) % n])


def _spatial(n=N, device="cpu"):
    return torch.randn(n, C, H, W, device=device)


def _vector(n=N, device="cpu"):
    return torch.randn(n, D, device=device)


def _fast_agg(**kw):
    """Minimal aggregator: 1 layer, no BN, no dropout — keeps tests fast."""
    return {"num_layers": 1, "use_batchnorm": False, "dropout_prob": 0.0, **kw}


# ──────────────────────────────────────────────────────────────────── #
# ConvMessagePassing                                                     #
# ──────────────────────────────────────────────────────────────────── #

class TestConvMessagePassing:
    OUT_C = 8

    def _layer(self, aggr="sum", residual=False, out_c=None):
        oc = out_c or self.OUT_C
        return ConvMessagePassing(
            (C, H, W), (oc, H, W),
            aggr=aggr, residual=residual,
            aggregator_params=_fast_agg(),
        )

    # -- output shape --

    def test_output_shape_sum(self):
        out = self._layer("sum")(_spatial(), _ei())
        assert out.shape == (N, self.OUT_C, H, W)

    def test_output_shape_mean(self):
        """C-06: mean aggregation must produce the correct shape for [N, C, H, W]."""
        out = self._layer("mean")(_spatial(), _ei())
        assert out.shape == (N, self.OUT_C, H, W)

    def test_output_values_finite_sum(self):
        out = self._layer("sum")(_spatial(), _ei())
        assert torch.isfinite(out).all()

    def test_output_values_finite_mean(self):
        out = self._layer("mean")(_spatial(), _ei())
        assert torch.isfinite(out).all()

    # -- aggr="max" must raise NotImplementedError (H-02) --

    def test_aggr_max_raises_not_implemented(self):
        """H-02: aggr='max' must raise NotImplementedError, not silently return sum."""
        layer = self._layer("max")
        with pytest.raises(NotImplementedError, match="aggr='max'"):
            layer(_spatial(), _ei())

    # -- backward pass --

    def test_backward_input_gradient(self):
        x = _spatial().requires_grad_(True)
        out = self._layer()(_spatial(), _ei())
        # use x via a separate forward that tracks x directly
        layer = self._layer()
        out = layer(x, _ei())
        out.sum().backward()
        assert x.grad is not None

    def test_backward_param_gradient(self):
        layer = self._layer()
        out = layer(_spatial(), _ei())
        out.sum().backward()
        first_param_grad = next(layer.parameters()).grad
        assert first_param_grad is not None

    def test_no_nan_in_gradients(self):
        x = _spatial().requires_grad_(True)
        layer = self._layer()
        layer(x, _ei()).sum().backward()
        assert torch.isfinite(x.grad).all()

    # -- residual connection --

    def test_residual_applied_when_shapes_match(self):
        """Residual skip should be active when in_channels == out_channels."""
        layer = ConvMessagePassing(
            (C, H, W), (C, H, W),
            residual=True, aggregator_params=_fast_agg(),
        )
        out = layer(_spatial(), _ei())
        assert out.shape == (N, C, H, W)
        assert layer.residual is True

    def test_residual_skipped_silently_when_shapes_differ(self):
        """When in_shape != out_shape, residual is skipped — no crash."""
        layer = ConvMessagePassing(
            (C, H, W), (self.OUT_C, H, W),
            residual=True, aggregator_params=_fast_agg(),
        )
        out = layer(_spatial(), _ei())
        assert out.shape == (N, self.OUT_C, H, W)

    # -- edge features --

    def test_with_edge_features(self):
        n_edges = N  # cycle has N edges
        layer = ConvMessagePassing(
            (C, H, W), (self.OUT_C, H, W),
            use_edge_features=True, aggregator_params=_fast_agg(),
        )
        ef = torch.randn(n_edges, C, H, W)
        out = layer(_spatial(), _ei(), edge_features=ef)
        assert out.shape == (N, self.OUT_C, H, W)

    # -- 1-D in_shape is rejected --

    def test_vector_in_shape_raises(self):
        """ConvMessagePassing requires spatial (≥3-D) node features."""
        with pytest.raises(ValueError):
            ConvMessagePassing((64,), (32,))

    # -- single-node self-loop edge case --

    def test_single_node_self_loop(self):
        x = torch.randn(1, C, H, W)
        ei = torch.tensor([[0], [0]], dtype=torch.long)
        out = self._layer()(x, ei)
        assert out.shape == (1, self.OUT_C, H, W)

    # -- empty edge set --

    def test_no_edges(self):
        x = torch.randn(4, C, H, W)
        ei = torch.zeros(2, 0, dtype=torch.long)
        out = self._layer()(x, ei)
        assert out.shape == (4, self.OUT_C, H, W)
        assert torch.isfinite(out).all()


# ──────────────────────────────────────────────────────────────────── #
# AttentionMessagePassing                                               #
# ──────────────────────────────────────────────────────────────────── #

class TestAttentionMessagePassing:
    OUT = 16

    # -- vector input branch (C-05 fix) --

    def test_vector_branch_instantiation(self):
        """C-05: vector branch used undefined names; must now instantiate cleanly."""
        AttentionMessagePassing(in_shape=(D,), out_shape=(self.OUT,))

    def test_vector_forward_shape(self):
        layer = AttentionMessagePassing(in_shape=(D,), out_shape=(self.OUT,))
        out = layer(_vector(), _ei())
        assert out.shape == (N, self.OUT)

    def test_vector_output_finite(self):
        layer = AttentionMessagePassing(in_shape=(D,), out_shape=(self.OUT,))
        out = layer(_vector(), _ei())
        assert torch.isfinite(out).all()

    def test_vector_backward(self):
        x = _vector().requires_grad_(True)
        layer = AttentionMessagePassing(in_shape=(D,), out_shape=(self.OUT,))
        layer(x, _ei()).sum().backward()
        assert x.grad is not None

    # -- spatial input branch --

    def test_spatial_forward_shape(self):
        layer = AttentionMessagePassing(in_shape=(C, H, W), out_shape=(self.OUT, H, W))
        out = layer(_spatial(), _ei())
        assert out.shape == (N, self.OUT, H, W)

    def test_spatial_output_finite(self):
        layer = AttentionMessagePassing(in_shape=(C, H, W), out_shape=(self.OUT, H, W))
        out = layer(_spatial(), _ei())
        assert torch.isfinite(out).all()

    def test_spatial_backward(self):
        x = _spatial().requires_grad_(True)
        layer = AttentionMessagePassing(in_shape=(C, H, W), out_shape=(self.OUT, H, W))
        layer(x, _ei()).sum().backward()
        assert x.grad is not None


# ──────────────────────────────────────────────────────────────────── #
# LinearMessagePassing (base layer, vector path)                        #
# ──────────────────────────────────────────────────────────────────── #

class TestLinearMessagePassing:
    def test_forward_shape(self):
        layer = LinearMessagePassing(in_shape=(D,), out_shape=(16,))
        x = _vector()
        out = layer(x, _ei())
        assert out.shape == (N, 16)

    def test_backward(self):
        x = _vector().requires_grad_(True)
        layer = LinearMessagePassing(in_shape=(D,), out_shape=(16,))
        layer(x, _ei()).sum().backward()
        assert x.grad is not None

    def test_mean_aggregation_vector(self):
        """C-06 fix must also work for vector features (shape [N, D])."""
        layer = LinearMessagePassing(in_shape=(D,), out_shape=(16,), aggr="mean")
        out = layer(_vector(), _ei())
        assert out.shape == (N, 16)
        assert torch.isfinite(out).all()
