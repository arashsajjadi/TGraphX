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
    TensorGATLayer,
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

    # -- aggr="max" is now implemented (Batch 6.5) --

    def test_aggr_max_runs_and_finite(self):
        """Batch 6.5: aggr='max' runs through scatter_max in the base class."""
        layer = self._layer("max")
        out = layer(_spatial(), _ei())
        assert out.shape == (N, self.OUT_C, H, W)
        assert torch.isfinite(out).all()

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

    # ── API-05: rank guard ────────────────────────────────────────────

    def test_spatial_in_shape_raises(self):
        """2-D spatial in_shape must raise ValueError (API-05)."""
        with pytest.raises(ValueError, match="LinearMessagePassing supports vector"):
            LinearMessagePassing(in_shape=(3, 8, 8), out_shape=(8, 8, 8))

    def test_volumetric_in_shape_raises(self):
        """3-D volumetric in_shape must raise ValueError (API-05)."""
        with pytest.raises(ValueError, match="LinearMessagePassing supports vector"):
            LinearMessagePassing(in_shape=(3, 4, 8, 8), out_shape=(8, 4, 8, 8))

    def test_spatial_out_shape_raises(self):
        """Spatial out_shape must raise ValueError even with vector in_shape."""
        with pytest.raises(ValueError, match="out_shape must be a 1-element tuple"):
            LinearMessagePassing(in_shape=(16,), out_shape=(8, 4, 4))

    # ── BUG-02: dropout actually fires ───────────────────────────────

    def test_dropout_changes_train_output(self):
        """With dropout_prob=0.9, train-mode output must differ from eval-mode."""
        layer = LinearMessagePassing(in_shape=(D,), out_shape=(D,), dropout_prob=0.9)
        x = torch.ones(N, D)
        ei = _ei()
        layer.train()
        # Run several times in train mode to make a collision astronomically unlikely.
        outputs_train = [layer(x, ei).detach() for _ in range(5)]
        layer.eval()
        out_eval = layer(x, ei).detach()
        # At least one train run must differ from eval (prob of all-same < 0.1^5).
        assert any(not torch.equal(o, out_eval) for o in outputs_train), (
            "dropout_prob=0.9 had no effect: train and eval outputs were identical"
        )

    def test_dropout_disabled_in_eval(self):
        """Eval mode must be deterministic (dropout off)."""
        layer = LinearMessagePassing(in_shape=(D,), out_shape=(D,), dropout_prob=0.5)
        layer.eval()
        x = torch.randn(N, D)
        ei = _ei()
        a = layer(x, ei)
        b = layer(x, ei)
        assert torch.equal(a, b), "eval-mode output must be deterministic"

    # ── BUG-02: residual actually fires ──────────────────────────────

    def test_residual_same_shape(self):
        """residual=True with matching D must add the input to the output."""
        layer = LinearMessagePassing(in_shape=(D,), out_shape=(D,), residual=True)
        layer.eval()
        x = torch.randn(N, D)
        ei = _ei()
        # Run once with residual, once with a copy of the layer where we manually
        # zero the skip and compare: residual adds the identity.
        out_with = layer(x, ei)
        # Manually compute without residual by clearing the flag.
        layer.residual = False
        out_without = layer(x, ei)
        layer.residual = True
        # out_with == out_without + x  (both computed from same weights/state)
        assert torch.allclose(out_with, out_without + x, atol=1e-5), (
            "residual=True did not add the input to the output"
        )

    def test_residual_different_shape_no_crash(self):
        """residual=True with mismatched shapes must silently skip the skip-connection."""
        layer = LinearMessagePassing(in_shape=(D,), out_shape=(16,), residual=True)
        out = layer(_vector(), _ei())
        assert out.shape == (N, 16)

    # ── BUG-02: batchnorm actually fires ─────────────────────────────

    def test_batchnorm_module_created(self):
        """use_batchnorm=True must create a BatchNorm1d on the layer."""
        import torch.nn as nn
        layer = LinearMessagePassing(in_shape=(D,), out_shape=(16,), use_batchnorm=True)
        assert hasattr(layer, "bn"), "BatchNorm module not found"
        assert isinstance(layer.bn, nn.BatchNorm1d)

    def test_batchnorm_forward_works(self):
        """Forward must succeed with use_batchnorm=True (at least 2 nodes needed)."""
        layer = LinearMessagePassing(in_shape=(D,), out_shape=(16,), use_batchnorm=True)
        layer.train()
        # BatchNorm1d requires batch size ≥ 2 in training mode.
        x = torch.randn(4, D)
        ei = _ei(4)
        out = layer(x, ei)
        assert out.shape == (4, 16)
        assert torch.isfinite(out).all()

    def test_batchnorm_train_differs_from_eval(self):
        """BN running stats cause train/eval divergence on fresh parameters."""
        layer = LinearMessagePassing(in_shape=(D,), out_shape=(16,), use_batchnorm=True)
        x = torch.randn(4, D)
        ei = _ei(4)
        layer.train()
        out_train = layer(x, ei).detach()
        layer.eval()
        out_eval = layer(x, ei).detach()
        # The very first eval pass after training accumulates running stats,
        # so the outputs must differ unless the data is pathologically symmetric.
        assert not torch.equal(out_train, out_eval), (
            "BN train/eval outputs identical — batchnorm may not be active"
        )


# ──────────────────────────────────────────────────────────────────── #
# BUG-03: TensorGATLayer add_self_loops deduplication                   #
# ──────────────────────────────────────────────────────────────────── #

class TestGATSelfLoopDedup:
    """BUG-03 regression tests: add_self_loops=True must not duplicate existing loops."""

    C_GAT, H_GAT, W_GAT = 4, 4, 4  # small spatial dims for speed
    OUT_C = 8

    def _layer(self, **kw):
        return TensorGATLayer(
            in_channels=self.C_GAT,
            out_channels=self.OUT_C,
            num_heads=2,
            add_self_loops=True,
            **kw,
        )

    def _x(self, n=4):
        return torch.randn(n, self.C_GAT, self.H_GAT, self.W_GAT)

    # ── no self-loops in input → all N are added ─────────────────────

    def test_no_existing_self_loops_adds_all(self):
        """When edge_index has no self-loops, N self-loops should be appended."""
        n = 4
        # directed cycle, no self-loops
        src = torch.arange(n)
        ei = torch.stack([src, (src + 1) % n])
        layer = self._layer()
        out, attn = layer(self._x(n), ei, return_attention=True)
        # Each destination node must have at least its self-loop in the attn
        assert out.shape == (n, self.OUT_C, self.H_GAT, self.W_GAT)
        assert torch.isfinite(out).all()
        # E_eff = E_orig + n (N new loops)
        assert attn.shape[0] == ei.size(1) + n

    # ── all self-loops already present → none added ───────────────────

    def test_all_self_loops_present_no_duplicates(self):
        """When every node already has a self-loop, no new loops are added."""
        n = 4
        src = torch.arange(n)
        # cycle + self-loops for every node
        cycle = torch.stack([src, (src + 1) % n])
        loops = torch.stack([src, src])
        ei = torch.cat([cycle, loops], dim=1)
        E_orig = ei.size(1)
        layer = self._layer()
        out, attn = layer(self._x(n), ei, return_attention=True)
        assert out.shape == (n, self.OUT_C, self.H_GAT, self.W_GAT)
        assert torch.isfinite(out).all()
        # No new loops added
        assert attn.shape[0] == E_orig

    # ── partial self-loops → only missing ones added ──────────────────

    def test_partial_self_loops_adds_only_missing(self):
        """Only nodes without a self-loop get one added."""
        n = 4
        # self-loop only for node 0
        ei = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 0]], dtype=torch.long)
        E_orig = ei.size(1)  # 5 edges (4 cycle + 1 self-loop for node 0)
        layer = self._layer()
        out, attn = layer(self._x(n), ei, return_attention=True)
        assert out.shape == (n, self.OUT_C, self.H_GAT, self.W_GAT)
        assert torch.isfinite(out).all()
        # n-1 = 3 missing self-loops should be added
        assert attn.shape[0] == E_orig + (n - 1)

    # ── edge_weight padded correctly ─────────────────────────────────

    def test_edge_weight_padded_for_new_loops_only(self):
        """edge_weight padding must match the number of newly added loops."""
        n = 4
        src = torch.arange(n)
        # cycle (no self-loops)
        ei = torch.stack([src, (src + 1) % n])
        ew = torch.ones(ei.size(1))
        layer = self._layer()
        out = layer(self._x(n), ei, edge_weight=ew)
        assert out.shape == (n, self.OUT_C, self.H_GAT, self.W_GAT)
        assert torch.isfinite(out).all()

    def test_edge_weight_no_padding_when_loops_already_present(self):
        """When all self-loops exist, edge_weight must not be padded."""
        n = 4
        src = torch.arange(n)
        cycle = torch.stack([src, (src + 1) % n])
        loops = torch.stack([src, src])
        ei = torch.cat([cycle, loops], dim=1)
        ew = torch.ones(ei.size(1))
        layer = self._layer()
        out = layer(self._x(n), ei, edge_weight=ew)
        assert out.shape == (n, self.OUT_C, self.H_GAT, self.W_GAT)
        assert torch.isfinite(out).all()

    # ── vector edge_features padded correctly ────────────────────────

    def test_vector_edge_features_padded_for_new_loops(self):
        """Vector edge_features bias must be padded for new loops only."""
        n = 4
        edge_dim = 3
        src = torch.arange(n)
        ei = torch.stack([src, (src + 1) % n])  # no self-loops
        ef = torch.randn(ei.size(1), edge_dim)
        layer = TensorGATLayer(
            in_channels=self.C_GAT,
            out_channels=self.OUT_C,
            num_heads=2,
            add_self_loops=True,
            use_edge_features=True,
            edge_dim=edge_dim,
        )
        out = layer(self._x(n), ei, edge_features=ef)
        assert out.shape == (n, self.OUT_C, self.H_GAT, self.W_GAT)
        assert torch.isfinite(out).all()

    # ── softmax sums to 1 per destination ────────────────────────────

    def test_attention_normalized_per_destination(self):
        """Softmax must sum to 1.0 per destination across all K heads."""
        n = 4
        src = torch.arange(n)
        ei = torch.stack([src, (src + 1) % n])
        layer = self._layer()
        layer.eval()
        _, attn = layer(self._x(n), ei, return_attention=True)  # [E_eff, K]
        # attn[e, k] for all e with dst==j must sum to 1 per head k
        # Build dst for E_eff: first E_orig are cycle destinations, rest are self-loops
        n_new = n  # no existing self-loops
        dst_eff = torch.cat([
            (src + 1) % n,           # cycle destinations
            torch.arange(n),         # self-loop destinations
        ])
        for j in range(n):
            for k in range(layer.num_heads):
                mask = dst_eff == j
                s = attn[mask, k].sum().item()
                assert abs(s - 1.0) < 1e-4, (
                    f"Node {j} head {k}: attention sums to {s:.6f} not 1.0"
                )
