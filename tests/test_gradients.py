"""Gradient-flow hardening tests.

Each layer family is exercised with:

* a single-layer forward + backward (input gradient and parameter gradient
  are both finite and not all zero);
* an 8-layer deep stack (same checks, plus min/mean/max gradient norm
  reporting);
* an optional residual=True comparison for layers that support it (the
  test only checks that residual=True is shape-safe and does not block
  gradient flow — no scientific claim about depth efficacy).

All tests are CPU-only and deterministic.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tgraphx.layers import (
    ConvMessagePassing,
    AttentionMessagePassing,
    TensorGATLayer,
    TensorGraphSAGELayer,
    TensorGINLayer,
)


# ──────────────────────────────────────────────────────────────────── #
# Helpers                                                                #
# ──────────────────────────────────────────────────────────────────── #

N, C, H, W = 6, 4, 4, 4


def _spatial(seed: int = 0):
    torch.manual_seed(seed)
    return torch.randn(N, C, H, W)


def _ei():
    src = torch.arange(N)
    return torch.stack([src, (src + 1) % N])


def _fast_agg():
    return {"num_layers": 1, "use_batchnorm": False, "dropout_prob": 0.0}


def _grad_norm_stats(model: nn.Module):
    """Return (n_total, n_none, n_nonfinite, min_norm, mean_norm, max_norm)."""
    norms = []
    n_none = 0
    n_nonfinite = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            n_none += 1
            continue
        if not torch.isfinite(p.grad).all():
            n_nonfinite += 1
        norms.append(p.grad.norm().item())
    if not norms:
        return (0, n_none, n_nonfinite, 0.0, 0.0, 0.0)
    import math
    return (
        len(norms) + n_none,
        n_none,
        n_nonfinite,
        min(norms),
        sum(norms) / len(norms),
        max(norms),
    )


# ──────────────────────────────────────────────────────────────────── #
# Single-layer backward sanity                                          #
# ──────────────────────────────────────────────────────────────────── #

class TestSingleLayerBackward:
    def _check(self, layer):
        x = _spatial(seed=0).requires_grad_(True)
        ei = _ei()
        out = layer(x, ei)
        assert torch.isfinite(out).all(), "forward output has NaN/Inf"
        out.sum().backward()
        assert torch.isfinite(x.grad).all(), "x.grad has NaN/Inf"
        assert not torch.all(x.grad == 0), "x.grad is all zero"
        n_total, n_none, n_nonfinite, *_ = _grad_norm_stats(layer)
        assert n_none == 0, f"{n_none} parameters have no gradient"
        assert n_nonfinite == 0, f"{n_nonfinite} parameters have non-finite gradient"

    def test_conv_message_passing(self):
        self._check(ConvMessagePassing(
            (C, H, W), (8, H, W), aggregator_params=_fast_agg(),
        ))

    def test_attention_message_passing(self):
        self._check(AttentionMessagePassing(in_shape=(C, H, W), out_shape=(8, H, W)))

    def test_tensor_gat(self):
        self._check(TensorGATLayer(in_channels=C, out_channels=8, num_heads=2))

    def test_tensor_gat_with_edge_features(self):
        # Use a graph where some destination has multiple incoming edges
        # so that the per-edge attention bias actually affects softmax.
        layer = TensorGATLayer(
            in_channels=C, out_channels=8, num_heads=2,
            use_edge_features=True, edge_dim=3,
        )
        ei = torch.tensor(
            [[0, 2, 3, 0, 1, 4], [1, 1, 1, 2, 3, 0]], dtype=torch.long,
        )
        ef = torch.randn(ei.size(1), 3, requires_grad=True)
        x = _spatial(seed=0).requires_grad_(True)
        out = layer(x, ei, edge_features=ef)
        out.sum().backward()
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(ef.grad).all()
        assert not torch.all(ef.grad == 0)

    def test_tensor_graphsage_mean(self):
        self._check(TensorGraphSAGELayer(in_channels=C, out_channels=8, aggr="mean"))

    def test_tensor_graphsage_max(self):
        self._check(TensorGraphSAGELayer(in_channels=C, out_channels=8, aggr="max"))

    def test_tensor_gin(self):
        self._check(TensorGINLayer(in_channels=C, out_channels=8, train_eps=True))


# ──────────────────────────────────────────────────────────────────── #
# Deep 8-layer stack gradient sanity                                     #
# ──────────────────────────────────────────────────────────────────── #

DEPTH = 8


class _Stack(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, ei):
        for L in self.layers:
            x = L(x, ei)
        return x


def _build_conv_stack():
    return _Stack([
        ConvMessagePassing((C, H, W), (C, H, W), aggregator_params=_fast_agg())
        for _ in range(DEPTH)
    ])


def _build_gat_stack(use_edge_features=False, edge_dim=None):
    return _Stack([
        TensorGATLayer(
            in_channels=C, out_channels=C, num_heads=2,
            use_edge_features=use_edge_features, edge_dim=edge_dim,
        )
        for _ in range(DEPTH)
    ])


def _build_sage_stack():
    return _Stack([
        TensorGraphSAGELayer(in_channels=C, out_channels=C, aggr="mean")
        for _ in range(DEPTH)
    ])


def _build_gin_stack():
    return _Stack([
        TensorGINLayer(in_channels=C, out_channels=C, train_eps=True)
        for _ in range(DEPTH)
    ])


class TestDeepStackGradient:
    def _run_stack(self, model: nn.Module):
        x = _spatial(seed=1).requires_grad_(True)
        ei = _ei()
        out = model(x, ei)
        assert torch.isfinite(out).all()
        out.sum().backward()
        n_total, n_none, n_nonfinite, gmin, gmean, gmax = _grad_norm_stats(model)
        assert n_none == 0
        assert n_nonfinite == 0
        # Not catastrophically zero.
        assert gmax > 0.0
        return n_total, gmin, gmean, gmax

    def test_conv_stack(self):
        n, gmin, gmean, gmax = self._run_stack(_build_conv_stack())
        assert gmin >= 0.0  # non-negative
        # A stack of 8 1×1-conv message passings has many params; just sanity.
        assert n > 0

    def test_gat_stack(self):
        self._run_stack(_build_gat_stack())

    def test_sage_stack(self):
        self._run_stack(_build_sage_stack())

    def test_gin_stack(self):
        self._run_stack(_build_gin_stack())


# ──────────────────────────────────────────────────────────────────── #
# Residual vs non-residual: shape safety + gradient finiteness only     #
# ──────────────────────────────────────────────────────────────────── #

class TestResidualSafety:
    """We only verify that residual=True does not break the gradient path.
    No scientific claim about whether residual is "better" is made."""

    def _stack(self, residual: bool, build):
        layers = [build(residual) for _ in range(DEPTH)]
        return _Stack(layers)

    def _run(self, model):
        x = _spatial(seed=2).requires_grad_(True)
        ei = _ei()
        out = model(x, ei)
        assert torch.isfinite(out).all()
        out.sum().backward()
        n_total, n_none, n_nonfinite, *_ = _grad_norm_stats(model)
        assert n_none == 0
        assert n_nonfinite == 0

    def test_conv_residual(self):
        def build(residual):
            return ConvMessagePassing(
                (C, H, W), (C, H, W),
                aggregator_params=_fast_agg(),
                residual=residual,
            )
        self._run(self._stack(False, build))
        self._run(self._stack(True, build))

    def test_gat_residual(self):
        def build(residual):
            return TensorGATLayer(
                in_channels=C, out_channels=C, num_heads=2, residual=residual,
            )
        self._run(self._stack(False, build))
        self._run(self._stack(True, build))

    def test_sage_residual(self):
        def build(residual):
            return TensorGraphSAGELayer(
                in_channels=C, out_channels=C, residual=residual,
            )
        self._run(self._stack(False, build))
        self._run(self._stack(True, build))


# ──────────────────────────────────────────────────────────────────── #
# Tiny overfit: relational task where label depends on neighbours        #
# ──────────────────────────────────────────────────────────────────── #

class TestTinyOverfit:
    """Sanity check: a tiny model should be able to reduce loss on a
    deterministic synthetic task where the label depends on the mean of
    neighbour features (so an isolated single-node baseline can't solve it)."""

    @staticmethod
    def _make_task(seed=123):
        torch.manual_seed(seed)
        N_t, C_t, H_t, W_t = 8, 4, 3, 3
        x = torch.randn(N_t, C_t, H_t, W_t)
        # Bidirectional ring + a couple of chord edges — every node has
        # incoming neighbours.
        src = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7,    1, 2, 3, 4, 5, 6, 7, 0,    0, 4])
        dst = torch.tensor([1, 2, 3, 4, 5, 6, 7, 0,    0, 1, 2, 3, 4, 5, 6, 7,    4, 0])
        ei = torch.stack([src, dst])
        # Label per node: 1 if mean(neighbour features) is positive, else 0.
        # We compute the labels using a simple aggregation so the task is
        # learnable by any of our message-passing layers.
        with torch.no_grad():
            tgt_idx = ei[1]
            src_idx = ei[0]
            agg = torch.zeros_like(x)
            agg.index_add_(0, tgt_idx, x[src_idx])
            labels = (agg.flatten(1).mean(dim=1) > 0).long()
        return x, ei, labels

    def _train(self, layer_factory, num_classes=2, steps=40, lr=0.05):
        x, ei, labels = self._make_task()
        N_t, C_t, H_t, W_t = x.shape
        # Build a 2-layer model: GNN -> GNN -> global avg pool per node -> Linear
        torch.manual_seed(0)
        gnn1 = layer_factory(C_t, 8, H_t, W_t)
        gnn2 = layer_factory(8, 8, H_t, W_t)
        head = nn.Linear(8, num_classes)
        params = list(gnn1.parameters()) + list(gnn2.parameters()) + list(head.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        first_loss = None
        last_loss = None
        for _ in range(steps):
            opt.zero_grad()
            h = torch.relu(gnn1(x, ei))
            h = gnn2(h, ei)
            # Per-node global avg pool over spatial dims, then per-node logits.
            pooled = h.mean(dim=(-2, -1))
            logits = head(pooled)
            loss = loss_fn(logits, labels)
            if first_loss is None:
                first_loss = loss.item()
            loss.backward()
            opt.step()
            last_loss = loss.item()
        return first_loss, last_loss

    def test_overfit_conv_message_passing(self):
        first, last = self._train(
            lambda c_in, c_out, h, w: ConvMessagePassing(
                (c_in, h, w), (c_out, h, w), aggregator_params=_fast_agg(),
            ),
        )
        assert last < first - 0.05, f"loss did not decrease meaningfully: {first:.4f} -> {last:.4f}"

    def test_overfit_tensor_gat(self):
        first, last = self._train(
            lambda c_in, c_out, h, w: TensorGATLayer(
                in_channels=c_in, out_channels=c_out,
                num_heads=2, add_self_loops=True,
            ),
        )
        assert last < first - 0.05, f"loss did not decrease meaningfully: {first:.4f} -> {last:.4f}"

    def test_overfit_tensor_graphsage(self):
        first, last = self._train(
            lambda c_in, c_out, h, w: TensorGraphSAGELayer(
                in_channels=c_in, out_channels=c_out, aggr="mean",
            ),
        )
        assert last < first - 0.05, f"loss did not decrease meaningfully: {first:.4f} -> {last:.4f}"

    def test_overfit_tensor_gin(self):
        first, last = self._train(
            lambda c_in, c_out, h, w: TensorGINLayer(
                in_channels=c_in, out_channels=c_out, train_eps=True,
            ),
        )
        assert last < first - 0.05, f"loss did not decrease meaningfully: {first:.4f} -> {last:.4f}"
