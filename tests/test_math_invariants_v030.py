"""Mathematical invariants that v0.3.0 ships with stable confidence."""
from __future__ import annotations

import pytest
import torch

from tgraphx import (
    APPNP,
    ConvMessagePassing,
    GATv2Conv,
    GCNConv,
    Graph,
    LinearMessagePassing,
    TensorGATLayer,
    TensorGINLayer,
    TensorGraphSAGELayer,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _vector_graph(N=6, D=4, E=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, D, generator=g)
    src = torch.randint(0, N, (E,), generator=g)
    dst = torch.randint(0, N, (E,), generator=g)
    return x, torch.stack([src, dst], dim=0).long()


def _spatial_graph(N=4, C=3, H=4, W=4, E=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, C, H, W, generator=g)
    src = torch.randint(0, N, (E,), generator=g)
    dst = torch.randint(0, N, (E,), generator=g)
    return x, torch.stack([src, dst], dim=0).long()


def _permute(x, ei, perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel())
    new_x = x[perm]
    new_ei = inv[ei]
    return new_x, new_ei, inv


# ── Permutation equivariance ─────────────────────────────────────────────────


class TestPermutationEquivariance:
    """For homogeneous layers ``layer(perm(x), perm(ei)) == perm(layer(x, ei))``."""

    def _check(self, layer, x, ei):
        layer.eval()
        with torch.no_grad():
            out_a = layer(x, ei)
            perm = torch.randperm(x.size(0))
            x_p, ei_p, inv = _permute(x, ei, perm)
            out_b = layer(x_p, ei_p)
            # un-permute the second output
            out_b_unperm = out_b[torch.argsort(perm)]
            assert torch.allclose(out_a, out_b_unperm, atol=1e-5)

    def test_gcn_conv(self):
        x, ei = _vector_graph(N=8, D=4)
        self._check(GCNConv(4, 4), x, ei)

    def test_gatv2(self):
        x, ei = _vector_graph(N=8, D=4)
        self._check(GATv2Conv(4, 4, num_heads=2), x, ei)

    def test_appnp(self):
        x, ei = _vector_graph(N=8, D=4)
        self._check(APPNP(K=3, alpha=0.1), x, ei)

    def test_linear_message_passing(self):
        x, ei = _vector_graph(N=8, D=4)
        self._check(LinearMessagePassing(in_shape=(4,), out_shape=(8,), aggr="mean"), x, ei)

    def test_tensor_conv_message(self):
        x, ei = _spatial_graph(N=4, C=3, H=4, W=4)
        layer = ConvMessagePassing(in_shape=(3, 4, 4), out_shape=(4, 4, 4), aggr="sum")
        self._check(layer, x, ei)

    def test_tensor_sage(self):
        x, ei = _spatial_graph(N=4, C=3, H=4, W=4)
        layer = TensorGraphSAGELayer(in_channels=3, out_channels=4, aggr="mean")
        self._check(layer, x, ei)

    def test_tensor_gin(self):
        x, ei = _spatial_graph(N=4, C=3, H=4, W=4)
        layer = TensorGINLayer(in_channels=3, out_channels=4)
        self._check(layer, x, ei)


# ── Edge-order invariance (for permutation-invariant aggregators) ────────────


class TestEdgeOrderInvariance:
    def _check(self, layer, x, ei):
        layer.eval()
        with torch.no_grad():
            out_a = layer(x, ei)
            perm = torch.randperm(ei.size(1))
            ei_p = ei[:, perm]
            out_b = layer(x, ei_p)
            assert torch.allclose(out_a, out_b, atol=1e-5)

    def test_gcn_conv(self):
        x, ei = _vector_graph(N=8, D=4)
        self._check(GCNConv(4, 4), x, ei)

    def test_appnp(self):
        x, ei = _vector_graph(N=8, D=4)
        self._check(APPNP(K=2, alpha=0.2), x, ei)


# ── Isolated-node behaviour ──────────────────────────────────────────────────


class TestIsolatedNodes:
    def test_gcn_conv(self):
        layer = GCNConv(4, 4)
        x = torch.randn(5, 4)
        # Node 4 has no edges — with self-loops, it should still produce finite output.
        ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = layer(x, ei)
        assert torch.isfinite(out).all()

    def test_appnp(self):
        prop = APPNP(K=2, alpha=0.5)
        x = torch.randn(5, 4)
        ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = prop(x, ei)
        assert torch.isfinite(out).all()


# ── GAT attention normalisation ──────────────────────────────────────────────


class TestGATAttention:
    def test_attention_sums_to_one_per_dest(self):
        torch.manual_seed(0)
        x = torch.randn(4, 4, 4, 4)
        # Make sure every destination has at least one incoming edge.
        ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        layer = TensorGATLayer(in_channels=4, out_channels=8, num_heads=2,
                               add_self_loops=False)
        layer.eval()
        with torch.no_grad():
            out, attn = layer(x, ei, return_attention=True)
        # Sum per destination per head must equal 1.
        sums = torch.zeros(4, 2)
        sums.index_add_(0, ei[1], attn)
        assert torch.allclose(sums, torch.ones(4, 2), atol=1e-5)

    def test_chunked_matches_unchunked(self):
        torch.manual_seed(0)
        x = torch.randn(8, 4, 4, 4)
        # Build a denser graph so chunking has work to do.
        src = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6])
        dst = torch.tensor([1, 2, 3, 4, 5, 6, 7, 0, 4, 5, 6, 7])
        ei = torch.stack([src, dst], dim=0).long()
        layer = TensorGATLayer(in_channels=4, out_channels=8, num_heads=2)
        layer.eval()
        with torch.no_grad():
            out_full = layer(x, ei)
            out_chunked = layer(x, ei, chunk_size=3)
        assert torch.allclose(out_full, out_chunked, atol=1e-4)
