"""Vector model-zoo layer tests (v0.3.0)."""
from __future__ import annotations

import pytest
import torch

from tgraphx import (
    APPNP,
    GATv2Conv,
    GCNConv,
    global_max_pool,
    global_mean_pool,
    global_sum_pool,
)
from tgraphx.models.model_zoo import list_layers, make_zoo_layer


# ── GCNConv ──────────────────────────────────────────────────────────────────


class TestGCNConv:
    def test_forward_shape(self):
        N, D, D_out = 6, 4, 8
        x = torch.randn(N, D)
        ei = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
        layer = GCNConv(D, D_out)
        out = layer(x, ei)
        assert out.shape == (N, D_out)

    def test_backward_gradients(self):
        layer = GCNConv(4, 8)
        x = torch.randn(5, 4, requires_grad=True)
        ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        out = layer(x, ei)
        loss = out.sum()
        loss.backward()
        assert torch.isfinite(layer.lin.weight.grad).all()
        assert (layer.lin.weight.grad.abs() > 0).any()

    def test_isolated_node_no_nan(self):
        layer = GCNConv(4, 4)
        x = torch.randn(5, 4)
        # Node 4 has no edges.
        ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = layer(x, ei)
        assert torch.isfinite(out).all()

    def test_invalid_shape_raises(self):
        layer = GCNConv(4, 4)
        with pytest.raises(ValueError, match="vector"):
            layer(torch.randn(3, 4, 4), torch.zeros((2, 0), dtype=torch.long))


# ── GATv2Conv ────────────────────────────────────────────────────────────────


class TestGATv2:
    def test_forward_shape_concat(self):
        layer = GATv2Conv(4, 8, num_heads=4, concat_heads=True)
        x = torch.randn(6, 4)
        ei = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
        out = layer(x, ei)
        assert out.shape == (6, 8)

    def test_forward_shape_mean(self):
        layer = GATv2Conv(4, 8, num_heads=4, concat_heads=False)
        x = torch.randn(6, 4)
        ei = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
        out = layer(x, ei)
        # head_dim = 8; output = mean over heads → [N, 8]
        assert out.shape == (6, 8)

    def test_invalid_concat_dim(self):
        with pytest.raises(ValueError, match="divisible"):
            GATv2Conv(4, 7, num_heads=4)

    def test_backward_gradients(self):
        layer = GATv2Conv(4, 8, num_heads=2)
        x = torch.randn(5, 4, requires_grad=True)
        ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        out = layer(x, ei)
        out.sum().backward()
        assert torch.isfinite(layer.W_l.weight.grad).all()


# ── APPNP ────────────────────────────────────────────────────────────────────


class TestAPPNP:
    def test_forward_finite(self):
        prop = APPNP(K=3, alpha=0.1)
        x = torch.randn(8, 4)
        ei = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
        out = prop(x, ei)
        assert torch.isfinite(out).all()
        assert out.shape == x.shape

    def test_invalid_K(self):
        with pytest.raises(ValueError, match="K must be"):
            APPNP(K=0)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha must be"):
            APPNP(alpha=0.0)


# ── Pooling ──────────────────────────────────────────────────────────────────


class TestPooling:
    def test_global_sum_mean_max(self):
        x = torch.tensor([
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 2.0],
            [10.0, 10.0],
        ])
        batch = torch.tensor([0, 0, 0, 1], dtype=torch.long)

        s = global_sum_pool(x, batch)
        assert torch.allclose(s, torch.tensor([[6.0, 3.0], [10.0, 10.0]]))

        m = global_mean_pool(x, batch)
        assert torch.allclose(m, torch.tensor([[2.0, 1.0], [10.0, 10.0]]))

        mx = global_max_pool(x, batch)
        assert torch.allclose(mx, torch.tensor([[3.0, 2.0], [10.0, 10.0]]))

    def test_pool_invalid_batch(self):
        x = torch.zeros(3, 2)
        with pytest.raises(ValueError, match="batch"):
            global_mean_pool(x, torch.zeros(2, dtype=torch.long))

    def test_pool_invalid_dtype(self):
        x = torch.zeros(3, 2)
        with pytest.raises(TypeError, match="torch.long"):
            global_mean_pool(x, torch.zeros(3, dtype=torch.float))


# ── Registry ─────────────────────────────────────────────────────────────────


class TestModelZooRegistry:
    def test_list_layers(self):
        names = list_layers()
        for n in ("gcn_conv", "gatv2", "appnp",
                  "global_mean_pool", "global_sum_pool", "global_max_pool"):
            assert n in names

    def test_make_layer_class(self):
        layer = make_zoo_layer("gcn_conv", in_dim=4, out_dim=8)
        assert isinstance(layer, GCNConv)

    def test_make_layer_pool(self):
        fn = make_zoo_layer("global_mean_pool")
        # Pooling helpers are returned as callables, not instances.
        assert callable(fn)

    def test_unknown_name(self):
        with pytest.raises(KeyError, match="Unknown zoo layer"):
            make_zoo_layer("nope")
