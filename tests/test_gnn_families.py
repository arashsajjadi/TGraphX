"""Tests for tensor-aware GNN families: GAT, GraphSAGE, GIN.

Also covers:
- the internal edge_softmax helper
- a custom subclass of TensorMessagePassingLayer
- CPU forward/backward, optional CUDA, optional MPS
"""

import pytest
import torch
import torch.nn as nn

from tgraphx.layers import (
    TensorGATLayer,
    TensorGraphSAGELayer,
    TensorGINLayer,
    TensorMessagePassingLayer,
)
from tgraphx.layers._scatter import (
    edge_softmax,
    scatter_sum,
    scatter_mean,
    scatter_max,
)


# ──────────────────────────────────────────────────────────────────── #
# Helpers                                                               #
# ──────────────────────────────────────────────────────────────────── #

N, C, H, W = 6, 4, 4, 4


def _ei(n=N, device="cpu"):
    """Directed cycle 0→1→…→(n-1)→0."""
    src = torch.arange(n, device=device)
    return torch.stack([src, (src + 1) % n])


def _spatial(n=N, c=C, h=H, w=W, device="cpu"):
    return torch.randn(n, c, h, w, device=device)


# ──────────────────────────────────────────────────────────────────── #
# edge_softmax helper                                                    #
# ──────────────────────────────────────────────────────────────────── #

class TestEdgeSoftmax:
    def test_sums_to_one_per_destination_1d(self):
        ei = torch.tensor([[0, 2, 3, 0, 1], [1, 1, 1, 2, 3]], dtype=torch.long)
        scores = torch.randn(5)
        attn = edge_softmax(scores, ei[1], num_nodes=4)
        sums = torch.zeros(4)
        sums.index_add_(0, ei[1], attn)
        # destinations 1, 2, 3 each receive at least one edge
        assert torch.allclose(sums[1:], torch.ones(3), atol=1e-5)
        # destination 0 receives no incoming edge
        assert sums[0].item() == 0.0

    def test_sums_to_one_per_destination_multihead(self):
        ei = torch.tensor([[0, 2, 3, 0, 1], [1, 1, 1, 2, 3]], dtype=torch.long)
        K = 4
        scores = torch.randn(5, K)
        attn = edge_softmax(scores, ei[1], num_nodes=4)
        sums = torch.zeros(4, K)
        sums.index_add_(0, ei[1], attn)
        assert torch.allclose(sums[1:], torch.ones(3, K), atol=1e-5)
        assert torch.allclose(sums[0], torch.zeros(K))

    def test_single_incoming_edge_gives_attention_one(self):
        ei = torch.tensor([[0], [1]], dtype=torch.long)
        scores = torch.randn(1)
        attn = edge_softmax(scores, ei[1], num_nodes=2)
        assert torch.allclose(attn, torch.ones(1), atol=1e-6)

    def test_numerical_stability_with_large_scores(self):
        target = torch.tensor([0, 0, 0], dtype=torch.long)
        scores = torch.tensor([1e3, 1e3, 1e3])
        attn = edge_softmax(scores, target, num_nodes=1)
        # Should be roughly uniform 1/3 each, no overflow
        assert torch.allclose(attn, torch.full((3,), 1.0 / 3), atol=1e-5)

    def test_empty_edges(self):
        scores = torch.empty(0)
        target = torch.empty(0, dtype=torch.long)
        out = edge_softmax(scores, target, num_nodes=4)
        assert out.numel() == 0

    def test_dtype_check(self):
        with pytest.raises(TypeError, match="torch.long"):
            edge_softmax(torch.randn(3), torch.zeros(3, dtype=torch.float), 4)

    def test_size_mismatch(self):
        with pytest.raises(ValueError, match="dim 0"):
            edge_softmax(torch.randn(3), torch.zeros(5, dtype=torch.long), 4)


class TestScatterHelpers:
    def test_scatter_sum_shape(self):
        x = torch.randn(5, 4, 3, 3)
        target = torch.tensor([0, 0, 1, 2, 1], dtype=torch.long)
        out = scatter_sum(x, target, num_nodes=3)
        assert out.shape == (3, 4, 3, 3)
        # destination 0 = x[0] + x[1]
        assert torch.allclose(out[0], x[0] + x[1])

    def test_scatter_mean_shape(self):
        x = torch.randn(5, 4, 3, 3)
        target = torch.tensor([0, 0, 1, 2, 1], dtype=torch.long)
        out = scatter_mean(x, target, num_nodes=3)
        assert out.shape == (3, 4, 3, 3)
        assert torch.allclose(out[0], (x[0] + x[1]) / 2)

    def test_scatter_max_isolated_zero(self):
        x = torch.randn(2, 3)
        target = torch.tensor([0, 1], dtype=torch.long)
        out = scatter_max(x, target, num_nodes=4)
        # destinations 2 and 3 receive nothing
        assert torch.all(out[2] == 0.0)
        assert torch.all(out[3] == 0.0)


# ──────────────────────────────────────────────────────────────────── #
# TensorGATLayer                                                         #
# ──────────────────────────────────────────────────────────────────── #

class TestTensorGATLayer:
    def test_forward_concat_heads(self):
        layer = TensorGATLayer(in_channels=C, out_channels=8, num_heads=4, concat_heads=True)
        out = layer(_spatial(), _ei())
        assert out.shape == (N, 8, H, W)
        assert torch.isfinite(out).all()

    def test_forward_average_heads(self):
        layer = TensorGATLayer(in_channels=C, out_channels=8, num_heads=4, concat_heads=False)
        out = layer(_spatial(), _ei())
        # When concat_heads=False, output is per-head channel count = out_channels.
        assert out.shape == (N, 8, H, W)

    def test_single_head(self):
        layer = TensorGATLayer(in_channels=C, out_channels=8, num_heads=1)
        out = layer(_spatial(), _ei())
        assert out.shape == (N, 8, H, W)

    def test_concat_divisibility_check(self):
        with pytest.raises(ValueError, match="divisible"):
            TensorGATLayer(in_channels=C, out_channels=7, num_heads=4, concat_heads=True)

    def test_invalid_num_heads(self):
        with pytest.raises(ValueError, match="num_heads"):
            TensorGATLayer(in_channels=C, out_channels=8, num_heads=0)

    def test_attention_sums_to_one_per_destination_per_head(self):
        """The principal correctness invariant for true GAT."""
        # Node 1 has three incoming edges; node 2 has one; node 0 has none.
        ei = torch.tensor([[0, 2, 3, 0], [1, 1, 1, 2]], dtype=torch.long)
        x = torch.randn(4, C, H, W)
        K = 3
        layer = TensorGATLayer(
            in_channels=C, out_channels=K * 5, num_heads=K, concat_heads=True,
            add_self_loops=False, attn_dropout=0.0,
        )
        layer.eval()  # disable any stochasticity
        _, attn = layer(x, ei, return_attention=True)
        assert attn.shape == (4, K)

        # Per-destination sum per head must be 1 for nodes that have incoming edges.
        sums = torch.zeros(4, K)
        sums.index_add_(0, ei[1], attn)
        for j in (1, 2):
            for h_idx in range(K):
                assert torch.allclose(
                    sums[j, h_idx], torch.tensor(1.0), atol=1e-5
                ), f"sum at dest {j} head {h_idx} = {sums[j, h_idx].item()}"
        # Node 0 has no incoming edges; sum stays zero.
        assert torch.allclose(sums[0], torch.zeros(K))

    def test_attention_with_self_loops_sums_to_one(self):
        ei = torch.tensor([[0, 2, 3], [1, 1, 1]], dtype=torch.long)
        layer = TensorGATLayer(
            in_channels=C, out_channels=4, num_heads=2,
            add_self_loops=True, attn_dropout=0.0,
        )
        layer.eval()
        x = torch.randn(4, C, H, W)
        _, attn = layer(x, ei, return_attention=True)
        # E_eff = 3 + 4 = 7
        assert attn.shape == (7, 2)
        # Now every node has ≥ 1 incoming edge (its self loop)
        N_t = 4
        # Re-derive the destination index used inside the layer (with loops).
        loop = torch.arange(N_t)
        dst_with_loops = torch.cat([ei[1], loop])
        sums = torch.zeros(N_t, 2)
        sums.index_add_(0, dst_with_loops, attn)
        assert torch.allclose(sums, torch.ones(N_t, 2), atol=1e-5)

    def test_isolated_node_outputs_zero_without_bias_or_residual(self):
        """No incoming edges → no attention → aggregated value is the zero tensor."""
        ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        x = torch.randn(N, C, H, W)
        layer = TensorGATLayer(
            in_channels=C, out_channels=8, num_heads=2,
            bias=False, residual=False, add_self_loops=False,
        )
        layer.eval()
        out = layer(x, ei)
        # Node index 5 has no incoming edge in this graph.
        assert torch.allclose(out[5], torch.zeros_like(out[5]), atol=1e-6)

    def test_self_loops_make_isolated_node_finite(self):
        ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        layer = TensorGATLayer(
            in_channels=C, out_channels=8, num_heads=2,
            bias=False, residual=False, add_self_loops=True,
        )
        out = layer(_spatial(), ei)
        # Every node now has at least one incoming edge (its self loop).
        assert torch.isfinite(out).all()
        assert not torch.allclose(out[5], torch.zeros_like(out[5]))

    def test_backward_input_and_params(self):
        x = _spatial().requires_grad_(True)
        layer = TensorGATLayer(in_channels=C, out_channels=8, num_heads=2)
        layer(x, _ei()).sum().backward()
        assert x.grad is not None
        for name, p in layer.named_parameters():
            assert p.grad is not None, f"no gradient on {name}"

    def test_residual_with_channel_change(self):
        layer = TensorGATLayer(
            in_channels=C, out_channels=8, num_heads=2, residual=True
        )
        out = layer(_spatial(), _ei())
        assert out.shape == (N, 8, H, W)

    def test_residual_passthrough_when_shapes_match(self):
        layer = TensorGATLayer(
            in_channels=C, out_channels=C, num_heads=2, residual=True,
        )
        out = layer(_spatial(), _ei())
        assert out.shape == (N, C, H, W)

    def test_unexpected_edge_features_raises(self):
        """Without use_edge_features, passing edge_features must raise ValueError."""
        layer = TensorGATLayer(in_channels=C, out_channels=8)
        with pytest.raises(ValueError, match="use_edge_features=False"):
            layer(_spatial(), _ei(), edge_features=torch.randn(N, 2))

    def test_spatial_edge_features_supported(self):
        """TensorGATLayer accepts 4-D spatial edge features (mean-pooled)."""
        layer = TensorGATLayer(
            in_channels=C, out_channels=8,
            use_edge_features=True, edge_dim=2,
        )
        # 4-D spatial edge tensor is mean-pooled to vector form before bias.
        out = layer(_spatial(), _ei(), edge_features=torch.randn(N, 2, H, W))
        assert out.shape == (N, 8, H, W)
        assert torch.isfinite(out).all()

    def test_volumetric_edge_features_rejected(self):
        """5-D volumetric edge features are explicitly rejected."""
        layer = TensorGATLayer(
            in_channels=C, out_channels=8,
            use_edge_features=True, edge_dim=2,
        )
        with pytest.raises(NotImplementedError, match="volumetric"):
            layer(_spatial(), _ei(), edge_features=torch.randn(N, 2, 2, H, W))

    def test_attn_dropout_in_train_mode(self):
        layer = TensorGATLayer(
            in_channels=C, out_channels=8, num_heads=2, attn_dropout=0.5
        )
        layer.train()
        out = layer(_spatial(), _ei())
        assert torch.isfinite(out).all()

    def test_invalid_edge_index_dtype(self):
        layer = TensorGATLayer(in_channels=C, out_channels=8)
        ei = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # float
        with pytest.raises(TypeError, match="torch.long"):
            layer(_spatial(), ei)

    def test_invalid_x_shape(self):
        layer = TensorGATLayer(in_channels=C, out_channels=8)
        with pytest.raises(ValueError, match=r"\[N, C, H, W\]"):
            layer(torch.randn(N, C, H), _ei())  # 3-D


# ──────────────────────────────────────────────────────────────────── #
# TensorGraphSAGELayer                                                  #
# ──────────────────────────────────────────────────────────────────── #

class TestTensorGraphSAGELayer:
    def test_forward_mean(self):
        layer = TensorGraphSAGELayer(in_channels=C, out_channels=8, aggr="mean")
        out = layer(_spatial(), _ei())
        assert out.shape == (N, 8, H, W)
        assert torch.isfinite(out).all()

    def test_forward_max(self):
        layer = TensorGraphSAGELayer(in_channels=C, out_channels=8, aggr="max")
        out = layer(_spatial(), _ei())
        assert out.shape == (N, 8, H, W)
        assert torch.isfinite(out).all()

    def test_invalid_aggr(self):
        with pytest.raises(ValueError, match="aggr"):
            TensorGraphSAGELayer(in_channels=C, out_channels=8, aggr="lstm")

    def test_isolated_node_uses_self_only(self):
        """Without incoming edges, output equals the self transform."""
        ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        x = torch.randn(N, C, H, W)
        layer = TensorGraphSAGELayer(
            in_channels=C, out_channels=8, aggr="mean", bias=False, residual=False,
        )
        layer.eval()
        out = layer(x, ei)
        # Node 5 has no incoming edges; W_neigh aggregate is zero.
        expected = layer.W_self(x[5:6]).squeeze(0)
        assert torch.allclose(out[5], expected, atol=1e-5)

    def test_backward(self):
        x = _spatial().requires_grad_(True)
        layer = TensorGraphSAGELayer(in_channels=C, out_channels=8, aggr="mean")
        layer(x, _ei()).sum().backward()
        assert x.grad is not None
        for p in layer.parameters():
            assert p.grad is not None

    def test_normalize_l2(self):
        layer = TensorGraphSAGELayer(
            in_channels=C, out_channels=8, normalize=True
        )
        out = layer(_spatial(), _ei())
        norms = out.pow(2).sum(dim=1).sqrt()
        # Per (n, h, w) the channel vector should have unit L2 norm.
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_with_edge_features(self):
        edge_dim = 2
        layer = TensorGraphSAGELayer(
            in_channels=C, out_channels=8,
            use_edge_features=True, edge_dim=edge_dim,
        )
        ef = torch.randn(N, edge_dim, H, W)  # cycle has N edges
        out = layer(_spatial(), _ei(), edge_features=ef)
        assert out.shape == (N, 8, H, W)

    def test_edge_features_shape_check(self):
        layer = TensorGraphSAGELayer(
            in_channels=C, out_channels=8, use_edge_features=True, edge_dim=2,
        )
        with pytest.raises(ValueError, match=r"edge_features"):
            layer(_spatial(), _ei(), edge_features=torch.randn(N, 3, H, W))  # wrong edge_dim

    def test_edge_features_required_when_enabled(self):
        layer = TensorGraphSAGELayer(
            in_channels=C, out_channels=8, use_edge_features=True, edge_dim=2,
        )
        with pytest.raises(ValueError, match="edge_features"):
            layer(_spatial(), _ei())

    def test_unexpected_edge_features(self):
        layer = TensorGraphSAGELayer(in_channels=C, out_channels=8)
        with pytest.raises(ValueError, match="edge_features"):
            layer(_spatial(), _ei(), edge_features=torch.randn(N, 2, H, W))

    def test_residual_proj_when_channels_differ(self):
        layer = TensorGraphSAGELayer(
            in_channels=C, out_channels=8, residual=True,
        )
        out = layer(_spatial(), _ei())
        assert out.shape == (N, 8, H, W)


# ──────────────────────────────────────────────────────────────────── #
# TensorGINLayer                                                         #
# ──────────────────────────────────────────────────────────────────── #

class TestTensorGINLayer:
    def test_forward_default(self):
        layer = TensorGINLayer(in_channels=C, out_channels=8)
        out = layer(_spatial(), _ei())
        assert out.shape == (N, 8, H, W)
        assert torch.isfinite(out).all()

    def test_eps_buffer_default(self):
        layer = TensorGINLayer(in_channels=C, out_channels=8, eps=0.5)
        param_names = {name for name, _ in layer.named_parameters()}
        assert "eps" not in param_names
        # buffer access
        assert float(layer.eps) == pytest.approx(0.5)

    def test_eps_learnable(self):
        layer = TensorGINLayer(in_channels=C, out_channels=8, eps=0.5, train_eps=True)
        param_names = {name for name, _ in layer.named_parameters()}
        assert "eps" in param_names
        # Backward must populate the eps gradient.
        layer(_spatial(), _ei()).sum().backward()
        assert layer.eps.grad is not None

    def test_backward_input_and_mlp(self):
        x = _spatial().requires_grad_(True)
        layer = TensorGINLayer(in_channels=C, out_channels=8, hidden_channels=12)
        layer(x, _ei()).sum().backward()
        assert x.grad is not None
        for p in layer.parameters():
            assert p.grad is not None

    def test_custom_mlp(self):
        custom = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=1),
        )
        layer = TensorGINLayer(in_channels=C, out_channels=8, mlp=custom)
        out = layer(_spatial(), _ei())
        assert out.shape == (N, 8, H, W)

    def test_with_edge_features(self):
        edge_dim = 2
        layer = TensorGINLayer(
            in_channels=C, out_channels=8,
            use_edge_features=True, edge_dim=edge_dim,
        )
        ef = torch.randn(N, edge_dim, H, W)
        out = layer(_spatial(), _ei(), edge_features=ef)
        assert out.shape == (N, 8, H, W)

    def test_edge_dim_equals_in_channels_uses_identity(self):
        layer = TensorGINLayer(
            in_channels=C, out_channels=8,
            use_edge_features=True, edge_dim=C,
        )
        assert isinstance(layer.edge_proj, nn.Identity)

    def test_edge_features_shape_check(self):
        layer = TensorGINLayer(
            in_channels=C, out_channels=8,
            use_edge_features=True, edge_dim=2,
        )
        with pytest.raises(ValueError):
            layer(_spatial(), _ei(), edge_features=torch.randn(N, 3, H, W))


# ──────────────────────────────────────────────────────────────────── #
# Custom subclass of TensorMessagePassingLayer                           #
# ──────────────────────────────────────────────────────────────────── #

class GatedConvCustom(TensorMessagePassingLayer):
    """Sigmoid-gated 1x1 conv message passing — used purely for the test."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__(
            in_shape=(c_in,), out_shape=(c_out,), aggr="mean",
        )
        self.W_g = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.W_v = nn.Conv2d(c_in, c_out, kernel_size=1)

    def message(self, src, dest, edge_attr):
        gate = torch.sigmoid(self.W_g(src + dest))
        return gate * self.W_v(src)

    def update(self, node_feature, aggregated_message):
        return aggregated_message  # no extra transform


class TestCustomSubclass:
    def test_forward_and_backward(self):
        layer = GatedConvCustom(c_in=C, c_out=8)
        x = _spatial().requires_grad_(True)
        out = layer(x, _ei())
        assert out.shape == (N, 8, H, W)
        out.sum().backward()
        assert x.grad is not None
        for p in layer.parameters():
            assert p.grad is not None


# ──────────────────────────────────────────────────────────────────── #
# Device tests                                                          #
# ──────────────────────────────────────────────────────────────────── #

@pytest.mark.cuda
class TestCUDA:
    def _device_test(self, layer, device="cuda"):
        x = _spatial(device=device).requires_grad_(True)
        ei = _ei(device=device)
        layer = layer.to(device)
        out = layer(x, ei)
        assert out.device.type == device
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert x.grad is not None

    def test_gat_cuda(self):
        self._device_test(
            TensorGATLayer(in_channels=C, out_channels=8, num_heads=2)
        )

    def test_sage_cuda_mean(self):
        self._device_test(
            TensorGraphSAGELayer(in_channels=C, out_channels=8, aggr="mean")
        )

    def test_sage_cuda_max(self):
        self._device_test(
            TensorGraphSAGELayer(in_channels=C, out_channels=8, aggr="max")
        )

    def test_gin_cuda(self):
        self._device_test(
            TensorGINLayer(in_channels=C, out_channels=8, train_eps=True)
        )


@pytest.mark.mps
class TestMPS:
    def _device_test(self, layer, device="mps"):
        x = _spatial(device=device)
        ei = _ei(device=device)
        layer = layer.to(device)
        out = layer(x, ei)
        assert out.device.type == device
        assert torch.isfinite(out).all()

    def test_gat_mps(self):
        self._device_test(
            TensorGATLayer(in_channels=C, out_channels=8, num_heads=2)
        )

    def test_sage_mps(self):
        self._device_test(
            TensorGraphSAGELayer(in_channels=C, out_channels=8, aggr="mean")
        )

    def test_gin_mps(self):
        self._device_test(TensorGINLayer(in_channels=C, out_channels=8))
