"""3-D volumetric node-feature support across all four message-passing layers.

Each layer is exercised on ``[N, C, D, H, W]`` node features in five modes:

1. plain (no edge fields),
2. ``edge_weight`` only,
3. vector ``edge_features`` only (where supported),
4. volumetric ``edge_features`` only (where supported),
5. ``edge_weight`` + volumetric ``edge_features`` together.

For every mode we check forward output shape, finite/non-zero values and
gradients (input + parameters), plus the structural invariants:

* edge-order invariance,
* node-permutation equivariance,
* isolated nodes finite,
* explicit ``NotImplementedError`` for the (2-D-rank, 5-D-edge) and
  (3-D-rank, 4-D-edge) mismatches on GAT.

A single ``GraphClassifier`` smoke test confirms that the 3-D path
end-to-end produces logits with shape ``[B, num_classes]``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tgraphx import Graph
from tgraphx.layers import (
    ConvMessagePassing,
    DeepCNNAggregator,
    TensorGATLayer,
    TensorGINLayer,
    TensorGraphSAGELayer,
)
from tgraphx.models import GraphClassifier


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

N, C, D, H, W = 6, 3, 3, 4, 4


def _x_3d(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(N, C, D, H, W)


def _ei_dense() -> torch.Tensor:
    """Dense edge set so every dst has multiple incoming edges (GAT softmax matters)."""
    return torch.tensor(
        [[0, 2, 3, 0, 1, 4, 5, 4],
         [1, 1, 1, 2, 3, 0, 4, 5]],
        dtype=torch.long,
    )


def _fast_agg():
    return {"num_layers": 1, "use_batchnorm": False, "dropout_prob": 0.0}


# =========================================================================== #
# Aggregator 3-D                                                                #
# =========================================================================== #

class TestDeepCNNAggregator3D:
    def test_forward_3d_shape_preserved(self):
        agg = DeepCNNAggregator(C, C, num_layers=2, use_batchnorm=False,
                                dropout_prob=0.0, spatial_rank=3)
        x = _x_3d()
        out = agg(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_forward_2d_unchanged(self):
        """2-D default path must still work and stay shape-preserving."""
        agg = DeepCNNAggregator(C, C, num_layers=2, use_batchnorm=False,
                                dropout_prob=0.0, spatial_rank=2)
        x = torch.randn(N, C, H, W)
        out = agg(x)
        assert out.shape == x.shape

    def test_invalid_rank_raises(self):
        with pytest.raises(ValueError, match="spatial_rank"):
            DeepCNNAggregator(C, C, spatial_rank=4)


# =========================================================================== #
# ConvMessagePassing 3-D                                                       #
# =========================================================================== #

class TestConvMessagePassing3D:
    def _layer(self, *, use_edge_features: bool = False):
        return ConvMessagePassing(
            (C, D, H, W), (8, D, H, W),
            use_edge_features=use_edge_features,
            aggregator_params=_fast_agg(),
        )

    def test_forward_backward_plain(self):
        layer = self._layer()
        x = _x_3d().requires_grad_(True)
        ei = _ei_dense()
        out = layer(x, ei)
        assert out.shape == (N, 8, D, H, W)
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert torch.isfinite(x.grad).all()
        for p in layer.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()

    def test_edge_weight_changes_output(self):
        layer = self._layer()
        layer.eval()
        x = _x_3d()
        ei = _ei_dense()
        E = ei.size(1)
        out_no = layer(x, ei)
        out_w = layer(x, ei, edge_weight=torch.linspace(0.5, 2.0, E))
        assert (out_w - out_no).abs().max().item() > 1e-4
        # edge_weight=ones round-trips
        out_ones = layer(x, ei, edge_weight=torch.ones(E))
        assert torch.allclose(out_no, out_ones, atol=1e-5)

    def test_volumetric_edge_features(self):
        """ConvMessagePassing requires edge channel = node channel; here both are C."""
        layer = self._layer(use_edge_features=True)
        x = _x_3d()
        ei = _ei_dense()
        E = ei.size(1)
        ef = torch.randn(E, C, D, H, W, requires_grad=True)
        out = layer(x, ei, edge_features=ef)
        assert out.shape == (N, 8, D, H, W)
        out.sum().backward()
        assert torch.isfinite(ef.grad).all()
        assert ef.grad.norm().item() > 0.0

    def test_volumetric_edge_channel_mismatch_clear_error(self):
        layer = self._layer(use_edge_features=True)
        x = _x_3d()
        ei = _ei_dense()
        with pytest.raises(ValueError, match="channel count"):
            layer(x, ei, edge_features=torch.randn(ei.size(1), C + 1, D, H, W))

    def test_edge_rank_mismatch_clear_error(self):
        layer = self._layer(use_edge_features=True)
        x = _x_3d()
        ei = _ei_dense()
        with pytest.raises(ValueError, match="same rank"):
            # 4-D edges into a 3-D-shape ConvMessagePassing.
            layer(x, ei, edge_features=torch.randn(ei.size(1), C, H, W))

    def test_edge_weight_plus_volumetric_edges(self):
        layer = self._layer(use_edge_features=True)
        x = _x_3d()
        ei = _ei_dense()
        E = ei.size(1)
        ef = torch.randn(E, C, D, H, W)
        out = layer(x, ei, edge_features=ef, edge_weight=torch.linspace(0.4, 1.6, E))
        assert out.shape == (N, 8, D, H, W)
        assert torch.isfinite(out).all()


# =========================================================================== #
# TensorGATLayer 3-D                                                            #
# =========================================================================== #

class TestTensorGATLayer3D:
    EDGE_DIM = 3

    def _layer(self, *, use_edge_features: bool = False):
        return TensorGATLayer(
            in_channels=C, out_channels=8, num_heads=2,
            spatial_rank=3,
            use_edge_features=use_edge_features,
            edge_dim=self.EDGE_DIM if use_edge_features else None,
            attn_dropout=0.0, residual=False, bias=False,
        )

    def test_forward_backward_plain(self):
        layer = self._layer()
        x = _x_3d().requires_grad_(True)
        ei = _ei_dense()
        out = layer(x, ei)
        assert out.shape == (N, 8, D, H, W)
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert torch.isfinite(x.grad).all()
        for name, p in layer.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name

    def test_attention_sums_to_one_3d(self):
        """The softmax invariant must hold in 3-D too."""
        ei = torch.tensor([[0, 2, 3, 0], [1, 1, 1, 2]], dtype=torch.long)
        x = torch.randn(4, C, D, H, W)
        K = 3
        layer = TensorGATLayer(
            in_channels=C, out_channels=K * 5, num_heads=K,
            spatial_rank=3, attn_dropout=0.0, add_self_loops=False,
        )
        layer.eval()
        _, attn = layer(x, ei, return_attention=True)
        assert attn.shape == (4, K)
        sums = torch.zeros(4, K)
        sums.index_add_(0, ei[1], attn)
        for j in (1, 2):
            assert torch.allclose(sums[j], torch.ones(K), atol=1e-5)

    def test_edge_weight_changes_output(self):
        layer = self._layer()
        layer.eval()
        x = _x_3d()
        ei = _ei_dense()
        E = ei.size(1)
        out_no = layer(x, ei)
        out_w = layer(x, ei, edge_weight=torch.linspace(0.5, 2.0, E))
        assert (out_w - out_no).abs().max().item() > 1e-4

    def test_vector_edge_features_3d(self):
        layer = self._layer(use_edge_features=True)
        x = _x_3d()
        ei = _ei_dense()
        E = ei.size(1)
        ef = torch.randn(E, self.EDGE_DIM, requires_grad=True)
        out = layer(x, ei, edge_features=ef)
        assert out.shape == (N, 8, D, H, W)
        out.sum().backward()
        assert torch.isfinite(ef.grad).all()
        assert ef.grad.norm().item() > 0.0

    def test_volumetric_edge_features_3d_pooled(self):
        layer = self._layer(use_edge_features=True)
        x = _x_3d()
        ei = _ei_dense()
        E = ei.size(1)
        ef = torch.randn(E, self.EDGE_DIM, D, H, W, requires_grad=True)
        out = layer(x, ei, edge_features=ef)
        assert out.shape == (N, 8, D, H, W)
        out.sum().backward()
        assert torch.isfinite(ef.grad).all()
        assert ef.grad.norm().item() > 0.0
        # mean-pool semantics: a constant-pixel volumetric tensor matches the
        # vector with that constant.
        layer.eval()
        const_vec = torch.randn(E, self.EDGE_DIM)
        const_vol = const_vec.view(E, self.EDGE_DIM, 1, 1, 1).expand(E, self.EDGE_DIM, D, H, W).contiguous()
        assert torch.allclose(
            layer(x, ei, edge_features=const_vec),
            layer(x, ei, edge_features=const_vol),
            atol=1e-5,
        )

    def test_2d_edges_into_3d_layer_raises(self):
        layer = self._layer(use_edge_features=True)
        x = _x_3d()
        ei = _ei_dense()
        with pytest.raises(NotImplementedError, match="2-D-spatial"):
            layer(x, ei, edge_features=torch.randn(ei.size(1), self.EDGE_DIM, H, W))

    def test_5d_edges_into_2d_layer_raises(self):
        """Mirror of the 3-D-into-2-D check (regression-pinned)."""
        layer = TensorGATLayer(
            in_channels=C, out_channels=8, num_heads=2,
            use_edge_features=True, edge_dim=self.EDGE_DIM,
            spatial_rank=2,
        )
        x = torch.randn(N, C, H, W)
        ei = _ei_dense()
        with pytest.raises(NotImplementedError, match="volumetric"):
            layer(x, ei, edge_features=torch.randn(ei.size(1), self.EDGE_DIM, D, H, W))

    def test_edge_weight_plus_volumetric_edges(self):
        layer = self._layer(use_edge_features=True)
        x = _x_3d()
        ei = _ei_dense()
        E = ei.size(1)
        ef = torch.randn(E, self.EDGE_DIM, D, H, W)
        w = torch.linspace(0.4, 1.6, E)
        out = layer(x, ei, edge_features=ef, edge_weight=w)
        assert out.shape == (N, 8, D, H, W)
        assert torch.isfinite(out).all()

    def test_self_loops_finite_3d(self):
        layer = TensorGATLayer(
            in_channels=C, out_channels=8, num_heads=2,
            spatial_rank=3, add_self_loops=True,
        )
        x = _x_3d()
        ei = _ei_dense()
        out = layer(x, ei)
        assert torch.isfinite(out).all()


# =========================================================================== #
# TensorGraphSAGELayer 3-D                                                      #
# =========================================================================== #

class TestTensorGraphSAGELayer3D:
    EDGE_DIM = 3

    def _layer(self, *, use_edge_features=False, kind="spatial", aggr="mean"):
        return TensorGraphSAGELayer(
            in_channels=C, out_channels=8, aggr=aggr,
            spatial_rank=3,
            use_edge_features=use_edge_features,
            edge_dim=self.EDGE_DIM if use_edge_features else None,
            edge_features_kind=kind,
        )

    def test_forward_backward_mean(self):
        layer = self._layer(aggr="mean")
        x = _x_3d().requires_grad_(True)
        ei = _ei_dense()
        out = layer(x, ei)
        assert out.shape == (N, 8, D, H, W)
        out.sum().backward()
        assert torch.isfinite(x.grad).all()
        for p in layer.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()

    def test_forward_backward_max(self):
        layer = self._layer(aggr="max")
        x = _x_3d().requires_grad_(True)
        ei = _ei_dense()
        out = layer(x, ei)
        assert out.shape == (N, 8, D, H, W)
        out.sum().backward()
        assert torch.isfinite(x.grad).all()

    def test_edge_weight_changes_output(self):
        layer = self._layer()
        layer.eval()
        x = _x_3d()
        ei = _ei_dense()
        E = ei.size(1)
        out_no = layer(x, ei)
        out_w = layer(x, ei, edge_weight=torch.linspace(0.5, 1.5, E))
        assert (out_w - out_no).abs().max().item() > 1e-4

    def test_volumetric_edge_features(self):
        layer = self._layer(use_edge_features=True, kind="spatial")
        x = _x_3d()
        ei = _ei_dense()
        E = ei.size(1)
        ef = torch.randn(E, self.EDGE_DIM, D, H, W, requires_grad=True)
        out = layer(x, ei, edge_features=ef)
        assert out.shape == (N, 8, D, H, W)
        out.sum().backward()
        assert torch.isfinite(ef.grad).all()
        assert ef.grad.norm().item() > 0.0

    def test_vector_edge_features(self):
        layer = self._layer(use_edge_features=True, kind="vector")
        x = _x_3d()
        ei = _ei_dense()
        E = ei.size(1)
        ef = torch.randn(E, self.EDGE_DIM)
        out = layer(x, ei, edge_features=ef)
        assert out.shape == (N, 8, D, H, W)

    def test_isolated_node_uses_self_only(self):
        """Node 5 is isolated in this graph; output equals W_self for that node."""
        ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        layer = TensorGraphSAGELayer(
            in_channels=C, out_channels=8, aggr="mean",
            spatial_rank=3, bias=False, residual=False,
        )
        layer.eval()
        x = _x_3d()
        out = layer(x, ei)
        assert torch.isfinite(out).all()
        expected = layer.W_self(x[5:6]).squeeze(0)
        assert torch.allclose(out[5], expected, atol=1e-5)

    def test_edge_weight_plus_volumetric_edges(self):
        layer = self._layer(use_edge_features=True, kind="spatial")
        x = _x_3d()
        ei = _ei_dense()
        E = ei.size(1)
        ef = torch.randn(E, self.EDGE_DIM, D, H, W)
        w = torch.linspace(0.4, 1.6, E)
        out = layer(x, ei, edge_features=ef, edge_weight=w)
        assert out.shape == (N, 8, D, H, W)


# =========================================================================== #
# TensorGINLayer 3-D                                                            #
# =========================================================================== #

class TestTensorGINLayer3D:
    EDGE_DIM = 3

    def _layer(self, *, use_edge_features=False, kind="spatial"):
        return TensorGINLayer(
            in_channels=C, out_channels=8,
            spatial_rank=3, train_eps=True,
            use_edge_features=use_edge_features,
            edge_dim=self.EDGE_DIM if use_edge_features else None,
            edge_features_kind=kind,
        )

    def test_forward_backward(self):
        layer = self._layer()
        x = _x_3d().requires_grad_(True)
        ei = _ei_dense()
        out = layer(x, ei)
        assert out.shape == (N, 8, D, H, W)
        out.sum().backward()
        assert torch.isfinite(x.grad).all()
        for p in layer.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()

    def test_eps_is_3d_aware(self):
        """eps is a scalar; broadcasts across [N, C, D, H, W] cleanly."""
        layer = self._layer()
        layer.eval()
        x = _x_3d()
        ei = _ei_dense()
        out = layer(x, ei)
        assert torch.isfinite(out).all()

    def test_edge_weight_changes_output(self):
        layer = self._layer()
        layer.eval()
        x = _x_3d()
        ei = _ei_dense()
        E = ei.size(1)
        out_no = layer(x, ei)
        out_w = layer(x, ei, edge_weight=torch.linspace(0.5, 1.5, E))
        assert (out_w - out_no).abs().max().item() > 1e-4

    def test_volumetric_edge_features(self):
        layer = self._layer(use_edge_features=True, kind="spatial")
        x = _x_3d()
        ei = _ei_dense()
        E = ei.size(1)
        ef = torch.randn(E, self.EDGE_DIM, D, H, W, requires_grad=True)
        out = layer(x, ei, edge_features=ef)
        assert out.shape == (N, 8, D, H, W)
        out.sum().backward()
        assert torch.isfinite(ef.grad).all()
        assert ef.grad.norm().item() > 0.0
        # When edge_dim == in_channels the projection is Identity.
        layer_id = TensorGINLayer(
            in_channels=C, out_channels=8, spatial_rank=3,
            use_edge_features=True, edge_dim=C, edge_features_kind="spatial",
        )
        assert isinstance(layer_id.edge_proj, nn.Identity)

    def test_vector_edge_features(self):
        layer = self._layer(use_edge_features=True, kind="vector")
        x = _x_3d()
        ei = _ei_dense()
        E = ei.size(1)
        ef = torch.randn(E, self.EDGE_DIM)
        out = layer(x, ei, edge_features=ef)
        assert out.shape == (N, 8, D, H, W)

    def test_edge_weight_plus_volumetric_edges(self):
        layer = self._layer(use_edge_features=True, kind="spatial")
        x = _x_3d()
        ei = _ei_dense()
        E = ei.size(1)
        ef = torch.randn(E, self.EDGE_DIM, D, H, W)
        w = torch.linspace(0.4, 1.6, E)
        out = layer(x, ei, edge_features=ef, edge_weight=w)
        assert out.shape == (N, 8, D, H, W)


# =========================================================================== #
# Edge-order invariance / node-permutation equivariance in 3-D                  #
# =========================================================================== #

def _layers_3d():
    EDGE_DIM = 3
    return [
        ("conv", lambda: ConvMessagePassing(
            (C, D, H, W), (8, D, H, W),
            aggr="mean", aggregator_params=_fast_agg(),
        ), False, None),
        ("gat", lambda: TensorGATLayer(
            in_channels=C, out_channels=8, num_heads=2, spatial_rank=3,
            use_edge_features=True, edge_dim=EDGE_DIM,
            attn_dropout=0.0, residual=False, bias=False,
        ), True, "spatial-pooled"),  # edges [E, edge_dim, D, H, W]
        ("sage", lambda: TensorGraphSAGELayer(
            in_channels=C, out_channels=8, aggr="mean", spatial_rank=3,
            use_edge_features=True, edge_dim=EDGE_DIM, edge_features_kind="spatial",
            bias=False,
        ), True, "spatial"),
        ("gin", lambda: TensorGINLayer(
            in_channels=C, out_channels=8, spatial_rank=3,
            use_edge_features=True, edge_dim=EDGE_DIM, edge_features_kind="spatial",
        ), True, "spatial"),
    ]


def _make_edge_features(name, ef_kind, E, edge_dim):
    if not ef_kind:
        return None
    if name == "conv":
        return torch.randn(E, C, D, H, W)
    return torch.randn(E, edge_dim, D, H, W)


class TestEdgeOrderInvariance3D:
    @pytest.mark.parametrize("name,factory,use_ef,kind", _layers_3d())
    def test_invariance(self, name, factory, use_ef, kind):
        torch.manual_seed(3)
        layer = factory()
        layer.eval()
        x = _x_3d()
        ei = _ei_dense()
        E = ei.size(1)
        ef = _make_edge_features(name, kind, E, edge_dim=3) if use_ef else None
        w = torch.linspace(0.5, 1.5, E)
        perm = torch.randperm(E)
        kwargs_a = {"edge_weight": w}
        kwargs_b = {"edge_weight": w[perm]}
        if use_ef:
            kwargs_a["edge_features"] = ef
            kwargs_b["edge_features"] = ef[perm]
        out_a = layer(x, ei, **kwargs_a)
        out_b = layer(x, ei[:, perm], **kwargs_b)
        assert torch.allclose(out_a, out_b, atol=1e-5), \
            f"{name}: edge-order invariance broken in 3-D"


class TestNodePermutationEquivariance3D:
    @pytest.mark.parametrize("name,factory,use_ef,kind", _layers_3d())
    def test_equivariance(self, name, factory, use_ef, kind):
        torch.manual_seed(11)
        layer = factory()
        layer.eval()
        x = _x_3d()
        ei = _ei_dense()
        E = ei.size(1)
        ef = _make_edge_features(name, kind, E, edge_dim=3) if use_ef else None
        w = torch.linspace(0.5, 1.5, E)
        kwargs = {"edge_weight": w}
        if use_ef:
            kwargs["edge_features"] = ef

        out_orig = layer(x, ei, **kwargs)

        node_perm = torch.randperm(N)
        inv_perm = torch.argsort(node_perm)
        x_perm = x[node_perm]
        ei_perm = inv_perm[ei]
        out_perm = layer(x_perm, ei_perm, **kwargs)

        assert torch.allclose(out_orig, out_perm[inv_perm], atol=1e-5), \
            f"{name}: node-permutation equivariance broken in 3-D"


# =========================================================================== #
# Isolated nodes finite                                                        #
# =========================================================================== #

class TestIsolatedNodesFinite3D:
    @pytest.mark.parametrize("name,factory,_use_ef,_kind", _layers_3d())
    def test_isolated_finite(self, name, factory, _use_ef, _kind):
        # Cycle 0->1->2->3->4 (no edge into node 5).
        ei = torch.tensor(
            [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long
        )
        layer = factory()
        layer.eval()
        x = _x_3d()
        E = ei.size(1)
        kwargs = {}
        if _use_ef:
            kwargs["edge_features"] = _make_edge_features(name, _kind, E, edge_dim=3)
        kwargs["edge_weight"] = torch.full((E,), 0.7)
        out = layer(x, ei, **kwargs)
        assert torch.isfinite(out).all()


# =========================================================================== #
# Graph object integration                                                     #
# =========================================================================== #

class TestGraphIntegration3D:
    EDGE_DIM = 3

    def test_graph_object_3d_through_gat(self):
        x = _x_3d()
        ei = _ei_dense()
        E = ei.size(1)
        ef = torch.randn(E, self.EDGE_DIM, D, H, W)
        w = torch.linspace(0.5, 1.5, E)
        g = Graph(x, ei, edge_weight=w, edge_features=ef)
        assert g.feature_shape == (C, D, H, W)
        assert g.edge_feature_shape == (self.EDGE_DIM, D, H, W)

        layer = TensorGATLayer(
            in_channels=C, out_channels=8, num_heads=2, spatial_rank=3,
            use_edge_features=True, edge_dim=self.EDGE_DIM,
        )
        layer.eval()
        out = layer(
            g.node_features, g.edge_index,
            edge_features=g.edge_features, edge_weight=g.edge_weight,
        )
        assert out.shape == (g.num_nodes, 8, D, H, W)


# =========================================================================== #
# GraphClassifier 3-D smoke test                                                #
# =========================================================================== #

class TestGraphClassifier3D:
    def test_3d_smoke(self):
        torch.manual_seed(0)
        # Two graphs each with N nodes, batched manually.
        x1 = _x_3d(seed=0)
        x2 = _x_3d(seed=1)
        x = torch.cat([x1, x2], dim=0)
        ei1 = _ei_dense()
        ei2 = _ei_dense() + N
        ei = torch.cat([ei1, ei2], dim=1)
        batch = torch.cat([torch.zeros(N, dtype=torch.long),
                           torch.ones(N, dtype=torch.long)])
        model = GraphClassifier(
            in_shape=(C, D, H, W),
            hidden_shape=(8, D, H, W),
            num_classes=3,
            num_layers=2,
            aggr="sum",
            pooling="mean",
        )
        logits = model(x, ei, batch=batch)
        assert logits.shape == (2, 3)
        assert torch.isfinite(logits).all()


# =========================================================================== #
# CUDA optional mirror                                                         #
# =========================================================================== #

@pytest.mark.cuda
class TestCUDA3D:
    EDGE_DIM = 3

    def test_gat_3d_cuda(self):
        layer = TensorGATLayer(
            in_channels=C, out_channels=8, num_heads=2, spatial_rank=3,
            use_edge_features=True, edge_dim=self.EDGE_DIM,
        ).cuda()
        x = _x_3d().cuda().requires_grad_(True)
        ei = _ei_dense().cuda()
        E = ei.size(1)
        ef = torch.randn(E, self.EDGE_DIM, D, H, W, device="cuda", requires_grad=True)
        w = torch.linspace(0.4, 1.6, E, device="cuda", requires_grad=True)
        out = layer(x, ei, edge_features=ef, edge_weight=w)
        out.sum().backward()
        assert out.device.type == "cuda"
        for p in (x, ef, w):
            assert torch.isfinite(p.grad).all()

    def test_sage_3d_cuda(self):
        layer = TensorGraphSAGELayer(
            in_channels=C, out_channels=8, spatial_rank=3,
            use_edge_features=True, edge_dim=self.EDGE_DIM,
            edge_features_kind="spatial",
        ).cuda()
        x = _x_3d().cuda()
        ei = _ei_dense().cuda()
        ef = torch.randn(ei.size(1), self.EDGE_DIM, D, H, W, device="cuda")
        out = layer(x, ei, edge_features=ef)
        assert out.shape == (N, 8, D, H, W)
        assert torch.isfinite(out).all()

    def test_gin_3d_cuda(self):
        layer = TensorGINLayer(
            in_channels=C, out_channels=8, spatial_rank=3,
            use_edge_features=True, edge_dim=self.EDGE_DIM,
            edge_features_kind="spatial",
        ).cuda()
        x = _x_3d().cuda()
        ei = _ei_dense().cuda()
        ef = torch.randn(ei.size(1), self.EDGE_DIM, D, H, W, device="cuda")
        out = layer(x, ei, edge_features=ef)
        assert out.shape == (N, 8, D, H, W)

    def test_conv_3d_cuda(self):
        layer = ConvMessagePassing(
            (C, D, H, W), (8, D, H, W), aggregator_params=_fast_agg(),
        ).cuda()
        x = _x_3d().cuda()
        ei = _ei_dense().cuda()
        out = layer(x, ei)
        assert out.shape == (N, 8, D, H, W)
