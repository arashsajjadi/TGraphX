"""Edge-weight and spatial-GAT-edge tests.

Covers the new layer-side functionality added in this phase:

* Spatial edge features ``[E, C_e, H, W]`` for ``TensorGATLayer`` (mean-pooled
  to a vector and projected to a per-(edge, head) attention bias).
* Per-edge ``edge_weight`` (``[E]``) scaling messages on every layer family:
  ``ConvMessagePassing``, ``TensorGATLayer``, ``TensorGraphSAGELayer``,
  ``TensorGINLayer``.

For each, the file checks:

1. *output changes* when the field is varied (the layer actually uses it),
2. *gradient flow* on both projection parameters and the input tensor,
3. *invariants*: GAT softmax still sums to 1 per (destination, head),
4. *equivariance*: output is invariant to edge-order permutation and
   equivariant to node permutation, even with spatial edges and weights,
5. *isolation*: nodes with no incoming edges still produce finite output,
6. *Graph integration*: a ``Graph`` carrying both ``edge_weight`` and
   ``edge_features`` works through the layer call site.

Volumetric (5-D) edge features and 3-D node features are rejected
explicitly — the corresponding tests pin those rejection paths.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tgraphx import Graph
from tgraphx.layers import (
    ConvMessagePassing,
    TensorGATLayer,
    TensorGINLayer,
    TensorGraphSAGELayer,
)


# --------------------------------------------------------------------------- #
# Shared graph fixtures                                                        #
# --------------------------------------------------------------------------- #

N, C, H, W = 6, 4, 4, 4


def _ei_dense(seed: int = 0) -> torch.Tensor:
    """Dense graph: every node has multiple incoming edges so attention bias
    actually matters."""
    return torch.tensor(
        [[0, 2, 3, 0, 1, 4, 5, 4],
         [1, 1, 1, 2, 3, 0, 4, 5]],
        dtype=torch.long,
    )


def _spatial(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(N, C, H, W)


def _fast_agg():
    return {"num_layers": 1, "use_batchnorm": False, "dropout_prob": 0.0}


# =========================================================================== #
# Spatial edge features for TensorGATLayer                                     #
# =========================================================================== #

class TestGATSpatialEdgeFeatures:
    EDGE_DIM = 3

    def _layer(self, num_heads: int = 2):
        return TensorGATLayer(
            in_channels=C, out_channels=8, num_heads=num_heads,
            use_edge_features=True, edge_dim=self.EDGE_DIM,
            attn_dropout=0.0, residual=False, bias=False,
        )

    def test_forward_shape_4d_edges(self):
        layer = self._layer()
        x = _spatial()
        ei = _ei_dense()
        ef = torch.randn(ei.size(1), self.EDGE_DIM, H, W)
        out = layer(x, ei, edge_features=ef)
        assert out.shape == (N, 8, H, W)
        assert torch.isfinite(out).all()

    def test_forward_shape_4d_edges_unequal_spatial(self):
        """Spatial dims of edges may differ from x's H, W (mean-pool collapses)."""
        layer = self._layer()
        x = _spatial()
        ei = _ei_dense()
        ef = torch.randn(ei.size(1), self.EDGE_DIM, 7, 5)  # not (H, W)
        out = layer(x, ei, edge_features=ef)
        assert out.shape == (N, 8, H, W)

    def test_output_changes_when_spatial_edges_zeroed(self):
        layer = self._layer()
        layer.eval()
        x = _spatial()
        ei = _ei_dense()
        ef = torch.randn(ei.size(1), self.EDGE_DIM, H, W)
        out_real = layer(x, ei, edge_features=ef)
        out_zero = layer(x, ei, edge_features=torch.zeros_like(ef))
        diff = (out_real - out_zero).abs().max().item()
        assert diff > 1e-4, f"spatial edge bias had no effect; max abs diff = {diff:.2e}"

    def test_pooled_4d_matches_2d_vector(self):
        """A 4-D edge tensor of constant pixels equals the 2-D vector with that mean."""
        layer = self._layer()
        layer.eval()
        x = _spatial()
        ei = _ei_dense()
        E = ei.size(1)
        torch.manual_seed(7)
        vec = torch.randn(E, self.EDGE_DIM)
        spatial = vec.view(E, self.EDGE_DIM, 1, 1).expand(E, self.EDGE_DIM, H, W).contiguous()
        out_vec = layer(x, ei, edge_features=vec)
        out_spat = layer(x, ei, edge_features=spatial)
        assert torch.allclose(out_vec, out_spat, atol=1e-5)

    def test_edge_projection_param_grad(self):
        layer = self._layer()
        x = _spatial()
        ei = _ei_dense()
        ef = torch.randn(ei.size(1), self.EDGE_DIM, H, W)
        layer(x, ei, edge_features=ef).sum().backward()
        wgrad = layer.edge_bias_proj.weight.grad
        assert wgrad is not None
        assert torch.isfinite(wgrad).all()
        assert wgrad.norm().item() > 0.0

    def test_edge_features_input_grad(self):
        layer = self._layer()
        x = _spatial()
        ei = _ei_dense()
        ef = torch.randn(ei.size(1), self.EDGE_DIM, H, W, requires_grad=True)
        layer(x, ei, edge_features=ef).sum().backward()
        assert ef.grad is not None
        assert torch.isfinite(ef.grad).all()
        assert ef.grad.norm().item() > 0.0

    def test_attention_sums_to_one_with_spatial_bias(self):
        """The softmax invariant holds even with spatial edge bias."""
        # Three incoming edges to node 1, one to node 2.
        ei = torch.tensor([[0, 2, 3, 0], [1, 1, 1, 2]], dtype=torch.long)
        x = torch.randn(4, C, H, W)
        K = 3
        layer = TensorGATLayer(
            in_channels=C, out_channels=K * 5, num_heads=K, concat_heads=True,
            add_self_loops=False, attn_dropout=0.0,
            use_edge_features=True, edge_dim=self.EDGE_DIM,
        )
        layer.eval()
        ef = torch.randn(ei.size(1), self.EDGE_DIM, H, W)
        _, attn = layer(x, ei, edge_features=ef, return_attention=True)
        sums = torch.zeros(4, K)
        sums.index_add_(0, ei[1], attn)
        for j in (1, 2):
            assert torch.allclose(sums[j], torch.ones(K), atol=1e-5)

    def test_volumetric_edges_rejected(self):
        layer = self._layer()
        x = _spatial()
        ei = _ei_dense()
        with pytest.raises(NotImplementedError, match="volumetric"):
            layer(x, ei, edge_features=torch.randn(ei.size(1), self.EDGE_DIM, 2, H, W))

    def test_invalid_3d_edge_shape_clear_error(self):
        layer = self._layer()
        x = _spatial()
        ei = _ei_dense()
        # 3-D is neither vector nor spatial — descriptive error.
        with pytest.raises(ValueError, match=r"\[E, edge_dim\]"):
            layer(x, ei, edge_features=torch.randn(ei.size(1), self.EDGE_DIM, H))

    def test_wrong_edge_dim_in_4d_raises(self):
        layer = self._layer()
        x = _spatial()
        ei = _ei_dense()
        with pytest.raises(ValueError, match="edge_dim"):
            layer(x, ei, edge_features=torch.randn(ei.size(1), self.EDGE_DIM + 1, H, W))


# =========================================================================== #
# edge_weight: Conv / GAT / SAGE / GIN                                         #
# =========================================================================== #

def _layer_factories():
    """Factories that produce each layer type with no edge features.

    Each returns a freshly-seeded layer so tests are deterministic.
    """
    def conv():
        torch.manual_seed(0)
        return ConvMessagePassing((C, H, W), (8, H, W), aggregator_params=_fast_agg())

    def gat():
        torch.manual_seed(0)
        return TensorGATLayer(in_channels=C, out_channels=8, num_heads=2, bias=False)

    def sage():
        torch.manual_seed(0)
        return TensorGraphSAGELayer(in_channels=C, out_channels=8, aggr="mean", bias=False)

    def gin():
        torch.manual_seed(0)
        return TensorGINLayer(in_channels=C, out_channels=8)

    return [("conv", conv), ("gat", gat), ("sage", sage), ("gin", gin)]


class TestEdgeWeightChangesOutput:
    """Output must depend on edge_weight in every layer."""

    @pytest.mark.parametrize("name,factory", _layer_factories())
    def test_weight_affects_output(self, name, factory):
        layer = factory()
        layer.eval()
        x = _spatial()
        ei = _ei_dense()
        E = ei.size(1)
        w = torch.linspace(0.5, 2.0, steps=E)
        out_w = layer(x, ei, edge_weight=w)
        out_no = layer(x, ei)
        diff = (out_w - out_no).abs().max().item()
        assert diff > 1e-4, f"{name}: edge_weight had no effect (max |diff|={diff:.2e})"

    @pytest.mark.parametrize("name,factory", _layer_factories())
    def test_weight_ones_matches_no_weight(self, name, factory):
        """edge_weight=1 must reproduce the no-weight behaviour exactly."""
        layer = factory()
        layer.eval()
        x = _spatial()
        ei = _ei_dense()
        E = ei.size(1)
        out_ones = layer(x, ei, edge_weight=torch.ones(E))
        out_no = layer(x, ei)
        # Conv path goes through aggregator with mean, so the messages must
        # match before update. In all cases the underlying numerics differ
        # only by a scaling factor of 1.
        assert torch.allclose(out_ones, out_no, atol=1e-5), \
            f"{name}: edge_weight=1 diverges from no-weight (max |diff|={(out_ones-out_no).abs().max():.2e})"

    def test_gat_zero_weight_kills_messages(self):
        """All-zero edge_weight + no bias/residual => output is the GAT bias only (zero)."""
        layer = TensorGATLayer(
            in_channels=C, out_channels=8, num_heads=2,
            bias=False, residual=False, add_self_loops=False,
        )
        layer.eval()
        x = _spatial()
        ei = _ei_dense()
        E = ei.size(1)
        out = layer(x, ei, edge_weight=torch.zeros(E))
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


class TestEdgeWeightGradient:
    @pytest.mark.parametrize("name,factory", _layer_factories())
    def test_weight_input_grad(self, name, factory):
        layer = factory()
        x = _spatial().requires_grad_(True)
        ei = _ei_dense()
        E = ei.size(1)
        w = torch.linspace(0.5, 2.0, steps=E).requires_grad_(True)
        out = layer(x, ei, edge_weight=w)
        out.sum().backward()
        assert w.grad is not None
        assert torch.isfinite(w.grad).all()
        assert w.grad.norm().item() > 0.0, f"{name}: edge_weight got zero gradient"
        assert torch.isfinite(x.grad).all()


class TestEdgeWeightInvalidShape:
    @pytest.mark.parametrize("name,factory", _layer_factories())
    def test_2d_weight_rejected(self, name, factory):
        layer = factory()
        x = _spatial()
        ei = _ei_dense()
        with pytest.raises(ValueError, match="1-D"):
            layer(x, ei, edge_weight=torch.randn(ei.size(1), 2))

    @pytest.mark.parametrize("name,factory", _layer_factories())
    def test_wrong_length_rejected(self, name, factory):
        layer = factory()
        x = _spatial()
        ei = _ei_dense()
        with pytest.raises(ValueError, match="edge_weight"):
            layer(x, ei, edge_weight=torch.randn(ei.size(1) + 3))


# =========================================================================== #
# edge_weight + edge_features together                                         #
# =========================================================================== #

class TestEdgeWeightWithEdgeFeatures:
    """Both fields must compose: same layer call, same forward."""

    EDGE_DIM = 2

    def test_gat_vector_edges_plus_weight(self):
        layer = TensorGATLayer(
            in_channels=C, out_channels=8, num_heads=2,
            use_edge_features=True, edge_dim=self.EDGE_DIM,
            bias=False, residual=False, add_self_loops=False,
        )
        layer.eval()
        x = _spatial()
        ei = _ei_dense()
        E = ei.size(1)
        ef = torch.randn(E, self.EDGE_DIM)
        w = torch.linspace(0.3, 1.7, E)
        out_both = layer(x, ei, edge_features=ef, edge_weight=w)
        out_ef = layer(x, ei, edge_features=ef)
        out_w = layer(x, ei, edge_features=ef, edge_weight=torch.ones(E))
        # both differs from each component-only call
        assert (out_both - out_ef).abs().max().item() > 1e-4
        # weight=1 with same ef equals the ef-only call (round-trip)
        assert torch.allclose(out_w, out_ef, atol=1e-5)

    def test_gat_spatial_edges_plus_weight(self):
        layer = TensorGATLayer(
            in_channels=C, out_channels=8, num_heads=2,
            use_edge_features=True, edge_dim=self.EDGE_DIM,
            bias=False, residual=False,
        )
        layer.eval()
        x = _spatial()
        ei = _ei_dense()
        E = ei.size(1)
        ef = torch.randn(E, self.EDGE_DIM, H, W, requires_grad=True)
        w = torch.linspace(0.3, 1.7, E, requires_grad=True)
        out = layer(x, ei, edge_features=ef, edge_weight=w)
        out.sum().backward()
        assert ef.grad is not None and torch.isfinite(ef.grad).all()
        assert w.grad is not None and torch.isfinite(w.grad).all()

    def test_sage_spatial_edges_plus_weight(self):
        layer = TensorGraphSAGELayer(
            in_channels=C, out_channels=8, aggr="mean",
            use_edge_features=True, edge_dim=self.EDGE_DIM,
        )
        layer.eval()
        x = _spatial()
        ei = _ei_dense()
        E = ei.size(1)
        ef = torch.randn(E, self.EDGE_DIM, H, W)
        out_both = layer(x, ei, edge_features=ef, edge_weight=torch.full((E,), 0.5))
        out_ef = layer(x, ei, edge_features=ef)
        # weight=0.5 must scale the neighbour messages: agg = mean(0.5 * msg) = 0.5 * mean(msg)
        # but self-transform is unaffected. So the diff is the 0.5x scaling on aggregate only.
        assert (out_both - out_ef).abs().max().item() > 1e-4

    def test_gin_spatial_edges_plus_weight(self):
        layer = TensorGINLayer(
            in_channels=C, out_channels=8,
            use_edge_features=True, edge_dim=self.EDGE_DIM,
            edge_features_kind="spatial", train_eps=True,
        )
        x = _spatial().requires_grad_(True)
        ei = _ei_dense()
        E = ei.size(1)
        ef = torch.randn(E, self.EDGE_DIM, H, W, requires_grad=True)
        w = torch.linspace(0.4, 1.6, E, requires_grad=True)
        out = layer(x, ei, edge_features=ef, edge_weight=w)
        out.sum().backward()
        for name, p in [("ef", ef), ("w", w), ("x", x)]:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all(), f"{name}.grad has NaN/Inf"

    def test_conv_spatial_edges_plus_weight(self):
        """ConvMessagePassing with edge features (channels = node channels) and weight."""
        layer = ConvMessagePassing(
            (C, H, W), (8, H, W),
            use_edge_features=True,
            aggregator_params=_fast_agg(),
        )
        layer.eval()
        x = _spatial()
        ei = _ei_dense()
        E = ei.size(1)
        # ConvMessagePassing wires edge channels = node channels (= C) in
        # message().
        ef = torch.randn(E, C, H, W)
        out_both = layer(x, ei, edge_features=ef, edge_weight=torch.linspace(0.5, 2.0, E))
        out_ef = layer(x, ei, edge_features=ef)
        assert (out_both - out_ef).abs().max().item() > 1e-4


# =========================================================================== #
# Spatial edge features: parameter & input gradient on every layer             #
# =========================================================================== #

class TestSpatialEdgeGradients:
    EDGE_DIM = 2

    def test_sage_spatial_param_and_input_grad(self):
        layer = TensorGraphSAGELayer(
            in_channels=C, out_channels=8,
            use_edge_features=True, edge_dim=self.EDGE_DIM,
        )
        x = _spatial()
        ei = _ei_dense()
        ef = torch.randn(ei.size(1), self.EDGE_DIM, H, W, requires_grad=True)
        layer(x, ei, edge_features=ef).sum().backward()
        assert layer.W_neigh.weight.grad is not None
        assert torch.isfinite(layer.W_neigh.weight.grad).all()
        assert layer.W_neigh.weight.grad.norm().item() > 0.0
        assert torch.isfinite(ef.grad).all()
        assert ef.grad.norm().item() > 0.0

    def test_gin_spatial_param_and_input_grad(self):
        layer = TensorGINLayer(
            in_channels=C, out_channels=8,
            use_edge_features=True, edge_dim=self.EDGE_DIM,
        )
        x = _spatial()
        ei = _ei_dense()
        ef = torch.randn(ei.size(1), self.EDGE_DIM, H, W, requires_grad=True)
        layer(x, ei, edge_features=ef).sum().backward()
        # When edge_dim != in_channels, edge_proj is a Conv2d.
        assert isinstance(layer.edge_proj, nn.Conv2d)
        assert layer.edge_proj.weight.grad is not None
        assert torch.isfinite(layer.edge_proj.weight.grad).all()
        assert layer.edge_proj.weight.grad.norm().item() > 0.0
        assert torch.isfinite(ef.grad).all()
        assert ef.grad.norm().item() > 0.0


# =========================================================================== #
# Edge-order invariance (with spatial edge features and weight)                #
# =========================================================================== #

def _permute_edges(ei: torch.Tensor, *tensors, perm: torch.Tensor):
    new_ei = ei[:, perm]
    return (new_ei,) + tuple(t[perm] for t in tensors)


class TestEdgeOrderInvariance:
    EDGE_DIM = 2

    def _check(self, layer, *, use_ef: bool, ef_kind: str | None):
        layer.eval()
        x = _spatial()
        ei = _ei_dense()
        E = ei.size(1)
        torch.manual_seed(2)
        kwargs_a: dict = {}
        kwargs_b: dict = {}
        perm = torch.randperm(E)

        if use_ef:
            if ef_kind == "vector":
                ef = torch.randn(E, self.EDGE_DIM)
            elif ef_kind == "spatial":
                ef = torch.randn(E, self.EDGE_DIM, H, W)
            elif ef_kind == "conv":
                # ConvMessagePassing requires edge channels = node channels.
                ef = torch.randn(E, C, H, W)
            else:
                raise ValueError(ef_kind)
            kwargs_a["edge_features"] = ef
            kwargs_b["edge_features"] = ef[perm]

        w = torch.linspace(0.5, 1.5, E)
        kwargs_a["edge_weight"] = w
        kwargs_b["edge_weight"] = w[perm]

        out_a = layer(x, ei, **kwargs_a)
        out_b = layer(x, ei[:, perm], **kwargs_b)
        assert torch.allclose(out_a, out_b, atol=1e-5), \
            f"edge-order invariance broken (max |diff|={(out_a-out_b).abs().max():.2e})"

    def test_conv(self):
        layer = ConvMessagePassing(
            (C, H, W), (8, H, W),
            aggr="mean",
            use_edge_features=True,
            aggregator_params=_fast_agg(),
        )
        self._check(layer, use_ef=True, ef_kind="conv")

    def test_gat_spatial(self):
        layer = TensorGATLayer(
            in_channels=C, out_channels=8, num_heads=2,
            use_edge_features=True, edge_dim=self.EDGE_DIM,
            attn_dropout=0.0, residual=False, bias=False,
        )
        self._check(layer, use_ef=True, ef_kind="spatial")

    def test_sage_spatial(self):
        layer = TensorGraphSAGELayer(
            in_channels=C, out_channels=8, aggr="mean",
            use_edge_features=True, edge_dim=self.EDGE_DIM,
        )
        self._check(layer, use_ef=True, ef_kind="spatial")

    def test_gin_spatial(self):
        layer = TensorGINLayer(
            in_channels=C, out_channels=8,
            use_edge_features=True, edge_dim=self.EDGE_DIM,
        )
        self._check(layer, use_ef=True, ef_kind="spatial")


# =========================================================================== #
# Node-permutation equivariance                                                #
# =========================================================================== #

class TestNodePermutationEquivariance:
    EDGE_DIM = 2

    def _check(self, layer, *, use_ef: bool, ef_kind: str):
        layer.eval()
        torch.manual_seed(11)
        x = _spatial()
        ei = _ei_dense()
        E = ei.size(1)

        # Build per-edge tensors.
        if use_ef:
            if ef_kind == "vector":
                ef = torch.randn(E, self.EDGE_DIM)
            elif ef_kind == "spatial":
                ef = torch.randn(E, self.EDGE_DIM, H, W)
            elif ef_kind == "conv":
                ef = torch.randn(E, C, H, W)
            else:
                raise ValueError(ef_kind)
        else:
            ef = None
        w = torch.linspace(0.4, 1.6, E)

        out_orig = layer(x, ei, edge_features=ef, edge_weight=w) if use_ef \
            else layer(x, ei, edge_weight=w)

        # Permute nodes, rewrite edge_index accordingly.
        node_perm = torch.randperm(N)
        inv_perm = torch.argsort(node_perm)
        x_perm = x[node_perm]
        ei_perm = inv_perm[ei]  # apply inverse to translate old -> new id

        out_perm = layer(x_perm, ei_perm, edge_features=ef, edge_weight=w) if use_ef \
            else layer(x_perm, ei_perm, edge_weight=w)

        # The output of the permuted graph at node `inv_perm[j]` should
        # equal the original output at node `j`.
        out_perm_restored = out_perm[inv_perm.argsort()]
        # Equivalently: out_perm[inv_perm[j]] == out_orig[j], i.e.
        # out_orig == out_perm[inv_perm].
        assert torch.allclose(out_orig, out_perm[inv_perm], atol=1e-5), \
            f"node-permutation equivariance broken (max |diff|={(out_orig - out_perm[inv_perm]).abs().max():.2e})"

    def test_conv(self):
        layer = ConvMessagePassing(
            (C, H, W), (8, H, W), aggr="mean",
            use_edge_features=True, aggregator_params=_fast_agg(),
        )
        self._check(layer, use_ef=True, ef_kind="conv")

    def test_gat_spatial(self):
        layer = TensorGATLayer(
            in_channels=C, out_channels=8, num_heads=2,
            use_edge_features=True, edge_dim=self.EDGE_DIM,
            attn_dropout=0.0, residual=False, bias=False,
        )
        self._check(layer, use_ef=True, ef_kind="spatial")

    def test_sage_spatial(self):
        layer = TensorGraphSAGELayer(
            in_channels=C, out_channels=8, aggr="mean",
            use_edge_features=True, edge_dim=self.EDGE_DIM,
            bias=False,
        )
        self._check(layer, use_ef=True, ef_kind="spatial")

    def test_gin_spatial(self):
        layer = TensorGINLayer(
            in_channels=C, out_channels=8,
            use_edge_features=True, edge_dim=self.EDGE_DIM,
        )
        self._check(layer, use_ef=True, ef_kind="spatial")


# =========================================================================== #
# Isolated nodes finite                                                        #
# =========================================================================== #

class TestIsolatedNodesFiniteWithWeight:
    """Nodes with no incoming edges must still produce finite outputs when
    edge_weight is provided."""

    def _no_edge_for(self, dst: int) -> torch.Tensor:
        # Cycle 0->1->2->3->4->5->0 — every node has exactly one incoming
        # edge. Drop the edge incoming to node 5 so node 5 is isolated.
        src = torch.tensor([0, 1, 2, 3, 5])
        dst_t = torch.tensor([1, 2, 3, 4, 0])
        return torch.stack([src, dst_t])

    def test_gat_isolated_finite(self):
        layer = TensorGATLayer(
            in_channels=C, out_channels=8, num_heads=2,
            bias=False, residual=False, add_self_loops=False,
        )
        layer.eval()
        ei = self._no_edge_for(5)
        x = _spatial()
        out = layer(x, ei, edge_weight=torch.full((ei.size(1),), 0.5))
        assert torch.isfinite(out).all()
        assert torch.allclose(out[5], torch.zeros_like(out[5]), atol=1e-6)

    def test_sage_isolated_finite(self):
        layer = TensorGraphSAGELayer(
            in_channels=C, out_channels=8, aggr="mean", bias=False, residual=False,
        )
        layer.eval()
        ei = self._no_edge_for(5)
        x = _spatial()
        out = layer(x, ei, edge_weight=torch.full((ei.size(1),), 0.5))
        assert torch.isfinite(out).all()
        # Isolated node only sees the self transform.
        expected = layer.W_self(x[5:6]).squeeze(0)
        assert torch.allclose(out[5], expected, atol=1e-5)

    def test_gin_isolated_finite(self):
        layer = TensorGINLayer(in_channels=C, out_channels=8)
        ei = self._no_edge_for(5)
        x = _spatial()
        out = layer(x, ei, edge_weight=torch.full((ei.size(1),), 0.5))
        assert torch.isfinite(out).all()

    def test_conv_isolated_finite(self):
        layer = ConvMessagePassing(
            (C, H, W), (8, H, W), aggr="mean", aggregator_params=_fast_agg()
        )
        ei = self._no_edge_for(5)
        x = _spatial()
        out = layer(x, ei, edge_weight=torch.full((ei.size(1),), 0.5))
        assert torch.isfinite(out).all()


# =========================================================================== #
# Graph-object integration                                                     #
# =========================================================================== #

class TestGraphIntegration:
    EDGE_DIM = 2

    def test_graph_carries_edge_weight_and_spatial_edges_through_gat(self):
        x = _spatial()
        ei = _ei_dense()
        E = ei.size(1)
        ef = torch.randn(E, self.EDGE_DIM, H, W)
        w = torch.linspace(0.4, 1.5, E)
        g = Graph(x, ei, edge_weight=w, edge_features=ef)

        layer = TensorGATLayer(
            in_channels=C, out_channels=8, num_heads=2,
            use_edge_features=True, edge_dim=self.EDGE_DIM,
        )
        layer.eval()
        out = layer(
            g.node_features, g.edge_index,
            edge_features=g.edge_features,
            edge_weight=g.edge_weight,
        )
        assert out.shape == (g.num_nodes, 8, H, W)
        assert torch.isfinite(out).all()

    def test_graph_through_sage_spatial(self):
        x = _spatial()
        ei = _ei_dense()
        E = ei.size(1)
        ef = torch.randn(E, self.EDGE_DIM, H, W)
        w = torch.linspace(0.5, 1.5, E)
        g = Graph(x, ei, edge_weight=w, edge_features=ef)

        layer = TensorGraphSAGELayer(
            in_channels=C, out_channels=8,
            use_edge_features=True, edge_dim=self.EDGE_DIM,
        )
        out = layer(
            g.node_features, g.edge_index,
            edge_features=g.edge_features,
            edge_weight=g.edge_weight,
        )
        assert out.shape == (g.num_nodes, 8, H, W)


# =========================================================================== #
# CUDA mirror                                                                  #
# =========================================================================== #

@pytest.mark.cuda
class TestCUDAEdgeWeightAndSpatial:
    EDGE_DIM = 2

    def test_gat_spatial_edges_cuda(self):
        layer = TensorGATLayer(
            in_channels=C, out_channels=8, num_heads=2,
            use_edge_features=True, edge_dim=self.EDGE_DIM,
        ).cuda()
        x = _spatial().cuda().requires_grad_(True)
        ei = _ei_dense().cuda()
        E = ei.size(1)
        ef = torch.randn(E, self.EDGE_DIM, H, W, device="cuda", requires_grad=True)
        w = torch.linspace(0.4, 1.6, E, device="cuda", requires_grad=True)
        out = layer(x, ei, edge_features=ef, edge_weight=w)
        out.sum().backward()
        assert out.device.type == "cuda"
        assert torch.isfinite(out).all()
        for p in (x, ef, w):
            assert torch.isfinite(p.grad).all()
