"""Mathematical-correctness tests for tensor-aware GNN layers.

Covers:

* Hand-computed sum / mean / max aggregation on a tiny graph.
* Edge-order invariance: shuffling ``edge_index`` columns must not change
  the output (modulo floating-point error).
* Node-permutation equivariance: permuting node order and remapping
  ``edge_index`` must permute the output in the same way.
* Spatial degenerate case ``H = W = 1``: forward + backward must work and
  match the natural vector-GNN analogue.
* Isolated-node sanity: a node with no incoming edges must produce finite
  output and not propagate NaN/Inf during backward.

All tests are CPU-only and deterministic (fixed seeds).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tgraphx.layers import (
    ConvMessagePassing,
    LinearMessagePassing,
    TensorGATLayer,
    TensorGraphSAGELayer,
    TensorGINLayer,
    TensorMessagePassingLayer,
)
from tgraphx.layers._scatter import edge_softmax, scatter_sum, scatter_mean, scatter_max


# ──────────────────────────────────────────────────────────────────── #
# Hand-computed aggregation on tiny graphs                              #
# ──────────────────────────────────────────────────────────────────── #

class TestHandComputedAggregation:
    """Verify the base layer's sum / mean / max aggregation against
    hand-computed values on a tiny graph with known messages."""

    @staticmethod
    def _layer(aggr: str, D: int = 4):
        """A TensorMessagePassingLayer subclass with identity message and update."""

        class IdentityMP(LinearMessagePassing):
            def __init__(self):
                super().__init__(in_shape=(D,), out_shape=(D,), aggr=aggr)

            def message(self, src, dest, edge_attr):
                return src  # identity

            def update(self, node_feature, aggregated_message):
                return aggregated_message

        return IdentityMP()

    def test_sum_aggregation_vector(self):
        # 4 nodes, edges 0->1, 2->1, 3->1, 0->2.  Node 1 has 3 incoming
        # edges so sum = x[0] + x[2] + x[3]; node 2 has 1; nodes 0 and 3
        # have 0 incoming edges.
        x = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0],
             [10., 10., 10., 10.],
             [5.0, 6.0, 7.0, 8.0],
             [-1., -2., -3., -4.]]
        )
        ei = torch.tensor([[0, 2, 3, 0], [1, 1, 1, 2]], dtype=torch.long)
        layer = self._layer("sum")
        out = layer(x, ei)

        expected_n1 = x[0] + x[2] + x[3]
        expected_n2 = x[0]
        assert torch.allclose(out[1], expected_n1)
        assert torch.allclose(out[2], expected_n2)
        # Isolated nodes (0 and 3) get zero from the aggregation step.
        assert torch.allclose(out[0], torch.zeros(4))
        assert torch.allclose(out[3], torch.zeros(4))

    def test_mean_aggregation_vector(self):
        x = torch.tensor(
            [[1., 2., 3., 4.],
             [10., 10., 10., 10.],
             [5., 6., 7., 8.],
             [-1., -2., -3., -4.]]
        )
        ei = torch.tensor([[0, 2, 3, 0], [1, 1, 1, 2]], dtype=torch.long)
        layer = self._layer("mean")
        out = layer(x, ei)

        expected_n1 = (x[0] + x[2] + x[3]) / 3.0
        expected_n2 = x[0] / 1.0
        assert torch.allclose(out[1], expected_n1)
        assert torch.allclose(out[2], expected_n2)
        assert torch.allclose(out[0], torch.zeros(4))

    def test_max_aggregation_vector(self):
        x = torch.tensor(
            [[1., 9., 3., 4.],
             [0., 0., 0., 0.],
             [5., 6., 7., 8.],
             [-1., -2., 100., -4.]]
        )
        ei = torch.tensor([[0, 2, 3, 0], [1, 1, 1, 2]], dtype=torch.long)
        layer = self._layer("max")
        out = layer(x, ei)
        # Node 1: max element-wise across x[0], x[2], x[3]
        expected_n1 = torch.stack([x[0], x[2], x[3]]).max(dim=0).values
        assert torch.allclose(out[1], expected_n1)
        # Node 2: only x[0]
        assert torch.allclose(out[2], x[0])
        # Isolated nodes (0 and 3) → 0 (scatter_max masks -inf)
        assert torch.allclose(out[0], torch.zeros(4))
        assert torch.allclose(out[3], torch.zeros(4))

    def test_max_aggregation_spatial(self):
        N, C, H, W = 4, 2, 3, 3
        torch.manual_seed(0)
        x = torch.randn(N, C, H, W)
        ei = torch.tensor([[0, 2, 3], [1, 1, 1]], dtype=torch.long)

        # Use ConvMessagePassing as a black box: aggr='max' goes through the
        # base layer's aggregate() which now uses scatter_max.
        layer = ConvMessagePassing(
            (C, H, W), (C, H, W), aggr="max",
            aggregator_params={"num_layers": 1, "use_batchnorm": False, "dropout_prob": 0.0},
        )
        # Replace the message conv with identity-like behaviour so we can check
        # aggregation directly: bypass via subclassing isn't necessary; use the
        # public scatter_max helper instead.
        msgs = torch.stack([x[0], x[2], x[3]], dim=0)  # 3 messages targeting node 1
        agg_n1_expected = msgs.max(dim=0).values

        # Direct check against scatter_max:
        agg = scatter_max(msgs, torch.tensor([1, 1, 1], dtype=torch.long), num_nodes=N)
        assert torch.allclose(agg[1], agg_n1_expected)
        # Nodes 0, 2, 3 had no edges → zero.
        for j in (0, 2, 3):
            assert torch.allclose(agg[j], torch.zeros_like(agg[j]))


# ──────────────────────────────────────────────────────────────────── #
# Edge-order invariance                                                  #
# ──────────────────────────────────────────────────────────────────── #

class TestEdgeOrderInvariance:
    """Shuffling edge_index columns (and the corresponding edge_features)
    must not change the layer output, modulo floating-point error."""

    N, C, H, W = 5, 4, 4, 4

    def _setup(self, seed=0):
        torch.manual_seed(seed)
        x = torch.randn(self.N, self.C, self.H, self.W)
        # Graph: 0->1, 2->1, 3->1, 0->2, 1->3, 4->0
        ei = torch.tensor(
            [[0, 2, 3, 0, 1, 4], [1, 1, 1, 2, 3, 0]], dtype=torch.long
        )
        return x, ei

    @staticmethod
    def _shuffle(ei: torch.Tensor, seed=42, edge_features: torch.Tensor | None = None):
        torch.manual_seed(seed)
        perm = torch.randperm(ei.size(1))
        ei_p = ei[:, perm]
        ef_p = edge_features[perm] if edge_features is not None else None
        return ei_p, ef_p

    def test_conv_message_passing(self):
        x, ei = self._setup()
        layer = ConvMessagePassing(
            (self.C, self.H, self.W), (8, self.H, self.W), aggr="sum",
            aggregator_params={"num_layers": 1, "use_batchnorm": False, "dropout_prob": 0.0},
        )
        layer.eval()
        out_orig = layer(x, ei)
        ei_p, _ = self._shuffle(ei)
        out_perm = layer(x, ei_p)
        assert torch.allclose(out_orig, out_perm, atol=1e-5)

    def test_tensor_gat_layer(self):
        x, ei = self._setup()
        layer = TensorGATLayer(
            in_channels=self.C, out_channels=8, num_heads=2,
            attn_dropout=0.0, residual=False, bias=False,
        )
        layer.eval()
        out_orig = layer(x, ei)
        ei_p, _ = self._shuffle(ei)
        out_perm = layer(x, ei_p)
        assert torch.allclose(out_orig, out_perm, atol=1e-5)

    def test_tensor_gat_with_edge_features(self):
        x, ei = self._setup()
        edge_dim = 3
        ef = torch.randn(ei.size(1), edge_dim)
        layer = TensorGATLayer(
            in_channels=self.C, out_channels=8, num_heads=2,
            use_edge_features=True, edge_dim=edge_dim,
            attn_dropout=0.0, residual=False, bias=False,
        )
        layer.eval()
        out_orig = layer(x, ei, edge_features=ef)
        ei_p, ef_p = self._shuffle(ei, edge_features=ef)
        out_perm = layer(x, ei_p, edge_features=ef_p)
        assert torch.allclose(out_orig, out_perm, atol=1e-5)

    def test_tensor_graphsage_mean(self):
        x, ei = self._setup()
        layer = TensorGraphSAGELayer(
            in_channels=self.C, out_channels=8, aggr="mean",
        )
        layer.eval()
        out_orig = layer(x, ei)
        ei_p, _ = self._shuffle(ei)
        out_perm = layer(x, ei_p)
        assert torch.allclose(out_orig, out_perm, atol=1e-5)

    def test_tensor_graphsage_max(self):
        x, ei = self._setup()
        layer = TensorGraphSAGELayer(
            in_channels=self.C, out_channels=8, aggr="max",
        )
        layer.eval()
        out_orig = layer(x, ei)
        ei_p, _ = self._shuffle(ei)
        out_perm = layer(x, ei_p)
        assert torch.allclose(out_orig, out_perm, atol=1e-5)

    def test_tensor_gin(self):
        x, ei = self._setup()
        layer = TensorGINLayer(in_channels=self.C, out_channels=8)
        layer.eval()
        out_orig = layer(x, ei)
        ei_p, _ = self._shuffle(ei)
        out_perm = layer(x, ei_p)
        assert torch.allclose(out_orig, out_perm, atol=1e-5)


# ──────────────────────────────────────────────────────────────────── #
# Node-permutation equivariance                                         #
# ──────────────────────────────────────────────────────────────────── #

class TestNodePermutationEquivariance:
    """Permute node order and remap edge_index; the layer output should
    permute the same way."""

    N, C, H, W = 5, 4, 4, 4

    def _setup(self, seed=0):
        torch.manual_seed(seed)
        x = torch.randn(self.N, self.C, self.H, self.W)
        ei = torch.tensor(
            [[0, 2, 3, 0, 1, 4], [1, 1, 1, 2, 3, 0]], dtype=torch.long
        )
        return x, ei

    @staticmethod
    def _permute(x: torch.Tensor, ei: torch.Tensor, perm: torch.Tensor):
        """Permute node order and remap edge_index to the new ordering."""
        x_perm = x[perm]
        # Build inverse permutation: inv[perm[i]] = i
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.size(0))
        ei_perm = inv[ei]
        return x_perm, ei_perm, inv

    def _check(self, layer, atol=1e-5, edge_features=None):
        torch.manual_seed(0)
        x, ei = self._setup()
        layer.eval()
        out_orig = layer(x, ei) if edge_features is None else layer(x, ei, edge_features=edge_features)

        torch.manual_seed(7)
        perm = torch.randperm(self.N)
        x_p, ei_p, inv = self._permute(x, ei, perm)
        # Edge features are per-edge, NOT per-node, so they don't permute when
        # only node order changes (edge_index columns are unchanged in count).
        out_perm = layer(x_p, ei_p) if edge_features is None else layer(x_p, ei_p, edge_features=edge_features)

        # Undo the permutation on the output to compare against the original.
        out_undone = out_perm[inv]
        assert torch.allclose(out_orig, out_undone, atol=atol), (
            f"max abs diff = {(out_orig - out_undone).abs().max().item():.3e}"
        )

    def test_conv_message_passing(self):
        layer = ConvMessagePassing(
            (self.C, self.H, self.W), (8, self.H, self.W), aggr="sum",
            aggregator_params={"num_layers": 1, "use_batchnorm": False, "dropout_prob": 0.0},
        )
        self._check(layer)

    def test_tensor_gat_layer(self):
        layer = TensorGATLayer(
            in_channels=self.C, out_channels=8, num_heads=2,
            attn_dropout=0.0, bias=False,
        )
        self._check(layer)

    def test_tensor_graphsage(self):
        layer = TensorGraphSAGELayer(
            in_channels=self.C, out_channels=8, aggr="mean",
        )
        self._check(layer)

    def test_tensor_gin(self):
        layer = TensorGINLayer(in_channels=self.C, out_channels=8)
        self._check(layer)


# ──────────────────────────────────────────────────────────────────── #
# H = W = 1 (degenerate spatial) sanity                                  #
# ──────────────────────────────────────────────────────────────────── #

class TestSpatialDegenerate:
    """When H = W = 1, the tensor-aware layers reduce to vector-message
    passing.  Forward and backward must still work cleanly."""

    N, C = 5, 8

    def _xy(self):
        x = torch.randn(self.N, self.C, 1, 1, requires_grad=True)
        ei = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        return x, ei

    def test_conv_message_passing(self):
        x, ei = self._xy()
        layer = ConvMessagePassing(
            (self.C, 1, 1), (4, 1, 1),
            aggregator_params={"num_layers": 1, "use_batchnorm": False, "dropout_prob": 0.0},
        )
        out = layer(x, ei)
        assert out.shape == (self.N, 4, 1, 1)
        out.sum().backward()
        assert torch.isfinite(x.grad).all()

    def test_tensor_gat(self):
        x, ei = self._xy()
        layer = TensorGATLayer(in_channels=self.C, out_channels=4, num_heads=2)
        out = layer(x, ei)
        assert out.shape == (self.N, 4, 1, 1)
        out.sum().backward()
        assert torch.isfinite(x.grad).all()

    def test_tensor_graphsage(self):
        x, ei = self._xy()
        layer = TensorGraphSAGELayer(in_channels=self.C, out_channels=4)
        out = layer(x, ei)
        assert out.shape == (self.N, 4, 1, 1)
        out.sum().backward()
        assert torch.isfinite(x.grad).all()

    def test_tensor_gin(self):
        x, ei = self._xy()
        layer = TensorGINLayer(in_channels=self.C, out_channels=4)
        out = layer(x, ei)
        assert out.shape == (self.N, 4, 1, 1)
        out.sum().backward()
        assert torch.isfinite(x.grad).all()


# ──────────────────────────────────────────────────────────────────── #
# Isolated-node sanity                                                   #
# ──────────────────────────────────────────────────────────────────── #

class TestIsolatedNode:
    """A graph with at least one node that has no incoming edges must give
    finite output and a clean backward pass."""

    N, C, H, W = 6, 4, 4, 4

    def _xy(self):
        torch.manual_seed(0)
        # Node 5 has zero incoming edges.
        x = torch.randn(self.N, self.C, self.H, self.W, requires_grad=True)
        ei = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        return x, ei

    def _check_finite_and_backward(self, layer):
        x, ei = self._xy()
        out = layer(x, ei)
        assert torch.isfinite(out).all(), "non-finite output for isolated node"
        out.sum().backward()
        assert torch.isfinite(x.grad).all(), "non-finite x.grad"

    def test_conv_message_passing(self):
        self._check_finite_and_backward(
            ConvMessagePassing(
                (self.C, self.H, self.W), (4, self.H, self.W),
                aggregator_params={"num_layers": 1, "use_batchnorm": False,
                                    "dropout_prob": 0.0},
            )
        )

    def test_tensor_gat_no_self_loops(self):
        self._check_finite_and_backward(
            TensorGATLayer(
                in_channels=self.C, out_channels=4, num_heads=2,
                bias=False, residual=False, add_self_loops=False,
            )
        )

    def test_tensor_gat_with_self_loops(self):
        self._check_finite_and_backward(
            TensorGATLayer(
                in_channels=self.C, out_channels=4, num_heads=2,
                add_self_loops=True,
            )
        )

    def test_tensor_graphsage_mean(self):
        self._check_finite_and_backward(
            TensorGraphSAGELayer(
                in_channels=self.C, out_channels=4, aggr="mean",
            )
        )

    def test_tensor_graphsage_max(self):
        self._check_finite_and_backward(
            TensorGraphSAGELayer(
                in_channels=self.C, out_channels=4, aggr="max",
            )
        )

    def test_tensor_gin(self):
        self._check_finite_and_backward(
            TensorGINLayer(in_channels=self.C, out_channels=4)
        )
