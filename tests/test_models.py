"""Smoke tests for high-level model classes.

Covers:
- CNNEncoder: output shapes, H-01 residual channel-change, SafeMaxPool2d
- GraphClassifier: pooling variants, optional edge_features (M-01), batch check
- NodeClassifier: output shape and backward
- CNN_GNN_Model: node-level and graph-level forward, gnn_dropout wiring (H-03),
  skip_cnn_to_classifier storage (M-04)

All tests use small synthetic tensors.  No real datasets or pretrained weights
are loaded (PreEncoder import is tested in test_imports.py).
"""

import pytest
import torch

from tgraphx.models import (
    CNNEncoder,
    GraphClassifier,
    NodeClassifier,
    CNN_GNN_Model,
)
from tgraphx.models.cnn_encoder import ResidualBlock  # internal; needed for H-01


# ──────────────────────────────────────────────────────────────────── #
# Helpers                                                               #
# ──────────────────────────────────────────────────────────────────── #

def _ei(n, device="cpu"):
    src = torch.arange(n, device=device)
    return torch.stack([src, (src + 1) % n])


def _batch_vec(sizes, device="cpu"):
    parts = [
        torch.full((n,), i, dtype=torch.long, device=device)
        for i, n in enumerate(sizes)
    ]
    return torch.cat(parts)


def _fast_agg(**kw):
    return {"num_layers": 1, "use_batchnorm": False, "dropout_prob": 0.0, **kw}


# ──────────────────────────────────────────────────────────────────── #
# ResidualBlock unit test (H-01)                                        #
# ──────────────────────────────────────────────────────────────────── #

class TestResidualBlock:
    def test_channel_change_no_crash(self):
        """H-01: stride=1 projection must not halve spatial dims when channels change."""
        import torch.nn as nn
        block = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU())
        rb = ResidualBlock(block, in_channels=3, out_channels=32)
        out = rb(torch.randn(2, 3, 16, 16))
        # stride=1 fix: spatial dims preserved; shape must be (2, 32, 16, 16)
        assert out.shape == (2, 32, 16, 16)

    def test_same_channels_no_projection(self):
        """When in == out channels, projection is never applied."""
        import torch.nn as nn
        block = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.ReLU())
        rb = ResidualBlock(block, in_channels=8, out_channels=8)
        x = torch.randn(2, 8, 16, 16)
        out = rb(x)
        assert out.shape == (2, 8, 16, 16)


# ──────────────────────────────────────────────────────────────────── #
# CNNEncoder                                                            #
# ──────────────────────────────────────────────────────────────────── #

class TestCNNEncoder:
    """Encoder config: 3→8 channels, 2 layers, 1 pool, fast (no BN, no dropout)."""

    def _enc(self, return_feature_map=True, **kw):
        defaults = dict(
            in_channels=3, out_features=8,
            num_layers=2, hidden_channels=8,
            dropout_prob=0.0, use_batchnorm=False,
            use_residual=False, pool_layers=1,
        )
        defaults.update(kw)
        return CNNEncoder(return_feature_map=return_feature_map, **defaults)

    def test_feature_map_shape(self):
        out = self._enc()(torch.randn(4, 3, 16, 16))
        # pool_layers=1 halves 16→8; out_features channels
        assert out.shape == (4, 8, 8, 8)

    def test_vector_output_shape(self):
        out = self._enc(return_feature_map=False)(torch.randn(4, 3, 16, 16))
        assert out.shape == (4, 8)

    def test_output_finite(self):
        out = self._enc()(torch.randn(4, 3, 16, 16))
        assert torch.isfinite(out).all()

    def test_backward(self):
        x = torch.randn(2, 3, 16, 16, requires_grad=True)
        self._enc()(x).sum().backward()
        assert x.grad is not None

    def test_residual_with_channel_change_no_crash(self):
        """H-01: a ResidualBlock inside CNNEncoder with pool_layers=0 must not crash."""
        # pool_layers=0 means all layers have no pooling, so ResidualBlocks
        # only need to handle channel mismatch — the scenario fixed by H-01.
        enc = CNNEncoder(
            in_channels=3, out_features=8,
            num_layers=3, hidden_channels=16,
            dropout_prob=0.0, use_batchnorm=False,
            use_residual=True, pool_layers=0,
            return_feature_map=True,
        )
        out = enc(torch.randn(2, 3, 16, 16))
        assert out.shape[0] == 2
        assert torch.isfinite(out).all()

    def test_safe_pool_small_spatial(self):
        """SafeMaxPool2d must not crash when H or W falls below kernel size."""
        enc = CNNEncoder(
            in_channels=3, out_features=8,
            num_layers=2, hidden_channels=8,
            dropout_prob=0.0, use_batchnorm=False,
            use_residual=False, pool_layers=2,
            return_feature_map=True,
        )
        out = enc(torch.randn(2, 3, 3, 3))  # 3×3 → pool → 1×1 → second pool skipped
        assert out.shape[0] == 2
        assert torch.isfinite(out).all()


# ──────────────────────────────────────────────────────────────────── #
# GraphClassifier                                                       #
# ──────────────────────────────────────────────────────────────────── #

class TestGraphClassifier:
    C, H, W = 3, 4, 4
    NUM_CLASSES = 5
    N = 6  # 2 graphs × 3 nodes

    def _clf(self, pooling="mean"):
        return GraphClassifier(
            in_shape=(self.C, self.H, self.W),
            hidden_shape=(8, self.H, self.W),
            num_classes=self.NUM_CLASSES,
            num_layers=1,
            aggr="sum",
            pooling=pooling,
        )

    def _data(self):
        x = torch.randn(self.N, self.C, self.H, self.W)
        ei = _ei(self.N)
        batch = _batch_vec([3, 3])
        return x, ei, batch

    # -- pooling variants --

    def test_mean_pooling_output_shape(self):
        x, ei, batch = self._data()
        out = self._clf("mean")(x, ei, batch=batch)
        assert out.shape == (2, self.NUM_CLASSES)

    def test_sum_pooling_output_shape(self):
        x, ei, batch = self._data()
        out = self._clf("sum")(x, ei, batch=batch)
        assert out.shape == (2, self.NUM_CLASSES)

    def test_max_pooling_output_shape(self):
        """M-06: vectorised max pooling must return the correct shape."""
        x, ei, batch = self._data()
        out = self._clf("max")(x, ei, batch=batch)
        assert out.shape == (2, self.NUM_CLASSES)

    def test_output_finite(self):
        x, ei, batch = self._data()
        assert torch.isfinite(self._clf()(x, ei, batch=batch)).all()

    # -- M-01: edge_features is now optional --

    def test_edge_features_omitted(self):
        """M-01: calling forward without edge_features must work."""
        x, ei, batch = self._data()
        out = self._clf()(x, ei, batch=batch)          # edge_features not passed
        assert out.shape == (2, self.NUM_CLASSES)

    def test_edge_features_passed_as_none(self):
        x, ei, batch = self._data()
        out = self._clf()(x, ei, None, batch)          # old positional style
        assert out.shape == (2, self.NUM_CLASSES)

    def test_missing_batch_raises(self):
        """batch is still required; calling without it must raise a clear error."""
        x, ei, _ = self._data()
        with pytest.raises(ValueError, match="batch"):
            self._clf()(x, ei)

    # -- backward --

    def test_backward(self):
        x, ei, batch = self._data()
        x = x.requires_grad_(True)
        self._clf()(x, ei, batch=batch).sum().backward()
        assert x.grad is not None


# ──────────────────────────────────────────────────────────────────── #
# NodeClassifier                                                        #
# ──────────────────────────────────────────────────────────────────── #

class TestNodeClassifier:
    D, N, NUM_CLASSES = 16, 6, 4

    def _clf(self):
        return NodeClassifier(
            in_shape=(self.D,),
            hidden_shape=(32,),
            num_classes=self.NUM_CLASSES,
            num_layers=2,
        )

    def test_output_shape(self):
        x = torch.randn(self.N, self.D)
        out = self._clf()(x, _ei(self.N))
        assert out.shape == (self.N, self.NUM_CLASSES)

    def test_output_finite(self):
        x = torch.randn(self.N, self.D)
        out = self._clf()(x, _ei(self.N))
        assert torch.isfinite(out).all()

    def test_backward(self):
        x = torch.randn(self.N, self.D, requires_grad=True)
        self._clf()(x, _ei(self.N)).sum().backward()
        assert x.grad is not None


# ──────────────────────────────────────────────────────────────────── #
# CNN_GNN_Model                                                         #
# ──────────────────────────────────────────────────────────────────── #

class TestCNNGNNModel:
    # CNN: 3→8 channels, pool_layers=1, input 16×16 → CNN output (8,8,8)
    C_IN, H_IN, W_IN = 3, 16, 16
    C_GNN, H_GNN, W_GNN = 8, 8, 8
    N = 4
    NUM_CLASSES = 3

    def _cnn_params(self):
        return dict(
            in_channels=self.C_IN,
            out_features=self.C_GNN,
            num_layers=2,
            hidden_channels=self.C_GNN,
            dropout_prob=0.0,
            use_batchnorm=False,
            use_residual=False,
            pool_layers=1,
            return_feature_map=True,
        )

    def _model(self, **kw):
        return CNN_GNN_Model(
            cnn_params=self._cnn_params(),
            gnn_in_dim=(self.C_GNN, self.H_GNN, self.W_GNN),
            gnn_hidden_dim=(self.C_GNN, self.H_GNN, self.W_GNN),
            num_classes=self.NUM_CLASSES,
            num_gnn_layers=2,
            aggregator_params=_fast_agg(),
            **kw,
        )

    def _raw(self):
        return torch.randn(self.N, self.C_IN, self.H_IN, self.W_IN)

    # -- forward shapes --

    def test_node_level_output_shape(self):
        """Without a batch vector the model returns per-node logits."""
        out = self._model()(self._raw(), _ei(self.N))
        assert out.shape == (self.N, self.NUM_CLASSES)

    def test_graph_level_output_shape(self):
        """With a batch vector the model returns per-graph logits."""
        batch = _batch_vec([2, 2])
        out = self._model()(self._raw(), _ei(self.N), batch=batch)
        assert out.shape == (2, self.NUM_CLASSES)

    def test_output_finite(self):
        out = self._model()(self._raw(), _ei(self.N))
        assert torch.isfinite(out).all()

    # -- H-03: gnn_dropout and residual forwarded --

    def test_gnn_dropout_reaches_aggregator(self):
        """H-03: gnn_dropout=0.5 must reach each layer's DeepCNNAggregator.

        We intentionally omit aggregator_params here so that gnn_dropout is the
        sole source of dropout_prob (setdefault only fills missing keys).
        """
        model = CNN_GNN_Model(
            cnn_params=self._cnn_params(),
            gnn_in_dim=(self.C_GNN, self.H_GNN, self.W_GNN),
            gnn_hidden_dim=(self.C_GNN, self.H_GNN, self.W_GNN),
            num_classes=self.NUM_CLASSES,
            num_gnn_layers=2,
            gnn_dropout=0.5,
            # No aggregator_params — gnn_dropout fills dropout_prob via setdefault.
        )
        for layer in model.gnn_layers:
            drops = [
                m for m in layer.aggregator.cnn.modules()
                if hasattr(m, "p") and abs(m.p - 0.5) < 1e-6
            ]
            assert drops, "gnn_dropout=0.5 did not reach DeepCNNAggregator"

    def test_residual_stored_on_gnn_layers(self):
        """H-03: residual=True must be stored on each ConvMessagePassing layer."""
        model = self._model(residual=True)
        for layer in model.gnn_layers:
            assert layer.residual is True

    # -- M-04: skip_cnn_to_classifier is a real attribute --

    def test_skip_cnn_to_classifier_default_false(self):
        """M-04: attribute must exist (not just checked with hasattr)."""
        model = self._model()
        assert model.skip_cnn_to_classifier is False

    def test_skip_cnn_to_classifier_true(self):
        model = self._model(skip_cnn_to_classifier=True)
        assert model.skip_cnn_to_classifier is True

    # -- backward --

    def test_backward(self):
        x = self._raw().requires_grad_(True)
        self._model()(x, _ei(self.N)).sum().backward()
        assert x.grad is not None

    # -- aggregator_params not mutated in-place --

    def test_aggregator_params_not_mutated(self):
        """CNN_GNN_Model must not mutate the caller's aggregator_params dict."""
        caller_params = {"num_layers": 1}
        original_keys = set(caller_params.keys())
        # Call directly (not via _model) to avoid the double-kwarg collision.
        CNN_GNN_Model(
            cnn_params=self._cnn_params(),
            gnn_in_dim=(self.C_GNN, self.H_GNN, self.W_GNN),
            gnn_hidden_dim=(self.C_GNN, self.H_GNN, self.W_GNN),
            num_classes=self.NUM_CLASSES,
            num_gnn_layers=2,
            aggregator_params=caller_params,
        )
        assert set(caller_params.keys()) == original_keys, (
            "CNN_GNN_Model mutated the caller's aggregator_params dict"
        )
