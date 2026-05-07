"""Tests for tgraphx.layers.factory and tgraphx.models.factory."""
import json
import tempfile

import pytest
import torch

from tgraphx.graph_builders import (
    build_grid_graph,
    build_grid_graph_3d,
    build_knn_graph,
    image_to_patches,
    volume_to_patches,
)
from tgraphx.layers.factory import make_layer
from tgraphx.models.edge_predictor import EdgePredictor
from tgraphx.models.factory import build_model, build_model_from_config

try:
    import yaml as _yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# =========================================================================== #
# Helpers                                                                       #
# =========================================================================== #

def _smoke_forward(model, x, edge_index, batch=None, ep_mode=False):
    """Run a forward pass and assert output is a finite tensor."""
    if ep_mode:
        out = model(x, edge_index)
    else:
        out = model(x, edge_index, batch=batch)
    assert isinstance(out, torch.Tensor)
    assert out.isfinite().all()
    return out


# =========================================================================== #
# Layer factory — make_layer                                                    #
# =========================================================================== #

class TestMakeLayerConv:

    def test_conv_2d(self):
        layer = make_layer("conv", (4, 8, 8), (8, 8, 8))
        x = torch.randn(5, 4, 8, 8)
        ei = build_grid_graph(2, 3, directed=False, self_loops=True)[:, :5*6]  # quick ei
        ei = torch.zeros(2, 0, dtype=torch.long)  # empty edge OK for shape check
        ei = build_grid_graph(2, 3, directed=False, self_loops=True)
        x = torch.randn(6, 4, 8, 8)
        out = layer(x, ei)
        assert out.shape == (6, 8, 8, 8)

    def test_conv_3d(self):
        layer = make_layer("conv", (2, 4, 4, 4), (4, 4, 4, 4))
        x = torch.randn(4, 2, 4, 4, 4)
        ei = build_grid_graph_3d(1, 2, 2, directed=False, self_loops=True)
        out = layer(x, ei)
        assert out.shape == (4, 4, 4, 4, 4)

    def test_conv_vector_raises(self):
        with pytest.raises(ValueError, match="requires a 2-D"):
            make_layer("conv", (32,), (64,))


class TestMakeLayerGAT:

    def test_gat_2d(self):
        layer = make_layer("gat", (4, 4, 4), (8, 4, 4), heads=2)
        x = torch.randn(9, 4, 4, 4)
        ei = build_grid_graph(3, 3, directed=False, self_loops=True)
        out = layer(x, ei)
        assert out.shape == (9, 8, 4, 4)

    def test_gat_3d(self):
        layer = make_layer("gat", (2, 2, 2, 2), (4, 2, 2, 2), heads=2, spatial_rank=3)
        x = torch.randn(8, 2, 2, 2, 2)
        ei = build_grid_graph_3d(2, 2, 2, directed=False, self_loops=True)
        out = layer(x, ei)
        assert out.shape == (8, 4, 2, 2, 2)

    def test_gat_kwargs_passed(self):
        layer = make_layer("gat", (4, 4, 4), (4, 4, 4),
                           heads=1, residual=True, dropout=0.0)
        assert layer.residual is True
        assert layer.num_heads == 1

    def test_gat_vector_raises(self):
        with pytest.raises(ValueError, match="requires a 2-D"):
            make_layer("gat", (32,), (64,))


class TestMakeLayerSAGE:

    def test_sage_2d(self):
        layer = make_layer("sage", (4, 4, 4), (8, 4, 4))
        x = torch.randn(4, 4, 4, 4)
        ei = build_grid_graph(2, 2, directed=False, self_loops=True)
        out = layer(x, ei)
        assert out.shape == (4, 8, 4, 4)

    def test_sage_3d(self):
        layer = make_layer("sage", (2, 2, 2, 2), (4, 2, 2, 2))
        x = torch.randn(8, 2, 2, 2, 2)
        ei = build_grid_graph_3d(2, 2, 2, directed=False, self_loops=True)
        out = layer(x, ei)
        assert out.shape == (8, 4, 2, 2, 2)

    def test_sage_vector_raises(self):
        with pytest.raises(ValueError, match="requires a 2-D"):
            make_layer("sage", (32,), (64,))


class TestMakeLayerGIN:

    def test_gin_2d(self):
        layer = make_layer("gin", (4, 4, 4), (8, 4, 4))
        x = torch.randn(4, 4, 4, 4)
        ei = build_grid_graph(2, 2, directed=False, self_loops=True)
        out = layer(x, ei)
        assert out.shape == (4, 8, 4, 4)

    def test_gin_3d(self):
        layer = make_layer("gin", (2, 2, 2, 2), (4, 2, 2, 2))
        x = torch.randn(8, 2, 2, 2, 2)
        ei = build_grid_graph_3d(2, 2, 2, directed=False, self_loops=True)
        out = layer(x, ei)
        assert out.shape == (8, 4, 2, 2, 2)

    def test_gin_vector_raises(self):
        with pytest.raises(ValueError, match="requires a 2-D"):
            make_layer("gin", (32,), (64,))

    # ── API-01: newly forwarded GIN kwargs ───────────────────────────

    def test_gin_eps_kwarg(self):
        """eps= forwarded; layer has that epsilon value."""
        layer = make_layer("gin", (4, 4, 4), (4, 4, 4), eps=0.5)
        import torch
        assert float(layer.eps) == pytest.approx(0.5)

    def test_gin_train_eps_kwarg(self):
        """train_eps=True makes eps a learnable nn.Parameter."""
        import torch.nn as nn
        layer = make_layer("gin", (4, 4, 4), (4, 4, 4), train_eps=True)
        assert isinstance(layer.eps, nn.Parameter)

    def test_gin_hidden_channels_kwarg(self):
        """hidden_channels= is forwarded; layer runs forward correctly."""
        layer = make_layer("gin", (4, 4, 4), (8, 4, 4), hidden_channels=16)
        x = torch.randn(4, 4, 4, 4)
        ei = build_grid_graph(2, 2, directed=False, self_loops=True)
        out = layer(x, ei)
        assert out.shape == (4, 8, 4, 4)

    def test_gin_use_batchnorm_kwarg(self):
        """use_batchnorm=True is forwarded; forward works."""
        layer = make_layer("gin", (4, 4, 4), (8, 4, 4), use_batchnorm=True)
        x = torch.randn(4, 4, 4, 4)
        ei = build_grid_graph(2, 2, directed=False, self_loops=True)
        out = layer(x, ei)
        assert out.shape == (4, 8, 4, 4)
        assert torch.isfinite(out).all()

    def test_gin_combined_kwargs(self):
        """Multiple GIN kwargs forwarded together."""
        layer = make_layer(
            "gin", (4, 4, 4), (4, 4, 4),
            eps=0.1, train_eps=True, hidden_channels=32, use_batchnorm=True,
        )
        import torch.nn as nn
        assert isinstance(layer.eps, nn.Parameter)
        assert float(layer.eps.detach()) == pytest.approx(0.1)
        x = torch.randn(4, 4, 4, 4)
        ei = build_grid_graph(2, 2, directed=False, self_loops=True)
        out = layer(x, ei)
        assert out.shape == (4, 4, 4, 4)


class TestMakeLayerLinear:

    def test_linear_vector(self):
        layer = make_layer("linear", (32,), (64,))
        x = torch.randn(6, 32)
        ei = build_grid_graph(2, 3, directed=False, self_loops=True)
        out = layer(x, ei)
        assert out.shape == (6, 64)

    def test_linear_spatial_raises(self):
        with pytest.raises(ValueError, match="vector in_shape"):
            make_layer("linear", (4, 8, 8), (8, 8, 8))

    # ── BUG-02 factory kwargs ─────────────────────────────────────────

    def test_linear_dropout_kwarg_forwarded(self):
        """make_layer('linear', dropout=...) must honour the flag."""
        layer = make_layer("linear", (32,), (32,), dropout=0.9)
        x = torch.randn(4, 32)
        ei = build_grid_graph(2, 2, directed=False, self_loops=True)
        layer.train()
        outputs = [layer(x, ei).detach() for _ in range(5)]
        layer.eval()
        out_eval = layer(x, ei).detach()
        assert any(not torch.equal(o, out_eval) for o in outputs), (
            "dropout kwarg forwarded but had no effect"
        )

    def test_linear_residual_kwarg_forwarded(self):
        """make_layer('linear', residual=True) must add the skip connection."""
        layer = make_layer("linear", (32,), (32,), residual=True)
        layer.eval()
        x = torch.randn(4, 32)
        ei = build_grid_graph(2, 2, directed=False, self_loops=True)
        out_with = layer(x, ei).detach()
        layer.residual = False
        out_without = layer(x, ei).detach()
        layer.residual = True
        assert torch.allclose(out_with, out_without + x, atol=1e-5)

    def test_linear_use_batchnorm_kwarg_forwarded(self):
        """make_layer('linear', use_batchnorm=True) must create a BN module."""
        import torch.nn as nn
        layer = make_layer("linear", (32,), (64,), use_batchnorm=True)
        assert hasattr(layer, "bn")
        assert isinstance(layer.bn, nn.BatchNorm1d)
        # forward must work
        x = torch.randn(4, 32)
        ei = build_grid_graph(2, 2, directed=False, self_loops=True)
        out = layer(x, ei)
        assert out.shape == (4, 64)


class TestMakeLayerErrors:

    def test_unknown_name(self):
        with pytest.raises(ValueError, match="Unknown layer name"):
            make_layer("unknown", (4, 4, 4), (8, 4, 4))

    def test_bad_rank(self):
        with pytest.raises(ValueError, match="1 .* 3 .* 4"):
            make_layer("gat", (4, 4), (8, 4))

    def test_legacy_attention_3d_raises(self):
        with pytest.raises(NotImplementedError, match="nn.Conv2d"):
            make_layer("legacy_attention", (2, 4, 4, 4), (4, 4, 4, 4))

    def test_legacy_attention_vector(self):
        layer = make_layer("legacy_attention", (32,), (64,))
        assert layer is not None

    def test_legacy_attention_2d(self):
        layer = make_layer("legacy_attention", (4, 8, 8), (8, 8, 8))
        assert layer is not None


# =========================================================================== #
# Model factory — build_model                                                   #
# =========================================================================== #

class TestBuildModelNodeClassification:

    def test_vector(self):
        model = build_model(
            task="node_classification",
            layer="linear",
            in_shape=(16,),
            hidden_shape=(32,),
            num_layers=2,
            num_classes=4,
        )
        N = 10
        x = torch.randn(N, 16)
        ei = build_knn_graph(torch.randn(N, 2), k=3, self_loops=False)
        out = _smoke_forward(model, x, ei)
        assert out.shape == (N, 4)

    def test_vector_single_layer(self):
        model = build_model(
            task="node_classification",
            layer="linear",
            in_shape=(8,),
            hidden_shape=(16,),
            num_layers=1,
            num_classes=3,
        )
        x = torch.randn(6, 8)
        ei = build_grid_graph(2, 3, directed=False, self_loops=True)
        out = _smoke_forward(model, x, ei)
        assert out.shape == (6, 3)


class TestBuildModelGraphClassification:

    def test_vector(self):
        model = build_model(
            task="graph_classification",
            layer="linear",
            in_shape=(16,),
            hidden_shape=(32,),
            num_layers=2,
            num_classes=5,
        )
        N = 12
        x = torch.randn(N, 16)
        ei = build_knn_graph(torch.randn(N, 2), k=3, self_loops=False)
        batch = torch.tensor([0]*6 + [1]*6)
        out = _smoke_forward(model, x, ei, batch=batch)
        assert out.shape == (2, 5)

    def test_2d_spatial(self):
        images = torch.randn(1, 2, 4, 4)
        patches = image_to_patches(images, patch_size=2)  # [1, 4, 2, 2, 2]
        x = patches[0]  # [4, 2, 2, 2]
        ei = build_grid_graph(2, 2, directed=False, self_loops=True)
        batch = torch.zeros(4, dtype=torch.long)

        model = build_model(
            task="graph_classification",
            layer="gat",
            in_shape=(2, 2, 2),
            hidden_shape=(4, 2, 2),
            num_layers=2,
            num_classes=3,
            heads=2,
        )
        out = _smoke_forward(model, x, ei, batch=batch)
        assert out.shape == (1, 3)

    def test_3d_volumetric(self):
        vols = torch.randn(1, 2, 4, 4, 4)
        patches = volume_to_patches(vols, patch_size=2)  # [1, 8, 2, 2, 2, 2]
        x = patches[0]  # [8, 2, 2, 2, 2]
        ei = build_grid_graph_3d(2, 2, 2, directed=False, self_loops=True)
        batch = torch.zeros(8, dtype=torch.long)

        model = build_model(
            task="graph_classification",
            layer="sage",
            in_shape=(2, 2, 2, 2),
            hidden_shape=(4, 2, 2, 2),
            num_layers=2,
            num_classes=2,
        )
        out = _smoke_forward(model, x, ei, batch=batch)
        assert out.shape == (1, 2)

    def test_missing_num_classes_raises(self):
        with pytest.raises(ValueError, match="num_classes is required"):
            build_model(
                task="graph_classification",
                layer="linear",
                in_shape=(16,),
                hidden_shape=(32,),
                num_layers=2,
            )

    def test_missing_batch_raises(self):
        model = build_model(
            task="graph_classification",
            layer="linear",
            in_shape=(8,),
            hidden_shape=(16,),
            num_layers=1,
            num_classes=2,
        )
        x = torch.randn(4, 8)
        ei = build_grid_graph(2, 2, directed=False, self_loops=True)
        with pytest.raises(ValueError, match="batch"):
            model(x, ei)


class TestBuildModelNodeRegression:

    def test_vector(self):
        model = build_model(
            task="node_regression",
            layer="linear",
            in_shape=(16,),
            hidden_shape=(32,),
            num_layers=2,
            out_dim=1,
        )
        N = 8
        x = torch.randn(N, 16)
        ei = build_knn_graph(torch.randn(N, 2), k=3, self_loops=False)
        out = _smoke_forward(model, x, ei)
        assert out.shape == (N, 1)

    def test_missing_out_dim_raises(self):
        with pytest.raises(ValueError, match="out_dim is required"):
            build_model(
                task="node_regression",
                layer="linear",
                in_shape=(8,),
                hidden_shape=(16,),
                num_layers=1,
            )


class TestBuildModelGraphRegression:

    def _batch(self, N, G):
        return torch.cat([torch.full((N // G,), i) for i in range(G)]).long()

    def test_vector(self):
        model = build_model(
            task="graph_regression",
            layer="linear",
            in_shape=(16,),
            hidden_shape=(32,),
            num_layers=2,
            out_dim=3,
        )
        N = 12
        x = torch.randn(N, 16)
        ei = build_knn_graph(torch.randn(N, 2), k=3, self_loops=False)
        batch = self._batch(N, 2)
        out = _smoke_forward(model, x, ei, batch=batch)
        assert out.shape == (2, 3)

    def test_2d_spatial(self):
        images = torch.randn(2, 2, 4, 4)
        patches = image_to_patches(images, patch_size=2)  # [2, 4, 2, 2, 2]
        x = torch.cat([patches[0], patches[1]], dim=0)   # [8, 2, 2, 2]
        ei = build_grid_graph(2, 2, directed=False, self_loops=True)
        # Two graphs in the batch, 4 nodes each (same topology, different offsets)
        from tgraphx.core.graph import Graph
        from tgraphx.core.graph import GraphBatch
        g1 = Graph(patches[0], ei)
        g2 = Graph(patches[1], ei)
        gb = GraphBatch([g1, g2])

        model = build_model(
            task="graph_regression",
            layer="gin",
            in_shape=(2, 2, 2),
            hidden_shape=(4, 2, 2),
            num_layers=2,
            out_dim=1,
        )
        out = model(gb.node_features, gb.edge_index, batch=gb.batch)
        assert out.shape == (2, 1)

    def test_3d_volumetric(self):
        vols = torch.randn(1, 2, 4, 4, 4)
        patches = volume_to_patches(vols, patch_size=2)  # [1, 8, 2, 2, 2, 2]
        x = patches[0]
        ei = build_grid_graph_3d(2, 2, 2, directed=False, self_loops=True)
        batch = torch.zeros(8, dtype=torch.long)

        model = build_model(
            task="graph_regression",
            layer="conv",
            in_shape=(2, 2, 2, 2),
            hidden_shape=(4, 2, 2, 2),
            num_layers=2,
            out_dim=2,
        )
        out = _smoke_forward(model, x, ei, batch=batch)
        assert out.shape == (1, 2)


class TestBuildModelEdgePrediction:

    def test_vector_smoke(self):
        model = build_model(
            task="edge_prediction",
            layer="linear",
            in_shape=(16,),
            hidden_shape=(32,),
            num_layers=2,
            out_dim=1,
        )
        N = 8
        x = torch.randn(N, 16)
        ei = build_knn_graph(torch.randn(N, 2), k=3, self_loops=False)
        out = _smoke_forward(model, x, ei, ep_mode=True)
        assert out.shape == (ei.shape[1], 1)

    def test_default_out_dim_1(self):
        model = build_model(
            task="edge_prediction",
            layer="linear",
            in_shape=(8,),
            hidden_shape=(16,),
            num_layers=1,
        )
        x = torch.randn(6, 8)
        ei = build_grid_graph(2, 3, directed=False, self_loops=False)
        out = model(x, ei)
        assert out.shape[1] == 1


class TestBuildModelErrors:

    def test_unknown_task(self):
        with pytest.raises(ValueError, match="Unknown task"):
            build_model(
                task="super_task",
                layer="linear",
                in_shape=(8,),
                hidden_shape=(16,),
                num_layers=1,
                num_classes=2,
            )

    def test_link_prediction_not_implemented(self):
        with pytest.raises(NotImplementedError, match="link_prediction"):
            build_model(
                task="link_prediction",
                layer="linear",
                in_shape=(8,),
                hidden_shape=(16,),
                num_layers=1,
            )

    def test_invalid_num_layers(self):
        with pytest.raises(ValueError, match="num_layers"):
            build_model(
                task="node_classification",
                layer="linear",
                in_shape=(8,),
                hidden_shape=(16,),
                num_layers=0,
                num_classes=2,
            )


# =========================================================================== #
# Config-based construction                                                     #
# =========================================================================== #

class TestBuildModelFromConfig:

    _base_cfg = {
        "model": {
            "task": "graph_classification",
            "layer": "linear",
            "in_shape": [16],
            "hidden_shape": [32],
            "num_layers": 2,
            "num_classes": 3,
        }
    }

    def test_from_dict(self):
        model = build_model_from_config(self._base_cfg)
        assert model is not None

    def test_from_dict_output_shape(self):
        model = build_model_from_config(self._base_cfg)
        N = 6
        x = torch.randn(N, 16)
        ei = build_knn_graph(torch.randn(N, 2), k=2, self_loops=False)
        batch = torch.zeros(N, dtype=torch.long)
        out = model(x, ei, batch=batch)
        assert out.shape == (1, 3)

    def test_from_json_tempfile(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(self._base_cfg, f)
            fname = f.name
        model = build_model_from_config(fname)
        assert model is not None

    @pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
    def test_from_yaml_tempfile(self):
        import yaml
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml.dump(self._base_cfg, f)
            fname = f.name
        model = build_model_from_config(fname)
        assert model is not None

    def test_missing_model_key(self):
        with pytest.raises(KeyError, match="'model'"):
            build_model_from_config({"architecture": {}})

    def test_missing_task_key(self):
        cfg = {"model": {"layer": "linear", "in_shape": [8], "hidden_shape": [16], "num_layers": 1}}
        with pytest.raises(KeyError, match="'task'"):
            build_model_from_config(cfg)

    def test_unsupported_layer(self):
        cfg = {"model": {**self._base_cfg["model"], "layer": "banana"}}
        with pytest.raises(ValueError, match="Unknown layer name"):
            build_model_from_config(cfg)

    def test_unsupported_task(self):
        cfg = {"model": {**self._base_cfg["model"], "task": "time_travel"}}
        with pytest.raises(ValueError, match="Unknown task"):
            build_model_from_config(cfg)

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            build_model_from_config(42)

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported config file"):
            build_model_from_config("/tmp/config.toml")

    def test_spatial_config(self):
        cfg = {
            "model": {
                "task": "graph_classification",
                "layer": "gat",
                "in_shape": [4, 4, 4],
                "hidden_shape": [8, 4, 4],
                "num_layers": 2,
                "num_classes": 3,
                "heads": 2,
                "residual": True,
                "dropout": 0.0,
            }
        }
        model = build_model_from_config(cfg)
        assert model is not None

    def test_3d_sage_config(self):
        cfg = {
            "model": {
                "task": "graph_classification",
                "layer": "sage",
                "in_shape": [2, 2, 2, 2],
                "hidden_shape": [4, 2, 2, 2],
                "num_layers": 2,
                "num_classes": 2,
            }
        }
        model = build_model_from_config(cfg)
        assert model is not None


# =========================================================================== #
# Integration with graph builders                                               #
# =========================================================================== #

class TestIntegrationWithBuilders:

    def test_gat_with_build_grid_graph(self):
        layer = make_layer("gat", (4, 4, 4), (8, 4, 4), heads=2)
        x = torch.randn(9, 4, 4, 4)
        ei = build_grid_graph(3, 3, directed=False, self_loops=True)
        out = layer(x, ei)
        assert out.shape == (9, 8, 4, 4)

    def test_sage_with_build_grid_graph_3d(self):
        layer = make_layer("sage", (2, 2, 2, 2), (4, 2, 2, 2))
        x = torch.randn(8, 2, 2, 2, 2)
        ei = build_grid_graph_3d(2, 2, 2, directed=False, self_loops=True)
        out = layer(x, ei)
        assert out.shape == (8, 4, 2, 2, 2)

    def test_graph_classification_2d_with_patches(self):
        images = torch.randn(1, 2, 4, 4)
        patches = image_to_patches(images, patch_size=2)
        x = patches[0]  # [4, 2, 2, 2]
        ei = build_grid_graph(2, 2, directed=False, self_loops=True)
        batch = torch.zeros(4, dtype=torch.long)

        model = build_model(
            task="graph_classification",
            layer="conv",
            in_shape=(2, 2, 2),
            hidden_shape=(4, 2, 2),
            num_layers=2,
            num_classes=3,
        )
        out = model(x, ei, batch=batch)
        assert out.shape == (1, 3)

    def test_graph_classification_3d_with_patches(self):
        vols = torch.randn(1, 2, 4, 4, 4)
        patches = volume_to_patches(vols, patch_size=2)
        x = patches[0]  # [8, 2, 2, 2, 2]
        ei = build_grid_graph_3d(2, 2, 2, directed=False, self_loops=True)
        batch = torch.zeros(8, dtype=torch.long)

        model = build_model(
            task="graph_classification",
            layer="gin",
            in_shape=(2, 2, 2, 2),
            hidden_shape=(4, 2, 2, 2),
            num_layers=2,
            num_classes=2,
        )
        out = model(x, ei, batch=batch)
        assert out.shape == (1, 2)


# =========================================================================== #
# EdgePredictor standalone                                                      #
# =========================================================================== #

class TestEdgePredictor:

    def test_vector_output_shape(self):
        predictor = EdgePredictor(in_dim=32, hidden_dim=64, out_dim=1)
        x = torch.randn(10, 32)
        ei = build_knn_graph(torch.randn(10, 2), k=3, self_loops=False)
        out = predictor(x, ei)
        assert out.shape == (ei.shape[1], 1)

    def test_spatial_input_pooled(self):
        predictor = EdgePredictor(in_dim=4, hidden_dim=16, out_dim=2)
        x = torch.randn(6, 4, 4, 4)  # [N, C, H, W]
        ei = build_grid_graph(2, 3, directed=False, self_loops=False)
        out = predictor(x, ei)
        assert out.shape == (ei.shape[1], 2)

    def test_wrong_in_dim_raises(self):
        predictor = EdgePredictor(in_dim=32)
        x = torch.randn(4, 16)  # wrong channel count
        ei = build_grid_graph(2, 2)
        with pytest.raises(ValueError, match="in_dim"):
            predictor(x, ei)
