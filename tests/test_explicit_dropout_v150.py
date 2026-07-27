"""v1.5.0 explicit-dropout regression suite.

TGraphX <= 1.4.2 silently applied ``dropout_prob=0.3`` inside
``CNNEncoder`` and ``DeepCNNAggregator`` (reached through
``ConvMessagePassing``, ``GraphClassifier``, ``CNN_GNN_Model``,
``make_layer('conv')``, and ``build_model(layer='conv')``, where the
``dropout`` kwarg was silently ignored).  Since v1.5.0:

- the documented default is 0.0 (no hidden dropout);
- omitting the value emits ``tgraphx.DropoutDefaultChangeWarning``;
- the effective value is visible in ``repr()`` and ``.config()``;
- the legacy behaviour is reconstructible via ``.legacy(...)``;
- ``make_layer('conv', dropout=...)`` and ``build_model(...,
  layer='conv', dropout=...)`` actually apply the requested value.
"""
from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

import tgraphx
from tgraphx import (
    CNNEncoder,
    ConvMessagePassing,
    DropoutDefaultChangeWarning,
    GraphClassifier,
    LEGACY_CNN_DROPOUT_PROB,
    build_model,
    build_model_from_config,
    load_checkpoint,
    make_layer,
    save_checkpoint,
)
from tgraphx.layers.aggregator import DeepCNNAggregator
from tgraphx.models.cnn_gnn_model import CNN_GNN_Model


def _dropout_probs(module: nn.Module) -> list[float]:
    """Probabilities of every active dropout module inside ``module``."""
    return [
        m.p
        for m in module.modules()
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d))
    ]


def _enc(**kw) -> CNNEncoder:
    defaults = dict(
        in_channels=3, out_features=8, num_layers=2, hidden_channels=8,
        use_batchnorm=False, use_residual=False, pool_layers=1,
        return_feature_map=True,
    )
    defaults.update(kw)
    return CNNEncoder(**defaults)


# ──────────────────────────────────────────────────────────────────── #
# A. Unspecified value: new default 0.0 + loud warning                  #
# ──────────────────────────────────────────────────────────────────── #

@pytest.mark.parametrize(
    "ctor",
    [
        lambda: _enc(),
        lambda: DeepCNNAggregator(4, 4, num_layers=2, use_batchnorm=False),
        lambda: ConvMessagePassing((4, 4, 4), (4, 4, 4)),
        lambda: GraphClassifier((4, 4, 4), (4, 4, 4), num_classes=2),
    ],
    ids=["CNNEncoder", "DeepCNNAggregator", "ConvMessagePassing", "GraphClassifier"],
)
def test_unspecified_dropout_warns_and_resolves_to_zero(ctor) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ctor()
    assert any(
        issubclass(w.category, DropoutDefaultChangeWarning) for w in caught
    ), "no DropoutDefaultChangeWarning emitted"
    assert model.dropout_prob == 0.0
    assert _dropout_probs(model) == [], "dropout modules present despite p=0"


def test_cnn_gnn_model_unspecified_cnn_dropout_warns() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = CNN_GNN_Model(
            cnn_params=dict(in_channels=3, out_features=4, num_layers=1,
                            hidden_channels=4, use_batchnorm=False,
                            use_residual=False, pool_layers=0,
                            return_feature_map=True),
            gnn_in_dim=(4, 8, 8), gnn_hidden_dim=(4, 8, 8),
            num_classes=2, num_gnn_layers=1,
        )
    assert any(
        issubclass(w.category, DropoutDefaultChangeWarning) for w in caught
    )
    assert model.encoder.dropout_prob == 0.0


def test_explicit_value_emits_no_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _enc(dropout_prob=0.0)
        _enc(dropout_prob=0.25)
        DeepCNNAggregator(4, 4, num_layers=1, dropout_prob=0.0)
        ConvMessagePassing((4, 4, 4), (4, 4, 4), dropout_prob=0.0)
        ConvMessagePassing(
            (4, 4, 4), (4, 4, 4), aggregator_params={"dropout_prob": 0.1}
        )
        GraphClassifier((4, 4, 4), (4, 4, 4), num_classes=2, dropout_prob=0.0)
    assert not any(
        issubclass(w.category, DropoutDefaultChangeWarning) for w in caught
    ), [str(w.message) for w in caught]


def test_invalid_dropout_rejected() -> None:
    with pytest.raises(ValueError):
        _enc(dropout_prob=1.0)
    with pytest.raises(ValueError):
        DeepCNNAggregator(4, 4, dropout_prob=-0.1)


# ──────────────────────────────────────────────────────────────────── #
# B. Configured value == effective value                                #
# ──────────────────────────────────────────────────────────────────── #

def test_configured_equals_effective_cnn_encoder() -> None:
    enc = _enc(dropout_prob=0.25)
    probs = _dropout_probs(enc)
    assert probs and all(p == 0.25 for p in probs)
    assert enc.dropout_prob == 0.25


def test_configured_equals_effective_aggregator() -> None:
    agg = DeepCNNAggregator(4, 4, num_layers=3, dropout_prob=0.4,
                            use_batchnorm=False)
    probs = _dropout_probs(agg)
    assert len(probs) == 3 and all(p == 0.4 for p in probs)


def test_zero_dropout_has_no_active_dropout_path() -> None:
    enc = _enc(dropout_prob=0.0)
    agg = DeepCNNAggregator(4, 4, num_layers=3, dropout_prob=0.0,
                            use_batchnorm=False)
    assert _dropout_probs(enc) == []
    assert _dropout_probs(agg) == []
    # Training-mode forward must be deterministic when dropout is 0.
    enc.train()
    x = torch.randn(2, 3, 8, 8)
    assert torch.equal(enc(x), enc(x))


def test_nonzero_dropout_active_in_train_inactive_in_eval() -> None:
    torch.manual_seed(0)
    enc = _enc(dropout_prob=0.5)
    x = torch.randn(4, 3, 8, 8)
    enc.train()
    a, b = enc(x), enc(x)
    assert not torch.equal(a, b), "train-mode dropout produced identical outputs"
    enc.eval()
    assert torch.equal(enc(x), enc(x))


# ──────────────────────────────────────────────────────────────────── #
# C. Visibility: repr / config parity                                   #
# ──────────────────────────────────────────────────────────────────── #

def test_repr_exposes_effective_dropout() -> None:
    assert "dropout_prob=0.25" in repr(_enc(dropout_prob=0.25))
    assert "dropout_prob=0.1" in repr(
        DeepCNNAggregator(4, 4, num_layers=1, dropout_prob=0.1)
    )
    layer = ConvMessagePassing((4, 4, 4), (4, 4, 4), dropout_prob=0.2)
    assert "dropout_prob=0.2" in repr(layer)


def test_config_round_trip_preserves_exact_value() -> None:
    enc = _enc(dropout_prob=0.37)
    clone = CNNEncoder(**enc.config())
    assert clone.dropout_prob == 0.37
    assert clone.config() == enc.config()

    agg = DeepCNNAggregator(4, 6, num_layers=2, dropout_prob=0.11)
    clone2 = DeepCNNAggregator(**agg.config())
    assert clone2.dropout_prob == 0.11
    assert clone2.config() == agg.config()


def test_conflicting_dropout_settings_raise() -> None:
    with pytest.raises(ValueError, match="conflicting dropout"):
        ConvMessagePassing(
            (4, 4, 4), (4, 4, 4),
            dropout_prob=0.1,
            aggregator_params={"dropout_prob": 0.2},
        )


# ──────────────────────────────────────────────────────────────────── #
# D. Factory paths actually honour dropout (the 1.4.2 silent-0.3 bug)   #
# ──────────────────────────────────────────────────────────────────── #

def test_make_layer_conv_forwards_dropout() -> None:
    layer = make_layer("conv", (4, 4, 4), (8, 4, 4), dropout=0.0)
    assert _dropout_probs(layer) == []
    layer2 = make_layer("conv", (4, 4, 4), (8, 4, 4), dropout=0.25)
    probs = _dropout_probs(layer2)
    assert probs and all(p == 0.25 for p in probs)


def test_make_layer_conv_use_batchnorm_forwarded() -> None:
    layer = make_layer("conv", (4, 4, 4), (8, 4, 4), dropout=0.0,
                       use_batchnorm=False)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in layer.modules())


def test_build_model_conv_respects_dropout_zero() -> None:
    model = build_model(
        "graph_classification", "conv", (4, 4, 4), (8, 4, 4),
        num_layers=2, num_classes=3, dropout=0.0,
    )
    assert _dropout_probs(model) == [], (
        "build_model(layer='conv', dropout=0.0) still contains dropout — "
        "the 1.4.2 silent-0.3 regression is back"
    )


def test_build_model_from_config_dropout_effective() -> None:
    cfg = {
        "model": {
            "task": "graph_classification",
            "layer": "conv",
            "in_shape": [4, 4, 4],
            "hidden_shape": [8, 4, 4],
            "num_layers": 1,
            "num_classes": 2,
            "dropout": 0.2,
        }
    }
    model = build_model_from_config(cfg)
    probs = _dropout_probs(model)
    assert probs and all(p == 0.2 for p in probs)


# ──────────────────────────────────────────────────────────────────── #
# E. Legacy behaviour reconstructible, checkpoints compatible           #
# ──────────────────────────────────────────────────────────────────── #

def test_legacy_constant_and_constructors() -> None:
    assert LEGACY_CNN_DROPOUT_PROB == 0.3
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        enc = CNNEncoder.legacy(3, 8, num_layers=2, hidden_channels=8)
        agg = DeepCNNAggregator.legacy(4, 4, num_layers=2)
    assert not any(
        issubclass(w.category, DropoutDefaultChangeWarning) for w in caught
    ), "legacy() must not warn — it is an intentional opt-in"
    assert enc.dropout_prob == 0.3 and enc.use_batchnorm and enc.use_residual
    assert agg.dropout_prob == 0.3 and agg.use_batchnorm
    probs = _dropout_probs(enc)
    assert probs and all(p == 0.3 for p in probs)


def test_legacy_and_new_state_dicts_are_interchangeable() -> None:
    """Dropout holds no parameters, so a 1.4.2-era checkpoint loads into a
    v1.5.0 model built with the new default (and vice versa)."""
    kw = dict(in_channels=3, out_features=8, num_layers=2, hidden_channels=8,
              pool_layers=1, return_feature_map=True)
    old_style = CNNEncoder.legacy(**kw)
    new_style = CNNEncoder(dropout_prob=0.0, use_batchnorm=True,
                           use_residual=True, **kw)
    assert set(old_style.state_dict()) == set(new_style.state_dict())
    new_style.load_state_dict(old_style.state_dict())
    old_style.eval()
    new_style.eval()
    x = torch.randn(2, 3, 8, 8)
    # Eval-mode outputs never depended on dropout_prob.
    assert torch.allclose(old_style(x), new_style(x))


def test_checkpoint_save_load_preserves_outputs(tmp_path) -> None:
    enc = _enc(dropout_prob=0.15)
    path = str(tmp_path / "enc.pt")
    save_checkpoint(enc, None, epoch=1, path=path, config=enc.config())
    rebuilt = CNNEncoder(
        **torch.load(path, weights_only=True)["config"]
    )
    load_checkpoint(rebuilt, None, path)
    assert rebuilt.dropout_prob == 0.15
    enc.eval()
    rebuilt.eval()
    x = torch.randn(2, 3, 8, 8)
    assert torch.allclose(enc(x), rebuilt(x))


def test_warning_is_publicly_importable_and_filterable() -> None:
    assert issubclass(tgraphx.DropoutDefaultChangeWarning, UserWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings(
            "ignore", category=tgraphx.DropoutDefaultChangeWarning
        )
        _enc()  # must not raise despite error filter
