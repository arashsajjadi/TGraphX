"""Tests for tgraphx.training utility functions."""
import os
import tempfile

import pytest
import torch
import torch.nn as nn

from tgraphx.training import (
    accuracy,
    count_parameters,
    load_checkpoint,
    mean_absolute_error,
    mean_squared_error,
    save_checkpoint,
    set_seed,
)


# ─────────────────────────────────────────────────────────────────────────────
# set_seed
# ─────────────────────────────────────────────────────────────────────────────

class TestSetSeed:

    def test_deterministic_tensor(self):
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        set_seed(0)
        a = torch.randn(10)
        set_seed(1)
        b = torch.randn(10)
        assert not torch.equal(a, b)

    def test_accepts_zero(self):
        set_seed(0)  # should not raise

    def test_no_file_created(self, tmp_path):
        initial = list(tmp_path.iterdir())
        set_seed(99)
        assert list(tmp_path.iterdir()) == initial


# ─────────────────────────────────────────────────────────────────────────────
# count_parameters
# ─────────────────────────────────────────────────────────────────────────────

class TestCountParameters:

    def test_linear_layer(self):
        m = nn.Linear(10, 5)
        # 10*5 weights + 5 bias = 55
        assert count_parameters(m) == 55

    def test_frozen_params_excluded(self):
        m = nn.Linear(10, 5)
        for p in m.parameters():
            p.requires_grad_(False)
        assert count_parameters(m, trainable_only=True) == 0
        assert count_parameters(m, trainable_only=False) == 55

    def test_nested_module(self):
        m = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        expected = (4 * 4 + 4) + (4 * 2 + 2)  # 20 + 10 = 30
        assert count_parameters(m) == expected

    def test_empty_model(self):
        class Empty(nn.Module):
            def forward(self, x): return x
        assert count_parameters(Empty()) == 0


# ─────────────────────────────────────────────────────────────────────────────
# save_checkpoint / load_checkpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckpoint:

    @pytest.fixture
    def model_and_opt(self):
        m = nn.Linear(4, 2)
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        return m, opt

    def test_save_creates_file(self, model_and_opt, tmp_path):
        m, opt = model_and_opt
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(m, opt, epoch=5, path=path)
        assert os.path.isfile(path)

    def test_load_returns_epoch(self, model_and_opt, tmp_path):
        m, opt = model_and_opt
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(m, opt, epoch=7, path=path)
        m2 = nn.Linear(4, 2)
        opt2 = torch.optim.SGD(m2.parameters(), lr=0.01)
        epoch = load_checkpoint(m2, opt2, path=path)
        assert epoch == 7

    def test_weights_restored(self, model_and_opt, tmp_path):
        m, opt = model_and_opt
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(m, opt, epoch=1, path=path, loss=0.42)
        m2 = nn.Linear(4, 2)
        opt2 = torch.optim.SGD(m2.parameters(), lr=0.01)
        load_checkpoint(m2, opt2, path=path)
        x = torch.randn(3, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m2(x))

    def test_save_without_optimizer(self, tmp_path):
        m = nn.Linear(4, 2)
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(m, None, epoch=3, path=path)
        m2 = nn.Linear(4, 2)
        epoch = load_checkpoint(m2, None, path=path, map_location="cpu")
        assert epoch == 3

    def test_extra_fields_stored(self, model_and_opt, tmp_path):
        m, opt = model_and_opt
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(m, opt, epoch=2, path=path, loss=0.99, tag="best")
        data = torch.load(path, weights_only=False)
        assert data["loss"] == 0.99
        assert data["tag"] == "best"

    def test_creates_parent_dir(self, model_and_opt, tmp_path):
        m, opt = model_and_opt
        nested = str(tmp_path / "a" / "b" / "ckpt.pt")
        save_checkpoint(m, opt, epoch=0, path=nested)
        assert os.path.isfile(nested)


# ─────────────────────────────────────────────────────────────────────────────
# accuracy
# ─────────────────────────────────────────────────────────────────────────────

class TestAccuracy:

    def test_perfect(self):
        logits = torch.tensor([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0]])
        labels = torch.tensor([0, 1, 0])
        assert accuracy(logits, labels) == pytest.approx(1.0)

    def test_zero(self):
        logits = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
        labels = torch.tensor([1, 1])
        assert accuracy(logits, labels) == pytest.approx(0.0)

    def test_half(self):
        logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
        labels = torch.tensor([0, 0])
        assert accuracy(logits, labels) == pytest.approx(0.5)

    def test_wrong_rank_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            accuracy(torch.randn(5), torch.zeros(5, dtype=torch.long))


# ─────────────────────────────────────────────────────────────────────────────
# Regression metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionMetrics:

    def test_mae_zero(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        assert mean_absolute_error(x, x) == pytest.approx(0.0)

    def test_mae_known(self):
        p = torch.tensor([1.0, 3.0])
        t = torch.tensor([2.0, 1.0])
        assert mean_absolute_error(p, t) == pytest.approx(1.5)

    def test_mse_zero(self):
        x = torch.tensor([1.0, 2.0])
        assert mean_squared_error(x, x) == pytest.approx(0.0)

    def test_mse_known(self):
        p = torch.tensor([0.0, 2.0])
        t = torch.tensor([1.0, 0.0])
        # errors: 1, 2 → squares: 1, 4 → mean: 2.5
        assert mean_squared_error(p, t) == pytest.approx(2.5)


# =========================================================================== #
# train_epoch / evaluate / fit                                                 #
# =========================================================================== #

import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tgraphx import Graph, GraphBatch, build_grid_graph
from tgraphx.core.dataloader import GraphDataLoader, GraphDataset
from tgraphx.tracking import CSVLogger
from tgraphx.training import evaluate, fit, train_epoch


# ── Shared fixtures ──────────────────────────────────────────────────────────

def _make_tensor_loader(n=20, in_dim=8, num_classes=3, batch_size=5):
    """Simple (x, y) DataLoader for vector classification."""
    torch.manual_seed(0)
    x = torch.randn(n, in_dim)
    y = torch.randint(0, num_classes, (n,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _make_graph_loader(n_graphs=6, nodes=9, in_dim=8, num_classes=3, batch_size=3):
    """GraphDataLoader with graph_labels."""
    torch.manual_seed(1)
    graphs = []
    ei = build_grid_graph(3, 3, directed=False, self_loops=True)
    for i in range(n_graphs):
        nf = torch.randn(nodes, in_dim)
        gl = torch.randint(0, num_classes, (1,))
        graphs.append(Graph(nf, ei, graph_label=gl))
    return GraphDataLoader(GraphDataset(graphs), batch_size=batch_size, shuffle=False)


def _simple_linear_model(in_dim=8, hidden=16, num_classes=3):
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, num_classes),
    )


class TestTrainEpoch:

    def test_returns_loss_dict(self):
        loader = _make_tensor_loader()
        model = _simple_linear_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        result = train_epoch(model, loader, opt, F.cross_entropy, device="cpu")
        assert "loss" in result
        assert isinstance(result["loss"], float)
        assert result["loss"] >= 0

    def test_with_metrics(self):
        from tgraphx.training import accuracy as acc_fn

        loader = _make_tensor_loader()
        model = _simple_linear_model()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        result = train_epoch(
            model, loader, opt, F.cross_entropy,
            device="cpu",
            metrics={"accuracy": acc_fn},
        )
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_no_files_logger_none(self, tmp_path):
        loader = _make_tensor_loader()
        model = _simple_linear_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        train_epoch(model, loader, opt, F.cross_entropy,
                    device="cpu", logger=None, log_level=0)
        assert list(tmp_path.iterdir()) == []

    def test_with_csvlogger(self, tmp_path):
        loader = _make_tensor_loader()
        model = _simple_linear_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        logdir = str(tmp_path / "run")
        with CSVLogger(logdir) as logger:
            train_epoch(model, loader, opt, F.cross_entropy,
                        device="cpu", logger=logger, epoch=0)
        import csv, os
        path = os.path.join(logdir, "metrics.csv")
        assert os.path.isfile(path)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert "train_loss" in rows[0]

    def test_graph_batch_loader(self):
        loader = _make_graph_loader()
        from tgraphx.models.factory import build_model
        model = build_model(
            task="graph_classification", layer="linear",
            in_shape=(8,), hidden_shape=(16,), num_layers=1, num_classes=3,
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        result = train_epoch(model, loader, opt, F.cross_entropy, device="cpu")
        assert "loss" in result
        assert result["loss"] >= 0

    def test_unsupported_batch_raises(self):
        model = _simple_linear_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        bad_loader = [{"x": torch.randn(4, 8), "y": torch.zeros(4, dtype=torch.long)}]
        with pytest.raises((ValueError, RuntimeError)):
            train_epoch(model, bad_loader, opt, F.cross_entropy, device="cpu")

    def test_grad_clip(self):
        loader = _make_tensor_loader()
        model = _simple_linear_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        result = train_epoch(model, loader, opt, F.cross_entropy,
                             device="cpu", grad_clip=1.0)
        assert "loss" in result


class TestEvaluate:

    def test_returns_loss_dict(self):
        loader = _make_tensor_loader()
        model = _simple_linear_model()
        result = evaluate(model, loader, F.cross_entropy, device="cpu")
        assert "loss" in result
        assert isinstance(result["loss"], float)

    def test_no_grad_no_file(self, tmp_path):
        loader = _make_tensor_loader()
        model = _simple_linear_model()
        evaluate(model, loader, F.cross_entropy, device="cpu")
        assert list(tmp_path.iterdir()) == []

    def test_with_metrics(self):
        from tgraphx.training import accuracy as acc_fn

        loader = _make_tensor_loader()
        model = _simple_linear_model()
        result = evaluate(model, loader, F.cross_entropy,
                          metrics={"accuracy": acc_fn}, device="cpu")
        assert "accuracy" in result

    def test_graph_loader(self):
        loader = _make_graph_loader()
        from tgraphx.models.factory import build_model
        model = build_model(
            task="graph_classification", layer="linear",
            in_shape=(8,), hidden_shape=(16,), num_layers=1, num_classes=3,
        )
        result = evaluate(model, loader, F.cross_entropy, device="cpu")
        assert "loss" in result


class TestFit:

    def test_returns_history_list(self):
        loader = _make_tensor_loader()
        model = _simple_linear_model()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = fit(model, loader, epochs=2, optimizer=opt,
                      loss_fn=F.cross_entropy, device="cpu")
        assert isinstance(history, list)
        assert len(history) == 2
        assert "train_loss" in history[0]
        assert "epoch" in history[0]

    def test_val_loader_adds_val_loss(self):
        train_loader = _make_tensor_loader()
        val_loader = _make_tensor_loader()
        model = _simple_linear_model()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = fit(model, train_loader, val_loader=val_loader,
                      epochs=2, optimizer=opt, loss_fn=F.cross_entropy, device="cpu")
        assert "val_loss" in history[0]

    def test_no_files_when_logger_none(self, tmp_path):
        loader = _make_tensor_loader()
        model = _simple_linear_model()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        fit(model, loader, epochs=1, optimizer=opt,
            loss_fn=F.cross_entropy, device="cpu",
            logger=None, log_level=0)
        assert list(tmp_path.iterdir()) == []

    def test_no_dashboard_starts(self):
        import threading
        before = [t.name for t in threading.enumerate() if "dashboard" in t.name.lower()]
        loader = _make_tensor_loader()
        model = _simple_linear_model()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        fit(model, loader, epochs=1, optimizer=opt,
            loss_fn=F.cross_entropy, device="cpu")
        after = [t.name for t in threading.enumerate() if "dashboard" in t.name.lower()]
        new_threads = set(after) - set(before)
        assert not new_threads, f"fit() unexpectedly started dashboard threads: {new_threads}"

    def test_with_csvlogger(self, tmp_path):
        loader = _make_tensor_loader()
        model = _simple_linear_model()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        logdir = str(tmp_path / "fit_run")
        with CSVLogger(logdir) as logger:
            fit(model, loader, epochs=3, optimizer=opt,
                loss_fn=F.cross_entropy, device="cpu", logger=logger)
        import csv, os
        path = os.path.join(logdir, "metrics.csv")
        assert os.path.isfile(path)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3

    def test_missing_optimizer_raises(self):
        loader = _make_tensor_loader()
        model = _simple_linear_model()
        with pytest.raises(ValueError, match="optimizer"):
            fit(model, loader, epochs=1, loss_fn=F.cross_entropy)

    def test_missing_loss_fn_raises(self):
        loader = _make_tensor_loader()
        model = _simple_linear_model()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        with pytest.raises(ValueError, match="loss_fn"):
            fit(model, loader, epochs=1, optimizer=opt)


# =========================================================================== #
# API-03: _call_model signature-aware dispatch                                 #
# =========================================================================== #

import warnings as _warnings_mod
from tgraphx.training import _call_model, _compute_metrics

class TestCallModel:

    def test_simple_model_no_kwargs(self):
        """Simple model that accepts only (x,) receives positional args, no kwargs."""
        model = _simple_linear_model()
        x = torch.randn(4, 8)
        result = _call_model(model, (x,), {})
        assert result.shape == (4, 3)

    def test_factory_model_receives_batch(self):
        """Factory model forward accepts batch=...; it must be passed through."""
        from tgraphx import build_model, build_grid_graph
        model = build_model(
            task="graph_classification", layer="linear",
            in_shape=(8,), hidden_shape=(16,), num_layers=1, num_classes=3,
        )
        nf = torch.randn(9, 8)
        ei = build_grid_graph(3, 3, directed=False, self_loops=True)
        batch = torch.zeros(9, dtype=torch.long)
        # batch kwarg must be forwarded (graph classification requires it)
        result = _call_model(model, (nf, ei), {"batch": batch})
        assert result.shape == (1, 3)

    def test_unsupported_kwarg_silently_filtered(self):
        """Kwargs not in the model's forward signature are dropped, not crashed."""
        model = _simple_linear_model()
        x = torch.randn(4, 8)
        # 'batch' is not in Sequential.forward — must be filtered, not crash
        result = _call_model(model, (x,), {"batch": torch.zeros(4, dtype=torch.long)})
        assert result.shape == (4, 3)

    def test_internal_typeerror_propagated(self):
        """A TypeError raised inside forward must become a RuntimeError with context."""
        class BuggyModel(nn.Module):
            def forward(self, x):
                raise TypeError("internal bug: cannot concatenate str and int")

        model = BuggyModel()
        with pytest.raises(RuntimeError, match="internal bug"):
            _call_model(model, (torch.randn(2, 4),), {})

    def test_graph_classification_training_still_works(self):
        """Full graph-classification train loop must complete without error."""
        loader = _make_graph_loader()
        from tgraphx.models.factory import build_model
        model = build_model(
            task="graph_classification", layer="linear",
            in_shape=(8,), hidden_shape=(16,), num_layers=1, num_classes=3,
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        result = train_epoch(model, loader, opt, F.cross_entropy, device="cpu")
        assert "loss" in result
        assert result["loss"] >= 0


# =========================================================================== #
# API-04: _compute_metrics warns once on failure                               #
# =========================================================================== #

class TestComputeMetrics:

    def test_successful_metric_returned(self):
        output = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        targets = torch.tensor([1, 0], dtype=torch.long)
        from tgraphx.training import accuracy
        result = _compute_metrics({"accuracy": accuracy}, output, targets)
        assert "accuracy" in result
        assert result["accuracy"] == pytest.approx(1.0)

    def test_failing_metric_triggers_warning(self):
        """A broken metric must emit a UserWarning instead of silently disappearing."""
        def broken_metric(output, targets):
            raise ValueError("shape mismatch in metric")

        output = torch.randn(4, 3)
        targets = torch.zeros(4, dtype=torch.long)
        # Reset the module-level warned set so we always get the warning.
        import tgraphx.training as _tr
        _tr._warned_metrics.discard("broken")
        with _warnings_mod.catch_warnings(record=True) as caught:
            _warnings_mod.simplefilter("always")
            result = _compute_metrics({"broken": broken_metric}, output, targets)
        assert "broken" not in result
        assert any("broken" in str(w.message) for w in caught), (
            "Expected a UserWarning mentioning the metric name 'broken'"
        )

    def test_warning_shown_only_once(self):
        """Second failure for the same metric name must not produce another warning."""
        def broken(output, targets):
            raise RuntimeError("always fails")

        import tgraphx.training as _tr
        _tr._warned_metrics.discard("once_metric")
        output = torch.randn(2, 2)
        targets = torch.zeros(2, dtype=torch.long)

        with _warnings_mod.catch_warnings(record=True) as first:
            _warnings_mod.simplefilter("always")
            _compute_metrics({"once_metric": broken}, output, targets)
        with _warnings_mod.catch_warnings(record=True) as second:
            _warnings_mod.simplefilter("always")
            _compute_metrics({"once_metric": broken}, output, targets)
        assert len(first) == 1
        assert len(second) == 0  # no second warning

    def test_training_still_completes_with_broken_metric(self):
        """train_epoch must complete normally even when a metric always fails."""
        def always_fail(output, targets):
            raise ValueError("always fails")

        loader = _make_tensor_loader()
        model = _simple_linear_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        import tgraphx.training as _tr
        _tr._warned_metrics.discard("bad")
        with _warnings_mod.catch_warnings(record=True):
            _warnings_mod.simplefilter("always")
            result = train_epoch(model, loader, opt, F.cross_entropy,
                                 device="cpu", metrics={"bad": always_fail})
        assert "loss" in result
        assert "bad" not in result


# =========================================================================== #
# BUG-04: _unpack_batch target squeeze — regression safety                     #
# =========================================================================== #

from tgraphx.training import _unpack_batch

class TestUnpackBatchSqueeze:
    """BUG-04: [B,1] integer targets squeezed; [B,1] float targets preserved."""

    def _make_batch(self, label_value, dtype):
        """Build a minimal GraphBatch with graph_labels of the given dtype."""
        from tgraphx import Graph, GraphBatch
        label = torch.tensor([label_value], dtype=dtype)
        # Provide a minimal edge_index so _unpack_batch can call .to(device).
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        g1 = Graph(torch.randn(4, 8), edge_index=ei, graph_label=label.clone())
        g2 = Graph(torch.randn(4, 8), edge_index=ei, graph_label=label.clone())
        return GraphBatch([g1, g2])

    def test_long_b1_squeezed_to_b(self):
        """[B,1] Long graph_labels must be squeezed to [B] for CrossEntropyLoss."""
        batch = self._make_batch(2, torch.long)
        _, _, targets = _unpack_batch(batch, torch.device("cpu"))
        # [2, 1] Long → squeezed → [2]
        assert targets.dim() == 1, (
            f"Expected [B] for Long targets, got shape {tuple(targets.shape)}"
        )

    def test_float_b1_preserved(self):
        """[B,1] float graph_labels must NOT be squeezed for MSELoss compatibility."""
        batch = self._make_batch(1.5, torch.float32)
        _, _, targets = _unpack_batch(batch, torch.device("cpu"))
        # [2, 1] float → must remain [2, 1]
        assert targets.dim() == 2 and targets.size(-1) == 1, (
            f"Expected [B,1] for float targets, got shape {tuple(targets.shape)}"
        )

    def test_graph_regression_mse_shape_matches(self):
        """Graph regression out_dim=1 with MSELoss: target and output shapes align."""
        from tgraphx import build_model, build_grid_graph, Graph, GraphBatch
        from tgraphx.core.dataloader import GraphDataLoader, GraphDataset

        model = build_model(
            task="graph_regression", layer="linear",
            in_shape=(8,), hidden_shape=(16,), num_layers=1, out_dim=1,
        )
        ei = build_grid_graph(3, 3, directed=False, self_loops=True)
        graphs = []
        for _ in range(4):
            nf = torch.randn(9, 8)
            gl = torch.tensor([1.5])  # float scalar label
            graphs.append(Graph(nf, ei, graph_label=gl))
        loader = GraphDataLoader(GraphDataset(graphs), batch_size=2, shuffle=False)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        # MSELoss must not receive mismatched shapes — train_epoch must not raise
        result = train_epoch(model, loader, opt, F.mse_loss, device="cpu")
        assert "loss" in result
        assert result["loss"] >= 0


# =========================================================================== #
# SEC-01: load_checkpoint safety (weights_only)                                #
# =========================================================================== #

import warnings as _warnings_mod

class TestCheckpointSafety:
    """SEC-01 regression tests for safe checkpoint loading."""

    @pytest.fixture
    def checkpoint(self, tmp_path):
        """Save a standard TGraphX checkpoint and return its path."""
        m = nn.Linear(4, 2)
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        # Populate optimizer state
        out = m(torch.randn(3, 4))
        out.sum().backward()
        opt.step()
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(m, opt, epoch=5, path=path, loss=0.42)
        return path, m

    def test_default_safe_load_works(self, checkpoint):
        """Default weights_only=True must succeed for standard checkpoints."""
        path, orig_model = checkpoint
        m2 = nn.Linear(4, 2)
        opt2 = torch.optim.SGD(m2.parameters(), lr=0.01)
        ep = load_checkpoint(m2, opt2, path=path)
        assert ep == 5
        x = torch.randn(3, 4)
        with torch.no_grad():
            assert torch.allclose(orig_model(x), m2(x))

    def test_default_safe_load_no_warning(self, checkpoint):
        """Safe load must not emit any security warning."""
        path, _ = checkpoint
        m2 = nn.Linear(4, 2)
        with _warnings_mod.catch_warnings(record=True) as w:
            _warnings_mod.simplefilter("always")
            load_checkpoint(m2, None, path=path)
        security_warns = [x for x in w if "weights_only" in str(x.message).lower()
                         or "untrusted" in str(x.message).lower()]
        assert not security_warns, f"Unexpected security warning on safe load: {security_warns}"

    def test_unsafe_mode_emits_warning(self, checkpoint):
        """weights_only=False must emit a UserWarning about untrusted files."""
        path, _ = checkpoint
        m2 = nn.Linear(4, 2)
        with _warnings_mod.catch_warnings(record=True) as w:
            _warnings_mod.simplefilter("always")
            load_checkpoint(m2, None, path=path, weights_only=False)
        assert any("weights_only=False" in str(wi.message) for wi in w), (
            "No security warning emitted when weights_only=False"
        )

    def test_unsafe_mode_still_loads_correctly(self, checkpoint):
        """Even in unsafe mode the checkpoint must load correctly."""
        path, orig_model = checkpoint
        m2 = nn.Linear(4, 2)
        with _warnings_mod.catch_warnings(record=True):
            _warnings_mod.simplefilter("always")
            ep = load_checkpoint(m2, None, path=path, weights_only=False)
        assert ep == 5
        x = torch.randn(3, 4)
        with torch.no_grad():
            assert torch.allclose(orig_model(x), m2(x))

    def test_safe_load_with_extra_fields(self, tmp_path):
        """Extra scalar/string fields stored by save_checkpoint survive safe load."""
        m = nn.Linear(4, 2)
        path = str(tmp_path / "extra.pt")
        save_checkpoint(m, None, epoch=3, path=path, tag="best", val_loss=0.15)
        m2 = nn.Linear(4, 2)
        # We cannot read extra fields through load_checkpoint, but loading must succeed.
        ep = load_checkpoint(m2, None, path=path)
        assert ep == 3

    def test_bad_path_raises_clearly(self, tmp_path):
        """A missing file must raise an error that mentions the path."""
        m = nn.Linear(4, 2)
        bad = str(tmp_path / "nonexistent.pt")
        with pytest.raises(Exception, match=r"nonexistent\.pt|No such file"):
            load_checkpoint(m, None, path=bad)

    def test_save_load_roundtrip_cpu(self, tmp_path):
        """Full round-trip on CPU: weights_only=True, map_location='cpu'."""
        m = nn.Linear(8, 4)
        path = str(tmp_path / "cpu.pt")
        save_checkpoint(m, None, epoch=0, path=path)
        m2 = nn.Linear(8, 4)
        ep = load_checkpoint(m2, None, path=path, map_location="cpu")
        assert ep == 0
        with torch.no_grad():
            x = torch.randn(2, 8)
            assert torch.allclose(m(x), m2(x))


# =========================================================================== #
# TRAIN-02: set_seed deterministic mode                                        #
# =========================================================================== #

class TestSetSeedDeterministic:
    """TRAIN-02 regression tests — deterministic=True flag."""

    def test_default_unchanged(self):
        """set_seed() with no extra args must behave exactly as before."""
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.equal(a, b)

    def test_deterministic_true_does_not_crash(self):
        """deterministic=True must never raise on any platform."""
        set_seed(42, deterministic=True)
        # Reset to avoid affecting other tests
        try:
            import torch.backends.cudnn as cudnn
            cudnn.deterministic = False
            cudnn.benchmark = True
        except AttributeError:
            pass

    def test_deterministic_sets_cudnn_flags(self):
        """deterministic=True must set cuDNN deterministic and disable benchmark."""
        try:
            import torch.backends.cudnn as cudnn
            # Record initial values so we can restore them after the test.
            orig_det = cudnn.deterministic
            orig_bench = cudnn.benchmark
        except AttributeError:
            pytest.skip("torch.backends.cudnn not available on this build")
        try:
            set_seed(0, deterministic=True)
            assert cudnn.deterministic is True
            assert cudnn.benchmark is False
        finally:
            cudnn.deterministic = orig_det
            cudnn.benchmark = orig_bench

    def test_deterministic_false_does_not_change_flags(self):
        """deterministic=False (default) must not touch cuDNN flags."""
        try:
            import torch.backends.cudnn as cudnn
        except AttributeError:
            pytest.skip("torch.backends.cudnn not available")
        orig_bench = cudnn.benchmark
        set_seed(42, deterministic=False)
        assert cudnn.benchmark == orig_bench  # unchanged

    def test_seed_still_deterministic_with_flag(self):
        """Seeding still produces reproducible tensors when deterministic=True."""
        set_seed(7, deterministic=True)
        a = torch.randn(10)
        set_seed(7, deterministic=True)
        b = torch.randn(10)
        assert torch.equal(a, b)
        # Reset flags
        try:
            import torch.backends.cudnn as cudnn
            cudnn.deterministic = False
            cudnn.benchmark = True
        except AttributeError:
            pass


# =========================================================================== #
# TRAIN-01: fit log_level=2 per-batch forwarding                               #
# =========================================================================== #

import io, sys

class TestFitLogLevel:
    """TRAIN-01 — fit() must forward log_level >= 2 to train_epoch."""

    def _make_loader(self):
        from torch.utils.data import DataLoader, TensorDataset
        torch.manual_seed(0)
        x = torch.randn(10, 4)
        y = torch.randint(0, 2, (10,))
        return DataLoader(TensorDataset(x, y), batch_size=5)

    def _make_model(self):
        return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(),
                                   torch.nn.Linear(8, 2))

    def test_log_level_0_silent(self):
        """log_level=0 must produce no stdout output."""
        model = self._make_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            fit(model, self._make_loader(), epochs=1, optimizer=opt,
                loss_fn=F.cross_entropy, device="cpu", log_level=0)
        finally:
            sys.stdout = old
        assert buf.getvalue() == ""

    def test_log_level_1_prints_epoch_summary(self):
        """log_level=1 must print an epoch summary line."""
        model = self._make_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            fit(model, self._make_loader(), epochs=1, optimizer=opt,
                loss_fn=F.cross_entropy, device="cpu", log_level=1)
        finally:
            sys.stdout = old
        output = buf.getvalue()
        assert "epoch" in output.lower() or "loss" in output.lower()

    def test_log_level_2_prints_batch_lines(self):
        """log_level=2 must produce per-batch [batch N] lines."""
        model = self._make_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            fit(model, self._make_loader(), epochs=1, optimizer=opt,
                loss_fn=F.cross_entropy, device="cpu", log_level=2)
        finally:
            sys.stdout = old
        output = buf.getvalue()
        assert "[batch" in output, (
            f"Expected per-batch [batch N] lines at log_level=2, got:\n{output!r}"
        )
