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
