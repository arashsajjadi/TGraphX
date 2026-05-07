"""Tests for tgraphx.tracking — CSVLogger behaviour and schema."""
import csv
import os
import tempfile

import pytest

from tgraphx.tracking import CSVLogger


# ─────────────────────────────────────────────────────────────────────────────
# Silent / off-by-default behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestSilentDefault:

    def test_import_creates_no_files(self, tmp_path):
        """Importing tracking module must not touch the filesystem."""
        import tgraphx.tracking  # noqa: F401
        # tmp_path is empty before and after
        assert list(tmp_path.iterdir()) == []

    def test_no_logger_no_files(self, tmp_path):
        """Not creating a CSVLogger leaves logdir empty."""
        # Simulate a trivial training loop with no logger
        for epoch in range(3):
            loss = 1.0 / (epoch + 1)
            # no logger.log() call
        assert list(tmp_path.iterdir()) == []

    def test_logger_created_but_not_called(self, tmp_path):
        """Creating a CSVLogger for a new dir must not write until log() is called."""
        logdir = str(tmp_path / "run")
        logger = CSVLogger(logdir)
        # CSV file must NOT exist yet
        assert not os.path.exists(logger.path)
        logger.close()


# ─────────────────────────────────────────────────────────────────────────────
# CSVLogger — creation and schema
# ─────────────────────────────────────────────────────────────────────────────

class TestCSVLoggerCreation:

    def test_creates_logdir(self, tmp_path):
        logdir = str(tmp_path / "new_run")
        assert not os.path.isdir(logdir)
        logger = CSVLogger(logdir)
        logger.log(epoch=1, train_loss=0.5)
        assert os.path.isdir(logdir)
        logger.close()

    def test_creates_csv(self, tmp_path):
        logdir = str(tmp_path)
        logger = CSVLogger(logdir)
        logger.log(epoch=1, train_loss=0.5)
        logger.close()
        assert os.path.isfile(os.path.join(logdir, "metrics.csv"))

    def test_custom_filename(self, tmp_path):
        logdir = str(tmp_path)
        logger = CSVLogger(logdir, filename="custom.csv")
        logger.log(epoch=1)
        logger.close()
        assert os.path.isfile(os.path.join(logdir, "custom.csv"))

    def test_path_property(self, tmp_path):
        logdir = str(tmp_path)
        logger = CSVLogger(logdir)
        assert logger.path.endswith("metrics.csv")
        logger.close()


# ─────────────────────────────────────────────────────────────────────────────
# CSVLogger — header and row content
# ─────────────────────────────────────────────────────────────────────────────

class TestCSVSchema:

    def _read(self, logdir):
        path = os.path.join(logdir, "metrics.csv")
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    def test_header_written(self, tmp_path):
        logdir = str(tmp_path)
        with CSVLogger(logdir) as logger:
            logger.log(epoch=1, train_loss=0.9)
        path = os.path.join(logdir, "metrics.csv")
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "epoch" in header
        assert "train_loss" in header

    def test_timestamp_auto_added(self, tmp_path):
        """timestamp column is added automatically if not provided."""
        logdir = str(tmp_path)
        with CSVLogger(logdir) as logger:
            logger.log(epoch=1, train_loss=0.5)
        rows = self._read(logdir)
        assert "timestamp" in rows[0]
        assert rows[0]["timestamp"] != ""

    def test_timestamp_utc_format(self, tmp_path):
        """Timestamp must contain 'T' (ISO-8601) and '+' or 'Z' for UTC."""
        logdir = str(tmp_path)
        with CSVLogger(logdir) as logger:
            logger.log(epoch=1)
        rows = self._read(logdir)
        ts = rows[0]["timestamp"]
        assert "T" in ts
        assert ("+" in ts or "Z" in ts or ts.endswith("+00:00"))

    def test_multiple_rows(self, tmp_path):
        logdir = str(tmp_path)
        with CSVLogger(logdir) as logger:
            for i in range(1, 4):
                logger.log(epoch=i, train_loss=1.0 / i)
        rows = self._read(logdir)
        assert len(rows) == 3
        assert rows[0]["epoch"] == "1"
        assert rows[2]["epoch"] == "3"

    def test_numeric_values_stored(self, tmp_path):
        logdir = str(tmp_path)
        with CSVLogger(logdir) as logger:
            logger.log(epoch=5, train_loss=0.1234, accuracy=0.9876)
        rows = self._read(logdir)
        assert float(rows[0]["train_loss"]) == pytest.approx(0.1234)
        assert float(rows[0]["accuracy"]) == pytest.approx(0.9876)

    def test_custom_timestamp_not_doubled(self, tmp_path):
        """If user provides timestamp, it must not be added twice."""
        logdir = str(tmp_path)
        with CSVLogger(logdir) as logger:
            logger.log(timestamp="2025-01-01T00:00:00+00:00", epoch=1)
        path = os.path.join(logdir, "metrics.csv")
        with open(path, newline="") as f:
            header = next(csv.reader(f))
        assert header.count("timestamp") == 1


# ─────────────────────────────────────────────────────────────────────────────
# CSVLogger — dashboard schema compatibility
# ─────────────────────────────────────────────────────────────────────────────

class TestDashboardSchemaCompatibility:
    """Verify metrics.csv written by CSVLogger matches /api/metrics expectations."""

    def test_epoch_column_parseable_as_float(self, tmp_path):
        logdir = str(tmp_path)
        with CSVLogger(logdir) as logger:
            logger.log(epoch=3, train_loss=0.5)
        rows = self._read(logdir)
        # Dashboard tries float(val) for each cell
        assert float(rows[0]["epoch"]) == 3.0

    def test_schema_matches_dashboard_example(self, tmp_path):
        """Full schema used in training_with_dashboard.py example."""
        logdir = str(tmp_path)
        with CSVLogger(logdir) as logger:
            logger.log(
                epoch=1, step=50,
                train_loss=0.82, val_loss=0.91,
                accuracy=0.56, learning_rate=0.001,
            )
        rows = self._read(logdir)
        expected_cols = {"epoch", "step", "train_loss", "val_loss",
                         "accuracy", "learning_rate", "timestamp"}
        assert expected_cols.issubset(set(rows[0].keys()))

    def _read(self, logdir):
        path = os.path.join(logdir, "metrics.csv")
        with open(path, newline="") as f:
            return list(csv.DictReader(f))


# ─────────────────────────────────────────────────────────────────────────────
# CSVLogger — append-mode (resume across runs)
# ─────────────────────────────────────────────────────────────────────────────

class TestCSVLoggerAppend:

    def test_appends_to_existing_file(self, tmp_path):
        logdir = str(tmp_path)
        with CSVLogger(logdir) as logger:
            logger.log(epoch=1, train_loss=0.5)
        # Second logger picks up existing headers
        with CSVLogger(logdir) as logger2:
            logger2.log(epoch=2, train_loss=0.4)
        path = os.path.join(logdir, "metrics.csv")
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[1]["epoch"] == "2"

    def test_context_manager(self, tmp_path):
        logdir = str(tmp_path)
        with CSVLogger(logdir) as logger:
            logger.log(epoch=1)
        assert os.path.isfile(os.path.join(logdir, "metrics.csv"))


# =========================================================================== #
# TensorBoardLogger                                                             #
# =========================================================================== #

import importlib

class TestTensorBoardLogger:

    def test_not_imported_at_module_level(self):
        """TensorBoard must not be imported when tgraphx.tracking is imported."""
        import sys
        # Re-import tracking in a subprocess to get a clean import state
        import subprocess
        result = subprocess.run(
            [sys.executable, "-c",
             "import tgraphx.tracking; import sys; "
             "assert 'tensorboard' not in sys.modules, "
             "'tensorboard was imported at module level'"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_import_error_when_missing(self, monkeypatch):
        """TensorBoardLogger raises clear ImportError when tensorboard absent."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch.utils.tensorboard":
                raise ImportError("tensorboard not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        # Need to re-import the class to trigger the lazy import inside __init__
        from tgraphx.tracking import TensorBoardLogger
        with pytest.raises(ImportError, match="pip install tensorboard"):
            TensorBoardLogger("/tmp/tb_test_dir")

    def test_tensorboard_logger_if_available(self, tmp_path):
        """If tensorboard is installed, TensorBoardLogger works correctly."""
        try:
            import torch.utils.tensorboard  # noqa: F401
        except ImportError:
            pytest.skip("tensorboard not installed")

        from tgraphx.tracking import TensorBoardLogger

        logdir = str(tmp_path / "tb_events")
        with TensorBoardLogger(logdir) as tb:
            tb.log(epoch=0, train_loss=0.9, val_loss=0.85)
            tb.log(epoch=1, train_loss=0.7, val_loss=0.65)
            tb.log_scalar("custom/metric", 1.23, step=5)
            tb.log_metrics({"a": 0.1, "b": 0.2}, step=10)

        import os
        assert os.path.isdir(logdir)
        files = list(os.listdir(logdir))
        assert any("tfevents" in f or ".v2" in f or f.startswith("events") for f in files), \
            f"No TensorBoard event files found: {files}"

    def test_context_manager_closes(self, tmp_path):
        """TensorBoardLogger closes cleanly via context manager."""
        try:
            import torch.utils.tensorboard  # noqa: F401
        except ImportError:
            pytest.skip("tensorboard not installed")

        from tgraphx.tracking import TensorBoardLogger

        logdir = str(tmp_path / "tb_cm")
        tb = TensorBoardLogger(logdir)
        tb.log(epoch=0, train_loss=0.5)
        tb.close()  # explicit close
        tb.close()  # double close must not raise


# =========================================================================== #
# TRACK-01: TensorBoardLogger.log zero-step / zero-epoch fix                   #
# =========================================================================== #

class _FakeWriter:
    """Minimal SummaryWriter stand-in that records add_scalar calls."""

    def __init__(self):
        self.calls: list[tuple] = []  # (tag, value, global_step)

    def add_scalar(self, tag: str, value: float, global_step: int) -> None:
        self.calls.append((tag, float(value), int(global_step)))

    def close(self) -> None:
        pass


def _make_logger(fake_writer: _FakeWriter):
    """Build a TensorBoardLogger backed by a fake writer without TensorBoard."""
    from tgraphx.tracking import TensorBoardLogger
    logger = TensorBoardLogger.__new__(TensorBoardLogger)
    logger._logdir = "/tmp/fake"
    logger._writer = fake_writer
    logger._step = 0
    return logger


class TestTensorBoardLoggerStepResolution:
    """TRACK-01 regression tests — epoch=0 / step=0 must not be skipped."""

    def test_epoch_zero_records_global_step_zero(self):
        """epoch=0 must map to global_step=0, not to the internal counter."""
        fw = _FakeWriter()
        logger = _make_logger(fw)
        logger.log(epoch=0, train_loss=0.5)
        assert len(fw.calls) == 1
        tag, val, gstep = fw.calls[0]
        assert tag == "train_loss"
        assert gstep == 0, f"Expected global_step=0, got {gstep}"

    def test_step_zero_records_global_step_zero(self):
        """step=0 must map to global_step=0 when epoch is absent."""
        fw = _FakeWriter()
        logger = _make_logger(fw)
        logger.log(step=0, train_loss=0.5)
        tag, val, gstep = fw.calls[0]
        assert gstep == 0, f"Expected global_step=0, got {gstep}"

    def test_epoch_nonzero_correct(self):
        fw = _FakeWriter()
        logger = _make_logger(fw)
        logger.log(epoch=3, train_loss=0.3)
        assert fw.calls[0][2] == 3

    def test_auto_step_increments(self):
        """Without explicit epoch/step the internal counter advances."""
        fw = _FakeWriter()
        logger = _make_logger(fw)
        logger.log(train_loss=0.9)   # step 0
        logger.log(train_loss=0.8)   # step 1
        logger.log(train_loss=0.7)   # step 2
        steps = [c[2] for c in fw.calls]
        assert steps == [0, 1, 2]

    def test_explicit_epoch_does_not_advance_auto_counter(self):
        """Auto-counter must stay at 0 after two explicit epoch calls."""
        fw = _FakeWriter()
        logger = _make_logger(fw)
        logger.log(epoch=0, train_loss=0.9)
        logger.log(epoch=1, train_loss=0.8)
        assert logger._step == 0, (
            f"_step should not advance on explicit epoch; got {logger._step}"
        )

    def test_mixed_auto_then_explicit_epoch_zero(self):
        """
        Reproduce the original TRACK-01 scenario:
          1. Two auto-step logs (counter reaches 2).
          2. epoch=0 log — must use step 0, not the auto counter (2).
          3. epoch=2 log — must use step 2, not collide.
        """
        fw = _FakeWriter()
        logger = _make_logger(fw)
        logger.log(train_loss=0.9)            # auto step 0
        logger.log(train_loss=0.8)            # auto step 1
        logger.log(epoch=0, train_loss=0.5)   # explicit step 0  ← bug repro
        logger.log(epoch=2, train_loss=0.3)   # explicit step 2

        # auto logs
        assert fw.calls[0][2] == 0
        assert fw.calls[1][2] == 1
        # explicit epoch=0 must not use _step (which is now 2)
        assert fw.calls[2][2] == 0, (
            f"epoch=0 was recorded at step {fw.calls[2][2]} instead of 0 "
            "(TRACK-01 regression)"
        )
        # explicit epoch=2
        assert fw.calls[3][2] == 2

    def test_epoch_key_missing_falls_to_step_key(self):
        """'step' key is used when 'epoch' is absent."""
        fw = _FakeWriter()
        logger = _make_logger(fw)
        logger.log(step=5, val_loss=0.4)
        assert fw.calls[0][2] == 5

    def test_epoch_none_falls_to_step(self):
        """epoch=None must be treated as absent; step key used instead."""
        fw = _FakeWriter()
        logger = _make_logger(fw)
        logger.log(epoch=None, step=7, val_loss=0.3)
        assert fw.calls[0][2] == 7

    def test_step_none_falls_to_auto(self):
        """step=None with no epoch key must use auto counter."""
        fw = _FakeWriter()
        logger = _make_logger(fw)
        logger.log(step=None, val_loss=0.3)
        assert fw.calls[0][2] == 0   # auto starts at 0
        assert logger._step == 1      # counter advanced

    def test_timestamp_never_logged_as_scalar(self):
        """'timestamp' key must be silently skipped."""
        fw = _FakeWriter()
        logger = _make_logger(fw)
        logger.log(epoch=1, timestamp="2025-01-01T00:00:00Z", train_loss=0.5)
        tags = [c[0] for c in fw.calls]
        assert "timestamp" not in tags
        assert "train_loss" in tags

    def test_non_numeric_values_skipped(self):
        """Non-numeric metric values must be silently skipped."""
        fw = _FakeWriter()
        logger = _make_logger(fw)
        logger.log(epoch=1, label="graph_clf", train_loss=0.5)
        tags = [c[0] for c in fw.calls]
        assert "label" not in tags
        assert "train_loss" in tags
