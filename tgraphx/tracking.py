"""Metric tracking utilities for TGraphX.

What IS provided
----------------
CSVLogger           — appends metric rows to ``metrics.csv`` (dashboard-
                      compatible schema).  Nothing is written until you
                      call :meth:`CSVLogger.log`.
TensorBoardLogger   — optional TensorBoard logger backed by
                      ``torch.utils.tensorboard.SummaryWriter``.
                      Requires: ``pip install tensorboard`` or
                      ``pip install "tgraphx[tracking]"``.
                      TensorBoard is imported lazily — only when you
                      instantiate :class:`TensorBoardLogger`.

What is NOT provided
--------------------
MLflowLogger is not implemented.  If you need MLflow, use the
``mlflow`` client directly: ``pip install mlflow``.

Silent / off-by-default contract
---------------------------------
* Importing ``tgraphx`` or ``tgraphx.tracking`` creates **no files**
  and does **not** import TensorBoard.
* A ``CSVLogger`` creates its logdir and CSV only on the first call to
  :meth:`CSVLogger.log`.
* A ``TensorBoardLogger`` creates its event directory only when you
  instantiate it (which triggers the lazy TensorBoard import).
* Not passing a logger to any training utility writes nothing.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# CSVLogger
# ─────────────────────────────────────────────────────────────────────────────

class CSVLogger:
    """Append-mode CSV metric logger compatible with the TGraphX dashboard.

    Schema written to ``metrics.csv`` (example)::

        epoch,step,train_loss,val_loss,accuracy,learning_rate,timestamp
        1,100,0.82,0.91,0.56,0.001,2025-01-01T12:00:00+00:00

    * Timestamps are ISO-8601 UTC.
    * Column order is determined by the first call to :meth:`log`.
    * Subsequent calls must pass a **consistent** set of keys; an extra
      key on row 2 is silently skipped (CSV header stays fixed after
      the first write).
    * ``None`` values are written as an empty string.

    Args:
        logdir:   Directory where ``metrics.csv`` will be written.
        filename: CSV filename (default ``"metrics.csv"``).

    Example::

        from tgraphx.tracking import CSVLogger
        logger = CSVLogger("runs/my_run")
        for epoch in range(10):
            # ... training ...
            logger.log(epoch=epoch, train_loss=loss, val_loss=val_loss)
        logger.close()
    """

    def __init__(self, logdir: str, filename: str = "metrics.csv") -> None:
        self._logdir = logdir
        self._path = os.path.join(logdir, filename)
        self._headers: Optional[list] = None
        self._file = None
        self._writer = None
        # Detect existing file so we can continue from a previous run.
        if os.path.isfile(self._path):
            with open(self._path, "r", newline="") as f:
                reader = csv.reader(f)
                try:
                    self._headers = next(reader)
                except StopIteration:
                    self._headers = None

    # ── Public API ────────────────────────────────────────────────────────────

    def log(self, **metrics: Any) -> None:
        """Append one metric row.

        A ``"timestamp"`` column (UTC ISO-8601) is always prepended
        unless you explicitly pass ``timestamp=...``.

        Args:
            **metrics: Metric name → value pairs.  Values must be
                numbers, strings, or ``None``.
        """
        if "timestamp" not in metrics:
            metrics = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       **metrics}

        if self._headers is None:
            # First write — establish headers and open the file.
            self._headers = list(metrics.keys())
            os.makedirs(self._logdir, exist_ok=True)
            self._file = open(self._path, "a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=self._headers,
                extrasaction="ignore",
            )
            # Write header only if the file was empty / new.
            if os.path.getsize(self._path) == 0:
                self._writer.writeheader()
        elif self._file is None:
            # Re-open for appending (file already existed with headers).
            os.makedirs(self._logdir, exist_ok=True)
            self._file = open(self._path, "a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=self._headers,
                extrasaction="ignore",
            )

        # Convert None → ""
        row = {k: ("" if v is None else v) for k, v in metrics.items()}
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @property
    def path(self) -> str:
        """Absolute path to the CSV file."""
        return os.path.abspath(self._path)

    def __repr__(self) -> str:
        return f"CSVLogger(path={self._path!r})"


# ─────────────────────────────────────────────────────────────────────────────
# TensorBoardLogger
# ─────────────────────────────────────────────────────────────────────────────

class TensorBoardLogger:
    """TensorBoard logger backed by ``torch.utils.tensorboard.SummaryWriter``.

    Requires TensorBoard to be installed::

        pip install tensorboard
        # or
        pip install "tgraphx[tracking]"

    Nothing is imported or written until you create an instance.

    Args:
        logdir:  Directory for TensorBoard event files.
        comment: Optional suffix appended to the auto-generated run name.

    Example::

        from tgraphx.tracking import TensorBoardLogger

        with TensorBoardLogger("runs/tb_run") as tb:
            for epoch in range(10):
                tb.log(epoch=epoch, train_loss=loss, val_accuracy=acc)

        # Or use directly with fit():
        model_history = fit(model, train_loader, val_loader=val_loader,
                            epochs=20, optimizer=opt, loss_fn=criterion,
                            logger=TensorBoardLogger("runs/tb"))
    """

    def __init__(self, logdir: str, comment: str = "") -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError(
                "TensorBoard is not installed. "
                "Install it with:  pip install tensorboard\n"
                "Or install the optional tracking extras: "
                "pip install 'tgraphx[tracking]'"
            ) from exc

        self._logdir = logdir
        self._writer = SummaryWriter(log_dir=logdir, comment=comment)
        self._step = 0

    # ── Scalar logging ────────────────────────────────────────────────────────

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Write one scalar value."""
        self._writer.add_scalar(tag, value, global_step=step)

    def log_metrics(self, metrics: dict, step: int) -> None:
        """Write multiple scalar values at ``step``."""
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self._writer.add_scalar(k, float(v), global_step=step)

    def log(self, **kwargs: Any) -> None:
        """Write metrics; compatible with the :class:`CSVLogger` ``log()`` interface.

        The ``"epoch"`` or ``"step"`` key is used as the TensorBoard
        global step.  ``"timestamp"`` is ignored (TensorBoard records
        wall-clock time internally).  All other numeric values are logged
        as scalars.

        Args:
            **kwargs: Metric key→value pairs.  Non-numeric values and
                ``"timestamp"`` are silently skipped.
        """
        step = kwargs.get("epoch") or kwargs.get("step") or self._step
        self._step += 1
        for k, v in kwargs.items():
            if k in ("epoch", "step", "timestamp"):
                continue
            if isinstance(v, (int, float)):
                self._writer.add_scalar(k, float(v), global_step=int(step))

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Flush and close the underlying ``SummaryWriter``."""
        self._writer.close()

    def __enter__(self) -> "TensorBoardLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"TensorBoardLogger(logdir={self._logdir!r})"


__all__ = ["CSVLogger", "TensorBoardLogger"]
