"""Experiment callbacks: early stopping, checkpointing, logging.

All callbacks share a tiny lifecycle: ``on_train_begin``,
``on_epoch_end``, ``on_train_end``.  They are always passed the same
``RunState`` object so they can read/write state without globals.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunState:
    """Mutable state shared between :class:`Runner` and callbacks."""

    run_dir: Path
    history: List[Dict[str, float]] = field(default_factory=list)
    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    best_state: Optional[Dict[str, Any]] = None
    should_stop: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)


class Callback:
    """Base callback (no-op by default)."""

    def on_train_begin(self, state: RunState) -> None:
        pass

    def on_epoch_end(self, state: RunState, epoch: int, metrics: Dict[str, float]) -> None:
        pass

    def on_train_end(self, state: RunState) -> None:
        pass


# ── Early stopping ───────────────────────────────────────────────────────────


class EarlyStopping(Callback):
    """Stop training when the monitored metric stops improving.

    Args:
        monitor: Metric key (e.g. ``"val_loss"`` or ``"train_loss"``).
        patience: Number of epochs without improvement before stopping.
        mode: ``"min"`` (default) or ``"max"``.
        min_delta: Minimum change in the monitored metric to qualify as
            an improvement.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max'; got {mode!r}")
        self.monitor = monitor
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self._wait = 0
        self._best: Optional[float] = None

    def _is_improvement(self, value: float) -> bool:
        if self._best is None:
            return True
        if self.mode == "min":
            return value < self._best - self.min_delta
        return value > self._best + self.min_delta

    def on_epoch_end(self, state: RunState, epoch: int, metrics: Dict[str, float]) -> None:
        if self.monitor not in metrics:
            # Fall back gracefully — many runs only report train_loss.
            return
        value = float(metrics[self.monitor])
        if self._is_improvement(value):
            self._best = value
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                state.should_stop = True


# ── Checkpoints ──────────────────────────────────────────────────────────────


class ModelCheckpoint(Callback):
    """Save best/latest model + optimizer state to disk.

    Saves under ``state.run_dir / 'checkpoints'`` only — no other path.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best: bool = True,
        save_latest: bool = True,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max'; got {mode!r}")
        self.monitor = monitor
        self.mode = mode
        self.save_best = bool(save_best)
        self.save_latest = bool(save_latest)
        self._best: Optional[float] = None

    def _is_better(self, value: float) -> bool:
        if self._best is None:
            return True
        return value < self._best if self.mode == "min" else value > self._best

    def on_epoch_end(self, state: RunState, epoch: int, metrics: Dict[str, float]) -> None:
        # Lazy import torch only when actually saving.
        import torch
        ckpt_dir = state.run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        model = state.extras.get("model")
        optimizer = state.extras.get("optimizer")
        if model is None:
            return  # Nothing to save yet.

        payload = {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
        if self.save_latest:
            torch.save(payload, ckpt_dir / "latest.pt")

        if self.save_best and self.monitor in metrics:
            value = float(metrics[self.monitor])
            if self._is_better(value):
                self._best = value
                state.best_metric = value
                state.best_epoch = int(epoch)
                torch.save(payload, ckpt_dir / "best.pt")


# ── CSV logger callback ──────────────────────────────────────────────────────


class CSVLoggerCallback(Callback):
    """Append per-epoch metrics to ``run_dir/metrics.csv``.

    The CSV format matches what the TGraphX dashboard already reads
    (``epoch`` plus arbitrary metric columns; ISO-8601 UTC timestamp
    if requested).
    """

    def __init__(self, filename: str = "metrics.csv", with_timestamp: bool = True) -> None:
        self.filename = filename
        self.with_timestamp = bool(with_timestamp)
        self._fieldnames: Optional[List[str]] = None

    def on_epoch_end(self, state: RunState, epoch: int, metrics: Dict[str, float]) -> None:
        path = state.run_dir / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        row: Dict[str, Any] = {"epoch": int(epoch)}
        for k, v in metrics.items():
            row[k] = float(v)
        if self.with_timestamp:
            row["timestamp"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(),
            )

        is_new = self._fieldnames is None
        if is_new:
            self._fieldnames = list(row.keys())
        # Preserve column order across epochs (drop new keys quietly).
        write_row = {k: row.get(k, "") for k in self._fieldnames}

        mode = "w" if is_new else "a"
        with path.open(mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if is_new:
                writer.writeheader()
            writer.writerow(write_row)


# ── Learning-rate logger ─────────────────────────────────────────────────────


class LearningRateLogger(Callback):
    """Stamp the current optimizer LR into the metrics dict each epoch."""

    def on_epoch_end(self, state: RunState, epoch: int, metrics: Dict[str, float]) -> None:
        opt = state.extras.get("optimizer")
        if opt is None or not opt.param_groups:
            return
        metrics["lr"] = float(opt.param_groups[0]["lr"])
