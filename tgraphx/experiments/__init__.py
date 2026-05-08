"""TGraphX experiment manager (v0.3.0).

A lightweight config-driven training runner.  Importing this module
does not start any background work; everything happens inside
:meth:`Runner.fit`.

Public surface:

* :func:`load_config`
* :class:`ExperimentConfig`
* :class:`Runner`
* :class:`GridRunner`
* :func:`expand_grid`
* :func:`summarize_runs` / :func:`write_markdown_report`
* Built-in callbacks: :class:`Callback`, :class:`EarlyStopping`,
  :class:`ModelCheckpoint`, :class:`CSVLoggerCallback`,
  :class:`LearningRateLogger`.
"""
from __future__ import annotations

from .callbacks import (
    Callback,
    CSVLoggerCallback,
    EarlyStopping,
    LearningRateLogger,
    ModelCheckpoint,
    RunState,
)
from .checkpoints import load_checkpoint, save_checkpoint
from .config import (
    CallbackConfig,
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    TransformConfig,
    load_config,
)
from .grid import GridRunner, expand_grid
from .runner import Runner
from .summarize import (
    discover_runs,
    summarize_runs,
    write_markdown_report,
    write_summary_csv,
)


__all__ = [
    # Config
    "load_config",
    "ExperimentConfig",
    "DatasetConfig",
    "ModelConfig",
    "TrainingConfig",
    "TransformConfig",
    "CallbackConfig",
    # Runner
    "Runner",
    "GridRunner",
    "expand_grid",
    # Callbacks
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "CSVLoggerCallback",
    "LearningRateLogger",
    "RunState",
    # Checkpoints
    "save_checkpoint",
    "load_checkpoint",
    # Summarize
    "discover_runs",
    "summarize_runs",
    "write_markdown_report",
    "write_summary_csv",
]
