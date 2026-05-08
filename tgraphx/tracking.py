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
MLflowLogger        — optional MLflow logger.
                      Requires: ``pip install mlflow`` or
                      ``pip install "tgraphx[mlflow]"``.
                      mlflow is imported lazily — only when you
                      instantiate :class:`MLflowLogger`.

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

        Step resolution (explicit zero is respected):

        1. If ``"epoch"`` is present and not ``None``, use ``int(epoch)``.
        2. Else if ``"step"`` is present and not ``None``, use ``int(step)``.
        3. Otherwise use the internal auto-step counter and advance it by 1.

        The internal counter is **not** advanced when an explicit
        ``epoch``/``step`` is provided, so callers that mix both styles
        stay in sync.

        Args:
            **kwargs: Metric key→value pairs.  Non-numeric values and
                ``"timestamp"`` are silently skipped.
        """
        if "epoch" in kwargs and kwargs["epoch"] is not None:
            step = int(kwargs["epoch"])
        elif "step" in kwargs and kwargs["step"] is not None:
            step = int(kwargs["step"])
        else:
            step = self._step
            self._step += 1
        for k, v in kwargs.items():
            if k in ("epoch", "step", "timestamp"):
                continue
            if isinstance(v, (int, float)):
                self._writer.add_scalar(k, float(v), global_step=step)

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


class MLflowLogger:
    """Optional MLflow metric logger.

    Wraps ``mlflow`` to provide an interface consistent with
    :class:`CSVLogger` and :class:`TensorBoardLogger`.  mlflow is imported
    lazily — only when this class is instantiated — so the base TGraphX
    install does not require mlflow.

    Requires: ``pip install mlflow`` or ``pip install "tgraphx[mlflow]"``.

    Off-by-default contract
    -----------------------
    * Importing ``tgraphx`` or ``tgraphx.tracking`` does **not** import
      mlflow.
    * No MLflow run is started until you call :meth:`__enter__` (context
      manager) or :meth:`start`.
    * All writes happen only through explicit :meth:`log` calls.

    Example::

        from tgraphx.tracking import MLflowLogger

        with MLflowLogger(run_name="my_run", experiment="gnn_exp") as mlf:
            for epoch in range(10):
                loss = train(...)
                mlf.log(epoch=epoch, train_loss=loss)

    Alternatively, use with ``fit()``::

        history = fit(model, train_loader, logger=mlf, ...)
    """

    def __init__(
        self,
        run_name: Optional[str] = None,
        experiment: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tags: Optional[dict] = None,
    ) -> None:
        """
        Args:
            run_name: Optional MLflow run name.
            experiment: MLflow experiment name.  If ``None``, uses the
                MLflow default experiment.
            tracking_uri: MLflow tracking URI.  ``None`` uses the
                environment-variable / default configured URI.
            tags: Optional dict of string tags attached to the run.
        """
        try:
            import mlflow as _mlflow  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "MLflowLogger requires mlflow.  Install it with:\n"
                "  pip install mlflow\n"
                "or:\n"
                "  pip install \"tgraphx[mlflow]\""
            ) from exc
        self._run_name = run_name
        self._experiment = experiment
        self._tracking_uri = tracking_uri
        self._tags = tags or {}
        self._run = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> "MLflowLogger":
        """Start the MLflow run.  Called automatically by ``__enter__``."""
        import mlflow
        if self._tracking_uri is not None:
            mlflow.set_tracking_uri(self._tracking_uri)
        if self._experiment is not None:
            mlflow.set_experiment(self._experiment)
        self._run = mlflow.start_run(
            run_name=self._run_name,
            tags=self._tags if self._tags else None,
        )
        return self

    def close(self) -> None:
        """End the MLflow run.  Called automatically by ``__exit__``."""
        import mlflow
        if self._run is not None:
            mlflow.end_run()
            self._run = None

    def __enter__(self) -> "MLflowLogger":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #

    def log(self, **metrics: Any) -> None:
        """Log a dict of scalar metrics.

        Compatible with the :class:`CSVLogger` / :class:`TensorBoardLogger`
        ``log(**kwargs)`` interface.  Non-numeric values are cast to
        ``float`` where possible and skipped otherwise.

        ``epoch`` and ``step`` keys are forwarded as the MLflow ``step``
        (``epoch`` takes precedence if both are present).

        Example::

            mlf.log(epoch=3, train_loss=0.42, val_acc=0.87)
        """
        import mlflow
        if self._run is None:
            raise RuntimeError(
                "MLflowLogger.log() called before start().  "
                "Use as a context manager or call .start() first."
            )
        step = None
        for k in ("epoch", "step"):
            if k in metrics:
                try:
                    step = int(metrics[k])
                except (TypeError, ValueError):
                    pass
                break

        metric_dict: dict[str, float] = {}
        for k, v in metrics.items():
            if k in ("epoch", "step"):
                continue
            try:
                metric_dict[k] = float(v)
            except (TypeError, ValueError):
                pass

        if metric_dict:
            mlflow.log_metrics(metric_dict, step=step)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar metric (compatible with TensorBoardLogger API)."""
        import mlflow
        if self._run is None:
            raise RuntimeError(
                "MLflowLogger.log_scalar() called before start()."
            )
        mlflow.log_metric(tag, float(value), step=step)

    @property
    def run_id(self) -> Optional[str]:
        """Active MLflow run ID, or ``None`` if not started."""
        return self._run.info.run_id if self._run is not None else None

    def __repr__(self) -> str:
        status = f"run_id={self.run_id!r}" if self._run else "not started"
        return f"MLflowLogger(run_name={self._run_name!r}, {status})"


def write_graph_stats(graph_obj: Any, path: str) -> None:
    """Write a precomputed graph-statistics JSON file readable by the dashboard.

    The dashboard's ``/api/graph_stats`` endpoint serves this file from
    ``{logdir}/graph_stats.json``.  Precomputing the statistics in your
    training script (rather than on the dashboard server) keeps the server
    fast and free of heavy graph-library dependencies.

    Supported fields (all optional, pass what you have):

    .. code-block:: python

        write_graph_stats(
            {
                "num_nodes": 100,
                "num_edges": 400,
                "directed": False,
                "self_loops": True,
                "avg_degree": 4.0,
                "min_degree": 1,
                "max_degree": 8,
                "density": 0.04,
                "connected_components": 1,
                "isolated_nodes": 0,
            },
            path="runs/demo/graph_stats.json",
        )

    Args:
        graph_obj: Either a :class:`~tgraphx.Graph` instance or a plain
            ``dict`` of pre-computed statistics. If a ``Graph`` is passed,
            ``num_nodes`` and ``num_edges`` are extracted automatically;
            all other fields are not computed here (pass them explicitly
            or compute them yourself).
        path: Destination file path.  Parent directory must exist.
    """
    import json as _json
    import os as _os

    if isinstance(graph_obj, dict):
        stats: Any = dict(graph_obj)
    else:
        # Duck-typed Graph support: pull the two cheapest attributes.
        stats = {}
        for attr in ("num_nodes", "num_edges"):
            if hasattr(graph_obj, attr):
                val = getattr(graph_obj, attr)
                stats[attr] = int(val) if hasattr(val, "__int__") else val

    parent = _os.path.dirname(_os.path.abspath(path))
    _os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(stats, f, ensure_ascii=False, indent=2)


__all__ = ["CSVLogger", "TensorBoardLogger", "write_graph_stats"]
