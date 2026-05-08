"""Experiment runner — turns an :class:`ExperimentConfig` into a full training run."""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .callbacks import (
    Callback,
    CSVLoggerCallback,
    EarlyStopping,
    LearningRateLogger,
    ModelCheckpoint,
    RunState,
)
from .checkpoints import load_checkpoint, save_checkpoint
from .config import ExperimentConfig

__all__ = ["Runner"]


_BUILTIN_CALLBACKS = {
    "csv_logger": CSVLoggerCallback,
    "early_stopping": EarlyStopping,
    "model_checkpoint": ModelCheckpoint,
    "lr_logger": LearningRateLogger,
}


def _build_callbacks(specs: list) -> List[Callback]:
    cbs: List[Callback] = []
    for spec in specs:
        cls = _BUILTIN_CALLBACKS.get(spec.name)
        if cls is None:
            raise ValueError(
                f"Unknown callback {spec.name!r}.  Built-ins: "
                f"{sorted(_BUILTIN_CALLBACKS)}"
            )
        cbs.append(cls(**spec.kwargs))
    return cbs


def _build_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {name!r}; expected adam/adamw/sgd.")


def _build_loss(name: str, task: str):
    name = name.lower()
    if name == "cross_entropy":
        return nn.functional.cross_entropy
    if name in ("mse", "mse_loss"):
        return nn.functional.mse_loss
    if name in ("mae", "l1", "l1_loss"):
        return nn.functional.l1_loss
    if name == "bce_with_logits":
        return nn.functional.binary_cross_entropy_with_logits
    if name == "auto":
        return nn.functional.cross_entropy if "classification" in task else nn.functional.mse_loss
    raise ValueError(f"Unknown loss {name!r}; expected cross_entropy/mse/mae/bce_with_logits/auto.")


def _build_transform(transforms_spec) -> Optional[Any]:
    if not transforms_spec:
        return None
    from tgraphx import transforms as T

    pipeline = []
    for spec in transforms_spec:
        cls = getattr(T, spec.name, None)
        if cls is None:
            raise ValueError(
                f"Unknown transform {spec.name!r}; not found in tgraphx.transforms."
            )
        pipeline.append(cls(**spec.kwargs))
    return T.Compose(pipeline)


def _resolve_device(name: str) -> torch.device:
    name = name.lower()
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Runner ───────────────────────────────────────────────────────────────────


class Runner:
    """Train a model from an :class:`ExperimentConfig`.

    Use:

    .. code-block:: python

        from tgraphx.experiments import Runner, load_config

        cfg = load_config("configs/synthetic_patch.yaml")
        runner = Runner(cfg)
        history = runner.fit()
    """

    def __init__(
        self,
        config: ExperimentConfig,
        run_dir: Optional[str | Path] = None,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        self.config = config
        # Resolve run_dir priority: explicit arg → config.run_dir → runs/<run_name>/<timestamp>
        if run_dir is not None:
            self.run_dir = Path(run_dir)
        elif config.run_dir:
            self.run_dir = Path(config.run_dir)
        else:
            stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            self.run_dir = Path("runs") / config.run_name / stamp
        self.run_dir = self.run_dir.expanduser()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._extra_callbacks = list(callbacks or [])

    # ── Setup helpers ────────────────────────────────────────────────────────

    def _save_provenance(self) -> None:
        """Write run_metadata.json + experiment_config.json into run_dir."""
        from tgraphx import __version__

        meta = {
            "run_name": self.config.run_name,
            "status": "running",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tgraphx_version": __version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "device": self.config.training.device,
            "seed": self.config.seed,
        }
        (self.run_dir / "run_metadata.json").write_text(
            json.dumps(meta, indent=2),
        )
        (self.run_dir / "experiment_config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2),
        )

    def _build_dataset(self):
        from tgraphx.datasets import get_dataset

        ds_cfg = self.config.dataset
        ds = get_dataset(ds_cfg.name, **ds_cfg.kwargs)
        # Apply transform pipeline by setting ``ds.transform`` (read by __getitem__).
        transform = _build_transform(self.config.transforms)
        if transform is not None and hasattr(ds, "transform"):
            ds.transform = transform
        return ds

    def _build_model(self) -> nn.Module:
        from tgraphx import build_model
        m = self.config.model
        kwargs: Dict[str, Any] = dict(m.extra)
        if m.in_shape is not None:
            kwargs["in_shape"] = tuple(m.in_shape)
        if m.hidden_shape is not None:
            kwargs["hidden_shape"] = tuple(m.hidden_shape)
        if m.num_classes is not None:
            kwargs["num_classes"] = m.num_classes
        if m.out_dim is not None:
            kwargs["out_dim"] = m.out_dim
        kwargs["pooling"] = m.pooling
        return build_model(
            task=m.task,
            layer=m.layer,
            num_layers=m.num_layers,
            **kwargs,
        )

    def _build_callbacks(self) -> List[Callback]:
        cbs = _build_callbacks(self.config.callbacks) + list(self._extra_callbacks)
        # Always have a CSV logger so dashboard sees something.
        if not any(isinstance(c, CSVLoggerCallback) for c in cbs):
            cbs.insert(0, CSVLoggerCallback())
        return cbs

    # ── Train loop ───────────────────────────────────────────────────────────

    def fit(self) -> List[Dict[str, float]]:
        """Run training; return the per-epoch metrics history."""
        from tgraphx import GraphBatch

        _set_seed(self.config.seed)
        self._save_provenance()
        device = _resolve_device(self.config.training.device)
        ds = self._build_dataset()

        # Build a single full-batch (suits the small synthetic datasets we
        # ship; user code can replace this with a real loader by subclassing).
        items = [ds[i] for i in range(len(ds))]
        if not items:
            raise ValueError(f"Dataset {self.config.dataset.name!r} produced 0 items.")
        first = items[0]
        is_single_graph = (
            len(items) == 1
            and not hasattr(first, "graph_label")  # node task, etc.
        ) or len(items) == 1
        # Either way, we work with either a single Graph or a GraphBatch.
        if len(items) > 1:
            batch = GraphBatch(items)
            batch_to_use = batch.to(device)
        else:
            batch_to_use = first.to(device)

        model = self._build_model().to(device)
        optimizer = _build_optimizer(
            self.config.training.optimizer,
            model.parameters(),
            self.config.training.lr,
            self.config.training.weight_decay,
        )
        loss_fn = _build_loss(self.config.training.loss, self.config.model.task)

        state = RunState(run_dir=self.run_dir)
        state.extras.update({"model": model, "optimizer": optimizer})

        callbacks = self._build_callbacks()
        for cb in callbacks:
            cb.on_train_begin(state)

        history: List[Dict[str, float]] = []
        for epoch in range(self.config.training.epochs):
            model.train()
            optimizer.zero_grad()
            loss_value = self._forward_loss(model, batch_to_use, loss_fn)
            loss_value.backward()
            optimizer.step()

            metrics = {"train_loss": float(loss_value.item())}
            for cb in callbacks:
                cb.on_epoch_end(state, epoch, metrics)
            history.append({"epoch": int(epoch), **metrics})
            state.history.append(metrics)
            if state.should_stop:
                break

        for cb in callbacks:
            cb.on_train_end(state)

        # Mark the run completed.
        meta_path = self.run_dir / "run_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["status"] = "completed"
            meta["total_epochs"] = len(history)
            meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            meta_path.write_text(json.dumps(meta, indent=2))

        # Dashboard-friendly metrics summary.
        summary = {
            "run_name": self.config.run_name,
            "epochs": len(history),
            "best_metric": state.best_metric,
            "best_epoch": state.best_epoch,
            "final_train_loss": history[-1].get("train_loss") if history else None,
        }
        (self.run_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2))

        return history

    def _forward_loss(self, model, batch_or_graph, loss_fn):
        """Forward + loss for the supported tasks."""
        task = self.config.model.task
        if task in ("graph_classification", "graph_regression"):
            if hasattr(batch_or_graph, "batch") and batch_or_graph.batch is not None:
                logits = model(
                    batch_or_graph.node_features,
                    batch_or_graph.edge_index,
                    batch=batch_or_graph.batch,
                )
                target = batch_or_graph.graph_labels
            else:
                # Single graph
                logits = model(
                    batch_or_graph.node_features.unsqueeze(0).reshape_as(batch_or_graph.node_features),
                    batch_or_graph.edge_index,
                )
                target = batch_or_graph.graph_label
            if task == "graph_classification":
                return loss_fn(logits, target.long().view(-1))
            return loss_fn(logits.view(-1), target.float().view(-1))
        if task in ("node_classification", "node_regression"):
            logits = model(batch_or_graph.node_features, batch_or_graph.edge_index)
            target = batch_or_graph.node_labels
            masks = (batch_or_graph.metadata or {}).get("masks") if hasattr(batch_or_graph, "metadata") else None
            if masks is not None and "train_mask" in masks:
                m = masks["train_mask"]
                logits = logits[m]
                target = target[m]
            if task == "node_classification":
                return loss_fn(logits, target.long())
            return loss_fn(logits.view(-1), target.float().view(-1))
        raise ValueError(f"Runner does not yet support task={task!r}")

    # ── Resume ───────────────────────────────────────────────────────────────

    def resume(self, checkpoint: str | Path = "checkpoints/latest.pt") -> Dict[str, Any]:
        """Load model + optimizer state from a checkpoint inside ``run_dir``."""
        ckpt_path = self.run_dir / checkpoint
        device = _resolve_device(self.config.training.device)
        model = self._build_model().to(device)
        optimizer = _build_optimizer(
            self.config.training.optimizer,
            model.parameters(),
            self.config.training.lr,
            self.config.training.weight_decay,
        )
        return load_checkpoint(ckpt_path, model, optimizer, map_location=device)
