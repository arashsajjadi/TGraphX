"""Experiment configuration loading and schema validation.

Configs are plain JSON / YAML files (no ``eval``, no ``exec``).  The
expected top-level structure is::

    seed: 0
    run_name: my_run
    run_dir: runs/my_run     # optional; defaults to runs/<run_name>/<timestamp>
    dataset:
        name: synthetic:patch_graph
        kwargs: { num_graphs: 32, seed: 0 }
    transforms: []           # optional list of {name, kwargs}
    model:
        task: graph_classification
        layer: conv
        in_shape: [1, 8, 8]
        hidden_shape: [8, 8, 8]
        num_layers: 2
        num_classes: 6
    training:
        epochs: 5
        batch_size: 8
        lr: 0.005
        optimizer: adam
        device: cpu
        loss: cross_entropy
    callbacks:
        - {name: csv_logger}
        - {name: early_stopping, kwargs: {patience: 3}}
        - {name: model_checkpoint, kwargs: {save_best: true}}

This module returns a validated :class:`ExperimentConfig` dataclass —
it does **not** instantiate datasets/models.  Construction happens
inside :class:`tgraphx.experiments.runner.Runner`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class DatasetConfig:
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    task: str
    layer: str = "linear"
    in_shape: Optional[List[int]] = None
    hidden_shape: Optional[List[int]] = None
    num_layers: int = 2
    num_classes: Optional[int] = None
    out_dim: Optional[int] = None
    pooling: str = "mean"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 8
    lr: float = 1e-3
    optimizer: str = "adam"
    device: str = "cpu"
    loss: str = "cross_entropy"
    val_ratio: float = 0.2
    test_ratio: float = 0.0
    weight_decay: float = 0.0


@dataclass
class CallbackConfig:
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformConfig:
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Validated experiment specification."""

    seed: int = 0
    run_name: str = "run"
    run_dir: Optional[str] = None
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(name="synthetic:patch_graph"),
    )
    transforms: List[TransformConfig] = field(default_factory=list)
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(task="graph_classification"),
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    callbacks: List[CallbackConfig] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "run_name": self.run_name,
            "run_dir": self.run_dir,
            "dataset": {"name": self.dataset.name, "kwargs": dict(self.dataset.kwargs)},
            "transforms": [
                {"name": t.name, "kwargs": dict(t.kwargs)} for t in self.transforms
            ],
            "model": {
                "task": self.model.task,
                "layer": self.model.layer,
                "in_shape": self.model.in_shape,
                "hidden_shape": self.model.hidden_shape,
                "num_layers": self.model.num_layers,
                "num_classes": self.model.num_classes,
                "out_dim": self.model.out_dim,
                "pooling": self.model.pooling,
                **self.model.extra,
            },
            "training": {
                "epochs": self.training.epochs,
                "batch_size": self.training.batch_size,
                "lr": self.training.lr,
                "optimizer": self.training.optimizer,
                "device": self.training.device,
                "loss": self.training.loss,
                "val_ratio": self.training.val_ratio,
                "test_ratio": self.training.test_ratio,
                "weight_decay": self.training.weight_decay,
            },
            "callbacks": [
                {"name": c.name, "kwargs": dict(c.kwargs)} for c in self.callbacks
            ],
        }


# ── Loading ──────────────────────────────────────────────────────────────────


def load_config(path_or_dict: Union[str, Path, Dict[str, Any]]) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML/JSON file or a dict.

    YAML uses ``yaml.safe_load`` (no ``!!python/object`` shenanigans).
    Unknown top-level keys raise :class:`ValueError`.
    """
    if isinstance(path_or_dict, dict):
        raw = path_or_dict
    else:
        path = Path(path_or_dict).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        suffix = path.suffix.lower()
        text = path.read_text(encoding="utf-8")
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover  (PyYAML is base dep)
                raise ImportError(
                    "YAML configs require PyYAML.  Install with `pip install pyyaml`."
                ) from exc
            raw = yaml.safe_load(text) or {}
        elif suffix == ".json":
            raw = json.loads(text)
        else:
            raise ValueError(
                f"Unsupported config format: {suffix!r}. Use .yaml/.yml or .json."
            )
    return _validate(raw)


def _validate(raw: Dict[str, Any]) -> ExperimentConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping; got {type(raw)}")
    allowed_top = {"seed", "run_name", "run_dir", "dataset", "transforms",
                   "model", "training", "callbacks"}
    unknown = set(raw) - allowed_top
    if unknown:
        raise ValueError(
            f"Unknown top-level config keys: {sorted(unknown)}. "
            f"Allowed: {sorted(allowed_top)}."
        )

    seed = int(raw.get("seed", 0))
    run_name = str(raw.get("run_name", "run"))
    run_dir = raw.get("run_dir")
    if run_dir is not None and not isinstance(run_dir, str):
        raise ValueError(f"run_dir must be a string; got {type(run_dir)}")

    ds_raw = raw.get("dataset")
    if ds_raw is None or not isinstance(ds_raw, dict):
        raise ValueError("config.dataset is required and must be a mapping")
    if "name" not in ds_raw:
        raise ValueError("config.dataset.name is required")
    dataset = DatasetConfig(
        name=str(ds_raw["name"]),
        kwargs=dict(ds_raw.get("kwargs") or {}),
    )

    transforms = []
    for i, t in enumerate(raw.get("transforms") or []):
        if not isinstance(t, dict) or "name" not in t:
            raise ValueError(
                f"config.transforms[{i}] must be a mapping with a 'name' key."
            )
        transforms.append(TransformConfig(
            name=str(t["name"]),
            kwargs=dict(t.get("kwargs") or {}),
        ))

    md = raw.get("model")
    if md is None or not isinstance(md, dict) or "task" not in md:
        raise ValueError("config.model is required and must include 'task'")
    model = ModelConfig(
        task=str(md["task"]),
        layer=str(md.get("layer", "linear")),
        in_shape=list(md["in_shape"]) if md.get("in_shape") is not None else None,
        hidden_shape=list(md["hidden_shape"]) if md.get("hidden_shape") is not None else None,
        num_layers=int(md.get("num_layers", 2)),
        num_classes=md.get("num_classes"),
        out_dim=md.get("out_dim"),
        pooling=str(md.get("pooling", "mean")),
        extra={k: v for k, v in md.items() if k not in
               {"task", "layer", "in_shape", "hidden_shape", "num_layers",
                "num_classes", "out_dim", "pooling"}},
    )

    tr = raw.get("training") or {}
    if not isinstance(tr, dict):
        raise ValueError("config.training must be a mapping")
    training = TrainingConfig(
        epochs=int(tr.get("epochs", 5)),
        batch_size=int(tr.get("batch_size", 8)),
        lr=float(tr.get("lr", 1e-3)),
        optimizer=str(tr.get("optimizer", "adam")),
        device=str(tr.get("device", "cpu")),
        loss=str(tr.get("loss", "cross_entropy")),
        val_ratio=float(tr.get("val_ratio", 0.2)),
        test_ratio=float(tr.get("test_ratio", 0.0)),
        weight_decay=float(tr.get("weight_decay", 0.0)),
    )

    callbacks = []
    for i, c in enumerate(raw.get("callbacks") or []):
        if not isinstance(c, dict) or "name" not in c:
            raise ValueError(
                f"config.callbacks[{i}] must be a mapping with a 'name' key."
            )
        callbacks.append(CallbackConfig(
            name=str(c["name"]),
            kwargs=dict(c.get("kwargs") or {}),
        ))

    return ExperimentConfig(
        seed=seed,
        run_name=run_name,
        run_dir=run_dir,
        dataset=dataset,
        transforms=transforms,
        model=model,
        training=training,
        callbacks=callbacks,
    )
