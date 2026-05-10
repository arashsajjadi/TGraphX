"""EasyResult and EasyConfig dataclasses for TGraphX easy mode."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch.nn as nn


@dataclass
class EasyResult:
    """Structured result returned by all easy-mode training workflows.

    Attributes:
        metrics: Final metrics dict (``loss``, ``accuracy``, etc.).
        history: List of per-epoch metric dicts.
        model: Trained ``nn.Module``.
        graph: Source ``Graph`` object.
        config: Resolved configuration dict with all defaults expanded.
        artifacts: Optional dict of generated artefact paths.
        loader: The last-used data loader (for inspection or resuming).
        optimizer: The last-used ``torch.optim.Optimizer``.
        elapsed: Wall-clock seconds for the run.
    """

    metrics: Dict[str, float] = field(default_factory=dict)
    history: List[Dict[str, float]] = field(default_factory=list)
    model: Optional[nn.Module] = None
    graph: Optional[Any] = None
    config: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    loader: Optional[Any] = None
    optimizer: Optional[Any] = None
    elapsed: float = 0.0

    def summary(self) -> str:
        """Print a human-readable summary and return it as a string."""
        lines = [
            "=" * 55,
            "TGraphX Easy Mode — Training Result",
            "=" * 55,
        ]
        lines.append("Metrics:")
        for k, v in self.metrics.items():
            lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        lines.append(f"Epochs: {len(self.history)}")
        lines.append(f"Elapsed: {self.elapsed:.1f}s")
        lines.append("Config (resolved defaults):")
        for k, v in self.config.items():
            lines.append(f"  {k}: {v}")
        if self.artifacts:
            lines.append("Artifacts:")
            for k, v in self.artifacts.items():
                lines.append(f"  {k}: {v}")
        text = "\n".join(lines)
        print(text)
        return text

    def print_summary(self) -> None:
        """Print a human-readable summary."""
        self.summary()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict (excludes non-serialisable objects)."""
        return {
            "metrics": self.metrics,
            "history": self.history,
            "config": self.config,
            "artifacts": self.artifacts,
            "elapsed": self.elapsed,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        """Return a markdown table of metrics."""
        if not self.metrics:
            return "_No metrics recorded._"
        lines = ["| Metric | Value |", "| --- | --- |"]
        for k, v in self.metrics.items():
            val = f"{v:.4f}" if isinstance(v, float) else str(v)
            lines.append(f"| {k} | {val} |")
        return "\n".join(lines)

    def save_report(self, path: str) -> None:
        """Write a JSON report to ``path``."""
        import pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(path).write_text(self.to_json())
        print(f"Report saved to {path}")

    def plot_loss(self) -> None:
        """Plot training loss history (requires matplotlib).

        Raises:
            ValueError: If no training history is available (train first).
            ImportError: Caught internally; prints an actionable install hint.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print(
                "matplotlib is not installed.  Install it with:\n"
                "    pip install matplotlib\n"
                "to enable plot_loss()."
            )
            return
        if not self.history:
            raise ValueError(
                "No training history is available — call a training workflow "
                "first (e.g. tgx.easy.train_node_classifier(...)) before "
                "calling result.plot_loss()."
            )
        losses = [e.get("loss", float("nan")) for e in self.history]
        if all(v != v for v in losses):  # all NaN means 'loss' key not present
            available = list(self.history[0].keys()) if self.history else []
            raise ValueError(
                f"No 'loss' key found in training history.  "
                f"Available metric keys: {available}.  "
                f"Use result.plot_metrics() to plot all available metrics."
            )
        plt.figure()
        plt.plot(losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.tight_layout()
        plt.show()

    def plot_metrics(self) -> None:
        """Plot all numeric metrics from history (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available.  Install it to use plot_metrics().")
            return
        if not self.history:
            raise ValueError(
                "No training history is available — call a training workflow "
                "first before calling result.plot_metrics()."
            )
        keys = [k for k in self.history[0] if isinstance(self.history[0][k], (int, float))]
        fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
        if len(keys) == 1:
            axes = [axes]
        for ax, k in zip(axes, keys):
            vals = [e.get(k, float("nan")) for e in self.history]
            ax.plot(vals, marker="o")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(k)
            ax.set_title(k)
        plt.tight_layout()
        plt.show()


@dataclass
class EasyConfig:
    """Configuration dataclass for easy-mode workflows.

    Every field has a sensible default.  Pass an ``EasyConfig`` instance to
    any ``train_*`` or ``fit_*`` function instead of individual keyword args.
    """

    task: str = "node_classification"
    model: Optional[str] = None
    sampler: str = "neighbor"
    optimizer: str = "adam"
    lr: float = 1e-3
    epochs: int = 5
    batch_size: int = 64
    device: str = "auto"
    seed: Optional[int] = None
    fanouts: List[int] = field(default_factory=lambda: [15, 10])
    hidden_channels: int = 16
    verbose: bool = True
    dashboard_dir: Optional[str] = None
