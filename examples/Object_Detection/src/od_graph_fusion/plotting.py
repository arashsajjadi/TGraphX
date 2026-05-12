"""Publication-friendly plots for the experiment."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


def plot_method_comparison(results: Dict[str, Dict[str, Any]], path: Path) -> Path:
    """Bar chart comparing AP@0.50 across methods."""
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = list(results.keys())
    aps = [results[m].get("AP@0.50", results[m].get("AP", 0.0)) for m in methods]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    bars = ax.bar(methods, aps)
    ax.set_ylabel("AP@0.50")
    ax.set_title("Detection fusion comparison (FAST_SMOKE / DEV unless stated)")
    ax.set_ylim(0, max(0.01, max(aps) * 1.2))
    for b, v in zip(bars, aps):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f"{v:.3f}", ha="center", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    return path


def plot_latency_breakdown(latencies: Dict[str, float], path: Path) -> Path:
    """Stacked bar of pipeline latency components (ms)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    labels = list(latencies.keys())
    values = [latencies[k] for k in labels]
    bottom = 0
    for label, v in zip(labels, values):
        ax.bar(["pipeline"], [v], bottom=[bottom], label=label)
        bottom += v
    ax.set_ylabel("ms / image")
    ax.set_title("Pipeline latency breakdown")
    ax.legend(loc="upper left")
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    return path


def plot_training_curves(history: Dict[str, Any], path: Path) -> Path:
    """Train / val loss curves."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    ax.plot(epochs, history.get("train_loss", []), marker="o", label="train")
    ax.plot(epochs, history.get("val_loss", []), marker="s", label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("TGraphX fusion training curves")
    ax.legend()
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    return path


def plot_detection_graph_sketch(meta, path: Path) -> Path:
    """Schematic of one detection graph: nodes colored by type."""
    path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    rng = np.random.default_rng(int(meta.image_id.encode()[0] if isinstance(meta.image_id, str) and meta.image_id else 0))
    N = meta.node_types.shape[0]
    types = meta.node_types.tolist()
    colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red"}
    labels_map = {0: "proposal", 1: "cluster", 2: "consensus", 3: "context"}
    pos = rng.normal(size=(N, 2))
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    for t in set(types):
        m = [i for i in range(N) if types[i] == t]
        ax.scatter(pos[m, 0], pos[m, 1], c=colors.get(t, "gray"),
                   s=80, alpha=0.8, label=labels_map.get(t, str(t)))
    ax.set_title(f"Detection graph for image {meta.image_id} "
                  f"(P={meta.num_proposals}, C={meta.num_clusters})")
    ax.legend()
    ax.set_axis_off()
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    return path
