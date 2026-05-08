"""Mining result plots: motifs, similarity, anomaly, membership, training.

Stability: Beta (v0.3.2+).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

__all__ = [
    "plot_motif_summary",
    "plot_graph_mining_summary",
    "plot_link_prediction_score_distribution",
    "plot_graph_similarity_heatmap",
    "plot_anomaly_scores",
    "plot_prototype_membership_scores",
    "plot_confusion_matrix",
    "plot_training_curves",
    "plot_community_assignments",
]


def _mpl():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError("pip install matplotlib  # required for TGraphX plotting") from exc


def _okabe():
    from .utils import OKABE_ITO
    return OKABE_ITO


# ── Motif summary ─────────────────────────────────────────────────────────────


def plot_motif_summary(
    motif_counts: Dict[str, Any],
    title: Optional[str] = None,
    ax=None,
):
    """Bar chart of motif/structural summary counts.

    Args:
        motif_counts: Dict from :func:`~tgraphx.mining.motif_counts`.
        title: Optional plot title.
        ax: Optional Matplotlib ``Axes``.

    Returns:
        ``(fig, ax)`` tuple.
    """
    plt = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig = ax.get_figure()

    keys = ["edges", "triangles", "wedges"]
    vals = [motif_counts.get(k, 0) for k in keys]
    colors = _okabe()[:len(keys)]
    ax.bar(keys, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(title or "Motif Counts", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.01, str(int(v)), ha="center", va="bottom", fontsize=9)
    return fig, ax


# ── Graph mining summary ──────────────────────────────────────────────────────


def plot_graph_mining_summary(
    summary: Dict[str, Any],
    title: Optional[str] = None,
    ax=None,
):
    """Horizontal bar chart of key graph summary statistics.

    Args:
        summary: Dict from :func:`~tgraphx.mining.graph_summary`.
        title: Optional title.
        ax: Optional Axes.

    Returns:
        ``(fig, ax)`` tuple.
    """
    plt = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    keys = ["num_nodes", "num_edges", "density",
            "mean_total_degree", "num_connected_components"]
    labels_display = ["Nodes", "Edges", "Density",
                      "Mean Degree", "Components"]
    vals = [summary.get(k, 0) or 0 for k in keys]
    colors = _okabe()

    ax.barh(labels_display, vals,
            color=[colors[i % len(colors)] for i in range(len(labels_display))],
            edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Value", fontsize=10)
    ax.set_title(title or "Graph Summary", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, f"{v:.3g}", va="center", fontsize=9)
    return fig, ax


# ── Link prediction ───────────────────────────────────────────────────────────


def plot_link_prediction_score_distribution(
    scores: Dict[str, Any],
    title: Optional[str] = None,
    ax=None,
    bins: int = 20,
):
    """Overlapping histogram of link prediction scores per scorer.

    Args:
        scores: Dict ``{scorer_name: FloatTensor[P] or list}``.
        title: Optional title.
        ax: Optional Axes.
        bins: Histogram bins.

    Returns:
        ``(fig, ax)`` tuple.
    """
    plt = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig = ax.get_figure()

    colors = _okabe()
    for i, (name, s) in enumerate(scores.items()):
        if hasattr(s, "tolist"):
            s = s.tolist()
        ax.hist(s, bins=bins, alpha=0.5, label=name,
                color=colors[i % len(colors)], edgecolor="none")
    ax.set_xlabel("Score", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(title or "Link Prediction Score Distributions", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


# ── Similarity heatmap ────────────────────────────────────────────────────────


def plot_graph_similarity_heatmap(
    matrix: Union[torch.Tensor, np.ndarray],
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    ax=None,
    max_size: int = 50,
):
    """Heatmap of a graph similarity matrix.

    Args:
        matrix: ``[G, G]`` similarity matrix.
        labels: Optional graph labels (displayed on axes).
        title: Optional title.
        ax: Optional Axes.
        max_size: Refuse if ``G > max_size`` to avoid unreadable plots.

    Returns:
        ``(fig, ax)`` tuple.
    """
    plt = _mpl()
    if hasattr(matrix, "cpu"):
        matrix = matrix.detach().cpu().numpy()

    G = matrix.shape[0]
    if G > max_size:
        raise ValueError(f"Matrix size {G} > max_size={max_size}.")

    if ax is None:
        size = max(4, min(10, G * 0.5))
        fig, ax = plt.subplots(figsize=(size, size * 0.9))
    else:
        fig = ax.get_figure()

    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if labels:
        ax.set_xticks(range(G))
        ax.set_yticks(range(G))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title or "Graph Similarity Matrix", fontsize=11)
    return fig, ax


# ── Anomaly ───────────────────────────────────────────────────────────────────


def plot_anomaly_scores(
    scores: Union[torch.Tensor, List[float]],
    top_k: int = 20,
    title: Optional[str] = None,
    ax=None,
):
    """Horizontal bar chart of top-k anomalous node scores.

    Args:
        scores: ``FloatTensor[N]`` or list of node anomaly scores.
        top_k: Number of top anomalous nodes to display.
        title: Optional title.
        ax: Optional Axes.

    Returns:
        ``(fig, ax)`` tuple.
    """
    plt = _mpl()
    if hasattr(scores, "tolist"):
        scores_list = scores.tolist()
    else:
        scores_list = list(scores)

    pairs = sorted(enumerate(scores_list), key=lambda x: -x[1])[:top_k]
    node_ids = [str(p[0]) for p in pairs]
    vals = [p[1] for p in pairs]

    if ax is None:
        height = max(3, min(10, len(pairs) * 0.35))
        fig, ax = plt.subplots(figsize=(6, height))
    else:
        fig = ax.get_figure()

    colors = _okabe()
    ax.barh(node_ids[::-1], vals[::-1], color=colors[5], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Anomaly Score", fontsize=10)
    ax.set_title(title or f"Top-{len(pairs)} Anomalous Nodes", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


# ── Prototype membership ──────────────────────────────────────────────────────


def plot_prototype_membership_scores(
    scores: Dict[int, float],
    true_label: Optional[int] = None,
    title: Optional[str] = None,
    ax=None,
):
    """Bar chart of per-class membership scores for a query.

    Args:
        scores: Dict ``{class_id: score}`` from the membership scorer.
        true_label: Optional true class (highlighted in green).
        title: Optional title.
        ax: Optional Axes.

    Returns:
        ``(fig, ax)`` tuple.
    """
    plt = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    else:
        fig = ax.get_figure()

    cls_ids = sorted(scores.keys())
    vals = [scores[c] for c in cls_ids]
    colors_base = _okabe()
    bar_colors = [
        colors_base[2] if true_label is not None and c == true_label else colors_base[1]
        for c in cls_ids
    ]
    ax.bar([str(c) for c in cls_ids], vals, color=bar_colors, edgecolor="white")
    ax.set_xlabel("Class", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(title or "Prototype Membership Scores", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if true_label is not None:
        ax.axhline(0, color="#888", linewidth=0.5)
    return fig, ax


# ── Confusion matrix ──────────────────────────────────────────────────────────


def plot_confusion_matrix(
    matrix: Union[List[List[int]], np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: Optional[str] = None,
    ax=None,
):
    """Annotated confusion matrix heatmap.

    Args:
        matrix: ``[C, C]`` confusion matrix.
        class_names: Optional class label strings.
        normalize: When ``True``, normalise rows to ``[0, 1]``.
        title: Optional title.
        ax: Optional Axes.

    Returns:
        ``(fig, ax)`` tuple.
    """
    plt = _mpl()
    if isinstance(matrix, list):
        M = np.array(matrix, dtype=np.float64)
    elif hasattr(matrix, "cpu"):
        M = matrix.detach().cpu().float().numpy()
    else:
        M = np.array(matrix, dtype=np.float64)

    C = M.shape[0]
    if normalize:
        row_sums = M.sum(axis=1, keepdims=True).clip(min=1)
        M = M / row_sums

    if ax is None:
        size = max(4, min(10, C * 0.8))
        fig, ax = plt.subplots(figsize=(size, size * 0.85))
    else:
        fig = ax.get_figure()

    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if class_names:
        ax.set_xticks(range(C))
        ax.set_yticks(range(C))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(title or "Confusion Matrix", fontsize=11)
    for i in range(C):
        for j in range(C):
            val = M[i, j]
            ax.text(j, i, f"{val:.2f}" if normalize else str(int(val)),
                    ha="center", va="center",
                    color="white" if val > 0.6 else "black", fontsize=9)
    return fig, ax


# ── Training curves ───────────────────────────────────────────────────────────


def plot_training_curves(
    history: Union[List[Dict[str, float]], Dict[str, List[float]]],
    metrics: Optional[List[str]] = None,
    title: Optional[str] = None,
    ax=None,
):
    """Line plot of training curves from a history object.

    Args:
        history: Either a list of per-epoch dicts (e.g. from
            :func:`~tgraphx.training.fit`) or a dict of
            ``{metric_name: [values]}``.
        metrics: Which keys to plot.  When ``None``, plots all
            numeric keys.
        title: Optional title.
        ax: Optional Axes.

    Returns:
        ``(fig, ax)`` tuple.
    """
    plt = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    # Normalise to dict-of-lists.
    if isinstance(history, list):
        if not history:
            ax.set_title(title or "Training Curves (no data)")
            return fig, ax
        keys = [k for k, v in history[0].items() if isinstance(v, (int, float))]
        data = {k: [h.get(k, float("nan")) for h in history] for k in keys}
    else:
        data = history

    if metrics is not None:
        data = {k: v for k, v in data.items() if k in metrics}

    colors = _okabe()
    for i, (name, vals) in enumerate(data.items()):
        ax.plot(vals, label=name, color=colors[i % len(colors)], linewidth=1.5)

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)
    ax.set_title(title or "Training Curves", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


# ── Community assignments ─────────────────────────────────────────────────────


def plot_community_assignments(
    edge_index: torch.Tensor,
    num_nodes: int,
    communities: torch.Tensor,
    layout: str = "spring",
    title: Optional[str] = None,
    ax=None,
    seed: Optional[int] = 42,
    max_nodes: int = 200,
):
    """Plot graph with nodes coloured by community assignment.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        communities: ``LongTensor[N]`` of community labels.
        layout: Layout algorithm.
        title: Optional title.
        ax: Optional Axes.
        seed: Layout seed.
        max_nodes: Size guard.

    Returns:
        ``(fig, ax)`` tuple.
    """
    from .graph import plot_graph
    return plot_graph(
        edge_index, num_nodes,
        node_values=communities.float(),
        layout=layout,
        title=title or "Community Assignments",
        ax=ax, seed=seed, max_nodes=max_nodes,
    )
