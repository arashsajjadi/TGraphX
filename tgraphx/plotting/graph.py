"""Graph structure plots: topology, degree distribution, adjacency.

Stability: Beta (v0.3.2+).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch

__all__ = [
    "plot_graph",
    "plot_degree_distribution",
    "plot_adjacency_matrix",
    "plot_connected_components",
]


def _require_mpl():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "pip install matplotlib  # required for TGraphX plotting"
        ) from exc


def _get_layout(layout: str, edge_index: torch.Tensor, num_nodes: int, seed: Optional[int]) -> np.ndarray:
    from .layouts import circular_layout, random_layout, spring_layout, grid_layout
    if layout == "circular":
        return circular_layout(num_nodes)
    if layout == "random":
        return random_layout(num_nodes, seed=seed)
    if layout == "grid":
        return grid_layout(num_nodes)
    if layout == "spring":
        return spring_layout(edge_index, num_nodes, seed=seed)
    raise ValueError(f"Unknown layout {layout!r}; use 'circular','random','grid','spring'.")


def plot_graph(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_values: Optional[torch.Tensor] = None,
    edge_values: Optional[torch.Tensor] = None,
    layout: str = "spring",
    node_size: float = 80.0,
    node_color: Optional[Union[str, Sequence]] = None,
    edge_color: str = "#888888",
    with_labels: bool = True,
    title: Optional[str] = None,
    ax=None,
    seed: Optional[int] = 42,
    max_nodes: int = 500,
    max_edges: int = 2000,
):
    """Plot a graph using Matplotlib.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        node_values: Optional ``FloatTensor[N]`` for node colouring.
        edge_values: Optional ``FloatTensor[E]`` for edge colouring.
        layout: ``"spring"`` (default), ``"circular"``, ``"random"``,
            ``"grid"``.
        node_size: Scatter marker size.
        node_color: Node colour or list of colours.
        edge_color: Edge colour.
        with_labels: Draw node id labels.
        title: Optional figure title.
        ax: Matplotlib ``Axes``; creates a new figure when ``None``.
        seed: Layout RNG seed.
        max_nodes: Refuse to draw if ``num_nodes > max_nodes``.
        max_edges: Refuse to draw if ``E > max_edges``.

    Returns:
        ``(fig, ax)`` tuple.
    """
    plt = _require_mpl()
    from .utils import OKABE_ITO

    if num_nodes > max_nodes:
        raise ValueError(f"num_nodes={num_nodes} > max_nodes={max_nodes}; "
                         "reduce max_nodes to override.")
    if edge_index.numel() > 0 and edge_index.size(1) > max_edges:
        raise ValueError(f"E={edge_index.size(1)} > max_edges={max_edges}; "
                         "reduce max_edges to override.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    if num_nodes == 0:
        ax.set_aspect("equal")
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=12)
        return fig, ax

    pos = _get_layout(layout, edge_index, num_nodes, seed)  # [N, 2]

    # Draw edges.
    if edge_index.numel():
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
        for u, v in zip(src, dst):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1], color=edge_color, lw=0.8, zorder=1, alpha=0.7)

    # Node colours.
    if node_color is not None:
        nc = node_color
    elif node_values is not None:
        vals = node_values.float().cpu().numpy()
        nc = vals
    else:
        nc = OKABE_ITO[1]  # sky blue default

    sc = ax.scatter(
        pos[:, 0], pos[:, 1],
        c=nc, s=node_size, zorder=2,
        edgecolors="white", linewidths=0.5,
        cmap="viridis" if node_values is not None else None,
    )

    if with_labels and num_nodes <= 30:
        for i in range(num_nodes):
            ax.annotate(str(i), (pos[i, 0], pos[i, 1]),
                        fontsize=7, ha="center", va="center", color="white",
                        fontweight="bold", zorder=3)

    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)
    return fig, ax


def plot_degree_distribution(
    edge_index: torch.Tensor,
    num_nodes: int,
    directed: bool = True,
    bins: int = 20,
    title: Optional[str] = None,
    ax=None,
):
    """Bar plot of node degree distribution.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        directed: When ``True``, shows out-degree distribution.
        bins: Number of histogram bins.
        title: Optional plot title.
        ax: Matplotlib ``Axes``; creates new figure when ``None``.

    Returns:
        ``(fig, ax)`` tuple.
    """
    plt = _require_mpl()
    from .utils import OKABE_ITO

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    else:
        fig = ax.get_figure()

    deg = torch.zeros(num_nodes, dtype=torch.long)
    if edge_index.numel() and num_nodes > 0:
        src = edge_index[0].to(torch.long)
        ones = torch.ones(src.size(0), dtype=torch.long)
        deg.scatter_add_(0, src, ones)

    ax.hist(deg.numpy(), bins=bins, color=OKABE_ITO[1], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Degree", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(title or ("Out-Degree Distribution" if directed else "Degree Distribution"), fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def plot_adjacency_matrix(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_nodes: int = 100,
    title: Optional[str] = None,
    ax=None,
):
    """Heatmap of the binary adjacency matrix.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        max_nodes: Refuse to plot if ``num_nodes > max_nodes``.
        title: Optional title.
        ax: Matplotlib ``Axes``; creates new figure when ``None``.

    Returns:
        ``(fig, ax)`` tuple.
    """
    plt = _require_mpl()

    if num_nodes > max_nodes:
        raise ValueError(f"num_nodes={num_nodes} > max_nodes={max_nodes}.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()

    A = torch.zeros(num_nodes, num_nodes, dtype=torch.float)
    if edge_index.numel():
        A[edge_index[0].long(), edge_index[1].long()] = 1.0

    ax.imshow(A.numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_title(title or "Adjacency Matrix", fontsize=11)
    ax.set_xlabel("Destination", fontsize=9)
    ax.set_ylabel("Source", fontsize=9)
    return fig, ax


def plot_connected_components(
    edge_index: torch.Tensor,
    num_nodes: int,
    layout: str = "spring",
    title: Optional[str] = None,
    ax=None,
    seed: Optional[int] = 42,
    max_nodes: int = 200,
):
    """Plot graph with nodes coloured by connected component.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        layout: Layout algorithm.
        title: Optional title.
        ax: Optional Matplotlib ``Axes``.
        seed: Layout seed.
        max_nodes: Size guard.

    Returns:
        ``(fig, ax)`` tuple.
    """
    from tgraphx.algorithms import connected_components
    labels = connected_components(edge_index, num_nodes)
    return plot_graph(
        edge_index, num_nodes,
        node_values=labels.float(),
        layout=layout, title=title or "Connected Components",
        ax=ax, seed=seed, max_nodes=max_nodes,
    )
