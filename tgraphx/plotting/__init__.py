"""TGraphX plotting and visualization utilities.

This package provides Matplotlib-based graph visualization and mining
result plots.  All plots are:

- Matplotlib-only (no mandatory seaborn, no mandatory networkx).
- Headless-safe (work with the ``Agg`` backend).
- Colorblind-friendly by default (Okabe-Ito palette).
- Saveable to PNG / SVG / PDF.
- Compatible with Colab, scripts, and Jupyter.

Imports are lazy so that ``import tgraphx`` does not pull in Matplotlib
unless the user actually calls a plotting function.

Stability: **Beta** (v0.3.2+).

Quick start::

    from tgraphx.plotting import (
        plot_graph,
        plot_degree_distribution,
        plot_motif_summary,
        plot_anomaly_scores,
        plot_confusion_matrix,
        plot_training_curves,
    )
"""
from __future__ import annotations

from .graph import (
    plot_graph,
    plot_degree_distribution,
    plot_adjacency_matrix,
    plot_connected_components,
)
from .mining import (
    plot_motif_summary,
    plot_graph_mining_summary,
    plot_link_prediction_score_distribution,
    plot_graph_similarity_heatmap,
    plot_anomaly_scores,
    plot_prototype_membership_scores,
    plot_confusion_matrix,
    plot_training_curves,
    plot_community_assignments,
)
from .layouts import (
    circular_layout,
    grid_layout,
    random_layout,
    spring_layout,
)
from .utils import save_figure

__all__ = [
    # Graph plots
    "plot_graph",
    "plot_degree_distribution",
    "plot_adjacency_matrix",
    "plot_connected_components",
    # Mining plots
    "plot_motif_summary",
    "plot_graph_mining_summary",
    "plot_link_prediction_score_distribution",
    "plot_graph_similarity_heatmap",
    "plot_anomaly_scores",
    "plot_prototype_membership_scores",
    "plot_confusion_matrix",
    "plot_training_curves",
    "plot_community_assignments",
    # Layouts
    "circular_layout",
    "grid_layout",
    "random_layout",
    "spring_layout",
    # Utilities
    "save_figure",
]
