"""Graph algorithms used by GNN workflows.

This package provides a small, focused set of graph algorithms in pure
PyTorch.  It is **not** a NetworkX replacement — only algorithms that
appear naturally in GNN training (component analysis, BFS traversal,
shortest paths, structural features) are included.

Public surface
--------------

Connectivity (:mod:`tgraphx.algorithms.connectivity`):
    - :func:`connected_components` — for undirected graphs (or treat a
      directed graph as undirected; see ``weakly_connected_components``).
    - :func:`weakly_connected_components` — directed → undirected.
    - :func:`is_connected` — single-component check.
    - :func:`number_connected_components` — count.

Traversal (:mod:`tgraphx.algorithms.traversal`):
    - :func:`bfs_edges` — BFS edges in visit order from a source.
    - :func:`bfs_layers` — BFS frontiers grouped by hop distance.
    - :func:`shortest_path_length` — unweighted distance from a source.

Stability
---------
Beta.  Signatures may evolve before v0.4.0, but the mathematical
contracts (component disjointness, BFS visit order, distance
correctness) are stable.
"""
from __future__ import annotations

from .connectivity import (
    connected_components,
    is_connected,
    number_connected_components,
    weakly_connected_components,
)
from .traversal import (
    bfs_edges,
    bfs_layers,
    shortest_path_length,
)
from .structural import (
    degree,
    degree_features,
)

__all__ = [
    "connected_components",
    "weakly_connected_components",
    "is_connected",
    "number_connected_components",
    "bfs_edges",
    "bfs_layers",
    "shortest_path_length",
    "degree",
    "degree_features",
]
