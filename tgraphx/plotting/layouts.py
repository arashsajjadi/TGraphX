"""Pure-Python/NumPy graph layout algorithms (no NetworkX required).

Stability: Beta (v0.3.2+).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

__all__ = ["circular_layout", "grid_layout", "random_layout", "spring_layout"]


def circular_layout(num_nodes: int) -> np.ndarray:
    """Place nodes equally spaced on a circle.

    Args:
        num_nodes: Number of nodes.

    Returns:
        ``ndarray[N, 2]`` of (x, y) coordinates.
    """
    if num_nodes == 0:
        return np.zeros((0, 2), dtype=np.float64)
    angles = np.linspace(0, 2 * math.pi, num_nodes, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def grid_layout(num_nodes: int, width: Optional[int] = None) -> np.ndarray:
    """Place nodes on a square-ish grid.

    Args:
        num_nodes: Number of nodes.
        width: Number of columns.  When ``None``, uses ``ceil(sqrt(N))``.

    Returns:
        ``ndarray[N, 2]``.
    """
    if num_nodes == 0:
        return np.zeros((0, 2), dtype=np.float64)
    W = width if width is not None else math.ceil(math.sqrt(num_nodes))
    positions = []
    for i in range(num_nodes):
        row, col = divmod(i, W)
        positions.append([float(col), float(-row)])
    return np.array(positions, dtype=np.float64)


def random_layout(num_nodes: int, seed: Optional[int] = None) -> np.ndarray:
    """Place nodes uniformly at random in ``[0, 1]²``.

    Args:
        num_nodes: Number of nodes.
        seed: Optional RNG seed.

    Returns:
        ``ndarray[N, 2]``.
    """
    rng = np.random.default_rng(seed)
    return rng.random((num_nodes, 2))


def spring_layout(
    edge_index: torch.Tensor,
    num_nodes: int,
    iterations: int = 50,
    k: Optional[float] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Fruchterman-Reingold spring-embedder layout (pure NumPy/Python).

    A classic force-directed layout without NetworkX.  Suitable for
    graphs with up to ~200 nodes.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        iterations: Number of force-directed iterations.
        k: Optimal distance parameter.  When ``None``, set to
            ``1 / sqrt(N)`` (standard FR heuristic).
        seed: Optional RNG seed for initial placement.

    Returns:
        ``ndarray[N, 2]`` of layout coordinates in ``[-1, 1]²``.
    """
    if num_nodes == 0:
        return np.zeros((0, 2), dtype=np.float64)

    rng = np.random.default_rng(seed)
    pos = rng.random((num_nodes, 2)) - 0.5  # in [-0.5, 0.5]

    if k is None:
        k_val = 1.0 / math.sqrt(max(num_nodes, 1))
    else:
        k_val = float(k)

    edges = edge_index.cpu().tolist() if edge_index.numel() else [[], []]
    src_list = edges[0] if edges else []
    dst_list = edges[1] if edges else []

    temp = 0.1  # initial temperature
    cooling = temp / max(iterations, 1)

    for _ in range(iterations):
        disp = np.zeros_like(pos)

        # Repulsive forces: all pairs.
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                delta = pos[i] - pos[j]
                dist = math.sqrt(delta[0] ** 2 + delta[1] ** 2) + 1e-9
                rep = k_val ** 2 / dist
                disp[i] += (delta / dist) * rep

        # Attractive forces: edges.
        for u, v in zip(src_list, dst_list):
            if u == v:
                continue
            delta = pos[u] - pos[v]
            dist = math.sqrt(delta[0] ** 2 + delta[1] ** 2) + 1e-9
            attr = dist ** 2 / k_val
            disp[u] -= (delta / dist) * attr
            disp[v] += (delta / dist) * attr

        # Apply displacement with temperature limit.
        for i in range(num_nodes):
            d = math.sqrt(disp[i, 0] ** 2 + disp[i, 1] ** 2) + 1e-9
            pos[i] += disp[i] / d * min(d, temp)

        temp -= cooling

    # Normalise to [-1, 1].
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    rng2 = hi - lo + 1e-9
    pos = 2.0 * (pos - lo) / rng2 - 1.0
    return pos
