"""Temporal graph mining utilities.

Utilities for working with time-stamped edge events and temporal
graph data structures.

Stability: Experimental (v0.3.2+).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

__all__ = [
    "temporal_degree",
    "sliding_window_edges",
    "temporal_chronological_split",
    "burst_score",
]


def temporal_degree(
    src: torch.Tensor,
    dst: torch.Tensor,
    timestamps: torch.Tensor,
    num_nodes: int,
    window_start: float,
    window_end: float,
) -> torch.Tensor:
    """Compute total (in + out) degree within a time window.

    Args:
        src: ``LongTensor[E]`` — source nodes.
        dst: ``LongTensor[E]`` — destination nodes.
        timestamps: ``FloatTensor[E]`` — event timestamps.
        num_nodes: Node count.
        window_start: Inclusive start of the window.
        window_end: Exclusive end of the window.

    Returns:
        ``LongTensor[num_nodes]`` — total degree in the window.
    """
    if src.shape != dst.shape or src.shape != timestamps.shape:
        raise ValueError("src, dst, timestamps must have the same shape")
    mask = (timestamps >= window_start) & (timestamps < window_end)
    src_w = src[mask].to(torch.long)
    dst_w = dst[mask].to(torch.long)
    deg = torch.zeros(num_nodes, dtype=torch.long)
    ones = torch.ones(src_w.size(0), dtype=torch.long)
    if src_w.numel():
        deg.scatter_add_(0, src_w, ones)
        deg.scatter_add_(0, dst_w, ones)
    return deg


def sliding_window_edges(
    src: torch.Tensor,
    dst: torch.Tensor,
    timestamps: torch.Tensor,
    window_size: float,
    step: float,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Slice edge events into sliding time windows.

    Args:
        src: ``LongTensor[E]`` — source nodes.
        dst: ``LongTensor[E]`` — destination nodes.
        timestamps: ``FloatTensor[E]`` — event timestamps.
        window_size: Width of each time window.
        step: Stride between windows.

    Returns:
        List of ``(src_window, dst_window, timestamps_window)`` tuples,
        one per window.  Empty windows are omitted.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive; got {window_size}")
    if step <= 0:
        raise ValueError(f"step must be positive; got {step}")
    if timestamps.numel() == 0:
        return []

    t_min = float(timestamps.min().item())
    t_max = float(timestamps.max().item())
    windows = []
    t_start = t_min
    while t_start <= t_max:
        t_end = t_start + window_size
        mask = (timestamps >= t_start) & (timestamps < t_end)
        if mask.any():
            windows.append((src[mask], dst[mask], timestamps[mask]))
        t_start += step
    return windows


def temporal_chronological_split(
    timestamps: torch.Tensor,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split edge indices chronologically without future leakage.

    Args:
        timestamps: ``FloatTensor[E]`` — edge timestamps.
        ratios: ``(train, val, test)`` fractions; must sum to 1.

    Returns:
        ``(train_mask, val_mask, test_mask)`` boolean tensors of shape ``[E]``.
    """
    if len(ratios) != 3:
        raise ValueError("ratios must have exactly 3 elements (train, val, test)")
    r_sum = sum(ratios)
    if abs(r_sum - 1.0) > 1e-5:
        raise ValueError(f"ratios must sum to 1; got {r_sum}")

    E = timestamps.size(0)
    if E == 0:
        empty = torch.zeros(0, dtype=torch.bool)
        return empty, empty, empty

    order = torch.argsort(timestamps, stable=True)
    n_train = int(round(ratios[0] * E))
    n_val = int(round(ratios[1] * E))
    n_test = E - n_train - n_val

    train_idx = order[:n_train]
    val_idx = order[n_train: n_train + n_val]
    test_idx = order[n_train + n_val:]

    train_mask = torch.zeros(E, dtype=torch.bool)
    val_mask = torch.zeros(E, dtype=torch.bool)
    test_mask = torch.zeros(E, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def burst_score(
    src: torch.Tensor,
    dst: torch.Tensor,
    timestamps: torch.Tensor,
    num_nodes: int,
    num_windows: int = 10,
) -> torch.Tensor:
    """Simple burst score per node: z-score of activity across time windows.

    Nodes with unusually high activity in some time windows compared to
    the overall mean receive a high burst score.

    Args:
        src: ``LongTensor[E]``.
        dst: ``LongTensor[E]``.
        timestamps: ``FloatTensor[E]``.
        num_nodes: Node count.
        num_windows: Number of equal-width windows.

    Returns:
        ``FloatTensor[num_nodes]`` — non-negative burst scores.
    """
    if timestamps.numel() == 0:
        return torch.zeros(num_nodes, dtype=torch.float)

    t_min = float(timestamps.min().item())
    t_max = float(timestamps.max().item())
    if t_max == t_min:
        # All events at the same time; no burst.
        return torch.zeros(num_nodes, dtype=torch.float)

    window_size = (t_max - t_min) / float(num_windows)
    deg_by_window = torch.zeros(num_nodes, num_windows, dtype=torch.float)

    for w in range(num_windows):
        t_start = t_min + w * window_size
        t_end = t_start + window_size if w < num_windows - 1 else t_max + 1e-9
        deg = temporal_degree(src, dst, timestamps, num_nodes, t_start, t_end)
        deg_by_window[:, w] = deg.float()

    mean_activity = deg_by_window.mean(dim=1)
    std_activity = deg_by_window.std(dim=1).clamp(min=1e-8)
    max_dev = (deg_by_window - mean_activity.unsqueeze(1)).abs().max(dim=1).values
    return (max_dev / std_activity).to(torch.float)
