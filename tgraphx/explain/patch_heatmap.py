"""Aggregate per-patch saliency back into an image / volume heatmap."""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def patch_saliency_to_image_grid(
    saliency: torch.Tensor,
    grid_shape: Tuple[int, int],
    reduce: str = "mean",
) -> torch.Tensor:
    """Collapse ``[P, C, ph, pw]`` patch saliency back to ``[H, W]``.

    Args:
        saliency: Per-patch saliency, shape ``[P, C, ph, pw]``.
        grid_shape: Patch grid ``(n_h, n_w)``; ``P`` must equal
            ``n_h * n_w``.
        reduce: How to reduce the channel dimension before tiling:
            ``"mean"`` (default), ``"sum"``, or ``"max"``.

    Returns:
        ``[n_h * ph, n_w * pw]`` float tensor — the heatmap.
    """
    if saliency.dim() != 4:
        raise ValueError(
            f"patch_saliency_to_image_grid expects [P, C, ph, pw]; "
            f"got {tuple(saliency.shape)}"
        )
    n_h, n_w = grid_shape
    P, C, ph, pw = saliency.shape
    if P != n_h * n_w:
        raise ValueError(
            f"grid_shape {grid_shape} → {n_h * n_w} patches, but saliency "
            f"has {P}."
        )
    if reduce == "mean":
        agg = saliency.mean(dim=1)
    elif reduce == "sum":
        agg = saliency.sum(dim=1)
    elif reduce == "max":
        agg = saliency.max(dim=1).values
    else:
        raise ValueError(f"reduce must be 'mean'/'sum'/'max'; got {reduce!r}")
    # agg: [P, ph, pw]; tile back to [n_h*ph, n_w*pw].
    agg = agg.view(n_h, n_w, ph, pw).permute(0, 2, 1, 3).contiguous()
    return agg.view(n_h * ph, n_w * pw)


def patch_saliency_to_volume_projection(
    saliency: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    axis: int = 0,
    reduce: str = "max",
) -> torch.Tensor:
    """Project a 3-D patch saliency back to a 2-D heatmap by max/mean over an axis.

    Args:
        saliency: ``[P, C, pd, ph, pw]``.
        grid_shape: ``(n_d, n_h, n_w)``.
        axis: Axis to project over (0 = depth, 1 = height, 2 = width).
        reduce: ``"max"`` (default) or ``"mean"``.
    """
    if saliency.dim() != 5:
        raise ValueError(
            f"patch_saliency_to_volume_projection expects [P, C, pd, ph, pw]; "
            f"got {tuple(saliency.shape)}"
        )
    n_d, n_h, n_w = grid_shape
    P, C, pd, ph, pw = saliency.shape
    if P != n_d * n_h * n_w:
        raise ValueError(f"grid_shape {grid_shape} != patches {P}")
    # Aggregate channels.
    agg = saliency.mean(dim=1)  # [P, pd, ph, pw]
    agg = agg.view(n_d, n_h, n_w, pd, ph, pw)
    # Reorder to [n_d, pd, n_h, ph, n_w, pw] then merge:
    full = agg.permute(0, 3, 1, 4, 2, 5).contiguous()
    full = full.view(n_d * pd, n_h * ph, n_w * pw)
    if axis == 0:
        return full.max(dim=0).values if reduce == "max" else full.mean(dim=0)
    if axis == 1:
        return full.max(dim=1).values if reduce == "max" else full.mean(dim=1)
    if axis == 2:
        return full.max(dim=2).values if reduce == "max" else full.mean(dim=2)
    raise ValueError(f"axis must be 0/1/2; got {axis}")
