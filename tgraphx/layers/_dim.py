"""Spatial-rank helpers for tensor-aware GNN layers.

TGraphX claims support for **three** node-feature layouts, no more:

* ``[N, D]``                      — vector features (rank 0).
* ``[N, C, H, W]``                — 2-D spatial features (rank 2).
* ``[N, C, D, H, W]``             — 3-D volumetric features (rank 3).

The convolutional layer families (``ConvMessagePassing``, ``TensorGATLayer``,
``TensorGraphSAGELayer``, ``TensorGINLayer``) operate on rank 2 or 3.  Rank
0 is handled by ``LinearMessagePassing`` and lives outside these helpers.

This module centralises the (small) rank-dependent choices: which 1×1
convolution module to use, which BatchNorm flavour, which dropout flavour,
and how to pool spatial axes.  Layers pick a single ``spatial_rank``
(``2`` by default for backward compatibility) at construction time and
stick with it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

CONV_RANKS = (2, 3)


def validate_spatial_rank(rank: int) -> int:
    """Raise unless ``rank`` is 2 or 3."""
    if rank not in CONV_RANKS:
        raise ValueError(
            f"spatial_rank must be 2 (for [N, C, H, W]) or 3 (for [N, C, D, H, W]); "
            f"got {rank!r}."
        )
    return rank


def expected_x_dim(rank: int) -> int:
    """Number of dimensions a node-feature tensor must have for ``rank``."""
    return 2 + rank  # leading [N, C] + spatial


def conv1x1(rank: int, in_ch: int, out_ch: int, bias: bool = True) -> nn.Module:
    """1x1 (rank 2) or 1x1x1 (rank 3) convolution."""
    if rank == 2:
        return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
    if rank == 3:
        return nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=bias)
    raise ValueError(f"conv1x1: spatial_rank must be 2 or 3; got {rank}")


def batchnorm(rank: int, channels: int) -> nn.Module:
    if rank == 2:
        return nn.BatchNorm2d(channels)
    if rank == 3:
        return nn.BatchNorm3d(channels)
    raise ValueError(f"batchnorm: spatial_rank must be 2 or 3; got {rank}")


def dropout_nd(rank: int, p: float) -> nn.Module:
    """Channel-wise dropout matching the spatial rank."""
    if rank == 2:
        return nn.Dropout2d(p=p)
    if rank == 3:
        return nn.Dropout3d(p=p)
    raise ValueError(f"dropout_nd: spatial_rank must be 2 or 3; got {rank}")


def mean_over_spatial(tensor: torch.Tensor, rank: int) -> torch.Tensor:
    """Mean over the trailing ``rank`` spatial axes (no-op when rank=0)."""
    if rank == 0:
        return tensor
    return tensor.mean(dim=tuple(range(-rank, 0)))


def view_for_channel_bias(rank: int, channels: int) -> tuple[int, ...]:
    """View used to broadcast a per-channel bias ``[C]`` over an ``[N, C, *spatial]`` tensor."""
    return (1, channels) + (1,) * rank


def trailing_ones(rank: int, extra: int = 0) -> tuple[int, ...]:
    """Tuple of ``rank + extra`` 1's — useful for ``view`` reshapes."""
    return (1,) * (rank + extra)


__all__ = [
    "CONV_RANKS",
    "validate_spatial_rank",
    "expected_x_dim",
    "conv1x1",
    "batchnorm",
    "dropout_nd",
    "mean_over_spatial",
    "view_for_channel_bias",
    "trailing_ones",
]
