"""Translate ``TensorGATLayer`` attention weights to per-edge scores."""
from __future__ import annotations

from typing import Optional

import torch


def attention_to_edge_scores(
    attn: torch.Tensor,
    edge_index: torch.Tensor,
    head_reduce: str = "mean",
) -> torch.Tensor:
    """Reduce ``[E, K]`` (or ``[E, K, C_head]``) attention to ``[E]`` edge scores.

    Args:
        attn: Attention tensor as returned by
            ``TensorGATLayer(..., return_attention=True)``.
        edge_index: ``[2, E]`` edge index used in the same forward pass.
        head_reduce: ``"mean"`` (default), ``"sum"``, or ``"max"``.

    Returns:
        ``[E]`` float tensor; values are in ``[0, 1]`` for
        ``head_reduce='mean'`` because attention weights are softmax-
        normalised per destination per head.
    """
    if attn.dim() < 2:
        raise ValueError(
            f"attn must be [E, K] or [E, K, C_head]; got {tuple(attn.shape)}"
        )
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"edge_index must be [2, E]; got {tuple(edge_index.shape)}"
        )
    if attn.size(0) != edge_index.size(1):
        raise ValueError(
            f"attn rows ({attn.size(0)}) must match edge_index columns "
            f"({edge_index.size(1)})"
        )

    a = attn.detach()
    if a.dim() == 3:
        # Average over channels for per-edge view (channel mode).
        a = a.mean(dim=-1)
    if head_reduce == "mean":
        return a.mean(dim=-1)
    if head_reduce == "sum":
        return a.sum(dim=-1)
    if head_reduce == "max":
        return a.max(dim=-1).values
    raise ValueError(
        f"head_reduce must be 'mean', 'sum', or 'max'; got {head_reduce!r}"
    )
