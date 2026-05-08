"""Graph-level pooling utilities (vector-friendly, batch-aware)."""
from __future__ import annotations

import torch


def _check_batch(batch: torch.Tensor, num_nodes: int) -> int:
    if batch.dim() != 1 or batch.numel() != num_nodes:
        raise ValueError(
            f"batch must be a 1-D LongTensor of length {num_nodes}; "
            f"got shape {tuple(batch.shape)}."
        )
    if batch.dtype != torch.long:
        raise TypeError(f"batch must be torch.long; got {batch.dtype}.")
    return int(batch.max().item()) + 1 if batch.numel() else 0


def global_sum_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Sum node features per graph.

    Args:
        x: ``[N, *]`` node features (any trailing dims).
        batch: ``[N]`` LongTensor mapping each node to its graph index.

    Returns:
        ``[G, *]`` per-graph aggregated features.
    """
    G = _check_batch(batch, x.size(0))
    out = x.new_zeros((G,) + tuple(x.shape[1:]))
    out.index_add_(0, batch, x)
    return out


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Mean of node features per graph (zero-graph protection)."""
    G = _check_batch(batch, x.size(0))
    out = x.new_zeros((G,) + tuple(x.shape[1:]))
    out.index_add_(0, batch, x)
    counts = torch.bincount(batch, minlength=G).clamp_min(1).to(x.dtype)
    while counts.dim() < out.dim():
        counts = counts.unsqueeze(-1)
    return out / counts


def global_max_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Per-graph max pool using ``scatter_reduce_(amax)``."""
    G = _check_batch(batch, x.size(0))
    out = x.new_full((G,) + tuple(x.shape[1:]), float("-inf"))
    # Expand batch indices to match trailing dims.
    expanded = batch
    for _ in range(x.dim() - 1):
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(x)
    out.scatter_reduce_(0, expanded, x, reduce="amax", include_self=True)
    # Replace -inf with 0 for empty groups (shouldn't happen in well-formed batches).
    out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    return out


__all__ = ["global_sum_pool", "global_mean_pool", "global_max_pool"]
