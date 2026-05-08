"""Heterogeneous graph readouts and pooling.

.. experimental::
    🧪 These readouts are experimental but functional.  They operate on
    ``x_dict`` outputs of :class:`tgraphx.layers.hetero.HeteroConv`
    optionally together with the per-type ``batch_dict`` produced by
    :class:`tgraphx.core.hetero_batch.HeteroGraphBatch`.

Functions
---------
hetero_mean_pool(x_dict, batch_dict=None)
    Per-type mean pool.  When ``batch_dict`` is provided, returns a dict
    of ``[B, *]`` tensors; otherwise returns ``[1, *]`` tensors.

hetero_sum_pool / hetero_max_pool
    Same as above with sum / max reduction.

hetero_concat_pool(x_dict, batch_dict=None, type_order=None, mode="mean")
    Pool per-type then concatenate into a single ``[B, sum_t D_t]`` tensor
    using a stable type ordering.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ..core.hetero_graph import NodeType

__all__ = [
    "hetero_mean_pool",
    "hetero_sum_pool",
    "hetero_max_pool",
    "hetero_concat_pool",
]


def _scatter_pool(
    x: torch.Tensor,
    batch: Optional[torch.Tensor],
    mode: str,
) -> torch.Tensor:
    """Pool ``x`` ([N, *]) per-graph using ``batch`` ([N], long).

    When ``batch`` is None, returns ``[1, *]`` (single-graph case).
    """
    if batch is None:
        if mode == "sum":
            return x.sum(dim=0, keepdim=True)
        if mode == "mean":
            return x.mean(dim=0, keepdim=True)
        if mode == "max":
            return x.max(dim=0, keepdim=True).values
        raise ValueError(f"unknown mode {mode!r}")

    if batch.numel() != x.size(0):
        raise ValueError(
            f"batch length {batch.numel()} != x rows {x.size(0)}"
        )
    if x.size(0) == 0:
        # No nodes — return a single zero row to keep shapes consistent.
        return x.new_zeros((int(batch.max().item()) + 1 if batch.numel() else 1,
                             *x.shape[1:]))
    B = int(batch.max().item()) + 1
    if mode == "sum":
        out = x.new_zeros((B, *x.shape[1:]))
        out.index_add_(0, batch, x)
        return out
    if mode == "mean":
        out = x.new_zeros((B, *x.shape[1:]))
        out.index_add_(0, batch, x)
        counts = x.new_zeros(B)
        counts.index_add_(0, batch, x.new_ones(x.size(0)))
        view = (B,) + (1,) * (out.dim() - 1)
        return out / counts.view(view).clamp_min(1.0)
    if mode == "max":
        out = x.new_full((B, *x.shape[1:]), float("-inf"))
        # Use scatter_reduce_ with amax (PyTorch >= 1.13)
        idx = batch.view(-1, *(1,) * (x.dim() - 1)).expand_as(x)
        out.scatter_reduce_(0, idx, x, reduce="amax", include_self=True)
        return out.masked_fill(torch.isinf(out) & (out < 0), 0.0)
    raise ValueError(f"unknown mode {mode!r}")


def hetero_mean_pool(
    x_dict: Dict[NodeType, torch.Tensor],
    batch_dict: Optional[Dict[NodeType, torch.Tensor]] = None,
) -> Dict[NodeType, torch.Tensor]:
    """Per-type mean readout."""
    return {
        t: _scatter_pool(x, batch_dict.get(t) if batch_dict else None, "mean")
        for t, x in x_dict.items()
    }


def hetero_sum_pool(
    x_dict: Dict[NodeType, torch.Tensor],
    batch_dict: Optional[Dict[NodeType, torch.Tensor]] = None,
) -> Dict[NodeType, torch.Tensor]:
    """Per-type sum readout."""
    return {
        t: _scatter_pool(x, batch_dict.get(t) if batch_dict else None, "sum")
        for t, x in x_dict.items()
    }


def hetero_max_pool(
    x_dict: Dict[NodeType, torch.Tensor],
    batch_dict: Optional[Dict[NodeType, torch.Tensor]] = None,
) -> Dict[NodeType, torch.Tensor]:
    """Per-type max readout."""
    return {
        t: _scatter_pool(x, batch_dict.get(t) if batch_dict else None, "max")
        for t, x in x_dict.items()
    }


def hetero_concat_pool(
    x_dict: Dict[NodeType, torch.Tensor],
    batch_dict: Optional[Dict[NodeType, torch.Tensor]] = None,
    type_order: Optional[List[NodeType]] = None,
    mode: str = "mean",
) -> torch.Tensor:
    """Pool per-type then concatenate into ``[B, sum_t D_t]``.

    Args:
        x_dict: Per-type node features.
        batch_dict: Per-type batch vectors (optional — single-graph if None).
        type_order: Stable ordering of node types for concatenation.
            Defaults to ``sorted(x_dict.keys())`` for determinism.
        mode: ``"mean"`` (default), ``"sum"``, or ``"max"``.

    Returns:
        Concatenated ``[B, total_dim]`` tensor.  Per-type features that
        have a spatial layout are flattened over their non-batch dims.
    """
    if mode == "mean":
        pooled = hetero_mean_pool(x_dict, batch_dict)
    elif mode == "sum":
        pooled = hetero_sum_pool(x_dict, batch_dict)
    elif mode == "max":
        pooled = hetero_max_pool(x_dict, batch_dict)
    else:
        raise ValueError(f"mode must be 'mean', 'sum', or 'max'; got {mode!r}")

    types = type_order if type_order is not None else sorted(x_dict.keys())
    flat_parts = [pooled[t].flatten(start_dim=1) for t in types]
    return torch.cat(flat_parts, dim=1)
