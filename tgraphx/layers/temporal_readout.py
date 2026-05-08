"""Temporal readouts: pool a sequence of per-snapshot embeddings.

.. experimental::
    🧪 The readouts are simple, transparent, and CPU-safe.  No recurrent
    memory module — that level of complexity (TGN/TGAT) is deferred.

API
---
``temporal_readout(sequence_embeddings, mode="last", mask=None)``

* ``sequence_embeddings``: tensor ``[T, B, D]`` — per-snapshot batch
  embeddings.
* ``mode``: ``"last"`` (default), ``"mean"``, or ``"max"``.  The
  ``"last"`` mode picks the last *valid* snapshot when ``mask`` is given.
* ``mask``: optional bool tensor ``[T, B]`` (True where the snapshot
  exists).  When provided, ``"mean"`` and ``"last"`` respect the mask.

Returns ``[B, D]``.
"""
from __future__ import annotations

from typing import Optional

import torch

__all__ = ["temporal_readout"]


def temporal_readout(
    sequence_embeddings: torch.Tensor,
    mode: str = "last",
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reduce a ``[T, B, D]`` sequence into ``[B, D]``."""
    if sequence_embeddings.dim() != 3:
        raise ValueError(
            f"sequence_embeddings must be 3-D [T, B, D]; got "
            f"{tuple(sequence_embeddings.shape)}"
        )
    T, B, D = sequence_embeddings.shape
    if mask is not None:
        if mask.shape != (T, B):
            raise ValueError(
                f"mask must have shape [T, B]={(T, B)}; got {tuple(mask.shape)}"
            )
        if mask.dtype != torch.bool:
            mask = mask.bool()

    if mode == "last":
        if mask is None:
            return sequence_embeddings[-1]
        # For each batch, find the last True index in mask.
        # mask: [T, B] → for each b, last_idx = max t such that mask[t, b].
        last_idx = torch.full((B,), -1, dtype=torch.long, device=mask.device)
        for t in range(T):
            last_idx = torch.where(mask[t], torch.full_like(last_idx, t), last_idx)
        # last_idx must be >= 0 for every batch.
        if (last_idx < 0).any():
            raise ValueError(
                "mode='last' requires every batch element to have at least "
                "one True entry in mask; some sequences are all-False."
            )
        # gather along T dim using advanced indexing
        out = sequence_embeddings[
            last_idx, torch.arange(B, device=sequence_embeddings.device)
        ]
        return out

    if mode == "mean":
        if mask is None:
            return sequence_embeddings.mean(dim=0)
        m = mask.to(sequence_embeddings.dtype).unsqueeze(-1)  # [T, B, 1]
        masked = sequence_embeddings * m
        denom = m.sum(dim=0).clamp_min(1.0)  # [B, 1]
        return masked.sum(dim=0) / denom

    if mode == "max":
        if mask is None:
            return sequence_embeddings.max(dim=0).values
        # Set masked-out positions to -inf so they cannot win the max.
        neg_inf = torch.finfo(sequence_embeddings.dtype).min
        m = mask.unsqueeze(-1)  # [T, B, 1]
        masked = sequence_embeddings.masked_fill(~m, neg_inf)
        out = masked.max(dim=0).values
        # Replace -inf (all-masked) with 0.
        return out.masked_fill(out == neg_inf, 0.0)

    raise ValueError(f"mode must be 'last', 'mean', or 'max'; got {mode!r}")
