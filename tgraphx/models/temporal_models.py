"""Experimental temporal graph models — snapshot-loop + readout.

.. experimental::
    🧪 ``TemporalGraphClassifier`` and ``TemporalGraphRegressor`` apply a
    **stateless** base graph model independently per snapshot, then pool
    the per-snapshot embeddings via :func:`temporal_readout`.  No
    recurrent memory module is implemented — TGN/TGAT-style architectures
    are explicitly deferred to v0.2.6+.

Workflow
--------
Given a base model ``M`` mapping a single :class:`Graph` (or
:class:`GraphBatch`) to a vector ``[B, D]`` (e.g., a graph classifier
without its final head), and a temporal batch:

    z_t = M(snapshot_t)          # [B, D]   for t = 0 .. T-1
    h   = readout(z_0..z_{T-1})  # [B, D]
    out = head(h)                # [B, num_classes / num_targets]

When a :class:`TemporalGraphBatch` is variable-length, the base model is
applied only to the active sequences at each time step and a mask is
forwarded to the readout.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn as nn

from ..core.temporal_batch import TemporalGraphBatch
from ..layers.temporal_readout import temporal_readout

__all__ = ["TemporalGraphClassifier", "TemporalGraphRegressor"]


def _apply_per_snapshot(
    base_model: Callable,
    temporal_batch: TemporalGraphBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ``base_model`` to each snapshot, return (embeddings [T, B, D], mask [T, B]).

    The base model must accept a :class:`GraphBatch` and return
    ``[active_B, D]``.  Inactive (masked-out) batch slots are filled with
    zeros.  The mask reflects which sequences had a snapshot at each time.
    """
    T = temporal_batch.max_length
    B = temporal_batch.num_sequences
    embeddings: List[torch.Tensor] = []
    mask = torch.zeros((T, B), dtype=torch.bool)
    feature_dim: Optional[int] = None
    for t, gb, m in temporal_batch:
        emb_active = base_model(gb)
        if emb_active.dim() != 2:
            raise ValueError(
                f"base_model must return a 2-D [active_B, D] tensor per "
                f"snapshot; got shape {tuple(emb_active.shape)}"
            )
        if feature_dim is None:
            feature_dim = emb_active.size(1)
        elif emb_active.size(1) != feature_dim:
            raise ValueError(
                f"base_model returned inconsistent feature dim: t=0 had "
                f"{feature_dim}, t={t} has {emb_active.size(1)}"
            )
        # Scatter active embeddings into [B, D].
        full = torch.zeros((B, feature_dim), device=emb_active.device,
                           dtype=emb_active.dtype)
        active_idx = torch.where(m)[0]
        full[active_idx] = emb_active
        embeddings.append(full)
        mask[t] = m

    return torch.stack(embeddings, dim=0), mask  # [T, B, D]


class TemporalGraphClassifier(nn.Module):
    """🧪 Experimental: classify each temporal sequence with snapshot loop + readout.

    Args:
        base_model: A stateless module mapping a :class:`GraphBatch` to a
            ``[active_B, D]`` embedding tensor.  Typically a graph
            classifier without its final classification head, or a
            small GNN+pool composition.
        feature_dim: ``D`` — the embedding dimension produced by
            ``base_model``.  Used to size the final classification head.
        num_classes: Output classification dimension.
        readout: ``"last"`` (default), ``"mean"``, or ``"max"``.
        dropout: Dropout applied after readout (default 0.0).
    """

    def __init__(
        self,
        base_model: nn.Module,
        feature_dim: int,
        num_classes: int,
        readout: str = "last",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if readout not in ("last", "mean", "max"):
            raise ValueError(
                f"readout must be 'last', 'mean', or 'max'; got {readout!r}"
            )
        self.base_model = base_model
        self.readout = readout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, temporal_batch: TemporalGraphBatch) -> torch.Tensor:
        embeddings, mask = _apply_per_snapshot(self.base_model, temporal_batch)
        # If not variable-length, we can pass mask=None for slightly faster path.
        use_mask = mask if temporal_batch.is_variable_length else None
        h = temporal_readout(embeddings, mode=self.readout, mask=use_mask)
        h = self.dropout(h)
        return self.head(h)


class TemporalGraphRegressor(nn.Module):
    """🧪 Experimental: regress each temporal sequence to a scalar/vector.

    Same architecture as :class:`TemporalGraphClassifier` with a regression
    head (no softmax/argmax intent).
    """

    def __init__(
        self,
        base_model: nn.Module,
        feature_dim: int,
        out_dim: int = 1,
        readout: str = "last",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if readout not in ("last", "mean", "max"):
            raise ValueError(
                f"readout must be 'last', 'mean', or 'max'; got {readout!r}"
            )
        self.base_model = base_model
        self.readout = readout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(feature_dim, out_dim)

    def forward(self, temporal_batch: TemporalGraphBatch) -> torch.Tensor:
        embeddings, mask = _apply_per_snapshot(self.base_model, temporal_batch)
        use_mask = mask if temporal_batch.is_variable_length else None
        h = temporal_readout(embeddings, mode=self.readout, mask=use_mask)
        h = self.dropout(h)
        return self.head(h)
