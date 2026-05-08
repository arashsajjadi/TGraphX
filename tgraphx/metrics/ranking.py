"""Ranking-style metrics for link prediction and retrieval.

All implementations are pure PyTorch.  Inputs:

* ``scores``: ``[N, M]`` float tensor where row ``i`` ranks ``M``
  candidates for query ``i``.
* ``targets``: either a ``[N]`` LongTensor of correct candidate
  indices, or a ``[N, M]`` boolean / float tensor where positive
  entries indicate relevant items.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch


def _ensure_2d(scores: torch.Tensor) -> torch.Tensor:
    if scores.dim() != 2:
        raise ValueError(
            f"scores must be [N, M]; got shape {tuple(scores.shape)}"
        )
    return scores


def hits_at_k(
    scores: torch.Tensor,
    target_idx: torch.Tensor,
    k: int = 10,
) -> float:
    """Fraction of queries whose correct item is in the top-``k``.

    Args:
        scores: ``[N, M]`` candidate scores (higher = better).
        target_idx: ``[N]`` LongTensor of correct candidate ids
            (one per query).
    """
    scores = _ensure_2d(scores).detach()
    target_idx = target_idx.detach().long()
    if scores.size(0) != target_idx.size(0):
        raise ValueError("scores and target_idx must have matching first dim")
    if k < 1:
        raise ValueError("k must be >= 1")
    if scores.numel() == 0:
        return 0.0
    k = min(int(k), scores.size(1))
    top = scores.topk(k, dim=-1).indices  # [N, k]
    hits = (top == target_idx.unsqueeze(-1)).any(dim=-1)
    return float(hits.float().mean().item())


def mean_reciprocal_rank(
    scores: torch.Tensor,
    target_idx: torch.Tensor,
) -> float:
    """Mean Reciprocal Rank (MRR).

    Rank 1 ⇒ contribution 1.0; rank 2 ⇒ 0.5; ...
    """
    scores = _ensure_2d(scores).detach()
    target_idx = target_idx.detach().long()
    if scores.size(0) != target_idx.size(0):
        raise ValueError("scores and target_idx must have matching first dim")
    if scores.numel() == 0:
        return 0.0
    # Argsort scores descending, find rank of target.
    order = scores.argsort(dim=-1, descending=True)
    target_pos = (order == target_idx.unsqueeze(-1)).float().argmax(dim=-1).float()
    ranks = target_pos + 1.0
    return float((1.0 / ranks).mean().item())


def ndcg_at_k(
    scores: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
) -> float:
    """Mean nDCG@``k`` across queries.

    Args:
        scores: ``[N, M]`` candidate scores.
        targets: ``[N, M]`` non-negative relevance grades; values
            higher than zero are considered relevant.
    """
    scores = _ensure_2d(scores).detach()
    if targets.dim() == 1:
        # If a 1-D index tensor was passed, lift it to a relevance matrix.
        oh = torch.zeros(scores.shape, dtype=torch.float)
        oh[torch.arange(scores.size(0)), targets.long()] = 1.0
        targets = oh
    elif targets.shape != scores.shape:
        raise ValueError(
            f"targets must be [N, M]; got {tuple(targets.shape)} for scores "
            f"{tuple(scores.shape)}"
        )
    targets = targets.detach().float()
    if scores.numel() == 0:
        return 0.0
    k = min(int(k), scores.size(1))
    if k < 1:
        return 0.0
    order = scores.argsort(dim=-1, descending=True)[:, :k]
    gains = torch.gather(targets, 1, order)
    discount = torch.log2(
        torch.arange(2, k + 2, device=scores.device, dtype=torch.float)
    )
    dcg = (gains / discount).sum(dim=-1)
    # Ideal DCG using the top-k of the *true* relevance.
    ideal = targets.sort(dim=-1, descending=True).values[:, :k]
    idcg = (ideal / discount).sum(dim=-1).clamp_min(1e-12)
    ndcg = dcg / idcg
    return float(ndcg.mean().item())
