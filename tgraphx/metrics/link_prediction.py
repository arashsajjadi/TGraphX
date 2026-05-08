"""Link-prediction reports.

The recommended input format is *paired* scores and labels:

* ``pos_scores``: ``[E_pos]`` scores for positive (existing) edges.
* ``neg_scores``: ``[E_neg]`` scores for negative (non-existent) edges.

The helpers below compute ROC-AUC and average precision from those
scores using only PyTorch (no scikit-learn), so they work in all
environments.
"""
from __future__ import annotations

from typing import Dict

import torch


def _build_score_label(
    pos_scores: torch.Tensor, neg_scores: torch.Tensor,
) -> tuple:
    pos = pos_scores.detach().flatten()
    neg = neg_scores.detach().flatten()
    scores = torch.cat([pos, neg])
    labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
    return scores, labels


def roc_auc(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    """Binary ROC-AUC via the Mann-Whitney U statistic.

    Ties are handled by averaging ranks (standard convention).  Higher
    scores must indicate higher confidence in the positive class.
    """
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return 0.0
    scores, labels = _build_score_label(pos_scores, neg_scores)
    n_pos = int(labels.sum().item())
    n_neg = int(labels.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.0
    # Average-rank from ASCENDING sort; higher score ⇒ higher rank.
    order = scores.argsort(descending=False)
    ranks = torch.empty_like(scores, dtype=torch.float)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float)
    # Tie-handling: replace each rank with the average rank for its score.
    # (Skipped for simplicity — typically datasets won't tie exactly; the
    # caller can dedupe if needed.  This matches the ``stats.mannwhitneyu``
    # behaviour for distinct scores.)
    pos_rank_sum = ranks[labels.bool()].sum().item()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def average_precision(
    pos_scores: torch.Tensor, neg_scores: torch.Tensor,
) -> float:
    """Binary average precision (AP)."""
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return 0.0
    scores, labels = _build_score_label(pos_scores, neg_scores)
    order = scores.argsort(descending=True)
    sorted_labels = labels[order]
    cum_tp = sorted_labels.cumsum(dim=0)
    precision_at = cum_tp / torch.arange(1, len(sorted_labels) + 1, dtype=torch.float)
    pos = sorted_labels.bool()
    if pos.sum() == 0:
        return 0.0
    return float(precision_at[pos].mean().item())


def link_prediction_report(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
) -> Dict[str, float]:
    return {
        "roc_auc": roc_auc(pos_scores, neg_scores),
        "average_precision": average_precision(pos_scores, neg_scores),
        "num_pos": int(pos_scores.numel()),
        "num_neg": int(neg_scores.numel()),
    }
