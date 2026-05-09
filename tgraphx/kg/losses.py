"""KG training losses.

All losses take positive and negative score tensors:
  pos_scores: FloatTensor[B]  — scores for positive triples
  neg_scores: FloatTensor[B*K] or FloatTensor[B, K]  — negative scores

Losses return a scalar.

Available:
  MarginRankingLoss   — max(0, γ - s_pos + s_neg)
  BCEKGLoss           — BCEWithLogits on labelled scores
  SoftplusKGLoss      — softplus(-s_pos) + softplus(s_neg)

Regularisers:
  l2_regularization   — L2 norm on all embedding weights
  n3_regularization   — N3 on complex embeddings

Stability: Beta.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "MarginRankingLoss",
    "BCEKGLoss",
    "SoftplusKGLoss",
    "l2_regularization",
]


class MarginRankingLoss(nn.Module):
    """Margin-based pairwise ranking loss.

    L = mean(max(0, γ - s_pos + s_neg))

    Args:
        margin: γ > 0.

    Stability: Beta.
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        if margin <= 0:
            raise ValueError(f"margin must be > 0; got {margin}")
        self.margin = float(margin)

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute margin loss.

        Args:
            pos_scores: ``FloatTensor[B]``.
            neg_scores: ``FloatTensor[B*K]`` or ``FloatTensor[B, K]``.
                Repeated or tiled to match pos_scores if K > 1.

        Returns:
            Scalar loss.
        """
        pos = pos_scores.view(-1)
        neg = neg_scores.view(-1)
        if neg.size(0) != pos.size(0):
            # K > 1: repeat positives to match.
            K = neg.size(0) // pos.size(0)
            pos = pos.repeat_interleave(K)
        return F.relu(self.margin - pos + neg).mean()


class BCEKGLoss(nn.Module):
    """Binary cross-entropy KG loss.

    L = -mean(log σ(s_pos)) - mean(log σ(-s_neg))
      = BCEWithLogits with positive labels=1, negative labels=0.

    Stability: Beta.
    """

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores.view(-1), torch.zeros(neg_scores.numel(), device=neg_scores.device)
        )
        return pos_loss + neg_loss


class SoftplusKGLoss(nn.Module):
    """Softplus loss (Trouillon et al., 2016).

    L = mean(softplus(-s_pos)) + mean(softplus(s_neg))
      = mean(log(1 + exp(-s_pos))) + mean(log(1 + exp(s_neg)))

    Stability: Beta.
    """

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        pos_loss = F.softplus(-pos_scores).mean()
        neg_loss = F.softplus(neg_scores.view(-1)).mean()
        return pos_loss + neg_loss


def l2_regularization(
    params: Iterable[torch.Tensor],
    weight: float = 1e-3,
) -> torch.Tensor:
    """Compute L2 regularisation on a sequence of parameters.

    Args:
        params: Iterable of tensors (e.g. embedding weights).
        weight: Regularisation coefficient λ.

    Returns:
        Scalar ``λ * Σ ||p||²``.
    """
    reg = torch.tensor(0.0)
    for p in params:
        reg = reg + p.pow(2).sum()
    return weight * reg
