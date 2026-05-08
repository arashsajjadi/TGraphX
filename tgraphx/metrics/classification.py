"""Classification metrics.

Inputs are either raw logits (preferred) or class indices.  Every
metric detaches its inputs and never retains an autograd graph.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


def _to_class_indices(preds: torch.Tensor) -> torch.Tensor:
    """Resolve the predicted class id for each sample.

    Accepts either ``[N]`` LongTensor of class ids or ``[N, C]`` float
    logits / probabilities.
    """
    if preds.dim() == 1:
        return preds.long()
    if preds.dim() == 2:
        return preds.argmax(dim=-1).long()
    raise ValueError(
        f"preds must be [N] or [N, C]; got shape {tuple(preds.shape)}"
    )


def _check_same_length(a: torch.Tensor, b: torch.Tensor, name_a: str = "preds",
                       name_b: str = "labels") -> None:
    if a.size(0) != b.size(0):
        raise ValueError(
            f"{name_a} and {name_b} length mismatch: {a.size(0)} vs {b.size(0)}"
        )


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 multiclass accuracy.

    Args:
        preds: ``[N]`` integer class ids OR ``[N, C]`` logits / probs.
        labels: ``[N]`` integer class ids.
    """
    preds = preds.detach()
    labels = labels.detach().long()
    pred_ids = _to_class_indices(preds)
    _check_same_length(pred_ids, labels)
    if labels.numel() == 0:
        return 0.0
    return float((pred_ids == labels).float().mean().item())


def top_k_accuracy(preds: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """Top-``k`` accuracy from logits / probabilities.

    Requires ``preds.shape == [N, C]`` with ``C >= k``.
    """
    if preds.dim() != 2:
        raise ValueError(
            f"top_k_accuracy expects [N, C] logits; got {tuple(preds.shape)}"
        )
    if k < 1:
        raise ValueError("k must be >= 1")
    preds = preds.detach()
    labels = labels.detach().long()
    _check_same_length(preds, labels)
    if labels.numel() == 0:
        return 0.0
    k = min(int(k), preds.size(1))
    topk = preds.topk(k, dim=-1).indices  # [N, k]
    hits = (topk == labels.unsqueeze(-1)).any(dim=-1)
    return float(hits.float().mean().item())


def confusion_matrix(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """Square ``[C, C]`` integer confusion matrix.

    Rows are true labels, columns are predicted labels.
    """
    preds = preds.detach()
    labels = labels.detach().long()
    pred_ids = _to_class_indices(preds)
    _check_same_length(pred_ids, labels)
    if num_classes is None:
        num_classes = int(max(int(pred_ids.max().item()) if pred_ids.numel() else 0,
                              int(labels.max().item()) if labels.numel() else 0) + 1)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    if pred_ids.numel() == 0:
        return cm
    flat = labels * num_classes + pred_ids
    counts = torch.bincount(flat, minlength=num_classes * num_classes)
    return counts.view(num_classes, num_classes)


def precision_recall_f1(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: Optional[int] = None,
    average: str = "macro",
    zero_division: float = 0.0,
) -> Dict[str, float]:
    """Per-class and aggregated precision / recall / F1.

    Args:
        preds, labels: as in :func:`accuracy`.
        num_classes: defaults to ``max(preds, labels) + 1``.
        average: ``"macro"`` (default) or ``"micro"`` for the
            aggregated values; per-class arrays are always returned.
        zero_division: value returned when a class has no predictions
            or no true samples (default ``0.0``).
    """
    if average not in ("macro", "micro"):
        raise ValueError(f"average must be 'macro' or 'micro'; got {average!r}")
    cm = confusion_matrix(preds, labels, num_classes=num_classes)
    C = cm.size(0)
    tp = cm.diag().float()
    pred_total = cm.sum(dim=0).float()
    true_total = cm.sum(dim=1).float()

    precision = torch.where(
        pred_total > 0, tp / pred_total.clamp_min(1.0), torch.full((C,), zero_division)
    )
    recall = torch.where(
        true_total > 0, tp / true_total.clamp_min(1.0), torch.full((C,), zero_division)
    )
    denom = (precision + recall).clamp_min(1e-12)
    f1 = torch.where(
        denom > 0, 2 * precision * recall / denom, torch.full((C,), zero_division)
    )

    if average == "macro":
        agg_p = float(precision.mean().item()) if C else 0.0
        agg_r = float(recall.mean().item()) if C else 0.0
        agg_f = float(f1.mean().item()) if C else 0.0
    else:  # micro
        total_tp = tp.sum().item()
        total_pred = pred_total.sum().item()
        total_true = true_total.sum().item()
        agg_p = total_tp / total_pred if total_pred > 0 else float(zero_division)
        agg_r = total_tp / total_true if total_true > 0 else float(zero_division)
        agg_f = (
            2 * agg_p * agg_r / (agg_p + agg_r)
            if (agg_p + agg_r) > 0 else float(zero_division)
        )

    return {
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "precision": float(agg_p),
        "recall": float(agg_r),
        "f1": float(agg_f),
        "average": average,
        "num_classes": int(C),
    }


def classification_report(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: Optional[int] = None,
    average: str = "macro",
) -> Dict[str, object]:
    """Combined accuracy + per-class P/R/F1 + confusion matrix.

    All values are JSON-friendly Python floats / lists.
    """
    cm = confusion_matrix(preds, labels, num_classes=num_classes)
    prf = precision_recall_f1(preds, labels, num_classes=cm.size(0), average=average)
    return {
        "accuracy": accuracy(preds, labels),
        "confusion_matrix": cm.tolist(),
        **prf,
    }
