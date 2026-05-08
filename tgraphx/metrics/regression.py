"""Regression metrics."""
from __future__ import annotations

from typing import Dict

import torch


def _detach_pair(preds: torch.Tensor, targets: torch.Tensor) -> tuple:
    if preds.shape != targets.shape:
        # Allow broadcasting through reshape; but warn loudly via error.
        raise ValueError(
            f"preds and targets shape mismatch: {tuple(preds.shape)} vs "
            f"{tuple(targets.shape)}"
        )
    return preds.detach().float(), targets.detach().float()


def mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    p, t = _detach_pair(preds, targets)
    if p.numel() == 0:
        return 0.0
    return float((p - t).abs().mean().item())


def mse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    p, t = _detach_pair(preds, targets)
    if p.numel() == 0:
        return 0.0
    return float(((p - t) ** 2).mean().item())


def rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return mse(preds, targets) ** 0.5


def r2_score(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Coefficient of determination ``R²``.

    Returns ``0.0`` when target variance is zero (degenerate case).
    """
    p, t = _detach_pair(preds, targets)
    if p.numel() == 0:
        return 0.0
    ss_res = ((t - p) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    if float(ss_tot.item()) == 0.0:
        return 0.0
    return float((1.0 - ss_res / ss_tot).item())


def regression_report(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    return {
        "mae": mae(preds, targets),
        "mse": mse(preds, targets),
        "rmse": rmse(preds, targets),
        "r2": r2_score(preds, targets),
    }
