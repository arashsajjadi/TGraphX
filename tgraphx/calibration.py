"""Calibration utilities for classification models.

Implements:
  - :func:`expected_calibration_error` — ECE on binned predictions.
  - :func:`reliability_diagram_data` — data for a reliability diagram.
  - :class:`TemperatureScaler` — post-hoc temperature scaling.
  - :func:`calibrate_temperature` — fit temperature on a validation set.

These utilities operate on raw logits (no autograd retained in metrics).

Stability: Beta (v0.5.0+).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "expected_calibration_error",
    "reliability_diagram_data",
    "TemperatureScaler",
    "calibrate_temperature",
]


# ── ECE ───────────────────────────────────────────────────────────────────────


@torch.no_grad()
def expected_calibration_error(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) from raw logits.

    ECE = Σ_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    where the sum is over equal-width confidence bins.

    Args:
        logits: ``FloatTensor[N, C]`` raw class logits.
        labels: ``LongTensor[N]`` ground-truth class indices.
        n_bins: Number of equally-spaced confidence bins.

    Returns:
        ECE in ``[0, 1]`` (float, no autograd).

    Notes:
        Empty bins contribute zero.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be [N, C]; got {tuple(logits.shape)}")
    if labels.dim() != 1 or labels.size(0) != logits.size(0):
        raise ValueError("labels must be [N] matching logits")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2; got {n_bins}")

    probs = F.softmax(logits.float(), dim=-1)
    confidence, predicted = probs.max(dim=-1)
    correct = (predicted == labels.to(predicted.device)).float()
    n = logits.size(0)

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=confidence.device)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = float(bin_boundaries[i]), float(bin_boundaries[i + 1])
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        if not mask.any():
            continue
        bin_conf = float(confidence[mask].mean().item())
        bin_acc = float(correct[mask].mean().item())
        ece += (int(mask.sum().item()) / n) * abs(bin_acc - bin_conf)
    return float(ece)


@torch.no_grad()
def reliability_diagram_data(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> Dict[str, List[float]]:
    """Compute data for a reliability diagram.

    Returns a dict with ``bin_confidences``, ``bin_accuracies``,
    ``bin_fractions``, and ``ece`` — all JSON-serialisable.
    """
    probs = F.softmax(logits.float(), dim=-1)
    confidence, predicted = probs.max(dim=-1)
    correct = (predicted == labels.to(predicted.device)).float()
    n = logits.size(0)

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=confidence.device)
    confs, accs, fracs = [], [], []
    for i in range(n_bins):
        lo = float(bin_boundaries[i])
        hi = float(bin_boundaries[i + 1])
        mask = (confidence >= lo) & (confidence < hi) if i < n_bins - 1 else (confidence >= lo) & (confidence <= hi)
        if mask.any():
            confs.append(round(float(confidence[mask].mean().item()), 4))
            accs.append(round(float(correct[mask].mean().item()), 4))
            fracs.append(round(int(mask.sum().item()) / n, 4))
        else:
            mid = round((lo + hi) / 2, 4)
            confs.append(mid)
            accs.append(0.0)
            fracs.append(0.0)
    return {
        "bin_confidences": confs,
        "bin_accuracies": accs,
        "bin_fractions": fracs,
        "ece": round(expected_calibration_error(logits, labels, n_bins), 6),
    }


# ── Temperature scaling ───────────────────────────────────────────────────────


class TemperatureScaler(nn.Module):
    """Single scalar temperature on logits: ``logits_scaled = logits / T``.

    Args:
        temperature: Initial temperature (must be > 0).

    Stability: Beta.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0; got {temperature}")
        self.temperature = nn.Parameter(torch.tensor([float(temperature)]))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-4)

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper (no_grad)."""
        with torch.no_grad():
            return self.forward(logits)

    def extra_repr(self) -> str:
        return f"T={float(self.temperature.item()):.4f}"


def calibrate_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    lr: float = 0.01,
    max_iter: int = 50,
) -> TemperatureScaler:
    """Fit a :class:`TemperatureScaler` on validation ``(logits, labels)``.

    Uses NLL minimisation.  Convergence is not guaranteed in a fixed
    number of steps; check ``scaler.temperature.item()`` after calling.

    Args:
        logits: ``FloatTensor[N, C]`` validation logits.
        labels: ``LongTensor[N]``.
        lr: LBFGS learning rate.
        max_iter: Optimiser steps.

    Returns:
        Fitted :class:`TemperatureScaler`.
    """
    scaler = TemperatureScaler(temperature=1.0)
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_iter)
    logits_ = logits.detach().float()
    labels_ = labels.detach().long()

    def closure():
        optimizer.zero_grad()
        scaled = scaler(logits_)
        loss = F.cross_entropy(scaled, labels_)
        loss.backward()
        return loss

    optimizer.step(closure)
    # Clamp to a sane range.
    with torch.no_grad():
        scaler.temperature.clamp_(min=0.01, max=100.0)
    return scaler
