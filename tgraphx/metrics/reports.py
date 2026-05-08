"""Convenience aggregations across metric families."""
from __future__ import annotations

from typing import Dict

import torch

from .classification import classification_report
from .regression import regression_report


def graph_classification_report(
    preds: torch.Tensor, labels: torch.Tensor, num_classes=None,
) -> Dict[str, object]:
    return classification_report(preds, labels, num_classes=num_classes)


def node_classification_report(
    preds: torch.Tensor, labels: torch.Tensor, num_classes=None,
    mask: torch.Tensor | None = None,
) -> Dict[str, object]:
    if mask is not None:
        preds = preds[mask]
        labels = labels[mask]
    return classification_report(preds, labels, num_classes=num_classes)


def edge_classification_report(
    preds: torch.Tensor, labels: torch.Tensor, num_classes=None,
) -> Dict[str, object]:
    return classification_report(preds, labels, num_classes=num_classes)


def graph_regression_report(
    preds: torch.Tensor, targets: torch.Tensor,
) -> Dict[str, float]:
    return regression_report(preds, targets)


__all__ = [
    "graph_classification_report",
    "node_classification_report",
    "edge_classification_report",
    "graph_regression_report",
]
