"""Integrated Gradients (Sundararajan et al., 2017) for graph models."""
from __future__ import annotations

from typing import Any, Optional

import torch

from .saliency import _wrap_graph
from .utils import call_model, select_target_logit


def integrated_gradients(
    model: torch.nn.Module,
    graph,
    target: Any = 0,
    baseline: Optional[torch.Tensor] = None,
    steps: int = 16,
) -> torch.Tensor:
    """Riemann-sum Integrated Gradients on ``graph.node_features``.

    Args:
        model: model with a ``(node_features, edge_index)`` signature.
        graph: TGraphX :class:`Graph`.
        target: target class for classification.
        baseline: tensor with the same shape as ``graph.node_features``.
            Defaults to a zero baseline.
        steps: integration steps (>=2).

    Returns:
        Tensor with the same shape as ``graph.node_features`` —
        the per-element attribution.
    """
    if steps < 2:
        raise ValueError(f"steps must be >= 2; got {steps}")
    x = graph.node_features.detach()
    if baseline is None:
        baseline = torch.zeros_like(x)
    if baseline.shape != x.shape:
        raise ValueError(
            f"baseline shape {tuple(baseline.shape)} != "
            f"node_features {tuple(x.shape)}"
        )

    model_was_training = model.training
    model.eval()
    try:
        alphas = torch.linspace(0.0, 1.0, steps=steps, device=x.device, dtype=x.dtype)
        total_grad = torch.zeros_like(x)
        for alpha in alphas:
            interp = (baseline + alpha * (x - baseline)).clone().requires_grad_(True)
            proxy = _wrap_graph(graph, node_features=interp)
            logits = call_model(model, proxy)
            scalar = select_target_logit(logits, target)
            grad, = torch.autograd.grad(scalar, interp,
                                        retain_graph=False, create_graph=False)
            total_grad = total_grad + grad.detach()
    finally:
        if model_was_training:
            model.train()

    avg_grad = total_grad / float(steps)
    return ((x - baseline) * avg_grad).detach()
