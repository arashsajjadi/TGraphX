"""Label-propagation estimator wrapping :func:`tgraphx.mining.label_propagation`."""
from __future__ import annotations

from typing import Any, Optional

import torch

from .base import BaseGraphEstimator

__all__ = ["LabelPropagationEstimator"]


class LabelPropagationEstimator(BaseGraphEstimator):
    """Sklearn-like wrapper around iterative label propagation.

    Args:
        num_iters: Propagation iterations.
        alpha: Smoothing weight (0 = pure copy, 1 = pure propagation).
        seed: Optional seed (forwarded to underlying utilities).

    Stability: Beta.
    """

    def __init__(
        self,
        num_iters: int = 50,
        alpha: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        self.num_iters = int(num_iters)
        self.alpha = float(alpha)
        self.seed = seed
        self._labels: Optional[torch.Tensor] = None
        self._num_classes: Optional[int] = None

    def fit(self, graph: Any, y: torch.Tensor = None) -> "LabelPropagationEstimator":
        """Fit by storing seed labels and running propagation.

        Args:
            graph: A :class:`tgraphx.Graph`.
            y: ``LongTensor[N]`` of labels with ``-1`` for unlabeled
                nodes.  When ``None``, uses ``graph.node_labels``.
        """
        if y is None:
            y = graph.node_labels
        if y is None:
            raise ValueError("y must be provided (graph.node_labels was None)")
        N = graph.num_nodes
        if y.numel() != N:
            raise ValueError(f"y must have shape [N]; got {tuple(y.shape)}")
        # Build one-hot soft-labels (uniform for unlabeled).
        labeled_mask = y >= 0
        if labeled_mask.sum().item() == 0:
            raise ValueError("at least one node must be labeled (y >= 0)")
        num_classes = int(y[labeled_mask].max().item()) + 1
        self._num_classes = num_classes
        F = torch.zeros(N, num_classes, dtype=torch.float)
        F[labeled_mask, y[labeled_mask]] = 1.0
        F[~labeled_mask] = 1.0 / num_classes
        # Iterative propagation: F_{t+1} = alpha * A_norm @ F_t + (1-alpha) * F_seed.
        F_seed = F.clone()
        if graph.edge_index is not None and graph.num_edges:
            ei = graph.edge_index.cpu()
            deg = torch.zeros(N, dtype=torch.float)
            ones = torch.ones(ei.size(1), dtype=torch.float)
            deg.scatter_add_(0, ei[1], ones)
            deg = deg.clamp(min=1.0)
        for _ in range(self.num_iters):
            if graph.edge_index is not None and graph.num_edges:
                msg = F[ei[0]]
                agg = torch.zeros_like(F)
                agg.scatter_add_(0, ei[1].unsqueeze(-1).expand_as(msg), msg)
                agg = agg / deg.unsqueeze(-1)
            else:
                agg = F
            F = self.alpha * agg + (1.0 - self.alpha) * F_seed
            # Clamp seeded rows to their original values.
            F[labeled_mask] = F_seed[labeled_mask]
        self._labels = F
        self._is_fitted = True
        return self

    def predict(self, graph: Any) -> torch.Tensor:
        if self._labels is None:
            raise RuntimeError("call fit() before predict()")
        return self._labels.argmax(dim=-1)

    def predict_proba(self, graph: Any) -> torch.Tensor:
        if self._labels is None:
            raise RuntimeError("call fit() before predict_proba()")
        # Row-normalise to a proper probability simplex.
        return self._labels / self._labels.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    def transform(self, graph: Any) -> torch.Tensor:
        return self.predict_proba(graph)
