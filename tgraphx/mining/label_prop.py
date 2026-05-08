"""Semi-supervised graph learning via label propagation.

Label propagation is a simple, effective baseline for semi-supervised
node classification on graphs.  It diffuses known labels through the
graph structure without any trainable parameters.

Stability: Beta (v0.4.2+).
"""
from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn.functional as F

__all__ = [
    "label_propagation",
    "LabelPropagationClassifier",
]


def label_propagation(
    edge_index: torch.Tensor,
    num_nodes: int,
    y: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
    alpha: float = 0.9,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> torch.Tensor:
    """Iterative label propagation for semi-supervised node classification.

    Iterates:
        Z_new ← alpha * D^{-1} A Z + (1-alpha) Y_0

    where ``Y_0`` is the one-hot initial label matrix (masked by ``mask``),
    ``A`` is the adjacency matrix (treated as undirected), and
    ``D`` is the degree matrix.

    Args:
        edge_index: ``LongTensor[2, E]`` (treated as undirected).
        num_nodes: Node count.
        y: ``LongTensor[N]`` class labels.  Unlabelled nodes can have any value.
        mask: ``BoolTensor[N]`` — ``True`` for labelled/seed nodes.
        num_classes: Number of classes.
        alpha: Propagation coefficient (0 = no propagation, 1 = full).
        max_iter: Maximum iterations.
        tol: Convergence tolerance (L1 norm of change per element).

    Returns:
        ``FloatTensor[N, num_classes]`` soft class probabilities.
        Argmax gives predicted class for every node.
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    N = num_nodes
    device = edge_index.device
    C = int(num_classes)

    # Build undirected symmetric edge_index.
    if edge_index.numel():
        ei_sym = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    else:
        ei_sym = edge_index

    # Out-degree for D^{-1} normalisation.
    deg = torch.zeros(N, dtype=torch.float, device=device)
    if ei_sym.numel():
        ones = torch.ones(ei_sym.size(1), dtype=torch.float, device=device)
        deg.scatter_add_(0, ei_sym[0], ones)
    deg_inv = 1.0 / deg.clamp(min=1e-12)

    # Initial label matrix Y_0 (labelled nodes only).
    Y0 = torch.zeros(N, C, dtype=torch.float, device=device)
    labeled_idx = mask.nonzero(as_tuple=False).view(-1)
    if labeled_idx.numel():
        Y0[labeled_idx] = F.one_hot(
            y[labeled_idx].to(torch.long), num_classes=C,
        ).float()

    Z = Y0.clone()
    src, dst = (ei_sym[0], ei_sym[1]) if ei_sym.numel() else (
        torch.zeros(0, dtype=torch.long, device=device),
        torch.zeros(0, dtype=torch.long, device=device),
    )

    for _ in range(max_iter):
        # Diffusion: Z_new[v] = alpha * Σ_{u→v} Z[u] / deg[u] + (1-alpha) * Y0
        agg = torch.zeros(N, C, dtype=torch.float, device=device)
        if src.numel():
            weighted_Z = Z[src] * deg_inv[src].unsqueeze(1)
            agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, C), weighted_Z)
        Z_new = alpha * agg + (1.0 - alpha) * Y0
        diff = (Z_new - Z).abs().mean().item()
        Z = Z_new
        if diff < tol:
            break

    return Z


class LabelPropagationClassifier:
    """Scikit-learn-style wrapper for label propagation.

    Provides ``fit`` / ``predict`` / ``predict_proba`` methods that
    operate on a fixed graph structure.

    Args:
        alpha: Propagation coefficient.  0 = ignore graph, 1 = propagate fully.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Example::

        clf = LabelPropagationClassifier(alpha=0.9)
        proba = clf.fit_predict(
            edge_index, num_nodes, y, train_mask, num_classes=C,
        )
        accuracy = (proba.argmax(1)[test_mask] == y[test_mask]).float().mean()

    Stability: Beta.
    """

    def __init__(
        self,
        alpha: float = 0.9,
        max_iter: int = 50,
        tol: float = 1e-4,
    ) -> None:
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self._Z: Optional[torch.Tensor] = None

    def fit_predict(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        y: torch.Tensor,
        train_mask: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """Propagate labels and return soft predictions for all nodes.

        Args:
            edge_index: ``LongTensor[2, E]``.
            num_nodes: Node count.
            y: ``LongTensor[N]`` with labels for ``train_mask`` nodes.
            train_mask: ``BoolTensor[N]`` — labeled nodes.
            num_classes: Class count.

        Returns:
            ``FloatTensor[N, num_classes]`` soft probabilities.
        """
        self._Z = label_propagation(
            edge_index, num_nodes, y, train_mask, num_classes,
            self.alpha, self.max_iter, self.tol,
        )
        return self._Z

    def predict(
        self,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return argmax predictions.

        Args:
            mask: Optional ``BoolTensor[N]`` to select a subset.

        Returns:
            ``LongTensor`` of predicted classes.
        """
        if self._Z is None:
            raise RuntimeError("Call fit_predict() first.")
        proba = self._Z[mask] if mask is not None else self._Z
        return proba.argmax(dim=1)

    def predict_proba(
        self,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return soft probability predictions.

        Args:
            mask: Optional ``BoolTensor[N]``.

        Returns:
            ``FloatTensor[N or mask.sum(), num_classes]``.
        """
        if self._Z is None:
            raise RuntimeError("Call fit_predict() first.")
        return self._Z[mask] if mask is not None else self._Z
