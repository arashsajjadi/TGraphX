"""VGAE estimator wrapping :class:`tgraphx.mining.vgae.VGAE`."""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .base import BaseGraphEstimator

__all__ = ["VGAEEstimator"]


class VGAEEstimator(BaseGraphEstimator):
    """Sklearn-like wrapper around VGAE for unsupervised link prediction.

    Args:
        hidden_dim: Encoder hidden dim.
        out_dim: Embedding dim.
        epochs: Training epochs.
        lr: Adam learning rate.
        beta: KL weight (0 → pure GAE).
        seed: Optional seed.

    Stability: Beta.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        out_dim: int = 16,
        epochs: int = 50,
        lr: float = 0.01,
        beta: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.beta = float(beta)
        self.seed = seed
        self._model: Optional[nn.Module] = None
        self._embeddings: Optional[torch.Tensor] = None

    def fit(
        self,
        graph: Any,
        y: Any = None,
        *,
        pos_edge_index: Optional[torch.Tensor] = None,
        neg_edge_index: Optional[torch.Tensor] = None,
    ) -> "VGAEEstimator":
        from ..mining.vgae import GCNEncoder, VGAE, train_gae_step
        if self.seed is not None:
            torch.manual_seed(int(self.seed))
        N = graph.num_nodes
        feat = graph.node_features
        in_dim = feat.size(-1) if feat.dim() == 2 else feat.numel() // N
        x = feat.view(N, -1).float()
        encoder = GCNEncoder(in_dim, self.hidden_dim, self.out_dim)
        model = VGAE(encoder)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        ei = graph.edge_index
        if pos_edge_index is None:
            pos_edge_index = ei
        if neg_edge_index is None:
            from ..sampling_negative import negative_sampling
            neg_edge_index = negative_sampling(
                edge_index=ei, num_nodes=N,
                num_neg_samples=pos_edge_index.size(1),
                seed=self.seed,
            )
        for _ in range(self.epochs):
            train_gae_step(
                model, opt, x, ei, pos_edge_index, neg_edge_index, N, self.beta,
            )
        model.eval()
        with torch.no_grad():
            z = model.encode(x, ei, N)
        self._model = model
        self._embeddings = z.detach()
        self._is_fitted = True
        return self

    def predict(self, graph: Any) -> torch.Tensor:
        return self.transform(graph)

    def transform(self, graph: Any) -> torch.Tensor:
        if self._embeddings is None:
            raise RuntimeError("call fit() before transform()")
        return self._embeddings

    def score(self, graph: Any, y: Any) -> float:
        """Reconstruction AUROC on (pos, neg) edges packed into ``y``.

        ``y`` should be a dict ``{"pos_edge_index": ..., "neg_edge_index": ...}``.
        """
        if self._model is None:
            raise RuntimeError("call fit() before score()")
        if not isinstance(y, dict) or "pos_edge_index" not in y or "neg_edge_index" not in y:
            raise ValueError("y must be {'pos_edge_index': ..., 'neg_edge_index': ...}")
        from ..mining.vgae import evaluate_link_prediction
        feat = graph.node_features
        N = graph.num_nodes
        x = feat.view(N, -1).float()
        result = evaluate_link_prediction(
            self._model, x, graph.edge_index,
            y["pos_edge_index"], y["neg_edge_index"], N,
        )
        return float(result["auroc"])
