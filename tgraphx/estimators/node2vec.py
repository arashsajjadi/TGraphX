"""Node2Vec estimator wrapping :class:`tgraphx.mining.node2vec.Node2Vec`."""
from __future__ import annotations

from typing import Any, Optional

import torch

from .base import BaseGraphEstimator

__all__ = ["Node2VecEstimator"]


class Node2VecEstimator(BaseGraphEstimator):
    """Sklearn-like wrapper around the existing Node2Vec helper.

    Args:
        embedding_dim: Embedding size.
        walk_length: Random walk length.
        num_walks_per_node: Walks per node.
        window: Skip-gram context window.
        epochs: Training epochs.
        lr: Learning rate.
        seed: Optional RNG seed.

    Stability: Beta.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        walk_length: int = 20,
        num_walks_per_node: int = 5,
        window: int = 5,
        epochs: int = 1,
        lr: float = 0.025,
        seed: Optional[int] = None,
    ) -> None:
        self.embedding_dim = int(embedding_dim)
        self.walk_length = int(walk_length)
        self.num_walks_per_node = int(num_walks_per_node)
        self.window = int(window)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.seed = seed
        self._embeddings: Optional[torch.Tensor] = None

    def fit(self, graph: Any, y: Any = None) -> "Node2VecEstimator":
        from ..mining.node2vec import (
            Node2VecEmbedding,
            node2vec_walks,
            generate_skipgram_pairs,
            train_node2vec_step,
        )
        if self.seed is not None:
            torch.manual_seed(int(self.seed))
        N = graph.num_nodes
        model = Node2VecEmbedding(N, embedding_dim=self.embedding_dim)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            walks = node2vec_walks(
                graph.edge_index, num_nodes=N,
                walk_length=self.walk_length,
                walks_per_node=self.num_walks_per_node,
                seed=self.seed,
            )
            centers, contexts, negatives = generate_skipgram_pairs(
                walks, window_size=self.window, num_nodes=N, seed=self.seed,
            )
            if centers.numel() == 0:
                break
            train_node2vec_step(model, opt, centers, contexts, negatives)
        self._embeddings = model.get_embeddings()
        self._is_fitted = True
        return self

    def predict(self, graph: Any) -> torch.Tensor:
        return self.transform(graph)

    def transform(self, graph: Any) -> torch.Tensor:
        if self._embeddings is None:
            raise RuntimeError("call fit() before transform()")
        return self._embeddings
