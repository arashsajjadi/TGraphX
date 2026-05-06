"""EdgePredictor: MLP-based edge scorer for link/edge prediction tasks.

Given node embeddings and an edge_index, concatenates the source and
destination embeddings for each edge and passes them through a two-layer
MLP to produce per-edge logits or scores.

Spatial node features (dim > 2) are globally average-pooled to
``[N, in_dim]`` vectors before edge scoring.  This keeps the module
decoupled from spatial resolution.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EdgePredictor(nn.Module):
    """Score each edge by concatenating its endpoint node embeddings.

    Args:
        in_dim: Node embedding dimension (after any spatial pooling).
        hidden_dim: MLP hidden layer size (default 64).
        out_dim: Output dimension per edge (default 1, i.e. a scalar logit).

    Forward signature::

        predictor(node_features, edge_index) -> Tensor [E, out_dim]

    where ``node_features`` can be:

    * ``[N, in_dim]`` — vector embeddings (used directly).
    * ``[N, in_dim, H, W]`` or ``[N, in_dim, D, H, W]`` — spatial tensors
      that are globally average-pooled to ``[N, in_dim]`` before scoring.

    Example::

        from tgraphx.models.edge_predictor import EdgePredictor

        predictor = EdgePredictor(in_dim=32, hidden_dim=64, out_dim=1)
        scores = predictor(node_emb, edge_index)  # [E, 1]
        probs  = torch.sigmoid(scores.squeeze(-1)) # [E]
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        if in_dim < 1:
            raise ValueError(f"in_dim must be >= 1; got {in_dim}")
        if out_dim < 1:
            raise ValueError(f"out_dim must be >= 1; got {out_dim}")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """Global average pool over spatial dims if x has rank > 2."""
        if x.dim() <= 2:
            return x
        return x.mean(dim=list(range(2, x.dim())))  # [N, C]

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Score edges.

        Args:
            node_features: ``[N, in_dim]`` or ``[N, in_dim, *spatial]``.
            edge_index: ``[2, E]`` LongTensor.

        Returns:
            ``[E, out_dim]`` score tensor.
        """
        x = self._pool(node_features)  # [N, in_dim]
        if x.size(1) != self.in_dim:
            raise ValueError(
                f"EdgePredictor expected in_dim={self.in_dim} channels after "
                f"spatial pooling; got {x.size(1)} from input shape "
                f"{tuple(node_features.shape)}."
            )
        src = edge_index[0]
        dst = edge_index[1]
        edge_feat = torch.cat([x[src], x[dst]], dim=1)  # [E, 2*in_dim]
        return self.mlp(edge_feat)

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}"


__all__ = ["EdgePredictor"]
