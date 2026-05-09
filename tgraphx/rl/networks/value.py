"""Value network for graph RL.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .policy import _gnn_layer

__all__ = ["GraphValueNetwork"]


class GraphValueNetwork(nn.Module):
    """GNN encoder -> global pool -> value head -> scalar V(s).

    Args:
        node_in_dim: Node feature dimension.
        edge_in_dim: Edge feature dimension (unused in simple version).
        hidden_dim: Hidden layer dimension.
        num_gnn_layers: Number of GNN layers.
        gnn_type: 'mean' (default).
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int = 0,
        hidden_dim: int = 64,
        num_gnn_layers: int = 2,
        gnn_type: str = "mean",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        layers = []
        in_d = node_in_dim
        for _ in range(num_gnn_layers):
            layers.append(_gnn_layer(in_d, hidden_dim))
            in_d = hidden_dim
        self.gnn_layers = nn.ModuleList(layers)

        self.value_head = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            node_features: [N, node_in_dim].
            edge_index: [2, E].
            batch: Optional LongTensor [N].

        Returns:
            FloatTensor [1, 1] — value scalar.
        """
        if node_features.dim() != 2:
            raise ValueError(
                f"GraphValueNetwork expects [N, F] node features but got {list(node_features.shape)}"
            )
        x = node_features
        n = x.shape[0]

        for layer in self.gnn_layers:
            x = layer(x, edge_index, n)

        pooled = x.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        return self.value_head(pooled)  # [1, 1]
