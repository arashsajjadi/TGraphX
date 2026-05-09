"""Q-network for graph RL.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .policy import _gnn_layer

__all__ = ["GraphQNetwork", "GraphDuelingQNetwork"]


class GraphQNetwork(nn.Module):
    """GNN encoder -> Q-values [1, A].

    Args:
        node_in_dim: Node feature dim.
        edge_in_dim: Edge feature dim (unused in simple version).
        hidden_dim: Hidden layer dim.
        num_actions: Action space size.
        num_gnn_layers: Number of GNN layers.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int = 0,
        hidden_dim: int = 64,
        num_actions: int = 10,
        num_gnn_layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        layers = []
        in_d = node_in_dim
        for _ in range(num_gnn_layers):
            layers.append(_gnn_layer(in_d, hidden_dim))
            in_d = hidden_dim
        self.gnn_layers = nn.ModuleList(layers)

        self.q_head = nn.Linear(hidden_dim, num_actions)
        nn.init.xavier_uniform_(self.q_head.weight)
        nn.init.zeros_(self.q_head.bias)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            node_features: [N, F].
            edge_index: [2, E].

        Returns:
            FloatTensor [1, num_actions].
        """
        if node_features.dim() != 2:
            raise ValueError(
                f"GraphQNetwork expects [N, F] node features but got {list(node_features.shape)}"
            )
        x = node_features
        n = x.shape[0]

        for layer in self.gnn_layers:
            x = layer(x, edge_index, n)

        pooled = x.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        return self.q_head(pooled)  # [1, num_actions]


class GraphDuelingQNetwork(nn.Module):
    """Dueling Q-network: Q(s,a) = V(s) + A(s,a) - mean_a A(s,a).

    Decomposes Q-values into state value V(s) and advantage A(s,a).

    Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))

    Args:
        node_in_dim: Node feature dim.
        edge_in_dim: Edge feature dim.
        hidden_dim: Hidden layer dim.
        num_actions: Action space size.
        num_gnn_layers: Number of GNN layers.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int = 0,
        hidden_dim: int = 64,
        num_actions: int = 10,
        num_gnn_layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        layers = []
        in_d = node_in_dim
        for _ in range(num_gnn_layers):
            layers.append(_gnn_layer(in_d, hidden_dim))
            in_d = hidden_dim
        self.gnn_layers = nn.ModuleList(layers)

        # Separate heads for value and advantage
        self.value_head = nn.Linear(hidden_dim, 1)
        self.advantage_head = nn.Linear(hidden_dim, num_actions)

        nn.init.xavier_uniform_(self.value_head.weight)
        nn.init.xavier_uniform_(self.advantage_head.weight)
        nn.init.zeros_(self.value_head.bias)
        nn.init.zeros_(self.advantage_head.bias)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)

        Args:
            node_features: [N, F].
            edge_index: [2, E].

        Returns:
            FloatTensor [1, num_actions].
        """
        if node_features.dim() != 2:
            raise ValueError(
                f"GraphDuelingQNetwork expects [N, F] node features but got {list(node_features.shape)}"
            )
        x = node_features
        n = x.shape[0]

        for layer in self.gnn_layers:
            x = layer(x, edge_index, n)

        pooled = x.mean(dim=0, keepdim=True)  # [1, hidden_dim]

        V = self.value_head(pooled)  # [1, 1]
        A = self.advantage_head(pooled)  # [1, num_actions]

        # Q = V + A - mean(A)
        Q = V + A - A.mean(dim=-1, keepdim=True)
        return Q
