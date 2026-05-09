"""Actor-Critic network for graph RL.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .policy import _gnn_layer

__all__ = ["GraphActorCriticNetwork"]


class GraphActorCriticNetwork(nn.Module):
    """Shared GNN encoder with separate policy and value heads.

    Architecture:
        Shared GNN -> global pool -> [policy_head, value_head]

    Args:
        node_in_dim: Node feature input dimension.
        edge_in_dim: Edge feature input dimension.
        hidden_dim: GNN hidden dimension.
        num_actions: Number of discrete actions.
        num_gnn_layers: Number of GNN layers.
        shared_encoder: Whether to share the GNN encoder.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int = 0,
        hidden_dim: int = 64,
        num_actions: int = 10,
        num_gnn_layers: int = 2,
        shared_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.shared_encoder = shared_encoder

        # Shared GNN encoder
        layers = []
        in_d = node_in_dim
        for _ in range(num_gnn_layers):
            layers.append(_gnn_layer(in_d, hidden_dim))
            in_d = hidden_dim
        self.shared_gnn = nn.ModuleList(layers)

        # Separate encoder for value if not shared
        if not shared_encoder:
            val_layers = []
            in_d = node_in_dim
            for _ in range(num_gnn_layers):
                val_layers.append(_gnn_layer(in_d, hidden_dim))
                in_d = hidden_dim
            self.value_gnn = nn.ModuleList(val_layers)

        # Policy head
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        nn.init.xavier_uniform_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def encode(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        gnn_layers: nn.ModuleList,
    ) -> torch.Tensor:
        """Run GNN encoder and global mean pool."""
        x = node_features
        n = x.shape[0]
        for layer in gnn_layers:
            x = layer(x, edge_index, n)
        return x.mean(dim=0, keepdim=True)  # [1, hidden_dim]

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            node_features: [N, F].
            edge_index: [2, E].
            batch: Optional LongTensor [N].

        Returns:
            (policy_logits [1, num_actions], value [1, 1])
        """
        if node_features.dim() != 2:
            raise ValueError(
                f"GraphActorCriticNetwork expects [N, F] node features but got {list(node_features.shape)}"
            )

        # Policy
        h_policy = self.encode(node_features, edge_index, self.shared_gnn)
        policy_logits = self.policy_head(h_policy)  # [1, num_actions]

        # Value
        if self.shared_encoder:
            h_value = h_policy
        else:
            h_value = self.encode(node_features, edge_index, self.value_gnn)
        value = self.value_head(h_value)  # [1, 1]

        return policy_logits, value
