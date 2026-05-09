"""Policy networks for graph RL.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "GraphPolicyNetwork",
    "MaskedCategoricalPolicy",
    "NodeActionPolicy",
    "EdgeActionPolicy",
    "GraphEditPolicy",
]

_MASK_FILL = -1e9  # Use -1e9 NOT -inf to avoid NaN in softmax


def _gnn_layer(in_dim: int, out_dim: int) -> nn.Module:
    """Simple mean-aggregation GNN layer."""
    class _MeanAgg(nn.Module):
        def __init__(self, in_d: int, out_d: int) -> None:
            super().__init__()
            self.lin = nn.Linear(in_d, out_d)
            nn.init.xavier_uniform_(self.lin.weight)
            nn.init.zeros_(self.lin.bias)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor, n: int) -> torch.Tensor:
            x = self.lin(x)
            if edge_index.numel() == 0 or n == 0:
                return F.relu(x)
            src, dst = edge_index[0], edge_index[1]
            agg = torch.zeros_like(x)
            agg.scatter_add_(0, dst.unsqueeze(1).expand_as(x[src]), x[src])
            cnt = torch.zeros(n, 1, dtype=x.dtype, device=x.device)
            cnt.scatter_add_(0, dst.unsqueeze(1), torch.ones(src.size(0), 1, dtype=x.dtype, device=x.device))
            cnt = cnt.clamp(min=1)
            return F.relu(agg / cnt)

    return _MeanAgg(in_dim, out_dim)


class GraphPolicyNetwork(nn.Module):
    """GNN encoder + global pool + policy head -> logits [B, A].

    Args:
        node_in_dim: Node feature input dimension.
        edge_in_dim: Edge feature input dimension (unused in simple version).
        hidden_dim: GNN hidden dimension.
        num_actions: Number of discrete actions.
        num_gnn_layers: Number of GNN layers.
        gnn_type: 'mean' (default).
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int = 0,
        hidden_dim: int = 64,
        num_actions: int = 10,
        num_gnn_layers: int = 2,
        gnn_type: str = "mean",
    ) -> None:
        super().__init__()
        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        layers = []
        in_d = node_in_dim
        for _ in range(num_gnn_layers):
            layers.append(_gnn_layer(in_d, hidden_dim))
            in_d = hidden_dim
        self.gnn_layers = nn.ModuleList(layers)

        self.policy_head = nn.Linear(hidden_dim, num_actions)
        nn.init.xavier_uniform_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            node_features: [N, node_in_dim] or [B*N, node_in_dim].
            edge_index: [2, E].
            batch: Optional LongTensor [N] mapping nodes to graph.

        Returns:
            FloatTensor [1, num_actions] (logits).
        """
        if node_features.dim() != 2:
            raise ValueError(
                f"GraphPolicyNetwork expects [N, F] node features but got {list(node_features.shape)}."
            )

        x = node_features
        n = x.shape[0]

        for layer in self.gnn_layers:
            x = layer(x, edge_index, n)

        # Global mean pool
        pooled = x.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        logits = self.policy_head(pooled)  # [1, num_actions]
        return logits


class MaskedCategoricalPolicy(nn.Module):
    """Categorical distribution with action masking.

    Masked logits are set to -1e9 (NOT -inf to avoid NaN in softmax).
    Entropy is computed only over valid actions.

    Args:
        logits: FloatTensor [B, A] or [A].
        action_mask: BoolTensor [B, A] or [A]. True = valid action.
    """

    def __init__(self, logits: torch.Tensor, action_mask: torch.Tensor) -> None:
        super().__init__()
        self._logits = logits
        self._mask = action_mask
        self._masked_logits = self._apply_mask(logits, action_mask)

    def _apply_mask(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = logits.clone()
        masked[~mask] = _MASK_FILL
        return masked

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample an action from the masked distribution.

        Returns:
            LongTensor (scalar or [B]).
        """
        probs = F.softmax(self._masked_logits, dim=-1)
        probs = probs.clamp(min=0)  # ensure non-negative after softmax
        if probs.dim() == 1:
            return torch.multinomial(probs, 1, generator=generator).squeeze(0)
        return torch.multinomial(probs.view(-1, probs.shape[-1]), 1, generator=generator).squeeze(-1)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Log probability of given action.

        Args:
            action: LongTensor.

        Returns:
            FloatTensor.
        """
        log_probs = F.log_softmax(self._masked_logits, dim=-1)
        if log_probs.dim() == 1:
            return log_probs[action]
        return log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)

    def entropy(self) -> torch.Tensor:
        """Entropy over valid actions only."""
        probs = F.softmax(self._masked_logits, dim=-1)
        valid_probs = probs * self._mask.float()
        total = valid_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        valid_probs = valid_probs / total
        log_probs = torch.log(valid_probs.clamp(min=1e-9))
        return -(valid_probs * log_probs).sum(dim=-1)


class NodeActionPolicy(nn.Module):
    """Per-node action policy (e.g., choose which node to act on).

    Args:
        node_in_dim: Input node feature dim.
        hidden_dim: Hidden layer dim.
        num_gnn_layers: GNN layers.
    """

    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 64,
        num_gnn_layers: int = 2,
    ) -> None:
        super().__init__()
        layers = []
        in_d = node_in_dim
        for _ in range(num_gnn_layers):
            layers.append(_gnn_layer(in_d, hidden_dim))
            in_d = hidden_dim
        self.gnn_layers = nn.ModuleList(layers)
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        valid_node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            node_features: [N, F].
            edge_index: [2, E].
            valid_node_mask: Optional BoolTensor [N].

        Returns:
            FloatTensor [N] — per-node logits.
        """
        x = node_features
        n = x.shape[0]
        for layer in self.gnn_layers:
            x = layer(x, edge_index, n)
        logits = self.score_head(x).squeeze(-1)  # [N]
        if valid_node_mask is not None:
            logits = logits.masked_fill(~valid_node_mask, _MASK_FILL)
        return logits


class EdgeActionPolicy(nn.Module):
    """Per-edge action policy.

    Args:
        node_in_dim: Input node feature dim.
        edge_in_dim: Input edge feature dim.
        hidden_dim: Hidden layer dim.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_in_dim, hidden_dim) if edge_in_dim > 0 else None
        self.score_head = nn.Linear(hidden_dim * 2 + (hidden_dim if edge_in_dim > 0 else 0), 1)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        valid_edge_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            node_features: [N, F].
            edge_index: [2, E].
            edge_features: Optional [E, Fe].
            valid_edge_mask: Optional BoolTensor [E].

        Returns:
            FloatTensor [E] — per-edge logits.
        """
        node_emb = F.relu(self.node_proj(node_features))  # [N, H]
        src_emb = node_emb[edge_index[0]]  # [E, H]
        dst_emb = node_emb[edge_index[1]]  # [E, H]

        parts = [src_emb, dst_emb]
        if edge_features is not None and self.edge_proj is not None:
            edge_emb = F.relu(self.edge_proj(edge_features))  # [E, H]
            parts.append(edge_emb)

        combined = torch.cat(parts, dim=-1)  # [E, H*2 or H*3]
        logits = self.score_head(combined).squeeze(-1)  # [E]

        if valid_edge_mask is not None:
            logits = logits.masked_fill(~valid_edge_mask, _MASK_FILL)
        return logits


class GraphEditPolicy(nn.Module):
    """Combined node + edge + stop policy for graph editing.

    Args:
        node_in_dim: Node feature input dim.
        edge_in_dim: Edge feature input dim.
        hidden_dim: Hidden layer dim.
        action_space_size: Total number of discrete actions.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int,
        action_space_size: int,
    ) -> None:
        super().__init__()
        self.node_policy = NodeActionPolicy(node_in_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_space_size)
        # Shared GNN encoder
        self.gnn = _gnn_layer(node_in_dim, hidden_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            node_features: [N, F].
            edge_index: [2, E].
            action_mask: Optional BoolTensor [action_space_size].

        Returns:
            FloatTensor [1, action_space_size] — logits.
        """
        n = node_features.shape[0]
        h = self.gnn(node_features, edge_index, n)
        pooled = h.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        logits = self.policy_head(pooled)  # [1, action_space_size]

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.unsqueeze(0), _MASK_FILL)

        return logits
