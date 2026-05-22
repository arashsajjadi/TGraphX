"""Reusable neural-network building blocks."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLP(nn.Module):
    """Two-layer MLP with a residual connection.

    Input and output must have the same dimension.
    """

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.fc1   = nn.Linear(dim, dim * 2)
        self.fc2   = nn.Linear(dim * 2, dim)
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.silu(self.fc1(h))
        h = self.drop(self.fc2(h))
        return x + h


class MLP(nn.Module):
    """Generic MLP with configurable depth, activations, and layer norm."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 n_layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PointEncoder(nn.Module):
    """Encodes the per-point features of a canonical board state.

    Input : [batch, 24, point_feature_dim]
    Output: [batch, state_dim]   (mean-pooled over 24 points)
    """

    def __init__(self, point_feat_dim: int, state_dim: int,
                 n_residual: int = 4) -> None:
        super().__init__()
        self.proj    = nn.Linear(point_feat_dim, state_dim)
        self.blocks  = nn.ModuleList(
            [ResidualMLP(state_dim) for _ in range(n_residual)]
        )
        self.norm    = nn.LayerNorm(state_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 24, P]
        h = self.proj(x)               # [B, 24, D]
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return h.mean(dim=1)           # [B, D]


def _init_orthogonal(module: nn.Module) -> None:
    """Apply orthogonal init to all Linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=1.41421)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
