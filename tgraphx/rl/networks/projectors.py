"""State/action feature projectors for graph RL networks.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tgraphx.generation.projectors import ImageNodeEncoder

__all__ = ["StateFeatureProjector", "ActionFeatureProjector"]


class StateFeatureProjector(nn.Module):
    """Project graph state features to a fixed-dim vector.

    Handles:
        - Vector node features [N, F] via a linear layer
        - Image node features [N, C, H, W] via ImageNodeEncoder
        - Edge features [E, Fe] via a linear layer
        - Graph-level features [F_g] via a linear layer

    All projected features are summed to out_dim.

    Args:
        node_in_dim: Input node feature dim (for vector features).
        edge_in_dim: Input edge feature dim. 0 means no edge features.
        graph_in_dim: Input graph feature dim. 0 means no graph features.
        out_dim: Output projection dimension.
        image_encoder: Optional ImageNodeEncoder for image node features.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        graph_in_dim: int,
        out_dim: int,
        image_encoder: Optional[ImageNodeEncoder] = None,
    ) -> None:
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.graph_in_dim = graph_in_dim
        self.out_dim = out_dim
        self.image_encoder = image_encoder

        self.node_proj = nn.Linear(node_in_dim, out_dim)
        nn.init.xavier_uniform_(self.node_proj.weight)

        self.edge_proj: Optional[nn.Linear] = None
        if edge_in_dim > 0:
            self.edge_proj = nn.Linear(edge_in_dim, out_dim)
            nn.init.xavier_uniform_(self.edge_proj.weight)

        self.graph_proj: Optional[nn.Linear] = None
        if graph_in_dim > 0:
            self.graph_proj = nn.Linear(graph_in_dim, out_dim)
            nn.init.xavier_uniform_(self.graph_proj.weight)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        graph_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            node_features: [N, F] or [N, C, H, W].
            edge_features: Optional [E, Fe].
            graph_features: Optional [F_g] or [B, F_g].

        Returns:
            FloatTensor [N, out_dim] (node-level embeddings).
        """
        if node_features.dim() == 4:
            # Image features — use encoder
            if self.image_encoder is None:
                raise ValueError(
                    "StateFeatureProjector received [N, C, H, W] node features "
                    "but no image_encoder was provided. Pass an ImageNodeEncoder."
                )
            node_proj = self.image_encoder(node_features)
            # Project to out_dim if needed
            if node_proj.shape[-1] != self.out_dim:
                node_proj = F.relu(self.node_proj(node_proj)) if self.image_encoder.out_dim == self.node_in_dim else node_proj
        elif node_features.dim() == 2:
            node_proj = F.relu(self.node_proj(node_features))
        else:
            raise ValueError(
                f"StateFeatureProjector expects [N, F] or [N, C, H, W] but got {list(node_features.shape)}"
            )

        result = node_proj

        # Fuse edge mean into node embeddings (optional)
        if edge_features is not None and self.edge_proj is not None:
            edge_emb = F.relu(self.edge_proj(edge_features))  # [E, out_dim]
            # Mean-pool edges (broadcast to node level)
            edge_mean = edge_emb.mean(dim=0, keepdim=True).expand(result.shape[0], -1)
            result = result + edge_mean

        # Fuse graph features (global bias)
        if graph_features is not None and self.graph_proj is not None:
            if graph_features.dim() == 1:
                gf = graph_features.unsqueeze(0)
            else:
                gf = graph_features
            graph_emb = F.relu(self.graph_proj(gf))  # [1, out_dim] or [B, out_dim]
            result = result + graph_emb.mean(dim=0, keepdim=True).expand(result.shape[0], -1)

        return result


class ActionFeatureProjector(nn.Module):
    """Project action features to a fixed embedding.

    Args:
        action_dim: Input action feature dimension.
        out_dim: Output dimension.
    """

    def __init__(self, action_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(action_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(x))
