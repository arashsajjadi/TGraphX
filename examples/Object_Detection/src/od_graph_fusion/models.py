"""TGraphX fusion model.

Architecture:
  1. CNN encoder over tensor crop node features [C, H, W].
  2. MLP encoder over scalar/vector metadata.
  3. Edge feature encoder (MLP).
  4. Two rounds of ConvMessagePassing on tensor features.
  5. Final scalar embedding via global pooling.
  6. Heads: objectness (BCE), class (CE), box-reg (SmoothL1).

Tensor-native semantics are preserved through the ConvMessagePassing layers;
flattening only happens at the head input.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tgraphx import ConvMessagePassing, Graph


class CropCNN(nn.Module):
    """Small CNN over [C, H, W] crops (used for both nodes and as preprocessing)."""

    def __init__(self, in_channels: int = 3, out_channels: int = 32, crop_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # crop_size/2
            nn.Conv2d(out_channels // 2, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # crop_size/4
        )
        self.out_channels = out_channels
        self.out_spatial = max(crop_size // 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LegacyDetectionFusionModel(nn.Module):
    """Legacy TGraphX fusion model (objectness/class/box heads).

    NOTE: edge_mlp is defined but edge_features are NOT used in forward.
    Do not use this model in V3/strict_source_router experiments.
    Use TGraphXSourceRouterV3 instead.

    Inputs (per graph):
        node_features:  [N, 3, crop_size, crop_size]
        edge_index:     [2, E]
        edge_attr:      [E, edge_feat_dim]
        metadata:       dict containing node_metadata [N, D_meta]
                        and node_types [N]
    Outputs (per node, restricted to cluster/consensus nodes at eval time):
        objectness_logits: [N]
        class_logits:      [N, num_classes]
        box_reg:           [N, 4]   (offset from candidate box to refined box)
    """

    def __init__(
        self,
        num_classes: int,
        num_detectors: int,
        crop_size: int,
        crop_channels: int = 32,
        hidden_dim: int = 64,
        metadata_dim: Optional[int] = None,
        edge_feat_dim: int = 14,
        num_message_passing: int = 2,
        num_node_types: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_detectors = num_detectors
        self.crop_size = crop_size
        self.num_node_types = num_node_types

        # Tensor encoder over crops
        self.crop_encoder = CropCNN(in_channels=3,
                                     out_channels=crop_channels,
                                     crop_size=crop_size)
        crop_spatial = self.crop_encoder.out_spatial

        # Encoder shape contract: [B, crop_channels, crop_spatial, crop_spatial]
        # Use ConvMessagePassing layers that preserve the spatial size.
        layers = []
        for _ in range(num_message_passing):
            layers.append(ConvMessagePassing(
                in_shape=(crop_channels, crop_spatial, crop_spatial),
                out_shape=(crop_channels, crop_spatial, crop_spatial),
            ))
        self.mp_layers = nn.ModuleList(layers)

        # Metadata + edge encoders
        md_dim = metadata_dim if metadata_dim is not None else 8 + num_detectors + num_classes
        self.metadata_mlp = nn.Sequential(
            nn.Linear(md_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Per-node global pool over [C, H, W] → vector of size crop_channels
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        # Heads (operate on hidden-dim node embeddings after fusion)
        fused_dim = crop_channels + hidden_dim
        self.fuse_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim), nn.ReLU(inplace=True),
        )
        self.objectness_head = nn.Linear(hidden_dim, 1)
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.box_head = nn.Linear(hidden_dim, 4)

    def forward(self, graph: Graph) -> Dict[str, torch.Tensor]:
        """Run a forward pass on a single graph (or batched graph)."""
        x = graph.node_features                  # [N, 3, H, W]
        ei = graph.edge_index                    # [2, E]
        md = graph.metadata.get("node_metadata") if isinstance(graph.metadata, dict) else None
        device = x.device

        # CNN encode crops
        h = self.crop_encoder(x)                 # [N, C_e, H_e, W_e]

        # Tensor-aware message passing
        for mp in self.mp_layers:
            h = F.relu(mp(h, ei)) + h

        # Spatial pool to vector per node
        v = self.spatial_pool(h).squeeze(-1).squeeze(-1)  # [N, C_e]

        # Metadata
        if md is None:
            md = torch.zeros(x.shape[0], self.metadata_mlp[0].in_features,
                              device=device, dtype=v.dtype)
        if md.device != device:
            md = md.to(device)
        m = self.metadata_mlp(md)                # [N, hidden]

        # Concatenate tensor embedding + metadata embedding
        fused = torch.cat([v, m], dim=1)
        z = self.fuse_head(fused)

        return {
            "objectness_logits": self.objectness_head(z).squeeze(-1),
            "class_logits": self.class_head(z),
            "box_reg": self.box_head(z),
            "node_embedding": z,
        }
DetectionFusionModel = LegacyDetectionFusionModel  # backward compat alias
