"""Feature projectors for multi-modal node/edge features.

These modules project various tensor layouts to a common embedding dimension.

IMPORTANT: If a [N, C, H, W] tensor arrives at a linear projector, it raises a
clear error rather than silently reshaping.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "VectorNodeProjector",
    "ImageNodeEncoder",
    "VolumeNodeEncoder",
    "EdgeFeatureProjector",
    "GraphFeatureProjector",
    "TensorFeatureFusion",
]


class VectorNodeProjector(nn.Module):
    """Project vector node features [N, in_dim] → [N, out_dim].

    Raises:
        ValueError: If input has more than 2 dimensions (e.g. [N, C, H, W]).
            Use ImageNodeEncoder for image features instead.

    Args:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
        activation: 'relu' or 'gelu' (default 'relu').
    """

    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu") -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: FloatTensor [N, in_dim].

        Returns:
            FloatTensor [N, out_dim].
        """
        if x.dim() != 2:
            raise ValueError(
                f"VectorNodeProjector expects [N, F] input but got shape {list(x.shape)}. "
                f"If you have image features [N, C, H, W], use ImageNodeEncoder instead."
            )
        if x.shape[1] != self.in_dim:
            raise ValueError(
                f"VectorNodeProjector expects in_dim={self.in_dim} but got {x.shape[1]}"
            )
        out = self.linear(x)
        if self.activation == "relu":
            return F.relu(out)
        elif self.activation == "gelu":
            return F.gelu(out)
        return out


class ImageNodeEncoder(nn.Module):
    """Encode image-format node features [N, C, H, W] → [N, out_dim].

    Uses a small CNN (2 conv layers + adaptive average pool) to project
    image node features to a fixed-size embedding.

    Args:
        in_channels: Input image channels C.
        out_dim: Output embedding dimension.
        spatial_size: Expected (H, W). Used only for shape validation in __init__.
            Set to None to skip spatial validation.
    """

    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        spatial_size: Optional[tuple] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.spatial_size = spatial_size
        hidden_ch = max(16, out_dim // 2)
        self.conv1 = nn.Conv2d(in_channels, hidden_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_ch, out_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: FloatTensor [N, C, H, W].

        Returns:
            FloatTensor [N, out_dim].
        """
        if x.dim() != 4:
            raise ValueError(
                f"ImageNodeEncoder expects [N, C, H, W] input but got shape {list(x.shape)}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"ImageNodeEncoder expects {self.in_channels} channels but got {x.shape[1]}"
            )
        if self.spatial_size is not None:
            if x.shape[2] != self.spatial_size[0] or x.shape[3] != self.spatial_size[1]:
                raise ValueError(
                    f"ImageNodeEncoder expects spatial_size={self.spatial_size} "
                    f"but got ({x.shape[2]}, {x.shape[3]})"
                )
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.pool(out)  # [N, out_dim, 1, 1]
        return out.view(x.shape[0], -1)  # [N, out_dim]


class VolumeNodeEncoder(nn.Module):
    """Encode volumetric node features [N, C, D, H, W] → [N, out_dim].

    Uses a small 3D-CNN to project volumetric node features.

    Args:
        in_channels: Input volume channels C.
        out_dim: Output embedding dimension.
    """

    def __init__(self, in_channels: int, out_dim: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        hidden_ch = max(8, out_dim // 2)
        self.conv1 = nn.Conv3d(in_channels, hidden_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(hidden_ch, out_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: FloatTensor [N, C, D, H, W].

        Returns:
            FloatTensor [N, out_dim].
        """
        if x.dim() != 5:
            raise ValueError(
                f"VolumeNodeEncoder expects [N, C, D, H, W] input but got shape {list(x.shape)}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"VolumeNodeEncoder expects {self.in_channels} channels but got {x.shape[1]}"
            )
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.pool(out)  # [N, out_dim, 1, 1, 1]
        return out.view(x.shape[0], -1)  # [N, out_dim]


class EdgeFeatureProjector(nn.Module):
    """Project edge features [E, in_dim] → [E, out_dim].

    Args:
        in_dim: Input edge feature dimension.
        out_dim: Output dimension.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: FloatTensor [E, in_dim].

        Returns:
            FloatTensor [E, out_dim].
        """
        if x.dim() != 2:
            raise ValueError(
                f"EdgeFeatureProjector expects [E, F] input but got {list(x.shape)}"
            )
        if x.shape[1] != self.in_dim:
            raise ValueError(
                f"EdgeFeatureProjector expects in_dim={self.in_dim} but got {x.shape[1]}"
            )
        return F.relu(self.linear(x))


class GraphFeatureProjector(nn.Module):
    """Project graph-level features [F] or [B, F] → projected.

    Args:
        in_dim: Input feature dimension.
        out_dim: Output dimension.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: FloatTensor [F] or [B, F].

        Returns:
            FloatTensor [out_dim] or [B, out_dim].
        """
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        if x.dim() != 2:
            raise ValueError(
                f"GraphFeatureProjector expects [F] or [B, F] input but got {list(x.shape)}"
            )
        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"GraphFeatureProjector expects in_dim={self.in_dim} but got {x.shape[-1]}"
            )
        out = F.relu(self.linear(x))
        return out.squeeze(0) if squeeze else out


class TensorFeatureFusion(nn.Module):
    """Fuse multiple modality projectors into a single embedding.

    Modes:
        - ``concat_project``: Concatenate projector outputs, then apply a linear layer.
        - ``add``: Element-wise sum (all projectors must have same out_dim).
        - ``gated``: Weighted sum using learned sigmoid gates.

    Args:
        projectors: List of nn.Modules each mapping their input → [N/B, proj_dim].
        out_dim: Final output dimension.
        mode: Fusion mode ('concat_project', 'add', 'gated').
    """

    def __init__(
        self,
        projectors: List[nn.Module],
        out_dim: int,
        mode: str = "concat_project",
    ) -> None:
        super().__init__()
        if mode not in ("concat_project", "add", "gated"):
            raise ValueError(
                f"mode={mode!r} not in ('concat_project', 'add', 'gated')"
            )
        self.projectors = nn.ModuleList(projectors)
        self.out_dim = out_dim
        self.mode = mode

        if mode == "concat_project":
            # Will be built lazily since we don't know each projector's out_dim at init
            # Use a fixed heuristic: assume each projector maps to out_dim
            total_in = out_dim * len(projectors)
            self.fusion_linear: Optional[nn.Linear] = nn.Linear(total_in, out_dim)
            self.gate_weights: Optional[nn.Parameter] = None
        elif mode == "gated":
            self.gate_weights = nn.Parameter(torch.ones(len(projectors)))
            self.fusion_linear = None
        else:
            self.gate_weights = None
            self.fusion_linear = None

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs: List of tensors, one per projector. Each projector handles
                    shape validation internally.

        Returns:
            FloatTensor [N, out_dim] or [B, out_dim].
        """
        if len(inputs) != len(self.projectors):
            raise ValueError(
                f"TensorFeatureFusion expects {len(self.projectors)} inputs "
                f"but got {len(inputs)}"
            )

        projected = [proj(x) for proj, x in zip(self.projectors, inputs)]

        if self.mode == "concat_project":
            cat = torch.cat(projected, dim=-1)
            if self.fusion_linear is not None and cat.shape[-1] == self.fusion_linear.in_features:
                return F.relu(self.fusion_linear(cat))
            else:
                # Rebuild linear lazily if shapes differ
                in_dim = cat.shape[-1]
                self.fusion_linear = nn.Linear(in_dim, self.out_dim).to(cat.device)
                return F.relu(self.fusion_linear(cat))

        elif self.mode == "add":
            result = projected[0]
            for p in projected[1:]:
                result = result + p
            return result

        elif self.mode == "gated":
            gates = torch.sigmoid(self.gate_weights)
            result = sum(g * p for g, p in zip(gates, projected))
            return result

        raise ValueError(f"Unknown mode: {self.mode}")
