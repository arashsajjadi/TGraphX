"""Model factories and helpers for TGraphX easy mode."""
from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

from ._exceptions import TGraphXShapeError, TGraphXUnknownNameError
from ._discovery import _MODELS


def make_tensor_node_classifier(
    in_shape: Tuple[int, ...],
    num_classes: int,
    hidden_channels: int = 16,
) -> nn.Module:
    """Create a simple tensor-aware node classifier.

    The model uses two :class:`~tgraphx.ConvMessagePassing` layers followed
    by global average pooling over spatial dimensions and a linear head.

    Args:
        in_shape: Per-node feature shape ``(C, H, W)`` or ``(C, D, H, W)``.
        num_classes: Number of output classes.
        hidden_channels: Hidden channel count for intermediate layers.

    Returns:
        ``nn.Module`` with ``forward(x, edge_index) -> logits``.
    """
    from tgraphx import ConvMessagePassing

    if len(in_shape) < 2:
        raise TGraphXShapeError(
            f"in_shape must be at least (C, H) for ConvMessagePassing, "
            f"got {in_shape}.  For vector features, use make_vector_node_classifier."
        )

    class _TensorNodeClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = ConvMessagePassing(
                in_shape, (hidden_channels, *in_shape[1:]), dropout_prob=0.0
            )
            self.conv2 = ConvMessagePassing(
                (hidden_channels, *in_shape[1:]),
                (hidden_channels, *in_shape[1:]),
                dropout_prob=0.0,
            )
            spatial_dims = (1,) * (len(in_shape) - 1)
            if len(in_shape) == 3:
                self.pool = nn.AdaptiveAvgPool2d(spatial_dims)
            else:
                self.pool = nn.AdaptiveAvgPool3d(spatial_dims)
            self.head = nn.Linear(hidden_channels, num_classes)

        def forward(self, x, edge_index):
            z = self.conv1(x, edge_index).relu()
            z = self.conv2(z, edge_index).relu()
            z = self.pool(z).flatten(1)
            return self.head(z)

    return _TensorNodeClassifier()


def make_vector_node_classifier(
    in_features: int,
    num_classes: int,
    hidden_channels: int = 64,
) -> nn.Module:
    """Create a vector-feature node classifier using GCNConv.

    Args:
        in_features: Number of input features per node.
        num_classes: Number of output classes.
        hidden_channels: Hidden dimension.

    Returns:
        ``nn.Module`` with ``forward(x, edge_index) -> logits``.
    """
    from tgraphx import GCNConv

    class _VectorNodeClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(in_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, num_classes)

        def forward(self, x, edge_index):
            z = self.conv1(x, edge_index).relu()
            return self.conv2(z, edge_index)

    return _VectorNodeClassifier()


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_model_name(model: Optional[str], node_shape: Tuple[int, ...]) -> str:
    if model is not None and model != "auto":
        return model
    if len(node_shape) >= 2:
        return "tensor_gcn"
    return "vector_gcn"


def _build_model(
    model_name: str,
    node_shape: Tuple[int, ...],
    num_classes: int,
    hidden_channels: int,
) -> nn.Module:
    if model_name in ("tensor_gcn", "tensor_sage"):
        return make_tensor_node_classifier(
            in_shape=node_shape,
            num_classes=num_classes,
            hidden_channels=hidden_channels,
        )
    elif model_name in ("vector_gcn", "linear"):
        in_features = node_shape[0] if node_shape else 1
        return make_vector_node_classifier(
            in_features=in_features,
            num_classes=num_classes,
            hidden_channels=hidden_channels,
        )
    else:
        available = list(_MODELS.get("node_classification", {}).keys())
        raise TGraphXUnknownNameError(
            f"Unknown model '{model_name}' for node classification. "
            f"Available: {available}.\n"
            f"Use list_models('node_classification') for descriptions."
        )
