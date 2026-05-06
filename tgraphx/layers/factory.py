"""Layer factory: create any TGraphX GNN layer by name.

Usage::

    from tgraphx.layers.factory import make_layer

    # 2-D spatial GAT
    layer = make_layer("gat", in_shape=(8, 4, 4), out_shape=(16, 4, 4), heads=2)

    # Vector linear layer
    layer = make_layer("linear", in_shape=(32,), out_shape=(64,))

Supported names
---------------
``"conv"``             ConvMessagePassing  (2-D / 3-D spatial)
``"gat"``              TensorGATLayer      (2-D / 3-D spatial)
``"sage"``             TensorGraphSAGELayer (2-D / 3-D spatial)
``"gin"``              TensorGINLayer      (2-D / 3-D spatial)
``"linear"``           LinearMessagePassing (vector only)
``"legacy_attention"`` AttentionMessagePassing (vector or 2-D spatial)

Shape convention
----------------
``(D,)``        — vector
``(C, H, W)``   — 2-D spatial
``(C, D, H, W)``— 3-D volumetric

Accepted kwargs (forwarded only where applicable)
-------------------------------------------------
aggr               str   "sum"/"mean"/"max"  (conv, sage, linear, legacy_attention)
heads              int   num_heads           (gat)
concat             bool  concat_heads        (gat, default True)
residual           bool  skip connection     (all)
dropout            float dropout rate        (gat → attn_dropout; linear/attention → dropout_prob)
use_edge_features  bool  enable edge input   (all)
edge_dim           int   edge channel count  (gat, sage, gin)
edge_features_kind str   "spatial"/"vector"  (sage, gin)
add_self_loops     bool  self-loops in fwd   (gat)
negative_slope     float leaky-ReLU slope    (gat)
normalize          bool  L2-normalise output (sage)
bias               bool  learnable bias      (gat, sage)
"""
from __future__ import annotations

import torch.nn as nn

from .attention_message import AttentionMessagePassing
from .base import LinearMessagePassing
from .conv_message import ConvMessagePassing
from .gat import TensorGATLayer
from .gin import TensorGINLayer
from .sage import TensorGraphSAGELayer

_SUPPORTED = ("conv", "gat", "sage", "gin", "linear", "legacy_attention")
_SPATIAL_ONLY = frozenset(("conv", "gat", "sage", "gin"))


def make_layer(
    name: str,
    in_shape,
    out_shape,
    **kwargs,
) -> nn.Module:
    """Create a TGraphX GNN layer by name.

    Args:
        name: Layer type (see module docstring for supported names).
        in_shape: Per-node input feature shape (tuple, no batch dim).
        out_shape: Per-node output feature shape (same rank as ``in_shape``).
        **kwargs: Layer-specific keyword arguments (see module docstring).

    Returns:
        Configured ``nn.Module``.

    Raises:
        ValueError: Unknown name, or unsupported shape/layer combination.
        NotImplementedError: ``legacy_attention`` with 3-D volumetric shape.
    """
    name = name.lower().strip()
    in_shape = tuple(in_shape)
    out_shape = tuple(out_shape)
    rank = len(in_shape)

    if rank == 1:
        spatial_rank = None
    elif rank == 3:
        spatial_rank = 2
    elif rank == 4:
        spatial_rank = 3
    else:
        raise ValueError(
            f"in_shape must have 1 (vector), 3 (2-D spatial), or 4 (3-D volumetric) "
            f"elements; got {rank} in in_shape={in_shape}."
        )

    if name not in _SUPPORTED:
        raise ValueError(
            f"Unknown layer name {name!r}. "
            f"Supported: {', '.join(repr(n) for n in _SUPPORTED)}."
        )

    if name in _SPATIAL_ONLY and spatial_rank is None:
        raise ValueError(
            f"Layer {name!r} requires a 2-D (C, H, W) or 3-D (C, D, H, W) "
            f"in_shape; got vector shape {in_shape}. "
            f"Use 'linear' or 'legacy_attention' for vector features."
        )

    if name == "linear" and spatial_rank is not None:
        raise ValueError(
            f"'linear' (LinearMessagePassing) supports only vector in_shape (D,); "
            f"got spatial shape {in_shape}. "
            f"Use 'conv', 'gat', 'sage', or 'gin' for spatial features."
        )

    if name == "legacy_attention" and spatial_rank == 3:
        raise NotImplementedError(
            f"'legacy_attention' (AttentionMessagePassing) uses nn.Conv2d internally "
            f"and does not support 3-D volumetric in_shape {in_shape}. "
            f"Use 'conv', 'gat', 'sage', or 'gin' for 3-D volumetric features."
        )

    in_ch = in_shape[0]
    out_ch = out_shape[0]

    if name == "conv":
        return ConvMessagePassing(
            in_shape=in_shape,
            out_shape=out_shape,
            aggr=kwargs.get("aggr", "sum"),
            use_edge_features=bool(kwargs.get("use_edge_features", False)),
            residual=bool(kwargs.get("residual", False)),
        )

    if name == "gat":
        return TensorGATLayer(
            in_channels=in_ch,
            out_channels=out_ch,
            num_heads=int(kwargs.get("heads", 1)),
            concat_heads=bool(kwargs.get("concat", True)),
            negative_slope=float(kwargs.get("negative_slope", 0.2)),
            attn_dropout=float(kwargs.get("dropout", 0.0)),
            residual=bool(kwargs.get("residual", False)),
            bias=bool(kwargs.get("bias", True)),
            add_self_loops=bool(kwargs.get("add_self_loops", False)),
            use_edge_features=bool(kwargs.get("use_edge_features", False)),
            edge_dim=kwargs.get("edge_dim", None),
            spatial_rank=spatial_rank,
        )

    if name == "sage":
        return TensorGraphSAGELayer(
            in_channels=in_ch,
            out_channels=out_ch,
            aggr=kwargs.get("aggr", "mean"),
            normalize=bool(kwargs.get("normalize", False)),
            bias=bool(kwargs.get("bias", True)),
            residual=bool(kwargs.get("residual", False)),
            use_edge_features=bool(kwargs.get("use_edge_features", False)),
            edge_dim=kwargs.get("edge_dim", None),
            edge_features_kind=kwargs.get("edge_features_kind", "spatial"),
            spatial_rank=spatial_rank,
        )

    if name == "gin":
        return TensorGINLayer(
            in_channels=in_ch,
            out_channels=out_ch,
            use_edge_features=bool(kwargs.get("use_edge_features", False)),
            edge_dim=kwargs.get("edge_dim", None),
            edge_features_kind=kwargs.get("edge_features_kind", "spatial"),
            spatial_rank=spatial_rank,
        )

    if name == "linear":
        return LinearMessagePassing(
            in_shape=in_shape,
            out_shape=out_shape,
            aggr=kwargs.get("aggr", "sum"),
            use_edge_features=bool(kwargs.get("use_edge_features", False)),
            dropout_prob=float(kwargs.get("dropout", 0.0)),
            residual=bool(kwargs.get("residual", False)),
        )

    if name == "legacy_attention":
        return AttentionMessagePassing(
            in_shape=in_shape,
            out_shape=out_shape,
            aggr=kwargs.get("aggr", "sum"),
            use_edge_features=bool(kwargs.get("use_edge_features", False)),
            dropout_prob=float(kwargs.get("dropout", 0.0)),
            residual=bool(kwargs.get("residual", False)),
        )

    raise AssertionError(f"Unhandled layer name {name!r}")  # pragma: no cover


__all__ = ["make_layer"]
