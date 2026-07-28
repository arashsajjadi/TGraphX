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
dropout            float dropout rate        (conv → aggregator dropout_prob;
                                              gat → attn_dropout;
                                              linear/attention → dropout_prob)
use_edge_features  bool  enable edge input   (all)
edge_dim           int   edge channel count  (gat, sage, gin)
edge_features_kind str   "spatial"/"vector"  (sage, gin)
add_self_loops     bool  self-loops in fwd   (gat)
negative_slope     float leaky-ReLU slope    (gat)
normalize          bool  L2-normalise output (sage)
bias               bool  learnable bias      (gat, sage)
use_batchnorm      bool  BatchNorm after agg (conv aggregator, gin, linear)
aggregator_params  dict  DeepCNNAggregator kwargs (conv)
eps                float GIN epsilon         (gin)
train_eps          bool  learn epsilon       (gin)
hidden_channels    int   GIN MLP hidden dim  (gin; defaults to out_channels)

Since v1.5.0, ``"conv"`` layers built without ``dropout`` (or an explicit
``aggregator_params["dropout_prob"]``) default to dropout 0.0 and emit
``tgraphx.DropoutDefaultChangeWarning`` — TGraphX <= 1.4.2 silently used
0.3 inside the conv aggregator and ignored the ``dropout`` kwarg.
"""
from __future__ import annotations

import torch.nn as nn

from .attention_message import AttentionMessagePassing
from .base import LinearMessagePassing
from .conv_message import ConvMessagePassing
from .gat import TensorGATLayer
from .gin import TensorGINLayer
from .graph_transformer import GraphTransformerLayer
from .sage import TensorGraphSAGELayer

_SUPPORTED = (
    "conv", "gat", "sage", "gin", "linear", "legacy_attention",
    "graph_transformer",
)
_SPATIAL_ONLY = frozenset(("conv", "gat", "sage", "gin"))
_VECTOR_ONLY = frozenset(("graph_transformer",))


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

    if name in ("set_transformer", "set_attention", "tgraphx_set_attention"):
        raise ValueError(
            f"{name!r} is a model-level family, not a per-layer "
            "operator: use tgraphx.build_model(task=..., "
            "layer='set_transformer', ...) or TGraphXSetAttention "
            "(alias SetTransformerModel) directly."
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
        agg_params = kwargs.get("aggregator_params") or {}
        agg_params = dict(agg_params)
        if "use_batchnorm" in kwargs:
            agg_params.setdefault("use_batchnorm", bool(kwargs["use_batchnorm"]))
        dropout = kwargs.get("dropout")
        return ConvMessagePassing(
            in_shape=in_shape,
            out_shape=out_shape,
            aggr=kwargs.get("aggr", "sum"),
            use_edge_features=bool(kwargs.get("use_edge_features", False)),
            residual=bool(kwargs.get("residual", False)),
            aggregator_params=agg_params or None,
            dropout_prob=float(dropout) if dropout is not None else None,
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
            attention_mode=kwargs.get("attention_mode", "scalar"),
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
            hidden_channels=kwargs.get("hidden_channels", None),
            eps=float(kwargs.get("eps", 0.0)),
            train_eps=bool(kwargs.get("train_eps", False)),
            use_batchnorm=bool(kwargs.get("use_batchnorm", False)),
            use_edge_features=bool(kwargs.get("use_edge_features", False)),
            edge_dim=kwargs.get("edge_dim", None),
            edge_features_kind=kwargs.get("edge_features_kind", "spatial"),
            spatial_rank=spatial_rank,
        )

    if name == "graph_transformer":
        if spatial_rank is not None:
            raise ValueError(
                f"'graph_transformer' currently supports only vector "
                f"in_shape (D,); got spatial in_shape {in_shape}. "
                f"Tensor-aware Graph Transformer is on the roadmap."
            )
        return GraphTransformerLayer(
            in_dim=in_ch,
            out_dim=out_ch,
            num_heads=int(kwargs.get("heads", 4)),
            ffn_dim=kwargs.get("ffn_dim", None),
            dropout=float(kwargs.get("dropout", 0.0)),
            attention_dropout=float(kwargs.get("attention_dropout", 0.0)),
            residual=bool(kwargs.get("residual", True)),
            layer_norm=bool(kwargs.get("layer_norm", True)),
            bias=bool(kwargs.get("bias", True)),
            edge_bias=bool(kwargs.get("edge_bias", False)),
            positional_encoding=kwargs.get("positional_encoding", None),
            pe_dim=int(kwargs.get("pe_dim", 0)),
        )

    if name == "linear":
        return LinearMessagePassing(
            in_shape=in_shape,
            out_shape=out_shape,
            aggr=kwargs.get("aggr", "sum"),
            use_edge_features=bool(kwargs.get("use_edge_features", False)),
            dropout_prob=float(kwargs.get("dropout", 0.0)),
            residual=bool(kwargs.get("residual", False)),
            use_batchnorm=bool(kwargs.get("use_batchnorm", False)),
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
