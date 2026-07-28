"""TGraphXSetAttention: learned implicit relations over tensor-valued nodes.

Canonical class name: :class:`TGraphXSetAttention` (paper/table label
``TGraphX-SetAttn``).  :class:`SetTransformerModel` remains a fully
compatible alias — same class object, same state dicts, same configs.

A first-class TGraphX platform component for the *learned implicit
relations* regime: every node attends to every other node in its graph
(set) through global self-attention, so pairwise interactions are
inferred from node content instead of being supplied as ``edge_index``.
Global content-based relation learning over tensor-valued entities
without requiring a supplied edge graph.

Conceptual contract
-------------------
* **Relation-aware**: self-attention learns content-dependent
  interactions among nodes; this is not a pooling-only (DeepSets-style)
  model.
* **Explicit-input-topology-blind**: a supplied ``edge_index`` is never
  consumed.  By default the model warns once
  (:class:`tgraphx.models.topology.TopologyIgnoredWarning`) when one is
  passed; set ``on_edge_index="error"`` to reject it or ``"ignore"`` to
  accept it silently.
* **Not TensorGAT**: TensorGAT attends only over supplied graph edges
  (topology source ``"given"``).
* **Not learned explicit topology**: no discrete edge set is
  constructed; see :mod:`tgraphx.learned_graph` for edge scorers.

Architecture
------------
``encoder`` (shared per-node tensor encoder → ``[N, embed_dim]``)
→ ``num_layers`` × multi-head self-attention blocks with key padding
masks (permutation-*equivariant*; pre-LN/GELU by default, post-LN and
ReLU available via ``norm_order`` / ``activation``)
→ permutation-*invariant* readout (``"attention"`` pooling by multi-head
attention with learned seed queries, or ``"mean"``/``"sum"``/``"max"``)
→ linear head (optionally one hidden layer via ``head_hidden_dim``).

Every architectural axis used by the TGraphX evaluation program is an
explicit constructor option: encoder architecture (``encoder_config``
``"architecture"``: ``"cnn"`` or ``"strided"``), encoder channel
schedule, normalization order, activation, per-site dropout, attention
block count/heads/FFN width, readout seeds, and head shape.
:meth:`TGraphXSetAttention.reference_config` returns the exact evaluated
reference configuration, and
:meth:`TGraphXSetAttention.map_reference_state_dict` maps checkpoints
saved from the torch-primitives reference layout onto this class.

Batching follows the TGraphX flat convention: node features ``[N, ...]``
plus a ``batch`` vector mapping nodes to graphs.  Dense ``[B, M, E]``
tokens and the padding mask are derived internally, so
:class:`tgraphx.GraphBatch`, :func:`tgraphx.fit`, and the experiment
runner work unchanged.

All regularization is explicit: ``dropout`` and ``attention_dropout``
default to ``0.0`` and appear in ``repr()`` and :meth:`config`.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..layers._dim import batchnorm as _batchnorm_nd
from ..layers.pooling import global_max_pool, global_mean_pool, global_sum_pool
from .cnn_encoder import CNNEncoder
from .topology import TopologyIgnoredWarning

__all__ = [
    "TGraphXSetAttention",
    "SetTransformerModel",
    "SetAttentionBlock",
    "AttentionPooling",
    "StridedConvEncoder",
]

_TASKS = (
    "node_classification",
    "node_regression",
    "graph_classification",
    "graph_regression",
)
_POOLINGS = ("attention", "mean", "sum", "max")
_ON_EDGE_INDEX = ("warn", "ignore", "error")
_NORM_ORDERS = ("pre", "post")
_ACTIVATIONS = ("gelu", "relu")
_ENCODER_ARCHITECTURES = ("cnn", "strided")

# Explicit defaults for the built-in tensor encoders (spatial inputs).
# dropout_prob is deliberately 0.0 — no hidden regularization (v1.5.0).
_DEFAULT_ENCODER_CONFIG: Dict[str, Any] = {
    "architecture": "cnn",
    "num_layers": 3,
    "hidden_channels": 32,
    "dropout_prob": 0.0,
    "use_batchnorm": True,
    "use_residual": False,
    "pool_layers": 1,
}

# Defaults for the "strided" spatial encoder (channel-growing strided
# convolutions, adaptive average pool, linear projection).
_STRIDED_ENCODER_CONFIG: Dict[str, Any] = {
    "architecture": "strided",
    "num_layers": 3,
    "hidden_channels": 32,
    "channel_multiplier": 2,
    "channel_schedule": None,
    "dropout_prob": 0.0,
    "use_batchnorm": True,
}


class SetAttentionBlock(nn.Module):
    """Multi-head self-attention block over padded set tokens.

    Permutation-equivariant: reordering the (real) tokens of a set
    reorders the outputs identically.

    ``norm_order`` selects where LayerNorm is applied:

    * ``"pre"`` (default) — pre-LN: ``x + attn(LN(x))`` then
      ``x + ffn(LN(x))``.
    * ``"post"`` — post-LN, matching
      :class:`torch.nn.TransformerEncoderLayer` with
      ``norm_first=False``: ``LN(x + attn(x))`` then ``LN(x + ffn(x))``.

    Args:
        embed_dim: Token width.
        num_heads: Attention heads (``embed_dim`` divisible by it).
        ffn_dim: Feed-forward hidden width (default ``2 * embed_dim``).
        dropout: Dropout after attention output and inside/after the FFN.
        attention_dropout: Dropout on attention weights.
        layer_norm: Apply LayerNorm around each sublayer.
        norm_order: ``"pre"`` (default) or ``"post"``.
        activation: FFN activation — ``"gelu"`` (default) or ``"relu"``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        layer_norm: bool = True,
        norm_order: str = "pre",
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        norm_order = str(norm_order).lower().strip()
        if norm_order not in _NORM_ORDERS:
            raise ValueError(
                f"norm_order must be one of {_NORM_ORDERS}; got {norm_order!r}."
            )
        activation = str(activation).lower().strip()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {_ACTIVATIONS}; got {activation!r}."
            )
        ffn_dim = int(ffn_dim) if ffn_dim is not None else 2 * embed_dim
        self.norm_order = norm_order
        self.activation = activation
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        act: nn.Module = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            act,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim) if layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim) if layer_norm else nn.Identity()
        self.out_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``tokens``: ``[B, M, E]``; ``key_padding_mask``: ``[B, M]`` bool,
        ``True`` marks padding positions."""
        if self.norm_order == "pre":
            h = self.norm1(tokens)
            attn_out, _ = self.attn(
                h, h, h, key_padding_mask=key_padding_mask, need_weights=False
            )
            tokens = tokens + self.out_dropout(attn_out)
            tokens = tokens + self.out_dropout(self.ffn(self.norm2(tokens)))
            return tokens
        # post-norm (torch.nn.TransformerEncoderLayer, norm_first=False)
        attn_out, _ = self.attn(
            tokens, tokens, tokens,
            key_padding_mask=key_padding_mask, need_weights=False,
        )
        tokens = self.norm1(tokens + self.out_dropout(attn_out))
        tokens = self.norm2(tokens + self.out_dropout(self.ffn(tokens)))
        return tokens

    def extra_repr(self) -> str:
        return f"norm_order={self.norm_order!r}, activation={self.activation!r}"


class AttentionPooling(nn.Module):
    """Permutation-invariant pooling by multi-head attention (PMA).

    ``num_seeds`` learned seed vectors attend over the set tokens; output
    is ``[B, num_seeds * embed_dim]``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_seeds: int = 1,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_seeds < 1:
            raise ValueError(f"num_seeds must be >= 1; got {num_seeds}")
        self.seeds = nn.Parameter(torch.empty(1, num_seeds, embed_dim))
        nn.init.trunc_normal_(self.seeds, std=0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.num_seeds = num_seeds
        self.embed_dim = embed_dim

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = self.seeds.expand(tokens.size(0), -1, -1)
        pooled, _ = self.attn(
            query, tokens, tokens,
            key_padding_mask=key_padding_mask, need_weights=False,
        )
        return pooled.reshape(tokens.size(0), self.num_seeds * self.embed_dim)

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, num_seeds={self.num_seeds}"


class StridedConvEncoder(nn.Module):
    """Strided channel-growing 2-D encoder for ``[N, C, H, W]`` nodes.

    Per layer: 3×3 pad-1 convolution (stride 2 from the second layer on,
    stride 1 for the first), optional BatchNorm, ReLU, optional 2-D
    dropout; then adaptive average pooling to 1×1 and a linear projection
    to ``embed_dim``.  No residual connections.

    The output channel count of layer ``i`` defaults to
    ``hidden_channels * channel_multiplier ** i`` (e.g. 32 → 64 → 128);
    pass ``channel_schedule`` for an explicit per-layer list.

    Selected inside :class:`TGraphXSetAttention` via
    ``encoder_config={"architecture": "strided", ...}``.  This is the
    node-encoder architecture of the evaluated reference set-attention
    configuration (see :meth:`TGraphXSetAttention.reference_config`), and
    its parameter names (``conv.*``, ``proj.*``) match the reference
    layout, so reference checkpoints load without renaming.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_layers: int = 3,
        hidden_channels: int = 32,
        channel_multiplier: int = 2,
        channel_schedule=None,
        dropout_prob: float = 0.0,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1; got {num_layers}")
        if channel_schedule is not None:
            schedule = [int(c) for c in channel_schedule]
            if len(schedule) != num_layers:
                raise ValueError(
                    f"channel_schedule has {len(schedule)} entries but "
                    f"num_layers={num_layers}."
                )
        else:
            schedule = [
                int(hidden_channels) * int(channel_multiplier) ** i
                for i in range(num_layers)
            ]
        blocks: list[nn.Module] = []
        c_in = int(in_channels)
        for i, c_out in enumerate(schedule):
            blocks.append(
                nn.Conv2d(
                    c_in, c_out, kernel_size=3, padding=1,
                    stride=2 if i > 0 else 1,
                )
            )
            if use_batchnorm:
                blocks.append(nn.BatchNorm2d(c_out))
            blocks.append(nn.ReLU(inplace=True))
            if dropout_prob > 0:
                blocks.append(nn.Dropout2d(p=dropout_prob))
            c_in = c_out
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(c_in, int(embed_dim))
        self.channel_schedule = schedule
        self.dropout_prob = float(dropout_prob)
        self.use_batchnorm = bool(use_batchnorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.pool(h).flatten(1)
        return self.proj(h)

    def extra_repr(self) -> str:
        return (
            f"channel_schedule={self.channel_schedule}, "
            f"dropout_prob={self.dropout_prob}, "
            f"use_batchnorm={self.use_batchnorm}"
        )


class _VolumeEncoder(nn.Module):
    """Minimal Conv3d encoder for ``[N, C, D, H, W]`` nodes → ``[N, embed_dim]``."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_layers: int = 3,
        hidden_channels: int = 32,
        dropout_prob: float = 0.0,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(nn.Conv3d(channels, hidden_channels, kernel_size=3, padding=1))
            if use_batchnorm:
                layers.append(_batchnorm_nd(3, hidden_channels))
            layers.append(nn.ReLU(inplace=True))
            if dropout_prob > 0:
                layers.append(nn.Dropout3d(p=dropout_prob))
            channels = hidden_channels
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(hidden_channels, embed_dim)
        self.dropout_prob = float(dropout_prob)
        self.use_batchnorm = use_batchnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pool(self.cnn(x)).flatten(1)
        return self.fc(out)


def _build_default_encoder(
    in_shape: tuple, embed_dim: int, encoder_config: Dict[str, Any]
) -> nn.Module:
    rank = len(in_shape)
    architecture = str(encoder_config.get("architecture", "cnn")).lower().strip()
    if architecture not in _ENCODER_ARCHITECTURES:
        raise ValueError(
            f"encoder_config['architecture'] must be one of "
            f"{_ENCODER_ARCHITECTURES}; got {architecture!r}."
        )
    if architecture == "strided" and rank != 3:
        raise ValueError(
            "encoder_config['architecture']='strided' requires a 2-D spatial "
            f"in_shape (C, H, W); got {in_shape}."
        )
    if rank == 1:
        hidden = int(encoder_config.get("hidden_channels", 2 * embed_dim))
        p = float(encoder_config.get("dropout_prob", 0.0))
        return nn.Sequential(
            nn.Linear(in_shape[0], hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p) if p > 0 else nn.Identity(),
            nn.Linear(hidden, embed_dim),
        )
    if rank == 3:
        cfg = {k: v for k, v in encoder_config.items() if k != "architecture"}
        if architecture == "strided":
            return StridedConvEncoder(
                in_channels=in_shape[0],
                embed_dim=embed_dim,
                **cfg,
            )
        return CNNEncoder(
            in_channels=in_shape[0],
            out_features=embed_dim,
            return_feature_map=False,
            **cfg,
        )
    if rank == 4:
        return _VolumeEncoder(
            in_channels=in_shape[0],
            embed_dim=embed_dim,
            num_layers=int(encoder_config.get("num_layers", 3)),
            hidden_channels=int(encoder_config.get("hidden_channels", 32)),
            dropout_prob=float(encoder_config.get("dropout_prob", 0.0)),
            use_batchnorm=bool(encoder_config.get("use_batchnorm", True)),
        )
    raise ValueError(
        f"TGraphXSetAttention: in_shape must have 1 (vector), 3 (2-D spatial), "
        f"or 4 (3-D volumetric) elements; got {in_shape}."
    )


def _dense_and_mask(h: torch.Tensor, batch: torch.Tensor):
    """Flat ``[N, E]`` + ``batch`` → dense ``[B, M, E]``, pad mask, scatter indices.

    Robust to unsorted ``batch`` vectors (stable sort keeps node order
    within each graph deterministic).
    """
    if batch.numel() == 0:
        raise ValueError(
            "TGraphXSetAttention: received an empty batch (0 nodes)."
        )
    num_graphs = int(batch.max().item()) + 1
    counts = torch.bincount(batch, minlength=num_graphs)
    if int(counts.min().item()) == 0:
        empty = (counts == 0).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            f"TGraphXSetAttention: graphs {empty} in the batch have zero "
            f"nodes; set attention over an empty set is undefined."
        )
    max_nodes = int(counts.max().item())
    order = torch.argsort(batch, stable=True)
    sorted_batch = batch[order]
    starts = torch.cumsum(counts, dim=0) - counts
    pos = torch.arange(batch.numel(), device=batch.device) - starts[sorted_batch]

    tokens = h.new_zeros(num_graphs, max_nodes, h.size(-1))
    tokens[sorted_batch, pos] = h[order]
    pad_mask = torch.ones(
        num_graphs, max_nodes, dtype=torch.bool, device=h.device
    )
    pad_mask[sorted_batch, pos] = False
    return tokens, pad_mask, order, sorted_batch, pos


class TGraphXSetAttention(nn.Module):
    """Set-attention model over tensor-valued nodes (topology source
    ``"learned_implicit"``): global content-based relation learning over
    tensor-valued entities without requiring a supplied edge graph.

    Canonical class name.  :class:`SetTransformerModel` is a fully
    compatible alias for this class (same object; imports, ``isinstance``
    checks, configs, state dicts, and pickled models continue to work).
    Paper/table label: ``TGraphX-SetAttn``.

    Args:
        task: ``"node_classification"``, ``"node_regression"``,
            ``"graph_classification"``, or ``"graph_regression"``.
        in_shape: Per-node input shape — ``(D,)`` vector, ``(C, H, W)``
            2-D spatial, or ``(C, D, H, W)`` 3-D volumetric.
        embed_dim: Token width used by the attention blocks.
        num_layers: Number of :class:`SetAttentionBlock` layers (>= 1).
        num_heads: Attention heads per block.
        ffn_dim: Feed-forward width inside each block
            (default ``2 * embed_dim``).
        dropout: Dropout in attention outputs and FFNs (default ``0.0``,
            explicit — never silently nonzero).
        attention_dropout: Dropout on attention weights inside the
            attention blocks (default ``0.0``).
        num_classes: Required for classification tasks.
        out_dim: Required for regression tasks.
        pooling: Graph-level readout: ``"attention"`` (default; pooling by
            multi-head attention with learned seeds), ``"mean"``,
            ``"sum"``, or ``"max"``.  Ignored for node-level tasks.
        num_seeds: Seed vectors for ``"attention"`` pooling; the head
            input width is ``num_seeds * embed_dim``.
        layer_norm: LayerNorm around each block sublayer (default
            ``True``).
        norm_order: LayerNorm placement in each block: ``"pre"``
            (default) or ``"post"`` (post-LN, the
            :class:`torch.nn.TransformerEncoderLayer` convention with
            ``norm_first=False``).
        activation: Block FFN activation: ``"gelu"`` (default) or
            ``"relu"``.
        pool_attention_dropout: Dropout on the ``"attention"``-pooling
            attention weights.  Default ``None`` follows
            ``attention_dropout``; pass an explicit value (e.g. ``0.0``)
            to decouple the readout from the block setting.
        head_hidden_dim: When set, the output head becomes
            ``Linear → ReLU → Linear`` with this hidden width; default
            ``None`` keeps a single linear head.
        encoder: Optional custom shared node encoder mapping
            ``[N, *in_shape]`` → ``[N, embed_dim]``.  When provided,
            ``encoder_config`` must be ``None`` and :meth:`config` cannot
            reconstruct the encoder (it records ``encoder="custom"``).
        encoder_config: Overrides for the built-in encoder.  For spatial
            inputs, ``{"architecture": "cnn"}`` (default) selects
            :class:`tgraphx.CNNEncoder` with explicit defaults
            ``{num_layers: 3, hidden_channels: 32, dropout_prob: 0.0,
            use_batchnorm: True, use_residual: False, pool_layers: 1}``;
            ``{"architecture": "strided"}`` selects
            :class:`StridedConvEncoder` (strided channel-growing
            convolutions, no residual) with explicit defaults
            ``{num_layers: 3, hidden_channels: 32,
            channel_multiplier: 2, channel_schedule: None,
            dropout_prob: 0.0, use_batchnorm: True}``.
        on_edge_index: What to do when ``forward`` receives a non-``None``
            ``edge_index``: ``"warn"`` (default; warn once per instance
            with :class:`TopologyIgnoredWarning`), ``"ignore"`` (silent),
            or ``"error"`` (raise ``ValueError``).

    Forward::

        out = model(x, edge_index=None, edge_features=None,
                    edge_weight=None, batch=None)

    ``batch`` is required for graph-level tasks with more than one graph;
    when ``None`` all nodes are treated as one set.  ``edge_index``,
    ``edge_features``, and ``edge_weight`` are accepted for pipeline
    compatibility but never consumed.
    """

    #: Relation regime of this family (see :mod:`tgraphx.models.topology`).
    topology_source = "learned_implicit"
    #: Factory family name (stable machine name; the canonical class name
    #: is ``TGraphXSetAttention``).
    model_family = "set_transformer"

    def __init__(
        self,
        task: str,
        in_shape,
        embed_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: Optional[int] = None,
        out_dim: Optional[int] = None,
        pooling: str = "attention",
        num_seeds: int = 1,
        layer_norm: bool = True,
        norm_order: str = "pre",
        activation: str = "gelu",
        pool_attention_dropout: Optional[float] = None,
        head_hidden_dim: Optional[int] = None,
        encoder: Optional[nn.Module] = None,
        encoder_config: Optional[Dict[str, Any]] = None,
        on_edge_index: str = "warn",
    ) -> None:
        super().__init__()
        task = task.lower().strip()
        if task not in _TASKS:
            raise ValueError(
                f"TGraphXSetAttention: unsupported task {task!r}. "
                f"Supported: {', '.join(repr(t) for t in _TASKS)}."
            )
        in_shape = tuple(int(s) for s in in_shape)
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1; got {num_layers}")
        pooling = str(pooling).lower().strip()
        if pooling not in _POOLINGS:
            raise ValueError(
                f"pooling must be one of {_POOLINGS}; got {pooling!r}."
            )
        norm_order = str(norm_order).lower().strip()
        if norm_order not in _NORM_ORDERS:
            raise ValueError(
                f"norm_order must be one of {_NORM_ORDERS}; got {norm_order!r}."
            )
        activation = str(activation).lower().strip()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {_ACTIVATIONS}; got {activation!r}."
            )
        if on_edge_index not in _ON_EDGE_INDEX:
            raise ValueError(
                f"on_edge_index must be one of {_ON_EDGE_INDEX}; "
                f"got {on_edge_index!r}."
            )
        classification = task.endswith("classification")
        if classification and num_classes is None:
            raise ValueError(f"num_classes is required for task={task!r}.")
        if not classification and out_dim is None:
            raise ValueError(f"out_dim is required for task={task!r}.")

        if encoder is not None and encoder_config is not None:
            raise ValueError(
                "Pass either a custom encoder or encoder_config, not both."
            )

        self.task = task
        self.in_shape = in_shape
        self.embed_dim = int(embed_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(ffn_dim) if ffn_dim is not None else 2 * int(embed_dim)
        self.dropout = float(dropout)
        self.attention_dropout = float(attention_dropout)
        self.num_classes = num_classes
        self.out_dim = out_dim
        self.pooling = pooling
        self.num_seeds = int(num_seeds)
        self.layer_norm = bool(layer_norm)
        self.norm_order = norm_order
        self.activation = activation
        self.pool_attention_dropout = (
            None if pool_attention_dropout is None else float(pool_attention_dropout)
        )
        self.head_hidden_dim = (
            None if head_hidden_dim is None else int(head_hidden_dim)
        )
        self.on_edge_index = on_edge_index
        self._custom_encoder = encoder is not None
        self._warned_edge_index = False

        if encoder is not None:
            self.encoder = encoder
            self.encoder_config: Dict[str, Any] = {}
        else:
            requested_arch = (encoder_config or {}).get("architecture", "cnn")
            if str(requested_arch).lower().strip() == "strided":
                cfg = dict(_STRIDED_ENCODER_CONFIG)
            else:
                cfg = dict(_DEFAULT_ENCODER_CONFIG)
            if len(in_shape) == 1:
                cfg = {
                    "hidden_channels": 2 * self.embed_dim,
                    "dropout_prob": 0.0,
                }
            if encoder_config:
                cfg.update(encoder_config)
            self.encoder = _build_default_encoder(in_shape, self.embed_dim, cfg)
            self.encoder_config = cfg

        self.blocks = nn.ModuleList(
            SetAttentionBlock(
                self.embed_dim,
                self.num_heads,
                ffn_dim=self.ffn_dim,
                dropout=self.dropout,
                attention_dropout=self.attention_dropout,
                layer_norm=self.layer_norm,
                norm_order=self.norm_order,
                activation=self.activation,
            )
            for _ in range(self.num_layers)
        )

        graph_level = task.startswith("graph")
        self.pool: Optional[AttentionPooling]
        if graph_level and pooling == "attention":
            self.pool = AttentionPooling(
                self.embed_dim,
                self.num_heads,
                num_seeds=self.num_seeds,
                attention_dropout=(
                    self.attention_dropout
                    if self.pool_attention_dropout is None
                    else self.pool_attention_dropout
                ),
            )
            head_in = self.num_seeds * self.embed_dim
        else:
            self.pool = None
            head_in = self.embed_dim
        target = num_classes if classification else out_dim
        assert target is not None  # guaranteed by the task checks above
        if self.head_hidden_dim is not None:
            self.head: nn.Module = nn.Sequential(
                nn.Linear(head_in, self.head_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.head_hidden_dim, int(target)),
            )
        else:
            self.head = nn.Linear(head_in, int(target))

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def _check_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"TGraphXSetAttention: x must be a Tensor; got {type(x).__name__}."
            )
        expected = 1 + len(self.in_shape)
        if x.dim() != expected or tuple(x.shape[1:]) != self.in_shape:
            raise ValueError(
                f"TGraphXSetAttention: expected node features of shape "
                f"[N, {', '.join(map(str, self.in_shape))}]; got {tuple(x.shape)}."
            )

    def _handle_edge_index(self, edge_index) -> None:
        if edge_index is None:
            return
        if self.on_edge_index == "error":
            raise ValueError(
                "TGraphXSetAttention received edge_index, but its topology "
                "source is 'learned_implicit': relations are inferred from "
                "node content and the supplied topology would be ignored. "
                "Use a 'given'-topology family (conv/gat/sage/gin) to "
                "consume edge_index, or construct with on_edge_index='warn' "
                "or 'ignore'."
            )
        if self.on_edge_index == "warn" and not self._warned_edge_index:
            self._warned_edge_index = True
            warnings.warn(
                "TGraphXSetAttention ignores the supplied edge_index "
                "(topology_source='learned_implicit'): relations are learned "
                "from node content by global self-attention. Pass "
                "on_edge_index='ignore' to silence, or 'error' to reject.",
                TopologyIgnoredWarning,
                stacklevel=3,
            )

    def _encode_tokens(self, x: torch.Tensor, batch: Optional[torch.Tensor]):
        """Shared encoder + attention stack → padded tokens and indices."""
        self._check_input(x)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        if batch.numel() != x.size(0):
            raise ValueError(
                f"batch has {batch.numel()} entries but x has {x.size(0)} nodes."
            )
        h = self.encoder(x)
        if h.dim() != 2 or h.size(-1) != self.embed_dim:
            raise ValueError(
                f"Encoder must produce [N, embed_dim={self.embed_dim}] "
                f"embeddings; got {tuple(h.shape)}. Check the custom encoder."
            )
        tokens, pad_mask, order, sorted_batch, pos = _dense_and_mask(h, batch)
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=pad_mask)
        return tokens, pad_mask, order, sorted_batch, pos, batch

    def _flatten_tokens(self, tokens, order, sorted_batch, pos) -> torch.Tensor:
        flat = tokens.new_empty(order.numel(), self.embed_dim)
        flat[order] = tokens[sorted_batch, pos]
        return flat

    def encode_nodes(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Permutation-equivariant per-node embeddings ``[N, embed_dim]``.

        Runs the shared encoder and all set-attention blocks but no
        readout.  Nodes only attend to nodes in the same graph.
        """
        tokens, _, order, sorted_batch, pos, _ = self._encode_tokens(x, batch)
        return self._flatten_tokens(tokens, order, sorted_batch, pos)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._handle_edge_index(edge_index)
        tokens, pad_mask, order, sorted_batch, pos, batch = self._encode_tokens(
            x, batch
        )

        if self.task.startswith("node"):
            return self.head(
                self._flatten_tokens(tokens, order, sorted_batch, pos)
            )

        # Graph-level readout (permutation-invariant).
        if self.pool is not None:
            pooled = self.pool(tokens, key_padding_mask=pad_mask)
        else:
            flat = self._flatten_tokens(tokens, order, sorted_batch, pos)
            if self.pooling == "mean":
                pooled = global_mean_pool(flat, batch)
            elif self.pooling == "sum":
                pooled = global_sum_pool(flat, batch)
            else:
                pooled = global_max_pool(flat, batch)
        return self.head(pooled)

    # ------------------------------------------------------------------ #
    # Config round trip                                                    #
    # ------------------------------------------------------------------ #

    def config(self) -> Dict[str, Any]:
        """Exact constructor configuration for deterministic reconstruction.

        ``TGraphXSetAttention.from_config(model.config())`` rebuilds an
        architecturally identical model (load its ``state_dict`` to
        restore weights).  Models built with a custom ``encoder`` record
        ``{"encoder": "custom"}`` and cannot be reconstructed from config
        alone.
        """
        cfg: Dict[str, Any] = {
            "task": self.task,
            "in_shape": list(self.in_shape),
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "num_classes": self.num_classes,
            "out_dim": self.out_dim,
            "pooling": self.pooling,
            "num_seeds": self.num_seeds,
            "layer_norm": self.layer_norm,
            "norm_order": self.norm_order,
            "activation": self.activation,
            "pool_attention_dropout": self.pool_attention_dropout,
            "head_hidden_dim": self.head_hidden_dim,
            "encoder_config": dict(self.encoder_config),
            "on_edge_index": self.on_edge_index,
            "model_family": self.model_family,
            "topology_source": self.topology_source,
        }
        if self._custom_encoder:
            cfg["encoder"] = "custom"
            cfg["encoder_config"] = {}
        return cfg

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "TGraphXSetAttention":
        """Reconstruct a model from :meth:`config` output.

        Accepts configs exported by earlier package versions (fields
        introduced later simply take their defaults).
        """
        cfg = dict(cfg)
        if cfg.pop("encoder", None) == "custom":
            raise ValueError(
                "This config was exported from a model with a custom "
                "encoder; rebuild it by passing the encoder module to "
                "TGraphXSetAttention(...) directly."
            )
        cfg.pop("model_family", None)
        cfg.pop("topology_source", None)
        encoder_config = cfg.pop("encoder_config", None) or None
        return cls(encoder_config=encoder_config, **cfg)

    # ------------------------------------------------------------------ #
    # Evaluated reference configuration                                    #
    # ------------------------------------------------------------------ #

    @classmethod
    def reference_config(
        cls,
        in_shape,
        num_classes: int,
        task: str = "graph_classification",
    ) -> Dict[str, Any]:
        """Constructor kwargs of the evaluated reference architecture.

        This is the exact set-attention architecture evaluated in the
        TGraphX experiment program: a :class:`StridedConvEncoder` node
        encoder (3 layers, channels 32→64→128, BatchNorm, no residual,
        no dropout), two post-LN ReLU attention blocks with dropout 0.1
        on attention weights, attention outputs, and FFNs (the
        :class:`torch.nn.TransformerEncoderLayer` defaults), a
        single-seed attention-pooling readout without attention-weight
        dropout, and a single linear head.

        Data-dependent sizes (``in_shape``, ``num_classes``) are
        required arguments — nothing here is dataset-specific.

        Example::

            cfg = TGraphXSetAttention.reference_config(
                in_shape=(13, 32, 32), num_classes=18)
            model = TGraphXSetAttention(**cfg)
        """
        return {
            "task": task,
            "in_shape": tuple(int(s) for s in in_shape),
            "embed_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_dim": 128,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "num_classes": int(num_classes),
            "pooling": "attention",
            "num_seeds": 1,
            "layer_norm": True,
            "norm_order": "post",
            "activation": "relu",
            "pool_attention_dropout": 0.0,
            "head_hidden_dim": None,
            "encoder_config": {
                "architecture": "strided",
                "num_layers": 3,
                "hidden_channels": 32,
                "channel_multiplier": 2,
                "channel_schedule": None,
                "dropout_prob": 0.0,
                "use_batchnorm": True,
            },
            "on_edge_index": "warn",
        }

    #: state-dict key prefixes of the torch-primitives reference layout →
    #: canonical :class:`TGraphXSetAttention` prefixes.
    _REFERENCE_KEY_MAP = (
        ("self_attn.layers.{i}.self_attn.", "blocks.{i}.attn."),
        ("self_attn.layers.{i}.linear1.", "blocks.{i}.ffn.0."),
        ("self_attn.layers.{i}.linear2.", "blocks.{i}.ffn.3."),
        ("self_attn.layers.{i}.norm1.", "blocks.{i}.norm1."),
        ("self_attn.layers.{i}.norm2.", "blocks.{i}.norm2."),
    )

    @staticmethod
    def map_reference_state_dict(
        state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Map a reference-layout state dict onto this class's key names.

        The reference layout is the torch-primitives implementation used
        by the TGraphX evaluation program: ``encoder.conv.*`` /
        ``encoder.proj.*`` (identical to :class:`StridedConvEncoder`),
        ``self_attn.layers.{i}.*`` (:class:`torch.nn.TransformerEncoder`),
        ``pma.query`` / ``pma.attn.*`` (single-seed pooling by multi-head
        attention), and ``head.net.*`` (classifier head).

        Returns a new dict whose keys strict-load into a model built from
        :meth:`reference_config` (all tensors are referenced, not
        copied).  Raises ``KeyError`` for keys that do not belong to the
        reference layout.
        """
        mapped: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                new_key = key
            elif key == "pma.query":
                new_key = "pool.seeds"
            elif key.startswith("pma.attn."):
                new_key = "pool.attn." + key[len("pma.attn."):]
            elif key.startswith("head.net."):
                new_key = "head." + key[len("head.net."):]
            elif key.startswith("self_attn.layers."):
                rest = key[len("self_attn.layers."):]
                index, sub = rest.split(".", 1)
                new_key = None
                for src, dst in TGraphXSetAttention._REFERENCE_KEY_MAP:
                    src_sub = src.split(".", 3)[3]  # part after the index
                    if sub.startswith(src_sub):
                        new_key = (
                            dst.format(i=index) + sub[len(src_sub):]
                        )
                        break
                if new_key is None:
                    raise KeyError(
                        f"Unrecognized reference attention key: {key!r}"
                    )
            else:
                raise KeyError(
                    f"Unrecognized reference state-dict key: {key!r}. "
                    "map_reference_state_dict only accepts the reference "
                    "layout (encoder.* / self_attn.layers.* / pma.* / "
                    "head.net.*)."
                )
            mapped[new_key] = value
        return mapped

    @classmethod
    def from_reference_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        in_shape,
        num_classes: int,
        task: str = "graph_classification",
    ) -> "TGraphXSetAttention":
        """Build the evaluated reference architecture and strict-load a
        reference-layout ``state_dict`` into it (see
        :meth:`reference_config` and :meth:`map_reference_state_dict`)."""
        model = cls(**cls.reference_config(in_shape, num_classes, task=task))
        model.load_state_dict(cls.map_reference_state_dict(state_dict), strict=True)
        return model

    def extra_repr(self) -> str:
        return (
            f"task={self.task!r}, in_shape={self.in_shape}, "
            f"embed_dim={self.embed_dim}, num_layers={self.num_layers}, "
            f"num_heads={self.num_heads}, ffn_dim={self.ffn_dim}, "
            f"dropout={self.dropout}, attention_dropout={self.attention_dropout}, "
            f"pooling={self.pooling!r}, num_seeds={self.num_seeds}, "
            f"layer_norm={self.layer_norm}, norm_order={self.norm_order!r}, "
            f"activation={self.activation!r}, "
            f"topology_source={self.topology_source!r}, "
            f"on_edge_index={self.on_edge_index!r}"
        )


#: Backward-compatible alias: the pre-1.5.1 public name of
#: :class:`TGraphXSetAttention`.  Same class object — imports, isinstance
#: checks, ``from_config``, state dicts, and pickled models all continue
#: to work.
SetTransformerModel = TGraphXSetAttention
