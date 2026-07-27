"""SetTransformer: learned implicit relations over tensor-valued nodes.

A first-class TGraphX platform component for the *learned implicit
relations* regime: every node attends to every other node in its graph
(set) through global self-attention, so pairwise interactions are
inferred from node content instead of being supplied as ``edge_index``.

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
→ ``num_layers`` × pre-LN multi-head self-attention blocks with key
padding masks (permutation-*equivariant*)
→ permutation-*invariant* readout (``"attention"`` pooling by multi-head
attention with learned seed queries, or ``"mean"``/``"sum"``/``"max"``)
→ linear head.

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

__all__ = ["SetTransformerModel", "SetAttentionBlock", "AttentionPooling"]

_TASKS = (
    "node_classification",
    "node_regression",
    "graph_classification",
    "graph_regression",
)
_POOLINGS = ("attention", "mean", "sum", "max")
_ON_EDGE_INDEX = ("warn", "ignore", "error")

# Explicit defaults for the built-in tensor encoder (spatial inputs).
# dropout_prob is deliberately 0.0 — no hidden regularization (v1.5.0).
_DEFAULT_ENCODER_CONFIG: Dict[str, Any] = {
    "num_layers": 3,
    "hidden_channels": 32,
    "dropout_prob": 0.0,
    "use_batchnorm": True,
    "use_residual": False,
    "pool_layers": 1,
}


class SetAttentionBlock(nn.Module):
    """Pre-LN multi-head self-attention block over padded set tokens.

    Permutation-equivariant: reordering the (real) tokens of a set
    reorders the outputs identically.

    Args:
        embed_dim: Token width.
        num_heads: Attention heads (``embed_dim`` divisible by it).
        ffn_dim: Feed-forward hidden width (default ``2 * embed_dim``).
        dropout: Dropout after attention output and inside/after the FFN.
        attention_dropout: Dropout on attention weights.
        layer_norm: Apply pre-LayerNorm before each sublayer.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        ffn_dim = int(ffn_dim) if ffn_dim is not None else 2 * embed_dim
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
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
        h = self.norm1(tokens)
        attn_out, _ = self.attn(
            h, h, h, key_padding_mask=key_padding_mask, need_weights=False
        )
        tokens = tokens + self.out_dropout(attn_out)
        tokens = tokens + self.out_dropout(self.ffn(self.norm2(tokens)))
        return tokens


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
        return CNNEncoder(
            in_channels=in_shape[0],
            out_features=embed_dim,
            return_feature_map=False,
            **encoder_config,
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
        f"SetTransformerModel: in_shape must have 1 (vector), 3 (2-D spatial), "
        f"or 4 (3-D volumetric) elements; got {in_shape}."
    )


def _dense_and_mask(h: torch.Tensor, batch: torch.Tensor):
    """Flat ``[N, E]`` + ``batch`` → dense ``[B, M, E]``, pad mask, scatter indices.

    Robust to unsorted ``batch`` vectors (stable sort keeps node order
    within each graph deterministic).
    """
    if batch.numel() == 0:
        raise ValueError(
            "SetTransformerModel: received an empty batch (0 nodes)."
        )
    num_graphs = int(batch.max().item()) + 1
    counts = torch.bincount(batch, minlength=num_graphs)
    if int(counts.min().item()) == 0:
        empty = (counts == 0).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            f"SetTransformerModel: graphs {empty} in the batch have zero "
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


class SetTransformerModel(nn.Module):
    """Set-attention model over tensor-valued nodes (topology source
    ``"learned_implicit"``).

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
        attention_dropout: Dropout on attention weights (default ``0.0``).
        num_classes: Required for classification tasks.
        out_dim: Required for regression tasks.
        pooling: Graph-level readout: ``"attention"`` (default; pooling by
            multi-head attention with learned seeds), ``"mean"``,
            ``"sum"``, or ``"max"``.  Ignored for node-level tasks.
        num_seeds: Seed vectors for ``"attention"`` pooling; the head
            input width is ``num_seeds * embed_dim``.
        layer_norm: Pre-LayerNorm in each block (default ``True``).
        encoder: Optional custom shared node encoder mapping
            ``[N, *in_shape]`` → ``[N, embed_dim]``.  When provided,
            ``encoder_config`` must be ``None`` and :meth:`config` cannot
            reconstruct the encoder (it records ``encoder="custom"``).
        encoder_config: Overrides for the built-in encoder.  For spatial
            inputs the built-in encoder is :class:`tgraphx.CNNEncoder`
            with explicit defaults
            ``{num_layers: 3, hidden_channels: 32, dropout_prob: 0.0,
            use_batchnorm: True, use_residual: False, pool_layers: 1}``.
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
    #: Factory family name.
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
        encoder: Optional[nn.Module] = None,
        encoder_config: Optional[Dict[str, Any]] = None,
        on_edge_index: str = "warn",
    ) -> None:
        super().__init__()
        task = task.lower().strip()
        if task not in _TASKS:
            raise ValueError(
                f"SetTransformerModel: unsupported task {task!r}. "
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
        self.on_edge_index = on_edge_index
        self._custom_encoder = encoder is not None
        self._warned_edge_index = False

        if encoder is not None:
            self.encoder = encoder
            self.encoder_config: Dict[str, Any] = {}
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
                attention_dropout=self.attention_dropout,
            )
            head_in = self.num_seeds * self.embed_dim
        else:
            self.pool = None
            head_in = self.embed_dim
        target = num_classes if classification else out_dim
        assert target is not None  # guaranteed by the task checks above
        self.head = nn.Linear(head_in, int(target))

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def _check_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"SetTransformerModel: x must be a Tensor; got {type(x).__name__}."
            )
        expected = 1 + len(self.in_shape)
        if x.dim() != expected or tuple(x.shape[1:]) != self.in_shape:
            raise ValueError(
                f"SetTransformerModel: expected node features of shape "
                f"[N, {', '.join(map(str, self.in_shape))}]; got {tuple(x.shape)}."
            )

    def _handle_edge_index(self, edge_index) -> None:
        if edge_index is None:
            return
        if self.on_edge_index == "error":
            raise ValueError(
                "SetTransformerModel received edge_index, but its topology "
                "source is 'learned_implicit': relations are inferred from "
                "node content and the supplied topology would be ignored. "
                "Use a 'given'-topology family (conv/gat/sage/gin) to "
                "consume edge_index, or construct with on_edge_index='warn' "
                "or 'ignore'."
            )
        if self.on_edge_index == "warn" and not self._warned_edge_index:
            self._warned_edge_index = True
            warnings.warn(
                "SetTransformerModel ignores the supplied edge_index "
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

        ``SetTransformerModel.from_config(model.config())`` rebuilds an
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
    def from_config(cls, cfg: Dict[str, Any]) -> "SetTransformerModel":
        """Reconstruct a model from :meth:`config` output."""
        cfg = dict(cfg)
        if cfg.pop("encoder", None) == "custom":
            raise ValueError(
                "This config was exported from a model with a custom "
                "encoder; rebuild it by passing the encoder module to "
                "SetTransformerModel(...) directly."
            )
        cfg.pop("model_family", None)
        cfg.pop("topology_source", None)
        encoder_config = cfg.pop("encoder_config", None) or None
        return cls(encoder_config=encoder_config, **cfg)

    def extra_repr(self) -> str:
        return (
            f"task={self.task!r}, in_shape={self.in_shape}, "
            f"embed_dim={self.embed_dim}, num_layers={self.num_layers}, "
            f"num_heads={self.num_heads}, ffn_dim={self.ffn_dim}, "
            f"dropout={self.dropout}, attention_dropout={self.attention_dropout}, "
            f"pooling={self.pooling!r}, num_seeds={self.num_seeds}, "
            f"layer_norm={self.layer_norm}, "
            f"topology_source={self.topology_source!r}, "
            f"on_edge_index={self.on_edge_index!r}"
        )
