"""Model factory for TGraphX.

High-level functions to create complete GNN models by task name:

    build_model(task, layer, in_shape, hidden_shape, num_layers, ...)
    build_model_from_config(path_or_dict)

Supported tasks
---------------
``"node_classification"``   [N, num_classes] logits
``"node_regression"``       [N, out_dim]
``"graph_classification"``  [G, num_classes] logits (requires batch tensor)
``"graph_regression"``      [G, out_dim]       (requires batch tensor)
``"edge_prediction"``       [E, out_dim] — MLP on concatenated node embeddings

``"link_prediction"`` is intentionally deferred; use ``"edge_prediction"``.

Supported layers
----------------
See :func:`tgraphx.layers.factory.make_layer` for the full list.
``"linear"`` and ``"legacy_attention"`` are limited to vector / 2-D spatial
shapes; all other layers support 2-D and 3-D spatial shapes.

Config support
--------------
``build_model_from_config`` accepts a Python dict, a JSON file path, or a
YAML file path (requires PyYAML).  No ``eval``, no ``exec``, YAML is
loaded with ``safe_load`` only.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.factory import make_layer
from .edge_predictor import EdgePredictor
from .set_transformer import TGraphXSetAttention
from .topology import topology_source_of

try:
    import yaml as _yaml
except ImportError:
    _yaml = None  # type: ignore[assignment]

_SUPPORTED_TASKS = (
    "node_classification",
    "node_regression",
    "graph_classification",
    "graph_regression",
    "edge_prediction",
)

# Families dispatched at the model level (not per-layer via make_layer).
_MODEL_LEVEL_FAMILIES = ("set_transformer",)

# Factory aliases resolving to the same canonical family.  All of these
# build a TGraphXSetAttention (family "set_transformer") — the alias
# never changes the architecture.
_FAMILY_ALIASES = {
    "tgraphx_set_attention": "set_transformer",
    "set_attention": "set_transformer",
    "set_transformer": "set_transformer",
}


def _tag_model(model: nn.Module, family: str, **layer_kwargs) -> nn.Module:
    """Record the family and topology source on a factory-built model."""
    model.model_family = family
    model.topology_source = topology_source_of(family, **layer_kwargs)
    return model


# --------------------------------------------------------------------------- #
# Internal helpers                                                              #
# --------------------------------------------------------------------------- #

def _graph_readout(
    x: torch.Tensor,
    batch: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    num_graphs = int(batch.max()) + 1
    if pooling == "mean":
        out = x.new_zeros(num_graphs, x.size(1))
        count = x.new_zeros(num_graphs)
        out.index_add_(0, batch, x)
        count.index_add_(0, batch, x.new_ones(x.size(0)))
        return out / count.clamp(min=1).unsqueeze(1)
    if pooling == "sum":
        out = x.new_zeros(num_graphs, x.size(1))
        out.index_add_(0, batch, x)
        return out
    if pooling == "max":
        idx = batch.unsqueeze(1).expand_as(x)
        out = x.new_full((num_graphs, x.size(1)), float("-inf"))
        out.scatter_reduce_(0, idx, x, reduce="amax", include_self=True)
        out = out.masked_fill(torch.isinf(out) & (out < 0), 0.0)
        return out
    raise ValueError(f"pooling must be 'mean', 'sum', or 'max'; got {pooling!r}")


# --------------------------------------------------------------------------- #
# Internal model modules                                                        #
# --------------------------------------------------------------------------- #

class _FactoryGNNModel(nn.Module):
    """Generic node/graph classification or regression model.

    Produced by :func:`build_model`.  Forward signature::

        out = model(x, edge_index,
                    edge_features=None, edge_weight=None, batch=None)

    ``batch`` is required for graph-level tasks.
    """

    def __init__(
        self,
        gnn_layers: list[nn.Module],
        task: str,
        out_linear: nn.Linear,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.gnn_layers = nn.ModuleList(gnn_layers)
        self.task = task
        self.out_linear = out_linear
        self.pooling = pooling

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features=None,
        edge_weight=None,
        batch=None,
    ) -> torch.Tensor:
        for layer in self.gnn_layers:
            x = F.relu(layer(x, edge_index, edge_features, edge_weight))

        # Spatial → vector: [N, C, *spatial] → [N, C]
        if x.dim() > 2:
            x = x.mean(dim=list(range(2, x.dim())))

        # Graph-level readout
        if self.task in ("graph_classification", "graph_regression"):
            if batch is None:
                raise ValueError(
                    "batch ([N] node-to-graph mapping) is required for "
                    f"task={self.task!r}. Pass batch=... to forward()."
                )
            x = _graph_readout(x, batch, self.pooling)

        return self.out_linear(x)

    def extra_repr(self) -> str:
        return f"task={self.task!r}, pooling={self.pooling!r}"


class _EdgePredictionModel(nn.Module):
    """GNN stack + MLP edge scorer.

    Produced by :func:`build_model` for task ``"edge_prediction"``.
    """

    def __init__(
        self,
        gnn_layers: list[nn.Module],
        predictor: EdgePredictor,
    ) -> None:
        super().__init__()
        self.gnn_layers = nn.ModuleList(gnn_layers)
        self.predictor = predictor

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features=None,
        edge_weight=None,
        **_,
    ) -> torch.Tensor:
        for layer in self.gnn_layers:
            x = F.relu(layer(x, edge_index, edge_features, edge_weight))
        return self.predictor(x, edge_index)


# --------------------------------------------------------------------------- #
# Public factory functions                                                      #
# --------------------------------------------------------------------------- #

def build_model(
    task: str,
    layer: Optional[str] = None,
    in_shape=None,
    hidden_shape=None,
    num_layers: Optional[int] = None,
    num_classes: Optional[int] = None,
    out_dim: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """Build a complete GNN model for a specific task.

    The model stacks ``num_layers`` GNN layers (first maps ``in_shape`` →
    ``hidden_shape``, subsequent map ``hidden_shape`` → ``hidden_shape``),
    applies global spatial average-pooling when features are tensors, and
    finishes with a task-specific linear head.

    Args:
        task: One of ``"node_classification"``, ``"node_regression"``,
            ``"graph_classification"``, ``"graph_regression"``,
            ``"edge_prediction"``.
        layer: GNN layer type (see :func:`tgraphx.layers.factory.make_layer`)
            or a model-level family name (``"set_transformer"``; the
            aliases ``"tgraphx_set_attention"`` and ``"set_attention"``
            resolve to the same family).
            ``family=`` is accepted as an alias for this argument.
            Families with topology source ``"given"`` (conv/gat/sage/gin/
            linear/legacy_attention) require ``edge_index`` in forward;
            ``"set_transformer"`` (topology source ``"learned_implicit"``)
            ignores a supplied ``edge_index`` and warns once (configurable
            via ``on_edge_index="warn"|"ignore"|"error"``).  Every returned
            model carries ``model.model_family`` and
            ``model.topology_source`` attributes.
        in_shape: Per-node input feature shape (no batch dim).
        hidden_shape: Per-node hidden feature shape.  Must be the same rank
            as ``in_shape``.  For GAT with ``concat=True``, ensure
            ``hidden_shape[0]`` is divisible by ``heads``.
        num_layers: Number of GNN layers (>= 1).
        num_classes: Required for classification tasks.
        out_dim: Required for regression and edge-prediction tasks.
            Defaults to ``1`` for ``edge_prediction`` if not specified.
        **kwargs: Forwarded to :func:`make_layer` and used for task options:
            ``pooling`` (graph readout, ``"mean"``/``"sum"``/``"max"``),
            ``edge_predictor_hidden`` (MLP hidden dim for edge prediction).

    Returns:
        ``nn.Module`` with forward signature::

            out = model(x, edge_index,
                        edge_features=None, edge_weight=None, batch=None)

    Raises:
        ValueError: Invalid task, invalid num_layers, or missing
            ``num_classes`` / ``out_dim`` for the chosen task.
        NotImplementedError: ``link_prediction`` — use ``edge_prediction``.
    """
    family = kwargs.pop("family", None)
    if family is not None:
        if layer is not None and str(layer).lower().strip() != str(family).lower().strip():
            raise ValueError(
                f"build_model received both layer={layer!r} and "
                f"family={family!r}; pass only one (they are aliases)."
            )
        layer = family
    if layer is None:
        raise ValueError("build_model requires layer= (or its alias family=).")
    if in_shape is None or hidden_shape is None or num_layers is None:
        raise ValueError(
            "build_model requires in_shape, hidden_shape, and num_layers."
        )

    task = task.lower().strip()
    layer = str(layer).lower().strip()
    layer = _FAMILY_ALIASES.get(layer, layer)
    in_shape = tuple(in_shape)
    hidden_shape = tuple(hidden_shape)

    if task == "link_prediction":
        raise NotImplementedError(
            "'link_prediction' is not implemented. "
            "Use task='edge_prediction' instead, which scores edges via an "
            "MLP on concatenated node embeddings."
        )

    if task not in _SUPPORTED_TASKS:
        raise ValueError(
            f"Unknown task {task!r}. "
            f"Supported: {', '.join(repr(t) for t in _SUPPORTED_TASKS)}."
        )

    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1; got {num_layers}")

    _classification = ("node_classification", "graph_classification")
    _regression = ("node_regression", "graph_regression", "edge_prediction")

    if task in _classification and num_classes is None:
        raise ValueError(
            f"num_classes is required for task={task!r}."
        )
    if task in ("node_regression", "graph_regression") and out_dim is None:
        raise ValueError(f"out_dim is required for task={task!r}.")

    # ── Model-level families (no per-layer make_layer stacking) ────────── #
    if layer == "set_transformer":
        if task == "edge_prediction":
            raise ValueError(
                "layer='set_transformer' does not support task="
                "'edge_prediction': the family does not consume edge_index. "
                "Use a 'given'-topology layer (conv/gat/sage/gin/linear)."
            )
        if len(hidden_shape) != 1:
            raise ValueError(
                "layer='set_transformer' expects hidden_shape=(embed_dim,) — "
                f"a 1-element tuple; got {hidden_shape}."
            )
        heads = kwargs.pop("heads", None)
        num_heads = kwargs.pop("num_heads", None)
        if heads is not None and num_heads is not None and heads != num_heads:
            raise ValueError(
                f"Pass only one of heads={heads!r} / num_heads={num_heads!r}."
            )
        resolved_heads = heads if heads is not None else (
            num_heads if num_heads is not None else 4
        )
        pool_attn_dropout = kwargs.pop("pool_attention_dropout", None)
        head_hidden = kwargs.pop("head_hidden_dim", None)
        model = TGraphXSetAttention(
            task=task,
            in_shape=in_shape,
            embed_dim=hidden_shape[0],
            num_layers=num_layers,
            num_heads=int(resolved_heads),
            ffn_dim=kwargs.pop("ffn_dim", None),
            dropout=float(kwargs.pop("dropout", 0.0)),
            attention_dropout=float(kwargs.pop("attention_dropout", 0.0)),
            num_classes=num_classes,
            out_dim=out_dim,
            pooling=str(kwargs.pop("pooling", "attention")),
            num_seeds=int(kwargs.pop("num_seeds", 1)),
            layer_norm=bool(kwargs.pop("layer_norm", True)),
            norm_order=str(kwargs.pop("norm_order", "pre")),
            activation=str(kwargs.pop("activation", "gelu")),
            pool_attention_dropout=(
                None if pool_attn_dropout is None else float(pool_attn_dropout)
            ),
            head_hidden_dim=(None if head_hidden is None else int(head_hidden)),
            encoder=kwargs.pop("encoder", None),
            encoder_config=kwargs.pop("encoder_config", None),
            on_edge_index=str(kwargs.pop("on_edge_index", "warn")),
        )
        return _tag_model(model, "set_transformer")

    pooling = str(kwargs.pop("pooling", "mean"))
    ep_hidden = int(kwargs.pop("edge_predictor_hidden", 64))

    # Build GNN layer stack
    layer_kwargs = kwargs  # remaining kwargs forwarded to make_layer
    gnn_layers: list[nn.Module] = []
    gnn_layers.append(make_layer(layer, in_shape, hidden_shape, **layer_kwargs))
    for _ in range(num_layers - 1):
        gnn_layers.append(make_layer(layer, hidden_shape, hidden_shape, **layer_kwargs))

    # Channel width after spatial pooling (always hidden_shape[0])
    hidden_dim = hidden_shape[0]

    # Edge prediction: GNN stack + EdgePredictor
    if task == "edge_prediction":
        ep_out = out_dim if out_dim is not None else 1
        predictor = EdgePredictor(
            in_dim=hidden_dim,
            hidden_dim=ep_hidden,
            out_dim=ep_out,
        )
        return _tag_model(
            _EdgePredictionModel(gnn_layers=gnn_layers, predictor=predictor),
            layer,
            **layer_kwargs,
        )

    # Node / graph classification and regression
    target_dim = num_classes if task in _classification else out_dim
    out_linear = nn.Linear(hidden_dim, target_dim)  # type: ignore[arg-type]
    return _tag_model(
        _FactoryGNNModel(
            gnn_layers=gnn_layers,
            task=task,
            out_linear=out_linear,
            pooling=pooling,
        ),
        layer,
        **layer_kwargs,
    )


# --------------------------------------------------------------------------- #
# Config-based construction                                                     #
# --------------------------------------------------------------------------- #

def build_model_from_config(
    path_or_dict: Union[str, Dict[str, Any]],
) -> nn.Module:
    """Build a model from a Python dict, JSON file, or YAML file.

    The config must have a top-level ``model`` key whose value is a dict
    with all required ``build_model`` arguments.

    Required keys inside ``model``:
        task, layer, in_shape, hidden_shape, num_layers

    Optional keys: any keyword accepted by ``build_model``.

    Example dict config::

        config = {
            "model": {
                "task": "graph_classification",
                "layer": "gat",
                "in_shape": [8, 4, 4],
                "hidden_shape": [16, 4, 4],
                "num_layers": 2,
                "num_classes": 3,
                "heads": 2,
                "residual": True,
                "dropout": 0.1,
            }
        }
        model = build_model_from_config(config)

    Args:
        path_or_dict: Python dict, path to a JSON file, or path to a YAML
            file (requires PyYAML).

    Returns:
        Configured ``nn.Module``.

    Raises:
        KeyError: Missing required key.
        ValueError: Unsupported file extension, invalid task/layer.
        ImportError: YAML file requested but PyYAML is not installed.
        TypeError: ``path_or_dict`` is not a dict or string.
    """
    if isinstance(path_or_dict, dict):
        config = path_or_dict
    elif isinstance(path_or_dict, str):
        path = path_or_dict
        if path.endswith(".json"):
            with open(path) as f:
                config = json.load(f)
        elif path.endswith((".yaml", ".yml")):
            if _yaml is None:
                raise ImportError(
                    "PyYAML is required to load YAML config files. "
                    "Install it with: pip install pyyaml"
                )
            with open(path) as f:
                config = _yaml.safe_load(f)
        else:
            raise ValueError(
                f"Unsupported config file extension for {path!r}. "
                f"Use '.json', '.yaml', or '.yml'."
            )
    else:
        raise TypeError(
            f"path_or_dict must be a dict or a file-path string; "
            f"got {type(path_or_dict).__name__}."
        )

    if "model" not in config:
        raise KeyError(
            "Config must have a top-level 'model' key. "
            f"Found keys: {list(config.keys())!r}"
        )

    cfg: Dict[str, Any] = dict(config["model"])

    required = ["task", "layer", "in_shape", "hidden_shape", "num_layers"]
    for key in required:
        if key not in cfg:
            raise KeyError(
                f"Config 'model' section is missing required key {key!r}. "
                f"Required keys: {required}"
            )

    cfg["in_shape"] = tuple(cfg["in_shape"])
    cfg["hidden_shape"] = tuple(cfg["hidden_shape"])
    cfg["num_layers"] = int(cfg["num_layers"])

    return build_model(**cfg)


__all__ = ["build_model", "build_model_from_config"]
