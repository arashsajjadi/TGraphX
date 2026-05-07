"""TGraphX: tensor-aware graph neural networks preserving spatial node feature layouts.

The package keeps multi-dimensional node features (e.g. ``[C, H, W]`` tensors)
intact through message passing, allowing CNN-style spatial reasoning to be
combined with GNN-style relational reasoning.

Common one-liner imports::

    from tgraphx import Graph, build_model, fit, CSVLogger, env_report
"""

# Keep this in sync with [project].version in pyproject.toml.
__version__ = "0.2.0"

# ── Core data structures ──────────────────────────────────────────────────────
from .core.graph import Graph, GraphBatch
from .core.dataloader import GraphDataset, GraphDataLoader
from .core.utils import load_config, get_device
from .core.graph_utils import (
    add_self_loops,
    coalesce_edges,
    is_undirected,
    make_undirected,
    remove_self_loops,
)

# ── GNN layers ────────────────────────────────────────────────────────────────
from .layers.base import TensorMessagePassingLayer, LinearMessagePassing
from .layers.conv_message import ConvMessagePassing
from .layers.attention_message import AttentionMessagePassing
from .layers.gat import TensorGATLayer
from .layers.sage import TensorGraphSAGELayer
from .layers.gin import TensorGINLayer

# ── Graph builders and patch helpers ─────────────────────────────────────────
from .graph_builders import (
    build_grid_graph,
    build_grid_graph_3d,
    build_fully_connected_graph,
    build_knn_graph,
    build_radius_graph,
    build_iou_graph,
    build_random_graph,
    patch_grid_shape,
    image_to_patches,
    volume_patch_grid_shape,
    volume_to_patches,
)

# ── Factories ─────────────────────────────────────────────────────────────────
from .layers.factory import make_layer
from .models.factory import build_model, build_model_from_config

# ── Model classes ─────────────────────────────────────────────────────────────
from .models.edge_predictor import EdgePredictor
from .models.regressors import NodeRegressor, GraphRegressor
from .models.graph_classifier import GraphClassifier
from .models.node_classifier import NodeClassifier

# ── Training utilities ────────────────────────────────────────────────────────
# These only import torch.nn — no heavy optional dependencies.
from .training import (
    set_seed,
    count_parameters,
    save_checkpoint,
    load_checkpoint,
    accuracy,
    mean_absolute_error,
    mean_squared_error,
    train_epoch,
    evaluate,
    fit,
)

# ── Metric logging ────────────────────────────────────────────────────────────
# CSVLogger: zero optional dependencies.
# TensorBoardLogger: the *class object* is imported here; tensorboard itself is
# loaded lazily inside TensorBoardLogger.__init__ only when instantiated.
# Importing tgraphx therefore does NOT require tensorboard to be installed.
from .tracking import CSVLogger, TensorBoardLogger, write_graph_stats

# ── Performance utilities ─────────────────────────────────────────────────────
from .performance import env_report, recommended_device, estimate_message_memory

__all__ = [
    "__version__",
    # Core
    "Graph",
    "GraphBatch",
    "GraphDataset",
    "GraphDataLoader",
    "load_config",
    "get_device",
    # Graph utilities
    "add_self_loops",
    "coalesce_edges",
    "is_undirected",
    "make_undirected",
    "remove_self_loops",
    # GNN layers
    "TensorMessagePassingLayer",
    "LinearMessagePassing",
    "ConvMessagePassing",
    "AttentionMessagePassing",
    "TensorGATLayer",
    "TensorGraphSAGELayer",
    "TensorGINLayer",
    # Graph builders
    "build_grid_graph",
    "build_grid_graph_3d",
    "build_fully_connected_graph",
    "build_knn_graph",
    "build_radius_graph",
    "build_iou_graph",
    "build_random_graph",
    # Patch helpers
    "patch_grid_shape",
    "image_to_patches",
    "volume_patch_grid_shape",
    "volume_to_patches",
    # Factories
    "make_layer",
    "build_model",
    "build_model_from_config",
    # Model classes
    "EdgePredictor",
    "NodeRegressor",
    "GraphRegressor",
    "GraphClassifier",
    "NodeClassifier",
    # Training utilities
    "set_seed",
    "count_parameters",
    "save_checkpoint",
    "load_checkpoint",
    "accuracy",
    "mean_absolute_error",
    "mean_squared_error",
    "train_epoch",
    "evaluate",
    "fit",
    # Metric logging
    "CSVLogger",
    "TensorBoardLogger",
    "write_graph_stats",
    # Performance utilities
    "env_report",
    "recommended_device",
    "estimate_message_memory",
]
