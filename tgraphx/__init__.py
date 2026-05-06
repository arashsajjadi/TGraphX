"""TGraphX: tensor-aware graph neural networks preserving spatial node feature layouts.

The package keeps multi-dimensional node features (e.g. ``[C, H, W]`` tensors)
intact through message passing, allowing CNN-style spatial reasoning to be
combined with GNN-style relational reasoning.
"""

# Keep this in sync with [project].version in pyproject.toml.
__version__ = "0.1.0"

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
from .layers.base import TensorMessagePassingLayer, LinearMessagePassing
from .layers.conv_message import ConvMessagePassing
from .layers.attention_message import AttentionMessagePassing
from .layers.gat import TensorGATLayer
from .layers.sage import TensorGraphSAGELayer
from .layers.gin import TensorGINLayer
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
from .layers.factory import make_layer
from .models.factory import build_model, build_model_from_config
from .models.edge_predictor import EdgePredictor
from .models.regressors import NodeRegressor, GraphRegressor

__all__ = [
    "__version__",
    "Graph",
    "GraphBatch",
    "GraphDataset",
    "GraphDataLoader",
    "load_config",
    "get_device",
    "add_self_loops",
    "coalesce_edges",
    "is_undirected",
    "make_undirected",
    "remove_self_loops",
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
    # New model classes
    "EdgePredictor",
    "NodeRegressor",
    "GraphRegressor",
]
