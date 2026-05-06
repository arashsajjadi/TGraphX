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
from .layers.base import TensorMessagePassingLayer, LinearMessagePassing
from .layers.conv_message import ConvMessagePassing
from .layers.attention_message import AttentionMessagePassing
from .layers.gat import TensorGATLayer
from .layers.sage import TensorGraphSAGELayer
from .layers.gin import TensorGINLayer

__all__ = [
    "__version__",
    "Graph",
    "GraphBatch",
    "GraphDataset",
    "GraphDataLoader",
    "load_config",
    "get_device",
    "TensorMessagePassingLayer",
    "LinearMessagePassing",
    "ConvMessagePassing",
    "AttentionMessagePassing",
    "TensorGATLayer",
    "TensorGraphSAGELayer",
    "TensorGINLayer",
]
