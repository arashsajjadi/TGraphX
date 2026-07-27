"""High-level model classes that combine CNN encoders with GNN message passing."""

from .cnn_encoder import CNNEncoder
from .cnn_gnn_model import CNN_GNN_Model
from .graph_classifier import GraphClassifier
from .node_classifier import NodeClassifier
from .pre_encoder import PreEncoder
from .edge_predictor import EdgePredictor
from .regressors import NodeRegressor, GraphRegressor
from .set_transformer import (
    AttentionPooling,
    SetAttentionBlock,
    SetTransformerModel,
)
from .topology import (
    TOPOLOGY_SOURCES,
    TopologyIgnoredWarning,
    topology_source_of,
)
from .factory import build_model, build_model_from_config

__all__ = [
    "CNNEncoder",
    "CNN_GNN_Model",
    "GraphClassifier",
    "NodeClassifier",
    "PreEncoder",
    "EdgePredictor",
    "NodeRegressor",
    "GraphRegressor",
    "SetTransformerModel",
    "SetAttentionBlock",
    "AttentionPooling",
    "TOPOLOGY_SOURCES",
    "TopologyIgnoredWarning",
    "topology_source_of",
    "build_model",
    "build_model_from_config",
]
