import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.base import LinearMessagePassing


class NodeClassifier(nn.Module):
    r"""Node classification model using a stack of GNN message passing layers.

    Args:
        in_shape (tuple): Shape of input node features (e.g., (C,)).
        hidden_shape (tuple): Shape of hidden representations.
        num_classes (int): Number of classes for prediction.
        num_layers (int): Total number of message passing layers.
        aggr (str): Aggregation type ('sum' or 'mean').
        use_edge_features (bool): Whether to use edge features in message computations.
    """
    def __init__(self, in_shape, hidden_shape, num_classes, num_layers=2, aggr='sum', use_edge_features=False):
        super(NodeClassifier, self).__init__()
        layers = []
        # First layer: from input to hidden
        layers.append(LinearMessagePassing(in_shape, hidden_shape, aggr=aggr, use_edge_features=use_edge_features))
        # Intermediate layers
        for _ in range(num_layers - 2):
            layers.append(LinearMessagePassing(hidden_shape, hidden_shape, aggr=aggr, use_edge_features=use_edge_features))
        # Final layer: from hidden to num_classes (output as logits)
        layers.append(LinearMessagePassing(hidden_shape, (num_classes,), aggr=aggr, use_edge_features=use_edge_features))
        self.layers = nn.ModuleList(layers)

    def forward(self, node_features, edge_index, edge_features=None):
        x = node_features
        for layer in self.layers:
            x = layer(x, edge_index, edge_features)
            x = F.relu(x)
        return x
