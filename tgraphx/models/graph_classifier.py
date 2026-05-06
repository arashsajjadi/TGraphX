import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.conv_message import ConvMessagePassing

class GraphClassifier(nn.Module):
    r"""Graph classification model that applies message passing followed by a graph-level pooling.

    Args:
        in_shape (tuple): Shape of input node features (e.g., (C,) or (C, H, W), etc.).
        hidden_shape (tuple): Shape of hidden representations.
        num_classes (int): Number of output classes.
        num_layers (int): Number of message passing layers.
        aggr (str): Aggregation type ('sum' or 'mean').
        use_edge_features (bool): Whether to include edge features.
        pooling (str): Pooling method for graph-level readout ('mean', 'sum', or 'max').
    """

    def __init__(self, in_shape, hidden_shape, num_classes, num_layers=2, aggr='sum', use_edge_features=False,
                 pooling='mean'):
        super(GraphClassifier, self).__init__()
        layers = []
        # For volumetric data, always use ConvMessagePassing.
        msg_layer = ConvMessagePassing
        # First layer: from input to hidden
        layers.append(msg_layer(in_shape, hidden_shape, aggr=aggr, use_edge_features=use_edge_features))
        # Additional layers
        for _ in range(num_layers - 1):
            layers.append(msg_layer(hidden_shape, hidden_shape, aggr=aggr, use_edge_features=use_edge_features))
        self.layers = nn.ModuleList(layers)
        self.pooling = pooling
        # Final classifier expects input features equal to the channel dimension (hidden_shape[0])
        self.classifier = nn.Linear(hidden_shape[0], num_classes)

    def forward(self, node_features, edge_index, edge_features=None, batch=None):
        """
        Args:
            node_features (Tensor): [N, ...] node features.
            edge_index (LongTensor): [2, E] edge indices.
            edge_features (Tensor | None): [E, ...] edge features. Default: None.
            batch (LongTensor): [N] tensor assigning each node to a graph in the batch.
                Required; pass explicitly.

        Returns:
            Tensor: Logits for each graph in the batch, shape [num_graphs, num_classes].
        """
        if batch is None:
            raise ValueError(
                "batch is required for graph-level pooling. "
                "Pass a LongTensor of shape [N] mapping each node to its graph index."
            )
        x = node_features
        for layer in self.layers:
            x = layer(x, edge_index, edge_features)
            x = F.relu(x)
        # Graph-level pooling
        num_graphs = batch.max().item() + 1
        if self.pooling == 'mean':
            pooled = torch.zeros(num_graphs, *x.shape[1:], device=x.device)
            count = torch.zeros(num_graphs, device=x.device)
            pooled = pooled.index_add(0, batch, x)
            ones = torch.ones(x.size(0), device=x.device)
            count = count.index_add(0, batch, ones)
            count = count.view(num_graphs, *([1] * (x.ndim - 1)))
            pooled = pooled / count.clamp(min=1)
        elif self.pooling == 'sum':
            pooled = torch.zeros(num_graphs, *x.shape[1:], device=x.device)
            pooled = pooled.index_add(0, batch, x)
        elif self.pooling == 'max':
            # Vectorised max pooling via scatter_reduce_ to avoid a Python loop.
            x_flat = x.flatten(1)                               # [N, F]
            idx = batch.unsqueeze(1).expand_as(x_flat)         # [N, F]
            pooled_flat = x_flat.new_full((num_graphs, x_flat.size(1)), float('-inf'))
            pooled_flat.scatter_reduce_(0, idx, x_flat, reduce='amax', include_self=True)
            # Guard: graphs with no nodes keep -inf; replace with 0.
            pooled_flat = pooled_flat.masked_fill(
                torch.isinf(pooled_flat) & (pooled_flat < 0), 0.0
            )
            pooled = pooled_flat.view(num_graphs, *x.shape[1:])
        else:
            raise ValueError("Invalid pooling type. Choose from 'mean', 'sum', or 'max'.")

        # Global average pooling over spatial dimensions if they exist.
        if pooled.ndim > 2:
            pooled = pooled.mean(dim=list(range(2, pooled.ndim)))
            # Now pooled has shape [num_graphs, channels]
        logits = self.classifier(pooled)
        return logits
