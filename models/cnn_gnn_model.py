# File: models/cnn_gnn_model.py
import torch
import torch.nn as nn
from models.cnn_encoder import CNNEncoder
from layers.conv_message import ConvMessagePassing

class CNN_GNN_Model(nn.Module):
    """
    A unified CNN‑GNN model that:
      1. Optionally pre-processes raw node image patches with a user‑supplied pre‑encoder.
      2. Passes the (pre‑processed) patches through a CNN encoder to obtain a multi-dimensional feature map.
      3. Applies a series of GNN layers (using ConvMessagePassing) whose aggregation is performed
         by a deep CNN aggregator.
      4. Pools (averages over spatial dimensions and nodes) and then classifies.
    """
    def __init__(self, cnn_params, gnn_in_dim, gnn_hidden_dim, num_classes, num_gnn_layers=2,
                 gnn_dropout=0.0, residual=False, aggregator_params=None, pre_encoder=None):
        super().__init__()
        # Pass the optional pre_encoder to the CNN encoder.
        cnn_params['pre_encoder'] = pre_encoder
        self.encoder = CNNEncoder(**cnn_params)
        layers = []
        # Build the GNN layers.
        layers.append(ConvMessagePassing(gnn_in_dim, gnn_hidden_dim, aggr='sum', use_edge_features=False,
                                         aggregator_params=aggregator_params))
        for _ in range(num_gnn_layers - 1):
            layers.append(ConvMessagePassing(gnn_hidden_dim, gnn_hidden_dim, aggr='sum', use_edge_features=False,
                                             aggregator_params=aggregator_params))
        self.gnn_layers = nn.ModuleList(layers)
        # Classifier works on flattened node features (after spatial pooling).
        self.classifier = nn.Linear(gnn_hidden_dim[0], num_classes)

    def forward(self, raw_node_data, edge_index, edge_features=None, batch=None):
        # Process raw node data through the CNN encoder (and optional pre-encoder).
        node_features = self.encoder(raw_node_data)
        x = node_features
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_features)
            x = torch.relu(x)
        # If x is multi-dimensional (e.g. [N, C, H, W]), average over spatial dimensions.
        if x.dim() > 2:
            x = x.mean(dim=tuple(range(2, x.dim())))
        # For graph-level tasks, aggregate nodes by batch.
        if batch is not None:
            num_graphs = batch.max().item() + 1
            pooled = torch.zeros(num_graphs, x.size(1), device=x.device)
            pooled = pooled.index_add(0, batch, x)
            counts = torch.zeros(num_graphs, device=x.device)
            ones = torch.ones(x.size(0), device=x.device)
            counts = counts.index_add(0, batch, ones).unsqueeze(1)
            pooled = pooled / counts.clamp(min=1)
            logits = self.classifier(pooled)
            return logits
        return self.classifier(x)
