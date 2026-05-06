# File: models/cnn_gnn_model.py
import torch
import torch.nn as nn
from .cnn_encoder import CNNEncoder
from ..layers.conv_message import ConvMessagePassing

class CNN_GNN_Model(nn.Module):
    """
    A unified CNN-GNN model.

    Pipeline (all stages after graph construction are end-to-end differentiable):
      1. Optionally pre-processes raw node patches with a user-supplied pre-encoder.
      2. Passes each patch through a CNN encoder to obtain a spatial feature map [C, H, W].
      3. Applies a series of ConvMessagePassing GNN layers over a *user-provided* edge_index.
         Graph construction (i.e. which nodes are connected) is NOT part of this model and
         must be computed before calling forward().
      4. Averages over spatial dimensions, then optionally aggregates across nodes per graph,
         and classifies.

    Args:
        cnn_params (dict): Keyword arguments forwarded to CNNEncoder (excluding pre_encoder).
        gnn_in_dim (tuple): Per-node feature shape entering the first GNN layer, e.g. (C, H, W).
        gnn_hidden_dim (tuple): Per-node feature shape for all subsequent GNN layers.
        num_classes (int): Number of output classes.
        num_gnn_layers (int): Total number of ConvMessagePassing layers.
        gnn_dropout (float): Dropout probability applied inside each GNN aggregator.
        residual (bool): Enable skip connections inside each GNN layer.
        aggregator_params (dict | None): Extra keyword arguments for DeepCNNAggregator.
            ``dropout_prob`` defaults to ``gnn_dropout`` if not already set here.
        pre_encoder (nn.Module | None): Optional module applied before CNNEncoder.
        skip_cnn_to_classifier (bool): If True, add a residual connection from the CNN
            output to the GNN output before spatial pooling (shapes must match).
    """
    def __init__(self, cnn_params, gnn_in_dim, gnn_hidden_dim, num_classes, num_gnn_layers=2,
                 gnn_dropout=0.0, residual=False, aggregator_params=None, pre_encoder=None,
                 skip_cnn_to_classifier=False):
        super().__init__()
        cnn_params['pre_encoder'] = pre_encoder
        self.encoder = CNNEncoder(**cnn_params)
        self.skip_cnn_to_classifier = skip_cnn_to_classifier

        # Merge gnn_dropout into aggregator_params without mutating the caller's dict.
        agg_params = dict(aggregator_params) if aggregator_params else {}
        agg_params.setdefault('dropout_prob', gnn_dropout)

        layers = []
        layers.append(ConvMessagePassing(
            gnn_in_dim, gnn_hidden_dim, aggr='sum', use_edge_features=False,
            aggregator_params=agg_params, residual=residual,
        ))
        for _ in range(num_gnn_layers - 1):
            layers.append(ConvMessagePassing(
                gnn_hidden_dim, gnn_hidden_dim, aggr='sum', use_edge_features=False,
                aggregator_params=agg_params, residual=residual,
            ))
        self.gnn_layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(gnn_hidden_dim[0], num_classes)

    def forward(self, raw_node_data, edge_index, edge_features=None, batch=None):
        """
        Args:
            raw_node_data (Tensor): Pre-split node patches, shape [N, C_in, H_in, W_in].
            edge_index (LongTensor): Graph connectivity, shape [2, E].
            edge_features (Tensor | None): Optional edge features.
            batch (LongTensor | None): Graph membership vector [N]. Required for
                graph-level classification; omit for node-level use.

        Returns:
            Tensor: Class logits of shape [num_graphs, num_classes] when batch is given,
                    or [N, num_classes] otherwise.
        """
        cnn_out = self.encoder(raw_node_data)   # [N, C, H, W]
        x = cnn_out

        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_features)
            x = torch.relu(x)

        if self.skip_cnn_to_classifier and cnn_out.shape == x.shape:
            x = x + cnn_out

        # Average over spatial dims (H, W, …) if present.
        if x.dim() > 2:
            x = x.mean(dim=tuple(range(2, x.dim())))

        if batch is not None:
            num_graphs = batch.max().item() + 1
            pooled = torch.zeros(num_graphs, x.size(1), device=x.device)
            pooled = pooled.index_add(0, batch, x)
            counts = torch.zeros(num_graphs, device=x.device)
            ones = torch.ones(x.size(0), device=x.device)
            counts = counts.index_add(0, batch, ones).unsqueeze(1).clamp(min=1)
            return self.classifier(pooled / counts)
        return self.classifier(x)
