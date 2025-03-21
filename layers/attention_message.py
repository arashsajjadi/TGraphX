import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import TensorMessagePassingLayer


class AttentionMessagePassing(TensorMessagePassingLayer):
    r"""Attention-based message passing layer.

    This layer computes attention coefficients for each edge based on projected and
    flattened node features. Optionally, edge features can be included.
    """
    def __init__(self, in_shape, out_shape, aggr='sum', use_edge_features=False):
        super().__init__(in_shape, out_shape, aggr)
        self.use_edge_features = use_edge_features
        # Flatten the input tensor shape (excluding batch) to a single dimension.
        self.flatten_dim = 1
        for d in in_shape:
            self.flatten_dim *= d
        self.query_proj = nn.Linear(self.flatten_dim, out_shape[0])
        self.key_proj = nn.Linear(self.flatten_dim, out_shape[0])
        self.value_proj = nn.Linear(self.flatten_dim, out_shape[0])
        if self.use_edge_features:
            self.edge_proj = nn.Linear(self.flatten_dim, out_shape[0])

    def message(self, src, dest, edge_attr):
        # Flatten node features (for each edge)
        src_flat = src.view(src.size(0), -1)
        dest_flat = dest.view(dest.size(0), -1)
        query = self.query_proj(dest_flat)  # query from destination node
        key = self.key_proj(src_flat)         # key from source node
        value = self.value_proj(src_flat)       # value from source node
        if self.use_edge_features and edge_attr is not None:
            edge_flat = edge_attr.view(edge_attr.size(0), -1)
            edge_value = self.edge_proj(edge_flat)
            value = value + edge_value
        # Compute raw attention scores (dot product)
        attn_scores = (query * key).sum(dim=-1, keepdim=True)
        attn_scores = F.leaky_relu(attn_scores)
        # Concatenate attention score and value so that aggregation can compute a weighted sum.
        return torch.cat([attn_scores, value], dim=-1)

    def aggregate(self, messages, edge_index, num_nodes):
        r"""Aggregate messages using attention.

        The first column of `messages` contains the raw attention scores, and the remaining
        columns contain the corresponding value vectors. A softmax is computed per destination node.
        """
        target = edge_index[1]
        attn_scores = messages[:, 0:1]  # shape: [E, 1]
        values = messages[:, 1:]        # shape: [E, out_channels]
        # Compute normalized attention per destination node.
        exp_scores = torch.exp(attn_scores)
        denom = torch.zeros(num_nodes, 1, device=messages.device)
        denom = denom.index_add(0, target, exp_scores)
        norm_attn = exp_scores / (denom[target] + 1e-16)
        weighted_values = norm_attn * values
        aggregated = torch.zeros(num_nodes, values.size(1), device=messages.device)
        aggregated = aggregated.index_add(0, target, weighted_values)
        return aggregated

    def update(self, node_feature, aggregated_message):
        return aggregated_message
