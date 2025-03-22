import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import TensorMessagePassingLayer


class AttentionMessagePassing(TensorMessagePassingLayer):
    def __init__(self, in_shape, out_shape, aggr='sum', use_edge_features=False,
                 dropout_prob=0.0, residual=False, use_batchnorm=False):
        super().__init__(in_shape, out_shape, aggr, dropout_prob=dropout_prob,
                         residual=residual, use_batchnorm=use_batchnorm)
        self.use_edge_features = use_edge_features
        self.out_dim = out_shape[0]
        # Scaling factor: 1/sqrt(d)
        self.scale = math.sqrt(self.out_dim)

        # If the input is multi-dimensional (e.g. (C, H, W)) then use 1x1 convolutions;
        # otherwise (if it's a vector) use Linear layers.
        if len(in_shape) > 1:
            # in_shape example: (C, H, W)
            self.query_proj = nn.Conv2d(in_shape[0], self.out_dim, kernel_size=1)
            self.key_proj = nn.Conv2d(in_shape[0], self.out_dim, kernel_size=1)
            self.value_proj = nn.Conv2d(in_shape[0], self.out_dim, kernel_size=1)
            if self.use_edge_features:
                self.edge_proj = nn.Conv2d(in_shape[0], self.out_dim, kernel_size=1)
        else:
            self.flatten_dim = in_shape[0]
            self.query_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.key_proj = nn.Linear(self.flatten_dim, self.out_dim)
            self.value_proj = nn.Linear(self.flatten_dim, self.out_dim)
            if self.use_edge_features:
                self.edge_proj = nn.Linear(self.flatten_dim, self.out_dim)

    def message(self, src, dest, edge_attr):
        # Flatten node features for each edge
        src_flat = src.view(src.size(0), -1)
        dest_flat = dest.view(dest.size(0), -1)
        # Compute projections
        query = self.query_proj(dest_flat)
        key = self.key_proj(src_flat)
        value = self.value_proj(src_flat)
        if self.use_edge_features and edge_attr is not None:
            edge_flat = edge_attr.view(edge_attr.size(0), -1)
            edge_value = self.edge_proj(edge_flat)
            value = value + edge_value
        # Compute raw attention scores with scaling
        raw_scores = (query * key).sum(dim=-1, keepdim=True) / self.scale
        attn_scores = F.leaky_relu(raw_scores)
        # Instead of concatenating, apply a sigmoid to get attention coefficients
        attn = torch.sigmoid(attn_scores)
        # Weight the value with the attention coefficients.
        # This keeps the output dimension the same as out_dim.
        return value * attn

