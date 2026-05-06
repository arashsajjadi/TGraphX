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

        # For spatial inputs (e.g. [E, C, H, W]), we use 1x1 convolutions
        # to compute projections while preserving H and W.
        if len(in_shape) > 1:
            self.query_proj = nn.Conv2d(in_shape[0], self.out_dim, kernel_size=1)
            self.key_proj = nn.Conv2d(in_shape[0], self.out_dim, kernel_size=1)
            self.value_proj = nn.Conv2d(in_shape[0], self.out_dim, kernel_size=1)
            if self.use_edge_features:
                self.edge_proj = nn.Conv2d(in_shape[0], self.out_dim, kernel_size=1)
        else:
            # This branch is for vector inputs; not used for spatial maps.
            self.flatten_dim = in_shape[0]
            self.query_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.key_proj = nn.Linear(self.flatten_dim, self.out_dim)
            self.value_proj = nn.Linear(self.flatten_dim, self.out_dim)
            if self.use_edge_features:
                self.edge_proj = nn.Linear(self.flatten_dim, self.out_dim)

    def message(self, src, dest, edge_attr):
        # Do not flatten: work directly on spatial feature maps.
        # src and dest are expected to have shape [E, C, H, W]
        query = self.query_proj(dest)    # [E, out_dim, H, W]
        key = self.key_proj(src)           # [E, out_dim, H, W]
        value = self.value_proj(src)       # [E, out_dim, H, W]

        if self.use_edge_features and edge_attr is not None:
            edge_value = self.edge_proj(edge_attr)  # Spatial processing of edge features.
            value = value + edge_value

        # Compute attention scores spatially, summing over the channel dimension.
        raw_scores = (query * key).sum(dim=1, keepdim=True) / self.scale  # [E, 1, H, W]
        attn_scores = F.leaky_relu(raw_scores)
        attn = torch.sigmoid(attn_scores)  # Spatial attention coefficients.
        return value * attn  # Element-wise multiplication over spatial dimensions.
