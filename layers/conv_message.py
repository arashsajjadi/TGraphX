import torch
import torch.nn as nn
from .base import TensorMessagePassingLayer


class ConvMessagePassing(TensorMessagePassingLayer):
    r"""Convolution-based message passing layer.

    Automatically selects a 2D or 3D convolution based on the input spatial dimensions.
    For each edge, the source and destination node features (and optionally edge features)
    are concatenated along the channel dimension and processed via a 1×1 convolution.
    """
    def __init__(self, in_shape, out_shape, aggr='sum', use_edge_features=False):
        super().__init__(in_shape, out_shape, aggr)
        self.use_edge_features = use_edge_features
        # in_shape is expected to be a tuple like (C, H, W) or (C, D, H, W)
        self.node_channels = in_shape[0]
        if self.use_edge_features:
            conv_in_channels = self.node_channels * 3  # src, dest, and edge features
        else:
            conv_in_channels = self.node_channels * 2  # src and dest only
        self.out_channels = out_shape[0]
        # Automatically choose between 2D and 3D convolution based on spatial dims
        if len(in_shape) == 3:
            # 2D spatial data: (C, H, W)
            self.conv = nn.Conv2d(conv_in_channels, self.out_channels, kernel_size=1)
        elif len(in_shape) == 4:
            # 3D spatial data: (C, D, H, W)
            self.conv = nn.Conv3d(conv_in_channels, self.out_channels, kernel_size=1)
        else:
            raise ValueError("ConvMessagePassing supports only 2D or 3D spatial node features.")

    def message(self, src, dest, edge_attr):
        # Assume src and dest are tensors of shape [E, C, ...]
        if self.use_edge_features and edge_attr is not None:
            msg_input = torch.cat([src, dest, edge_attr], dim=1)  # concatenate along channel dim
        else:
            msg_input = torch.cat([src, dest], dim=1)
        return self.conv(msg_input)

    def update(self, node_feature, aggregated_message):
        return aggregated_message
