# File: layers/conv_message.py
import torch
import torch.nn as nn
from .base import TensorMessagePassingLayer
from .aggregator import DeepCNNAggregator


class ConvMessagePassing(TensorMessagePassingLayer):
    """
    Convolution-based message passing layer that concatenates source and destination
    features (and optionally edge features) and processes them via a 1x1 convolution.
    It then uses a deep CNN aggregator (by default 4 layers) to process the aggregated messages.
    """

    def __init__(self, in_shape, out_shape, aggr='sum', use_edge_features=False, aggregator_params=None):
        super().__init__(in_shape, out_shape, aggr)
        self.use_edge_features = use_edge_features
        # Expect in_shape to be a tuple like (C, H, W)
        self.node_channels = in_shape[0]
        if self.use_edge_features:
            conv_in_channels = self.node_channels * 3  # for src, dest, and edge features
        else:
            conv_in_channels = self.node_channels * 2  # for src and dest only
        self.out_channels = out_shape[0]
        if len(in_shape) == 3:
            self.conv = nn.Conv2d(conv_in_channels, self.out_channels, kernel_size=1)
        elif len(in_shape) == 4:
            self.conv = nn.Conv3d(conv_in_channels, self.out_channels, kernel_size=1)
        else:
            raise ValueError("ConvMessagePassing supports only 2D or 3D spatial node features.")

        # Set up the deep CNN aggregator for the aggregation (default: 4-layer CNN)
        if aggregator_params is None:
            aggregator_params = {}
        self.aggregator = DeepCNNAggregator(in_channels=self.out_channels,
                                            out_channels=self.out_channels,
                                            **aggregator_params)

    def message(self, src, dest, edge_attr):
        if self.use_edge_features and edge_attr is not None:
            msg_input = torch.cat([src, dest, edge_attr], dim=1)
        else:
            msg_input = torch.cat([src, dest], dim=1)
        return self.conv(msg_input)

    def update(self, node_feature, aggregated_message):
        # Apply the deep CNN aggregator to the aggregated message.
        aggregated_message = self.aggregator(aggregated_message)
        return aggregated_message
