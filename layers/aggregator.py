# File: layers/aggregator.py
import torch
import torch.nn as nn


class DeepCNNAggregator(nn.Module):
    """
    A deep CNN aggregator that processes a spatial tensor [N, C, H, W] using a series of
    convolutional layers, batch normalization, ReLU activations, and dropout.
    This module is designed to aggregate messages from the GNN while preserving spatial details.

    Parameters:
      - in_channels: Number of input channels.
      - out_channels: Desired number of output channels.
      - num_layers: Number of convolutional layers (default: 4).
      - hidden_channels: Number of channels in intermediate layers (if None, defaults to out_channels).
      - dropout_prob: Dropout probability.
      - use_batchnorm: Whether to use BatchNorm.
    """

    def __init__(self, in_channels, out_channels, num_layers=4, hidden_channels=None,
                 dropout_prob=0.3, use_batchnorm=True):
        super().__init__()
        layers = []
        if hidden_channels is None:
            hidden_channels = out_channels
        current_channels = in_channels
        for i in range(num_layers):
            conv = nn.Conv2d(current_channels, hidden_channels, kernel_size=3, padding=1)
            layers.append(conv)
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout2d(p=dropout_prob))
            current_channels = hidden_channels
        # If necessary, adjust the channel count at the end.
        if current_channels != out_channels:
            layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=1))
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        # Process the entire spatial feature map.
        return self.cnn(x)
