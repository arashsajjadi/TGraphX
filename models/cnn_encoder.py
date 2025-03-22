# File: cnn_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.safe_pool import SafeMaxPool2d  # Use our new safe pooling module


class ResidualBlock(nn.Module):
    def __init__(self, block, in_channels, out_channels, debug=False):
        super().__init__()
        self.block = block
        self.debug = debug
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x):
        out = self.block(x)
        if x.shape != out.shape:
            if self.debug:
                print("ResidualBlock shape mismatch:", x.shape, out.shape)
            x = self.proj(x)
        if self.debug:
            print("ResidualBlock output shape:", (x + out).shape)
        return x + out


class CNNEncoder(nn.Module):
    """
    A CNN encoder that converts raw node data (e.g. an image patch) into a feature map.
    When `return_feature_map` is True, the network returns a multi-dimensional output
    (e.g. [N, out_features, H, W]) using a 1x1 convolution instead of flattening.

    If a pre_encoder is provided, it is applied first and its output channel count is used
    as the input channel count for the CNN layers.
    """

    def __init__(self, in_channels, out_features, num_layers=5, hidden_channels=64,
                 dropout_prob=0.3, use_batchnorm=True, use_residual=True, pool_layers=2,
                 debug=False, return_feature_map=False, pre_encoder=None):
        super().__init__()
        self.return_feature_map = return_feature_map
        self.pre_encoder = pre_encoder  # Optional pre-encoder stage.
        self.debug = debug
        # Determine starting channels: if a pre_encoder is provided, use its out_channels.
        if self.pre_encoder is not None and hasattr(self.pre_encoder, "out_channels"):
            channels = self.pre_encoder.out_channels
        else:
            channels = in_channels

        layers = []
        for i in range(num_layers):
            conv = nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1)
            bn = nn.BatchNorm2d(hidden_channels) if use_batchnorm else nn.Identity()
            relu = nn.ReLU(inplace=True)
            dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()
            # Use safe pooling for early layers.
            if i < pool_layers:
                pool = SafeMaxPool2d(2)
                block = nn.Sequential(conv, bn, relu, dropout, pool)
            else:
                block = nn.Sequential(conv, bn, relu, dropout)
            if use_residual and i > 0:
                block = ResidualBlock(block, in_channels=channels, out_channels=hidden_channels, debug=debug)
            layers.append(block)
            channels = hidden_channels
        self.cnn = nn.Sequential(*layers)
        if self.return_feature_map:
            self.conv1x1 = nn.Conv2d(hidden_channels, out_features, kernel_size=1)
        else:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(hidden_channels, out_features)

    def forward(self, x):
        if self.pre_encoder is not None:
            x = self.pre_encoder(x)
        out = self.cnn(x)
        if self.debug:
            print("CNN output shape:", out.shape)
        if self.return_feature_map:
            out = self.conv1x1(out)
            if self.debug:
                print("Feature map shape after 1x1 conv:", out.shape)
            return out
        else:
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            if self.debug:
                print("Flattened encoder output shape:", out.shape)
            out = self.fc(out)
            return out
