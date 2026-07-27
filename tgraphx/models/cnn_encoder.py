# File: cnn_encoder.py
from __future__ import annotations

import logging

import torch.nn as nn

from .._compat import LEGACY_CNN_DROPOUT_PROB, resolve_dropout_prob
from ..layers.safe_pool import SafeMaxPool2d

_log = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    def __init__(self, block, in_channels, out_channels, debug=False):
        super().__init__()
        self.block = block
        self.debug = debug
        # stride=1 so the projection only adjusts channels without changing spatial dims.
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.block(x)
        if x.shape != out.shape:
            _log.debug("ResidualBlock: projecting skip from %s to %s", x.shape, out.shape)
            x = self.proj(x)
        return x + out


class CNNEncoder(nn.Module):
    """
    A CNN encoder that converts raw node data (e.g. an image patch) into a feature map.
    When `return_feature_map` is True, the network returns a multi-dimensional output
    (e.g. [N, out_features, H, W]) using a 1x1 convolution instead of flattening.

    If a pre_encoder is provided, it is applied first and its output channel count is used
    as the input channel count for the CNN layers.

    Regularization is explicit (v1.5.0):
        ``dropout_prob`` defaults to ``0.0`` (no dropout).  TGraphX <= 1.4.2
        silently used ``0.3``; constructing without an explicit value now
        emits :class:`tgraphx.DropoutDefaultChangeWarning`.  Use
        :meth:`CNNEncoder.legacy` to reconstruct the pre-1.5 defaults
        intentionally.  ``use_batchnorm`` / ``use_residual`` keep their
        historical defaults (``True``) because changing them would alter
        the parameter layout of existing checkpoints; both are surfaced in
        ``repr()`` and :meth:`config`.
    """

    def __init__(self, in_channels, out_features, num_layers=5, hidden_channels=64,
                 dropout_prob=None, use_batchnorm=True, use_residual=True, pool_layers=2,
                 debug=False, return_feature_map=False, pre_encoder=None):
        super().__init__()
        dropout_prob = resolve_dropout_prob(dropout_prob, owner="CNNEncoder")
        self.in_channels = in_channels
        self.out_features = out_features
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dropout_prob = dropout_prob
        self.use_batchnorm = use_batchnorm
        self.use_residual = use_residual
        self.pool_layers = pool_layers
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

    @classmethod
    def legacy(cls, in_channels, out_features, **kwargs):
        """Construct with the TGraphX <= 1.4.2 implicit defaults, explicitly.

        Sets ``dropout_prob=0.3``, ``use_batchnorm=True``, ``use_residual=True``
        unless overridden in ``kwargs``.  No warning is emitted: the caller
        is opting into the legacy behaviour by name.
        """
        kwargs.setdefault("dropout_prob", LEGACY_CNN_DROPOUT_PROB)
        kwargs.setdefault("use_batchnorm", True)
        kwargs.setdefault("use_residual", True)
        return cls(in_channels, out_features, **kwargs)

    def config(self) -> dict:
        """Exact constructor configuration (excluding ``pre_encoder``).

        ``CNNEncoder(**enc.config())`` reconstructs an architecturally
        identical encoder with the same effective dropout probability.
        """
        return {
            "in_channels": self.in_channels,
            "out_features": self.out_features,
            "num_layers": self.num_layers,
            "hidden_channels": self.hidden_channels,
            "dropout_prob": self.dropout_prob,
            "use_batchnorm": self.use_batchnorm,
            "use_residual": self.use_residual,
            "pool_layers": self.pool_layers,
            "return_feature_map": self.return_feature_map,
        }

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_features={self.out_features}, "
            f"num_layers={self.num_layers}, hidden_channels={self.hidden_channels}, "
            f"dropout_prob={self.dropout_prob}, use_batchnorm={self.use_batchnorm}, "
            f"use_residual={self.use_residual}, pool_layers={self.pool_layers}, "
            f"return_feature_map={self.return_feature_map}"
        )

    def forward(self, x):
        if self.pre_encoder is not None:
            x = self.pre_encoder(x)
        out = self.cnn(x)
        _log.debug("CNN output shape: %s", out.shape)
        if self.return_feature_map:
            out = self.conv1x1(out)
            _log.debug("Feature map shape after 1x1 conv: %s", out.shape)
            return out
        else:
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            _log.debug("Flattened encoder output shape: %s", out.shape)
            out = self.fc(out)
            return out
