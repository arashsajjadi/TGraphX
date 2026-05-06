"""Deep CNN aggregator with rank-aware (2-D / 3-D) plumbing.

Used by :class:`tgraphx.layers.ConvMessagePassing` after destination-wise
aggregation: the aggregator processes each node's accumulated message tensor
through a small CNN block while preserving its spatial layout.

Backward-compatible: ``spatial_rank`` defaults to 2 (``Conv2d`` /
``BatchNorm2d`` / ``Dropout2d``).  Pass ``spatial_rank=3`` to operate on
``[N, C, D, H, W]`` tensors using ``Conv3d`` / ``BatchNorm3d`` /
``Dropout3d``.  Only ranks 2 and 3 are supported — vector features go
through ``LinearMessagePassing`` instead.
"""

from __future__ import annotations

import torch.nn as nn

from ._dim import batchnorm, conv1x1, dropout_nd, validate_spatial_rank


class DeepCNNAggregator(nn.Module):
    """A small CNN block for per-node message tensors.

    Args:
        in_channels: Channel count of the aggregated message tensor.
        out_channels: Output channel count.
        num_layers: Number of 3×3 (or 3×3×3) convolutional layers.
        hidden_channels: Channels in the intermediate stages.  Defaults to
            ``out_channels``.
        dropout_prob: Probability for the rank-matching dropout layer.
        use_batchnorm: Whether to insert BatchNorm after each conv.
        spatial_rank: ``2`` (default) for ``[N, C, H, W]`` inputs or ``3``
            for ``[N, C, D, H, W]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 4,
        hidden_channels: int | None = None,
        dropout_prob: float = 0.3,
        use_batchnorm: bool = True,
        spatial_rank: int = 2,
    ) -> None:
        super().__init__()
        validate_spatial_rank(spatial_rank)
        if hidden_channels is None:
            hidden_channels = out_channels

        if spatial_rank == 2:
            Conv = nn.Conv2d
        else:
            Conv = nn.Conv3d

        layers: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(num_layers):
            layers.append(
                Conv(current_channels, hidden_channels, kernel_size=3, padding=1)
            )
            if use_batchnorm:
                layers.append(batchnorm(spatial_rank, hidden_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(dropout_nd(spatial_rank, dropout_prob))
            current_channels = hidden_channels
        if current_channels != out_channels:
            layers.append(conv1x1(spatial_rank, current_channels, out_channels))
        self.cnn = nn.Sequential(*layers)
        self.spatial_rank = spatial_rank

    def forward(self, x):
        return self.cnn(x)
