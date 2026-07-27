"""Deep CNN aggregator with rank-aware (2-D / 3-D) plumbing.

Used by :class:`tgraphx.layers.ConvMessagePassing` after destination-wise
aggregation: the aggregator processes each node's accumulated message tensor
through a small CNN block while preserving its spatial layout.

Backward-compatible: ``spatial_rank`` defaults to 2 (``Conv2d`` /
``BatchNorm2d`` / ``Dropout2d``).  Pass ``spatial_rank=3`` to operate on
``[N, C, D, H, W]`` tensors using ``Conv3d`` / ``BatchNorm3d`` /
``Dropout3d``.  Only ranks 2 and 3 are supported — vector features go
through ``LinearMessagePassing`` instead.

Regularization is explicit (v1.5.0): ``dropout_prob`` defaults to ``0.0``.
TGraphX <= 1.4.2 silently used ``0.3``; constructing without an explicit
value now emits :class:`tgraphx.DropoutDefaultChangeWarning`.  Use
:meth:`DeepCNNAggregator.legacy` to reconstruct the pre-1.5 defaults
intentionally.  ``use_batchnorm`` keeps its historical default (``True``)
because changing it would alter the parameter layout of existing
checkpoints; note that its usefulness is graph-density dependent — it can
help on dense graphs but hurt when many nodes have zero incoming edges.
"""

from __future__ import annotations

import torch.nn as nn

from .._compat import LEGACY_CNN_DROPOUT_PROB, resolve_dropout_prob
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
            Defaults to ``0.0`` since v1.5.0 (previously a silent ``0.3``);
            omitting it emits :class:`tgraphx.DropoutDefaultChangeWarning`.
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
        dropout_prob: float | None = None,
        use_batchnorm: bool = True,
        spatial_rank: int = 2,
    ) -> None:
        super().__init__()
        dropout_prob = resolve_dropout_prob(dropout_prob, owner="DeepCNNAggregator")
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
            layers.append(
                dropout_nd(spatial_rank, dropout_prob) if dropout_prob > 0
                else nn.Identity()
            )
            current_channels = hidden_channels
        if current_channels != out_channels:
            layers.append(conv1x1(spatial_rank, current_channels, out_channels))
        self.cnn = nn.Sequential(*layers)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dropout_prob = dropout_prob
        self.use_batchnorm = use_batchnorm
        self.spatial_rank = spatial_rank

    @classmethod
    def legacy(cls, in_channels: int, out_channels: int, **kwargs) -> "DeepCNNAggregator":
        """Construct with the TGraphX <= 1.4.2 implicit defaults, explicitly.

        Sets ``dropout_prob=0.3`` and ``use_batchnorm=True`` unless
        overridden in ``kwargs``.  No warning is emitted: the caller is
        opting into the legacy behaviour by name.
        """
        kwargs.setdefault("dropout_prob", LEGACY_CNN_DROPOUT_PROB)
        kwargs.setdefault("use_batchnorm", True)
        return cls(in_channels, out_channels, **kwargs)

    def config(self) -> dict:
        """Exact constructor configuration for deterministic reconstruction."""
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "num_layers": self.num_layers,
            "hidden_channels": self.hidden_channels,
            "dropout_prob": self.dropout_prob,
            "use_batchnorm": self.use_batchnorm,
            "spatial_rank": self.spatial_rank,
        }

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"num_layers={self.num_layers}, hidden_channels={self.hidden_channels}, "
            f"dropout_prob={self.dropout_prob}, use_batchnorm={self.use_batchnorm}, "
            f"spatial_rank={self.spatial_rank}"
        )

    def forward(self, x):
        return self.cnn(x)
