"""Convolution-based message passing for 2-D and 3-D spatial node features.

Builds a per-edge message by concatenating ``[h_src, h_dst]`` (and
optionally ``[h_src, h_dst, e_ij]``) along the channel axis, projecting via
a 1×1 (or 1×1×1) convolution, and aggregating with sum/mean/max.  The
aggregated per-node tensor is then passed through ``DeepCNNAggregator``,
which is itself rank-aware.

Supported ``in_shape`` / ``out_shape`` ranks:

* ``len(...) == 3`` → ``(C, H, W)``    — 2-D spatial.
* ``len(...) == 4`` → ``(C, D, H, W)`` — 3-D volumetric.

Edge feature contract (preserved from earlier phases):

When ``use_edge_features=True`` the edge tensor is concatenated to the
message-conv input.  Its **channel count must equal the node channel
count** (``in_shape[0]``) and it must match the spatial rank of the node
features.  This is the same constraint as the 2-D case; for 3-D we extend
it to ``[E, C, D, H, W]``.  Vector edge tensors are not accepted by
``ConvMessagePassing`` — use ``TensorGAT/SAGE/GINLayer`` for those.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .aggregator import DeepCNNAggregator
from .base import TensorMessagePassingLayer


class ConvMessagePassing(TensorMessagePassingLayer):
    """Convolutional message passing over 2-D or 3-D spatial node features."""

    def __init__(
        self,
        in_shape,
        out_shape,
        aggr: str = "sum",
        use_edge_features: bool = False,
        aggregator_params: dict | None = None,
        residual: bool = False,
    ) -> None:
        super().__init__(in_shape, out_shape, aggr, residual=residual)
        self.use_edge_features = use_edge_features
        self.node_channels = in_shape[0]
        self.out_channels = out_shape[0]

        # Choose 2-D vs 3-D based on the spatial rank encoded in ``in_shape``.
        if len(in_shape) == 3:
            spatial_rank = 2
            Conv = nn.Conv2d
        elif len(in_shape) == 4:
            spatial_rank = 3
            Conv = nn.Conv3d
        else:
            raise ValueError(
                "ConvMessagePassing supports only 2-D ([C, H, W]) or 3-D "
                f"([C, D, H, W]) spatial node features; got in_shape={tuple(in_shape)}."
            )
        if len(out_shape) != len(in_shape):
            raise ValueError(
                f"in_shape and out_shape must have the same rank; got "
                f"{tuple(in_shape)} vs {tuple(out_shape)}."
            )
        self.spatial_rank = spatial_rank

        if self.use_edge_features:
            conv_in_channels = self.node_channels * 3  # src, dest, edge
        else:
            conv_in_channels = self.node_channels * 2  # src, dest
        self.conv = Conv(conv_in_channels, self.out_channels, kernel_size=1)

        if aggregator_params is None:
            aggregator_params = {}
        self.aggregator = DeepCNNAggregator(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            spatial_rank=spatial_rank,
            **aggregator_params,
        )

    # ------------------------------------------------------------------ #
    # Message / Update                                                     #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Chunked forward (optional memory-saving path)                       #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        node_features,
        edge_index,
        edge_features=None,
        edge_weight=None,
        chunk_size=None,
    ):
        """Message-passing forward with optional edge chunking.

        Args:
            node_features: ``[N, C, *spatial]`` node feature tensor.
            edge_index:    ``[2, E]`` LongTensor.
            edge_features: Optional ``[E, C_e, *spatial]`` edge features.
            edge_weight:   Optional ``[E]`` per-edge scalar weights.
            chunk_size:    If set, process edges in chunks of this size to
                reduce peak memory.  ``None`` (default) uses the standard
                single-pass path.  Supported for ``aggr='sum'`` and
                ``aggr='mean'``; falls back to the standard path for
                ``aggr='max'``.

        Returns:
            Updated node features ``[N, out_C, *spatial]``.
        """
        if chunk_size is None or edge_index.size(1) <= chunk_size:
            return super().forward(node_features, edge_index, edge_features, edge_weight)
        if self.aggr == "max":
            # Chunked max requires a running-max scatter which is supported,
            # but the scatter_reduce_ API varies across PyTorch versions.
            # Fall back to unchunked for safety.
            import warnings
            warnings.warn(
                "ConvMessagePassing: chunk_size is ignored for aggr='max' "
                "(unchunked path used). Use aggr='sum' or 'mean' for chunked forward.",
                stacklevel=2,
            )
            return super().forward(node_features, edge_index, edge_features, edge_weight)
        return self._chunked_forward(
            node_features, edge_index, edge_features, edge_weight, chunk_size
        )

    def _chunked_forward(self, node_features, edge_index, edge_features, edge_weight, chunk_size):
        """Chunked forward for aggr='sum' and aggr='mean'."""
        from ._scatter import broadcast_edge_weight

        E = edge_index.size(1)
        N = node_features.size(0)
        device = node_features.device
        dtype = node_features.dtype
        spatial = node_features.shape[2:]

        aggregated = torch.zeros(N, self.out_channels, *spatial, device=device, dtype=dtype)
        counts = (
            torch.zeros(N, device=device, dtype=dtype)
            if self.aggr == "mean"
            else None
        )

        for start in range(0, E, chunk_size):
            end = min(start + chunk_size, E)
            chunk_ei = edge_index[:, start:end]
            chunk_size_actual = end - start

            src_c = node_features[chunk_ei[0]]
            dst_c = node_features[chunk_ei[1]]
            ef_c = edge_features[start:end] if edge_features is not None else None

            msg = self.message(src_c, dst_c, ef_c)  # [chunk, out_C, *spatial]

            if edge_weight is not None:
                w_c = edge_weight[start:end]
                msg = msg * broadcast_edge_weight(w_c, msg, num_edges=chunk_size_actual)

            target = chunk_ei[1]
            aggregated.index_add_(0, target, msg)

            if counts is not None:
                counts.index_add_(
                    0, target, torch.ones(chunk_size_actual, device=device, dtype=dtype)
                )

        if counts is not None:
            view = (N,) + (1,) * (aggregated.dim() - 1)
            aggregated = aggregated / counts.view(view).clamp(min=1)

        return self.update(node_features, aggregated)

    # ------------------------------------------------------------------ #
    # Message / Update                                                     #
    # ------------------------------------------------------------------ #

    def message(self, src, dest, edge_attr):
        if self.use_edge_features and edge_attr is not None:
            if edge_attr.dim() != src.dim():
                raise ValueError(
                    f"ConvMessagePassing: edge_features must have the same "
                    f"rank as node features. Node tensor has {src.dim()} dims, "
                    f"edge tensor has {edge_attr.dim()} dims (shape "
                    f"{tuple(edge_attr.shape)}). For 3-D node features, "
                    f"edge_features must be [E, C, D, H, W]."
                )
            if edge_attr.size(1) != self.node_channels:
                raise ValueError(
                    f"ConvMessagePassing: edge_features channel count "
                    f"{edge_attr.size(1)} does not match node channel count "
                    f"{self.node_channels}. ConvMessagePassing requires the "
                    f"two to match because src/dst/edge are concatenated "
                    f"along the channel axis. Use TensorGraphSAGELayer or "
                    f"TensorGINLayer for an arbitrary edge_dim."
                )
            msg_input = torch.cat([src, dest, edge_attr], dim=1)
        else:
            msg_input = torch.cat([src, dest], dim=1)
        return self.conv(msg_input)

    def update(self, node_feature, aggregated_message):
        aggregated_message = self.aggregator(aggregated_message)
        if self.residual and node_feature.shape == aggregated_message.shape:
            aggregated_message = node_feature + aggregated_message
        return aggregated_message
