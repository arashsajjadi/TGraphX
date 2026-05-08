"""Tensor-aware GraphSAGE layer for 2-D or 3-D node features.

Adapts Hamilton, Ying & Leskovec (2017) to spatial / volumetric features:

    h_j' = W_self(h_j) + W_neigh( AGG_{i ∈ N(j)} h_i )    [+ edge bias]

``W_self`` and ``W_neigh`` are 1×1 (or 1×1×1) convolutions so the spatial
layout ``[C, H, W]`` (rank 2) or ``[C, D, H, W]`` (rank 3) is preserved.
``AGG`` is mean (default) or max.

Edge features (optional) come in two forms, selected via
``edge_features_kind``:

* ``"spatial"`` (default): ``e_ij`` has shape ``[E, edge_dim, *spatial]``
  matching the layer's ``spatial_rank``, and is concatenated to the source
  feature map along the channel dim before ``W_neigh`` is applied.
* ``"vector"``: ``e_ij`` has shape ``[E, edge_dim]`` and is projected via
  ``Linear(edge_dim, out_channels)`` into a per-edge channel bias of shape
  ``[E, out_channels, 1, ...]``, broadcast across the spatial grid and
  added to ``W_neigh(h_src)`` before aggregation.

``edge_weight`` (``[E]``) scales each neighbour message after the edge
projection / concatenation, before scatter aggregation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._dim import (
    conv1x1,
    expected_x_dim,
    trailing_ones,
    validate_spatial_rank,
)
from ._scatter import broadcast_edge_weight, scatter_max, scatter_mean


class TensorGraphSAGELayer(nn.Module):
    """Tensor-aware GraphSAGE layer for 2-D or 3-D node features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = "mean",
        normalize: bool = False,
        bias: bool = True,
        residual: bool = False,
        use_edge_features: bool = False,
        edge_dim: int | None = None,
        edge_features_kind: str = "spatial",
        spatial_rank: int = 2,
    ) -> None:
        super().__init__()
        if aggr not in ("mean", "max"):
            raise ValueError(
                f"aggr must be 'mean' or 'max'; got {aggr!r}. "
                f"LSTM-style aggregation is not implemented."
            )
        if use_edge_features and edge_dim is None:
            raise ValueError("edge_dim must be set when use_edge_features=True")
        if edge_features_kind not in ("spatial", "vector"):
            raise ValueError(
                f"edge_features_kind must be 'spatial' or 'vector'; got "
                f"{edge_features_kind!r}."
            )
        validate_spatial_rank(spatial_rank)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.normalize = normalize
        self.use_edge_features = use_edge_features
        self.edge_dim = edge_dim
        self.edge_features_kind = edge_features_kind
        self.residual = residual
        self.spatial_rank = spatial_rank

        self.W_self = conv1x1(spatial_rank, in_channels, out_channels, bias=bias)

        # W_neigh input channel count depends on whether spatial edge tensors
        # are concatenated.  Vector edges are added as a post-projection bias.
        if use_edge_features and edge_features_kind == "spatial":
            neigh_in = in_channels + edge_dim
        else:
            neigh_in = in_channels
        self.W_neigh = conv1x1(spatial_rank, neigh_in, out_channels, bias=False)

        if use_edge_features and edge_features_kind == "vector":
            self.edge_bias_proj: nn.Module = nn.Linear(edge_dim, out_channels)
        else:
            self.edge_bias_proj = None

        if residual and in_channels != out_channels:
            self.res_proj = conv1x1(spatial_rank, in_channels, out_channels, bias=False)
        else:
            self.res_proj = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_self.weight)
        if self.W_self.bias is not None:
            nn.init.zeros_(self.W_self.bias)
        nn.init.xavier_uniform_(self.W_neigh.weight)
        if self.edge_bias_proj is not None:
            nn.init.xavier_uniform_(self.edge_bias_proj.weight)
            if self.edge_bias_proj.bias is not None:
                nn.init.zeros_(self.edge_bias_proj.bias)
        if self.res_proj is not None:
            nn.init.xavier_uniform_(self.res_proj.weight)

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Message-passing forward with optional edge chunking.

        Args:
            x: Node features ``[N, C, *spatial]``.
            edge_index: ``[2, E]`` LongTensor.
            edge_features: Optional spatial or vector edge features.
            edge_weight: Optional ``[E]`` per-edge scalar weights.
            chunk_size: If set, process edges in chunks of this size to reduce
                peak message-buffer memory.  ``None`` (default) uses the
                standard single-pass path.  Supported for both
                ``aggr='mean'`` and ``aggr='max'``.

        Returns:
            Updated node features ``[N, out_channels, *spatial]``.
        """
        rank = self.spatial_rank
        x_dim = expected_x_dim(rank)
        if x.dim() != x_dim:
            shape_str = "[N, C, H, W]" if rank == 2 else "[N, C, D, H, W]"
            raise ValueError(
                f"x must have shape {shape_str} (spatial_rank={rank}); "
                f"got {tuple(x.shape)}."
            )
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"edge_index must have shape [2, E]; got {tuple(edge_index.shape)}."
            )
        if edge_index.dtype != torch.long:
            raise TypeError(
                f"edge_index must have dtype torch.long; got {edge_index.dtype}."
            )
        if self.use_edge_features and edge_features is None:
            raise ValueError(
                "Layer was constructed with use_edge_features=True, but "
                "edge_features was None."
            )
        if (not self.use_edge_features) and edge_features is not None:
            raise ValueError(
                "Layer was constructed with use_edge_features=False; do not "
                "pass edge_features."
            )

        N = x.size(0)
        src = edge_index[0]
        dst = edge_index[1]
        E = src.size(0)

        # Self transform applied to every node (not chunked — full node op).
        self_out = self.W_self(x)

        # Edge feature validation (shared by both paths).
        if self.use_edge_features and edge_features is not None:
            if edge_features.size(0) != E:
                raise ValueError(
                    f"edge_features has {edge_features.size(0)} rows but "
                    f"edge_index has {E} edges."
                )
            if self.edge_features_kind == "spatial":
                expected_dim = 2 + rank
                spatial_str = "[E, edge_dim, H, W]" if rank == 2 else "[E, edge_dim, D, H, W]"
                if edge_features.dim() != expected_dim:
                    raise ValueError(
                        f"edge_features must have shape {spatial_str} when "
                        f"edge_features_kind='spatial' and spatial_rank={rank}; "
                        f"got {tuple(edge_features.shape)}."
                    )
                if edge_features.size(1) != self.edge_dim:
                    raise ValueError(
                        f"edge_features channel count {edge_features.size(1)} "
                        f"does not match edge_dim={self.edge_dim}."
                    )
            else:  # "vector"
                if edge_features.dim() != 2:
                    raise ValueError(
                        f"edge_features must have shape [E, edge_dim] when "
                        f"edge_features_kind='vector'; got "
                        f"{tuple(edge_features.shape)}."
                    )
                if edge_features.size(1) != self.edge_dim:
                    raise ValueError(
                        f"edge_features last-dim {edge_features.size(1)} does "
                        f"not match edge_dim={self.edge_dim}."
                    )

        if chunk_size is not None and E > chunk_size:
            agg = self._chunked_forward(
                x, src, dst, N, E, rank, edge_features, edge_weight, chunk_size
            )
        else:
            agg = self._full_forward(x, src, dst, N, E, rank, edge_features, edge_weight)

        out = self_out + agg

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=1)

        if self.residual:
            if self.res_proj is not None:
                out = out + self.res_proj(x)
            elif x.shape == out.shape:
                out = out + x

        return out

    # ------------------------------------------------------------------ #
    # Internal message-passing helpers                                     #
    # ------------------------------------------------------------------ #

    def _full_forward(
        self,
        x: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        N: int,
        E: int,
        rank: int,
        edge_features: torch.Tensor | None,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        """Standard single-pass aggregation (original unchunked path)."""
        if self.use_edge_features and edge_features is not None:
            if self.edge_features_kind == "spatial":
                cat = torch.cat([x.index_select(0, src), edge_features], dim=1)
                messages = self.W_neigh(cat)
            else:  # "vector"
                src_msg = self.W_neigh(x.index_select(0, src))
                edge_bias = self.edge_bias_proj(edge_features)
                bias_view = (edge_bias.size(0), self.out_channels) + trailing_ones(rank)
                messages = src_msg + edge_bias.view(bias_view)
        else:
            h_neigh = self.W_neigh(x)
            messages = h_neigh.index_select(0, src)

        if edge_weight is not None:
            weight_b = broadcast_edge_weight(edge_weight, messages, num_edges=E)
            messages = messages * weight_b

        if self.aggr == "mean":
            return scatter_mean(messages, dst, N)
        return scatter_max(messages, dst, N)

    def _compute_chunk_messages(
        self,
        x: torch.Tensor,
        src_c: torch.Tensor,
        ef_c: torch.Tensor | None,
        n_c: int,
        rank: int,
        h_neigh: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute per-edge messages for one chunk of edges."""
        if self.use_edge_features and ef_c is not None:
            if self.edge_features_kind == "spatial":
                cat = torch.cat([x.index_select(0, src_c), ef_c], dim=1)
                return self.W_neigh(cat)
            else:  # "vector"
                src_msg = self.W_neigh(x.index_select(0, src_c))
                edge_bias = self.edge_bias_proj(ef_c)
                bias_view = (n_c, self.out_channels) + trailing_ones(rank)
                return src_msg + edge_bias.view(bias_view)
        else:
            # h_neigh was pre-computed outside the chunk loop.
            assert h_neigh is not None
            return h_neigh.index_select(0, src_c)

    def _chunked_forward(
        self,
        x: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        N: int,
        E: int,
        rank: int,
        edge_features: torch.Tensor | None,
        edge_weight: torch.Tensor | None,
        chunk_size: int,
    ) -> torch.Tensor:
        """Chunked edge processing — reduces peak message-buffer memory.

        For ``aggr='mean'``: accumulates per-node sum and count, divides at end.
        For ``aggr='max'``: running per-node max via scatter_reduce_.
        Both produce output identical to the unchunked path up to float precision.
        """
        # Pre-compute W_neigh(x) once for the no-edge-features path to avoid
        # redundant conv calls across chunks.
        h_neigh = None if self.use_edge_features else self.W_neigh(x)

        agg: torch.Tensor | None = None
        counts: torch.Tensor | None = None  # used for mean only

        for start in range(0, E, chunk_size):
            end = min(start + chunk_size, E)
            n_c = end - start
            src_c = src[start:end]
            dst_c = dst[start:end]
            ef_c = edge_features[start:end] if edge_features is not None else None

            msg_c = self._compute_chunk_messages(x, src_c, ef_c, n_c, rank, h_neigh)

            if edge_weight is not None:
                ew_c = edge_weight[start:end].to(dtype=msg_c.dtype)
                ew_b = ew_c.view(n_c, *(1,) * (msg_c.dim() - 1))
                msg_c = msg_c * ew_b

            # Initialise aggregation buffers from the first chunk's dtype/device.
            if agg is None:
                if self.aggr == "mean":
                    agg = msg_c.new_zeros(N, *msg_c.shape[1:])
                    counts = msg_c.new_zeros(N)
                else:  # "max" — initial -inf sentinel; updated below
                    agg = msg_c.new_full((N, *msg_c.shape[1:]), float("-inf"))

            if self.aggr == "mean":
                agg.index_add_(0, dst_c, msg_c)
                counts.index_add_(0, dst_c, msg_c.new_ones(n_c))  # type: ignore[union-attr]
            else:  # "max"
                # Create a FRESH buffer per chunk so scatter_reduce_ is called only
                # once per buffer (single in-place op → autograd-safe).  Combine
                # chunk contributions via out-of-place torch.maximum.
                chunk_buf = msg_c.new_full((N, *msg_c.shape[1:]), float("-inf"))
                tgt_b = dst_c.view(n_c, *(1,) * (msg_c.dim() - 1)).expand_as(msg_c)
                chunk_buf.scatter_reduce_(0, tgt_b, msg_c, reduce="amax", include_self=True)
                agg = torch.maximum(agg, chunk_buf)  # type: ignore[arg-type]

        if agg is None:
            # E == 0 was caught by the caller condition; this should not happen.
            # Return zeros with the correct shape by falling back to scatter_mean.
            return scatter_mean(x.new_zeros(0, self.out_channels, *x.shape[2:]),
                                dst[:0], N)

        if self.aggr == "mean":
            view = (N,) + (1,) * (agg.dim() - 1)
            agg = agg / counts.view(view).clamp_min(1.0)  # type: ignore[union-attr]
        else:  # "max" — replace -inf (isolated nodes) with 0
            agg = agg.masked_fill(torch.isinf(agg) & (agg < 0), 0.0)

        return agg

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"aggr={self.aggr!r}, normalize={self.normalize}, "
            f"spatial_rank={self.spatial_rank}"
        )
