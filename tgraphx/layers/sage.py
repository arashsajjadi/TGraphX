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
    ) -> torch.Tensor:
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

        # Self transform applied to every node.
        self_out = self.W_self(x)

        # Per-edge messages.
        if self.use_edge_features and edge_features is not None:
            if edge_features.size(0) != edge_index.size(1):
                raise ValueError(
                    f"edge_features has {edge_features.size(0)} rows but "
                    f"edge_index has {edge_index.size(1)} edges."
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
                cat = torch.cat([x.index_select(0, src), edge_features], dim=1)
                messages = self.W_neigh(cat)
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
                src_msg = self.W_neigh(x.index_select(0, src))  # [E, out, *spatial]
                edge_bias = self.edge_bias_proj(edge_features)   # [E, out]
                # broadcast as [E, out, 1, ...] (rank trailing 1's for spatial dims).
                bias_view = (edge_bias.size(0), self.out_channels) + trailing_ones(rank)
                messages = src_msg + edge_bias.view(bias_view)
        else:
            h_neigh = self.W_neigh(x)
            messages = h_neigh.index_select(0, src)

        if edge_weight is not None:
            weight_b = broadcast_edge_weight(
                edge_weight, messages, num_edges=edge_index.size(1)
            )
            messages = messages * weight_b

        if self.aggr == "mean":
            agg = scatter_mean(messages, dst, N)
        else:  # "max"
            agg = scatter_max(messages, dst, N)

        out = self_out + agg

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=1)

        if self.residual:
            if self.res_proj is not None:
                out = out + self.res_proj(x)
            elif x.shape == out.shape:
                out = out + x

        return out

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"aggr={self.aggr!r}, normalize={self.normalize}, "
            f"spatial_rank={self.spatial_rank}"
        )
