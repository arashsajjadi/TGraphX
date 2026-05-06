"""Tensor-aware multi-head Graph Attention Network layer.

GAT (Veličković et al. 2018) adapted to 2-D and 3-D spatial node feature
maps:

* 2-D: ``[N, C, H, W]``      (``spatial_rank=2``, default)
* 3-D: ``[N, C, D, H, W]``   (``spatial_rank=3``)

Mathematics (split-attention form, equivalent to the original concat-and-dot):

    h_i^k       = W^k x_i                                  # per-head value tensor
    score_ij^k  = LeakyReLU( a_dst^k · pool(h_j^k)
                           + a_src^k · pool(h_i^k)
                           + b^k(e_ij) )                    # optional edge bias
    α_ij^k      = softmax_i ( score_ij^k )                  # over incoming edges to j
    m_ij^k      = α_ij^k · w_ij · h_i^k                     # w_ij optional
    o_j^k       = sum_i  m_ij^k                             # per-head value
    o_j         = concat_k o_j^k   or   mean_k o_j^k

The attention score is **scalar per (edge, head)**: spatial dimensions are
mean-pooled before the dot product with the learned attention vectors.
The values themselves keep their full ``[C_head, *spatial]`` layout in the
aggregation.  Per-channel, per-pixel and per-voxel attention are
intentionally not implemented.

Edge feature formats (when ``use_edge_features=True``):

* **Vector** ``[E, edge_dim]`` — projected directly to ``[E, num_heads]``
  attention bias.
* **Matching-rank spatial** — ``[E, edge_dim, H_e, W_e]`` for 2-D nodes,
  ``[E, edge_dim, D_e, H_e, W_e]`` for 3-D nodes.  Spatial dims are
  mean-pooled to a vector before the bias projection runs (so spatial
  edge dims need not match node spatial dims).
* **Mismatched-rank spatial** raises ``NotImplementedError`` (e.g. 5-D
  edges into a 2-D-configured GAT, or 4-D edges into a 3-D-configured
  GAT).

Edge weight (``edge_weight``):

A per-edge ``[E]`` scalar that scales values **after** softmax-normalised
attention, before the destination-wise sum.  Self-loops (when
``add_self_loops=True``) implicitly use weight ``1``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._dim import (
    conv1x1,
    expected_x_dim,
    mean_over_spatial,
    validate_spatial_rank,
    view_for_channel_bias,
)
from ._scatter import broadcast_edge_weight, edge_softmax


class TensorGATLayer(nn.Module):
    """Tensor-aware multi-head GAT layer for 2-D or 3-D node features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 1,
        concat_heads: bool = True,
        negative_slope: float = 0.2,
        attn_dropout: float = 0.0,
        residual: bool = False,
        bias: bool = True,
        add_self_loops: bool = False,
        use_edge_features: bool = False,
        edge_dim: int | None = None,
        spatial_rank: int = 2,
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1; got {num_heads}")
        if concat_heads:
            if out_channels % num_heads != 0:
                raise ValueError(
                    f"With concat_heads=True, out_channels ({out_channels}) "
                    f"must be divisible by num_heads ({num_heads})."
                )
            head_channels = out_channels // num_heads
            out_total = num_heads * head_channels  # == out_channels
        else:
            head_channels = out_channels
            out_total = head_channels  # heads averaged
        if use_edge_features and (edge_dim is None or edge_dim <= 0):
            raise ValueError(
                "edge_dim must be a positive integer when use_edge_features=True."
            )
        validate_spatial_rank(spatial_rank)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.head_channels = head_channels
        self._out_total = out_total
        self.negative_slope = float(negative_slope)
        self.attn_dropout_p = float(attn_dropout)
        self.residual = residual
        self.add_self_loops = add_self_loops
        self.use_edge_features = use_edge_features
        self.edge_dim = edge_dim
        self.spatial_rank = spatial_rank

        # Linear projection W: rank-aware 1×1 (or 1×1×1) convolution.
        self.W = conv1x1(spatial_rank, in_channels, num_heads * head_channels, bias=False)

        # Per-head split-attention parameters.
        self.a_dst = nn.Parameter(torch.empty(num_heads, head_channels))
        self.a_src = nn.Parameter(torch.empty(num_heads, head_channels))

        if use_edge_features:
            self.edge_bias_proj: nn.Module = nn.Linear(edge_dim, num_heads, bias=False)
        else:
            self.edge_bias_proj = None

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_total))
        else:
            self.register_parameter("bias", None)

        if residual and in_channels != out_total:
            self.res_proj = conv1x1(spatial_rank, in_channels, out_total, bias=False)
        else:
            self.res_proj = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_dst)
        nn.init.xavier_uniform_(self.a_src)
        if self.edge_bias_proj is not None:
            nn.init.xavier_uniform_(self.edge_bias_proj.weight)
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
        return_attention: bool = False,
    ):
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
        if (not self.use_edge_features) and edge_features is not None:
            raise ValueError(
                "Layer was constructed with use_edge_features=False; "
                "do not pass edge_features."
            )
        if self.use_edge_features and edge_features is None:
            raise ValueError(
                "Layer was constructed with use_edge_features=True; "
                "edge_features must be provided."
            )

        # Edge feature shape validation: vector OR matching-rank spatial.
        edge_pool: torch.Tensor | None = None
        spatial_edge_dim = 2 + rank  # 4 for 2-D nodes, 5 for 3-D nodes
        if self.use_edge_features and edge_features is not None:
            if edge_features.size(0) != edge_index.size(1):
                raise ValueError(
                    f"edge_features has {edge_features.size(0)} rows but "
                    f"edge_index has {edge_index.size(1)} edges."
                )
            if edge_features.dim() == 2:
                if edge_features.size(1) != self.edge_dim:
                    raise ValueError(
                        f"edge_features.shape[-1]={edge_features.size(1)} does not "
                        f"match edge_dim={self.edge_dim}."
                    )
                edge_pool = edge_features
            elif edge_features.dim() == spatial_edge_dim:
                if edge_features.size(1) != self.edge_dim:
                    raise ValueError(
                        f"edge_features channel count {edge_features.size(1)} "
                        f"does not match edge_dim={self.edge_dim}."
                    )
                # Mean-pool spatial / volumetric dims to a vector per (edge, channel).
                edge_pool = mean_over_spatial(edge_features, rank)
            else:
                # 5-D into a 2-D-configured GAT == "volumetric edges"; reject
                # explicitly with NotImplementedError so callers can pin the
                # contract.  Other mismatched ranks are a plain shape error.
                if rank == 2 and edge_features.dim() == 5:
                    raise NotImplementedError(
                        f"TensorGATLayer (spatial_rank=2) does not support 5-D "
                        f"volumetric edge features [E, C_e, D, H, W]; got "
                        f"{tuple(edge_features.shape)}. Construct the layer with "
                        f"spatial_rank=3 to enable volumetric edges."
                    )
                if rank == 3 and edge_features.dim() == 4:
                    raise NotImplementedError(
                        f"TensorGATLayer (spatial_rank=3) does not accept 4-D "
                        f"2-D-spatial edge features [E, C_e, H, W]; got "
                        f"{tuple(edge_features.shape)}. Provide volumetric edge "
                        f"features [E, edge_dim, D, H, W] or vector "
                        f"[E, edge_dim] instead."
                    )
                raise ValueError(
                    f"TensorGATLayer expects edge_features of shape "
                    f"[E, edge_dim] or [E, edge_dim, " +
                    ("H, W]" if rank == 2 else "D, H, W]") +
                    f"; got {tuple(edge_features.shape)}."
                )

        N = x.size(0)
        spatial = x.shape[2:]  # (H, W) or (D, H, W)
        src = edge_index[0]
        dst = edge_index[1]
        E_orig = src.size(0)

        if self.add_self_loops:
            loop = torch.arange(N, device=x.device, dtype=torch.long)
            src = torch.cat([src, loop], dim=0)
            dst = torch.cat([dst, loop], dim=0)
        E_eff = src.size(0)

        # Linear projection: [N, K * C_head, *spatial] -> [N, K, C_head, *spatial].
        h = self.W(x).view(N, self.num_heads, self.head_channels, *spatial)

        # Gather per-edge per-head values.
        h_src = h.index_select(0, src)  # [E_eff, K, C_head, *spatial]
        h_dst = h.index_select(0, dst)

        # Resolve and broadcast edge_weight (with self-loop padding = 1).
        weight_b: torch.Tensor | None = None
        if edge_weight is not None:
            broadcast_edge_weight(edge_weight, x, num_edges=E_orig)  # validate against E_orig
            if self.add_self_loops:
                pad = edge_weight.new_ones(N)
                full_w = torch.cat([edge_weight, pad], dim=0)
            else:
                full_w = edge_weight
            weight_b = broadcast_edge_weight(full_w, h_src, num_edges=E_eff)

        # Spatial mean pool to get a vector representation per (edge, head)
        # for the scoring step.  Values keep their full layout for sum.
        h_src_pool = mean_over_spatial(h_src, rank)  # [E_eff, K, C_head]
        h_dst_pool = mean_over_spatial(h_dst, rank)

        score_src = (h_src_pool * self.a_src.unsqueeze(0)).sum(dim=-1)  # [E_eff, K]
        score_dst = (h_dst_pool * self.a_dst.unsqueeze(0)).sum(dim=-1)
        scores_pre = score_src + score_dst

        if self.use_edge_features and edge_pool is not None:
            edge_bias = self.edge_bias_proj(edge_pool)  # [E_orig, K]
            if self.add_self_loops:
                pad = edge_bias.new_zeros(N, self.num_heads)
                edge_bias = torch.cat([edge_bias, pad], dim=0)
            scores_pre = scores_pre + edge_bias

        scores = F.leaky_relu(scores_pre, negative_slope=self.negative_slope)

        attn = edge_softmax(scores, dst, N)  # [E_eff, K]
        if self.attn_dropout_p > 0.0 and self.training:
            attn_dropped = F.dropout(attn, p=self.attn_dropout_p, training=True)
        else:
            attn_dropped = attn

        # Weight values and aggregate by destination.  attn has shape [E, K]
        # and h_src has shape [E, K, C_head, *spatial]; broadcast attn over
        # the trailing channel + spatial dims.
        trailing_ones = (1,) * (h_src.dim() - attn_dropped.dim())
        weighted = attn_dropped.view(*attn_dropped.shape, *trailing_ones) * h_src
        if weight_b is not None:
            weighted = weighted * weight_b
        out_per_head = h.new_zeros((N, self.num_heads, self.head_channels, *spatial))
        out_per_head.index_add_(0, dst, weighted)

        if self.concat_heads:
            out = out_per_head.reshape(N, self._out_total, *spatial)
        else:
            out = out_per_head.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias.view(*view_for_channel_bias(rank, self._out_total))

        if self.residual:
            if self.res_proj is not None:
                out = out + self.res_proj(x)
            elif x.shape == out.shape:
                out = out + x

        if return_attention:
            return out, attn
        return out

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"num_heads={self.num_heads}, concat_heads={self.concat_heads}, "
            f"head_channels={self.head_channels}, spatial_rank={self.spatial_rank}"
        )
