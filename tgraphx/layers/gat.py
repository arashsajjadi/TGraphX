"""Tensor-aware multi-head Graph Attention Network layer.

This is the canonical GAT (Veličković et al. 2018) adapted to spatial node
feature maps ``[N, C, H, W]``.

Mathematics (split-attention form, equivalent to the original concat-and-dot):

    h_i^k       = W^k x_i                                # [N, C_head, H, W] per head k
    score_ij^k  = LeakyReLU( a_dst^k · pool(h_j^k)
                           + a_src^k · pool(h_i^k) )    # scalar per edge per head
    α_ij^k      = softmax_i ( score_ij^k )              # over incoming edges to j
    o_j^k       = sum_i  α_ij^k * h_i^k                  # [N, C_head, H, W] per head
    o_j         = concat_k o_j^k   or   mean_k o_j^k    # head combination

The score is scalar per ``(edge, head)``: spatial dimensions are mean-pooled
*before* the dot product with the learned attention vectors.  The values
themselves keep their full ``[C_head, H, W]`` layout when aggregated.  This
is the "scalar attention per edge per head" mode requested by the design.
Per-channel and per-pixel attention modes are reasonable next steps but are
intentionally not implemented here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._scatter import edge_softmax


class TensorGATLayer(nn.Module):
    """Tensor-aware multi-head GAT layer.

    Args:
        in_channels: Input channel count ``C_in``.
        out_channels: Output channel count.

            * ``concat_heads=True``: ``out_channels`` must be divisible by
              ``num_heads``; each head produces ``out_channels // num_heads``
              channels.
            * ``concat_heads=False``: ``out_channels`` is the per-head
              channel count and heads are averaged.
        num_heads: Number of attention heads.
        concat_heads: Concatenate heads (``True``, default GAT) or average
            them (``False``, GAT's last-layer convention).
        negative_slope: LeakyReLU slope before edge softmax.
        attn_dropout: Dropout probability on attention weights (training
            only).
        residual: Add a residual connection from the input.  Auto-projects
            to match channels.
        bias: Add a learnable output bias.
        add_self_loops: Append self-loops to ``edge_index`` so every node
            attends to itself.

    Shape conventions:
        * input  ``x``           : ``[N, in_channels, H, W]``
        * input  ``edge_index``  : ``[2, E]`` (``dtype=torch.long``)
        * output                 : ``[N, out_channels, H, W]``

    Edge features are not yet supported.  Passing a non-``None``
    ``edge_features`` raises ``NotImplementedError``.

    Examples:
        >>> layer = TensorGATLayer(in_channels=8, out_channels=16,
        ...                        num_heads=4, add_self_loops=True)
        >>> out, attn = layer(x, edge_index, return_attention=True)
        >>> attn.shape   # [E (+ N if self-loops added), num_heads]
    """

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

        # Linear projection W shared by source and destination (standard
        # GAT).  Implemented as a single 1x1 Conv2d producing
        # ``num_heads * head_channels`` channels, then reshaped.
        self.W = nn.Conv2d(
            in_channels, num_heads * head_channels, kernel_size=1, bias=False
        )

        # Per-head split-attention parameters.
        self.a_dst = nn.Parameter(torch.empty(num_heads, head_channels))
        self.a_src = nn.Parameter(torch.empty(num_heads, head_channels))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_total))
        else:
            self.register_parameter("bias", None)

        if residual and in_channels != out_total:
            self.res_proj = nn.Conv2d(in_channels, out_total, kernel_size=1, bias=False)
        else:
            self.res_proj = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_dst)
        nn.init.xavier_uniform_(self.a_src)
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
        return_attention: bool = False,
    ):
        """Run multi-head GAT over a graph with spatial node features.

        Args:
            x: ``[N, in_channels, H, W]`` node feature tensor.
            edge_index: ``[2, E]`` ``torch.long`` edge index.
            edge_features: not yet supported — must be ``None``.
            return_attention: if ``True``, return ``(out, attn)`` where
                ``attn`` has shape ``[E_eff, num_heads]`` with ``E_eff = E``
                or ``E + N`` if ``add_self_loops=True``.

        Returns:
            ``out`` of shape ``[N, out_channels, H, W]``, or
            ``(out, attn)`` if ``return_attention=True``.
        """
        if edge_features is not None:
            raise NotImplementedError(
                "TensorGATLayer does not yet support edge features. "
                "Pass edge_features=None or use TensorGraphSAGELayer / "
                "TensorGINLayer for edge-feature support."
            )
        if x.dim() != 4:
            raise ValueError(
                f"x must have shape [N, C, H, W]; got {tuple(x.shape)}."
            )
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"edge_index must have shape [2, E]; got {tuple(edge_index.shape)}."
            )
        if edge_index.dtype != torch.long:
            raise TypeError(
                f"edge_index must have dtype torch.long; got {edge_index.dtype}."
            )

        N, _, H, W = x.shape
        src = edge_index[0]
        dst = edge_index[1]

        if self.add_self_loops:
            loop = torch.arange(N, device=x.device, dtype=torch.long)
            src = torch.cat([src, loop], dim=0)
            dst = torch.cat([dst, loop], dim=0)

        # Linear projection: [N, K * C_head, H, W] -> [N, K, C_head, H, W].
        h = self.W(x).view(N, self.num_heads, self.head_channels, H, W)

        # Gather per-edge features.
        h_src = h.index_select(0, src)  # [E, K, C_head, H, W]
        h_dst = h.index_select(0, dst)

        # Spatial mean pool to get a vector representation per (edge, head)
        # for the scoring step.  Values keep their full spatial layout.
        h_src_pool = h_src.mean(dim=(-2, -1))  # [E, K, C_head]
        h_dst_pool = h_dst.mean(dim=(-2, -1))

        # Scalar score per (edge, head).  ``a_src`` and ``a_dst`` have shape
        # ``[K, C_head]`` so the elementwise multiply broadcasts over E.
        score_src = (h_src_pool * self.a_src.unsqueeze(0)).sum(dim=-1)  # [E, K]
        score_dst = (h_dst_pool * self.a_dst.unsqueeze(0)).sum(dim=-1)
        scores = F.leaky_relu(score_src + score_dst, negative_slope=self.negative_slope)

        # Edge softmax over destinations per head: each j has weights summing to 1.
        attn = edge_softmax(scores, dst, N)  # [E, K]
        if self.attn_dropout_p > 0.0 and self.training:
            attn_dropped = F.dropout(attn, p=self.attn_dropout_p, training=True)
        else:
            attn_dropped = attn

        # Weight values and aggregate by destination.
        weighted = attn_dropped.view(*attn_dropped.shape, 1, 1, 1) * h_src
        out_per_head = h.new_zeros(
            (N, self.num_heads, self.head_channels, H, W)
        )
        out_per_head.index_add_(0, dst, weighted)

        # Combine heads.
        if self.concat_heads:
            out = out_per_head.reshape(N, self._out_total, H, W)
        else:
            out = out_per_head.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        if self.residual:
            if self.res_proj is not None:
                out = out + self.res_proj(x)
            elif x.shape == out.shape:
                out = out + x
            # otherwise silently skip (shapes incompatible and no auto-projection)

        if return_attention:
            return out, attn
        return out

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"num_heads={self.num_heads}, concat_heads={self.concat_heads}, "
            f"head_channels={self.head_channels}"
        )
