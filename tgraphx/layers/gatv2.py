"""GATv2 (Brody et al., 2022) for vector node features.

Differences from GATv1:

* ``a^T LeakyReLU([Wx_i ‖ Wx_j])`` instead of ``LeakyReLU(a^T [...])``.
* The non-linearity sits *between* the linear projection and the
  attention dot product, which avoids the static-attention pathology.

Tensor-aware spatial/volumetric variants are not implemented in v0.3.0
because :class:`tgraphx.layers.TensorGATLayer` already covers the
spatial case.  This module is vector-only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._scatter import edge_softmax


class GATv2Conv(nn.Module):
    """Multi-head GATv2 convolution for vector node features.

    Args:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension; must be divisible by
            ``num_heads`` when ``concat_heads=True``.
        num_heads: Number of attention heads.
        concat_heads: When ``True`` (default), heads are concatenated
            (output is ``out_dim``); when ``False``, heads are averaged
            (output is ``out_dim // num_heads`` after the linear).
        negative_slope: LeakyReLU slope used inside the attention.
        attn_dropout: Dropout applied to attention weights.
        bias: Whether to include a bias on the output.
        add_self_loops: When ``True`` (default), every node attends to
            itself — recommended for stability.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        concat_heads: bool = True,
        negative_slope: float = 0.2,
        attn_dropout: float = 0.0,
        bias: bool = True,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1; got {num_heads}")
        if concat_heads and out_dim % num_heads != 0:
            raise ValueError(
                f"With concat_heads=True, out_dim ({out_dim}) must be "
                f"divisible by num_heads ({num_heads})."
            )
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_heads = int(num_heads)
        self.head_dim = out_dim // num_heads if concat_heads else out_dim
        self.concat_heads = bool(concat_heads)
        self.negative_slope = float(negative_slope)
        self.attn_dropout_p = float(attn_dropout)
        self.add_self_loops = bool(add_self_loops)

        self.W_l = nn.Linear(in_dim, num_heads * self.head_dim, bias=False)
        self.W_r = nn.Linear(in_dim, num_heads * self.head_dim, bias=False)
        self.att = nn.Parameter(torch.empty(num_heads, self.head_dim))
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(num_heads * self.head_dim if concat_heads else self.head_dim)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_l.weight)
        nn.init.xavier_uniform_(self.W_r.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(
                f"GATv2Conv expects [N, D] vector features; got {tuple(x.shape)}"
            )
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"edge_index must be [2, E]; got {tuple(edge_index.shape)}"
            )
        N = x.size(0)
        K = self.num_heads
        d = self.head_dim
        device = x.device

        src = edge_index[0]
        dst = edge_index[1]
        if self.add_self_loops:
            self_idx = torch.arange(N, device=device, dtype=torch.long)
            src = torch.cat([src, self_idx], dim=0)
            dst = torch.cat([dst, self_idx], dim=0)

        h_l = self.W_l(x).view(N, K, d)        # left projection
        h_r = self.W_r(x).view(N, K, d)        # right (used as the "query" side)
        h_src = h_l.index_select(0, src)        # [E, K, d]
        h_dst = h_r.index_select(0, dst)
        # GATv2: a^T · LeakyReLU(h_src + h_dst)
        e = (h_src + h_dst)
        e = F.leaky_relu(e, negative_slope=self.negative_slope)
        score = (e * self.att.unsqueeze(0)).sum(dim=-1)   # [E, K]
        attn = edge_softmax(score, dst, N)                # softmax over incoming edges
        if self.attn_dropout_p > 0.0 and self.training:
            attn = F.dropout(attn, p=self.attn_dropout_p, training=True)

        weighted = h_src * attn.unsqueeze(-1)              # [E, K, d]
        out = torch.zeros(N, K, d, device=device, dtype=x.dtype)
        out.index_add_(0, dst, weighted)
        if self.concat_heads:
            out = out.reshape(N, K * d)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
                f"num_heads={self.num_heads}, head_dim={self.head_dim}")
