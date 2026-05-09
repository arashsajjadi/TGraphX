"""Heterogeneous Attention Network (HAN) layer.

Reference: Wang et al. (WWW 2019) — "Heterogeneous Graph Attention
Network".

HAN combines two attention mechanisms over a hetero graph:

1. **Node-level attention** within each metapath: a GAT-style
   attention computes weights over each node's metapath neighbours.
2. **Semantic attention** across metapaths: a learnable scoring head
   weighs the contribution of each metapath's embeddings before
   summing.

This implementation expects pre-computed metapath neighbour edge
indices (``metapath_edge_index_dict[name] = LongTensor[2, E_m]``) — the
caller is responsible for materialising the metapaths from the
:class:`~tgraphx.HeteroGraph`.  This decoupling keeps HAN general and
avoids hard-coding metapath schemas.

The layer is **Experimental**: API and semantics may evolve.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HANConv"]


def _scatter_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Per-destination softmax over edges; numerically stable."""
    # Per-dst max for numerical stability.
    max_val = torch.full((num_nodes,), float("-inf"), dtype=scores.dtype, device=scores.device)
    max_val.scatter_reduce_(0, dst, scores, reduce="amax", include_self=True)
    stable = scores - max_val[dst]
    exp = stable.exp()
    sum_exp = torch.zeros(num_nodes, dtype=scores.dtype, device=scores.device)
    sum_exp.scatter_add_(0, dst, exp)
    return exp / (sum_exp[dst] + 1e-12)


class _NodeLevelAttention(nn.Module):
    """GAT-style node-level attention for a single metapath."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.heads = int(heads)
        self.out_dim = int(out_dim)
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att = nn.Parameter(torch.empty(1, heads, 2 * out_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        H, D = self.heads, self.out_dim
        h = self.lin(x).view(N, H, D)  # [N, H, D]
        if edge_index.numel() == 0:
            return h.mean(dim=1)
        src, dst = edge_index[0], edge_index[1]
        h_src = h[src]  # [E, H, D]
        h_dst = h[dst]
        cat = torch.cat([h_src, h_dst], dim=-1)  # [E, H, 2D]
        e = (cat * self.att).sum(dim=-1)  # [E, H]
        e = F.leaky_relu(e, negative_slope=0.2)
        # Per-head softmax over destination.
        alpha = torch.stack([
            _scatter_softmax(e[:, k], dst, N) for k in range(H)
        ], dim=-1)
        alpha = self.dropout(alpha)
        msg = h_src * alpha.unsqueeze(-1)  # [E, H, D]
        out = torch.zeros(N, H, D, dtype=h.dtype, device=h.device)
        out.scatter_add_(
            0,
            dst.view(-1, 1, 1).expand_as(msg),
            msg,
        )
        return out.mean(dim=1)  # [N, D] — average heads


class HANConv(nn.Module):
    """HAN convolution: per-metapath attention + cross-metapath attention.

    Args:
        in_dim: Input feature dimension (assumed shared across metapaths).
        out_dim: Output feature dimension.
        num_heads: Heads for the inner GAT-style attention.
        dropout: Dropout on attention weights.
        semantic_hidden: Hidden dim of the semantic-attention MLP.

    Forward:
        ``x`` — ``FloatTensor[N, in_dim]`` shared across metapaths.
        ``metapath_edge_index_dict`` — ``{metapath_name: LongTensor[2, E_m]}``.

    Returns:
        ``FloatTensor[N, out_dim]`` semantic-attention-weighted output.

    Stability: Experimental.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        semantic_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_heads = int(num_heads)
        # Per-metapath attention modules are created lazily on first forward
        # (we don't yet know which metapaths the user will pass in).
        self._mp_attn: nn.ModuleDict = nn.ModuleDict()
        self._dropout = float(dropout)
        # Semantic attention MLP.
        self.semantic = nn.Sequential(
            nn.Linear(out_dim, semantic_hidden, bias=True),
            nn.Tanh(),
            nn.Linear(semantic_hidden, 1, bias=False),
        )
        for m in self.semantic.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_mp_attn(self, name: str, device: torch.device, dtype: torch.dtype) -> _NodeLevelAttention:
        if name not in self._mp_attn:
            mod = _NodeLevelAttention(
                self.in_dim, self.out_dim,
                heads=self.num_heads, dropout=self._dropout,
            ).to(device=device, dtype=dtype)
            self._mp_attn[name] = mod
        return self._mp_attn[name]  # type: ignore[return-value]

    def forward(
        self,
        x: torch.Tensor,
        metapath_edge_index_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not metapath_edge_index_dict:
            raise ValueError("metapath_edge_index_dict must be non-empty")
        embeds = []
        for name, ei in metapath_edge_index_dict.items():
            attn = self._get_mp_attn(name, x.device, x.dtype)
            embeds.append(attn(x, ei))
        # Stack to [P, N, D] where P = #metapaths.
        Z = torch.stack(embeds, dim=0)
        # Semantic attention: weight each metapath.
        # Per-metapath summary = mean over nodes.
        summary = Z.mean(dim=1)  # [P, D]
        s = self.semantic(summary).squeeze(-1)  # [P]
        beta = F.softmax(s, dim=0)  # [P]
        out = (beta.view(-1, 1, 1) * Z).sum(dim=0)  # [N, D]
        return out
