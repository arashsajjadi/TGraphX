"""Vector GCNConv — Kipf & Welling (2017).

Implements the renormalised symmetric propagation
``H' = σ( D^{-1/2} (A + I) D^{-1/2} H W )`` for vector node features
``[N, D_in] → [N, D_out]``.

This is a stable v0.3.0 layer — vector node features only, gradient-
tested, OS-agnostic.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GCNConv(nn.Module):
    """Renormalised symmetric GCN convolution for vector node features.

    Args:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
        bias: When ``True`` (default), include a learnable bias.
        add_self_loops: When ``True`` (default), insert ``i->i`` for every
            node prior to normalisation (matches the canonical
            renormalisation trick).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
        self.add_self_loops = bool(add_self_loops)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        if self.lin.bias is not None:
            nn.init.zeros_(self.lin.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(
                f"GCNConv expects vector node features [N, D]; got {tuple(x.shape)}"
            )
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"edge_index must be [2, E]; got {tuple(edge_index.shape)}"
            )
        N = x.size(0)
        device = x.device
        src = edge_index[0]
        dst = edge_index[1]
        if self.add_self_loops:
            self_idx = torch.arange(N, device=device, dtype=torch.long)
            src = torch.cat([src, self_idx], dim=0)
            dst = torch.cat([dst, self_idx], dim=0)
            if edge_weight is not None:
                edge_weight = torch.cat([
                    edge_weight,
                    torch.ones(N, device=device, dtype=edge_weight.dtype),
                ], dim=0)

        if edge_weight is None:
            edge_weight = torch.ones(src.numel(), device=device, dtype=x.dtype)
        else:
            edge_weight = edge_weight.to(dtype=x.dtype)

        # D = degree.  We use 1/sqrt(deg) on each side.
        deg = torch.zeros(N, device=device, dtype=x.dtype)
        deg.index_add_(0, dst, edge_weight)
        deg_inv_sqrt = deg.clamp_min(1e-12).rsqrt()

        norm = deg_inv_sqrt[src] * edge_weight * deg_inv_sqrt[dst]
        # Linear projection happens after aggregation for numerical stability
        # at large N (W applied to compact aggregate).
        h_src = x[src] * norm.unsqueeze(-1)
        out = torch.zeros(N, x.size(1), device=device, dtype=x.dtype)
        out.index_add_(0, dst, h_src)
        return self.lin(out)

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}"
