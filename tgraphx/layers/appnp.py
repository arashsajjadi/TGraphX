"""APPNP propagation (Klicpera et al., 2019).

Approximate personalised PageRank: ``H_{k+1} = (1 - α) Â H_k + α H_0``
with ``Â = D^{-1/2} (A + I) D^{-1/2}``.  Useful as a non-trainable
post-MLP propagation block on vector node features.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class APPNP(nn.Module):
    """Approximate personalised PageRank propagation.

    Args:
        K: Number of propagation steps.
        alpha: Teleport probability ``α ∈ (0, 1]``.
        add_self_loops: Whether to add ``A + I`` (default ``True``).
    """

    def __init__(self, K: int = 10, alpha: float = 0.1, add_self_loops: bool = True) -> None:
        super().__init__()
        if K < 1:
            raise ValueError(f"K must be >= 1; got {K}")
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1]; got {alpha}")
        self.K = int(K)
        self.alpha = float(alpha)
        self.add_self_loops = bool(add_self_loops)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if h.dim() != 2:
            raise ValueError(
                f"APPNP expects vector features [N, D]; got {tuple(h.shape)}"
            )
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"edge_index must be [2, E]; got {tuple(edge_index.shape)}"
            )
        N = h.size(0)
        device = h.device
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
            edge_weight = torch.ones(src.numel(), device=device, dtype=h.dtype)
        else:
            edge_weight = edge_weight.to(dtype=h.dtype)
        deg = torch.zeros(N, device=device, dtype=h.dtype)
        deg.index_add_(0, dst, edge_weight)
        deg_inv_sqrt = deg.clamp_min(1e-12).rsqrt()
        norm = deg_inv_sqrt[src] * edge_weight * deg_inv_sqrt[dst]

        h_0 = h
        out = h
        for _ in range(self.K):
            h_src = out[src] * norm.unsqueeze(-1)
            agg = torch.zeros_like(out)
            agg.index_add_(0, dst, h_src)
            out = (1.0 - self.alpha) * agg + self.alpha * h_0
        return out

    def extra_repr(self) -> str:
        return f"K={self.K}, alpha={self.alpha}"
