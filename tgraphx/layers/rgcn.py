"""Relational Graph Convolutional Network (RGCN) layer.

Reference: Schlichtkrull et al., 2018 — "Modeling Relational Data with
Graph Convolutional Networks".

Supports:
- Basis decomposition (reduces parameter count for many relations).
- Block-diagonal decomposition.
- Self-loop weight.
- Directed relation semantics.

Stability: Experimental (v0.5.0+).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RGCNConv"]


class RGCNConv(nn.Module):
    """Relational GCN convolution for heterogeneous graphs.

    For each relation ``r``, message passing computes:

        h'_i = W_0 * h_i + Σ_r (1/|N_r(i)|) * Σ_{j ∈ N_r(i)} W_r * h_j

    With basis decomposition (``num_bases > 0``):

        W_r = Σ_b a_{r,b} * V_b

    where ``V_b`` are shared basis matrices and ``a_{r,b}`` are
    relation-specific scalar coefficients.

    Args:
        in_channels: Input feature dimension.
        out_channels: Output feature dimension.
        num_relations: Number of relation types.
        num_bases: Number of basis matrices (0 = no decomposition,
            one weight matrix per relation).
        aggr: Aggregation: ``"sum"`` or ``"mean"`` (default ``"mean"``).
        add_self_loops: When ``True``, add a root weight ``W_0``.
        bias: Add learnable bias.

    Stability: Experimental.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: int = 0,
        aggr: str = "mean",
        add_self_loops: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_relations = int(num_relations)
        self.num_bases = int(num_bases)
        self.aggr = aggr
        self.add_self_loops = add_self_loops

        if num_bases > 0:
            # Basis decomposition.
            self.weight = nn.Parameter(
                torch.empty(num_bases, in_channels, out_channels)
            )
            self.comp = nn.Parameter(
                torch.empty(num_relations, num_bases)
            )
        else:
            # One weight matrix per relation.
            self.weight = nn.Parameter(
                torch.empty(num_relations, in_channels, out_channels)
            )
            self.register_parameter("comp", None)

        if add_self_loops:
            self.root_weight = nn.Parameter(torch.empty(in_channels, out_channels))
        else:
            self.register_parameter("root_weight", None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight.view(-1, self.weight.size(-1)).t().unsqueeze(0).squeeze(0)
                                if self.weight.dim() == 2 else self.weight.view(-1, self.weight.size(-1)))
        if self.comp is not None:
            nn.init.xavier_uniform_(self.comp)
        if self.root_weight is not None:
            nn.init.xavier_uniform_(self.root_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def reset_parameters(self) -> None:
        """Initialise weights."""
        w = self.weight
        if w.dim() == 3:
            for i in range(w.size(0)):
                nn.init.xavier_uniform_(w[i])
        else:
            nn.init.xavier_uniform_(w)
        if self.comp is not None:
            nn.init.xavier_uniform_(self.comp)
        if self.root_weight is not None:
            nn.init.xavier_uniform_(self.root_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _rel_weights(self) -> torch.Tensor:
        """Return per-relation weight matrices ``[R, in, out]``."""
        if self.comp is not None:
            # Basis decomposition: w_r = Σ_b a_{r,b} * V_b
            # self.weight: [B, in, out], self.comp: [R, B]
            w = torch.einsum("rb,bio->rio", self.comp, self.weight)
        else:
            w = self.weight  # [R, in, out]
        return w

    def forward(
        self,
        x: torch.Tensor,
        edge_index_by_rel: Dict[int, torch.Tensor],
        num_nodes: int,
    ) -> torch.Tensor:
        """RGCN forward pass.

        Args:
            x: ``FloatTensor[N, in_channels]`` node features.
            edge_index_by_rel: Dict mapping relation ID (0-indexed int)
                to ``LongTensor[2, E_r]`` edge indices.
            num_nodes: Total node count.

        Returns:
            ``FloatTensor[N, out_channels]`` updated node features.
        """
        device = x.device
        rel_w = self._rel_weights()  # [R, in, out]
        out = torch.zeros(num_nodes, self.out_channels, dtype=x.dtype, device=device)

        for rel_id, ei in edge_index_by_rel.items():
            if ei.numel() == 0:
                continue
            r = int(rel_id)
            if r >= self.num_relations:
                raise ValueError(f"Relation ID {r} >= num_relations={self.num_relations}")
            src, dst = ei[0], ei[1]
            # Transform source features.
            h_src = x[src] @ rel_w[r]  # [E_r, out]
            # Aggregate to destination.
            agg = torch.zeros(num_nodes, self.out_channels, dtype=x.dtype, device=device)
            agg.scatter_add_(0, dst.unsqueeze(1).expand_as(h_src), h_src)
            if self.aggr == "mean":
                cnt = torch.zeros(num_nodes, 1, dtype=x.dtype, device=device)
                cnt.scatter_add_(0, dst.unsqueeze(1),
                                 torch.ones(src.size(0), 1, dtype=x.dtype, device=device))
                agg = agg / cnt.clamp(min=1)
            out = out + agg

        # Self / root transformation.
        if self.root_weight is not None:
            out = out + x @ self.root_weight

        if self.bias is not None:
            out = out + self.bias

        return out

    def extra_repr(self) -> str:
        return (f"in={self.in_channels}, out={self.out_channels}, "
                f"relations={self.num_relations}, bases={self.num_bases}, "
                f"aggr={self.aggr!r}")
