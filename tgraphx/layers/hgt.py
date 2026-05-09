"""Heterogeneous Graph Transformer (HGT) layer.

Reference: Hu et al. (WWW 2020) — "Heterogeneous Graph Transformer".

HGT learns relation-specific projections (Q/K/V) per edge type and a
multi-head attention with per-node-type / per-edge-type parameters.
This implementation is a faithful but compact foundation:

* Per-node-type linear projections for Q, K, V.
* Per-edge-type relation matrices applied to K and V (the
  type-specific message transformation).
* Multi-head dot-product attention over edges, normalised per
  destination node.
* A residual connection plus per-node-type output projection.

Inputs:
    ``x_dict`` — ``{node_type: FloatTensor[N_t, in_dim]}``.
    ``edge_index_dict`` — ``{(src_t, rel, dst_t): LongTensor[2, E]}``.

Output:
    ``{node_type: FloatTensor[N_t, out_dim]}``.

This layer is **Experimental**.  The implementation favours clarity and
correctness over peak throughput; a future revision may fuse the
relation matmuls.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HGTConv"]


EdgeType = Tuple[str, str, str]


def _scatter_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    max_val = torch.full((num_nodes,), float("-inf"), dtype=scores.dtype, device=scores.device)
    max_val.scatter_reduce_(0, dst, scores, reduce="amax", include_self=True)
    stable = scores - max_val[dst]
    exp = stable.exp()
    sum_exp = torch.zeros(num_nodes, dtype=scores.dtype, device=scores.device)
    sum_exp.scatter_add_(0, dst, exp)
    return exp / (sum_exp[dst] + 1e-12)


class HGTConv(nn.Module):
    """One layer of Heterogeneous Graph Transformer (HGT).

    Args:
        in_dim: Input feature dimension (shared across node types for
            simplicity; users should pre-project disparate types).
        out_dim: Output feature dimension.
        node_types: List of node type names.
        edge_types: List of ``(src_t, rel, dst_t)`` tuples.
        num_heads: Number of attention heads.
        dropout: Attention dropout probability.

    Stability: Experimental.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        node_types,
        edge_types,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.head_dim = out_dim // num_heads
        self.num_heads = int(num_heads)
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)

        # Per-node-type Q/K/V/Out linears.
        self.q_lin = nn.ModuleDict({nt: nn.Linear(in_dim, out_dim, bias=False) for nt in self.node_types})
        self.k_lin = nn.ModuleDict({nt: nn.Linear(in_dim, out_dim, bias=False) for nt in self.node_types})
        self.v_lin = nn.ModuleDict({nt: nn.Linear(in_dim, out_dim, bias=False) for nt in self.node_types})
        self.out_lin = nn.ModuleDict({nt: nn.Linear(out_dim, out_dim, bias=True) for nt in self.node_types})
        # Residual projection (Identity when in_dim == out_dim).
        if in_dim != out_dim:
            self.res_lin = nn.ModuleDict({nt: nn.Linear(in_dim, out_dim, bias=False) for nt in self.node_types})
        else:
            self.res_lin = None  # type: ignore[assignment]

        # Per-edge-type relation matrices for K and V (per-head).
        self.relation_pri = nn.ParameterDict()
        for et in self.edge_types:
            key = self._etype_key(et)
            # Per-head scalar prior weights (relation importance).
            self.relation_pri[key] = nn.Parameter(torch.ones(num_heads))

        self.dropout = nn.Dropout(dropout)
        self.skip = nn.ParameterDict({
            nt: nn.Parameter(torch.tensor(1.0)) for nt in self.node_types
        })
        self._reset()

    @staticmethod
    def _etype_key(et: EdgeType) -> str:
        return f"{et[0]}__{et[1]}__{et[2]}"

    def _reset(self) -> None:
        for m in self.q_lin.values():
            nn.init.xavier_uniform_(m.weight)
        for m in self.k_lin.values():
            nn.init.xavier_uniform_(m.weight)
        for m in self.v_lin.values():
            nn.init.xavier_uniform_(m.weight)
        for m in self.out_lin.values():
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        if self.res_lin is not None:
            for m in self.res_lin.values():
                nn.init.xavier_uniform_(m.weight)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """HGT forward pass.

        Args:
            x_dict: Node features per type (must include every type in
                ``self.node_types``).
            edge_index_dict: Per-edge-type edge indices.

        Returns:
            Dict of new node features per type.
        """
        for nt in self.node_types:
            if nt not in x_dict:
                raise KeyError(f"x_dict missing node type {nt!r}")

        # Project Q/K/V per node type.
        q_per_type: Dict[str, torch.Tensor] = {}
        k_per_type: Dict[str, torch.Tensor] = {}
        v_per_type: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            x = x_dict[nt]
            N = x.size(0)
            q_per_type[nt] = self.q_lin[nt](x).view(N, self.num_heads, self.head_dim)
            k_per_type[nt] = self.k_lin[nt](x).view(N, self.num_heads, self.head_dim)
            v_per_type[nt] = self.v_lin[nt](x).view(N, self.num_heads, self.head_dim)

        # Aggregate messages per destination type.
        agg_per_dst: Dict[str, torch.Tensor] = {
            nt: torch.zeros(x_dict[nt].size(0), self.num_heads, self.head_dim,
                            dtype=x_dict[nt].dtype, device=x_dict[nt].device)
            for nt in self.node_types
        }
        # Per-dst attention normalisation.
        for et in self.edge_types:
            ei = edge_index_dict.get(et)
            if ei is None or ei.numel() == 0:
                continue
            src_t, _rel, dst_t = et
            key = self._etype_key(et)
            pri = self.relation_pri[key]  # [H]
            src, dst = ei[0], ei[1]
            q_dst = q_per_type[dst_t][dst]  # [E, H, D]
            k_src = k_per_type[src_t][src]  # [E, H, D]
            v_src = v_per_type[src_t][src]  # [E, H, D]
            # Attention score per head: (q · k) / sqrt(D).
            scores = (q_dst * k_src).sum(dim=-1) / math.sqrt(self.head_dim)  # [E, H]
            scores = scores * pri.unsqueeze(0)  # relation prior
            # Softmax per destination node, per head.
            N_dst = x_dict[dst_t].size(0)
            alpha = torch.stack([
                _scatter_softmax(scores[:, h], dst, N_dst) for h in range(self.num_heads)
            ], dim=-1)
            alpha = self.dropout(alpha)
            msg = v_src * alpha.unsqueeze(-1)  # [E, H, D]
            agg_per_dst[dst_t].scatter_add_(
                0,
                dst.view(-1, 1, 1).expand_as(msg),
                msg,
            )

        # Output projection + residual.
        out_dict: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            agg = agg_per_dst[nt].reshape(-1, self.num_heads * self.head_dim)
            out = self.out_lin[nt](agg)
            # Residual: project to out_dim when needed.
            if self.res_lin is not None:
                res = self.res_lin[nt](x_dict[nt])
            else:
                res = x_dict[nt]
            skip = torch.sigmoid(self.skip[nt])
            out = skip * out + (1.0 - skip) * res
            out_dict[nt] = out
        return out_dict
