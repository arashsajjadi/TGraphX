"""TGAT-style temporal attention layer (foundation).

Reference: Xu et al. (ICLR 2020) — "Inductive Representation Learning
on Temporal Graphs".

TGAT augments standard graph attention with a *time encoding* on each
edge:

    score(i, j, t) = softmax_j ( q_i · k_j  +  q_i · k_t )

where ``k_t = W_t · phi(t_i - t_j)`` is the projected time-difference
encoding, and ``phi`` is a sinusoidal or Time2Vec encoder.

This implementation is a foundation: it wires the time encoding into a
multi-head attention layer and aggregates messages from temporal
neighbours.  Caller is responsible for filtering edges to those
respecting the cutoff time (``temporal_neighbor_sample`` in
``tgraphx.temporal_sampling``).

Stability: Experimental.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .time_encoding import LearnableTimeEncoding, sinusoidal_time_encoding

__all__ = ["TGATConv"]


def _scatter_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    max_val = torch.full((num_nodes,), float("-inf"), dtype=scores.dtype, device=scores.device)
    max_val.scatter_reduce_(0, dst, scores, reduce="amax", include_self=True)
    stable = scores - max_val[dst]
    exp = stable.exp()
    sum_exp = torch.zeros(num_nodes, dtype=scores.dtype, device=scores.device)
    sum_exp.scatter_add_(0, dst, exp)
    return exp / (sum_exp[dst] + 1e-12)


class TGATConv(nn.Module):
    """Temporal attention convolution.

    Args:
        in_dim: Input feature dimension (shared src/dst).
        out_dim: Output feature dimension.
        time_dim: Time encoding dimension.
        num_heads: Attention heads (must divide ``out_dim``).
        time_encoding: ``"sinusoidal"`` (default, parameter-free) or
            ``"learnable"`` (Time2Vec).
        dropout: Attention dropout.

    Forward:
        ``x`` — ``FloatTensor[N, in_dim]`` node features.
        ``edge_index`` — ``LongTensor[2, E]`` (src → dst).
        ``edge_time`` — ``FloatTensor[E]`` event time per edge.
        ``query_time`` — ``FloatTensor[N]`` cutoff time per destination
            node (inference time).  ``Δt = query_time[dst] - edge_time``
            is fed through the time encoder.

    Stability: Experimental.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        time_dim: int = 16,
        num_heads: int = 2,
        time_encoding: str = "sinusoidal",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.time_dim = int(time_dim)
        self.num_heads = int(num_heads)
        self.head_dim = out_dim // num_heads
        if time_encoding not in ("sinusoidal", "learnable"):
            raise ValueError(
                f"time_encoding must be 'sinusoidal' or 'learnable'; got {time_encoding!r}"
            )
        self.time_encoding_kind = time_encoding
        if time_encoding == "learnable":
            self.time_enc: Optional[nn.Module] = LearnableTimeEncoding(time_dim)
        else:
            self.time_enc = None  # use functional encoder

        # Q comes from dst; K, V come from src; time encoding is added to K.
        self.q_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.k_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.v_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.t_lin = nn.Linear(time_dim, out_dim, bias=False)
        self.out_lin = nn.Linear(out_dim, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        for m in (self.q_lin, self.k_lin, self.v_lin, self.t_lin):
            nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(self.out_lin.weight)
        nn.init.zeros_(self.out_lin.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_time: torch.Tensor,
        query_time: torch.Tensor,
    ) -> torch.Tensor:
        N = x.size(0)
        if edge_index.numel() == 0:
            # No temporal neighbours: identity output projection.
            return self.out_lin(self.v_lin(x))
        src, dst = edge_index[0], edge_index[1]
        # Time encoding on Δt = query_time[dst] - edge_time.
        dt = query_time[dst] - edge_time
        if self.time_enc is not None:
            t_enc = self.time_enc(dt)
        else:
            t_enc = sinusoidal_time_encoding(dt, self.time_dim)

        H, D = self.num_heads, self.head_dim
        q = self.q_lin(x).view(N, H, D)
        k = self.k_lin(x).view(N, H, D)
        v = self.v_lin(x).view(N, H, D)
        kt = self.t_lin(t_enc).view(-1, H, D)

        q_dst = q[dst]            # [E, H, D]
        k_src = k[src] + kt       # [E, H, D]
        v_src = v[src]            # [E, H, D]
        scores = (q_dst * k_src).sum(dim=-1) / math.sqrt(self.head_dim)  # [E, H]
        # Per-dst softmax per head.
        alpha = torch.stack([
            _scatter_softmax(scores[:, h], dst, N) for h in range(H)
        ], dim=-1)
        alpha = self.dropout(alpha)
        msg = v_src * alpha.unsqueeze(-1)
        agg = torch.zeros(N, H, D, dtype=x.dtype, device=x.device)
        agg.scatter_add_(0, dst.view(-1, 1, 1).expand_as(msg), msg)
        out = self.out_lin(agg.reshape(N, H * D))
        return out
