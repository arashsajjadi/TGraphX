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

Attention modes (``attention_mode`` constructor argument):

* ``"scalar"`` (default) — one scalar score per ``(edge, head)``.
  Spatial dims are mean-pooled before the attention vector dot product.
  Backward-compatible with all previous releases.

* ``"channel"`` (experimental) — one score per ``(edge, head, channel)``.
  Spatial dims are still mean-pooled, but the final sum over channels is
  dropped so that ``a_src`` / ``a_dst`` act as per-channel attention
  weights rather than a single projected score.  Memory cost for the score
  tensor scales as ``E × K × C_head`` instead of ``E × K``.
  Mark: **🧪 Experimental** — API may change in future releases.

Chunked forward (``chunk_size`` argument to ``forward()``):

When ``chunk_size=K`` is set, edges are processed in groups of ``K`` using
a two-pass algorithm:

* **Pass 1** — collect per-destination/head max for log-sum-exp stability.
* **Pass 2** — accumulate exp-weighted values and the denominator in one pass.

The output matches unchunked within floating-point tolerance.  Memory use
for intermediate edge tensors scales as ``O(K × K_heads × C_head × spatial)``
instead of ``O(E × ...)``.  ``chunk_size=None`` (default) is unchanged.

Edge feature formats (when ``use_edge_features=True``):

* **Vector** ``[E, edge_dim]`` — projected directly to ``[E, num_heads]``
  attention bias.
* **Matching-rank spatial** — mean-pooled over spatial dims before the bias
  projection (so spatial edge dims need not match node spatial dims).
* **Mismatched-rank spatial** raises ``NotImplementedError``.

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
        attention_mode: str = "scalar",
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1; got {num_heads}")
        if attention_mode not in ("scalar", "channel"):
            raise ValueError(
                f"attention_mode must be 'scalar' or 'channel'; got {attention_mode!r}. "
                f"'spatial' (per-pixel) and 'voxel' (per-voxel) modes are planned for a "
                f"future release; their O(E×K×H×W) score tensors are currently "
                f"prohibitively memory-intensive for typical spatial GNN workloads."
            )
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
        self.attention_mode = attention_mode

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
        chunk_size: int | None = None,
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
                    "TensorGATLayer expects edge_features of shape "
                    "[E, edge_dim] or [E, edge_dim, " +
                    ("H, W]" if rank == 2 else "D, H, W]") +
                    f"; got {tuple(edge_features.shape)}."
                )

        N = x.size(0)
        spatial = x.shape[2:]  # (H, W) or (D, H, W)
        src = edge_index[0]
        dst = edge_index[1]
        E_orig = src.size(0)

        # Self-loop insertion.
        n_new = 0
        if self.add_self_loops:
            self_mask = edge_index[0] == edge_index[1]
            already = torch.zeros(N, dtype=torch.bool, device=x.device)
            if self_mask.any():
                already[edge_index[0][self_mask]] = True
            new_nodes = torch.arange(N, device=x.device, dtype=torch.long)[~already]
            n_new = new_nodes.numel()
            if n_new > 0:
                src = torch.cat([src, new_nodes], dim=0)
                dst = torch.cat([dst, new_nodes], dim=0)
        E_eff = src.size(0)

        # Pad edge_pool for new self-loops (zero bias).
        if self.use_edge_features and edge_pool is not None and self.add_self_loops and n_new > 0:
            pad = edge_pool.new_zeros(n_new, edge_pool.size(1))
            edge_pool = torch.cat([edge_pool, pad], dim=0)

        # Linear projection: [N, K * C_head, *spatial] -> [N, K, C_head, *spatial].
        h = self.W(x).view(N, self.num_heads, self.head_channels, *spatial)

        # Validate edge_weight before use.
        if edge_weight is not None:
            broadcast_edge_weight(edge_weight, x, num_edges=E_orig)

        # Resolve full edge_weight (with padding for new self-loops).
        full_w: torch.Tensor | None = None
        if edge_weight is not None:
            if self.add_self_loops and n_new > 0:
                full_w = torch.cat([edge_weight, edge_weight.new_ones(n_new)], dim=0)
            else:
                full_w = edge_weight

        # Route to chunked or unchunked path.
        if chunk_size is not None and E_eff > chunk_size:
            out_per_head, attn = self._chunked_forward(
                h, src, dst, N, E_eff, rank, spatial,
                edge_pool, full_w, chunk_size, return_attention,
            )
        else:
            out_per_head, attn = self._unchunked_forward(
                h, src, dst, N, E_eff, rank, spatial, edge_pool, full_w,
            )

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

    # ------------------------------------------------------------------ #
    # Internal attention helpers                                           #
    # ------------------------------------------------------------------ #

    def _compute_scores(
        self,
        h_src_pool: torch.Tensor,   # [E_or_chunk, K, C_head]
        h_dst_pool: torch.Tensor,
        edge_pool_c: torch.Tensor | None,  # [E_or_chunk, K] or None
    ) -> torch.Tensor:
        """Compute pre-activation attention scores for a batch of edges.

        Returns shape:
          * ``"scalar"`` mode: ``[E, K]``
          * ``"channel"`` mode: ``[E, K, C_head]``
        """
        if self.attention_mode == "scalar":
            score_src = (h_src_pool * self.a_src.unsqueeze(0)).sum(dim=-1)  # [E, K]
            score_dst = (h_dst_pool * self.a_dst.unsqueeze(0)).sum(dim=-1)
            scores_pre = score_src + score_dst
            if edge_pool_c is not None:
                scores_pre = scores_pre + edge_pool_c
            return F.leaky_relu(scores_pre, negative_slope=self.negative_slope)
        else:  # "channel"
            # Element-wise per-channel (no final sum over C_head).
            score_src = h_src_pool * self.a_src.unsqueeze(0)  # [E, K, C_head]
            score_dst = h_dst_pool * self.a_dst.unsqueeze(0)
            scores_pre = score_src + score_dst
            if edge_pool_c is not None:
                # edge_pool_c is [E, K] — broadcast over C_head.
                scores_pre = scores_pre + edge_pool_c.unsqueeze(-1)
            return F.leaky_relu(scores_pre, negative_slope=self.negative_slope)

    def _get_edge_pool_chunk(
        self,
        edge_pool: torch.Tensor | None,
        start: int,
        end: int,
    ) -> torch.Tensor | None:
        if edge_pool is None:
            return None
        ep = edge_pool[start:end]
        return self.edge_bias_proj(ep)  # [chunk, K]

    def _weighted_values(
        self,
        attn: torch.Tensor,       # [E, K] or [E, K, C_head]
        h_src: torch.Tensor,      # [E, K, C_head, *spatial]
        weight_c: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute attn-weighted values; cast attn to h_src.dtype (AMP safety)."""
        attn_cast = attn.to(dtype=h_src.dtype)
        if self.attention_mode == "scalar":
            trailing = (1,) * (h_src.dim() - attn_cast.dim())
            weighted = attn_cast.view(*attn_cast.shape, *trailing) * h_src
        else:  # "channel" — attn is [E, K, C_head]
            trailing = (1,) * len(h_src.shape[3:])  # spatial dims only
            weighted = attn_cast.view(*attn_cast.shape, *trailing) * h_src
        if weight_c is not None:
            weighted = weighted * weight_c
        return weighted

    def _unchunked_forward(
        self,
        h: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        N: int,
        E_eff: int,
        rank: int,
        spatial: tuple,
        edge_pool: torch.Tensor | None,
        full_w: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard single-pass forward (original algorithm)."""
        h_src = h.index_select(0, src)  # [E_eff, K, C_head, *spatial]
        h_dst = h.index_select(0, dst)

        weight_b: torch.Tensor | None = None
        if full_w is not None:
            weight_b = broadcast_edge_weight(full_w, h_src, num_edges=E_eff)

        h_src_pool = mean_over_spatial(h_src, rank)  # [E_eff, K, C_head]
        h_dst_pool = mean_over_spatial(h_dst, rank)

        edge_bias = (
            self.edge_bias_proj(edge_pool) if edge_pool is not None else None
        )  # [E_eff, K] or None

        scores = self._compute_scores(h_src_pool, h_dst_pool, edge_bias)
        attn = edge_softmax(scores, dst, N)  # [E_eff, K] or [E_eff, K, C_head]

        if self.attn_dropout_p > 0.0 and self.training:
            attn_dropped = F.dropout(attn, p=self.attn_dropout_p, training=True)
        else:
            attn_dropped = attn

        weighted = self._weighted_values(attn_dropped, h_src, weight_b)
        out_per_head = h.new_zeros((N, self.num_heads, self.head_channels, *spatial))
        out_per_head.index_add_(0, dst, weighted)
        return out_per_head, attn

    def _chunked_forward(
        self,
        h: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        N: int,
        E_eff: int,
        rank: int,
        spatial: tuple,
        edge_pool: torch.Tensor | None,
        full_w: torch.Tensor | None,
        chunk_size: int,
        return_attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Two-pass chunked forward for memory-efficient GAT.

        Memory savings: peak edge-buffer goes from O(E × K × C_head × spatial)
        to O(chunk × K × C_head × spatial).  The value tensor H [N, K, C_head,
        *spatial] is kept in memory throughout; savings are greatest when E ≫ N.

        Algorithm
        ---------
        Pass 1  — accumulate per-destination/head max over all edge chunks
                  (log-sum-exp stability constant, detached from grad graph).
        Pass 2  — recompute logits chunk-by-chunk:
                    exp_c  = exp(score_c − max[dst_c])          per chunk
                    sum_exp[dst] += scatter(exp_c)
                    out[dst] += scatter(exp_c_broadcast × h_src_c × w_c)
        Normalise — out = out / sum_exp.

        Return attention weights only when ``return_attention=True``; this
        requires an additional [E_eff, K] buffer but avoids a third pass.
        """
        K = self.num_heads
        C = self.head_channels

        # ── Pass 1: max_per_dest ─────────────────────────────────────────────
        # Shape: [N, K] for scalar; [N, K, C_head] for channel.
        if self.attention_mode == "scalar":
            max_shape = (N, K)
        else:
            max_shape = (N, K, C)
        max_per_dest = h.new_full(max_shape, float("-inf"))

        for start in range(0, E_eff, chunk_size):
            end = min(start + chunk_size, E_eff)
            src_c = src[start:end]
            dst_c = dst[start:end]
            h_src_c = h.index_select(0, src_c)
            h_dst_c = h.index_select(0, dst_c)
            h_src_pool_c = mean_over_spatial(h_src_c, rank)
            h_dst_pool_c = mean_over_spatial(h_dst_c, rank)
            ep_c = self._get_edge_pool_chunk(edge_pool, start, end)
            scores_c = self._compute_scores(h_src_pool_c, h_dst_pool_c, ep_c)
            # Cast to max_per_dest dtype: under AMP a_src/a_dst (fp32 params)
            # promote scores to fp32 even if h is bf16.
            scores_c_det = scores_c.detach().to(dtype=max_per_dest.dtype)
            # Running max per destination.
            if self.attention_mode == "scalar":
                tgt_b = dst_c.unsqueeze(1).expand_as(scores_c_det)
            else:
                tgt_b = dst_c.view(-1, 1, 1).expand_as(scores_c_det)
            max_per_dest.scatter_reduce_(
                0, tgt_b, scores_c_det, reduce="amax", include_self=True
            )

        max_per_dest = max_per_dest.detach()  # stability constant — no gradients

        # ── Pass 2: sum_exp + weighted value accumulation ────────────────────
        sum_exp = h.new_zeros(max_shape)  # [N, K] or [N, K, C]
        out_per_head = h.new_zeros((N, K, C, *spatial))

        # Optional: collect exp_scores for return_attention.
        all_exp: torch.Tensor | None = (
            h.new_zeros(E_eff, K) if (return_attention and self.attention_mode == "scalar") else
            h.new_zeros(E_eff, K, C) if (return_attention and self.attention_mode == "channel") else
            None
        )

        for start in range(0, E_eff, chunk_size):
            end = min(start + chunk_size, E_eff)
            n_c = end - start
            src_c = src[start:end]
            dst_c = dst[start:end]
            h_src_c = h.index_select(0, src_c)          # [chunk, K, C, *spatial]
            h_dst_c = h.index_select(0, dst_c)
            h_src_pool_c = mean_over_spatial(h_src_c, rank)
            h_dst_pool_c = mean_over_spatial(h_dst_c, rank)
            ep_c = self._get_edge_pool_chunk(edge_pool, start, end)
            scores_c = self._compute_scores(h_src_pool_c, h_dst_pool_c, ep_c)

            # Globally normalised exp (shifted by per-destination max).
            # Cast max to scores dtype to avoid mismatch under AMP.
            if self.attention_mode == "scalar":
                max_c = max_per_dest.index_select(0, dst_c).to(dtype=scores_c.dtype)
            else:
                max_c = max_per_dest.index_select(0, dst_c).to(dtype=scores_c.dtype)
            exp_c = (scores_c - max_c).exp()  # [chunk, K] or [chunk, K, C]

            # Attention dropout.
            if self.attn_dropout_p > 0.0 and self.training:
                exp_c_drop = F.dropout(exp_c, p=self.attn_dropout_p, training=True)
            else:
                exp_c_drop = exp_c

            # Accumulate sum_exp.  Cast exp to sum_exp.dtype (AMP-safe).
            exp_c_acc = exp_c_drop.to(dtype=sum_exp.dtype)
            if self.attention_mode == "scalar":
                tgt_b = dst_c.unsqueeze(1).expand_as(exp_c_acc)
            else:
                tgt_b = dst_c.view(-1, 1, 1).expand_as(exp_c_acc)
            sum_exp.index_add_(0, dst_c, exp_c_acc)

            # Edge weight for this chunk.
            w_c: torch.Tensor | None = None
            if full_w is not None:
                w_chunk = full_w[start:end].to(dtype=h_src_c.dtype)
                w_c = w_chunk.view(n_c, *(1,) * (h_src_c.dim() - 1))

            # Weighted value accumulation.
            weighted_c = self._weighted_values(exp_c_drop, h_src_c, w_c)
            out_per_head.index_add_(0, dst_c, weighted_c)

            # Store exp scores for return_attention.
            if all_exp is not None:
                all_exp[start:end] = exp_c_drop.detach()

        # ── Normalise ────────────────────────────────────────────────────────
        denom = sum_exp.clamp_min(1e-16)
        if self.attention_mode == "scalar":
            denom_b = denom.view(N, K, 1, *(1,) * rank)  # broadcast over C, spatial
        else:
            denom_b = denom.view(N, K, C, *(1,) * rank)  # broadcast over spatial
        out_per_head = out_per_head / denom_b

        # Build return attention [E, K] or [E, K, C].
        attn: torch.Tensor | None = None
        if return_attention and all_exp is not None:
            denom_per_edge = denom.index_select(0, dst)  # [E, K] or [E, K, C]
            attn = all_exp / denom_per_edge.clamp_min(1e-16)
        elif not return_attention:
            attn = None  # caller ignores it

        return out_per_head, attn

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"num_heads={self.num_heads}, concat_heads={self.concat_heads}, "
            f"head_channels={self.head_channels}, spatial_rank={self.spatial_rank}, "
            f"attention_mode={self.attention_mode!r}"
        )
