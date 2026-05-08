"""Experimental GraphTransformerLayer — vector-only global self-attention.

.. experimental::
    This layer is **🧪 Experimental**.  API may change in future releases.
    Only **vector node features** ``[N, D]`` are supported.
    Tensor-aware spatial / volumetric inputs are deferred.

The layer implements the Transformer architecture from
Dwivedi & Bresson (2020) "A Generalization of Transformers to Graphs":

    Q = x W_Q           [N, D]
    K = x W_K           [N, D]
    V = x W_V           [N, D]
    A = softmax( Q K^T / sqrt(d_head) )   [N, N]   — global, O(N²)
    out = A V                              [N, D]
    out = LN( x + out )                   residual + layer norm
    out = LN( out + FFN(out) )            feed-forward + layer norm

``A[i, j]`` is the attention from node ``i`` to node ``j`` over the whole
graph (not restricted to edges).  This makes the layer **O(N²)** in
memory and compute.

For edge-conditioned or linear-complexity variants (not yet implemented),
supply ``edge_index`` in a future release.

.. warning::
    **O(N²) attention.** A ``UserWarning`` is emitted for N > 1 000.

Factory / config usage::

    from tgraphx.layers.graph_transformer import GraphTransformerLayer

    layer = GraphTransformerLayer(
        in_dim=64,
        out_dim=64,
        num_heads=4,
        ffn_dim=128,
        dropout=0.1,
    )
    out = layer(x)   # x: [N, 64]
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

_N_WARN = 1_000

__all__ = ["GraphTransformerLayer"]


class GraphTransformerLayer(nn.Module):
    """🧪 Experimental: global self-attention transformer layer for graphs.

    Operates on **vector node features** ``[N, in_dim]``.  Tensor-aware
    (spatial / volumetric) support is deferred to a future release.

    Args:
        in_dim: Input node feature dimension.
        out_dim: Output node feature dimension.
        num_heads: Number of attention heads.  ``out_dim`` must be divisible
            by ``num_heads``.
        ffn_dim: Hidden dimension of the feed-forward sublayer.  Defaults
            to ``4 * out_dim``.
        dropout: Dropout applied to attention weights and after the FFN
            (default ``0.0``).
        attention_dropout: Dropout applied only to attention weights
            (default ``0.0``).  Added to ``dropout`` if both are non-zero.
        residual: If ``True`` (default), add a skip connection from input
            to output of the multi-head attention sublayer.  A learnable
            linear projection is added when ``in_dim != out_dim``.
        layer_norm: If ``True`` (default), apply LayerNorm after each
            sublayer.
        bias: If ``True`` (default), add bias to Q, K, V projections.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        residual: bool = True,
        layer_norm: bool = True,
        bias: bool = True,
        edge_bias: bool = False,
        positional_encoding: str | None = None,
        pe_dim: int = 0,
    ) -> None:
        """
        Additional v0.2.7 arguments
        ---------------------------
        edge_bias: If ``True``, ``forward`` accepts an optional
            ``edge_bias_dense`` argument (shape ``[N, N]`` or
            ``[num_heads, N, N]``) that is added to the pre-softmax
            attention logits.  Use :func:`build_adjacency_bias` from
            :mod:`tgraphx.layers.transformer_encodings` to construct one
            from ``edge_index``.

        positional_encoding: ``None`` (default), ``"degree"``, or
            ``"laplacian"``.  When set, ``forward`` accepts an optional
            ``edge_index`` and computes a per-node positional encoding
            that is added to the input projection.  Users can also
            pre-compute encodings and pass them via the ``positional``
            argument to ``forward`` (zero dependencies).

        pe_dim: Dimension of the positional encoding when
            ``positional_encoding`` is set.  Must be > 0 when
            ``positional_encoding`` is given; ignored otherwise.
        """
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError(
                f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})."
            )
        if positional_encoding not in (None, "degree", "laplacian"):
            raise ValueError(
                f"positional_encoding must be None, 'degree', or 'laplacian'; "
                f"got {positional_encoding!r}"
            )
        if positional_encoding is not None and pe_dim <= 0:
            raise ValueError(
                f"pe_dim must be > 0 when positional_encoding is set; got {pe_dim}"
            )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_p = float(dropout)
        self.attn_dropout_p = float(attention_dropout)
        self.residual = residual
        self.edge_bias = bool(edge_bias)
        self.positional_encoding = positional_encoding
        self.pe_dim = int(pe_dim)
        if positional_encoding is not None:
            # Project encoding into the model dim so it can be summed with
            # the input projection.
            self.pe_proj = nn.Linear(pe_dim, in_dim, bias=False)
        else:
            self.pe_proj = None

        ffn_dim = ffn_dim or out_dim * 4

        self.W_Q = nn.Linear(in_dim, out_dim, bias=bias)
        self.W_K = nn.Linear(in_dim, out_dim, bias=bias)
        self.W_V = nn.Linear(in_dim, out_dim, bias=bias)
        self.W_O = nn.Linear(out_dim, out_dim, bias=bias)

        self.ffn = nn.Sequential(
            nn.Linear(out_dim, ffn_dim, bias=bias),
            nn.GELU(),
            nn.Dropout(p=self.dropout_p) if self.dropout_p > 0 else nn.Identity(),
            nn.Linear(ffn_dim, out_dim, bias=bias),
        )

        if residual and in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.res_proj = None

        self.norm1 = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()
        self.drop = nn.Dropout(p=self.dropout_p) if self.dropout_p > 0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.W_O.weight)
        if self.res_proj is not None:
            nn.init.xavier_uniform_(self.res_proj.weight)
        if self.pe_proj is not None:
            nn.init.xavier_uniform_(self.pe_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        positional: torch.Tensor | None = None,
        edge_bias_dense: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Global self-attention forward pass.

        Args:
            x: ``[N, in_dim]`` node features.
            edge_index: Accepted for API consistency but **currently
                ignored** (global attention is always applied).  A future
                release may use edge_index for edge-conditioned attention.

        Returns:
            ``[N, out_dim]`` updated node features.

        Raises:
            ValueError: If ``x`` is not 2-D or the feature dim does not
                match ``in_dim``.
        """
        if x.dim() != 2:
            raise ValueError(
                f"GraphTransformerLayer expects 2-D vector input [N, D]; "
                f"got {tuple(x.shape)}. "
                f"Tensor-aware (spatial/volumetric) input is not yet supported."
            )
        if x.size(1) != self.in_dim:
            raise ValueError(
                f"Feature dim {x.size(1)} != in_dim {self.in_dim}."
            )
        N = x.size(0)
        if N > _N_WARN:
            warnings.warn(
                f"GraphTransformerLayer: N={N} > {_N_WARN}. "
                f"Global self-attention allocates an O(N²) attention matrix "
                f"({N}×{N} per head).  Consider subgraph sampling or a "
                f"linear-complexity variant for large graphs.",
                stacklevel=2,
            )

        # Optional positional encoding addition.
        if self.pe_proj is not None and positional is not None:
            if positional.shape != (N, self.pe_dim):
                raise ValueError(
                    f"positional must have shape [N, pe_dim]={(N, self.pe_dim)}; "
                    f"got {tuple(positional.shape)}"
                )
            x = x + self.pe_proj(positional)

        K = self.num_heads
        d = self.head_dim

        Q = self.W_Q(x).view(N, K, d)          # [N, K, d]
        Kt = self.W_K(x).view(N, K, d)
        V = self.W_V(x).view(N, K, d)

        # Scaled dot-product attention: [K, N, d] @ [K, d, N] → [K, N, N]
        Q_t = Q.transpose(0, 1)                 # [K, N, d]
        K_t = Kt.transpose(0, 1)                # [K, N, d]
        V_t = V.transpose(0, 1)                 # [K, N, d]

        attn_logits = torch.bmm(Q_t, K_t.transpose(1, 2)) * self.scale  # [K, N, N]

        # Optional structural / edge bias added to logits before softmax.
        if self.edge_bias and edge_bias_dense is not None:
            if edge_bias_dense.dim() == 2:
                if edge_bias_dense.shape != (N, N):
                    raise ValueError(
                        f"edge_bias_dense must be [N, N]={(N, N)} or "
                        f"[K, N, N]; got {tuple(edge_bias_dense.shape)}"
                    )
                attn_logits = attn_logits + edge_bias_dense.unsqueeze(0)
            elif edge_bias_dense.dim() == 3:
                if edge_bias_dense.shape != (K, N, N):
                    raise ValueError(
                        f"edge_bias_dense [K,N,N] expected {(K, N, N)}; "
                        f"got {tuple(edge_bias_dense.shape)}"
                    )
                attn_logits = attn_logits + edge_bias_dense
            else:
                raise ValueError(
                    f"edge_bias_dense must be 2-D or 3-D; got {edge_bias_dense.dim()}-D"
                )

        attn = F.softmax(attn_logits, dim=-1)   # [K, N, N]

        if self.attn_dropout_p > 0 and self.training:
            attn = F.dropout(attn, p=self.attn_dropout_p, training=True)

        out = torch.bmm(attn, V_t)              # [K, N, d]
        out = out.transpose(0, 1).reshape(N, self.out_dim)  # [N, K*d]
        out = self.drop(self.W_O(out))

        # Residual + layer norm (sublayer 1: attention)
        if self.residual:
            skip = self.res_proj(x) if self.res_proj is not None else x
            out = self.norm1(skip + out)
        else:
            out = self.norm1(out)

        # FFN sublayer
        ffn_out = self.drop(self.ffn(out))
        if self.residual:
            out = self.norm2(out + ffn_out)
        else:
            out = self.norm2(ffn_out)

        return out

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
            f"num_heads={self.num_heads}, head_dim={self.head_dim}"
        )
