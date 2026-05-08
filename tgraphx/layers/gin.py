"""Tensor-aware GIN / GINEConv layer for 2-D or 3-D node features.

Adapts Xu et al. (2019) Graph Isomorphism Network to spatial / volumetric
features:

    h_j' = MLP( (1 + ε) · h_j + Σ_{i ∈ N(j)} h_i )

For tensor features, ``MLP`` is a small 1×1 (or 1×1×1) ``Conv2d`` /
``Conv3d`` block by default so the spatial layout ``[C, H, W]`` (rank 2)
or ``[C, D, H, W]`` (rank 3) is preserved.  Users may pass any custom
module as the MLP as long as it preserves the leading ``[N, ...]`` shape
and maps ``in_channels`` to ``out_channels``.

When ``use_edge_features=True``, this becomes a tensor-aware GINEConv:

    h_j' = MLP( (1 + ε) · h_j + Σ_i ReLU( h_i + φ(e_ij) ) )

The edge projection ``φ`` adapts to the input format:

* ``edge_features_kind="spatial"`` (default) — ``e_ij`` has shape
  ``[E, edge_dim, *spatial]`` matching the layer's ``spatial_rank``;
  ``φ`` is a 1×1 (or 1×1×1) convolution mapping ``edge_dim → in_channels``
  (or identity when ``edge_dim == in_channels``).
* ``edge_features_kind="vector"``  — ``e_ij`` has shape ``[E, edge_dim]``;
  ``φ`` is ``nn.Linear(edge_dim, in_channels)`` followed by an unsqueeze
  to ``[E, in_channels, 1, ...]`` (one ``1`` per spatial dim) so the bias
  broadcasts over the spatial grid.

``edge_weight`` (``[E]``) scales each neighbour message before scatter-sum.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._dim import (
    batchnorm,
    conv1x1,
    expected_x_dim,
    trailing_ones,
    validate_spatial_rank,
)
from ._scatter import broadcast_edge_weight, scatter_sum


class TensorGINLayer(nn.Module):
    """Tensor-aware GIN / GINEConv layer for 2-D or 3-D node features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | None = None,
        eps: float = 0.0,
        train_eps: bool = False,
        use_batchnorm: bool = False,
        mlp: nn.Module | None = None,
        use_edge_features: bool = False,
        edge_dim: int | None = None,
        edge_features_kind: str = "spatial",
        spatial_rank: int = 2,
    ) -> None:
        super().__init__()
        if use_edge_features and edge_dim is None:
            raise ValueError("edge_dim must be set when use_edge_features=True")
        if edge_features_kind not in ("spatial", "vector"):
            raise ValueError(
                f"edge_features_kind must be 'spatial' or 'vector'; got "
                f"{edge_features_kind!r}."
            )
        validate_spatial_rank(spatial_rank)
        if hidden_channels is None:
            hidden_channels = out_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_edge_features = use_edge_features
        self.edge_dim = edge_dim
        self.edge_features_kind = edge_features_kind
        self.spatial_rank = spatial_rank

        if train_eps:
            self.eps = nn.Parameter(torch.tensor(float(eps)))
        else:
            self.register_buffer("eps", torch.tensor(float(eps)))

        if mlp is None:
            layers: list[nn.Module] = [
                conv1x1(spatial_rank, in_channels, hidden_channels)
            ]
            if use_batchnorm:
                layers.append(batchnorm(spatial_rank, hidden_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(conv1x1(spatial_rank, hidden_channels, out_channels))
            if use_batchnorm:
                layers.append(batchnorm(spatial_rank, out_channels))
            self.mlp = nn.Sequential(*layers)
        else:
            self.mlp = mlp

        if use_edge_features:
            if edge_features_kind == "spatial":
                if edge_dim == in_channels:
                    self.edge_proj: nn.Module = nn.Identity()
                else:
                    self.edge_proj = conv1x1(spatial_rank, edge_dim, in_channels)
            else:  # "vector"
                self.edge_proj = nn.Linear(edge_dim, in_channels)
        else:
            self.edge_proj = nn.Identity()  # unused; kept for state-dict stability

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Message-passing forward with optional edge chunking.

        Args:
            x: Node features ``[N, C, *spatial]``.
            edge_index: ``[2, E]`` LongTensor.
            edge_features: Optional spatial or vector edge features.
            edge_weight: Optional ``[E]`` per-edge scalar weights.
            chunk_size: If set, process edges in chunks of this size to reduce
                peak message-buffer memory.  ``None`` (default) uses the
                standard single-pass path.  The sum aggregation is exact up
                to floating-point associativity — output matches unchunked
                within float32 rounding.

        Returns:
            Updated node features ``[N, out_channels, *spatial]``.
        """
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
        if self.use_edge_features and edge_features is None:
            raise ValueError(
                "Layer was constructed with use_edge_features=True, but "
                "edge_features was None."
            )
        if (not self.use_edge_features) and edge_features is not None:
            raise ValueError(
                "Layer was constructed with use_edge_features=False; do not "
                "pass edge_features."
            )

        N = x.size(0)
        src = edge_index[0]
        dst = edge_index[1]
        E = src.size(0)

        if self.use_edge_features and edge_features is not None:
            if edge_features.size(0) != E:
                raise ValueError(
                    f"edge_features has {edge_features.size(0)} rows but "
                    f"edge_index has {E} edges."
                )
            if self.edge_features_kind == "spatial":
                expected_dim = 2 + rank
                spatial_str = "[E, edge_dim, H, W]" if rank == 2 else "[E, edge_dim, D, H, W]"
                if edge_features.dim() != expected_dim:
                    raise ValueError(
                        f"edge_features must have shape {spatial_str} when "
                        f"edge_features_kind='spatial' and spatial_rank={rank}; "
                        f"got {tuple(edge_features.shape)}."
                    )
                if edge_features.size(1) != self.edge_dim:
                    raise ValueError(
                        f"edge_features channel count {edge_features.size(1)} "
                        f"does not match edge_dim={self.edge_dim}."
                    )
            else:  # "vector"
                if edge_features.dim() != 2:
                    raise ValueError(
                        f"edge_features must have shape [E, edge_dim] when "
                        f"edge_features_kind='vector'; got "
                        f"{tuple(edge_features.shape)}."
                    )
                if edge_features.size(1) != self.edge_dim:
                    raise ValueError(
                        f"edge_features last-dim {edge_features.size(1)} does "
                        f"not match edge_dim={self.edge_dim}."
                    )

        if chunk_size is not None and E > chunk_size:
            agg = self._chunked_forward(
                x, src, dst, N, E, rank, edge_features, edge_weight, chunk_size
            )
        else:
            agg = self._full_forward(x, src, dst, N, E, rank, edge_features, edge_weight)

        combined = (1.0 + self.eps) * x + agg
        return self.mlp(combined)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _compute_chunk_messages(
        self,
        x: torch.Tensor,
        src_c: torch.Tensor,
        ef_c: torch.Tensor | None,
        n_c: int,
        rank: int,
    ) -> torch.Tensor:
        """Compute per-edge messages for one chunk."""
        if self.use_edge_features and ef_c is not None:
            if self.edge_features_kind == "spatial":
                edge_term = self.edge_proj(ef_c)
            else:  # "vector"
                edge_vec = self.edge_proj(ef_c)
                view = (n_c, self.in_channels) + trailing_ones(rank)
                edge_term = edge_vec.view(view)
            return F.relu(x.index_select(0, src_c) + edge_term)
        return x.index_select(0, src_c)

    def _full_forward(
        self,
        x: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        N: int,
        E: int,
        rank: int,
        edge_features: torch.Tensor | None,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        """Standard single-pass sum aggregation (original unchunked path)."""
        if self.use_edge_features and edge_features is not None:
            if self.edge_features_kind == "spatial":
                edge_term = self.edge_proj(edge_features)
            else:  # "vector"
                edge_vec = self.edge_proj(edge_features)
                view = (edge_vec.size(0), self.in_channels) + trailing_ones(rank)
                edge_term = edge_vec.view(view)
            messages = F.relu(x.index_select(0, src) + edge_term)
        else:
            messages = x.index_select(0, src)

        if edge_weight is not None:
            weight_b = broadcast_edge_weight(edge_weight, messages, num_edges=E)
            messages = messages * weight_b

        return scatter_sum(messages, dst, N)

    def _chunked_forward(
        self,
        x: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        N: int,
        E: int,
        rank: int,
        edge_features: torch.Tensor | None,
        edge_weight: torch.Tensor | None,
        chunk_size: int,
    ) -> torch.Tensor:
        """Chunked sum aggregation — reduces peak message-buffer memory.

        Sum is associative so chunked output matches unchunked up to
        floating-point rounding order differences (well within float32
        epsilon for typical models).
        """
        agg: torch.Tensor | None = None

        for start in range(0, E, chunk_size):
            end = min(start + chunk_size, E)
            n_c = end - start
            src_c = src[start:end]
            dst_c = dst[start:end]
            ef_c = edge_features[start:end] if edge_features is not None else None

            msg_c = self._compute_chunk_messages(x, src_c, ef_c, n_c, rank)

            if edge_weight is not None:
                ew_c = edge_weight[start:end].to(dtype=msg_c.dtype)
                ew_b = ew_c.view(n_c, *(1,) * (msg_c.dim() - 1))
                msg_c = msg_c * ew_b

            if agg is None:
                agg = msg_c.new_zeros(N, *msg_c.shape[1:])

            agg.index_add_(0, dst_c, msg_c)

        if agg is None:
            # E == 0 caught by caller; unreachable here.
            return scatter_sum(
                x.new_zeros(0, self.in_channels, *x.shape[2:]), dst[:0], N
            )
        return agg

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"eps={float(self.eps):.4f}, "
            f"train_eps={isinstance(self.eps, nn.Parameter)}, "
            f"use_edge_features={self.use_edge_features}, "
            f"spatial_rank={self.spatial_rank}"
        )
