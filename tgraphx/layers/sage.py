"""Tensor-aware GraphSAGE layer.

Adapts Hamilton, Ying & Leskovec (2017) to spatial node feature maps:

    h_j' = W_self(h_j) + W_neigh( AGG_{i ∈ N(j)} h_i )

``W_self`` and ``W_neigh`` are 1x1 ``Conv2d`` so the spatial layout
``[C, H, W]`` is preserved.  ``AGG`` is mean (default) or max.  When
``use_edge_features=True``, edge features are concatenated to the source
features before ``W_neigh`` is applied per edge.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._scatter import scatter_mean, scatter_max


class TensorGraphSAGELayer(nn.Module):
    """Tensor-aware GraphSAGE layer.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        aggr: Neighbour aggregation, ``"mean"`` or ``"max"``.
        normalize: If ``True``, apply L2 normalisation along the channel
            dimension to the output (per node, per spatial location).
        bias: Add a learnable bias on the self transform.
        residual: Add an input residual.  Auto-projects when channel counts
            differ.
        use_edge_features: Concatenate edge features to source features
            before applying ``W_neigh``.
        edge_dim: Edge feature channel count, required when
            ``use_edge_features=True``.

    Shape conventions:
        * ``x``              ``[N, in_channels, H, W]``
        * ``edge_index``     ``[2, E]`` (``torch.long``)
        * ``edge_features``  ``[E, edge_dim, H, W]`` (only when used)
        * output             ``[N, out_channels, H, W]``

    Isolated nodes (no incoming edges) receive only the self transform
    contribution; the neighbour aggregate is zero.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = "mean",
        normalize: bool = False,
        bias: bool = True,
        residual: bool = False,
        use_edge_features: bool = False,
        edge_dim: int | None = None,
    ) -> None:
        super().__init__()
        if aggr not in ("mean", "max"):
            raise ValueError(
                f"aggr must be 'mean' or 'max'; got {aggr!r}. "
                f"LSTM-style aggregation is not implemented."
            )
        if use_edge_features and edge_dim is None:
            raise ValueError("edge_dim must be set when use_edge_features=True")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.normalize = normalize
        self.use_edge_features = use_edge_features
        self.edge_dim = edge_dim
        self.residual = residual

        self.W_self = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        neigh_in = in_channels + (edge_dim if use_edge_features else 0)
        self.W_neigh = nn.Conv2d(neigh_in, out_channels, kernel_size=1, bias=False)

        if residual and in_channels != out_channels:
            self.res_proj = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            )
        else:
            self.res_proj = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_self.weight)
        if self.W_self.bias is not None:
            nn.init.zeros_(self.W_self.bias)
        nn.init.xavier_uniform_(self.W_neigh.weight)
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
    ) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"x must have shape [N, C, H, W]; got {tuple(x.shape)}."
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

        N, _, _, _ = x.shape
        src = edge_index[0]
        dst = edge_index[1]

        # Self transform applied to every node.
        self_out = self.W_self(x)  # [N, out, H, W]

        # Per-edge messages.
        if self.use_edge_features and edge_features is not None:
            if edge_features.dim() != 4:
                raise ValueError(
                    f"edge_features must have shape [E, edge_dim, H, W]; "
                    f"got {tuple(edge_features.shape)}."
                )
            if edge_features.size(1) != self.edge_dim:
                raise ValueError(
                    f"edge_features channel count {edge_features.size(1)} "
                    f"does not match edge_dim={self.edge_dim}."
                )
            cat = torch.cat([x.index_select(0, src), edge_features], dim=1)
            messages = self.W_neigh(cat)
        else:
            # Cheaper: project all nodes once, then gather sources.
            h_neigh = self.W_neigh(x)
            messages = h_neigh.index_select(0, src)

        # Aggregate.
        if self.aggr == "mean":
            agg = scatter_mean(messages, dst, N)
        else:  # "max"
            agg = scatter_max(messages, dst, N)

        out = self_out + agg

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=1)

        if self.residual:
            if self.res_proj is not None:
                out = out + self.res_proj(x)
            elif x.shape == out.shape:
                out = out + x
            # otherwise silently skip

        return out

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"aggr={self.aggr!r}, normalize={self.normalize}"
        )
