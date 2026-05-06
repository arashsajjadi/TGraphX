"""Tensor-aware GraphSAGE layer.

Adapts Hamilton, Ying & Leskovec (2017) to spatial node feature maps:

    h_j' = W_self(h_j) + W_neigh( AGG_{i ∈ N(j)} h_i )    [+ edge bias]

``W_self`` and ``W_neigh`` are 1x1 ``Conv2d`` so the spatial layout
``[C, H, W]`` is preserved.  ``AGG`` is mean (default) or max.

Edge features (optional) come in two forms, selected via
``edge_features_kind``:

* ``"spatial"`` (default): ``e_ij`` has shape ``[E, edge_dim, H, W]`` and
  is concatenated to the source feature map along the channel dim before
  ``W_neigh`` is applied.
* ``"vector"``: ``e_ij`` has shape ``[E, edge_dim]`` and is projected via
  ``Linear(edge_dim, out_channels)`` into a per-edge channel bias of
  shape ``[E, out_channels, 1, 1]``, which is added to ``W_neigh(h_src)``
  before aggregation by destination.
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
        use_edge_features: Enable edge features in the neighbour transform.
            See ``edge_features_kind`` for the two supported formats.
        edge_dim: Edge feature channel/vector count, required when
            ``use_edge_features=True``.
        edge_features_kind: ``"spatial"`` (default) for ``[E, edge_dim, H, W]``
            edge tensors concatenated to source features before ``W_neigh``,
            or ``"vector"`` for ``[E, edge_dim]`` per-edge vectors that are
            projected to ``[E, out_channels, 1, 1]`` and added to
            ``W_neigh(h_src)`` before aggregation.

    Shape conventions:
        * ``x``              ``[N, in_channels, H, W]``
        * ``edge_index``     ``[2, E]`` (``torch.long``)
        * ``edge_features``  ``[E, edge_dim, H, W]`` if ``edge_features_kind=
          "spatial"``, ``[E, edge_dim]`` if ``"vector"`` (only when
          ``use_edge_features=True``).
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
        edge_features_kind: str = "spatial",
    ) -> None:
        super().__init__()
        if aggr not in ("mean", "max"):
            raise ValueError(
                f"aggr must be 'mean' or 'max'; got {aggr!r}. "
                f"LSTM-style aggregation is not implemented."
            )
        if use_edge_features and edge_dim is None:
            raise ValueError("edge_dim must be set when use_edge_features=True")
        if edge_features_kind not in ("spatial", "vector"):
            raise ValueError(
                f"edge_features_kind must be 'spatial' or 'vector'; got "
                f"{edge_features_kind!r}."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.normalize = normalize
        self.use_edge_features = use_edge_features
        self.edge_dim = edge_dim
        self.edge_features_kind = edge_features_kind
        self.residual = residual

        self.W_self = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        # W_neigh input channel count depends on whether spatial edge tensors
        # are concatenated.  Vector edges are added as a post-projection bias.
        if use_edge_features and edge_features_kind == "spatial":
            neigh_in = in_channels + edge_dim
        else:
            neigh_in = in_channels
        self.W_neigh = nn.Conv2d(neigh_in, out_channels, kernel_size=1, bias=False)

        if use_edge_features and edge_features_kind == "vector":
            self.edge_bias_proj: nn.Module = nn.Linear(edge_dim, out_channels)
        else:
            self.edge_bias_proj = None

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
        if self.edge_bias_proj is not None:
            nn.init.xavier_uniform_(self.edge_bias_proj.weight)
            if self.edge_bias_proj.bias is not None:
                nn.init.zeros_(self.edge_bias_proj.bias)
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
            if edge_features.size(0) != edge_index.size(1):
                raise ValueError(
                    f"edge_features has {edge_features.size(0)} rows but "
                    f"edge_index has {edge_index.size(1)} edges."
                )
            if self.edge_features_kind == "spatial":
                if edge_features.dim() != 4:
                    raise ValueError(
                        f"edge_features must have shape [E, edge_dim, H, W] "
                        f"when edge_features_kind='spatial'; got "
                        f"{tuple(edge_features.shape)}."
                    )
                if edge_features.size(1) != self.edge_dim:
                    raise ValueError(
                        f"edge_features channel count {edge_features.size(1)} "
                        f"does not match edge_dim={self.edge_dim}."
                    )
                cat = torch.cat([x.index_select(0, src), edge_features], dim=1)
                messages = self.W_neigh(cat)
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
                # Project source features (no concatenation), then add a
                # broadcast per-edge channel bias derived from the vector.
                src_msg = self.W_neigh(x.index_select(0, src))  # [E, out, H, W]
                edge_bias = self.edge_bias_proj(edge_features)   # [E, out]
                messages = src_msg + edge_bias.unsqueeze(-1).unsqueeze(-1)
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
