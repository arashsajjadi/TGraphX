"""Tensor-aware GIN / GINEConv layer.

Adapts Xu et al. (2019)'s Graph Isomorphism Network to spatial node features:

    h_j' = MLP( (1 + ε) · h_j + Σ_{i ∈ N(j)} h_i )

For tensor features, ``MLP`` is a small 1x1 ``Conv2d`` block by default so the
spatial layout ``[C, H, W]`` is preserved.  Users may pass any custom module
as the MLP as long as it maps ``[N, in_channels, H, W]`` to
``[N, out_channels, H, W]``.

When ``use_edge_features=True``, this becomes a tensor-aware GINEConv:

    h_j' = MLP( (1 + ε) · h_j + Σ_i ReLU( h_i + φ(e_ij) ) )

The edge projection ``φ`` adapts to the input format:

* ``edge_features_kind="spatial"`` (default) — ``e_ij`` has shape
  ``[E, edge_dim, H, W]``; ``φ`` is a 1x1 ``Conv2d`` mapping
  ``edge_dim → in_channels`` (or identity when ``edge_dim == in_channels``).
* ``edge_features_kind="vector"``  — ``e_ij`` has shape ``[E, edge_dim]``;
  ``φ`` is ``nn.Linear(edge_dim, in_channels)`` followed by an unsqueeze to
  ``[E, in_channels, 1, 1]`` so the bias broadcasts over the spatial grid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._scatter import scatter_sum


class TensorGINLayer(nn.Module):
    """Tensor-aware GIN / GINEConv layer.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        hidden_channels: Hidden channel count in the default MLP.  Defaults
            to ``out_channels``.
        eps: Initial value of the ε parameter.
        train_eps: If ``True``, ε is a learnable scalar; otherwise it is a
            fixed buffer.
        use_batchnorm: If ``True``, insert ``BatchNorm2d`` layers after each
            convolution in the default MLP.
        mlp: Optional user-supplied module used in place of the default
            two-layer 1x1 Conv MLP.  Must map ``[N, in_channels, H, W]`` to
            ``[N, out_channels, H, W]``.
        use_edge_features: Enable GINEConv-style edge inclusion.
        edge_dim: Edge feature channel/vector count, required when
            ``use_edge_features=True``.
        edge_features_kind: ``"spatial"`` (default) for ``[E, edge_dim, H, W]``
            edge tensors, or ``"vector"`` for ``[E, edge_dim]`` per-edge
            vectors that are broadcast over the ``H × W`` grid.

    Shape conventions:
        * ``x``              ``[N, in_channels, H, W]``
        * ``edge_index``     ``[2, E]`` (``torch.long``)
        * ``edge_features``  ``[E, edge_dim, H, W]`` if ``edge_features_kind=
          "spatial"``, ``[E, edge_dim]`` if ``"vector"`` (only when
          ``use_edge_features=True``).
        * output             ``[N, out_channels, H, W]``
    """

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
    ) -> None:
        super().__init__()
        if use_edge_features and edge_dim is None:
            raise ValueError("edge_dim must be set when use_edge_features=True")
        if edge_features_kind not in ("spatial", "vector"):
            raise ValueError(
                f"edge_features_kind must be 'spatial' or 'vector'; got "
                f"{edge_features_kind!r}."
            )
        if hidden_channels is None:
            hidden_channels = out_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_edge_features = use_edge_features
        self.edge_dim = edge_dim
        self.edge_features_kind = edge_features_kind

        if train_eps:
            self.eps = nn.Parameter(torch.tensor(float(eps)))
        else:
            self.register_buffer("eps", torch.tensor(float(eps)))

        if mlp is None:
            layers: list[nn.Module] = [
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
            ]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            self.mlp = nn.Sequential(*layers)
        else:
            self.mlp = mlp

        if use_edge_features:
            if edge_features_kind == "spatial":
                if edge_dim == in_channels:
                    self.edge_proj: nn.Module = nn.Identity()
                else:
                    self.edge_proj = nn.Conv2d(edge_dim, in_channels, kernel_size=1)
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

        N = x.size(0)
        src = edge_index[0]
        dst = edge_index[1]

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
                edge_term = self.edge_proj(edge_features)  # [E, in_channels, H, W]
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
                # [E, edge_dim] -> [E, in_channels] -> [E, in_channels, 1, 1]
                edge_term = self.edge_proj(edge_features).unsqueeze(-1).unsqueeze(-1)
            messages = F.relu(x.index_select(0, src) + edge_term)
        else:
            messages = x.index_select(0, src)

        agg = scatter_sum(messages, dst, N)  # [N, in_channels, H, W]
        combined = (1.0 + self.eps) * x + agg
        return self.mlp(combined)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"eps={float(self.eps):.4f}, "
            f"train_eps={isinstance(self.eps, nn.Parameter)}, "
            f"use_edge_features={self.use_edge_features}"
        )
