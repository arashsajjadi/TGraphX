"""Patch / image / volume transforms.

Most users will go directly through :mod:`tgraphx.graph_builders`;
these wrappers exist so a single :class:`Compose` pipeline can chain
patchifying with feature normalisation, augmentation, etc.
"""
from __future__ import annotations

from typing import Tuple

import torch

from ..core.graph import Graph
from ..graph_builders import (
    build_grid_graph,
    build_grid_graph_3d,
    build_knn_graph,
    build_radius_graph,
    image_to_patches,
    volume_to_patches,
)
from .graph import _shallow_copy


class PatchifyImage:
    """Replace ``node_features`` (shape ``[1, C, H, W]``) with patch tensors.

    Expects a graph that wraps a single image as ``node_features``
    (batch dim 1).  Produces a new graph whose ``node_features`` is
    ``[P, C, ph, pw]``.
    """

    def __init__(
        self,
        patch_size: int,
        padding: str = "auto",
    ) -> None:
        if padding not in ("none", "auto"):
            raise ValueError(f"padding must be 'none' or 'auto'; got {padding!r}")
        self.patch_size = int(patch_size)
        self.padding = padding

    def __call__(self, graph: Graph) -> Graph:
        x = graph.node_features
        if x.dim() != 4 or x.size(0) != 1:
            raise ValueError(
                f"PatchifyImage expects node_features of shape [1, C, H, W]; "
                f"got {tuple(x.shape)}"
            )
        ps = self.patch_size
        C, H, W = x.shape[1:]
        if self.padding == "auto":
            pad_h = (-H) % ps
            pad_w = (-W) % ps
            if pad_h or pad_w:
                x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), value=0.0)
        elif H % ps or W % ps:
            raise ValueError(
                f"PatchifyImage: image {H}x{W} not divisible by {ps}; "
                f"pass padding='auto'"
            )
        patches = image_to_patches(x, patch_size=ps)[0]
        n_h, n_w = x.shape[2] // ps, x.shape[3] // ps

        new = _shallow_copy(graph)
        new.node_features = patches
        meta = dict(new.metadata or {})
        meta["grid_shape"] = (n_h, n_w)
        meta["patch_size"] = ps
        new.metadata = meta
        return new


class PatchifyVolume:
    """Replace ``node_features`` (shape ``[1, C, D, H, W]``) with volumetric patches."""

    def __init__(self, patch_size: int, padding: str = "auto") -> None:
        if padding not in ("none", "auto"):
            raise ValueError(f"padding must be 'none' or 'auto'; got {padding!r}")
        self.patch_size = int(patch_size)
        self.padding = padding

    def __call__(self, graph: Graph) -> Graph:
        x = graph.node_features
        if x.dim() != 5 or x.size(0) != 1:
            raise ValueError(
                f"PatchifyVolume expects node_features of shape "
                f"[1, C, D, H, W]; got {tuple(x.shape)}"
            )
        ps = self.patch_size
        C, D, H, W = x.shape[1:]
        if self.padding == "auto":
            pad_d, pad_h, pad_w = (-D) % ps, (-H) % ps, (-W) % ps
            if pad_d or pad_h or pad_w:
                x = torch.nn.functional.pad(
                    x, (0, pad_w, 0, pad_h, 0, pad_d), value=0.0,
                )
        elif D % ps or H % ps or W % ps:
            raise ValueError(
                f"PatchifyVolume: volume {D}x{H}x{W} not divisible by {ps}; "
                f"pass padding='auto'"
            )
        patches = volume_to_patches(x, patch_size=ps)[0]
        grid = (x.shape[2] // ps, x.shape[3] // ps, x.shape[4] // ps)
        new = _shallow_copy(graph)
        new.node_features = patches
        meta = dict(new.metadata or {})
        meta["grid_shape"] = grid
        meta["patch_size"] = ps
        new.metadata = meta
        return new


class BuildGridGraph:
    """Replace ``edge_index`` with a 2-D grid built from ``metadata['grid_shape']``."""

    def __init__(self, directed: bool = False, self_loops: bool = True) -> None:
        self.directed = bool(directed)
        self.self_loops = bool(self_loops)

    def __call__(self, graph: Graph) -> Graph:
        meta = graph.metadata or {}
        if "grid_shape" not in meta:
            raise ValueError(
                "BuildGridGraph requires graph.metadata['grid_shape'].  Run "
                "PatchifyImage/PatchifyVolume first or set it manually."
            )
        gs = meta["grid_shape"]
        if len(gs) == 2:
            ei = build_grid_graph(gs[0], gs[1], directed=self.directed, self_loops=self.self_loops)
        elif len(gs) == 3:
            ei = build_grid_graph_3d(*gs, directed=self.directed, self_loops=self.self_loops)
        else:
            raise ValueError(f"grid_shape must be 2- or 3-tuple; got {gs!r}")
        new = _shallow_copy(graph)
        new.edge_index = ei
        return new


class BuildKNNGraph:
    """Replace ``edge_index`` with kNN over node-feature centroids.

    Operates on vector node features.  Use
    ``flatten_spatial=True`` to flatten ``[P, C, H, W]`` patches
    into ``[P, C*H*W]`` before computing the kNN.
    """

    def __init__(self, k: int, flatten_spatial: bool = True,
                 directed: bool = False, self_loops: bool = True) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = int(k)
        self.flatten_spatial = bool(flatten_spatial)
        self.directed = bool(directed)
        self.self_loops = bool(self_loops)

    def __call__(self, graph: Graph) -> Graph:
        x = graph.node_features
        if x.dim() == 2:
            coords = x
        elif self.flatten_spatial:
            coords = x.reshape(x.size(0), -1)
        else:
            raise ValueError(
                f"BuildKNNGraph: expected vector features or flatten_spatial=True; "
                f"got shape {tuple(x.shape)}"
            )
        new = _shallow_copy(graph)
        new.edge_index = build_knn_graph(
            coords.detach(), k=self.k, directed=self.directed, self_loops=self.self_loops,
        )
        return new


class BuildRadiusGraph:
    def __init__(self, radius: float, flatten_spatial: bool = True,
                 directed: bool = False, self_loops: bool = True) -> None:
        if radius <= 0:
            raise ValueError("radius must be > 0")
        self.radius = float(radius)
        self.flatten_spatial = bool(flatten_spatial)
        self.directed = bool(directed)
        self.self_loops = bool(self_loops)

    def __call__(self, graph: Graph) -> Graph:
        x = graph.node_features
        if x.dim() == 2:
            coords = x
        elif self.flatten_spatial:
            coords = x.reshape(x.size(0), -1)
        else:
            raise ValueError(
                "BuildRadiusGraph: expected vector features or flatten_spatial=True"
            )
        new = _shallow_copy(graph)
        new.edge_index = build_radius_graph(
            coords.detach(), radius=self.radius,
            directed=self.directed, self_loops=self.self_loops,
        )
        return new
