"""Feature-level graph transforms (normalise, standardise, augment)."""
from __future__ import annotations

from typing import Optional

import torch

from ..core.graph import Graph
from .graph import _shallow_copy


class NormalizeFeatures:
    """L1-normalise vector node features (rows sum to 1).

    Zero-feature rows are left unchanged (no division by zero).
    """

    def __init__(self, ord: int = 1) -> None:
        if ord not in (1, 2):
            raise ValueError(f"ord must be 1 or 2; got {ord}")
        self.ord = ord

    def __call__(self, graph: Graph) -> Graph:
        new = _shallow_copy(graph)
        if new.node_features.dim() != 2:
            return new  # only meaningful on vector features
        x = new.node_features
        if self.ord == 1:
            denom = x.abs().sum(dim=1, keepdim=True)
        else:
            denom = x.norm(p=2, dim=1, keepdim=True)
        denom = denom.clamp_min(1e-12)
        new.node_features = x / denom
        # Restore exact zero-rows (clamp introduced 1e-12-scaled noise).
        zero_rows = (x.abs().sum(dim=1) == 0)
        if zero_rows.any():
            new.node_features[zero_rows] = 0.0
        return new


class StandardizeFeatures:
    """Per-feature standardisation: ``(x - mean) / std``.

    Computed across the *node dimension* of the current graph (not a
    global running stat).  Suitable for tutorials; in production use a
    dataset-level stat instead.
    """

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = float(eps)

    def __call__(self, graph: Graph) -> Graph:
        new = _shallow_copy(graph)
        if new.node_features.dim() != 2:
            return new
        x = new.node_features
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp_min(self.eps)
        new.node_features = (x - mean) / std
        return new


class NormalizeEdgeFeatures:
    """L2-normalise per-edge vector features (zero rows preserved)."""

    def __call__(self, graph: Graph) -> Graph:
        if graph.edge_features is None or graph.edge_features.dim() != 2:
            return _shallow_copy(graph)
        new = _shallow_copy(graph)
        ef = new.edge_features
        denom = ef.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        new.edge_features = ef / denom
        zero = (ef.abs().sum(dim=1) == 0)
        if zero.any():
            new.edge_features[zero] = 0.0
        return new


class AddDegreeFeatures:
    """Append per-node degree features.

    Args:
        direction: ``"in"``, ``"out"``, or ``"both"``.
        normalize: When ``True``, divide each degree by ``num_nodes``.
    """

    def __init__(self, direction: str = "both", normalize: bool = False) -> None:
        if direction not in ("in", "out", "both"):
            raise ValueError(f"direction must be 'in'/'out'/'both'; got {direction!r}")
        self.direction = direction
        self.normalize = bool(normalize)

    def __call__(self, graph: Graph) -> Graph:
        new = _shallow_copy(graph)
        if new.node_features.dim() != 2:
            raise ValueError(
                "AddDegreeFeatures requires vector node features [N, D]"
            )
        N = new.num_nodes
        device = new.node_features.device
        dtype = new.node_features.dtype
        in_deg = torch.zeros(N, device=device, dtype=dtype)
        out_deg = torch.zeros(N, device=device, dtype=dtype)
        if new.edge_index is not None and new.num_edges > 0:
            ones = torch.ones(new.num_edges, device=device, dtype=dtype)
            in_deg.index_add_(0, new.edge_index[1], ones)
            out_deg.index_add_(0, new.edge_index[0], ones)
        if self.normalize:
            in_deg = in_deg / max(1, N)
            out_deg = out_deg / max(1, N)
        if self.direction == "in":
            extra = in_deg.unsqueeze(-1)
        elif self.direction == "out":
            extra = out_deg.unsqueeze(-1)
        else:
            extra = torch.stack([in_deg, out_deg], dim=-1)
        new.node_features = torch.cat([new.node_features, extra], dim=-1)
        return new


class AddConstantFeatures:
    """Append a constant scalar feature (e.g. ``1.0``) to every node.

    Useful for graphs where nodes have no features at all.
    """

    def __init__(self, value: float = 1.0, num_features: int = 1) -> None:
        self.value = float(value)
        self.num_features = int(num_features)
        if num_features < 1:
            raise ValueError(f"num_features must be >= 1; got {num_features}")

    def __call__(self, graph: Graph) -> Graph:
        new = _shallow_copy(graph)
        if new.node_features.dim() != 2:
            raise ValueError(
                "AddConstantFeatures requires vector node features [N, D]"
            )
        const = torch.full(
            (new.num_nodes, self.num_features),
            self.value,
            device=new.node_features.device,
            dtype=new.node_features.dtype,
        )
        new.node_features = torch.cat([new.node_features, const], dim=-1)
        return new


class FeatureNoise:
    """Add Gaussian noise of std ``sigma`` to vector node features.

    Deterministic when ``seed`` is provided.
    """

    def __init__(self, sigma: float = 0.05, seed: Optional[int] = None) -> None:
        self.sigma = float(sigma)
        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(int(seed))

    def __call__(self, graph: Graph) -> Graph:
        new = _shallow_copy(graph)
        if self.sigma == 0.0:
            return new
        noise = torch.randn(
            new.node_features.shape, generator=self._gen,
        ).to(new.node_features.dtype)
        new.node_features = new.node_features + self.sigma * noise
        return new


class NodeFeatureMask:
    """Zero out a fraction of node feature entries (Bernoulli mask)."""

    def __init__(self, p: float = 0.1, seed: Optional[int] = None) -> None:
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p must be in [0, 1); got {p}")
        self.p = float(p)
        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(int(seed))

    def __call__(self, graph: Graph) -> Graph:
        new = _shallow_copy(graph)
        if self.p == 0.0:
            return new
        mask = torch.bernoulli(
            torch.full_like(new.node_features, 1.0 - self.p,
                            dtype=torch.float),
            generator=self._gen,
        ).to(new.node_features.dtype)
        new.node_features = new.node_features * mask
        return new
