"""Native, learnable synthetic graph datasets.

Every dataset here:

* generates data deterministically from a ``seed``,
* exposes a clear ``task`` (classification / regression / hetero / temporal),
* uses *learnable* labels (not pure noise) so tiny-overfit smoke tests
  can show a real loss decrease,
* never touches the network,
* is small enough by default to run in CI in well under a second.

These are sanity / tutorial datasets — not benchmarks.  Documentation
makes that explicit.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch

from ..core.graph import Graph
from ..core.hetero_graph import HeteroGraph
from ..core.temporal import TemporalGraphSequence
from ..graph_builders import (
    build_grid_graph,
    build_grid_graph_3d,
    build_knn_graph,
    build_radius_graph,
)
from .base import InMemoryGraphDataset, TransformFn, TargetTransformFn
from .metadata import DatasetMetadata

__all__ = [
    "SyntheticPatchGraphDataset",
    "SyntheticVolumeGraphDataset",
    "SyntheticNodeClassificationDataset",
    "SyntheticEdgePredictionDataset",
    "SyntheticGraphRegressionDataset",
    "SyntheticHeteroGraphDataset",
    "SyntheticTemporalGraphDataset",
]


# ── Pattern generators ───────────────────────────────────────────────────────


def _pattern_image(
    pattern: str,
    H: int,
    W: int,
    C: int,
    rng: torch.Generator,
    intensity: float,
) -> torch.Tensor:
    """Produce a deterministic ``[C, H, W]`` image carrying ``pattern``."""
    img = torch.zeros(C, H, W)
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float),
        torch.arange(W, dtype=torch.float),
        indexing="ij",
    )
    if pattern == "horizontal_stripe":
        mask = (yy >= H // 3) & (yy < 2 * H // 3)
    elif pattern == "vertical_stripe":
        mask = (xx >= W // 3) & (xx < 2 * W // 3)
    elif pattern == "diagonal":
        mask = (yy - xx).abs() < min(H, W) // 4
    elif pattern == "checkerboard":
        block = max(1, H // 4)
        mask = ((yy // block).int() + (xx // block).int()) % 2 == 0
    elif pattern == "central_blob":
        cy, cx = H / 2, W / 2
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 < (min(H, W) / 4) ** 2
    elif pattern == "corner_blob":
        mask = (yy < H // 3) & (xx < W // 3)
    elif pattern == "noisy_background":
        mask = torch.rand((H, W), generator=rng) > 0.5
    else:
        raise ValueError(f"Unknown pattern {pattern!r}")
    img += mask.float().unsqueeze(0) * float(intensity)
    img += 0.05 * torch.randn(C, H, W, generator=rng)
    return img


def _pattern_volume(
    pattern: str,
    D: int,
    H: int,
    W: int,
    C: int,
    rng: torch.Generator,
    intensity: float,
) -> torch.Tensor:
    """Produce a deterministic ``[C, D, H, W]`` volume."""
    vol = torch.zeros(C, D, H, W)
    zz, yy, xx = torch.meshgrid(
        torch.arange(D, dtype=torch.float),
        torch.arange(H, dtype=torch.float),
        torch.arange(W, dtype=torch.float),
        indexing="ij",
    )
    if pattern == "sphere":
        cz, cy, cx = D / 2, H / 2, W / 2
        mask = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 < (min(D, H, W) / 3) ** 2
    elif pattern == "cube":
        mask = (
            (zz >= D // 4) & (zz < 3 * D // 4) &
            (yy >= H // 4) & (yy < 3 * H // 4) &
            (xx >= W // 4) & (xx < 3 * W // 4)
        )
    elif pattern == "tube_x":
        mask = ((yy - H / 2) ** 2 + (zz - D / 2) ** 2) < (min(H, D) / 4) ** 2
    elif pattern == "tube_y":
        mask = ((xx - W / 2) ** 2 + (zz - D / 2) ** 2) < (min(W, D) / 4) ** 2
    elif pattern == "tube_z":
        mask = ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) < (min(W, H) / 4) ** 2
    elif pattern == "diagonal_structure":
        mask = (xx - yy).abs() + (yy - zz).abs() < min(D, H, W) // 3
    else:
        raise ValueError(f"Unknown volume pattern {pattern!r}")
    vol += mask.float().unsqueeze(0) * float(intensity)
    vol += 0.05 * torch.randn(C, D, H, W, generator=rng)
    return vol


# ── Helpers ──────────────────────────────────────────────────────────────────


def _patchify_2d(image: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Split ``[C, H, W]`` into ``[P, C, ph, pw]`` row-major.

    Requires divisible dims (datasets pre-size their images).
    """
    C, H, W = image.shape
    assert H % patch_size == 0 and W % patch_size == 0
    n_h, n_w = H // patch_size, W // patch_size
    img = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # img: [C, n_h, n_w, ps, ps]
    patches = img.permute(1, 2, 0, 3, 4).contiguous().view(n_h * n_w, C, patch_size, patch_size)
    return patches, (n_h, n_w)


def _patchify_3d(volume: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    C, D, H, W = volume.shape
    assert D % patch_size == 0 and H % patch_size == 0 and W % patch_size == 0
    n_d, n_h, n_w = D // patch_size, H // patch_size, W // patch_size
    vol = (
        volume.unfold(1, patch_size, patch_size)
              .unfold(2, patch_size, patch_size)
              .unfold(3, patch_size, patch_size)
    )
    # vol: [C, n_d, n_h, n_w, ps, ps, ps]
    patches = vol.permute(1, 2, 3, 0, 4, 5, 6).contiguous().view(
        n_d * n_h * n_w, C, patch_size, patch_size, patch_size,
    )
    return patches, (n_d, n_h, n_w)


def _patch_centres(grid_shape: Tuple[int, ...]) -> torch.Tensor:
    """Return integer-coordinate centres for kNN/radius graph builders."""
    ranges = [torch.arange(n, dtype=torch.float) for n in grid_shape]
    grids = torch.meshgrid(*ranges, indexing="ij")
    coords = torch.stack([g.flatten() for g in grids], dim=-1)
    return coords


def _build_edge_index(
    grid_shape: Tuple[int, ...],
    builder: str,
    knn_k: int = 4,
    radius: float = 1.5,
) -> torch.Tensor:
    if builder == "grid":
        if len(grid_shape) == 2:
            return build_grid_graph(grid_shape[0], grid_shape[1], directed=False, self_loops=True)
        return build_grid_graph_3d(*grid_shape, directed=False, self_loops=True)
    coords = _patch_centres(grid_shape)
    if builder == "knn":
        return build_knn_graph(coords, k=knn_k, directed=False, self_loops=True)
    if builder == "radius":
        return build_radius_graph(coords, radius=radius, directed=False, self_loops=True)
    raise ValueError(f"Unknown graph_builder {builder!r}")


# ── 1. Patch graph dataset (2-D) ─────────────────────────────────────────────


PATCH_PATTERNS = (
    "horizontal_stripe", "vertical_stripe", "diagonal",
    "checkerboard", "central_blob", "corner_blob",
)


class SyntheticPatchGraphDataset(InMemoryGraphDataset):
    """Synthetic 2-D image-patch graph dataset.

    Each sample is a :class:`Graph` whose nodes are the patches of a
    small synthetic image carrying one of :data:`PATCH_PATTERNS`.  The
    ``graph_label`` is the pattern's class id (so a tiny GNN can
    overfit the dataset in a few epochs).

    Args:
        num_graphs: Number of samples.
        image_size: Side length ``H = W`` of each image (must be
            divisible by ``patch_size``).
        channels: Number of image channels.
        patch_size: Side of each square patch.
        graph_builder: ``"grid"`` (default), ``"knn"``, ``"radius"``.
        knn_k: ``k`` when ``graph_builder="knn"``.
        radius: Cutoff when ``graph_builder="radius"``.
        intensity_range: Tuple ``(low, high)`` controlling the
            per-graph pattern intensity (also exposed as
            ``graph_label`` when ``task='graph_regression'``).
        task: ``"graph_classification"`` (default) or
            ``"graph_regression"``.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        num_graphs: int = 32,
        image_size: int = 32,
        channels: int = 1,
        patch_size: int = 8,
        graph_builder: str = "grid",
        knn_k: int = 4,
        radius: float = 1.5,
        intensity_range: Tuple[float, float] = (0.5, 1.0),
        task: str = "graph_classification",
        seed: int = 0,
        root: Optional[str | Path] = None,
        split: Optional[str] = None,
        transform: TransformFn = None,
        target_transform: TargetTransformFn = None,
    ) -> None:
        if task not in ("graph_classification", "graph_regression"):
            raise ValueError(
                f"task must be 'graph_classification' or "
                f"'graph_regression'; got {task!r}"
            )
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by "
                f"patch_size ({patch_size})."
            )
        self.num_graphs = int(num_graphs)
        self.image_size = int(image_size)
        self.channels = int(channels)
        self.patch_size = int(patch_size)
        self.graph_builder = graph_builder
        self.knn_k = knn_k
        self.radius = radius
        self.intensity_range = intensity_range
        self.task = task
        self.seed = int(seed)
        super().__init__(
            root=root, split=split, transform=transform,
            target_transform=target_transform,
        )

    def _populate(self) -> None:
        rng = torch.Generator().manual_seed(self.seed)
        H = W = self.image_size
        ps = self.patch_size
        n_h, n_w = H // ps, W // ps
        edge_index = _build_edge_index(
            (n_h, n_w), self.graph_builder,
            knn_k=self.knn_k, radius=self.radius,
        )

        graphs: List[Graph] = []
        lo, hi = self.intensity_range
        for i in range(self.num_graphs):
            # Cycle through patterns so every class appears.
            pattern_idx = i % len(PATCH_PATTERNS)
            pattern = PATCH_PATTERNS[pattern_idx]
            intensity = lo + (hi - lo) * torch.rand((), generator=rng).item()
            image = _pattern_image(
                pattern, H, W, self.channels, rng=rng, intensity=intensity,
            )
            patches, _ = _patchify_2d(image, ps)
            if self.task == "graph_classification":
                label = torch.tensor(pattern_idx, dtype=torch.long)
            else:  # regression on intensity
                label = torch.tensor(intensity, dtype=torch.float).view(1)
            g = Graph(
                node_features=patches,
                edge_index=edge_index.clone(),
                graph_label=label,
                metadata={
                    "pattern": pattern,
                    "intensity": float(intensity),
                    "grid_shape": (n_h, n_w),
                    "task": self.task,
                },
            )
            graphs.append(g)
        self.graphs = graphs

    def _build_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="synthetic:patch_graph",
            source="TGraphX synthetic",
            upstream_library=None,
            license="MIT (TGraphX-generated synthetic data)",
            task=self.task,
            graph_type="homogeneous",
            num_graphs=self.num_graphs,
            num_classes=(len(PATCH_PATTERNS) if self.task == "graph_classification" else None),
            extra={
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "channels": self.channels,
                "graph_builder": self.graph_builder,
                "seed": self.seed,
            },
        )


# ── 2. Volume graph dataset (3-D) ────────────────────────────────────────────


VOLUME_PATTERNS = ("sphere", "cube", "tube_x", "tube_y", "tube_z", "diagonal_structure")


class SyntheticVolumeGraphDataset(InMemoryGraphDataset):
    """Synthetic 3-D volume-patch graph dataset (graph classification).

    Mirrors :class:`SyntheticPatchGraphDataset` but with volumetric
    patches ``[C, pd, ph, pw]`` and a 3-D grid graph.
    """

    def __init__(
        self,
        num_graphs: int = 16,
        volume_size: int = 16,
        channels: int = 1,
        patch_size: int = 4,
        intensity_range: Tuple[float, float] = (0.6, 1.0),
        seed: int = 0,
        root: Optional[str | Path] = None,
        split: Optional[str] = None,
        transform: TransformFn = None,
        target_transform: TargetTransformFn = None,
    ) -> None:
        if volume_size % patch_size != 0:
            raise ValueError(
                f"volume_size ({volume_size}) must be divisible by "
                f"patch_size ({patch_size})."
            )
        self.num_graphs = int(num_graphs)
        self.volume_size = int(volume_size)
        self.channels = int(channels)
        self.patch_size = int(patch_size)
        self.intensity_range = intensity_range
        self.seed = int(seed)
        super().__init__(
            root=root, split=split, transform=transform,
            target_transform=target_transform,
        )

    def _populate(self) -> None:
        rng = torch.Generator().manual_seed(self.seed)
        D = H = W = self.volume_size
        ps = self.patch_size
        grid = (D // ps, H // ps, W // ps)
        edge_index = build_grid_graph_3d(*grid, directed=False, self_loops=True)
        lo, hi = self.intensity_range
        graphs: List[Graph] = []
        for i in range(self.num_graphs):
            pattern_idx = i % len(VOLUME_PATTERNS)
            pattern = VOLUME_PATTERNS[pattern_idx]
            intensity = lo + (hi - lo) * torch.rand((), generator=rng).item()
            volume = _pattern_volume(
                pattern, D, H, W, self.channels, rng=rng, intensity=intensity,
            )
            patches, _ = _patchify_3d(volume, ps)
            graphs.append(
                Graph(
                    node_features=patches,
                    edge_index=edge_index.clone(),
                    graph_label=torch.tensor(pattern_idx, dtype=torch.long),
                    metadata={
                        "pattern": pattern,
                        "intensity": float(intensity),
                        "grid_shape": grid,
                    },
                )
            )
        self.graphs = graphs

    def _build_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="synthetic:volume_graph",
            source="TGraphX synthetic",
            license="MIT (TGraphX-generated synthetic data)",
            task="graph_classification",
            graph_type="homogeneous",
            num_graphs=self.num_graphs,
            num_classes=len(VOLUME_PATTERNS),
            extra={
                "volume_size": self.volume_size,
                "patch_size": self.patch_size,
                "channels": self.channels,
                "seed": self.seed,
            },
        )


# ── 3. Node classification (single graph, vector features) ──────────────────


class SyntheticNodeClassificationDataset(InMemoryGraphDataset):
    """Single-graph node classification benchmark.

    Stochastic block model: ``num_classes`` clusters, edges denser
    inside a cluster than between.  Node features are class-conditional
    Gaussian centres + small noise — easily separable with a couple of
    GNN layers.

    Train/val/test masks are stored both as :class:`Graph.metadata`
    (under ``"masks"``) and as ``train_mask`` / ``val_mask`` /
    ``test_mask`` attributes on the metadata for convenience.
    """

    def __init__(
        self,
        num_nodes: int = 200,
        num_classes: int = 4,
        feature_dim: int = 16,
        intra_class_prob: float = 0.05,
        inter_class_prob: float = 0.005,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        seed: int = 0,
        root: Optional[str | Path] = None,
        split: Optional[str] = None,
        transform: TransformFn = None,
        target_transform: TargetTransformFn = None,
    ) -> None:
        if not 0.0 <= train_ratio + val_ratio <= 1.0:
            raise ValueError("train_ratio + val_ratio must be in [0, 1]")
        self.num_nodes = int(num_nodes)
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.intra_class_prob = float(intra_class_prob)
        self.inter_class_prob = float(inter_class_prob)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        super().__init__(
            root=root, split=split, transform=transform,
            target_transform=target_transform,
        )

    def _populate(self) -> None:
        rng = torch.Generator().manual_seed(self.seed)
        N = self.num_nodes
        K = self.num_classes
        D = self.feature_dim

        # Class centres: well-separated Gaussians on the unit sphere-ish.
        centres = torch.randn(K, D, generator=rng) * 2.5

        labels = torch.randint(0, K, (N,), generator=rng)
        x = centres[labels] + 0.5 * torch.randn(N, D, generator=rng)

        # Build SBM edges.
        edges_src: List[int] = []
        edges_dst: List[int] = []
        rand = torch.rand(N, N, generator=rng)
        same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
        prob = torch.where(
            same_class, torch.full_like(rand, self.intra_class_prob),
            torch.full_like(rand, self.inter_class_prob),
        )
        # Upper-triangular only, then symmetrise.
        triu = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        connect = (rand < prob) & triu
        ij = connect.nonzero(as_tuple=False)
        for i, j in ij.tolist():
            edges_src.extend([i, j])
            edges_dst.extend([j, i])
        if not edges_src:
            # Add a self-loop on every node so downstream layers don't hit
            # zero-edge edge cases.
            edges_src = list(range(N))
            edges_dst = list(range(N))
        edge_index = torch.stack(
            [torch.tensor(edges_src, dtype=torch.long),
             torch.tensor(edges_dst, dtype=torch.long)],
            dim=0,
        )

        # Splits.
        perm = torch.randperm(N, generator=rng)
        n_train = int(self.train_ratio * N)
        n_val = int(self.val_ratio * N)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]
        train_mask = torch.zeros(N, dtype=torch.bool)
        val_mask = torch.zeros(N, dtype=torch.bool)
        test_mask = torch.zeros(N, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        g = Graph(
            node_features=x,
            edge_index=edge_index,
            node_labels=labels,
            metadata={
                "masks": {
                    "train_mask": train_mask,
                    "val_mask": val_mask,
                    "test_mask": test_mask,
                },
                "num_classes": K,
                "task": "node_classification",
            },
        )
        self.graphs = [g]

    def _build_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="synthetic:node_classification",
            source="TGraphX synthetic",
            license="MIT (TGraphX-generated synthetic data)",
            task="node_classification",
            graph_type="homogeneous",
            num_graphs=1,
            num_nodes=self.num_nodes,
            num_classes=self.num_classes,
            splits={
                "train": int(self.train_ratio * self.num_nodes),
                "val": int(self.val_ratio * self.num_nodes),
                "test": self.num_nodes
                - int(self.train_ratio * self.num_nodes)
                - int(self.val_ratio * self.num_nodes),
            },
            extra={"feature_dim": self.feature_dim, "seed": self.seed},
        )


# ── 4. Edge prediction ────────────────────────────────────────────────────────


class SyntheticEdgePredictionDataset(InMemoryGraphDataset):
    """Single-graph edge prediction sanity dataset.

    Nodes have random vector features.  Positive edges are pairs of
    *similar* nodes (cosine similarity > threshold); negative edges are
    pairs of *dissimilar* nodes.  Both populate ``edge_labels`` (0 or
    1) so users can train binary edge classifiers.
    """

    def __init__(
        self,
        num_nodes: int = 80,
        feature_dim: int = 16,
        num_pos: int = 100,
        num_neg: int = 100,
        sim_threshold: float = 0.6,
        seed: int = 0,
        root: Optional[str | Path] = None,
        split: Optional[str] = None,
        transform: TransformFn = None,
        target_transform: TargetTransformFn = None,
    ) -> None:
        self.num_nodes = int(num_nodes)
        self.feature_dim = int(feature_dim)
        self.num_pos = int(num_pos)
        self.num_neg = int(num_neg)
        self.sim_threshold = float(sim_threshold)
        self.seed = int(seed)
        super().__init__(
            root=root, split=split, transform=transform,
            target_transform=target_transform,
        )

    def _populate(self) -> None:
        rng = torch.Generator().manual_seed(self.seed)
        N, D = self.num_nodes, self.feature_dim
        x = torch.randn(N, D, generator=rng)
        x_norm = torch.nn.functional.normalize(x, dim=-1)
        sim = x_norm @ x_norm.t()
        # Make sure self-similarity isn't picked.
        sim.fill_diagonal_(-2.0)

        pos_pairs = (sim > self.sim_threshold).nonzero(as_tuple=False)
        neg_pairs = (sim < -self.sim_threshold).nonzero(as_tuple=False)

        if pos_pairs.size(0) < self.num_pos or neg_pairs.size(0) < self.num_neg:
            # Fall back to top-k pairs by absolute similarity.
            _, top = sim.flatten().topk(min(self.num_pos, N * (N - 1) // 2))
            pos_pairs = torch.stack([top // N, top % N], dim=1)
            _, low = (-sim.flatten()).topk(min(self.num_neg, N * (N - 1) // 2))
            neg_pairs = torch.stack([low // N, low % N], dim=1)

        pos = pos_pairs[torch.randperm(pos_pairs.size(0), generator=rng)[:self.num_pos]]
        neg = neg_pairs[torch.randperm(neg_pairs.size(0), generator=rng)[:self.num_neg]]

        ei = torch.cat([pos.t(), neg.t()], dim=1)
        labels = torch.cat([
            torch.ones(pos.size(0), dtype=torch.long),
            torch.zeros(neg.size(0), dtype=torch.long),
        ])

        g = Graph(
            node_features=x,
            edge_index=ei,
            edge_labels=labels,
            metadata={
                "num_pos": int(pos.size(0)),
                "num_neg": int(neg.size(0)),
                "task": "edge_prediction",
            },
        )
        self.graphs = [g]

    def _build_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="synthetic:edge_prediction",
            source="TGraphX synthetic",
            license="MIT (TGraphX-generated synthetic data)",
            task="edge_prediction",
            graph_type="homogeneous",
            num_graphs=1,
            num_nodes=self.num_nodes,
            num_classes=2,
            extra={
                "feature_dim": self.feature_dim,
                "num_pos": self.num_pos,
                "num_neg": self.num_neg,
                "seed": self.seed,
            },
        )


# ── 5. Graph regression ──────────────────────────────────────────────────────


class SyntheticGraphRegressionDataset(InMemoryGraphDataset):
    """Predict scalar pattern intensity from a patch graph."""

    def __init__(
        self,
        num_graphs: int = 32,
        image_size: int = 24,
        channels: int = 1,
        patch_size: int = 6,
        seed: int = 0,
        root: Optional[str | Path] = None,
        split: Optional[str] = None,
        transform: TransformFn = None,
        target_transform: TargetTransformFn = None,
    ) -> None:
        # Reuse the patch-graph generator with task='graph_regression'.
        self._inner = SyntheticPatchGraphDataset(
            num_graphs=num_graphs,
            image_size=image_size,
            channels=channels,
            patch_size=patch_size,
            graph_builder="grid",
            task="graph_regression",
            seed=seed,
        )
        super().__init__(
            root=root, split=split, transform=transform,
            target_transform=target_transform,
        )
        self.graphs = list(self._inner.graphs)

    def _build_metadata(self) -> DatasetMetadata:
        meta = self._inner.metadata
        meta.name = "synthetic:graph_regression"
        meta.task = "graph_regression"
        return meta


# ── 6. Hetero ────────────────────────────────────────────────────────────────


class SyntheticHeteroGraphDataset(InMemoryGraphDataset):
    """Mini bipartite-ish hetero graph: paper / author / venue.

    A single :class:`HeteroGraph` with three node types and three
    relation types, vector features, and per-paper class labels.
    """

    def __init__(
        self,
        num_papers: int = 30,
        num_authors: int = 20,
        num_venues: int = 5,
        feature_dim_paper: int = 8,
        feature_dim_author: int = 6,
        feature_dim_venue: int = 4,
        num_classes: int = 3,
        seed: int = 0,
        root: Optional[str | Path] = None,
        split: Optional[str] = None,
        transform: TransformFn = None,
        target_transform: TargetTransformFn = None,
    ) -> None:
        self.num_papers = int(num_papers)
        self.num_authors = int(num_authors)
        self.num_venues = int(num_venues)
        self.feature_dim_paper = int(feature_dim_paper)
        self.feature_dim_author = int(feature_dim_author)
        self.feature_dim_venue = int(feature_dim_venue)
        self.num_classes = int(num_classes)
        self.seed = int(seed)
        super().__init__(
            root=root, split=split, transform=transform,
            target_transform=target_transform,
        )

    def _populate(self) -> None:
        rng = torch.Generator().manual_seed(self.seed)
        K = self.num_classes
        # Class-conditional papers.
        paper_class = torch.randint(0, K, (self.num_papers,), generator=rng)
        paper_centres = torch.randn(K, self.feature_dim_paper, generator=rng) * 2.0
        x_paper = (
            paper_centres[paper_class]
            + 0.4 * torch.randn(self.num_papers, self.feature_dim_paper, generator=rng)
        )
        x_author = torch.randn(self.num_authors, self.feature_dim_author, generator=rng)
        x_venue = torch.randn(self.num_venues, self.feature_dim_venue, generator=rng)

        # Edges: each paper has 1-3 authors; each paper is published in 1 venue;
        # 30 random paper-cites-paper edges.
        author_writes_paper = []
        for p in range(self.num_papers):
            n = int(torch.randint(1, 4, (), generator=rng).item())
            authors = torch.randperm(self.num_authors, generator=rng)[:n]
            for a in authors.tolist():
                author_writes_paper.append((a, p))
        ei_aw = torch.tensor(author_writes_paper, dtype=torch.long).t().contiguous()

        venues = torch.randint(0, self.num_venues, (self.num_papers,), generator=rng)
        ei_pv = torch.stack([torch.arange(self.num_papers), venues], dim=0).long()

        n_cite = max(self.num_papers, 30)
        src = torch.randint(0, self.num_papers, (n_cite,), generator=rng)
        dst = torch.randint(0, self.num_papers, (n_cite,), generator=rng)
        keep = src != dst
        ei_pc = torch.stack([src[keep], dst[keep]], dim=0).long()

        hg = HeteroGraph(
            node_stores={
                "paper": x_paper,
                "author": x_author,
                "venue": x_venue,
            },
            edge_stores={
                ("author", "writes", "paper"): ei_aw,
                ("paper", "published_in", "venue"): ei_pv,
                ("paper", "cites", "paper"): ei_pc,
            },
            node_label_stores={"paper": paper_class},
            metadata={"task": "hetero_node_classification"},
        )
        self.graphs = [hg]

    def _build_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="synthetic:hetero",
            source="TGraphX synthetic",
            license="MIT (TGraphX-generated synthetic data)",
            task="hetero_node_classification",
            graph_type="heterogeneous",
            num_graphs=1,
            num_classes=self.num_classes,
            extra={
                "num_papers": self.num_papers,
                "num_authors": self.num_authors,
                "num_venues": self.num_venues,
                "seed": self.seed,
            },
        )


# ── 7. Temporal ──────────────────────────────────────────────────────────────


class SyntheticTemporalGraphDataset(InMemoryGraphDataset):
    """Sequence-of-snapshots dataset.

    Each item is a :class:`TemporalGraphSequence` of ``T`` snapshots.
    The trend at each time step (increasing / decreasing / event)
    determines the sequence-level label so a temporal classifier can
    overfit it.
    """

    TRENDS = ("increasing", "decreasing", "event")

    def __init__(
        self,
        num_sequences: int = 16,
        sequence_length: int = 6,
        num_nodes: int = 12,
        feature_dim: int = 8,
        seed: int = 0,
        root: Optional[str | Path] = None,
        split: Optional[str] = None,
        transform: TransformFn = None,
        target_transform: TargetTransformFn = None,
    ) -> None:
        self.num_sequences = int(num_sequences)
        self.sequence_length = int(sequence_length)
        self.num_nodes = int(num_nodes)
        self.feature_dim = int(feature_dim)
        self.seed = int(seed)
        super().__init__(
            root=root, split=split, transform=transform,
            target_transform=target_transform,
        )

    def _populate(self) -> None:
        rng = torch.Generator().manual_seed(self.seed)
        sequences: List[TemporalGraphSequence] = []
        labels: List[int] = []
        N = self.num_nodes
        T = self.sequence_length
        D = self.feature_dim
        for s in range(self.num_sequences):
            trend_idx = s % len(self.TRENDS)
            trend = self.TRENDS[trend_idx]
            base = torch.randn(N, D, generator=rng)
            snapshots: List[Graph] = []
            timestamps: List[float] = []
            for t in range(T):
                if trend == "increasing":
                    intensity = (t + 1) / T
                elif trend == "decreasing":
                    intensity = 1.0 - t / T
                else:  # event: small spike at the middle
                    intensity = 1.0 if t == T // 2 else 0.2
                feats = base + intensity * torch.ones(N, D)
                src = torch.arange(N, dtype=torch.long)
                dst = (src + 1) % N
                ei = torch.stack([src, dst], dim=0)
                snapshots.append(
                    Graph(
                        node_features=feats,
                        edge_index=ei,
                        graph_label=torch.tensor(trend_idx, dtype=torch.long),
                    )
                )
                timestamps.append(float(t))
            sequences.append(
                TemporalGraphSequence(
                    graphs=snapshots,
                    timestamps=timestamps,
                    metadata={"trend": trend, "label": trend_idx},
                )
            )
            labels.append(trend_idx)
        self.graphs = sequences
        self._labels = labels

    def _build_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="synthetic:temporal",
            source="TGraphX synthetic",
            license="MIT (TGraphX-generated synthetic data)",
            task="temporal_graph_classification",
            graph_type="temporal",
            num_graphs=self.num_sequences,
            num_classes=len(self.TRENDS),
            extra={
                "sequence_length": self.sequence_length,
                "num_nodes": self.num_nodes,
                "seed": self.seed,
            },
        )
