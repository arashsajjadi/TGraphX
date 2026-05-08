"""Datasets that read user-owned image / volume folders.

These datasets never download anything.  They walk a local directory
laid out as ``root/class_name/file.{png,jpg,...}`` (or
``root/file.{npy,pt}`` for unlabelled volumes) and convert each file
to a TGraphX :class:`~tgraphx.Graph` of patches.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
from .base import InMemoryGraphDataset, TransformFn, TargetTransformFn
from .errors import OptionalDependencyError
from .metadata import DatasetMetadata

__all__ = [
    "ImageFolderPatchGraphDataset",
    "VolumeFolderPatchGraphDataset",
]


# Image extensions handled when PIL is available.
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
# Volume extensions handled natively (no extra deps needed).
_VOLUME_EXTS = (".npy", ".npz", ".pt")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _list_class_subdirs(root: Path) -> Dict[str, Path]:
    classes: Dict[str, Path] = {}
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            classes[child.name] = child
    return classes


def _build_edge_index_from_grid(
    grid_shape: Tuple[int, ...],
    builder: str,
    knn_k: int,
    radius: float,
) -> torch.Tensor:
    if builder == "grid":
        if len(grid_shape) == 2:
            return build_grid_graph(grid_shape[0], grid_shape[1], directed=False, self_loops=True)
        return build_grid_graph_3d(*grid_shape, directed=False, self_loops=True)
    coords_ranges = [torch.arange(n, dtype=torch.float) for n in grid_shape]
    grids = torch.meshgrid(*coords_ranges, indexing="ij")
    coords = torch.stack([g.flatten() for g in grids], dim=-1)
    if builder == "knn":
        return build_knn_graph(coords, k=knn_k, directed=False, self_loops=True)
    if builder == "radius":
        return build_radius_graph(coords, radius=radius, directed=False, self_loops=True)
    raise ValueError(f"Unknown graph_builder {builder!r}")


# ── Image folder ─────────────────────────────────────────────────────────────


class ImageFolderPatchGraphDataset(InMemoryGraphDataset):
    """Walk ``root/class_name/*.{png,jpg,...}`` and build patch graphs.

    Lazy-imports PIL.  If PIL is not installed, raises
    :class:`OptionalDependencyError` with an install hint.

    Args:
        root: Path containing class subdirectories.
        patch_size: Square patch side.
        graph_builder: ``"grid"`` (default), ``"knn"``, or ``"radius"``.
        knn_k / radius: Used when their respective builder is selected.
        padding: ``"none"`` (default) or ``"auto"``.  ``"auto"`` pads the
            image with zeros so it is exactly divisible by
            ``patch_size``.
        normalize: When ``True``, divide by 255 so pixel values fall in
            ``[0, 1]``.
        image_transform: Optional callable applied to the raw
            ``[C, H, W]`` tensor before patchifying (e.g. resize).
        graph_transform: Optional :class:`tgraphx.transforms` callable
            applied to each :class:`Graph` (alias for ``transform``).
        max_images_per_class: Cap to keep the dataset small for CI/demo.
    """

    def __init__(
        self,
        root: str | Path,
        patch_size: int = 8,
        graph_builder: str = "grid",
        knn_k: int = 4,
        radius: float = 1.5,
        padding: str = "auto",
        normalize: bool = True,
        image_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        graph_transform: TransformFn = None,
        max_images_per_class: Optional[int] = None,
        target_transform: TargetTransformFn = None,
    ) -> None:
        self._root_path = Path(root).expanduser()
        if not self._root_path.exists():
            raise FileNotFoundError(
                f"Image folder root does not exist: {self._root_path}"
            )
        self.patch_size = int(patch_size)
        self.graph_builder = graph_builder
        self.knn_k = int(knn_k)
        self.radius = float(radius)
        if padding not in ("none", "auto"):
            raise ValueError(f"padding must be 'none' or 'auto'; got {padding!r}")
        self.padding = padding
        self.normalize = bool(normalize)
        self.image_transform = image_transform
        self.max_images_per_class = max_images_per_class

        # Build class index by sorted subdir names — stable across OSes.
        self._class_dirs = _list_class_subdirs(self._root_path)
        self.class_to_idx: Dict[str, int] = {
            cls: i for i, cls in enumerate(self._class_dirs)
        }

        super().__init__(
            root=root, split=None,
            transform=graph_transform,
            target_transform=target_transform,
        )

    @staticmethod
    def _require_pil():
        try:
            from PIL import Image  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise OptionalDependencyError(
                "Pillow",
                install_hint="pip install Pillow",
            ) from exc

    def _populate(self) -> None:
        self._require_pil()
        import numpy as _np
        from PIL import Image  # local import (lazy)

        graphs: List[Graph] = []
        for cls, dirpath in self._class_dirs.items():
            files = sorted(p for p in dirpath.iterdir()
                           if p.suffix.lower() in _IMAGE_EXTS)
            if self.max_images_per_class is not None:
                files = files[: int(self.max_images_per_class)]
            cls_idx = self.class_to_idx[cls]
            for path in files:
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    arr = torch.from_numpy(
                        _np.array(im, dtype=_np.uint8)
                    ).permute(2, 0, 1).contiguous()
                img = arr.to(dtype=torch.float)
                if self.normalize:
                    img = img / 255.0
                if self.image_transform is not None:
                    img = self.image_transform(img)

                if img.dim() != 3:
                    raise ValueError(
                        f"image_transform must keep [C, H, W]; got "
                        f"shape {tuple(img.shape)} for {path}"
                    )

                ps = self.patch_size
                C, H, W = img.shape
                pad_mode = self.padding
                # image_to_patches handles padding="auto" itself.
                patches = image_to_patches(
                    img.unsqueeze(0), patch_size=ps,
                    padding=pad_mode if pad_mode == "auto" else "none",
                )[0]
                # Recompute grid shape after padding (image_to_patches uses
                # padded dimensions internally).
                if pad_mode == "auto":
                    pad_h = (-H) % ps
                    pad_w = (-W) % ps
                    H_pad = H + pad_h
                    W_pad = W + pad_w
                else:
                    if H % ps or W % ps:
                        raise ValueError(
                            f"{path}: image size {H}x{W} not divisible by "
                            f"patch_size={ps}; pass padding='auto'"
                        )
                    H_pad, W_pad = H, W
                n_h, n_w = H_pad // ps, W_pad // ps

                ei = _build_edge_index_from_grid(
                    (n_h, n_w), self.graph_builder,
                    self.knn_k, self.radius,
                )
                graphs.append(
                    Graph(
                        node_features=patches,
                        edge_index=ei,
                        graph_label=torch.tensor(cls_idx, dtype=torch.long),
                        metadata={
                            "path": str(path),
                            "class_name": cls,
                            "grid_shape": (n_h, n_w),
                            "patch_size": ps,
                        },
                    )
                )
        self.graphs = graphs

    def _build_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="image_folder_patch",
            source=str(self._root_path),
            license="user-supplied (image folder)",
            task="graph_classification",
            graph_type="homogeneous",
            num_graphs=len(self.graphs),
            num_classes=len(self.class_to_idx),
            extra={
                "patch_size": self.patch_size,
                "graph_builder": self.graph_builder,
                "padding": self.padding,
            },
        )


# ── Volume folder ────────────────────────────────────────────────────────────


def _load_volume_file(path: Path) -> torch.Tensor:
    """Load .npy / .npz / .pt and return a 4-D ``[C, D, H, W]`` tensor."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise OptionalDependencyError("numpy", "pip install numpy") from exc
        arr = np.load(path)
        t = torch.from_numpy(arr.copy()).float()
    elif suffix == ".npz":
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise OptionalDependencyError("numpy", "pip install numpy") from exc
        with np.load(path) as f:
            keys = list(f.keys())
            if not keys:
                raise ValueError(f"{path} contains no arrays")
            arr = f[keys[0]]
        t = torch.from_numpy(arr.copy()).float()
    elif suffix == ".pt":
        try:
            obj = torch.load(path, weights_only=True)
        except TypeError:  # pragma: no cover
            obj = torch.load(path)
        if not isinstance(obj, torch.Tensor):
            raise ValueError(f"{path} must contain a torch.Tensor; got {type(obj)}")
        t = obj.float()
    else:
        raise ValueError(
            f"Unsupported volume extension {suffix}. Supported: {_VOLUME_EXTS}"
        )

    if t.dim() == 3:
        t = t.unsqueeze(0)  # [1, D, H, W]
    if t.dim() != 4:
        raise ValueError(
            f"{path}: expected 3-D [D,H,W] or 4-D [C,D,H,W]; got {tuple(t.shape)}"
        )
    return t


class VolumeFolderPatchGraphDataset(InMemoryGraphDataset):
    """Walk ``root/class_name/*.{npy,npz,pt}`` and build 3-D patch graphs.

    File formats are handled natively — no nibabel / h5py dependency.
    Add support for those formats yourself by subclassing and
    overriding :meth:`_load`.
    """

    def __init__(
        self,
        root: str | Path,
        patch_size: int = 4,
        graph_builder: str = "grid",
        padding: str = "auto",
        graph_transform: TransformFn = None,
        target_transform: TargetTransformFn = None,
        max_volumes_per_class: Optional[int] = None,
    ) -> None:
        self._root_path = Path(root).expanduser()
        if not self._root_path.exists():
            raise FileNotFoundError(
                f"Volume folder root does not exist: {self._root_path}"
            )
        self.patch_size = int(patch_size)
        self.graph_builder = graph_builder
        if padding not in ("none", "auto"):
            raise ValueError(f"padding must be 'none' or 'auto'; got {padding!r}")
        self.padding = padding
        self.max_volumes_per_class = max_volumes_per_class

        self._class_dirs = _list_class_subdirs(self._root_path)
        self.class_to_idx: Dict[str, int] = {
            cls: i for i, cls in enumerate(self._class_dirs)
        }
        super().__init__(
            root=root, split=None,
            transform=graph_transform,
            target_transform=target_transform,
        )

    def _load(self, path: Path) -> torch.Tensor:
        return _load_volume_file(path)

    def _populate(self) -> None:
        graphs: List[Graph] = []
        for cls, dirpath in self._class_dirs.items():
            files = sorted(p for p in dirpath.iterdir()
                           if p.suffix.lower() in _VOLUME_EXTS)
            if self.max_volumes_per_class is not None:
                files = files[: int(self.max_volumes_per_class)]
            cls_idx = self.class_to_idx[cls]
            for path in files:
                vol = self._load(path)  # [C, D, H, W]
                ps = self.patch_size
                C, D, H, W = vol.shape
                if self.padding == "auto":
                    pad_d, pad_h, pad_w = (-D) % ps, (-H) % ps, (-W) % ps
                    if pad_d or pad_h or pad_w:
                        vol = torch.nn.functional.pad(
                            vol, (0, pad_w, 0, pad_h, 0, pad_d), value=0.0,
                        )
                else:
                    if D % ps or H % ps or W % ps:
                        raise ValueError(
                            f"{path}: volume size {D}x{H}x{W} not divisible "
                            f"by patch_size={ps}; pass padding='auto'"
                        )
                _, D2, H2, W2 = vol.shape
                patches = volume_to_patches(vol.unsqueeze(0), patch_size=ps)[0]
                grid = (D2 // ps, H2 // ps, W2 // ps)
                ei = _build_edge_index_from_grid(grid, self.graph_builder, 4, 1.5)
                graphs.append(
                    Graph(
                        node_features=patches,
                        edge_index=ei,
                        graph_label=torch.tensor(cls_idx, dtype=torch.long),
                        metadata={
                            "path": str(path),
                            "class_name": cls,
                            "grid_shape": grid,
                            "patch_size": ps,
                        },
                    )
                )
        self.graphs = graphs

    def _build_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="volume_folder_patch",
            source=str(self._root_path),
            license="user-supplied (volume folder)",
            task="graph_classification",
            graph_type="homogeneous",
            num_graphs=len(self.graphs),
            num_classes=len(self.class_to_idx),
            extra={"patch_size": self.patch_size},
        )
