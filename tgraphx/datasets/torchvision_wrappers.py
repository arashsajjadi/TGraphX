"""Optional torchvision-backed patch-graph datasets.

These wrappers use upstream torchvision loaders (so we never
redistribute MNIST / CIFAR / SVHN / etc.) and convert each image into
a TGraphX patch :class:`Graph`.  They:

* import torchvision **lazily** inside ``__init__``;
* never download unless the user passes ``download=True``;
* make conversion happen on-the-fly inside ``get(idx)`` (no eager
  iteration of the full upstream dataset).

The license / citation responsibilities stay with the upstream
dataset.  TGraphX is only a converter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch

from .base import (
    BaseGraphDataset,
    ExternalDatasetAdapter,
    TargetTransformFn,
    TransformFn,
)
from .converters import torchvision_image_to_patch_graph
from .errors import OptionalDependencyError
from .metadata import DatasetMetadata


__all__ = [
    "TorchvisionPatchGraphDataset",
    "MNISTPatchGraphDataset",
    "FashionMNISTPatchGraphDataset",
    "KMNISTPatchGraphDataset",
    "CIFAR10PatchGraphDataset",
    "CIFAR100PatchGraphDataset",
    "SVHNPatchGraphDataset",
    "STL10PatchGraphDataset",
    "FakeDataPatchGraphDataset",
]


_TORCHVISION_HINT = (
    "Torchvision-backed datasets require torchvision (a base TGraphX dependency). "
    "Install or reinstall with `pip install torchvision`."
)


def _require_torchvision():
    try:
        import torchvision  # noqa: F401
        from torchvision import datasets as tv_datasets  # noqa: F401
        from torchvision.transforms import functional as TF  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("torchvision", _TORCHVISION_HINT) from exc


class TorchvisionPatchGraphDataset(ExternalDatasetAdapter):
    """Generic adapter wrapping any torchvision image classification dataset.

    Args:
        dataset_class_or_name: Either a :mod:`torchvision.datasets`
            class, or a string naming one (``"MNIST"``, ``"CIFAR10"``,
            ``"FakeData"``, ...).
        root: Directory the upstream loader will use for caching the
            raw data.  Defaults to TGraphX's cache root.
        download: Forwarded to the upstream constructor.  ``False`` by
            default — *no hidden downloads*.
        train / split: Whichever the upstream class accepts; passed
            through as keyword arguments.
        upstream_kwargs: Extra keyword arguments forwarded to the
            upstream constructor.
        patch_size, graph_builder, knn_k, radius, padding: Patch-graph
            parameters.
        transform: TGraphX-side graph transform (applied after
            conversion).
        target_transform: Forwarded — applied to the integer class id
            returned by the upstream dataset before being attached to
            the graph.
    """

    upstream_library = "torchvision"
    upstream_install_hint = _TORCHVISION_HINT

    def __init__(
        self,
        dataset_class_or_name: str | type,
        *,
        root: Optional[str | Path] = None,
        download: bool = False,
        train: Optional[bool] = None,
        split: Optional[str] = None,
        upstream_kwargs: Optional[Dict[str, Any]] = None,
        patch_size: int = 7,
        graph_builder: str = "grid",
        knn_k: int = 4,
        radius: float = 1.5,
        padding: str = "auto",
        transform: TransformFn = None,
        target_transform: TargetTransformFn = None,
    ) -> None:
        _require_torchvision()
        from torchvision import datasets as tv_datasets

        if isinstance(dataset_class_or_name, str):
            cls = getattr(tv_datasets, dataset_class_or_name, None)
            if cls is None:
                raise ValueError(
                    f"torchvision.datasets has no attribute "
                    f"{dataset_class_or_name!r}"
                )
        else:
            cls = dataset_class_or_name

        # Resolve the upstream root.  We let torchvision manage its own
        # subdirectory layout under our cache root.
        from .cache import resolve_dataset_root
        upstream_root = resolve_dataset_root(root, f"torchvision/{cls.__name__}")
        upstream_root.mkdir(parents=True, exist_ok=True)

        kw = dict(upstream_kwargs or {})
        if train is not None:
            kw.setdefault("train", train)
        if split is not None:
            kw.setdefault("split", split)
        kw.setdefault("download", bool(download))

        # Inspect the constructor: torchvision datasets that take a `root`
        # argument include MNIST/CIFAR/SVHN; FakeData doesn't.  Pass `root`
        # only if the constructor accepts it.
        import inspect
        sig = inspect.signature(cls.__init__)
        accepts_root = "root" in sig.parameters
        # Some constructors don't accept 'download' either (e.g. FakeData).
        accepts_download = "download" in sig.parameters
        if not accepts_download:
            kw.pop("download", None)
        try:
            if accepts_root:
                self._upstream = cls(str(upstream_root), **kw)
            else:
                self._upstream = cls(**kw)
        except TypeError:
            # Last-resort retry without train/split.
            for k in ("train", "split"):
                kw.pop(k, None)
            if accepts_root:
                self._upstream = cls(str(upstream_root), **kw)
            else:
                self._upstream = cls(**kw)

        self.patch_size = int(patch_size)
        self.graph_builder = graph_builder
        self.knn_k = int(knn_k)
        self.radius = float(radius)
        self.padding = padding
        self._target_transform_fn = target_transform

        super().__init__(
            root=root, split=split or ("train" if train else None),
            transform=transform,
            target_transform=target_transform,
        )

    def __len__(self) -> int:
        return len(self._upstream)

    def get(self, idx: int):
        sample, target = self._upstream[idx]
        if self._target_transform_fn is not None:
            target = self._target_transform_fn(target)
        # Convert PIL → tensor via torchvision.transforms.functional
        from torchvision.transforms import functional as TF
        if not isinstance(sample, torch.Tensor):
            sample = TF.to_tensor(sample)  # [C, H, W], float in [0,1]
        return torchvision_image_to_patch_graph(
            sample, target,
            patch_size=self.patch_size,
            graph_builder=self.graph_builder,
            knn_k=self.knn_k,
            radius=self.radius,
            padding=self.padding,
        )

    def _build_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name=f"torchvision:{type(self._upstream).__name__.lower()}_patch",
            source="torchvision",
            upstream_library="torchvision",
            source_url="https://pytorch.org/vision/stable/datasets.html",
            citation="See upstream dataset documentation.",
            license="See upstream dataset documentation.",
            task="graph_classification",
            graph_type="homogeneous",
            num_graphs=len(self._upstream),
            extra={
                "upstream_class": type(self._upstream).__name__,
                "patch_size": self.patch_size,
                "graph_builder": self.graph_builder,
            },
        )


# ── Curated convenience subclasses ───────────────────────────────────────────


def _make_curated(name: str, default_kwargs: Dict[str, Any]):
    """Factory that builds a thin subclass with sensible defaults."""

    class _Wrapped(TorchvisionPatchGraphDataset):
        def __init__(self, **kwargs):
            kw = {**default_kwargs, **kwargs}
            kw["dataset_class_or_name"] = name
            super().__init__(**kw)

    _Wrapped.__name__ = f"{name}PatchGraphDataset"
    _Wrapped.__qualname__ = _Wrapped.__name__
    return _Wrapped


MNISTPatchGraphDataset = _make_curated("MNIST", {"patch_size": 7, "train": True})
FashionMNISTPatchGraphDataset = _make_curated("FashionMNIST", {"patch_size": 7, "train": True})
KMNISTPatchGraphDataset = _make_curated("KMNIST", {"patch_size": 7, "train": True})
CIFAR10PatchGraphDataset = _make_curated("CIFAR10", {"patch_size": 8, "train": True})
CIFAR100PatchGraphDataset = _make_curated("CIFAR100", {"patch_size": 8, "train": True})
SVHNPatchGraphDataset = _make_curated("SVHN", {"patch_size": 8, "split": "train"})
STL10PatchGraphDataset = _make_curated("STL10", {"patch_size": 8, "split": "train"})
FakeDataPatchGraphDataset = _make_curated("FakeData", {"patch_size": 8})
