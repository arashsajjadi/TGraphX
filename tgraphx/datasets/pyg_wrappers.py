"""Optional PyG-backed dataset adapter.

PyG (``torch_geometric``) is **optional**.  Importing this module is
fine without PyG installed; constructing any of these adapters lazily
imports it and raises a helpful :class:`OptionalDependencyError` if
the package is missing.

The adapters are *converters*, not API replacements: they expose the
underlying PyG dataset and translate each item to a TGraphX
:class:`~tgraphx.Graph` (or :class:`~tgraphx.HeteroGraph`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import ExternalDatasetAdapter, TargetTransformFn, TransformFn
from .converters import from_pyg_data, from_pyg_heterodata
from .errors import OptionalDependencyError
from .metadata import DatasetMetadata

__all__ = [
    "PyGDatasetAdapter",
    "PyGPlanetoidDataset",
    "PyGTUDatasetAdapter",
]


_PYG_HINT = (
    "PyG-backed datasets require torch-geometric. "
    "Install with `pip install torch-geometric` "
    "(see https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)."
)


def _require_pyg():
    try:
        import torch_geometric  # noqa: F401
        from torch_geometric import datasets as _pyg_datasets  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("torch_geometric", _PYG_HINT) from exc


class PyGDatasetAdapter(ExternalDatasetAdapter):
    """Generic wrapper around any ``torch_geometric.datasets.<Cls>``.

    Args:
        dataset_cls: Class name or class object inside
            :mod:`torch_geometric.datasets`.
        root: Cache directory (forwarded to the upstream loader).
        download: Forwarded to the upstream constructor.  Note: PyG's
            own ``Planetoid`` / ``TUDataset`` will *download by default*
            unless you pass an existing root with raw files; we pass
            this flag through but cannot fully prevent PyG's behaviour.
            **For a no-network test, point ``root`` at a directory
            that already contains the raw files.**
        dataset_kwargs: Forwarded to the upstream class constructor.
        is_hetero: When ``True``, treat each item as a PyG
            ``HeteroData`` and convert to :class:`HeteroGraph`.
        transform: TGraphX-side post-conversion transform.
        target_transform: Optional transform applied to graph labels.
    """

    upstream_library = "torch_geometric"
    upstream_install_hint = _PYG_HINT

    def __init__(
        self,
        dataset_cls: str | type,
        root: Optional[str | Path] = None,
        download: bool = False,  # noqa: ARG002 — informational; PyG decides on its own
        dataset_kwargs: Optional[Dict[str, Any]] = None,
        is_hetero: bool = False,
        transform: TransformFn = None,
        target_transform: TargetTransformFn = None,
    ) -> None:
        _require_pyg()
        from torch_geometric import datasets as pyg_datasets

        if isinstance(dataset_cls, str):
            cls = getattr(pyg_datasets, dataset_cls, None)
            if cls is None:
                raise ValueError(
                    f"torch_geometric.datasets has no attribute "
                    f"{dataset_cls!r}"
                )
        else:
            cls = dataset_cls

        from .cache import resolve_dataset_root
        upstream_root = resolve_dataset_root(root, f"pyg/{cls.__name__}")
        upstream_root.mkdir(parents=True, exist_ok=True)

        kw = dict(dataset_kwargs or {})
        try:
            self._upstream = cls(root=str(upstream_root), **kw)
        except TypeError:
            self._upstream = cls(str(upstream_root), **kw)
        self._is_hetero = bool(is_hetero)
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
        )

    def __len__(self) -> int:
        return len(self._upstream)

    def get(self, idx: int):
        item = self._upstream[idx]
        if self._is_hetero:
            return from_pyg_heterodata(item)
        return from_pyg_data(item)

    def _build_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name=f"pyg:{type(self._upstream).__name__.lower()}",
            source="torch_geometric",
            upstream_library="torch_geometric",
            source_url="https://pytorch-geometric.readthedocs.io/",
            citation="Fey & Lenssen, ICLR-W 2019; see upstream dataset citation.",
            license="See upstream dataset documentation.",
            task=None,
            graph_type="heterogeneous" if self._is_hetero else "homogeneous",
            num_graphs=len(self._upstream),
            extra={"upstream_class": type(self._upstream).__name__},
        )


# ── Curated subclasses ───────────────────────────────────────────────────────


class PyGPlanetoidDataset(PyGDatasetAdapter):
    """Wrapper for ``torch_geometric.datasets.Planetoid`` (Cora/CiteSeer/PubMed)."""

    def __init__(
        self,
        name: str,
        root: Optional[str | Path] = None,
        download: bool = False,
        transform: TransformFn = None,
        target_transform: TargetTransformFn = None,
    ) -> None:
        super().__init__(
            dataset_cls="Planetoid",
            root=root,
            download=download,
            dataset_kwargs={"name": name},
            transform=transform,
            target_transform=target_transform,
        )


class PyGTUDatasetAdapter(PyGDatasetAdapter):
    """Wrapper for ``torch_geometric.datasets.TUDataset`` (MUTAG / PROTEINS / ...)."""

    def __init__(
        self,
        name: str,
        root: Optional[str | Path] = None,
        download: bool = False,
        transform: TransformFn = None,
        target_transform: TargetTransformFn = None,
    ) -> None:
        super().__init__(
            dataset_cls="TUDataset",
            root=root,
            download=download,
            dataset_kwargs={"name": name},
            transform=transform,
            target_transform=target_transform,
        )
