"""Optional DGL-backed dataset adapter.

DGL is **optional** — see the install notes upstream because their
wheels are platform-sensitive and shouldn't be a TGraphX hard
dependency.  This module imports DGL lazily.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .base import ExternalDatasetAdapter, TargetTransformFn, TransformFn
from .converters import from_dgl_graph, from_dgl_heterograph
from .errors import OptionalDependencyError
from .metadata import DatasetMetadata

__all__ = [
    "DGLDatasetAdapter",
    "DGLCitationDatasetAdapter",
]


_DGL_HINT = (
    "DGL-backed datasets require dgl. "
    "Install: see https://www.dgl.ai/pages/start.html (DGL wheels are "
    "platform-specific; we do not pin them as a TGraphX extra)."
)


def _require_dgl():
    try:
        import dgl  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("dgl", _DGL_HINT) from exc


class DGLDatasetAdapter(ExternalDatasetAdapter):
    """Generic wrapper around any DGL ``DGLDataset`` subclass.

    Args:
        dataset_cls: Class object or string name resolved against
            :mod:`dgl.data`.
        root: Cache directory (passed as ``raw_dir`` to most DGL
            datasets).
        download: Forwarded to the upstream constructor.  DGL caches
            decisions on its own; with ``download=False`` and missing
            files the upstream class will typically raise.
        dataset_kwargs: Extra kwargs forwarded to the upstream
            constructor.
        is_hetero: Treat the upstream graph as a heterograph.
    """

    upstream_library = "dgl"
    upstream_install_hint = _DGL_HINT

    def __init__(
        self,
        dataset_cls: str | type,
        root: Optional[str | Path] = None,
        download: bool = False,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
        is_hetero: bool = False,
        transform: TransformFn = None,
        target_transform: TargetTransformFn = None,
    ) -> None:
        _require_dgl()
        import dgl
        from dgl import data as dgl_data

        if isinstance(dataset_cls, str):
            cls = getattr(dgl_data, dataset_cls, None)
            if cls is None:
                raise ValueError(
                    f"dgl.data has no attribute {dataset_cls!r}"
                )
        else:
            cls = dataset_cls

        from .cache import resolve_dataset_root
        upstream_root = resolve_dataset_root(root, f"dgl/{cls.__name__}")
        upstream_root.mkdir(parents=True, exist_ok=True)

        kw = dict(dataset_kwargs or {})
        # DGL's constructor uses raw_dir / save_dir.
        kw.setdefault("raw_dir", str(upstream_root))
        # 'download' and 'force_reload' are honoured by some DGL datasets only.
        if "force_reload" not in kw:
            kw["force_reload"] = False
        try:
            self._upstream = cls(**kw)
        except TypeError:
            kw.pop("raw_dir", None)
            self._upstream = cls(**kw)
        self._is_hetero = bool(is_hetero)
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
        )

    def __len__(self) -> int:
        try:
            return len(self._upstream)
        except TypeError:
            return 1  # single-graph DGL datasets aren't always sized

    def get(self, idx: int):
        # DGL datasets sometimes return (g, label) tuples or just g.
        item = self._upstream[idx] if hasattr(self._upstream, "__getitem__") else self._upstream
        label = None
        if isinstance(item, tuple):
            graph_obj, label = item[0], item[1] if len(item) > 1 else None
        else:
            graph_obj = item

        if self._is_hetero:
            graph = from_dgl_heterograph(graph_obj)
        else:
            graph = from_dgl_graph(graph_obj)

        if label is not None and graph.graph_label is None:
            import torch
            graph.graph_label = label if isinstance(label, torch.Tensor) else torch.tensor(label)
        return graph

    def _build_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name=f"dgl:{type(self._upstream).__name__.lower()}",
            source="dgl",
            upstream_library="dgl",
            source_url="https://www.dgl.ai/",
            citation="Wang et al., DGL — see upstream dataset citation.",
            license="See upstream dataset documentation.",
            graph_type="heterogeneous" if self._is_hetero else "homogeneous",
            extra={"upstream_class": type(self._upstream).__name__},
        )


class DGLCitationDatasetAdapter(DGLDatasetAdapter):
    """Curated wrapper for DGL citation datasets (Cora / CiteSeer / PubMed)."""

    _CITATION_TO_CLASS = {
        "cora": "CoraGraphDataset",
        "citeseer": "CiteseerGraphDataset",
        "pubmed": "PubmedGraphDataset",
    }

    def __init__(
        self,
        name: str,
        root: Optional[str | Path] = None,
        download: bool = False,
        transform: TransformFn = None,
        target_transform: TargetTransformFn = None,
    ) -> None:
        key = name.lower()
        if key not in self._CITATION_TO_CLASS:
            raise ValueError(
                f"Unknown citation dataset {name!r}; expected one of "
                f"{list(self._CITATION_TO_CLASS)}"
            )
        super().__init__(
            dataset_cls=self._CITATION_TO_CLASS[key],
            root=root,
            download=download,
            transform=transform,
            target_transform=target_transform,
        )
