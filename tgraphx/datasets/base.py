"""Base classes for TGraphX datasets.

Three layers:

* :class:`BaseGraphDataset` — abstract iterator + metadata + transform
  hook.  Doesn't assume anything about how data is produced.
* :class:`InMemoryGraphDataset` — concrete subclass that holds a
  ``list[Graph]`` and stores processed payloads with
  :func:`torch.save` / :func:`torch.load`.
* :class:`DownloadableGraphDataset` — base class for datasets that
  fetch raw files from the internet.  Provides a ``prepare()`` method
  that orchestrates ``download → process → load`` and refuses to do
  any of those silently.

Subclasses are responsible for actually defining the data-generation /
parsing logic; everything else is plumbing.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import torch

from ..core.graph import Graph
from .cache import (
    atomic_save_torch,
    resolve_dataset_root,
    safe_mkdir,
)
from .errors import DatasetFilesNotFoundError
from .metadata import DatasetMetadata


# Type alias for "any TGraphX item a dataset might return".
GraphLike = Any  # Graph | HeteroGraph | TemporalGraphSequence

TransformFn = Optional[Callable[[GraphLike], GraphLike]]
TargetTransformFn = Optional[Callable[[Any], Any]]


class BaseGraphDataset(ABC):
    """Abstract base for all TGraphX datasets.

    Subclasses must implement :meth:`__len__` and :meth:`get` — never
    :meth:`__getitem__`, which is the public hook that applies
    ``transform`` and ``target_transform``.

    Args:
        root: Optional dataset root; if ``None`` the default cache is
            used (``$TGRAPHX_DATA`` or ``~/.cache/tgraphx/datasets``).
        split: Optional split name (``"train"`` / ``"val"`` / ``"test"``
            or any subclass-specific value).
        transform: Function applied to each graph object before
            returning it.  Pure (no in-place mutation) is recommended
            but not enforced.
        target_transform: Function applied to graph labels (when
            applicable).  Useful for one-hot encoding, label smoothing,
            etc.  Subclasses decide how / whether to invoke it.
        metadata: Optional pre-built :class:`DatasetMetadata`.  When
            ``None``, subclasses build one in :meth:`_build_metadata`.
    """

    def __init__(
        self,
        root: Optional[str | Path] = None,
        split: Optional[str] = None,
        transform: TransformFn = None,
        target_transform: TargetTransformFn = None,
        metadata: Optional[DatasetMetadata] = None,
    ) -> None:
        self._root_arg = root
        self._split = split
        self.transform = transform
        self.target_transform = target_transform
        self._metadata: Optional[DatasetMetadata] = metadata

    # ── Mandatory subclass surface ───────────────────────────────────────────

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def get(self, idx: int) -> GraphLike:
        """Return the raw item at ``idx`` (no transform applied)."""

    # ── Public API ───────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> GraphLike:
        if not (-len(self) <= idx < len(self)):
            raise IndexError(
                f"Dataset index {idx} out of range [0, {len(self)})"
            )
        if idx < 0:
            idx += len(self)
        item = self.get(idx)
        if self.transform is not None:
            item = self.transform(item)
        return item

    def __iter__(self) -> Iterable[GraphLike]:
        for i in range(len(self)):
            yield self[i]

    @property
    def split(self) -> Optional[str]:
        return self._split

    @property
    def root(self) -> Path:
        """Resolved on-disk root for this dataset."""
        return resolve_dataset_root(self._root_arg, self._dataset_slug())

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.root / "processed"

    @property
    def metadata(self) -> DatasetMetadata:
        if self._metadata is None:
            self._metadata = self._build_metadata()
        return self._metadata

    # ── Subclass hooks ──────────────────────────────────────────────────────

    def _dataset_slug(self) -> str:
        """Override to control the on-disk directory name; default = name."""
        if self._metadata is not None:
            return self._metadata.name
        return type(self).__name__

    def _build_metadata(self) -> DatasetMetadata:
        """Build a :class:`DatasetMetadata` lazily; override in subclasses."""
        return DatasetMetadata(name=type(self).__name__)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        meta = self.metadata
        sample_shape: Optional[List[int]] = None
        if len(self) > 0:
            try:
                sample = self.get(0)
                if hasattr(sample, "node_features"):
                    sample_shape = list(sample.node_features.shape)
            except Exception:  # pragma: no cover  (best-effort summary)
                sample_shape = None
        return {
            "name": meta.name,
            "task": meta.task,
            "graph_type": meta.graph_type,
            "len": len(self),
            "split": self._split,
            "root": str(self.root),
            "sample_node_features_shape": sample_shape,
        }

    def describe(self) -> str:
        meta = self.metadata
        lines: List[str] = [meta.short_summary()]
        if self._split:
            lines.append(f"split: {self._split}")
        lines.append(f"len: {len(self)}")
        lines.append(f"root: {self.root}")
        if meta.citation:
            lines.append(f"citation: {meta.citation}")
        if meta.license:
            lines.append(f"license: {meta.license}")
        if meta.source_url:
            lines.append(f"source: {meta.source_url}")
        return "\n".join(lines)


# ── In-memory ────────────────────────────────────────────────────────────────


class InMemoryGraphDataset(BaseGraphDataset):
    """Holds a ``list[Graph]`` (or other graph-like objects) in memory.

    Subclasses override :meth:`_generate` (or fill ``self.graphs``
    directly) and :meth:`_build_metadata`.

    Processed-cache helpers (:meth:`save_processed` /
    :meth:`load_processed`) are optional — datasets that are cheap to
    regenerate (synthetic, folder-backed) typically skip them entirely.
    """

    def __init__(
        self,
        root: Optional[str | Path] = None,
        split: Optional[str] = None,
        transform: TransformFn = None,
        target_transform: TargetTransformFn = None,
        metadata: Optional[DatasetMetadata] = None,
    ) -> None:
        super().__init__(
            root=root, split=split, transform=transform,
            target_transform=target_transform, metadata=metadata,
        )
        self.graphs: List[GraphLike] = []
        self._populate()

    # Subclasses fill self.graphs.  Default: empty.
    def _populate(self) -> None:
        pass

    # ── BaseGraphDataset surface ────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.graphs)

    def get(self, idx: int) -> GraphLike:
        return self.graphs[idx]

    # ── Processed-cache helpers ─────────────────────────────────────────────

    def processed_path(self, fname: str = "data.pt") -> Path:
        return self.processed_dir / fname

    def save_processed(self, fname: str = "data.pt") -> Path:
        """Persist the in-memory list to disk via :func:`torch.save`."""
        safe_mkdir(self.processed_dir)
        path = self.processed_path(fname)
        atomic_save_torch({"graphs": self.graphs}, path)
        # Save metadata next to it for human readers.
        try:
            self.metadata.processed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.metadata.save_json(self.processed_dir / "metadata.json")
        except Exception:  # pragma: no cover
            pass
        return path

    def load_processed(self, fname: str = "data.pt") -> None:
        """Load processed list from disk into ``self.graphs``."""
        path = self.processed_path(fname)
        if not path.exists():
            raise DatasetFilesNotFoundError(
                f"Processed file {path} is missing. Pass download=True or "
                f"call .save_processed() first."
            )
        try:
            payload = torch.load(path, weights_only=False)
        except TypeError:  # pragma: no cover  (older torch)
            payload = torch.load(path)
        self.graphs = payload["graphs"]


# ── Downloadable ─────────────────────────────────────────────────────────────


class DownloadableGraphDataset(InMemoryGraphDataset):
    """Base class for datasets that fetch raw files from the network.

    Subclasses override:

    * :attr:`raw_file_names` — sequence of filenames expected under
      ``raw_dir`` after download.
    * :meth:`download` — actually fetch (only called when
      ``download=True``).
    * :meth:`process` — read raw → fill ``self.graphs``.

    The :meth:`prepare` orchestrator wires these together and refuses
    to download anything unless the user explicitly asks.
    """

    raw_file_names: Sequence[str] = ()
    processed_file_names: Sequence[str] = ("data.pt",)

    def __init__(
        self,
        root: Optional[str | Path] = None,
        split: Optional[str] = None,
        transform: TransformFn = None,
        target_transform: TargetTransformFn = None,
        download: bool = False,
        force_reload: bool = False,
        metadata: Optional[DatasetMetadata] = None,
    ) -> None:
        # Skip InMemory's _populate(); we orchestrate it ourselves.
        BaseGraphDataset.__init__(
            self, root=root, split=split, transform=transform,
            target_transform=target_transform, metadata=metadata,
        )
        self.graphs: List[GraphLike] = []
        self._download_flag = bool(download)
        self._force_reload = bool(force_reload)
        self.prepare(download=self._download_flag, force_reload=self._force_reload)

    # ── Public surface ──────────────────────────────────────────────────────

    @property
    def raw_paths(self) -> List[Path]:
        return [self.raw_dir / n for n in self.raw_file_names]

    @property
    def processed_paths(self) -> List[Path]:
        return [self.processed_dir / n for n in self.processed_file_names]

    def has_raw(self) -> bool:
        return all(p.exists() for p in self.raw_paths)

    def has_processed(self) -> bool:
        return all(p.exists() for p in self.processed_paths)

    def prepare(self, download: bool = False, force_reload: bool = False) -> None:
        """Ensure ``self.graphs`` is populated."""
        safe_mkdir(self.raw_dir)
        safe_mkdir(self.processed_dir)

        if force_reload or not self.has_processed():
            if not self.has_raw():
                if not download:
                    raise DatasetFilesNotFoundError(
                        f"Raw dataset files missing under {self.raw_dir}. "
                        f"Pass download=True to fetch them, or place files "
                        f"manually at: {[str(p) for p in self.raw_paths]}"
                    )
                self.download()
                if not self.has_raw():
                    raise DatasetFilesNotFoundError(
                        f"download() did not produce expected raw files: "
                        f"{[str(p) for p in self.raw_paths]}"
                    )
            self.process()
            self.save_processed(self.processed_file_names[0])
        else:
            self.load_processed(self.processed_file_names[0])

    # ── Subclass hooks ──────────────────────────────────────────────────────

    def download(self) -> None:  # pragma: no cover  (subclass-specific)
        raise NotImplementedError

    def process(self) -> None:  # pragma: no cover  (subclass-specific)
        raise NotImplementedError

    # ── Cleanup helpers ─────────────────────────────────────────────────────

    def clear_processed(self) -> None:
        for p in self.processed_paths:
            try:
                p.unlink()
            except OSError:
                pass

    def clear_raw(self) -> None:
        for p in self.raw_paths:
            try:
                p.unlink()
            except OSError:
                pass


# ── External adapter base ────────────────────────────────────────────────────


class ExternalDatasetAdapter(BaseGraphDataset):
    """Base for adapters that wrap a third-party loader (PyG / DGL / OGB / torchvision).

    Subclasses are responsible for:

    * Lazy-importing the upstream package inside ``__init__`` /
      ``get`` / ``_load_upstream`` (never at module top level).
    * Translating each upstream sample to a TGraphX
      :class:`Graph` / :class:`HeteroGraph` /
      :class:`TemporalGraphSequence`.
    * Raising :class:`OptionalDependencyError` with a clear install
      hint when the upstream package is missing.

    The base class provides the common attribute layout
    (``upstream_name``, ``upstream_library``) and reuses
    :class:`BaseGraphDataset`'s metadata/root/transform plumbing.
    """

    upstream_library: str = ""
    upstream_install_hint: str = ""
