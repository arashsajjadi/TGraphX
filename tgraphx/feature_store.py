"""Feature store foundations for large-scale graph learning.

Provides :class:`InMemoryFeatureStore` (default, pure-PyTorch) and an
optional :class:`MemmapFeatureStore` backed by NumPy memory-mapped files.

TGraphX does **not** claim billion-node production readiness.  These stores
are designed for single-machine out-of-core workflows where node features
are too large to fit in GPU memory simultaneously.

Stability: Beta (v0.5.0+).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

__all__ = [
    "InMemoryFeatureStore",
    "MemmapFeatureStore",
    "FeatureStoreError",
]


class FeatureStoreError(RuntimeError):
    """Raised for FeatureStore contract violations."""


# ── InMemoryFeatureStore ──────────────────────────────────────────────────────


class InMemoryFeatureStore:
    """Pure-PyTorch in-memory feature store.

    Stores named tensors and supports indexed fetch/update.  Suitable
    for graphs where all features fit in RAM/VRAM.

    Example::

        store = InMemoryFeatureStore()
        store.put("x", torch.randn(100, 64))           # all features
        x_batch = store.get("x", ids=torch.arange(10)) # first 10 nodes

    Args:
        device: Default device for stored tensors.  Individual put()
            calls may override this.

    Stability: Beta.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self._device = device or torch.device("cpu")
        self._store: Dict[str, torch.Tensor] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}

    # ── Write ─────────────────────────────────────────────────────────────────

    def put(
        self,
        name: str,
        tensor: torch.Tensor,
        ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Store a tensor (all rows or specific rows by ID).

        Args:
            name: Feature name (e.g. ``"x"``, ``"edge_attr"``).
            tensor: ``Tensor[N, *]`` of feature values.
            ids: Optional ``LongTensor[K]`` of row indices.  When
                ``None``, replaces the entire stored tensor.
        """
        if ids is None:
            self._store[name] = tensor.detach().to(self._device)
            self._meta[name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(self._device),
                "backend": "in_memory",
            }
        else:
            if name not in self._store:
                raise FeatureStoreError(f"Feature '{name}' not yet put; call put(name, full_tensor) first.")
            self._store[name][ids] = tensor.detach().to(self._device)

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(
        self,
        name: str,
        ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fetch a feature tensor.

        Args:
            name: Feature name.
            ids: Optional ``LongTensor[K]`` of row indices.

        Returns:
            ``Tensor[K, *]`` if ``ids`` provided, else the full tensor.

        Raises:
            FeatureStoreError: If ``name`` is not found.
        """
        if name not in self._store:
            raise FeatureStoreError(f"Feature '{name}' not found in store.")
        t = self._store[name]
        return t if ids is None else t[ids]

    def contains(self, name: str) -> bool:
        """Return ``True`` if feature ``name`` is stored."""
        return name in self._store

    def feature_names(self) -> List[str]:
        """Return list of stored feature names."""
        return list(self._store.keys())

    def metadata(self, name: str) -> Dict[str, Any]:
        """Return metadata dict for feature ``name``."""
        if name not in self._meta:
            raise FeatureStoreError(f"Feature '{name}' not found.")
        return dict(self._meta[name])

    def memory_estimate_bytes(self) -> int:
        """Estimate total RAM usage in bytes."""
        total = 0
        for t in self._store.values():
            total += t.numel() * t.element_size()
        return total

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "backend": "in_memory",
            "num_features": len(self._store),
            "features": {
                n: {
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                    "size_bytes": t.numel() * t.element_size(),
                }
                for n, t in self._store.items()
            },
            "total_bytes": self.memory_estimate_bytes(),
        }

    def to(self, device: Union[str, torch.device]) -> "InMemoryFeatureStore":
        """Move all stored tensors to ``device`` in-place."""
        device = torch.device(device)
        self._device = device
        for name in self._store:
            self._store[name] = self._store[name].to(device)
        return self


# ── MemmapFeatureStore ────────────────────────────────────────────────────────


class MemmapFeatureStore:
    """Disk-backed feature store using NumPy memory-mapped arrays.

    Features are written to ``.npy`` files in ``root`` and accessed
    lazily.  Suitable for features too large for RAM.

    Requires NumPy (``pip install numpy``).

    Args:
        root: Directory for ``.npy`` files and a ``metadata.json`` sidecar.

    Security:
        Uses ``numpy.load(..., allow_pickle=False)`` — no unsafe pickle.
        Root directory must be provided explicitly; no hidden writes.

    Stability: Beta.
    """

    def __init__(self, root: Union[str, Path]) -> None:
        self._root = Path(root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._meta_path = self._root / "feature_store_metadata.json"
        self._meta: Dict[str, Dict[str, Any]] = self._load_meta()

    # ── Metadata helpers ──────────────────────────────────────────────────────

    def _load_meta(self) -> Dict[str, Dict[str, Any]]:
        if self._meta_path.exists():
            try:
                return json.loads(self._meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_meta(self) -> None:
        text = json.dumps(self._meta, indent=2)
        fd, tmp = tempfile.mkstemp(dir=str(self._root), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
            os.replace(tmp, str(self._meta_path))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _npy_path(self, name: str) -> Path:
        return self._root / f"{name}.npy"

    # ── Write ─────────────────────────────────────────────────────────────────

    def put(
        self,
        name: str,
        tensor: torch.Tensor,
        ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Write tensor to disk.

        Args:
            name: Feature name.
            tensor: ``Tensor[N, *]``.
            ids: When ``None``, write the full tensor.  When provided,
                must have called ``put(name, full_tensor)`` first; only
                updates the indexed rows.
        """
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("MemmapFeatureStore requires numpy.") from exc

        if ids is None:
            arr = tensor.detach().cpu().numpy()
            np.save(str(self._npy_path(name)), arr)
            self._meta[name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "backend": "memmap",
                "path": str(self._npy_path(name)),
            }
            self._save_meta()
        else:
            if name not in self._meta:
                raise FeatureStoreError(f"Feature '{name}' not yet put.")
            arr = np.load(str(self._npy_path(name)), allow_pickle=False, mmap_mode="r+")
            arr[ids.cpu().numpy()] = tensor.detach().cpu().numpy()

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(
        self,
        name: str,
        ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Load feature from disk (chunked if ids provided).

        Args:
            name: Feature name.
            ids: Optional ``LongTensor[K]`` row indices.

        Returns:
            ``FloatTensor[K, *]`` or full tensor.
        """
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("MemmapFeatureStore requires numpy.") from exc
        if name not in self._meta:
            raise FeatureStoreError(f"Feature '{name}' not found.")
        arr = np.load(str(self._npy_path(name)), allow_pickle=False, mmap_mode="r")
        if ids is None:
            return torch.from_numpy(arr.copy())
        return torch.from_numpy(arr[ids.cpu().numpy()].copy())

    def contains(self, name: str) -> bool:
        return name in self._meta

    def feature_names(self) -> List[str]:
        return list(self._meta.keys())

    def metadata(self, name: str) -> Dict[str, Any]:
        if name not in self._meta:
            raise FeatureStoreError(f"Feature '{name}' not found.")
        return dict(self._meta[name])

    def summary(self) -> Dict[str, Any]:
        return {
            "backend": "memmap",
            "root": str(self._root),
            "num_features": len(self._meta),
            "features": {
                n: {"shape": m.get("shape"), "dtype": m.get("dtype")}
                for n, m in self._meta.items()
            },
        }
