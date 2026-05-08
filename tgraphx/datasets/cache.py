"""Dataset cache layout, atomic IO, and root-directory resolution.

Default root priority:

1. Explicit ``root`` argument passed by the user.
2. ``TGRAPHX_DATA`` environment variable.
3. ``~/.cache/tgraphx/datasets`` (XDG-style).

Atomic writes go through a ``.tmp`` file then ``os.replace`` to avoid
partial-file corruption on crashes.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

__all__ = [
    "get_default_cache_root",
    "resolve_dataset_root",
    "safe_mkdir",
    "atomic_write_bytes",
    "atomic_save_torch",
    "atomic_save_json",
    "load_json",
    "cache_summary",
    "clear_cache",
]


# ── Root resolution ──────────────────────────────────────────────────────────


def get_default_cache_root() -> Path:
    """Return the directory used when the user does not specify ``root``.

    Honours ``TGRAPHX_DATA``; otherwise falls back to ``~/.cache/tgraphx/datasets``.
    Does **not** create the directory — that happens lazily when the first
    dataset writes to it.
    """
    env = os.environ.get("TGRAPHX_DATA")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "tgraphx" / "datasets"


def resolve_dataset_root(root: Optional[str | Path], dataset_name: str) -> Path:
    """Resolve the on-disk root for one specific dataset.

    Args:
        root: User-supplied root directory (parent of all datasets).  When
            ``None``, uses :func:`get_default_cache_root`.
        dataset_name: Canonical dataset name; safe-slugified before joining.

    Returns:
        ``Path`` that points at ``<root>/<dataset_slug>``.  The directory
        is **not** created here; callers materialise it as needed.
    """
    base = Path(root).expanduser() if root else get_default_cache_root()
    safe = _slugify_dataset_name(dataset_name)
    return base / safe


def _slugify_dataset_name(name: str) -> str:
    """Make a dataset name safe to use as a directory name on every OS."""
    bad = '<>:"/\\|?*\0'
    out = "".join("_" if ch in bad else ch for ch in name)
    return out.strip().strip(".") or "dataset"


# ── Filesystem helpers ───────────────────────────────────────────────────────


def safe_mkdir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if absent; return as :class:`Path`."""
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """Atomically write ``data`` to ``path`` via a sibling ``.tmp`` file.

    Designed to work on Linux, macOS, and Windows: uses
    :func:`os.replace`, which is atomic on POSIX and an
    ``MoveFileExW`` on Windows.
    """
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_path, target)
    except Exception:
        # Clean up partial file.
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return target


def atomic_save_torch(obj: Any, path: str | Path) -> Path:
    """Atomically ``torch.save`` an object to disk.

    Use this for processed tensor caches.  Note: ``torch.load`` should be
    called by callers with ``weights_only=True`` whenever the payload is
    purely tensors.
    """
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent),
    )
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return target


def atomic_save_json(obj: Any, path: str | Path) -> Path:
    text = json.dumps(obj, indent=2, default=str)
    return atomic_write_bytes(path, text.encode("utf-8"))


def load_json(path: str | Path) -> Any:
    text = Path(path).expanduser().read_text(encoding="utf-8")
    return json.loads(text)


# ── Cache management ─────────────────────────────────────────────────────────


def cache_summary(root: Optional[str | Path] = None) -> Dict[str, Any]:
    """Return a small summary dict describing the cache directory."""
    base = Path(root).expanduser() if root else get_default_cache_root()
    info: Dict[str, Any] = {"root": str(base), "exists": base.exists(), "datasets": []}
    if not base.exists():
        return info
    for child in sorted(base.iterdir()):
        if child.is_dir():
            size = sum(p.stat().st_size for p in child.rglob("*") if p.is_file())
            info["datasets"].append({
                "name": child.name,
                "path": str(child),
                "size_bytes": size,
            })
    return info


def clear_cache(
    root: Optional[str | Path] = None,
    dataset_name: Optional[str] = None,
    dry_run: bool = True,
) -> List[str]:
    """List or remove cached files.

    Defaults to ``dry_run=True`` so accidental ``clear_cache()`` calls do
    not lose data.
    """
    import shutil

    base = Path(root).expanduser() if root else get_default_cache_root()
    if not base.exists():
        return []
    targets: List[Path]
    if dataset_name is None:
        targets = sorted(p for p in base.iterdir() if p.is_dir())
    else:
        slug = _slugify_dataset_name(dataset_name)
        candidate = base / slug
        targets = [candidate] if candidate.exists() else []
    if dry_run:
        return [str(p) for p in targets]
    removed: List[str] = []
    for p in targets:
        shutil.rmtree(p, ignore_errors=True)
        removed.append(str(p))
    return removed
