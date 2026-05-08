"""Safe download / verification / extraction utilities.

These helpers are intentionally minimal — they never run at import time
and never run during tests (every dataset gates them behind an explicit
``download=True``).  Tests in this repo monkey-patch
:func:`download_url` rather than hitting the network.

Security:

* Atomic downloads via a sibling ``.tmp`` file.
* Optional checksum verification (SHA-256 by default).
* Archive extraction blocks path traversal on every OS — refuses to
  write members whose resolved path lies outside the destination.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen

from .errors import DatasetExtractionError, DatasetIntegrityError

__all__ = [
    "download_url",
    "verify_checksum",
    "is_archive",
    "extract_archive",
    "safe_extract_zip",
    "safe_extract_tar",
    "maybe_download",
]


# ── Download ─────────────────────────────────────────────────────────────────


def download_url(
    url: str,
    dst: str | Path,
    checksum: Optional[str] = None,
    algorithm: str = "sha256",
    timeout: float = 60.0,
    chunk_size: int = 1 << 16,
) -> Path:
    """Atomically download ``url`` to ``dst`` and (optionally) verify checksum.

    Args:
        url: Source URL (``http://`` / ``https://``).
        dst: Destination file path.  Parent directories are created.
        checksum: Hex digest the downloaded file must match (when given).
        algorithm: Hash algorithm name (default ``"sha256"``).
        timeout: Network timeout in seconds.
        chunk_size: Stream chunk size.

    Returns:
        :class:`Path` to the finalised file.

    Raises:
        URLError: On network failure.
        DatasetIntegrityError: If a checksum was supplied and did not match.
    """
    target = Path(dst).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".part", dir=str(target.parent),
    )
    os.close(fd)

    try:
        with urlopen(url, timeout=timeout) as resp, open(tmp_path, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
        if checksum is not None:
            verify_checksum(tmp_path, checksum, algorithm=algorithm)
        os.replace(tmp_path, target)
    except (URLError, DatasetIntegrityError, OSError):
        # Clean up partial download on any error.
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return target


def verify_checksum(
    path: str | Path,
    checksum: str,
    algorithm: str = "sha256",
) -> None:
    """Raise :class:`DatasetIntegrityError` unless ``path`` hashes to ``checksum``.

    Hash algorithm name follows :mod:`hashlib`.  Comparison is hex-string,
    case-insensitive.
    """
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    actual = h.hexdigest().lower()
    expected = checksum.lower()
    if actual != expected:
        raise DatasetIntegrityError(
            f"{path}: {algorithm} checksum mismatch (expected {expected}, "
            f"got {actual}). The download may be corrupt or tampered with."
        )


# ── Archive extraction ───────────────────────────────────────────────────────

_ARCHIVE_EXTENSIONS = (
    ".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".tar.xz", ".txz",
)


def is_archive(path: str | Path) -> bool:
    name = str(path).lower()
    return any(name.endswith(ext) for ext in _ARCHIVE_EXTENSIONS)


def _is_within(target_dir: Path, member_path: Path) -> bool:
    """``True`` iff ``member_path`` resolves to somewhere inside ``target_dir``."""
    try:
        target = target_dir.resolve()
        member = (target_dir / member_path).resolve()
    except OSError:
        return False
    try:
        member.relative_to(target)
        return True
    except ValueError:
        return False


def safe_extract_zip(archive: Path, dst: Path) -> None:
    """Extract a .zip archive while rejecting path-traversal entries.

    Refuses any member whose resolved path lies outside ``dst``.
    """
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or not _is_within(dst, member_path):
                raise DatasetExtractionError(
                    f"Refusing to extract '{member.filename}' from "
                    f"{archive}: would escape destination {dst}."
                )
        zf.extractall(dst)


def safe_extract_tar(archive: Path, dst: Path) -> None:
    """Extract a .tar / .tar.gz / .tar.bz2 / .tar.xz archive safely."""
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or not _is_within(dst, member_path):
                raise DatasetExtractionError(
                    f"Refusing to extract '{member.name}' from "
                    f"{archive}: would escape destination {dst}."
                )
            # Disallow special members that could hide path tricks.
            if (member.issym() or member.islnk()) and member.linkname:
                link_target = (dst / member_path).parent / member.linkname
                if not _is_within(dst, link_target.relative_to(dst) if link_target.is_relative_to(dst) else Path("..")):
                    raise DatasetExtractionError(
                        f"Refusing to extract symlink '{member.name}' "
                        f"pointing to '{member.linkname}' from {archive}."
                    )
        # Python 3.12+ supports `filter='data'` which adds extra hardening.
        try:
            tf.extractall(dst, filter="data")
        except TypeError:  # pragma: no cover  (older Python)
            tf.extractall(dst)


def extract_archive(
    archive: str | Path,
    dst: str | Path,
    remove_finished: bool = False,
) -> Path:
    """Extract a known archive type to ``dst``."""
    archive = Path(archive).expanduser()
    target = Path(dst).expanduser()
    target.mkdir(parents=True, exist_ok=True)

    name = str(archive).lower()
    if name.endswith(".zip"):
        safe_extract_zip(archive, target)
    elif any(name.endswith(ext) for ext in (".tar", ".tar.gz", ".tgz",
                                              ".tar.bz2", ".tbz",
                                              ".tar.xz", ".txz")):
        safe_extract_tar(archive, target)
    else:
        raise DatasetExtractionError(
            f"Unsupported archive type: {archive}. Supported extensions: "
            f"{_ARCHIVE_EXTENSIONS}"
        )

    if remove_finished:
        try:
            archive.unlink()
        except OSError:
            pass
    return target


# ── Convenience: download-only-if-asked ──────────────────────────────────────


def maybe_download(
    url: str,
    dst: str | Path,
    checksum: Optional[str] = None,
    download: bool = False,
    algorithm: str = "sha256",
) -> Path:
    """Download ``url`` to ``dst`` only when ``download=True``.

    If the file already exists, return its path without touching the
    network.  If ``download=False`` and the file is missing, raise
    :class:`FileNotFoundError` with a clear hint — never download
    silently.
    """
    target = Path(dst).expanduser()
    if target.exists():
        if checksum is not None:
            verify_checksum(target, checksum, algorithm=algorithm)
        return target
    if not download:
        raise FileNotFoundError(
            f"{target} is missing. Pass download=True to fetch from {url}, "
            f"or place the file at this path manually."
        )
    return download_url(url, target, checksum=checksum, algorithm=algorithm)
