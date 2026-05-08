"""Mocked download / safe-extraction tests (v0.2.9). No network."""
from __future__ import annotations

import hashlib
import os
import tarfile
import zipfile
from pathlib import Path

import pytest

from tgraphx.datasets import (
    DatasetExtractionError,
    DatasetIntegrityError,
    download_url,
    extract_archive,
    maybe_download,
    verify_checksum,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TestVerifyChecksum:
    def test_pass(self, tmp_path):
        f = tmp_path / "a.bin"
        data = b"hello"
        f.write_bytes(data)
        verify_checksum(f, _sha256_bytes(data))

    def test_fail(self, tmp_path):
        f = tmp_path / "b.bin"
        f.write_bytes(b"hello")
        with pytest.raises(DatasetIntegrityError):
            verify_checksum(f, "0" * 64)


class TestMaybeDownload:
    def test_existing_file_no_network(self, tmp_path):
        target = tmp_path / "x.bin"
        target.write_bytes(b"abc")
        # download=False but file exists → returns immediately.
        out = maybe_download("http://example.invalid/x", target, download=False)
        assert out == target

    def test_missing_with_download_false_raises(self, tmp_path):
        target = tmp_path / "missing.bin"
        with pytest.raises(FileNotFoundError):
            maybe_download("http://example.invalid/x", target, download=False)


class TestDownloadUrlMocked:
    def test_atomic_via_monkeypatched_urlopen(self, tmp_path, monkeypatch):
        from io import BytesIO
        from urllib.request import urlopen as real_urlopen  # noqa: F401

        class _FakeResp:
            def __init__(self, payload: bytes) -> None:
                self._buf = BytesIO(payload)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                self._buf.close()

            def read(self, n=-1):
                return self._buf.read(n)

        payload = b"controlled bytes"

        def fake_urlopen(url, timeout=60.0):
            return _FakeResp(payload)

        # Patch the urlopen used inside download.py.
        import tgraphx.datasets.download as dl
        monkeypatch.setattr(dl, "urlopen", fake_urlopen)

        target = tmp_path / "downloaded.bin"
        out = download_url("http://example.invalid/x", target,
                           checksum=_sha256_bytes(payload))
        assert out.read_bytes() == payload

    def test_download_checksum_failure_cleans_up(self, tmp_path, monkeypatch):
        from io import BytesIO

        class _FakeResp:
            def __init__(self, payload: bytes) -> None:
                self._buf = BytesIO(payload)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                self._buf.close()

            def read(self, n=-1):
                return self._buf.read(n)

        def fake_urlopen(url, timeout=60.0):
            return _FakeResp(b"bytes that won't match")

        import tgraphx.datasets.download as dl
        monkeypatch.setattr(dl, "urlopen", fake_urlopen)

        target = tmp_path / "out.bin"
        with pytest.raises(DatasetIntegrityError):
            download_url("http://example.invalid/x", target,
                         checksum="0" * 64)
        # No partial file left behind.
        assert not target.exists()
        leftovers = list(tmp_path.glob("out.bin.*"))
        assert leftovers == []


class TestSafeExtract:
    def test_zip_normal(self, tmp_path):
        archive = tmp_path / "ok.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("a/b.txt", "hello")
        dst = tmp_path / "out"
        extract_archive(archive, dst)
        assert (dst / "a" / "b.txt").read_text() == "hello"

    def test_zip_path_traversal_blocked(self, tmp_path):
        archive = tmp_path / "evil.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("../escape.txt", "boom")
        dst = tmp_path / "out"
        with pytest.raises(DatasetExtractionError):
            extract_archive(archive, dst)
        assert not (tmp_path / "escape.txt").exists()

    def test_tar_normal(self, tmp_path):
        archive = tmp_path / "ok.tar"
        with tarfile.open(archive, "w") as tf:
            f = tmp_path / "x.txt"
            f.write_bytes(b"abc")
            tf.add(f, arcname="x.txt")
        dst = tmp_path / "out"
        extract_archive(archive, dst)
        assert (dst / "x.txt").read_bytes() == b"abc"

    def test_tar_path_traversal_blocked(self, tmp_path):
        archive = tmp_path / "evil.tar"
        with tarfile.open(archive, "w") as tf:
            payload = tmp_path / "_payload.txt"
            payload.write_text("escape")
            tf.add(payload, arcname="../escape.txt")
        dst = tmp_path / "out"
        with pytest.raises(DatasetExtractionError):
            extract_archive(archive, dst)

    def test_unknown_archive_type_rejected(self, tmp_path):
        weird = tmp_path / "data.bin"
        weird.write_bytes(b"x")
        with pytest.raises(DatasetExtractionError):
            extract_archive(weird, tmp_path / "out")
