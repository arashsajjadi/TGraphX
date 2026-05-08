"""Cache + atomic IO tests (v0.2.9). No network."""
from __future__ import annotations

import json
import os

import torch

from tgraphx.datasets.cache import (
    atomic_save_json,
    atomic_save_torch,
    atomic_write_bytes,
    cache_summary,
    clear_cache,
    get_default_cache_root,
    load_json,
    resolve_dataset_root,
)


class TestRootResolution:
    def test_explicit_root(self, tmp_path):
        p = resolve_dataset_root(tmp_path, "synthetic:foo")
        assert str(p).startswith(str(tmp_path))
        assert "synthetic" in p.name and "foo" in p.name

    def test_env_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TGRAPHX_DATA", str(tmp_path / "envroot"))
        root = get_default_cache_root()
        assert str(root).endswith("envroot")
        # Resolve is consistent.
        p = resolve_dataset_root(None, "x:y")
        assert str(p).startswith(str(tmp_path / "envroot"))

    def test_default_no_env(self, monkeypatch):
        monkeypatch.delenv("TGRAPHX_DATA", raising=False)
        root = get_default_cache_root()
        assert ".cache/tgraphx" in str(root) or "AppData" in str(root) or "Library" in str(root)

    def test_slugify_safe(self, tmp_path):
        p = resolve_dataset_root(tmp_path, "synthetic:patch_graph")
        # Colon must not appear as a directory separator.
        assert ":" not in p.name
        # Forward slashes in dataset names are slugified.
        p2 = resolve_dataset_root(tmp_path, "pyg:planetoid/cora")
        assert "/cora" not in p2.name


class TestAtomicIO:
    def test_atomic_write_bytes(self, tmp_path):
        target = tmp_path / "sub" / "file.bin"
        atomic_write_bytes(target, b"hello")
        assert target.read_bytes() == b"hello"

    def test_atomic_save_torch_roundtrip(self, tmp_path):
        target = tmp_path / "obj.pt"
        atomic_save_torch({"x": torch.tensor([1.0, 2.0])}, target)
        try:
            loaded = torch.load(target, weights_only=False)
        except TypeError:  # pragma: no cover
            loaded = torch.load(target)
        assert torch.equal(loaded["x"], torch.tensor([1.0, 2.0]))

    def test_atomic_save_json_roundtrip(self, tmp_path):
        target = tmp_path / "x.json"
        atomic_save_json({"a": 1, "b": [1, 2, 3]}, target)
        loaded = load_json(target)
        assert loaded == {"a": 1, "b": [1, 2, 3]}

    def test_atomic_failure_cleans_tmp(self, tmp_path, monkeypatch):
        target = tmp_path / "x.json"

        # Force os.replace to fail; ensure tmp file is cleaned up.
        original = os.replace

        def boom(*a, **kw):
            raise OSError("simulated failure")

        monkeypatch.setattr(os, "replace", boom)
        import pytest
        with pytest.raises(OSError):
            atomic_write_bytes(target, b"x")
        # No leftover .tmp files.
        leftovers = list(tmp_path.glob("x.json.*tmp"))
        assert leftovers == []
        monkeypatch.setattr(os, "replace", original)


class TestCacheManagement:
    def test_summary_empty(self, tmp_path):
        info = cache_summary(tmp_path)
        assert info["root"] == str(tmp_path)
        assert info["datasets"] == []

    def test_summary_with_files(self, tmp_path):
        d = tmp_path / "dataset1"
        d.mkdir()
        (d / "a.bin").write_bytes(b"x" * 100)
        info = cache_summary(tmp_path)
        assert any(item["name"] == "dataset1" for item in info["datasets"])

    def test_clear_cache_dry_run(self, tmp_path):
        (tmp_path / "ds_a").mkdir()
        (tmp_path / "ds_b").mkdir()
        listed = clear_cache(tmp_path, dry_run=True)
        assert len(listed) == 2
        # Files still exist.
        assert (tmp_path / "ds_a").exists()

    def test_clear_cache_actual(self, tmp_path):
        (tmp_path / "ds_a").mkdir()
        clear_cache(tmp_path, dry_run=False)
        assert not (tmp_path / "ds_a").exists()
