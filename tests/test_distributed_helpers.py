"""Tests for the v0.5.0 distributed-helpers additions."""
from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path

import torch

from tgraphx.distributed import (
    detect_distributed_environment,
    rank_seed,
    distributed_device,
    shard_indices,
    write_distributed_run_summary,
    is_distributed,
)


def test_detect_distributed_no_env(monkeypatch):
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "SLURM_PROCID", "SLURM_NTASKS"):
        monkeypatch.delenv(k, raising=False)
    env = detect_distributed_environment()
    assert env["initialized"] is False
    assert env["rank"] == 0
    assert env["world_size"] == 1
    assert env["launcher"] == "none"


def test_detect_distributed_torchrun_env(monkeypatch):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")
    env = detect_distributed_environment()
    # When torch.distributed is not actually initialised, we still report env.
    assert env["launcher"] == "torchrun"


def test_rank_seed_deterministic():
    a = rank_seed(42, rank=0)
    b = rank_seed(42, rank=0)
    c = rank_seed(42, rank=1)
    assert a == b
    assert a != c


def test_shard_indices_balanced():
    idx = torch.arange(10)
    parts = [shard_indices(idx, rank=r, world_size=4) for r in range(4)]
    # Reconstruct: every index appears exactly once.
    cat = torch.cat(parts).sort().values
    assert torch.equal(cat, idx)
    # Sizes within 1.
    sizes = [int(p.numel()) for p in parts]
    assert max(sizes) - min(sizes) <= 1


def test_shard_indices_drop_last():
    idx = torch.arange(10)
    parts = [shard_indices(idx, rank=r, world_size=3, drop_last=True) for r in range(3)]
    sizes = {p.numel() for p in parts}
    assert len(sizes) == 1


def test_distributed_device_no_cuda(monkeypatch):
    # Force the no-CUDA path.
    if torch.cuda.is_available():
        return  # skip on CUDA boxes
    dev = distributed_device(0)
    assert dev.type == "cpu"


def test_write_distributed_run_summary_rank0(monkeypatch, tmp_path):
    monkeypatch.delenv("RANK", raising=False)
    out = tmp_path / "distributed_run_summary.json"
    write_distributed_run_summary(str(out), base_seed=123, model="toy")
    payload = json.loads(out.read_text())
    assert payload["world_size"] == 1
    assert payload["base_seed"] == 123
    assert payload["model"] == "toy"
