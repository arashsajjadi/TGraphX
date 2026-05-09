"""Tests for the distributed smoke example."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def test_single_process_smoke(tmp_path):
    """World-size=1 path: always runs, no subprocess launch."""
    # Do NOT pass RANK/WORLD_SIZE in env — those trigger the torchrun-worker branch
    # which has no --json output.  Use a clean env without distributed env vars.
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in ("RANK", "WORLD_SIZE", "LOCAL_RANK",
                              "TORCHELASTIC_RESTART_COUNT", "DIST_SMOKE_WORKER")}
    result = subprocess.run(
        [sys.executable, "examples/distributed_smoke.py",
         "--world-size", "1", "--output-dir", str(tmp_path), "--json"],
        capture_output=True, text=True, timeout=30, env=clean_env,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    data = json.loads(result.stdout)
    assert data["passed"] is True
    summary = tmp_path / "distributed_run_summary.json"
    assert summary.exists()
    payload = json.loads(summary.read_text())
    assert payload.get("step_completed") is True
    assert payload.get("loss_finite") is True


def test_two_process_gloo_smoke(tmp_path):
    """World-size=2 subprocess-pair: validates multi-process DDP on CPU with gloo."""
    result = subprocess.run(
        [sys.executable, "examples/distributed_smoke.py",
         "--world-size", "2", "--subprocess-pair",
         "--output-dir", str(tmp_path), "--json", "--timeout", "60"],
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    data = json.loads(result.stdout)
    assert data["passed"] is True
    assert data["world_size"] == 2
    assert data["backend"] == "gloo"
    # Rank-zero artifact written.
    summary = tmp_path / "distributed_run_summary.json"
    assert summary.exists()
    payload = json.loads(summary.read_text())
    assert payload.get("step_completed") is True
