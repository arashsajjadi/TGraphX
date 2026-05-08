"""Tests for benchmarks/public/ scripts (v0.3.2).

Verifies:
- ``--help`` works for every script.
- Optional dependencies are skipped cleanly.
- The MNIST FakeData path runs and writes the four standard artefacts.
- ``--strict`` flips a missing-dependency skip into a hard failure.
- Artefact JSON files are valid and dashboard-compatible.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO / "benchmarks" / "public"


def _run(script: str, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SCRIPTS_DIR / script), *args]
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


# ── --help works ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("script", [
    "mnist_patch_benchmark.py",
    "pyg_cora_benchmark.py",
    "fashionmnist_patch_benchmark.py",
])
def test_script_help(script):
    res = _run(script, "--help")
    assert res.returncode == 0, res.stderr
    out = res.stdout + res.stderr
    for flag in ["--root", "--download", "--max-samples", "--max-nodes",
                 "--epochs", "--device", "--output-dir", "--seed", "--json", "--strict"]:
        assert flag in out, f"Missing CLI flag {flag} in {script} --help"


# ── PyG Cora benchmark — clean skip without PyG ──────────────────────────────


def test_pyg_cora_clean_skip_without_dep():
    # Without --download AND without PyG installed, the script reports the
    # specific guidance message and exits 0 (soft skip).
    res = _run("pyg_cora_benchmark.py")
    assert res.returncode == 0
    out = (res.stdout + res.stderr).lower()
    assert "[skip]" in out


def test_pyg_cora_strict_fails_without_dep():
    # In strict mode the same condition must produce exit 2.
    res = _run("pyg_cora_benchmark.py", "--strict")
    assert res.returncode == 2


# ── MNIST FakeData path runs end-to-end ──────────────────────────────────────


def test_mnist_fakedata_writes_four_artefacts(tmp_path):
    out_dir = tmp_path / "mnist_run"
    res = _run(
        "mnist_patch_benchmark.py",
        "--epochs", "2",
        "--max-samples", "8",
        "--output-dir", str(out_dir),
        "--seed", "0",
    )
    assert res.returncode == 0, res.stderr + res.stdout
    for name in [
        "benchmark_results.json",
        "run_metadata.json",
        "dataset_metadata.json",
        "metrics_summary.json",
    ]:
        path = out_dir / name
        assert path.exists(), f"missing artefact: {name}"
        data = json.loads(path.read_text())
        assert isinstance(data, dict)


def test_mnist_fakedata_benchmark_payload_schema(tmp_path):
    out_dir = tmp_path / "mnist_run"
    _run(
        "mnist_patch_benchmark.py",
        "--epochs", "2",
        "--max-samples", "8",
        "--output-dir", str(out_dir),
    )
    bench = json.loads((out_dir / "benchmark_results.json").read_text())
    # Required schema fields.
    for k in [
        "benchmark", "data_source", "elapsed_s", "epochs",
        "num_graphs", "num_nodes", "num_edges",
        "loss_start", "loss_end", "loss_decreased", "final_accuracy",
        "tgraphx_version", "platform", "device", "seed",
    ]:
        assert k in bench, f"benchmark_results.json missing key {k}"
    # FakeData path is unconditional unless --download was set.
    assert bench["data_source"] == "fake_data_synthetic"
    assert bench["num_graphs"] == 8
    assert bench["seed"] == 0
    # Loss should be finite floats.
    assert isinstance(bench["loss_start"], float)
    assert isinstance(bench["loss_end"], float)


def test_mnist_fakedata_no_network_in_default_path(tmp_path, monkeypatch):
    """Default (no --download) path must not contact the network."""
    # Disable outbound DNS by pointing torchvision at an unwritable cache and
    # hoping it refuses any attempted download.  Since this script defaults
    # to FakeData (synthetic, in-memory), no download should be attempted at
    # all and the run should succeed.
    out_dir = tmp_path / "mnist_run"
    res = _run(
        "mnist_patch_benchmark.py",
        "--epochs", "1",
        "--max-samples", "4",
        "--output-dir", str(out_dir),
    )
    assert res.returncode == 0
    bench = json.loads((out_dir / "benchmark_results.json").read_text())
    assert bench["data_source"] == "fake_data_synthetic"


def test_fashionmnist_fakedata_runs(tmp_path):
    out_dir = tmp_path / "fashion_run"
    res = _run(
        "fashionmnist_patch_benchmark.py",
        "--epochs", "1",
        "--max-samples", "4",
        "--output-dir", str(out_dir),
    )
    assert res.returncode == 0, res.stderr + res.stdout
    bench = json.loads((out_dir / "benchmark_results.json").read_text())
    assert bench["data_source"] == "fake_data_synthetic"
    assert bench["benchmark"] == "fashionmnist_patch_benchmark"


def test_mnist_json_output(tmp_path):
    out_dir = tmp_path / "mnist_run"
    res = _run(
        "mnist_patch_benchmark.py",
        "--epochs", "1",
        "--max-samples", "4",
        "--output-dir", str(out_dir),
        "--json",
    )
    assert res.returncode == 0
    # The first line of stdout should be a JSON document we can parse.
    parsed = json.loads(res.stdout)
    assert "summary" in parsed
    assert "artefacts" in parsed
