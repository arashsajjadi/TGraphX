"""Benchmark --small mode smoke (v0.2.9)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
BENCH = ROOT / "benchmarks"


def _run(name: str, *args, output: Path | None = None):
    cmd = [sys.executable, str(BENCH / name), "--small", *args]
    if output:
        cmd += ["--output", str(output)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (
        f"{name} failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return result


class TestBenchmarkSmoke:
    def test_dataset_loading(self, tmp_path):
        out = tmp_path / "ds.json"
        _run("benchmark_dataset_loading.py", output=out)
        data = json.loads(out.read_text())
        assert data["small"] is True
        assert isinstance(data["results"], list) and len(data["results"]) >= 4

    def test_transforms(self, tmp_path):
        out = tmp_path / "tr.json"
        _run("benchmark_transforms.py", output=out)
        data = json.loads(out.read_text())
        assert "results" in data

    def test_metrics(self, tmp_path):
        out = tmp_path / "me.json"
        _run("benchmark_metrics.py", output=out)
        data = json.loads(out.read_text())
        assert "results" in data

    def test_training_synthetic(self, tmp_path):
        out = tmp_path / "tr2.json"
        _run("benchmark_training_synthetic.py", output=out)
        data = json.loads(out.read_text())
        assert "results" in data

    def test_tensor_vs_flatten(self, tmp_path):
        out = tmp_path / "tvf.json"
        _run("benchmark_tensor_vs_flatten.py", output=out)
        data = json.loads(out.read_text())
        assert "tensor" in data and "flatten" in data

    def test_make_benchmark_report(self, tmp_path):
        # Generate two benchmark JSONs first.
        ds_json = tmp_path / "ds.json"
        _run("benchmark_dataset_loading.py", output=ds_json)
        report = tmp_path / "report.md"
        cmd = [
            sys.executable, str(BENCH / "make_benchmark_report.py"),
            str(ds_json), "--output", str(report),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        text = report.read_text()
        assert "TGraphX benchmark report" in text
