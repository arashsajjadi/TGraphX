"""End-to-end FAST_SMOKE integration test."""
import subprocess, sys
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]


def test_fast_smoke_pipeline_runs():
    env = {
        "PYTHONPATH": str(PROJECT / "src"),
        "PATH": __import__("os").environ.get("PATH", ""),
    }
    result = subprocess.run(
        [sys.executable, "-m", "od_graph_fusion.cli",
         "--config", str(PROJECT / "configs" / "fast_smoke.yaml")],
        cwd=str(PROJECT), env=env,
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, (
        f"FAST_SMOKE failed (exit={result.returncode}):\n"
        f"stdout: {result.stdout[-1500:]}\n"
        f"stderr: {result.stderr[-1500:]}"
    )
    # Check report file exists
    run_dir = PROJECT / "runs" / "fast_smoke"
    assert (run_dir / "report.md").exists()
    assert (run_dir / "method_results.json").exists()
    assert (run_dir / "env_report.json").exists()
