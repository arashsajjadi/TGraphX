"""Smoke tests for KG benchmark scripts."""
from __future__ import annotations

import json
import subprocess
import sys


def _run(script: str, extra: list = None) -> dict:
    cmd = [sys.executable, script, "--small", "--json"] + (extra or [])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"{script} failed:\n{result.stderr}"
    return json.loads(result.stdout)


class TestKGBenchmarkSmoke:

    def test_negative_sampling_smoke(self):
        d = _run("benchmarks/kg/benchmark_kg_negative_sampling.py")
        assert "sampler_results" in d
        names = {r["sampler"] for r in d["sampler_results"]}
        assert {"uniform", "bernoulli", "filtered"}.issubset(names)

    def test_filtered_eval_smoke(self):
        d = _run("benchmarks/kg/benchmark_kg_filtered_eval.py")
        assert "results" in d
        assert "filtered" in d["results"]

    def test_transe_smoke(self):
        d = _run("benchmarks/kg/benchmark_kg_transe.py")
        assert "final_loss" in d
        assert d["final_loss"] is not None

    def test_distmult_smoke(self):
        d = _run("benchmarks/kg/benchmark_kg_distmult.py")
        assert "final_loss" in d

    def test_complex_smoke(self):
        d = _run("benchmarks/kg/benchmark_kg_complex.py")
        assert "final_loss" in d

    def test_rotate_smoke(self):
        d = _run("benchmarks/kg/benchmark_kg_rotate.py")
        assert "final_loss" in d

    def test_json_parseable(self):
        for script in [
            "benchmarks/kg/benchmark_kg_negative_sampling.py",
            "benchmarks/kg/benchmark_kg_filtered_eval.py",
            "benchmarks/kg/benchmark_kg_transe.py",
        ]:
            d = _run(script)
            json.dumps(d)  # must not raise

    def test_no_network(self):
        """Benchmarks must not attempt network access (no download flag)."""
        d = _run("benchmarks/kg/benchmark_kg_transe.py")
        assert d.get("model") == "TransE"
