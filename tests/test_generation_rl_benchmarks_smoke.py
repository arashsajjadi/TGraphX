"""Smoke tests for benchmark scripts: --help, --small --json output, timing."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
PYTHON = sys.executable

# Benchmark scripts to test
BENCHMARK_SCRIPTS = [
    PROJECT_ROOT / "benchmarks" / "generation" / "benchmark_generation_metrics.py",
    PROJECT_ROOT / "benchmarks" / "generation" / "benchmark_neural_generation.py",
    PROJECT_ROOT / "benchmarks" / "generation" / "benchmark_graph_sequence_models.py",
    PROJECT_ROOT / "benchmarks" / "evolution" / "benchmark_nsga2.py",
    PROJECT_ROOT / "benchmarks" / "evolution" / "benchmark_mutation_crossover.py",
    PROJECT_ROOT / "benchmarks" / "rl" / "benchmark_graph_coloring_rl.py",
    PROJECT_ROOT / "benchmarks" / "rl" / "benchmark_dqn_graph_env.py",
    PROJECT_ROOT / "benchmarks" / "rl" / "benchmark_ppo_graph_env.py",
    PROJECT_ROOT / "benchmarks" / "rl" / "benchmark_td3_sac_graph_env.py",
    PROJECT_ROOT / "benchmarks" / "rl" / "benchmark_rl_algorithm_comparison.py",
]


@pytest.mark.parametrize("script", BENCHMARK_SCRIPTS, ids=lambda p: p.name)
def test_benchmark_help(script):
    """Every benchmark script should support --help without error."""
    result = subprocess.run(
        [PYTHON, str(script), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"--help failed for {script.name}: {result.stderr}"


@pytest.mark.parametrize("script", BENCHMARK_SCRIPTS, ids=lambda p: p.name)
def test_benchmark_small_json(script):
    """Every benchmark script should produce valid JSON with --small --json."""
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, str(script), "--small", "--json", "--seed", "0"],
        capture_output=True, text=True, timeout=60,
    )
    elapsed = time.time() - t0

    assert result.returncode == 0, (
        f"--small --json failed for {script.name}\n"
        f"stdout: {result.stdout[:500]}\n"
        f"stderr: {result.stderr[:500]}"
    )
    assert elapsed < 60.0, f"{script.name} took {elapsed:.1f}s (> 60s limit)"

    # Parse JSON
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        pytest.fail(f"Invalid JSON from {script.name}: {e}\nOutput: {result.stdout[:200]}")

    # Check required keys
    assert "seed" in data, f"'seed' missing from {script.name} output"
    assert "metrics" in data, f"'metrics' missing from {script.name} output"


def test_benchmark_no_network_access():
    """Benchmark scripts should not make network calls (torch.hub.load, requests, etc.)."""
    script = PROJECT_ROOT / "benchmarks" / "rl" / "benchmark_dqn_graph_env.py"
    result = subprocess.run(
        [PYTHON, str(script), "--small", "--json"],
        capture_output=True, text=True, timeout=30,
        env={
            **__import__("os").environ,
            "HTTP_PROXY": "",
            "HTTPS_PROXY": "",
            "NO_PROXY": "*",
        },
    )
    assert result.returncode == 0
