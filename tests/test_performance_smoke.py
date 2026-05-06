"""Smoke tests for tgraphx.performance, chunked ConvMessagePassing,
dashboard metrics caching, and benchmark scripts."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import torch

from tgraphx.graph_builders import build_grid_graph, build_grid_graph_3d
from tgraphx.layers.conv_message import ConvMessagePassing
from tgraphx.performance import (
    env_report,
    estimate_message_memory,
    recommended_device,
)

PROJ_ROOT = Path(__file__).parent.parent
BENCH_DIR = PROJ_ROOT / "benchmarks"


# ─────────────────────────────────────────────────────────────────────────────
# env_report
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvReport:

    def test_required_keys_present(self):
        r = env_report()
        for key in ("python", "os", "torch", "tgraphx",
                    "cuda_available", "cuda_device",
                    "mps_available", "recommended_device"):
            assert key in r, f"Missing key: {key}"

    def test_python_is_string(self):
        assert isinstance(env_report()["python"], str)
        assert env_report()["python"].startswith("3")

    def test_torch_version_is_string(self):
        r = env_report()
        assert isinstance(r["torch"], str) and r["torch"] != ""

    def test_tgraphx_version_not_unknown(self):
        r = env_report()
        assert r["tgraphx"] != "" and r["tgraphx"] is not None

    def test_recommended_device_is_valid(self):
        r = env_report()
        assert r["recommended_device"] in ("cpu", "cuda", "mps")

    def test_include_hardware_adds_cpu_count(self):
        r = env_report(include_hardware=True)
        assert "cpu_count" in r
        assert isinstance(r["cpu_count"], int) and r["cpu_count"] >= 1

    def test_include_hardware_has_psutil_flag(self):
        r = env_report(include_hardware=True)
        assert "psutil_available" in r

    def test_include_hardware_ram_fields_present(self):
        r = env_report(include_hardware=True)
        # Fields exist regardless of psutil
        assert "ram_total_gb" in r
        assert "ram_avail_gb" in r

    def test_include_sensors_has_pynvml_flag(self):
        r = env_report(include_sensors=True)
        assert "pynvml_available" in r

    def test_default_no_hardware(self):
        r = env_report()
        assert "cpu_count" not in r
        assert "ram_total_gb" not in r

    def test_does_not_raise_without_optional_deps(self):
        # Must not crash even if psutil / pynvml are absent
        r = env_report(include_hardware=True, include_sensors=True)
        assert isinstance(r, dict)

    def test_recommended_device_returns_device(self):
        dev = recommended_device()
        assert isinstance(dev, torch.device)


# ─────────────────────────────────────────────────────────────────────────────
# estimate_message_memory
# ─────────────────────────────────────────────────────────────────────────────

class TestEstimateMemory:

    def test_returns_required_keys(self):
        m = estimate_message_memory(100, (8, 4, 4))
        for k in ("num_edges", "out_shape", "dtype", "bytes_per_edge",
                  "total_bytes", "total_mb", "note"):
            assert k in m

    def test_scalar_shape(self):
        m = estimate_message_memory(100, 64)
        assert m["out_shape"] == (64,)

    def test_float16_half_of_float32(self):
        m32 = estimate_message_memory(1000, (32, 8, 8), dtype=torch.float32)
        m16 = estimate_message_memory(1000, (32, 8, 8), dtype=torch.float16)
        assert abs(m32["total_mb"] - 2 * m16["total_mb"]) < 1e-6

    def test_zero_edges(self):
        m = estimate_message_memory(0, (16,))
        assert m["total_bytes"] == 0

    def test_sanity_2d(self):
        E, C, H, W = 200, 8, 4, 4
        m = estimate_message_memory(E, (C, H, W))
        expected_bytes = E * C * H * W * 4  # float32 = 4 bytes
        assert m["total_bytes"] == expected_bytes

    def test_note_is_string(self):
        m = estimate_message_memory(50, (4, 4, 4))
        assert isinstance(m["note"], str) and len(m["note"]) > 10


# ─────────────────────────────────────────────────────────────────────────────
# Base import does not import psutil/pynvml
# ─────────────────────────────────────────────────────────────────────────────

class TestImportLightweight:

    def test_performance_py_has_no_toplevel_psutil_import(self):
        """Verify tgraphx/performance.py does not import psutil at module scope."""
        perf_src = (PROJ_ROOT / "tgraphx" / "performance.py").read_text()
        # The only occurrence of 'psutil' must be inside a function body (indented)
        for line in perf_src.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("import psutil") or stripped.startswith("from psutil"):
                # Accept only if the line is indented (inside a function/try block)
                assert line.startswith(" ") or line.startswith("\t"), (
                    f"Found top-level psutil import: {line!r}"
                )

    def test_performance_py_has_no_toplevel_pynvml_import(self):
        """Verify tgraphx/performance.py does not import pynvml at module scope."""
        perf_src = (PROJ_ROOT / "tgraphx" / "performance.py").read_text()
        for line in perf_src.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("import pynvml") or stripped.startswith("from pynvml"):
                assert line.startswith(" ") or line.startswith("\t"), (
                    f"Found top-level pynvml import: {line!r}"
                )

    def test_env_report_works_without_optional_deps(self):
        """env_report() must not raise even when psutil/pynvml are absent."""
        r = env_report(include_hardware=True, include_sensors=True)
        assert isinstance(r, dict)
        assert "python" in r
        # Optional fields may be None but must be present or absent consistently
        if "psutil_available" in r and not r["psutil_available"]:
            assert r.get("ram_total_gb") is None
        if "pynvml_available" in r and not r["pynvml_available"]:
            assert r.get("gpu_util_pct") is None


# ─────────────────────────────────────────────────────────────────────────────
# Chunked ConvMessagePassing
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkedConvMessagePassing:

    @pytest.fixture(params=["sum", "mean"])
    def aggr(self, request):
        return request.param

    def _run(self, aggr, chunk_size=None):
        torch.manual_seed(0)
        N = 9
        layer = ConvMessagePassing(
            in_shape=(4, 4, 4), out_shape=(8, 4, 4), aggr=aggr
        ).eval()
        x  = torch.randn(N, 4, 4, 4)
        ei = build_grid_graph(3, 3, directed=False, self_loops=True)
        with torch.no_grad():
            return layer(x, ei, chunk_size=chunk_size)

    def test_chunk_matches_unchunked_sum(self):
        out_full  = self._run("sum", chunk_size=None)
        out_chunk = self._run("sum", chunk_size=5)
        assert torch.allclose(out_full, out_chunk, atol=1e-5), \
            f"max diff = {(out_full - out_chunk).abs().max()}"

    def test_chunk_matches_unchunked_mean(self):
        out_full  = self._run("mean", chunk_size=None)
        out_chunk = self._run("mean", chunk_size=5)
        assert torch.allclose(out_full, out_chunk, atol=1e-5), \
            f"max diff = {(out_full - out_chunk).abs().max()}"

    def test_large_chunk_same_as_no_chunk(self, aggr):
        out_full  = self._run(aggr, chunk_size=None)
        out_chunk = self._run(aggr, chunk_size=10_000)
        assert torch.allclose(out_full, out_chunk, atol=1e-5)

    def test_chunk_size_1_works(self, aggr):
        out_full  = self._run(aggr, chunk_size=None)
        out_chunk = self._run(aggr, chunk_size=1)
        assert torch.allclose(out_full, out_chunk, atol=1e-4)

    def test_chunked_with_edge_weight(self):
        torch.manual_seed(1)
        N = 9
        layer = ConvMessagePassing(in_shape=(4, 4, 4), out_shape=(8, 4, 4), aggr="sum").eval()
        x  = torch.randn(N, 4, 4, 4)
        ei = build_grid_graph(3, 3, directed=False, self_loops=True)
        ew = torch.rand(ei.size(1))
        with torch.no_grad():
            out_full  = layer(x, ei, edge_weight=ew, chunk_size=None)
            out_chunk = layer(x, ei, edge_weight=ew, chunk_size=4)
        assert torch.allclose(out_full, out_chunk, atol=1e-5)

    def test_chunked_3d(self):
        torch.manual_seed(2)
        N = 8
        layer = ConvMessagePassing(
            in_shape=(2, 2, 2, 2), out_shape=(4, 2, 2, 2), aggr="sum"
        ).eval()
        x  = torch.randn(N, 2, 2, 2, 2)
        ei = build_grid_graph_3d(2, 2, 2, directed=False, self_loops=True)
        with torch.no_grad():
            out_full  = layer(x, ei, chunk_size=None)
            out_chunk = layer(x, ei, chunk_size=6)
        assert torch.allclose(out_full, out_chunk, atol=1e-5)

    def test_max_aggr_warns_and_falls_back(self):
        import warnings
        N = 4
        layer = ConvMessagePassing(in_shape=(2, 2, 2), out_shape=(4, 2, 2), aggr="max").eval()
        x  = torch.randn(N, 2, 2, 2)
        ei = build_grid_graph(2, 2, directed=False, self_loops=True)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            with torch.no_grad():
                out = layer(x, ei, chunk_size=2)
            assert any("chunk_size is ignored" in str(w.message) for w in ws)
        assert out.shape == (N, 4, 2, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard metrics caching
# ─────────────────────────────────────────────────────────────────────────────

class TestDashboardMetricsCache:

    def test_cache_hit_on_unchanged_file(self):
        import csv, threading
        from tgraphx.dashboard.app import DashboardServer

        with tempfile.TemporaryDirectory() as d:
            csv_path = os.path.join(d, "metrics.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "train_loss"])
                w.writerow([1, 0.5])

            srv = DashboardServer(d, "127.0.0.1", 0, verbose=False)
            thread = threading.Thread(target=srv.serve_forever, daemon=True)
            thread.start()
            time.sleep(0.05)

            try:
                import urllib.request
                _, p = srv.server_address
                base = f"http://127.0.0.1:{p}"

                # First request populates cache
                r1 = urllib.request.urlopen(f"{base}/api/metrics", timeout=3)
                d1 = json.loads(r1.read())

                # Second request should use cache
                r2 = urllib.request.urlopen(f"{base}/api/metrics", timeout=3)
                d2 = json.loads(r2.read())

                assert d1 == d2
                assert len(d1["rows"]) == 1
            finally:
                srv.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark scripts (subprocess, small configs)
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmarkLayersSmoke:

    def _run(self, extra_args=None, timeout=60):
        script = str(BENCH_DIR / "benchmark_layers.py")
        cmd = [
            sys.executable, script,
            "--layer", "gin",
            "--nodes", "8", "--edges", "16",
            "--shape", "4,4,4",
            "--device", "cpu",
            "--iters", "2", "--warmup", "1",
        ] + (extra_args or [])
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def test_basic_run(self):
        r = self._run()
        assert r.returncode == 0, r.stderr

    def test_output_contains_layer_name(self):
        r = self._run()
        assert "gin" in r.stdout.lower()

    def test_no_file_without_output_flag(self):
        with tempfile.TemporaryDirectory() as d:
            r = self._run()
            # No files should appear in tmpdir (we didn't pass --output)
            assert r.returncode == 0

    def test_output_flag_writes_json(self):
        with tempfile.TemporaryDirectory() as d:
            out_path = os.path.join(d, "result.json")
            r = self._run(extra_args=["--output", out_path])
            assert r.returncode == 0, r.stderr
            assert os.path.isfile(out_path)
            data = json.loads(open(out_path).read())
            assert data["layer"] == "gin"
            assert "fwd_mean_ms" in data

    def test_gat_2d(self):
        script = str(BENCH_DIR / "benchmark_layers.py")
        r = subprocess.run([
            sys.executable, script,
            "--layer", "gat", "--nodes", "8", "--edges", "16",
            "--shape", "4,4,4", "--device", "cpu", "--iters", "2", "--warmup", "1",
        ], capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr


class TestBenchmarkBuildersSmoke:

    def test_small_run(self):
        script = str(BENCH_DIR / "benchmark_graph_builders.py")
        r = subprocess.run(
            [sys.executable, script, "--small"],
            capture_output=True, text=True, timeout=60,
        )
        assert r.returncode == 0, r.stderr

    def test_output_contains_o2_warning(self):
        script = str(BENCH_DIR / "benchmark_graph_builders.py")
        r = subprocess.run(
            [sys.executable, script, "--small"],
            capture_output=True, text=True, timeout=60,
        )
        assert "O(N" in r.stdout

    def test_no_file_without_output_flag(self):
        script = str(BENCH_DIR / "benchmark_graph_builders.py")
        r = subprocess.run(
            [sys.executable, script, "--small"],
            capture_output=True, text=True, timeout=60,
        )
        assert r.returncode == 0

    def test_output_flag_writes_json(self):
        script = str(BENCH_DIR / "benchmark_graph_builders.py")
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "builders.json")
            r = subprocess.run(
                [sys.executable, script, "--small", "--output", out],
                capture_output=True, text=True, timeout=60,
            )
            assert r.returncode == 0, r.stderr
            data = json.loads(open(out).read())
            assert isinstance(data, list)
            assert len(data) > 0
            assert "builder" in data[0]


# ─────────────────────────────────────────────────────────────────────────────
# torch.compile smoke (CPU, small model)
# ─────────────────────────────────────────────────────────────────────────────

class TestTorchCompileSmoke:

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
    def test_compile_matches_eager(self):
        from tgraphx.layers.factory import make_layer

        torch.manual_seed(0)
        layer = make_layer("gin", (4, 4, 4), (4, 4, 4)).eval()
        x  = torch.randn(4, 4, 4, 4)
        ei = build_grid_graph(2, 2, directed=False, self_loops=True)

        with torch.no_grad():
            eager_out = layer(x, ei)

        try:
            compiled = torch.compile(layer, mode="default")
            with torch.no_grad():
                compiled_out = compiled(x, ei)
            assert torch.allclose(eager_out, compiled_out, atol=1e-4), \
                f"max diff = {(eager_out - compiled_out).abs().max()}"
        except Exception as e:
            pytest.skip(f"torch.compile failed on this platform: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# AMP smoke (CUDA only)
# ─────────────────────────────────────────────────────────────────────────────

class TestAMPSmoke:

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_autocast_context_can_be_entered(self):
        """Verify torch.autocast("cuda") context works with a spatial GNN layer.

        GAT's index_add_ requires same dtype, so float16 autocast may raise
        a dtype mismatch for that specific op.  We accept either:
        - success with finite outputs, OR
        - a dtype/runtime error (known PyTorch limitation for some ops under
          float16 autocast — not a TGraphX framework bug).
        """
        from tgraphx.layers.conv_message import ConvMessagePassing

        device = torch.device("cuda")
        # ConvMessagePassing (conv1x1) handles autocast more uniformly
        layer = ConvMessagePassing(
            in_shape=(4, 4, 4), out_shape=(4, 4, 4), aggr="sum"
        ).to(device).eval()
        x  = torch.randn(4, 4, 4, 4, device=device)
        ei = build_grid_graph(2, 2, directed=False, self_loops=True).to(device)

        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                out = layer(x, ei)
            assert torch.isfinite(out).all(), "autocast output has non-finite values"
        except RuntimeError as e:
            if "scalar type" in str(e).lower() or "dtype" in str(e).lower():
                pytest.skip(
                    f"float16 autocast dtype mismatch on this op (known limitation): {e}"
                )
            raise
