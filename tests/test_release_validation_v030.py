"""Release-blocking validation tests for v0.3.0 (no network)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "examples"
PUBLIC = EXAMPLES / "public_datasets"


# ── Public-dataset script imports + skip behaviour ──────────────────────────


class TestPublicDatasetScripts:
    @pytest.mark.parametrize("script", [
        "fake_torchvision_patch_smoke.py",
        "mnist_patch_smoke.py",
        "pyg_cora_smoke.py",
        "ogb_arxiv_smoke.py",
        "dgl_cora_smoke.py",
    ])
    def test_help(self, script):
        result = subprocess.run(
            [sys.executable, str(PUBLIC / script), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"{script} --help failed:\n{result.stderr}"

    def test_fake_torchvision_runs(self, tmp_path):
        """FakeData script must not need the network and must produce dashboard files."""
        result = subprocess.run(
            [sys.executable, str(PUBLIC / "fake_torchvision_patch_smoke.py"),
             "--epochs", "2", "--max-samples", "4",
             "--output-run-dir", str(tmp_path / "fake_run")],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        run_dir = tmp_path / "fake_run"
        assert (run_dir / "metrics.csv").exists()
        assert (run_dir / "run_metadata.json").exists()
        assert (run_dir / "dataset_metadata.json").exists()
        assert (run_dir / "metrics_summary.json").exists()
        # Status flipped to completed.
        meta = json.loads((run_dir / "run_metadata.json").read_text())
        assert meta["status"] == "completed"

    def test_mnist_without_download_does_not_network(self, tmp_path):
        """``mnist_patch_smoke`` without ``--download`` must exit cleanly without network."""
        result = subprocess.run(
            [sys.executable, str(PUBLIC / "mnist_patch_smoke.py")],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        # Either skip message (torchvision missing) or "requires --download" message.
        assert "download" in result.stdout.lower() or "skip" in result.stdout.lower()

    def test_pyg_without_dep_skips_or_requires_download(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(PUBLIC / "pyg_cora_smoke.py")],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0


# ── Dashboard validation script ─────────────────────────────────────────────


class TestDashboardArtifactValidation:
    def test_runs(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(EXAMPLES / "dashboard_artifact_validation.py"),
             "--output-run-dir", str(tmp_path / "dash")],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        # Parse the JSON report (it is the only thing on stdout).
        report = json.loads(result.stdout)
        assert report["passed"] is True
        # Required artefacts.
        for f in ("metrics.csv", "run_metadata.json",
                  "dataset_metadata.json", "transform_metadata.json",
                  "metrics_summary.json", "benchmark_results.json",
                  "explanation_metadata.json", "explanation_edges.csv",
                  "explanation_patch_heatmap.json",
                  "experiment_config.json", "experiment_summary.json",
                  "hardware_report.json", "sampling_metadata.json",
                  "hetero_graph_metadata.json", "temporal_metadata.json",
                  "snapshot.html"):
            assert f in report["files"], f"Missing artefact: {f}"


# ── Device validation script ────────────────────────────────────────────────


class TestDeviceValidation:
    def test_cpu_quick(self, tmp_path):
        out = tmp_path / "cpu.json"
        result = subprocess.run(
            [sys.executable, str(EXAMPLES / "device_validation.py"),
             "--device", "cpu", "--quick",
             "--output-json", str(out)],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(out.read_text())
        assert data["all_passed"] is True
        assert "vector" in data["results"]


# ── Experiment + explainability end-to-end ──────────────────────────────────


class TestEndToEndScripts:
    def test_experiment_end_to_end(self, tmp_path):
        result = subprocess.run(
            [sys.executable,
             str(EXAMPLES / "experiment_end_to_end_validation.py"),
             "--epochs", "2",
             "--output-run-dir", str(tmp_path / "exp_e2e")],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr

    def test_explainability_end_to_end(self, tmp_path):
        result = subprocess.run(
            [sys.executable,
             str(EXAMPLES / "explainability_end_to_end_validation.py"),
             "--epochs", "3",
             "--output-run-dir", str(tmp_path / "explain_e2e")],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr


# ── README honest-claim guards ──────────────────────────────────────────────


class TestReadmeHonesty:
    def test_readme_does_not_claim_cuda_ci(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
        # We allow "no GPU runners in CI" / "Local tests" but never "CUDA CI".
        assert "cuda ci" not in text, "README must not claim 'CUDA CI'"

    def test_readme_does_not_claim_full_mps(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
        # Forbidden positive claim.
        assert "full mps" not in text

    def test_readme_does_not_claim_full_automatic_ddp(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        # The phrase "automatic multi-GPU training framework" may appear ONLY in a
        # denial context (e.g. "not an automatic multi-GPU training framework").
        phrase = "automatic multi-GPU training framework"
        if phrase.lower() in text.lower():
            for line in text.splitlines():
                if phrase.lower() in line.lower():
                    assert "not" in line.lower() or "no " in line.lower(), (
                        f"Positive automatic multi-GPU framework claim detected: {line}"
                    )

    def test_readme_does_not_claim_all_public_datasets_validated(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
        # Forbidden phrases.
        for bad in ["all public datasets validated",
                    "every public dataset",
                    "all torchvision datasets validated",
                    "all pyg datasets validated",
                    "all dgl datasets validated"]:
            assert bad not in text, f"README contains forbidden claim: {bad!r}"

    def test_readme_does_not_claim_sota(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        for bad in ["SOTA", "state-of-the-art"]:
            # Allow denial phrasing only.
            if bad.lower() in text.lower():
                # Find the line and ensure it is in a denial context.
                for line in text.splitlines():
                    if bad.lower() in line.lower():
                        assert any(neg in line.lower()
                                   for neg in ["no ", "not ", "never",
                                               "make no", "no real"]), (
                            f"README contains positive {bad!r} claim: {line}"
                        )

    def test_readme_no_full_replacement_claim(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
        for line in text.splitlines():
            if "drop-in replacement for pyg" in line and "not" not in line:
                pytest.fail(f"README contains positive PyG drop-in claim: {line}")
            if "drop-in replacement for dgl" in line and "not" not in line:
                pytest.fail(f"README contains positive DGL drop-in claim: {line}")
