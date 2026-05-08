"""Documentation honesty tests for the v0.2.9 dataset/transforms/metrics ecosystem."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
README = ROOT / "README.md"
DOCS = ROOT / "docs"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


# ── README claims ────────────────────────────────────────────────────────────


class TestReadmeWording:
    def test_says_no_bundled_datasets(self):
        text = _read(README).lower()
        # Either of these honest statements must appear.
        assert (
            "does not redistribute" in text
            or "no bundled datasets" in text
            or "not redistribute third-party datasets" in text
        )

    def test_mentions_download_true(self):
        text = _read(README).lower()
        assert "download=true" in text or "explicit download" in text

    def test_no_full_pyg_replacement_claim(self):
        text = _read(README).lower()
        assert "drop-in replacement for pyg" not in text or \
               "not a drop-in replacement for pyg" in text

    def test_no_sota_for_synthetic(self):
        text = _read(README).lower()
        # Either no positive SOTA wording, or we explicitly disclaim.
        bad = re.compile(
            r"\b(?:tgraphx|synthetic)\s+(?:is|are|achieves?)\s+state-of-the-art\b",
            re.IGNORECASE,
        )
        assert bad.search(text) is None


# ── Lazy import discipline ──────────────────────────────────────────────────


class TestLazyImports:
    def test_importing_datasets_does_not_load_pyg_dgl_ogb(self, tmp_path):
        import subprocess
        prog = "\n".join([
            "import sys",
            "import tgraphx",
            "import tgraphx.datasets",
            "import tgraphx.transforms",
            "import tgraphx.metrics",
            "for m in ('torch_geometric', 'dgl', 'ogb'):",
            "    assert m not in sys.modules, f'{m} imported eagerly'",
        ])
        f = tmp_path / "check.py"
        f.write_text(prog)
        result = subprocess.run(
            [sys.executable, str(f)], capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr

    def test_dataset_info_does_not_import_optionals(self):
        # Running list_datasets() / dataset_info() must not import optional deps.
        before = {m for m in sys.modules}
        from tgraphx.datasets import list_datasets, dataset_info
        list_datasets()
        dataset_info("synthetic:patch_graph")
        for m in ("torch_geometric", "dgl", "ogb"):
            assert m not in sys.modules


# ── Docs files exist ────────────────────────────────────────────────────────


class TestDocsExist:
    @pytest.mark.parametrize("name", [
        "datasets.md",
        "transforms.md",
        "metrics.md",
        "benchmarks.md",
        "dataset_license_policy.md",
    ])
    def test_docs_files_present(self, name):
        assert (DOCS / name).exists(), f"docs/{name} is missing"

    def test_dataset_doc_mentions_no_redistribution(self):
        text = _read(DOCS / "datasets.md").lower()
        assert (
            "does not redistribute" in text
            or "no bundled datasets" in text
            or "not redistribute" in text
        )
