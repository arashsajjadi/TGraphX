"""Notebook structural tests (v1.3).

Validates every notebook/*.ipynb without executing code.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))

_PRIVATE_PATH_RE = re.compile(r"/home/[a-zA-Z0-9_]+|/Users/[a-zA-Z0-9_]+")
_SECRET_RE = re.compile(r"(?i)(token\s*=|password\s*=|api_key\s*=)['\"]?\w{8,}")


@pytest.fixture(scope="module")
def all_notebooks():
    assert NOTEBOOKS_DIR.exists(), f"notebooks/ directory not found: {NOTEBOOKS_DIR}"
    assert len(NOTEBOOKS) > 0, f"No .ipynb files in {NOTEBOOKS_DIR}"
    return {nb.name: json.loads(nb.read_text(encoding="utf-8")) for nb in NOTEBOOKS}


class TestNotebooksExist:
    def test_notebooks_dir_exists(self):
        assert NOTEBOOKS_DIR.exists()

    def test_at_least_7_notebooks(self):
        assert len(NOTEBOOKS) >= 7, f"Expected >=7 notebooks, found {len(NOTEBOOKS)}"

    @pytest.mark.parametrize("name", [
        "01_easy_tensor_node_classification.ipynb",
        "02_image_patch_tensor_graph.ipynb",
        "03_kg_completion_rescal_simple_hpo.ipynb",
        "04_graph_generation_and_optimization.ipynb",
        "05_graph_rl_coloring_and_navigation.ipynb",
        "06_graph_io_roundtrip.ipynb",
        "07_benchmark_suite_and_dashboard.ipynb",
    ])
    def test_specific_notebook_exists(self, name):
        assert (NOTEBOOKS_DIR / name).exists(), f"Missing: {name}"


class TestNotebookStructure:
    def test_all_valid_json(self, all_notebooks):
        # Already validated by the fixture (json.loads raises on invalid).
        assert len(all_notebooks) >= 7

    def test_all_have_nbformat_4(self, all_notebooks):
        for name, nb in all_notebooks.items():
            assert nb.get("nbformat") == 4, f"{name}: expected nbformat 4"

    def test_all_have_cells(self, all_notebooks):
        for name, nb in all_notebooks.items():
            assert len(nb.get("cells", [])) >= 3, f"{name}: too few cells"

    def test_all_have_markdown_cells(self, all_notebooks):
        for name, nb in all_notebooks.items():
            md = [c for c in nb["cells"] if c.get("cell_type") == "markdown"]
            assert len(md) >= 2, f"{name}: needs at least 2 Markdown cells"

    def test_all_have_code_cells(self, all_notebooks):
        for name, nb in all_notebooks.items():
            code = [c for c in nb["cells"] if c.get("cell_type") == "code"]
            assert len(code) >= 2, f"{name}: needs at least 2 code cells"

    def test_first_cell_is_heading(self, all_notebooks):
        for name, nb in all_notebooks.items():
            first = nb["cells"][0]
            src = "".join(first.get("source", []))
            assert first["cell_type"] == "markdown", f"{name}: first cell must be Markdown"
            assert src.strip().startswith("#"), f"{name}: first cell must start with #"


class TestNotebookContent:
    def _all_source(self, nb: dict) -> str:
        return "\n".join(
            "".join(c.get("source", [])) for c in nb.get("cells", [])
        )

    def test_references_tgraphx(self, all_notebooks):
        for name, nb in all_notebooks.items():
            assert "tgraphx" in self._all_source(nb).lower(), \
                f"{name}: does not reference tgraphx"

    def test_no_private_paths(self, all_notebooks):
        for name, nb in all_notebooks.items():
            matches = _PRIVATE_PATH_RE.findall(self._all_source(nb))
            assert not matches, f"{name}: private path found: {matches}"

    def test_no_secrets(self, all_notebooks):
        for name, nb in all_notebooks.items():
            matches = _SECRET_RE.findall(self._all_source(nb))
            assert not matches, f"{name}: potential secret: {matches}"

    def test_no_excessive_outputs(self, all_notebooks):
        for name, nb in all_notebooks.items():
            for i, cell in enumerate(nb.get("cells", [])):
                for out in cell.get("outputs", []):
                    text = out.get("text", [])
                    n = len(text) if isinstance(text, list) else text.count("\n")
                    assert n < 200, f"{name} cell {i}: excessive output ({n} lines)"


class TestNotebookValidationTool:
    def test_validate_notebooks_passes(self):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "tools/validate_notebooks.py"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, \
            f"tools/validate_notebooks.py failed:\n{result.stdout}\n{result.stderr}"


class TestColabGallery:
    def test_colab_gallery_doc_exists(self):
        assert Path("docs/colab_gallery.md").exists()

    def test_colab_gallery_lists_notebooks(self):
        text = Path("docs/colab_gallery.md").read_text()
        for i in range(1, 8):
            assert f"0{i}_" in text, f"Notebook {i} not in colab_gallery.md"
