"""Executed-notebook tests for advanced real-dataset notebooks 31–35 (v1.3.8).

These tests verify that the SHIPPED notebooks contain executed cells with
outputs — preventing regressions where notebook files claim a workflow but
the cells were never executed.

Run:
    pytest tests/test_advanced_notebook_execution_v138.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
NB_DIR = REPO / "colab_drafts" / "advanced_real_datasets"

NB_FILES = [
    "31_mnist_class_graph_membership_tensor_nodes.ipynb",
    "32_cifar10_visual_similarity_patch_graph.ipynb",
    "33_cora_citation_network_sampling_and_dashboard.ipynb",
    "34_movielens_user_item_kg_recommendation.ipynb",
    "35_molecular_graph_classification_mutag_or_qm9.ipynb",
]


def _load(name: str) -> dict:
    """Load a shipped notebook. Skips test if file is absent.

    Per repo policy, .ipynb files in colab_drafts/ are NOT tracked in git
    (they live in Google Drive/Colab). These execution tests run against
    locally-generated notebooks; CI environments that have not run
    `tools/build_advanced_notebooks.py` + `tools/execute_advanced_colab_drafts.py`
    should skip these tests cleanly.
    """
    p = NB_DIR / name
    if not p.exists():
        pytest.skip(
            f"Notebook {name} not present (gitignored). "
            "Run tools/build_advanced_notebooks.py and "
            "tools/execute_advanced_colab_drafts.py locally to generate and execute."
        )
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.mark.parametrize("nb_name", NB_FILES)
def test_all_code_cells_executed(nb_name: str) -> None:
    nb = _load(nb_name)
    code = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    unexec = []
    for i, cell in enumerate(code):
        src = "".join(cell.get("source", [])).strip()
        if src and cell.get("execution_count") is None:
            unexec.append(i)
    assert not unexec, (
        f"{nb_name}: code cells not executed (idx={unexec}). "
        "Run tools/execute_advanced_colab_drafts.py to execute notebooks."
    )


@pytest.mark.parametrize("nb_name", NB_FILES)
def test_at_least_some_outputs(nb_name: str) -> None:
    nb = _load(nb_name)
    code = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    n_outputs = sum(len(c.get("outputs", [])) for c in code)
    assert n_outputs >= 5, (
        f"{nb_name}: too few code-cell outputs ({n_outputs}); "
        "notebook may not be fully executed."
    )


@pytest.mark.parametrize("nb_name", NB_FILES)
def test_no_error_outputs(nb_name: str) -> None:
    nb = _load(nb_name)
    errors = []
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                ename = out.get("ename", "?")
                evalue = out.get("evalue", "")
                errors.append(f"cell[{i}] {ename}: {evalue[:120]}")
    assert not errors, (
        f"{nb_name}: error outputs found: {errors}"
    )


@pytest.mark.parametrize("nb_name", NB_FILES)
def test_final_completion_message_in_outputs(nb_name: str) -> None:
    """The notebook must print 'passed all checks' or 'Notebook completed'
    in some cell output as evidence that the final assertion ran successfully."""
    nb = _load(nb_name)
    all_output_text = ""
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                text = out.get("text", "")
                if isinstance(text, list):
                    text = "".join(text)
                all_output_text += text
    assert ("passed all checks" in all_output_text
            or "Notebook completed" in all_output_text), (
        f"{nb_name}: missing final completion message in outputs"
    )


@pytest.mark.parametrize("nb_name", NB_FILES)
def test_outputs_not_oversized(nb_name: str) -> None:
    """Reject notebooks bloated with huge outputs (>200KB single blob)."""
    nb = _load(nb_name)
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            for mime, content in data.items():
                blob = content if isinstance(content, str) else "".join(content)
                assert len(blob) < 200_000, (
                    f"{nb_name}: cell[{i}] mime={mime} oversized ({len(blob)} chars)"
                )
            text = out.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            assert len(text) < 200_000, (
                f"{nb_name}: cell[{i}] stream text oversized ({len(text)} chars)"
            )


@pytest.mark.parametrize("nb_name", NB_FILES)
def test_artifact_path_present_in_outputs(nb_name: str) -> None:
    """Verify the notebook reports an artifact directory in stdout."""
    nb = _load(nb_name)
    txt = ""
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            t = out.get("text", "")
            if isinstance(t, list):
                t = "".join(t)
            txt += t
    assert "runs/advanced_notebooks/" in txt or "Artifacts written" in txt, (
        f"{nb_name}: no artifact-directory mention in outputs"
    )


@pytest.mark.parametrize("nb_name", NB_FILES)
def test_tgraphx_version_printed(nb_name: str) -> None:
    nb = _load(nb_name)
    txt = ""
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            t = out.get("text", "")
            if isinstance(t, list):
                t = "".join(t)
            txt += t
    assert "TGraphX v" in txt, f"{nb_name}: no TGraphX version printed in outputs"


@pytest.mark.parametrize("nb_name", NB_FILES)
def test_device_printed(nb_name: str) -> None:
    nb = _load(nb_name)
    txt = ""
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            t = out.get("text", "")
            if isinstance(t, list):
                t = "".join(t)
            txt += t
    assert "device=" in txt or "Device:" in txt, (
        f"{nb_name}: no device print in outputs"
    )
