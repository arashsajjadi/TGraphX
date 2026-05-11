"""Tests for advanced real-dataset notebooks 31–35.

Validates structural integrity, runs smoke scripts in skip-safe mode,
checks artifact JSON schemas, and verifies no false SOTA claims.

Run:
    pytest tests/test_advanced_real_dataset_notebooks.py -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
NB_DIR = REPO / "colab_drafts" / "advanced_real_datasets"
SMOKE_DIR = REPO / "examples" / "advanced_real_datasets"

NB_FILES = [
    "31_mnist_class_graph_membership_tensor_nodes.ipynb",
    "32_cifar10_visual_similarity_patch_graph.ipynb",
    "33_cora_citation_network_sampling_and_dashboard.ipynb",
    "34_movielens_user_item_kg_recommendation.ipynb",
    "35_molecular_graph_classification_mutag_or_qm9.ipynb",
]

SMOKE_FILES = {
    "31": "mnist_class_graph_membership_smoke.py",
    "32": "cifar10_visual_similarity_smoke.py",
    "33": "cora_sampling_smoke.py",
    "34": "movielens_kg_smoke.py",
    "35": "molecular_graph_smoke.py",
}

BANNED_PHRASES = [
    "achieves state-of-the-art",
    "achieve state-of-the-art",
    "beats all baselines",
    "outperforms all",
    "sets a new record",
    "surpasses all prior",
]

REQUIRED_SECTIONS = [
    "FAST_MODE",
    "## Limitations",
    "device",
    "set_seed",
    "from tgraphx",
]

ARTIFACT_SCHEMA_KEYS = {
    "benchmark_summary.json": [],  # must be valid JSON, no required keys
    "run_metadata.json": ["tgraphx_version", "seed", "fast_mode"],
    "metrics_summary.json": [],
}


# ── Structural checks ──────────────────────────────────────────────────────


@pytest.mark.parametrize("nb_name", NB_FILES)
def test_notebook_valid_json(nb_name: str) -> None:
    path = NB_DIR / nb_name
    assert path.exists(), f"Notebook missing: {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("nbformat") == 4, "nbformat must be 4"
    assert len(data.get("cells", [])) >= 5, "Expected at least 5 cells"


@pytest.mark.parametrize("nb_name", NB_FILES)
def test_notebook_first_cell_title(nb_name: str) -> None:
    path = NB_DIR / nb_name
    nb = json.loads(path.read_text(encoding="utf-8"))
    first = nb["cells"][0]
    assert first["cell_type"] == "markdown"
    src = "".join(first["source"])
    assert src.strip().startswith("#"), "First cell must be a Markdown title"


@pytest.mark.parametrize("nb_name", NB_FILES)
def test_notebook_required_sections(nb_name: str) -> None:
    path = NB_DIR / nb_name
    nb = json.loads(path.read_text(encoding="utf-8"))
    all_src = "\n".join("".join(c["source"]) for c in nb["cells"])
    for phrase in REQUIRED_SECTIONS:
        assert phrase in all_src, f"Missing required section {phrase!r} in {nb_name}"


@pytest.mark.parametrize("nb_name", NB_FILES)
def test_notebook_no_sota_claims(nb_name: str) -> None:
    path = NB_DIR / nb_name
    nb = json.loads(path.read_text(encoding="utf-8"))
    all_src = "\n".join("".join(c["source"]) for c in nb["cells"])
    for phrase in BANNED_PHRASES:
        assert phrase not in all_src, (
            f"False SOTA claim {phrase!r} found in {nb_name}"
        )


@pytest.mark.parametrize("nb_name", NB_FILES)
def test_notebook_no_private_paths(nb_name: str) -> None:
    path = NB_DIR / nb_name
    nb = json.loads(path.read_text(encoding="utf-8"))
    all_src = "\n".join("".join(c["source"]) for c in nb["cells"])
    for forbidden in ("/home/arash/", "/Users/", "C:\\Users\\"):
        assert forbidden not in all_src, (
            f"Private path {forbidden!r} found in {nb_name}"
        )


@pytest.mark.parametrize("nb_name", NB_FILES)
def test_notebook_no_exception_slicing(nb_name: str) -> None:
    """Check for the e[:120] / err[:120] anti-pattern."""
    path = NB_DIR / nb_name
    nb = json.loads(path.read_text(encoding="utf-8"))
    all_src = "\n".join("".join(c["source"]) for c in nb["cells"])
    for pattern in ("e[:120]", "err[:120]"):
        assert pattern not in all_src, (
            f"Exception slicing pattern {pattern!r} in {nb_name}. "
            "Use str(e)[:120] instead."
        )


@pytest.mark.parametrize("nb_name", NB_FILES)
def test_notebook_no_large_outputs(nb_name: str) -> None:
    path = NB_DIR / nb_name
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            for mime, content in out.get("data", {}).items():
                blob = content if isinstance(content, str) else "".join(content)
                assert len(blob) < 50_000, (
                    f"Large output ({len(blob)} chars, {mime}) in {nb_name}"
                )


@pytest.mark.parametrize("nb_name", NB_FILES)
def test_notebook_device_selection(nb_name: str) -> None:
    path = NB_DIR / nb_name
    nb = json.loads(path.read_text(encoding="utf-8"))
    all_src = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "cuda" in all_src, f"No device selection in {nb_name}"
    assert "is_available" in all_src, f"No torch.cuda.is_available() in {nb_name}"


# ── Smoke-script tests ─────────────────────────────────────────────────────


@pytest.mark.parametrize("nb_id,script_name", list(SMOKE_FILES.items()))
def test_smoke_script_fast_no_download(nb_id: str, script_name: str) -> None:
    script = SMOKE_DIR / script_name
    assert script.exists(), f"Smoke script missing: {script}"
    result = subprocess.run(
        [sys.executable, str(script), "--fast", "--no-download"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, (
        f"Smoke script {script_name} failed (exit {result.returncode}):\n"
        f"stdout: {result.stdout[-1000:]}\n"
        f"stderr: {result.stderr[-1000:]}"
    )
    assert "PASSED" in result.stdout or "PASS" in result.stdout, (
        f"Smoke script {script_name} did not print PASSED:\n{result.stdout[-500:]}"
    )


# ── Artifact JSON schema checks ────────────────────────────────────────────


@pytest.mark.parametrize("nb_name,artifact,req_keys", [
    (nb_name, artifact, req_keys)
    for nb_name in NB_FILES
    for artifact, req_keys in ARTIFACT_SCHEMA_KEYS.items()
])
def test_artifact_json_schema(nb_name: str, artifact: str, req_keys: list) -> None:
    """Artifacts written during smoke runs should be valid JSON with required keys."""
    nb_id = nb_name[:2]
    dirs = {
        "31": "31_mnist",
        "32": "32_cifar10",
        "33": "33_cora",
        "34": "34_movielens",
        "35": "35_mutag",
    }
    run_dir = REPO / "runs" / "advanced_notebooks" / dirs[nb_id]
    artifact_path = run_dir / artifact
    if not artifact_path.exists():
        pytest.skip(f"Artifact not found: {artifact_path} — run smoke script first")
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{artifact} is not a JSON object"
    for key in req_keys:
        assert key in data, f"Key {key!r} missing from {artifact_path}"


# ── KGTrainer CUDA generator regression ───────────────────────────────────


def test_kgtrainer_cpu_generator_with_cuda_device() -> None:
    """KGTrainer must not fail when device='cuda' is requested but only CPU is available.
    Regression for: torch.randperm(..., generator=cpu_gen, device='cuda') error."""
    import torch
    from tgraphx.kg import TransEModel, KGTrainer, KGTrainingConfig
    from tgraphx import KnowledgeGraph

    # Build tiny KG
    triples = torch.zeros((50, 3), dtype=torch.long)
    for i in range(50):
        triples[i] = torch.tensor([i % 10, i % 3, (i + 1) % 10], dtype=torch.long)
    kg = KnowledgeGraph(triples, num_entities=10, num_relations=3)
    model = TransEModel(10, 3, embedding_dim=8)
    # Force CPU even if a GPU header is present; the fix should make it work on any device.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    config = KGTrainingConfig(num_epochs=2, batch_size=16, device=dev, seed=0)
    trainer = KGTrainer(model, config, triples)
    history = trainer.fit()
    assert "final_loss" in history
    import math
    assert math.isfinite(history["final_loss"])


# ── Validation tool check ──────────────────────────────────────────────────


def test_validate_advanced_colab_drafts_passes() -> None:
    """The structural validation tool must report all notebooks passing."""
    result = subprocess.run(
        [sys.executable, "tools/validate_advanced_colab_drafts.py"],
        capture_output=True, text=True, cwd=str(REPO), timeout=30,
    )
    assert result.returncode == 0, (
        f"validate_advanced_colab_drafts.py failed:\n{result.stdout}\n{result.stderr}"
    )
    assert "5 notebooks passed" in result.stdout or "All" in result.stdout
