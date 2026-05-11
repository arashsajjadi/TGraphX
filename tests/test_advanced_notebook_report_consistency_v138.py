"""Strict report-consistency tests for advanced notebooks (v1.3.8).

Asserts that every notebook actually contains the content claimed in the
release report. These tests prevent claims from drifting away from code.

Run:
    pytest tests/test_advanced_notebook_report_consistency_v138.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
NB_DIR = REPO / "colab_drafts" / "advanced_real_datasets"

NB = {
    "31": NB_DIR / "31_mnist_class_graph_membership_tensor_nodes.ipynb",
    "32": NB_DIR / "32_cifar10_visual_similarity_patch_graph.ipynb",
    "33": NB_DIR / "33_cora_citation_network_sampling_and_dashboard.ipynb",
    "34": NB_DIR / "34_movielens_user_item_kg_recommendation.ipynb",
    "35": NB_DIR / "35_molecular_graph_classification_mutag_or_qm9.ipynb",
}


def _text(nbid: str) -> str:
    """Load notebook source; skip if file is absent (gitignored)."""
    p = NB[nbid]
    if not p.exists():
        pytest.skip(
            f"Notebook {p.name} not present (gitignored). "
            "Run tools/build_advanced_notebooks.py to generate locally."
        )
    nb = json.loads(p.read_text(encoding="utf-8"))
    return "\n".join("".join(c.get("source", [])) for c in nb["cells"])


# ── Global ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("nbid", list(NB.keys()))
def test_scientific_notes_section(nbid: str) -> None:
    """Every notebook must have an explicit scientific notes section."""
    assert "## Scientific and methodological notes" in _text(nbid), (
        f"NB{nbid}: missing '## Scientific and methodological notes' section"
    )


@pytest.mark.parametrize("nbid", list(NB.keys()))
def test_leakage_policy_present(nbid: str) -> None:
    t = _text(nbid)
    assert "Leakage policy" in t or "leakage policy" in t.lower(), (
        f"NB{nbid}: missing leakage policy"
    )


@pytest.mark.parametrize("nbid", list(NB.keys()))
def test_split_policy_documented(nbid: str) -> None:
    t = _text(nbid)
    assert "Split policy" in t or "split policy" in t.lower() or "Split:" in t, (
        f"NB{nbid}: missing split policy"
    )


@pytest.mark.parametrize("nbid", list(NB.keys()))
def test_no_sota_claim(nbid: str) -> None:
    t = _text(nbid)
    for bad in ("achieves state-of-the-art", "beats all baselines", "outperforms all"):
        assert bad not in t, f"NB{nbid}: false SOTA claim {bad!r}"


@pytest.mark.parametrize("nbid", list(NB.keys()))
def test_fast_mode_disclaimer(nbid: str) -> None:
    """Every notebook must disclaim that FAST_MODE metrics are not benchmark claims."""
    t = _text(nbid).lower()
    assert ("fast_mode metrics are not benchmark claims" in t
            or "fast_mode is" in t
            or "not benchmark claim" in t), (
        f"NB{nbid}: missing FAST_MODE disclaimer"
    )


@pytest.mark.parametrize("nbid", list(NB.keys()))
def test_device_handling_in_source(nbid: str) -> None:
    t = _text(nbid)
    assert "is_available" in t and "cuda" in t.lower(), (
        f"NB{nbid}: missing device/CUDA handling"
    )


@pytest.mark.parametrize("nbid", list(NB.keys()))
def test_no_temporary_directory_artifacts(nbid: str) -> None:
    t = _text(nbid)
    assert "TemporaryDirectory" not in t, (
        f"NB{nbid}: uses tempfile.TemporaryDirectory — artifacts must be persistent"
    )


@pytest.mark.parametrize("nbid", list(NB.keys()))
def test_no_repo_local_paths(nbid: str) -> None:
    t = _text(nbid)
    for bad in ("/home/arash/", "/Users/", "C:\\Users\\"):
        assert bad not in t, f"NB{nbid}: private path {bad!r}"


# ── NB31 specifics ────────────────────────────────────────────────────────


def test_nb31_edge_types() -> None:
    t = _text("31")
    assert "edge_type 0 (visual_similarity)" in t, "NB31: missing visual_similarity edge_type label"
    assert "edge_type 1 (prototype_membership)" in t, "NB31: missing prototype_membership edge_type label"


def test_nb31_edge_attr_on_graph() -> None:
    t = _text("31")
    assert "edge_attr=all_edge_attr" in t, "NB31: edge_attr not passed to Graph"


def test_nb31_train_only_prototypes() -> None:
    t = _text("31")
    assert "TRAINING nodes only" in t or "train labels only" in t.lower() or "train-mask labels" in t, (
        "NB31: train-only prototype policy not documented in code"
    )


def test_nb31_flatten_mlp_class() -> None:
    t = _text("31")
    assert "class FlattenMLP" in t, "NB31: FlattenMLP class missing"


# ── NB32 specifics ────────────────────────────────────────────────────────


def test_nb32_cifar10_patch_dataset() -> None:
    t = _text("32")
    assert "CIFAR10PatchGraphDataset" in t, "NB32: missing CIFAR10PatchGraphDataset"


def test_nb32_graph_data_loader() -> None:
    assert "GraphDataLoader" in _text("32"), "NB32: missing GraphDataLoader"


def test_nb32_mean_and_max_pool() -> None:
    t = _text("32")
    assert "global_mean_pool" in t and "global_max_pool" in t, (
        "NB32: missing mean+max pooling"
    )


def test_nb32_patch_tensor_shape() -> None:
    t = _text("32")
    assert "PATCH_SIZE" in t and "patch_shape" in t, (
        "NB32: patch tensor shape not declared"
    )


def test_nb32_baseline_actually_trained() -> None:
    t = _text("32")
    assert "Training FlattenMLP" in t and "model_baseline = FlattenMLP" in t, (
        "NB32: FlattenMLP baseline must be actually trained"
    )


def test_nb32_inductive_declared() -> None:
    t = _text("32")
    assert "inductive" in t.lower() or "graph classification" in t.lower(), (
        "NB32: inductive task not declared"
    )


# ── NB33 specifics ────────────────────────────────────────────────────────


def test_nb33_no_graph_rl_training() -> None:
    assert "graph RL training" not in _text("33"), (
        "NB33: forbidden phrase 'graph RL training'"
    )


def test_nb33_transductive_declared() -> None:
    assert "transductive" in _text("33").lower(), (
        "NB33: transductive setting not declared"
    )


def test_nb33_flatten_mlp_baseline() -> None:
    t = _text("33")
    assert "class FlattenMLP" in t, "NB33: FlattenMLP baseline missing"


def test_nb33_sampling_metadata() -> None:
    t = _text("33")
    assert "sampling_metadata.json" in t and "write_sampling_metadata" in t, (
        "NB33: sampling metadata not written"
    )


def test_nb33_persistent_artifacts_not_tempdir() -> None:
    t = _text("33")
    assert "runs/advanced_notebooks/33_cora" in t, (
        "NB33: missing persistent runs/ directory"
    )


# ── NB34 specifics ────────────────────────────────────────────────────────


def test_nb34_rated_high_relation() -> None:
    assert "rated_high" in _text("34"), "NB34: missing rated_high"


def test_nb34_has_genre_relation() -> None:
    assert "has_genre" in _text("34"), "NB34: missing has_genre"


def test_nb34_has_occupation_relation() -> None:
    assert "has_occupation" in _text("34") or "REL_HAS_OCC" in _text("34"), (
        "NB34: missing has_occupation"
    )


def test_nb34_entity_features() -> None:
    assert "entity_features" in _text("34"), "NB34: missing entity_features"


def test_nb34_run_kg_hpo() -> None:
    assert "run_kg_hpo" in _text("34"), "NB34: missing run_kg_hpo"


def test_nb34_separate_val_test() -> None:
    t = _text("34")
    assert "val_triples" in t and "test_triples" in t, (
        "NB34: missing separate val/test splits"
    )


def test_nb34_topk_with_titles() -> None:
    t = _text("34")
    assert "title" in t.lower() and "Top-5" in t, (
        "NB34: missing top-K with movie titles"
    )


def test_nb34_popularity_baseline() -> None:
    t = _text("34")
    assert "Popularity" in t or "popularity" in t.lower(), (
        "NB34: missing popularity baseline"
    )


def test_nb34_filtered_evaluation() -> None:
    t = _text("34")
    assert "filtered=True" in t or "filtered ranking" in t.lower(), (
        "NB34: filtered evaluation not declared"
    )


# ── NB35 specifics ────────────────────────────────────────────────────────


def test_nb35_edge_attr_in_graph() -> None:
    t = _text("35")
    assert "edge_attr=bond_feat" in t or "edge_attr=" in t, (
        "NB35: missing edge_attr= in Graph constructor"
    )


def test_nb35_mean_max_pool() -> None:
    t = _text("35")
    assert "global_mean_pool" in t and "global_max_pool" in t, (
        "NB35: missing mean+max readout"
    )


def test_nb35_edge_feature_projection() -> None:
    t = _text("35")
    assert "edge_proj" in t or "edge_feature projection" in t.lower(), (
        "NB35: missing edge-feature projection in model"
    )


def test_nb35_no_no_edge_features_claim() -> None:
    t = _text("35")
    assert "No edge features are used" not in t, (
        "NB35: stale 'No edge features are used' claim found"
    )


def test_nb35_motif_mining() -> None:
    t = _text("35")
    assert "motif_profile" in t and "graph_summary" in t, (
        "NB35: missing motif/mining summary"
    )


def test_nb35_baseline_actually_trained() -> None:
    t = _text("35")
    assert "DegreeFeatureBaseline" in t and "train_baseline" in t, (
        "NB35: degree baseline not actually trained"
    )
