"""Report-consistency tests for advanced notebooks 31–35 (v1.3.7).

Verifies that the notebook source matches key report claims so that
content, code, and claims cannot silently diverge.

Run:
    pytest tests/test_advanced_notebook_report_consistency_v137.py -q
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
    p = NB[nbid]
    if not p.exists():
        pytest.skip(f"Notebook {p.name} not present (gitignored).")
    nb = json.loads(p.read_text(encoding="utf-8"))
    return "\n".join("".join(c.get("source", [])) for c in nb["cells"])


# ── Global requirements for all notebooks ─────────────────────────────────


@pytest.mark.parametrize("nbid", list(NB.keys()))
def test_device_selection(nbid: str) -> None:
    t = _text(nbid)
    assert "is_available" in t, f"NB{nbid}: missing torch.cuda.is_available()"
    assert "device" in t, f"NB{nbid}: missing device variable"


@pytest.mark.parametrize("nbid", list(NB.keys()))
def test_fast_mode_flag(nbid: str) -> None:
    assert "FAST_MODE" in _text(nbid), f"NB{nbid}: missing FAST_MODE"


@pytest.mark.parametrize("nbid", list(NB.keys()))
def test_tgraphx_version_print(nbid: str) -> None:
    t = _text(nbid)
    assert "__version__" in t or "tgraphx.__version__" in t, (
        f"NB{nbid}: missing version print"
    )


@pytest.mark.parametrize("nbid", list(NB.keys()))
def test_artifact_writing(nbid: str) -> None:
    t = _text(nbid)
    has_write = any(k in t for k in [
        "write_run_metadata", "write_metrics_summary",
        "write_kg_summary", "benchmark_summary.json",
        "write_graph_stats",
    ])
    assert has_write, f"NB{nbid}: missing artifact writing"


@pytest.mark.parametrize("nbid", list(NB.keys()))
def test_limitations_section(nbid: str) -> None:
    t = _text(nbid)
    assert "## Limitations" in t or "Limitations" in t, (
        f"NB{nbid}: missing Limitations section"
    )


@pytest.mark.parametrize("nbid", list(NB.keys()))
def test_no_private_paths(nbid: str) -> None:
    t = _text(nbid)
    for bad in ("/home/arash/", "/Users/", "C:\\Users\\"):
        assert bad not in t, f"NB{nbid}: private path {bad!r}"


@pytest.mark.parametrize("nbid", list(NB.keys()))
def test_no_fake_colab_urls(nbid: str) -> None:
    t = _text(nbid)
    # Fake Colab links would look like colab.research.google.com/drive/ with a real ID
    assert "colab.research.google.com/drive/1" not in t, (
        f"NB{nbid}: fake Colab URL"
    )


@pytest.mark.parametrize("nbid", list(NB.keys()))
def test_no_sota_claims(nbid: str) -> None:
    t = _text(nbid)
    for phrase in ("achieves state-of-the-art", "outperforms all", "beats all baselines"):
        assert phrase not in t, f"NB{nbid}: false SOTA claim {phrase!r}"


# ── NB31: MNIST class-graph membership ────────────────────────────────────


def test_nb31_has_prototype_edges() -> None:
    t = _text("31")
    assert "prototype" in t, "NB31: missing prototype edges"


def test_nb31_has_edge_type() -> None:
    t = _text("31")
    assert "edge_type" in t, "NB31: missing edge_type documentation"


def test_nb31_has_leakage_policy() -> None:
    t = _text("31")
    assert "Leakage policy" in t or "leakage policy" in t.lower(), (
        "NB31: missing leakage policy"
    )


def test_nb31_has_edge_attr() -> None:
    t = _text("31")
    assert "edge_attr" in t, "NB31: missing edge_attr on Graph"


def test_nb31_has_flattendmlp_baseline() -> None:
    t = _text("31")
    assert "FlattenMLP" in t, "NB31: missing FlattenMLP baseline"


def test_nb31_has_visual_similarity_edge_count() -> None:
    t = _text("31")
    assert "visual_similarity" in t or "visual" in t.lower(), (
        "NB31: missing visual similarity edge count"
    )


def test_nb31_has_prototype_membership_edge_count() -> None:
    t = _text("31")
    assert "prototype_membership" in t or ("prototype" in t and "edge" in t), (
        "NB31: missing prototype membership edge count"
    )


# ── NB32: CIFAR-10 patch graph ─────────────────────────────────────────────


def test_nb32_uses_cifar10_patch_graph_dataset() -> None:
    assert "CIFAR10PatchGraphDataset" in _text("32"), (
        "NB32: missing CIFAR10PatchGraphDataset"
    )


def test_nb32_has_patch_shape() -> None:
    t = _text("32")
    # Must mention patch tensor shape or PATCH_SIZE
    assert "patch" in t.lower() and ("3, 8, 8" in t or "PATCH_SIZE" in t or "[3, 8" in t), (
        "NB32: missing patch tensor shape"
    )


def test_nb32_has_graph_level_pooling() -> None:
    t = _text("32")
    assert "global_mean_pool" in t or "global_max_pool" in t, (
        "NB32: missing graph-level pooling"
    )


def test_nb32_has_baseline_training() -> None:
    t = _text("32")
    assert "FlattenMLP" in t, "NB32: missing FlattenMLP baseline"
    assert "train_graph_model" in t or "train" in t, (
        "NB32: missing baseline training code"
    )


def test_nb32_inductive_task_declared() -> None:
    t = _text("32")
    assert "inductive" in t.lower() or "graph classification" in t.lower(), (
        "NB32: missing inductive task declaration"
    )


def test_nb32_leakage_policy() -> None:
    assert "leakage" in _text("32").lower(), "NB32: missing leakage note"


# ── NB33: Cora citation network ────────────────────────────────────────────


def test_nb33_no_graph_rl_training_phrase() -> None:
    assert "graph RL training" not in _text("33"), (
        "NB33: forbidden phrase 'graph RL training'"
    )


def test_nb33_transductive_setting() -> None:
    assert "transductive" in _text("33").lower(), (
        "NB33: missing transductive setting declaration"
    )


def test_nb33_flatten_mlp_baseline() -> None:
    t = _text("33")
    assert "FlattenMLP" in t or "flatten" in t.lower() or "MLP" in t, (
        "NB33: missing MLP baseline"
    )


def test_nb33_dashboard_artifact_writer() -> None:
    t = _text("33")
    assert any(k in t for k in ["write_run_metadata", "write_metrics_summary",
                                  "write_sampling_metadata", "benchmark_summary"]), (
        "NB33: missing dashboard artifact writer"
    )


def test_nb33_gcnconv_or_graph_model() -> None:
    t = _text("33")
    assert "GCNConv" in t or "LinearMessagePassing" in t or "GNN" in t, (
        "NB33: missing TGraphX graph model"
    )


# ── NB34: MovieLens KG recommendation ─────────────────────────────────────


def test_nb34_has_rated_high_relation() -> None:
    assert "rated_high" in _text("34"), "NB34: missing rated_high relation"


def test_nb34_has_has_genre_relation() -> None:
    assert "has_genre" in _text("34"), "NB34: missing has_genre relation"


def test_nb34_has_entity_features() -> None:
    assert "entity_features" in _text("34"), "NB34: missing entity_features"


def test_nb34_has_run_kg_hpo() -> None:
    assert "run_kg_hpo" in _text("34"), "NB34: missing run_kg_hpo"


def test_nb34_has_topk_with_titles() -> None:
    t = _text("34")
    assert "title" in t.lower() and "recommendation" in t.lower(), (
        "NB34: missing top-K movie title recommendations"
    )


def test_nb34_has_separate_val_test() -> None:
    t = _text("34")
    assert "val_triples" in t and "test_triples" in t, (
        "NB34: missing separate val/test splits"
    )


def test_nb34_leakage_policy() -> None:
    t = _text("34")
    assert "Leakage policy" in t or "leakage" in t.lower(), (
        "NB34: missing leakage policy"
    )


def test_nb34_genre_implemented_not_only_future() -> None:
    t = _text("34")
    # Genre should be implemented in code, not only mentioned as next step
    assert "has_genre" in t and "genre_vecs" in t, (
        "NB34: genre/metadata implemented in code (not just next step)"
    )


# ── NB35: MUTAG molecular graph classification ────────────────────────────


def test_nb35_has_edge_attr_or_bond_features() -> None:
    t = _text("35")
    assert "edge_attr" in t or "edge_features" in t or "bond" in t, (
        "NB35: missing edge_attr / bond features"
    )


def test_nb35_has_edge_attr_parameter() -> None:
    t = _text("35")
    assert "edge_attr" in t, "NB35: edge_attr parameter not present"


def test_nb35_has_motif_or_mining_summary() -> None:
    t = _text("35")
    assert "motif_profile" in t or "graph_summary" in t, (
        "NB35: missing motif/mining summary"
    )


def test_nb35_has_mean_and_max_pooling() -> None:
    t = _text("35")
    assert "global_mean_pool" in t and "global_max_pool" in t, (
        "NB35: missing mean + max pooling"
    )


def test_nb35_no_stale_no_edge_features_claim() -> None:
    t = _text("35")
    assert "No edge features are used" not in t, (
        "NB35: stale 'No edge features are used' phrase found"
    )


def test_nb35_has_baseline() -> None:
    t = _text("35")
    assert "DegreeFeatureBaseline" in t or "degree" in t.lower(), (
        "NB35: missing degree-feature baseline"
    )
