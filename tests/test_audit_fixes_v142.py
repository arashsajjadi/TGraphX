"""Regression tests for the v1.4.2 audit-fix release.

Each test pins down one of the validated audit findings so it cannot
silently regress.  Tests prefer synthetic inputs and CPU-only behaviour
so they stay deterministic and offline-safe in CI.
"""
from __future__ import annotations

import subprocess
import sys

import pytest
import torch

import tgraphx
from tgraphx import Graph


# ── Codex/Composer TGX-AUDIT-002 / 001: make_graph + NetworkX  ──────────────


def _require_nx():
    try:
        import networkx  # noqa: F401
    except ImportError:
        pytest.skip("networkx not installed")
    import networkx as nx
    return nx


def test_make_graph_networkx_preserves_external_features() -> None:
    nx = _require_nx()
    G = nx.path_graph(3)
    x = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    g = tgraphx.make_graph(x=x, networkx_graph=G)
    assert g.node_features.shape == (3, 5)
    # Original tensor content survives (no zero placeholder).
    assert torch.equal(g.node_features, x)
    # Topology is non-trivial (path 0-1-2, undirected => 4 directed edges).
    assert g.num_edges == 4


def test_make_graph_networkx_preserves_labels() -> None:
    nx = _require_nx()
    G = nx.path_graph(3)
    y = torch.tensor([1, 0, 2], dtype=torch.long)
    g = tgraphx.make_graph(x=torch.zeros(3, 4), networkx_graph=G, labels=y)
    assert g.node_labels is not None
    assert torch.equal(g.node_labels, y)


def test_make_graph_networkx_rejects_shape_mismatch() -> None:
    nx = _require_nx()
    G = nx.path_graph(3)
    with pytest.raises(ValueError, match="rows but NetworkX graph"):
        tgraphx.make_graph(x=torch.zeros(4, 2), networkx_graph=G)


# ── Codex TGX-AUDIT-001 / Composer TGX-AUDIT-005: public_api registry  ──────


def test_public_api_registry_includes_v141_helpers() -> None:
    expected = {
        "classify_nodes",
        "kg_completion",
        "make_graph",
        "generate_graph",
        "optimize_graph",
        "train_graph_rl",
        "audit_package_readiness",
    }
    for name in expected:
        # Should not raise; must report a non-empty stability level.
        level = tgraphx.api_status(name)
        assert level, f"api_status({name!r}) returned empty level"


def test_public_api_registry_recognizes_v141_aliases() -> None:
    # `build_graph` is an alias of `make_graph`; api_status must resolve it.
    level = tgraphx.api_status("build_graph")
    assert level, "api_status('build_graph') returned empty level"
    aliases = tgraphx.list_aliases("make_graph")
    assert "build_graph" in aliases, aliases


# ── Codex TGX-AUDIT-003 / Composer TGX-AUDIT-004 / 010: top-level __all__  ──


def test_v141_aliases_in_top_level_all() -> None:
    expected = {
        "generate", "graph_generation_report", "compare_generated_graphs",
        "generation_metrics", "graph_evolution", "run_evolution", "run_rl",
        # Composer TGX-AUDIT-010 additions
        "KnowledgeGraph", "KGTrainer", "KGTrainingConfig",
        "run_graph_generation", "run_evolutionary_optimization", "run_graph_rl",
    }
    missing = expected.difference(tgraphx.__all__)
    assert not missing, f"Missing from tgraphx.__all__: {sorted(missing)}"


# ── Codex TGX-AUDIT-005: no fake `tgraphx[vision]` extra  ───────────────────


def test_optional_dependency_messages_reference_existing_extras() -> None:
    msg = tgraphx.explain_error("missing optional torchvision")
    assert "tgraphx[vision]" not in msg, (
        "explain_error must not suggest a non-existing tgraphx[vision] extra "
        "(torchvision is a mandatory dependency)."
    )


# ── Codex TGX-AUDIT-009: serialization preserves edge_labels & graph_features


def test_graph_save_load_preserves_edge_labels_and_graph_features(tmp_path) -> None:
    ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    g = Graph(
        node_features=torch.randn(3, 4),
        edge_index=ei,
        edge_labels=torch.tensor([7, 9]),
        graph_features=torch.tensor([1.0, 2.0, 3.0]),
    )
    p = tmp_path / "g.tgx"
    g.save(p)
    loaded = Graph.load(p)
    assert torch.equal(loaded.edge_labels, g.edge_labels)
    assert torch.equal(loaded.graph_features, g.graph_features)


def test_graph_save_load_backward_compat_missing_new_fields(tmp_path) -> None:
    """An old-format payload without the new keys must still load."""
    from tgraphx.ux.serialization import _payload_to_graph

    payload = {
        "node_features": torch.zeros(2, 3),
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
    }
    g = _payload_to_graph(payload)
    assert g.num_nodes == 2 and g.num_edges == 1
    assert g.edge_labels is None
    assert g.graph_features is None


# ── Codex TGX-AUDIT-010: training _unpack_batch with edge_index=None  ───────


def test_unpack_batch_raises_clear_error_for_none_edge_index() -> None:
    from tgraphx.training import _unpack_batch
    from tgraphx import Graph, GraphBatch

    # Build a batch whose edge_index is None (manually patched after creation).
    g = Graph(node_features=torch.zeros(3, 4),
              edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
              y=torch.tensor([0, 1, 0]))
    batch = GraphBatch([g])
    batch.edge_index = None
    with pytest.raises(ValueError, match="edge_index is None"):
        _unpack_batch(batch, torch.device("cpu"))


# ── Codex TGX-AUDIT-011: Graph.num_classes on empty labels  ─────────────────


def test_num_classes_empty_labels_returns_zero() -> None:
    g = Graph(node_features=torch.zeros(0, 4),
              edge_index=torch.zeros(2, 0, dtype=torch.long),
              y=torch.zeros(0, dtype=torch.long))
    assert g.num_classes == 0


# ── Codex TGX-AUDIT-012: from_adjacency rejects non-square sparse adjacency


def test_from_adjacency_rejects_nonsquare_sparse() -> None:
    try:
        import scipy.sparse as sp
    except ImportError:
        pytest.skip("scipy not installed")
    adj = sp.csr_matrix([[1, 0, 0], [0, 1, 0]])  # 2x3, not square
    with pytest.raises(ValueError, match="square"):
        Graph.from_adjacency(adj)


# ── Codex TGX-AUDIT-013: Graph.to() also moves masks in metadata  ───────────


def test_graph_to_moves_masks_in_metadata() -> None:
    g = Graph(
        node_features=torch.zeros(3, 4),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        metadata={"masks": {"train": torch.tensor([True, False, True])}},
    )
    # CPU->CPU is the only universally-available transition; we assert that
    # Graph.to() touches the mask without raising and the dtype/values stay.
    g.to(torch.device("cpu"))
    m = g.train_mask
    assert m is not None
    assert m.device.type == "cpu"
    assert m.tolist() == [True, False, True]


# ── Codex TGX-AUDIT-016: train_graph_rl forwards max_steps  ─────────────────


def test_train_graph_rl_max_steps_is_recorded() -> None:
    result = tgraphx.train_graph_rl(
        env="max_cut",
        algorithm="random",
        episodes=2,
        max_steps=7,
        num_nodes=6,
        seed=0,
        fast_mode=False,
    )
    config = getattr(result, "config", None)
    assert isinstance(config, dict)
    assert config.get("max_steps") == 7


# ── Composer TGX-AUDIT-003: lazy torchvision import  ────────────────────────


def test_torchvision_not_imported_eagerly() -> None:
    code = (
        "import sys, tgraphx;"
        "import json; print(json.dumps({"
        "'torchvision': 'torchvision' in sys.modules,"
        "'torchvision_models': 'torchvision.models' in sys.modules}))"
    )
    out = subprocess.check_output([sys.executable, "-c", code]).decode()
    last = out.strip().splitlines()[-1]
    import json
    info = json.loads(last)
    assert info["torchvision"] is False, (
        "`import tgraphx` should not eagerly import torchvision."
    )
    assert info["torchvision_models"] is False


# ── Composer TGX-AUDIT-007: CLI list-methods regression  ────────────────────


def test_cli_list_methods() -> None:
    res = subprocess.run(
        [sys.executable, "-m", "tgraphx", "list-methods"],
        capture_output=True, text=True, timeout=60,
    )
    assert res.returncode == 0, f"stderr={res.stderr}"
    # `list-methods` enumerates graph-generation methods registered by the
    # generation subsystem. Two stable methods we ship since v0.7 are ER and BA.
    assert "erdos_renyi" in res.stdout, res.stdout
    assert "barabasi_albert" in res.stdout, res.stdout


# ── Composer TGX-AUDIT-015: readiness dependency classification  ────────────


def test_readiness_dependency_classification() -> None:
    report = tgraphx.audit_package_readiness()
    assert "required_dependencies" in report
    # torchvision must be in required, not optional, per pyproject.toml.
    assert "torchvision" in report["required_dependencies"]
    assert "torchvision" not in report.get("optional_dependencies", {})
