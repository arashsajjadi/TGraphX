"""Documentation honesty tests.

These tests catch obvious claim drift in README.md and docs/*.md without
enforcing prose too rigidly.  They are intentionally coarse: a failure
means a documented claim has diverged from reality and must be resolved
before release.

Checks:
- No "Fully supported" in hardware rows that are not in CI.
- No stale "not implemented" for APIs that ARE implemented.
- TensorGATLayer spatial edge feature claim is consistent (mean-pooled, not ✗).
- No "TensorBoard replacement" claim.
- No PyG/DGL compatibility claim.
- No SOTA / superiority claim.
- O(N²) builder runtime warnings are actually emitted above threshold.
- TensorGATLayer spatial edge features actually work (code matches docs).
- env_report returns expected keys.
- Top-level re-exports match __all__.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest
import torch

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
README = ROOT / "README.md"
LIMITATIONS = ROOT / "docs" / "limitations.md"
PERFORMANCE = ROOT / "docs" / "performance.md"
DASHBOARD = ROOT / "docs" / "dashboard.md"
COMPARISON = ROOT / "docs" / "comparison.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _readme() -> str:
    return _read(README)


def _all_docs() -> str:
    return "\n".join(
        _read(p)
        for p in [README, LIMITATIONS, PERFORMANCE, DASHBOARD, COMPARISON]
        if p.exists()
    )


# ── 1. No stale "Fully supported" for platforms without CI ───────────────────

class TestNoOverclaim:
    def test_windows_not_called_fully_supported(self):
        """Windows must not be labelled 'Fully supported' — it has no CI."""
        readme = _readme()
        lines = readme.splitlines()
        for line in lines:
            if "Windows" in line and "Fully supported" in line:
                pytest.fail(
                    f"README overclaims Windows as 'Fully supported' (no CI):\n  {line}"
                )

    def test_macos_not_called_fully_supported(self):
        readme = _readme()
        for line in readme.splitlines():
            if "macOS" in line and "Fully supported" in line:
                pytest.fail(
                    f"README overclaims macOS as 'Fully supported' (no CI):\n  {line}"
                )

    def test_no_sota_claim(self):
        """No positive state-of-the-art or superiority claims allowed.

        Denial phrases ("no SOTA claims", "does not claim SOTA", "make no ...
        claims") are fine and are excluded by the regex.
        """
        # Match positive assertions; skip lines that explicitly deny the claim.
        bad = re.compile(
            r"\b(?:is|are|achieves?|provides?|delivers?)\s+(?:the\s+)?state[- ]of[- ]the[- ]art\b"
            r"|\boutperforms\b"
            r"|\bbeats\s+(?:pyg|dgl|pytorch[- ]geometric)\b",
            re.IGNORECASE,
        )
        for doc_path in [README, COMPARISON]:
            if not doc_path.exists():
                continue
            text = _read(doc_path)
            for line in text.splitlines():
                if bad.search(line):
                    pytest.fail(
                        f"{doc_path.name}: SOTA/superiority claim found:\n  {line}"
                    )


# ── 2. No stale "not implemented" for implemented APIs ───────────────────────

class TestNoStaleNotImplemented:
    def test_train_epoch_not_claimed_missing(self):
        """train_epoch is implemented; no doc should say it is not."""
        text = _all_docs()
        stale = re.compile(
            r"train_epoch\b.*\bnot\s+implemented\b", re.IGNORECASE
        )
        assert not stale.search(text), (
            "A doc claims train_epoch is not implemented, but it is."
        )

    def test_evaluate_not_claimed_missing(self):
        text = _all_docs()
        stale = re.compile(r"evaluate\b.*\bnot\s+implemented\b", re.IGNORECASE)
        assert not stale.search(text)

    def test_fit_not_claimed_missing(self):
        text = _all_docs()
        stale = re.compile(r"\bfit\b.*\bnot\s+implemented\b", re.IGNORECASE)
        assert not stale.search(text)

    def test_tensorboard_logger_not_claimed_missing(self):
        """TensorBoardLogger is implemented (lazy import); docs must not say otherwise."""
        text = _all_docs()
        stale = re.compile(
            r"TensorBoardLogger\b.*\bnot\s+implemented\b", re.IGNORECASE
        )
        assert not stale.search(text), (
            "A doc claims TensorBoardLogger is not implemented, but it is."
        )

    def test_offline_export_not_claimed_missing(self):
        text = _all_docs()
        stale = re.compile(
            r"export[_-]?dashboard[_-]?html\b.*\bnot\s+implemented\b",
            re.IGNORECASE,
        )
        assert not stale.search(text)

    def test_write_graph_stats_not_claimed_missing(self):
        text = _all_docs()
        stale = re.compile(
            r"write_graph_stats\b.*\bnot\s+implemented\b", re.IGNORECASE
        )
        assert not stale.search(text)


# ── 3. TensorGATLayer spatial edge feature consistency ───────────────────────

class TestGATEdgeFeatureConsistency:
    def test_readme_does_not_mark_spatial_edges_unsupported_for_gat(self):
        """The edge-feature table must not mark TensorGATLayer spatial as plain ✗.

        Accepted forms: ⚠️, mean-pool, supported, etc.
        Plain ✗ with no qualifier would be a contradiction.
        """
        readme = _readme()
        lines = readme.splitlines()
        for i, line in enumerate(lines):
            if "TensorGATLayer" in line and "|" in line:
                # Look for a plain ✗ in the spatial column
                # The row has 3 | delimiters: | layer | vector | spatial |
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4:
                    spatial_col = parts[3] if len(parts) > 3 else ""
                    # Fail only if spatial column is literally "✗" with nothing else
                    if spatial_col.strip() in ("✗", "❌"):
                        pytest.fail(
                            f"README line {i+1}: TensorGATLayer spatial edge "
                            f"features are marked as plain ✗/❌, but the code "
                            f"accepts them (mean-pooled). Update the table.\n"
                            f"  Line: {line}"
                        )

    def test_code_accepts_spatial_edges_with_mean_pool(self):
        """Runtime: TensorGATLayer actually accepts spatial edge features."""
        from tgraphx.layers import TensorGATLayer

        torch.manual_seed(0)
        N, C, H, W = 4, 4, 4, 4
        x = torch.randn(N, C, H, W)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        E = edge_index.size(1)
        edge_dim = 2

        layer = TensorGATLayer(
            in_channels=C,
            out_channels=8,
            num_heads=2,
            use_edge_features=True,
            edge_dim=edge_dim,
            spatial_rank=2,
        )

        # Vector edge features — always works
        ef_vec = torch.randn(E, edge_dim)
        out_vec = layer(x, edge_index, edge_features=ef_vec)
        assert out_vec.shape == (N, 8, H, W)

        # Spatial edge features — must also work (mean-pooled internally)
        ef_spatial = torch.randn(E, edge_dim, H, W)
        out_spatial = layer(x, edge_index, edge_features=ef_spatial)
        assert out_spatial.shape == (N, 8, H, W)
        assert torch.isfinite(out_spatial).all()

    def test_code_accepts_volumetric_edges_with_mean_pool(self):
        """TensorGATLayer(spatial_rank=3) accepts volumetric edge features."""
        from tgraphx.layers import TensorGATLayer

        torch.manual_seed(1)
        N, C, D, H, W = 4, 4, 4, 4, 4
        x = torch.randn(N, C, D, H, W)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        E = edge_index.size(1)
        edge_dim = 2

        layer = TensorGATLayer(
            in_channels=C,
            out_channels=8,
            num_heads=2,
            use_edge_features=True,
            edge_dim=edge_dim,
            spatial_rank=3,
        )
        ef_vol = torch.randn(E, edge_dim, D, H, W)
        out = layer(x, edge_index, edge_features=ef_vol)
        assert out.shape == (N, 8, D, H, W)
        assert torch.isfinite(out).all()

    def test_mismatched_rank_raises_not_implemented(self):
        """5-D edges into a 2-D GAT must raise NotImplementedError."""
        from tgraphx.layers import TensorGATLayer

        torch.manual_seed(2)
        N, C, H, W = 4, 4, 4, 4
        x = torch.randn(N, C, H, W)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        E = edge_index.size(1)

        layer = TensorGATLayer(
            in_channels=C, out_channels=8, num_heads=2,
            use_edge_features=True, edge_dim=2, spatial_rank=2,
        )
        ef_5d = torch.randn(E, 2, 4, 4, 4)  # volumetric into 2-D GAT
        with pytest.raises(NotImplementedError):
            layer(x, edge_index, edge_features=ef_5d)


# ── 4. Dashboard is NOT a TensorBoard replacement ────────────────────────────

class TestDashboardClaims:
    def test_no_tensorboard_replacement_claim(self):
        """Dashboard docs must not claim to be a TensorBoard replacement."""
        text = _all_docs()
        bad = re.compile(
            r"(dashboard|tgraphx).*(?:is|as)\s+(?:a\s+)?tensorboard\s+replacement",
            re.IGNORECASE,
        )
        assert not bad.search(text), (
            "A doc claims the dashboard is a TensorBoard replacement."
        )

    def test_no_pyg_dgl_compat_claim(self):
        """No doc must positively claim PyG/DGL drop-in compatibility.

        Denial phrases ("is **not** a drop-in replacement for PyG") are
        acceptable and are excluded by requiring the claim to lack negation
        on the same line.
        """
        bad = re.compile(
            r"(?<!\bnot\s)(?<!\bnot\sa\s)drop[- ]?in\s+replacement\s+for\s+(pyg|pytorch[- ]?geometric|dgl)",
            re.IGNORECASE,
        )
        text = _all_docs()
        for line in text.splitlines():
            # Skip lines that explicitly say "not a drop-in replacement"
            if re.search(r"\bnot\b.*drop[- ]?in\s+replacement", line, re.IGNORECASE):
                continue
            if bad.search(line):
                pytest.fail(
                    f"A doc positively claims PyG/DGL drop-in compatibility:\n  {line}"
                )


# ── 5. O(N²) runtime warnings are actually emitted ───────────────────────────

class TestO2Warnings:
    def _large_coords(self, n):
        return torch.randn(n, 2)

    def test_knn_warns_large_n(self):
        from tgraphx import build_knn_graph
        N = 10_001
        coords = self._large_coords(N)
        with pytest.warns(UserWarning, match="O\\(N²\\)|O.N.2|num_nodes"):
            build_knn_graph(coords, k=3)

    def test_radius_warns_large_n(self):
        from tgraphx import build_radius_graph
        N = 10_001
        coords = self._large_coords(N)
        with pytest.warns(UserWarning, match="O\\(N²\\)|O.N.2|num_nodes"):
            build_radius_graph(coords, radius=0.1)

    def test_fully_connected_warns_large_n(self):
        from tgraphx import build_fully_connected_graph
        with pytest.warns(UserWarning, match="O\\(N²\\)|O.N.2|num_nodes"):
            build_fully_connected_graph(5_001)

    def test_iou_warns_large_n(self):
        from tgraphx import build_iou_graph
        N = 5_001
        boxes = torch.rand(N, 4)
        # make valid x1<x2, y1<y2
        boxes[:, 2] = boxes[:, 0] + 0.1
        boxes[:, 3] = boxes[:, 1] + 0.1
        with pytest.warns(UserWarning, match="O\\(N²\\)|O.N.2|num_nodes"):
            build_iou_graph(boxes, threshold=0.0)

    def test_knn_no_warn_small_n(self):
        from tgraphx import build_knn_graph
        coords = self._large_coords(100)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            build_knn_graph(coords, k=3)  # must not warn

    def test_fully_connected_no_warn_small_n(self):
        from tgraphx import build_fully_connected_graph
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            build_fully_connected_graph(50)  # must not warn


# ── 6. env_report returns expected keys ──────────────────────────────────────

class TestEnvReport:
    def test_env_report_has_required_keys(self):
        from tgraphx import env_report
        info = env_report()
        for key in ("python", "torch", "tgraphx", "cuda_available", "recommended_device"):
            assert key in info, f"env_report missing key: {key}"

    def test_env_report_hardware_flag(self):
        """include_hardware=True should not raise even without psutil."""
        from tgraphx import env_report
        try:
            info = env_report(include_hardware=True)
            assert isinstance(info, dict)
        except Exception as exc:
            pytest.fail(f"env_report(include_hardware=True) raised: {exc}")


# ── 7. Top-level exports match __all__ ────────────────────────────────────────

class TestTopLevelExports:
    def test_all_exports_importable(self):
        """Every symbol in tgraphx.__all__ must be accessible on the module."""
        import tgraphx
        for name in tgraphx.__all__:
            assert hasattr(tgraphx, name), f"tgraphx.{name} is in __all__ but not importable"

    def test_write_graph_stats_importable(self):
        from tgraphx import write_graph_stats  # noqa: F401

    def test_csv_logger_importable(self):
        from tgraphx import CSVLogger  # noqa: F401
        import tgraphx
        assert tgraphx.CSVLogger is CSVLogger

    def test_tensorboard_logger_importable_without_tensorboard(self):
        """TensorBoardLogger class must be importable even without tensorboard installed."""
        from tgraphx import TensorBoardLogger  # noqa: F401


# ── 8. ConvMessagePassing max aggregation works (not NotImplementedError) ────

class TestConvMaxAggregation:
    def test_conv_max_aggr_forward(self):
        """aggr='max' must forward without raising NotImplementedError."""
        from tgraphx.layers import ConvMessagePassing

        torch.manual_seed(0)
        N, C, H, W = 4, 4, 4, 4
        x = torch.randn(N, C, H, W)
        ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        layer = ConvMessagePassing(
            (C, H, W), (8, H, W), aggr="max",
            aggregator_params={"num_layers": 1, "use_batchnorm": False, "dropout_prob": 0.0},
        )
        out = layer(x, ei)
        assert out.shape == (N, 8, H, W)
        assert torch.isfinite(out).all()

    def test_conv_max_chunked_emits_warning_not_error(self):
        """chunk_size with aggr='max' must warn and succeed — not raise."""
        from tgraphx.layers import ConvMessagePassing

        torch.manual_seed(0)
        N, C, H, W = 4, 4, 4, 4
        x = torch.randn(N, C, H, W)
        ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        layer = ConvMessagePassing(
            (C, H, W), (8, H, W), aggr="max",
            aggregator_params={"num_layers": 1, "use_batchnorm": False, "dropout_prob": 0.0},
        )
        with pytest.warns(UserWarning, match="chunk_size"):
            out = layer(x, ei, chunk_size=2)
        assert out.shape == (N, 8, H, W)


# ── 9. v0.2.4 feature claim audit ─────────────────────────────────────────────

class TestV024FeatureClaims:
    """If README/docs claim a v0.2.4 feature exists, the code must back it up."""

    def test_gat_chunked_forward_actually_works(self):
        from tgraphx.layers import TensorGATLayer
        from tgraphx import build_grid_graph
        torch.manual_seed(0)
        x = torch.randn(9, 4, 4, 4)
        ei = build_grid_graph(3, 3, directed=False, self_loops=True)
        l = TensorGATLayer(4, 4, num_heads=2).eval()
        with torch.no_grad():
            full = l(x, ei)
            chunked = l(x, ei, chunk_size=5)
        assert torch.allclose(full, chunked, atol=1e-4)

    def test_gat_channel_attention_mode_actually_works(self):
        from tgraphx.layers import TensorGATLayer
        from tgraphx import build_grid_graph
        x = torch.randn(9, 4, 4, 4)
        ei = build_grid_graph(3, 3, directed=False, self_loops=True)
        l = TensorGATLayer(4, 4, num_heads=2, attention_mode="channel").eval()
        with torch.no_grad():
            out = l(x, ei)
        assert out.shape == (9, 4, 4, 4)
        assert torch.isfinite(out).all()

    def test_patch_padding_auto_actually_works(self):
        from tgraphx.graph_builders import image_to_patches
        imgs = torch.randn(2, 3, 9, 9)  # not divisible by 4
        patches = image_to_patches(imgs, patch_size=4, padding="auto")
        assert patches.shape[1] == 9  # 3x3 grid

    def test_mlflow_logger_class_importable(self):
        from tgraphx import MLflowLogger
        import inspect
        assert inspect.isclass(MLflowLogger)

    def test_pyg_dgl_converters_module_importable(self):
        """tgraphx.interop must be importable without PyG/DGL installed."""
        from tgraphx.interop import (
            to_pyg_data, from_pyg_data, to_dgl_graph, from_dgl_graph,
        )
        assert all(callable(f) for f in (to_pyg_data, from_pyg_data,
                                          to_dgl_graph, from_dgl_graph))

    def test_learned_graph_helpers_actually_work(self):
        from tgraphx.learned_graph import (
            soft_adjacency_from_embeddings,
            top_k_edges_from_scores,
            build_knn_graph_from_embeddings,
            EdgeScorer,
        )
        z = torch.randn(8, 16)
        A = soft_adjacency_from_embeddings(z)
        assert A.shape == (8, 8)
        ei, _ = top_k_edges_from_scores(A, k=3)
        assert ei.shape == (2, 24)

    def test_hetero_graph_container_actually_works(self):
        from tgraphx.core.hetero_graph import HeteroGraph
        g = HeteroGraph(
            node_stores={"a": torch.randn(3, 4)},
            edge_stores={
                ("a", "rel", "a"): torch.tensor(
                    [[0, 1], [1, 2]], dtype=torch.long
                )
            },
        )
        assert g.num_nodes("a") == 3

    def test_temporal_sequence_container_actually_works(self):
        from tgraphx.core.temporal import TemporalGraphSequence
        from tgraphx import Graph
        seq = TemporalGraphSequence(
            graphs=[Graph(torch.randn(3, 4), None) for _ in range(2)]
        )
        assert seq.num_snapshots == 2

    def test_graph_transformer_layer_actually_works(self):
        from tgraphx.layers.graph_transformer import GraphTransformerLayer
        l = GraphTransformerLayer(16, 16, num_heads=4).eval()
        x = torch.randn(8, 16)
        out = l(x)
        assert out.shape == (8, 16)
        assert torch.isfinite(out).all()


# ── 9b. v0.2.8 honesty: no stale "Planned v0.2.4" left in README ─────────────


class TestReadmeHonesty:
    def test_readme_does_not_mark_implemented_features_planned(self):
        """README must not call shipped features 'Planned v0.2.4'."""
        readme = _readme()
        # GAT chunked forward is shipped — must not be 'Planned' anywhere.
        for line in readme.splitlines():
            if "TensorGATLayer" in line and "chunked" in line:
                if re.search(r"Planned\s+v?0\.2\.4", line, re.IGNORECASE):
                    pytest.fail(
                        f"README still calls TensorGATLayer chunked forward "
                        f"'Planned v0.2.4' even though it ships:\n  {line}"
                    )

    def test_readme_per_channel_attention_not_marked_unsupported(self):
        """attention_mode='channel' is implemented — README must not call it
        'Not supported'."""
        readme = _readme()
        for line in readme.splitlines():
            if "Per-channel attention" in line and "❌ Not supported" in line:
                pytest.fail(
                    f"README marks per-channel attention as ❌ Not supported, "
                    f"but `attention_mode='channel'` is implemented:\n  {line}"
                )

    def test_readme_does_not_say_sampling_out_of_scope(self):
        readme = _readme()
        for line in readme.splitlines():
            if (
                "out of scope" in line.lower()
                and "sampling" in line.lower()
                and "neighbor" in line.lower()
            ):
                pytest.fail(
                    f"README still says neighbour sampling is out of scope, "
                    f"but tgraphx.sampling ships:\n  {line}"
                )

    def test_readme_does_not_say_hetero_temporal_only_containers(self):
        readme = _readme()
        for line in readme.splitlines():
            low = line.lower()
            if (
                "hetero" in low
                and "temporal" in low
                and "containers" in low
                and "not gnn implementations" in low
            ):
                pytest.fail(
                    f"README still calls hetero/temporal 'containers, not GNN "
                    f"implementations', but HeteroConv / classifiers ship:\n"
                    f"  {line}"
                )


# ── 9c. v0.2.8 sampling helpers actually work ────────────────────────────────


class TestV028SamplingClaims:
    def test_random_walk_sample_works(self):
        from tgraphx import Graph, random_walk_sample
        x = torch.randn(6, 4)
        ei = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
        sub = random_walk_sample(Graph(x, ei), torch.tensor([0]), 4, seed=0)
        assert sub.num_nodes >= 1
        assert sub.metadata["sampling"]["kind"] == "random_walk_sample"

    def test_hetero_sampling_works(self):
        from tgraphx import (
            HeteroGraph,
            hetero_induced_subgraph,
            hetero_neighbor_sample,
        )
        g = HeteroGraph(
            node_stores={
                "p": torch.randn(4, 4), "a": torch.randn(3, 2),
            },
            edge_stores={
                ("a", "writes", "p"): torch.tensor(
                    [[0, 1, 2], [0, 1, 2]], dtype=torch.long,
                ),
            },
        )
        sub = hetero_induced_subgraph(
            g, {"p": torch.tensor([0, 1]), "a": torch.tensor([0, 1])},
        )
        assert sub.num_nodes("p") == 2
        assert sub.num_nodes("a") == 2

        sampled = hetero_neighbor_sample(
            g,
            seed_nodes_dict={"p": torch.tensor([0])},
            fanouts=[{("a", "writes", "p"): 1}],
            seed=0, direction="in",
        )
        assert "sampling" in sampled.metadata

    def test_readme_has_no_scary_symbols(self):
        """v0.3.0 contract: the main README must not look like a warning board."""
        text = _read(README)
        bad = ["⚠️", "❌", "⛔", "⏳", "🧪", "🚫"]
        found = [b for b in bad if b in text]
        assert not found, (
            f"README contains scary status symbols: {found}.  "
            f"Move detailed limitations to docs/limitations.md and rephrase "
            f"calmly in README."
        )

    def test_readme_uses_calm_language(self):
        """No 'Best-effort'/'Out of scope'/'Planned'/'Deferred' wall-of-warning prose."""
        text = _read(README).lower()
        # We allow these words in CHANGELOG-style references inside docs/, but
        # not in the user-facing README body.
        bad = [
            "best-effort", "out of scope", "future release", "current release",
            "planned v0.2", "⏳ planned",
        ]
        found = [b for b in bad if b in text]
        assert not found, (
            f"README contains warning-board language: {found}. "
            f"Use calm, current-state wording; move details to docs/."
        )

    def test_temporal_window_sample_works(self):
        from tgraphx import (
            Graph, TemporalGraphSequence, TemporalGraphBatch,
            temporal_window_sample, temporal_window_sample_batch,
        )
        snaps = [Graph(torch.randn(4, 3), None) for _ in range(5)]
        seq = TemporalGraphSequence(graphs=snaps)
        sub = temporal_window_sample(seq, 1, 4)
        assert sub.num_snapshots == 3

        batch = TemporalGraphBatch([seq, seq])
        sub_batch = temporal_window_sample_batch(batch, 1, 4)
        assert sub_batch.num_sequences == 2


# ── 10. Lazy-import contracts for new optional integrations ──────────────────

class TestLazyImports:
    """Importing tgraphx must not pull in optional heavy deps."""

    def test_no_eager_mlflow(self):
        """mlflow must not be imported at tgraphx import time."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c",
             "import tgraphx, sys; "
             "assert 'mlflow' not in sys.modules, 'mlflow imported eagerly'"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_no_eager_torch_geometric(self):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c",
             "import tgraphx; import tgraphx.interop; import sys; "
             "assert 'torch_geometric' not in sys.modules"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_no_eager_dgl(self):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c",
             "import tgraphx; import tgraphx.interop; import sys; "
             "assert 'dgl' not in sys.modules"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
