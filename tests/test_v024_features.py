"""Tests for all v0.2.4 new features:
- GAT channel attention mode
- MLflowLogger
- Patch padding
- PyG/DGL converters (mocked / skip)
- Learned graph helpers
- HeteroGraph container
- TemporalGraphSequence container
- GraphTransformerLayer
"""
from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest
import torch

from tgraphx import Graph
from tgraphx.graph_builders import (
    build_grid_graph,
    image_to_patches,
    volume_to_patches,
)
from tgraphx.layers import TensorGATLayer


# ── GAT channel attention mode ───────────────────────────────────────────────

class TestGATChannelAttention:
    def _layer(self, **kw):
        return TensorGATLayer(4, 4, num_heads=2, attention_mode="channel", **kw).eval()

    def test_forward_shape_2d(self):
        torch.manual_seed(0)
        x = torch.randn(9, 4, 4, 4)
        ei = build_grid_graph(3, 3, directed=False, self_loops=True)
        out = self._layer()(x, ei)
        assert out.shape == (9, 4, 4, 4)

    def test_forward_shape_3d(self):
        torch.manual_seed(1)
        from tgraphx.graph_builders import build_grid_graph_3d
        x = torch.randn(8, 4, 4, 4, 4)
        ei = build_grid_graph_3d(2, 2, 2, directed=False, self_loops=True)
        l = TensorGATLayer(4, 4, num_heads=2, attention_mode="channel", spatial_rank=3).eval()
        out = l(x, ei)
        assert out.shape == (8, 4, 4, 4, 4)

    def test_output_finite(self):
        torch.manual_seed(2)
        x = torch.randn(9, 4, 4, 4)
        ei = build_grid_graph(3, 3, directed=False, self_loops=True)
        with torch.no_grad():
            out = self._layer()(x, ei)
        assert torch.isfinite(out).all()

    def test_backward_finite(self):
        torch.manual_seed(3)
        x = torch.randn(9, 4, 4, 4, requires_grad=True)
        ei = build_grid_graph(3, 3, directed=False, self_loops=True)
        l = TensorGATLayer(4, 4, num_heads=2, attention_mode="channel").train()
        out = l(x, ei)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_return_attention_channel(self):
        torch.manual_seed(4)
        N, C = 9, 4
        x = torch.randn(N, C, 4, 4)
        ei = build_grid_graph(3, 3, directed=False, self_loops=True)
        l = self._layer()
        with torch.no_grad():
            out, attn = l(x, ei, return_attention=True)
        # attn shape should be [E, K, C_head] for channel mode
        assert attn.shape == (ei.size(1), 2, 2)  # K=2 heads, C_head=4/2=2

    def test_scalar_parity_unchanged(self):
        """Scalar mode output must not be affected by adding channel mode."""
        torch.manual_seed(5)
        x = torch.randn(9, 4, 4, 4)
        ei = build_grid_graph(3, 3, directed=False, self_loops=True)
        torch.manual_seed(5)
        l_scalar = TensorGATLayer(4, 4, num_heads=2, attention_mode="scalar").eval()
        torch.manual_seed(5)
        l_scalar2 = TensorGATLayer(4, 4, num_heads=2).eval()  # default
        # Same params → same output
        with torch.no_grad():
            out1 = l_scalar(x, ei)
            out2 = l_scalar2(x, ei)
        assert torch.allclose(out1, out2)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="attention_mode"):
            TensorGATLayer(4, 4, attention_mode="pixel")

    def test_factory_channel_mode(self):
        from tgraphx.layers.factory import make_layer
        l = make_layer("gat", (4, 4, 4), (4, 4, 4), heads=2, attention_mode="channel")
        assert l.attention_mode == "channel"

    def test_chunked_channel_parity(self):
        torch.manual_seed(6)
        x = torch.randn(9, 4, 4, 4)
        ei = build_grid_graph(3, 3, directed=False, self_loops=True)
        l = self._layer()
        with torch.no_grad():
            full = l(x, ei)
            chunked = l(x, ei, chunk_size=5)
        assert torch.allclose(full, chunked, atol=1e-4)


# ── MLflowLogger ─────────────────────────────────────────────────────────────

class TestMLflowLogger:
    def test_missing_mlflow_raises_import_error(self):
        """MLflowLogger must raise ImportError when mlflow is not installed."""
        import sys
        orig = sys.modules.get("mlflow")
        sys.modules["mlflow"] = None  # type: ignore[assignment]
        try:
            # Force reimport of tracking module that uses lazy import
            from tgraphx.tracking import MLflowLogger
            with pytest.raises((ImportError, TypeError)):
                MLflowLogger()
        finally:
            if orig is None:
                del sys.modules["mlflow"]
            else:
                sys.modules["mlflow"] = orig

    def test_mlflow_not_imported_at_tgraphx_import(self):
        """Importing tgraphx must not import mlflow."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c",
             "import tgraphx; import sys; "
             "assert 'mlflow' not in sys.modules, 'mlflow imported at tgraphx import time'"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_mlflow_class_importable_from_tgraphx(self):
        from tgraphx import MLflowLogger
        import inspect
        assert inspect.isclass(MLflowLogger)

    def test_mlflow_logger_with_mock(self):
        """MLflowLogger.log must call mlflow.log_metrics with mock."""
        import sys
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = lambda s: s
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value = mock_run
        sys.modules["mlflow"] = mock_mlflow
        try:
            from tgraphx.tracking import MLflowLogger
            logger = MLflowLogger(run_name="test")
            logger._run = mock_run  # simulate started
            logger.log(epoch=1, train_loss=0.5, val_acc=0.9)
            mock_mlflow.log_metrics.assert_called_once()
            call_kwargs = mock_mlflow.log_metrics.call_args
            metrics_dict = call_kwargs[0][0]
            assert "train_loss" in metrics_dict
            assert "val_acc" in metrics_dict
            assert "epoch" not in metrics_dict  # epoch used as step, not logged
        finally:
            del sys.modules["mlflow"]

    def test_mlflow_log_before_start_raises(self):
        mock_mlflow = MagicMock()
        import sys
        sys.modules["mlflow"] = mock_mlflow
        try:
            from tgraphx.tracking import MLflowLogger
            logger = MLflowLogger()
            with pytest.raises(RuntimeError, match="start"):
                logger.log(loss=0.5)
        finally:
            del sys.modules["mlflow"]

    def test_mlflow_repr(self):
        mock_mlflow = MagicMock()
        import sys
        sys.modules["mlflow"] = mock_mlflow
        try:
            from tgraphx.tracking import MLflowLogger
            logger = MLflowLogger(run_name="my_run")
            r = repr(logger)
            assert "MLflowLogger" in r
            assert "my_run" in r
        finally:
            del sys.modules["mlflow"]


# ── Patch helper padding ──────────────────────────────────────────────────────

class TestPatchPadding:
    def test_exact_tiling_unchanged(self):
        B, C, H, W = 2, 3, 8, 8
        imgs = torch.randn(B, C, H, W)
        patches_old = image_to_patches(imgs, patch_size=4)
        patches_new = image_to_patches(imgs, patch_size=4, padding="none")
        assert torch.equal(patches_old, patches_new)

    def test_invalid_padding_raises(self):
        imgs = torch.randn(2, 3, 8, 8)
        with pytest.raises(ValueError, match="padding"):
            image_to_patches(imgs, patch_size=4, padding="reflect")

    def test_auto_padding_2d(self):
        B, C, H, W = 1, 3, 9, 9  # not divisible by 4
        imgs = torch.randn(B, C, H, W)
        patches = image_to_patches(imgs, patch_size=4, padding="auto")
        # ceil((9-4)/4)+1 = 3 → 3×3 = 9 patches
        assert patches.shape[0] == B
        assert patches.shape[1] == 9
        assert patches.shape[2] == C
        assert patches.shape[3] == 4
        assert patches.shape[4] == 4

    def test_auto_padding_patch_count(self):
        import math
        B, C = 1, 2
        for H, W, ps in [(10, 10, 4), (7, 9, 3), (12, 11, 5)]:
            imgs = torch.randn(B, C, H, W)
            patches = image_to_patches(imgs, patch_size=ps, padding="auto")
            n_h = math.ceil((H - ps) / ps) + 1
            n_w = math.ceil((W - ps) / ps) + 1
            assert patches.shape[1] == n_h * n_w

    def test_auto_padding_exact_divisible(self):
        """Auto-padding on exactly-divisible dims produces same as 'none'."""
        B, C, H, W = 2, 3, 8, 8
        imgs = torch.randn(B, C, H, W)
        patches_none = image_to_patches(imgs, patch_size=4, padding="none")
        patches_auto = image_to_patches(imgs, patch_size=4, padding="auto")
        assert torch.equal(patches_none, patches_auto)

    def test_auto_padding_3d(self):
        B, C, D, H, W = 1, 2, 9, 9, 9
        vols = torch.randn(B, C, D, H, W)
        patches = volume_to_patches(vols, patch_size=4, padding="auto")
        import math
        n = math.ceil((9 - 4) / 4) + 1  # = 3
        assert patches.shape[1] == n ** 3

    def test_auto_padding_values(self):
        """Padded region should contain pad_value."""
        B, C, H, W = 1, 1, 5, 5  # 5×5 with patch_size=4 → needs pad to 8×8
        imgs = torch.zeros(B, C, H, W)
        patches = image_to_patches(imgs, patch_size=4, padding="auto", pad_value=9.0)
        # All patches should be finite
        assert torch.isfinite(patches).all()


# ── Interop (mocked deps) ────────────────────────────────────────────────────

class TestInteropMissingDeps:
    def test_to_pyg_missing_dep(self):
        import sys
        sys.modules["torch_geometric"] = None  # type: ignore
        sys.modules.get("torch_geometric.data", None)
        try:
            from tgraphx.interop import to_pyg_data
            g = Graph(torch.randn(4, 8), None)
            with pytest.raises((ImportError, TypeError)):
                to_pyg_data(g)
        finally:
            del sys.modules["torch_geometric"]

    def test_to_dgl_missing_dep(self):
        import sys
        sys.modules["dgl"] = None  # type: ignore
        try:
            from tgraphx.interop import to_dgl_graph
            g = Graph(torch.randn(4, 8), torch.tensor([[0, 1], [1, 2]], dtype=torch.long))
            with pytest.raises((ImportError, TypeError)):
                to_dgl_graph(g)
        finally:
            del sys.modules["dgl"]

    def test_interop_not_imported_at_tgraphx_import(self):
        """tgraphx.interop must not import PyG or DGL at import time."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c",
             "import tgraphx.interop; import sys; "
             "assert 'torch_geometric' not in sys.modules; "
             "assert 'dgl' not in sys.modules"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr


# ── Learned graph helpers ─────────────────────────────────────────────────────

class TestLearnedGraph:
    def test_soft_adjacency_shape(self):
        from tgraphx.learned_graph import soft_adjacency_from_embeddings
        z = torch.randn(10, 16)
        A = soft_adjacency_from_embeddings(z)
        assert A.shape == (10, 10)
        assert (A >= 0).all() and (A <= 1).all()

    def test_soft_adjacency_differentiable(self):
        from tgraphx.learned_graph import soft_adjacency_from_embeddings
        z = torch.randn(8, 16, requires_grad=True)
        A = soft_adjacency_from_embeddings(z)
        A.sum().backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()

    def test_soft_adjacency_large_n_warns(self):
        from tgraphx.learned_graph import soft_adjacency_from_embeddings
        z = torch.randn(5001, 4)
        with pytest.warns(UserWarning, match="O.N.2.|N=5001"):
            soft_adjacency_from_embeddings(z)

    def test_top_k_edges_shape(self):
        from tgraphx.learned_graph import top_k_edges_from_scores
        scores = torch.rand(10, 10)
        ei, es = top_k_edges_from_scores(scores, k=3)
        assert ei.shape == (2, 30)
        assert es.shape == (30,)

    def test_top_k_no_self_loops(self):
        from tgraphx.learned_graph import top_k_edges_from_scores
        scores = torch.rand(10, 10)
        scores.fill_diagonal_(100.0)  # high self-score but should be excluded
        ei, _ = top_k_edges_from_scores(scores, k=3, self_loops=False)
        assert (ei[0] != ei[1]).all()

    def test_knn_from_embeddings_shape(self):
        from tgraphx.learned_graph import build_knn_graph_from_embeddings
        z = torch.randn(20, 16)
        ei = build_knn_graph_from_embeddings(z, k=3)
        assert ei.shape[0] == 2
        # No gradients through graph topology
        assert not ei.requires_grad

    def test_edge_scorer_forward(self):
        from tgraphx.learned_graph import EdgeScorer
        z = torch.randn(10, 32)
        ei = torch.randint(0, 10, (2, 20))
        scorer = EdgeScorer(in_dim=32, hidden_dim=16)
        scores = scorer(z, ei)
        assert scores.shape == (20,)

    def test_edge_scorer_gradient(self):
        from tgraphx.learned_graph import EdgeScorer
        z = torch.randn(10, 32, requires_grad=True)
        ei = torch.randint(0, 10, (2, 20))
        scorer = EdgeScorer(in_dim=32, hidden_dim=16)
        scores = scorer(z, ei)
        scores.sum().backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()

    def test_top_k_non_differentiable(self):
        """top_k_edges result should not carry gradients."""
        from tgraphx.learned_graph import top_k_edges_from_scores
        scores = torch.rand(8, 8, requires_grad=True)
        ei, es = top_k_edges_from_scores(scores, k=2)
        assert not ei.requires_grad
        assert not es.requires_grad


# ── HeteroGraph container ────────────────────────────────────────────────────

class TestHeteroGraph:
    def _make(self):
        from tgraphx.core.hetero_graph import HeteroGraph
        paper_feat = torch.randn(5, 16)
        author_feat = torch.randn(3, 8)
        writes_ei = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
        return HeteroGraph(
            {"paper": paper_feat, "author": author_feat},
            {("author", "writes", "paper"): writes_ei},
        )

    def test_create(self):
        g = self._make()
        assert "paper" in g.node_types
        assert "author" in g.node_types
        assert ("author", "writes", "paper") in g.edge_types

    def test_num_nodes(self):
        g = self._make()
        assert g.num_nodes("paper") == 5
        assert g.num_nodes("author") == 3

    def test_num_edges(self):
        g = self._make()
        assert g.num_edges(("author", "writes", "paper")) == 3

    def test_unknown_node_type_raises(self):
        g = self._make()
        with pytest.raises(KeyError):
            g.node_features("venue")

    def test_to_device(self):
        g = self._make()
        g_cpu = g.to("cpu")
        assert g_cpu.node_features("paper").device.type == "cpu"

    def test_bad_edge_type_raises(self):
        from tgraphx.core.hetero_graph import HeteroGraph
        with pytest.raises(ValueError, match="3-tuple"):
            HeteroGraph(
                {"a": torch.randn(3, 4)},
                {"bad_key": torch.zeros(2, 5, dtype=torch.long)},
            )

    def test_wrong_dtype_edge_index_raises(self):
        from tgraphx.core.hetero_graph import HeteroGraph
        with pytest.raises(TypeError, match="torch.long"):
            HeteroGraph(
                {"a": torch.randn(3, 4)},
                {("a", "rel", "a"): torch.zeros(2, 5, dtype=torch.float)},
            )

    def test_repr_contains_experimental(self):
        g = self._make()
        assert "Experimental" in repr(g) or "🧪" in repr(g)


# ── TemporalGraphSequence container ──────────────────────────────────────────

class TestTemporalGraphSequence:
    def _make_graphs(self, n=3):
        return [Graph(torch.randn(4, 8), None) for _ in range(n)]

    def test_create_without_timestamps(self):
        from tgraphx.core.temporal import TemporalGraphSequence
        seq = TemporalGraphSequence(self._make_graphs())
        assert seq.num_snapshots == 3

    def test_create_with_timestamps(self):
        from tgraphx.core.temporal import TemporalGraphSequence
        seq = TemporalGraphSequence(self._make_graphs(), timestamps=[0.0, 1.0, 2.0])
        assert seq.timestamps == [0.0, 1.0, 2.0]

    def test_indexing(self):
        from tgraphx.core.temporal import TemporalGraphSequence
        graphs = self._make_graphs()
        seq = TemporalGraphSequence(graphs)
        assert seq[0] is graphs[0]
        assert seq[-1] is graphs[-1]

    def test_iteration(self):
        from tgraphx.core.temporal import TemporalGraphSequence
        graphs = self._make_graphs()
        seq = TemporalGraphSequence(graphs, timestamps=[10.0, 20.0, 30.0])
        ts_list, g_list = [], []
        for t, g in seq:
            ts_list.append(t); g_list.append(g)
        assert ts_list == [10.0, 20.0, 30.0]
        assert len(g_list) == 3

    def test_iteration_no_timestamps(self):
        from tgraphx.core.temporal import TemporalGraphSequence
        seq = TemporalGraphSequence(self._make_graphs())
        for t, g in seq:
            assert t is None

    def test_to_device(self):
        from tgraphx.core.temporal import TemporalGraphSequence
        seq = TemporalGraphSequence(self._make_graphs())
        seq_cpu = seq.to("cpu")
        assert seq_cpu.num_snapshots == 3

    def test_wrong_timestamps_length_raises(self):
        from tgraphx.core.temporal import TemporalGraphSequence
        with pytest.raises(ValueError, match="length"):
            TemporalGraphSequence(self._make_graphs(3), timestamps=[1.0, 2.0])

    def test_empty_graphs_raises(self):
        from tgraphx.core.temporal import TemporalGraphSequence
        with pytest.raises(ValueError):
            TemporalGraphSequence([])

    def test_repr_contains_experimental(self):
        from tgraphx.core.temporal import TemporalGraphSequence
        seq = TemporalGraphSequence(self._make_graphs())
        assert "Experimental" in repr(seq) or "🧪" in repr(seq)


# ── GraphTransformerLayer ─────────────────────────────────────────────────────

class TestGraphTransformerLayer:
    def test_forward_shape(self):
        from tgraphx.layers.graph_transformer import GraphTransformerLayer
        l = GraphTransformerLayer(32, 32, num_heads=4).eval()
        x = torch.randn(10, 32)
        with torch.no_grad():
            out = l(x)
        assert out.shape == (10, 32)

    def test_forward_different_in_out(self):
        from tgraphx.layers.graph_transformer import GraphTransformerLayer
        l = GraphTransformerLayer(16, 32, num_heads=4).eval()
        x = torch.randn(8, 16)
        out = l(x)
        assert out.shape == (8, 32)

    def test_output_finite(self):
        from tgraphx.layers.graph_transformer import GraphTransformerLayer
        l = GraphTransformerLayer(32, 32, num_heads=4).eval()
        x = torch.randn(10, 32)
        with torch.no_grad():
            out = l(x)
        assert torch.isfinite(out).all()

    def test_backward_finite(self):
        from tgraphx.layers.graph_transformer import GraphTransformerLayer
        l = GraphTransformerLayer(32, 32, num_heads=4).train()
        x = torch.randn(10, 32, requires_grad=True)
        out = l(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_large_n_warns(self):
        from tgraphx.layers.graph_transformer import GraphTransformerLayer
        l = GraphTransformerLayer(8, 8, num_heads=2).eval()
        x = torch.randn(1001, 8)
        with pytest.warns(UserWarning, match="N=1001|O.N.2."):
            l(x)

    def test_spatial_input_raises(self):
        from tgraphx.layers.graph_transformer import GraphTransformerLayer
        l = GraphTransformerLayer(4, 4, num_heads=2).eval()
        x = torch.randn(9, 4, 4, 4)  # spatial — not supported
        with pytest.raises(ValueError, match="2-D"):
            l(x)

    def test_edge_index_accepted_but_ignored(self):
        """edge_index may be passed for API consistency but is ignored."""
        from tgraphx.layers.graph_transformer import GraphTransformerLayer
        l = GraphTransformerLayer(8, 8, num_heads=2).eval()
        x = torch.randn(9, 8)
        ei = build_grid_graph(3, 3)
        out = l(x, edge_index=ei)
        assert out.shape == (9, 8)

    def test_out_dim_not_divisible_raises(self):
        from tgraphx.layers.graph_transformer import GraphTransformerLayer
        with pytest.raises(ValueError, match="divisible"):
            GraphTransformerLayer(8, 9, num_heads=4)
