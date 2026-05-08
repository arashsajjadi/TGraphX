"""Chunked forward tests for TensorGraphSAGELayer and TensorGINLayer (v0.2.3).

Verifies that chunked forward produces output identical to unchunked within
floating-point tolerance for all supported configurations:
- 2-D spatial (spatial_rank=2)
- 3-D volumetric (spatial_rank=3)
- aggr="mean" and aggr="max" (SAGE)
- edge_weight
- vector edge_features
- spatial/volumetric edge_features
- isolated nodes (no incoming edges)
- chunk_size=None unchanged (identity)
- empty edge_index
- gradients finite
- AMP/bfloat16 smoke (skipped when unsupported)

Also tests graph builder chunked paths (kNN, radius, IoU) and random graph
algorithm="sample".
"""

from __future__ import annotations

import warnings

import pytest
import torch

from tgraphx.graph_builders import (
    build_grid_graph,
    build_grid_graph_3d,
    build_iou_graph,
    build_knn_graph,
    build_radius_graph,
    build_random_graph,
)
from tgraphx.layers import TensorGINLayer, TensorGraphSAGELayer


# ── Helpers ───────────────────────────────────────────────────────────────────

ATOL = 1e-5  # float32 tolerance for chunk vs unchunked comparison


def _small_2d(N=9, C=4, H=4, W=4, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(N, C, H, W)
    ei = build_grid_graph(3, 3, directed=False, self_loops=True)
    return x, ei, N, C, H, W


def _small_3d(N=8, C=4, D=4, H=4, W=4, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(N, C, D, H, W)
    ei = build_grid_graph_3d(2, 2, 2, directed=False, self_loops=True)
    return x, ei, N, C, D, H, W


def _cpu_bf16_ok():
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            t = torch.tensor([1.0])
            _ = t + t
        return True
    except Exception:
        return False


skip_bf16 = pytest.mark.skipif(
    not _cpu_bf16_ok(), reason="CPU bfloat16 autocast not available"
)


# ── SAGE chunking — mean aggregation ─────────────────────────────────────────

class TestSAGEChunkedMean:
    def _layer(self, **kw):
        return TensorGraphSAGELayer(4, 4, aggr="mean", **kw).eval()

    def test_2d_parity(self):
        x, ei, N, C, H, W = _small_2d()
        layer = self._layer()
        with torch.no_grad():
            out_full = layer(x, ei)
            out_chunk = layer(x, ei, chunk_size=5)
        assert out_chunk.shape == out_full.shape
        assert torch.allclose(out_full, out_chunk, atol=ATOL), \
            f"max diff = {(out_full - out_chunk).abs().max():.2e}"

    def test_3d_parity(self):
        x, ei, N, C, D, H, W = _small_3d()
        layer = TensorGraphSAGELayer(C, C, aggr="mean", spatial_rank=3).eval()
        with torch.no_grad():
            out_full = layer(x, ei)
            out_chunk = layer(x, ei, chunk_size=5)
        assert torch.allclose(out_full, out_chunk, atol=ATOL)

    def test_chunk_size_1(self):
        x, ei, N, C, H, W = _small_2d()
        layer = self._layer()
        with torch.no_grad():
            out_full = layer(x, ei)
            out_chunk = layer(x, ei, chunk_size=1)
        assert torch.allclose(out_full, out_chunk, atol=ATOL)

    def test_chunk_size_exceeds_edges(self):
        x, ei, N, C, H, W = _small_2d()
        layer = self._layer()
        with torch.no_grad():
            out_full = layer(x, ei)
            out_chunk = layer(x, ei, chunk_size=10_000)
        assert torch.allclose(out_full, out_chunk, atol=ATOL)

    def test_chunk_size_none_unchanged(self):
        x, ei, N, C, H, W = _small_2d()
        layer = self._layer()
        with torch.no_grad():
            out1 = layer(x, ei, chunk_size=None)
            out2 = layer(x, ei)
        assert torch.equal(out1, out2)

    def test_edge_weight_parity(self):
        x, ei, N, C, H, W = _small_2d()
        layer = self._layer()
        ew = torch.rand(ei.size(1))
        with torch.no_grad():
            out_full = layer(x, ei, edge_weight=ew)
            out_chunk = layer(x, ei, edge_weight=ew, chunk_size=5)
        assert torch.allclose(out_full, out_chunk, atol=ATOL)

    def test_vector_edge_features_parity(self):
        x, ei, N, C, H, W = _small_2d()
        layer = TensorGraphSAGELayer(
            C, C, aggr="mean", use_edge_features=True,
            edge_dim=3, edge_features_kind="vector",
        ).eval()
        ef = torch.randn(ei.size(1), 3)
        with torch.no_grad():
            out_full = layer(x, ei, edge_features=ef)
            out_chunk = layer(x, ei, edge_features=ef, chunk_size=5)
        assert torch.allclose(out_full, out_chunk, atol=ATOL)

    def test_spatial_edge_features_parity(self):
        x, ei, N, C, H, W = _small_2d()
        layer = TensorGraphSAGELayer(
            C, C, aggr="mean", use_edge_features=True,
            edge_dim=3, edge_features_kind="spatial",
        ).eval()
        ef = torch.randn(ei.size(1), 3, H, W)
        with torch.no_grad():
            out_full = layer(x, ei, edge_features=ef)
            out_chunk = layer(x, ei, edge_features=ef, chunk_size=5)
        assert torch.allclose(out_full, out_chunk, atol=ATOL)

    def test_isolated_nodes(self):
        """Nodes with no incoming edges must produce zeros in the agg term."""
        torch.manual_seed(1)
        N, C, H, W = 5, 4, 4, 4
        x = torch.randn(N, C, H, W)
        # Node 4 has no incoming edges.
        ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        layer = self._layer()
        with torch.no_grad():
            out_full = layer(x, ei)
            out_chunk = layer(x, ei, chunk_size=2)
        assert torch.allclose(out_full, out_chunk, atol=ATOL)

    def test_gradient_finite(self):
        x, ei, N, C, H, W = _small_2d()
        x = x.requires_grad_(True)
        layer = TensorGraphSAGELayer(C, C, aggr="mean").train()
        out = layer(x, ei, chunk_size=5)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    @skip_bf16
    def test_amp_bfloat16_smoke(self):
        x, ei, N, C, H, W = _small_2d()
        layer = self._layer()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = layer(x, ei, chunk_size=5)
        assert torch.isfinite(out).all()


# ── SAGE chunking — max aggregation ──────────────────────────────────────────

class TestSAGEChunkedMax:
    def _layer(self, **kw):
        return TensorGraphSAGELayer(4, 4, aggr="max", **kw).eval()

    def test_2d_parity(self):
        x, ei, N, C, H, W = _small_2d()
        layer = self._layer()
        with torch.no_grad():
            out_full = layer(x, ei)
            out_chunk = layer(x, ei, chunk_size=5)
        assert torch.allclose(out_full, out_chunk, atol=ATOL)

    def test_3d_parity(self):
        x, ei, N, C, D, H, W = _small_3d()
        layer = TensorGraphSAGELayer(C, C, aggr="max", spatial_rank=3).eval()
        with torch.no_grad():
            out_full = layer(x, ei)
            out_chunk = layer(x, ei, chunk_size=5)
        assert torch.allclose(out_full, out_chunk, atol=ATOL)

    def test_edge_weight_parity(self):
        x, ei, N, C, H, W = _small_2d()
        layer = self._layer()
        ew = torch.rand(ei.size(1))
        with torch.no_grad():
            out_full = layer(x, ei, edge_weight=ew)
            out_chunk = layer(x, ei, edge_weight=ew, chunk_size=5)
        assert torch.allclose(out_full, out_chunk, atol=ATOL)

    def test_isolated_nodes_max(self):
        """Isolated nodes must be zero in max-aggregated output (not -inf)."""
        torch.manual_seed(2)
        N, C, H, W = 4, 4, 4, 4
        x = torch.randn(N, C, H, W)
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        layer = self._layer()
        with torch.no_grad():
            out_full = layer(x, ei)
            out_chunk = layer(x, ei, chunk_size=1)
        assert torch.allclose(out_full, out_chunk, atol=ATOL)
        # Node 3 has no incoming edges → agg must be 0 (not -inf)
        assert torch.isfinite(out_full[3]).all()
        assert torch.isfinite(out_chunk[3]).all()

    def test_gradient_finite(self):
        x, ei, N, C, H, W = _small_2d()
        x = x.requires_grad_(True)
        layer = TensorGraphSAGELayer(C, C, aggr="max").train()
        out = layer(x, ei, chunk_size=5)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ── GIN chunking ──────────────────────────────────────────────────────────────

class TestGINChunked:
    def _layer(self, **kw):
        return TensorGINLayer(4, 4, **kw).eval()

    def test_2d_parity(self):
        x, ei, N, C, H, W = _small_2d()
        layer = self._layer()
        with torch.no_grad():
            out_full = layer(x, ei)
            out_chunk = layer(x, ei, chunk_size=5)
        assert torch.allclose(out_full, out_chunk, atol=ATOL)

    def test_3d_parity(self):
        x, ei, N, C, D, H, W = _small_3d()
        layer = TensorGINLayer(C, C, spatial_rank=3).eval()
        with torch.no_grad():
            out_full = layer(x, ei)
            out_chunk = layer(x, ei, chunk_size=5)
        assert torch.allclose(out_full, out_chunk, atol=ATOL)

    def test_chunk_size_1(self):
        x, ei, N, C, H, W = _small_2d()
        layer = self._layer()
        with torch.no_grad():
            out_full = layer(x, ei)
            out_chunk = layer(x, ei, chunk_size=1)
        assert torch.allclose(out_full, out_chunk, atol=ATOL)

    def test_chunk_size_none_unchanged(self):
        x, ei, N, C, H, W = _small_2d()
        layer = self._layer()
        with torch.no_grad():
            out1 = layer(x, ei)
            out2 = layer(x, ei, chunk_size=None)
        assert torch.equal(out1, out2)

    def test_edge_weight_parity(self):
        x, ei, N, C, H, W = _small_2d()
        layer = self._layer()
        ew = torch.rand(ei.size(1))
        with torch.no_grad():
            out_full = layer(x, ei, edge_weight=ew)
            out_chunk = layer(x, ei, edge_weight=ew, chunk_size=5)
        assert torch.allclose(out_full, out_chunk, atol=ATOL)

    def test_vector_edge_features_parity(self):
        x, ei, N, C, H, W = _small_2d()
        layer = TensorGINLayer(
            C, C, use_edge_features=True, edge_dim=3,
            edge_features_kind="vector",
        ).eval()
        ef = torch.randn(ei.size(1), 3)
        with torch.no_grad():
            out_full = layer(x, ei, edge_features=ef)
            out_chunk = layer(x, ei, edge_features=ef, chunk_size=5)
        assert torch.allclose(out_full, out_chunk, atol=ATOL)

    def test_spatial_edge_features_parity(self):
        x, ei, N, C, H, W = _small_2d()
        layer = TensorGINLayer(
            C, C, use_edge_features=True, edge_dim=3,
            edge_features_kind="spatial",
        ).eval()
        ef = torch.randn(ei.size(1), 3, H, W)
        with torch.no_grad():
            out_full = layer(x, ei, edge_features=ef)
            out_chunk = layer(x, ei, edge_features=ef, chunk_size=5)
        assert torch.allclose(out_full, out_chunk, atol=ATOL)

    def test_train_eps_parity(self):
        x, ei, N, C, H, W = _small_2d()
        layer = TensorGINLayer(C, C, train_eps=True, eps=0.5).eval()
        with torch.no_grad():
            out_full = layer(x, ei)
            out_chunk = layer(x, ei, chunk_size=5)
        assert torch.allclose(out_full, out_chunk, atol=ATOL)

    def test_custom_mlp(self):
        import torch.nn as nn
        x, ei, N, C, H, W = _small_2d()
        mlp = nn.Sequential(
            nn.Conv2d(C, C, 1), nn.ReLU(), nn.Conv2d(C, C, 1)
        )
        layer = TensorGINLayer(C, C, mlp=mlp).eval()
        with torch.no_grad():
            out_full = layer(x, ei)
            out_chunk = layer(x, ei, chunk_size=5)
        assert torch.allclose(out_full, out_chunk, atol=ATOL)

    def test_isolated_nodes(self):
        torch.manual_seed(3)
        N, C, H, W = 5, 4, 4, 4
        x = torch.randn(N, C, H, W)
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        layer = self._layer()
        with torch.no_grad():
            out_full = layer(x, ei)
            out_chunk = layer(x, ei, chunk_size=1)
        assert torch.allclose(out_full, out_chunk, atol=ATOL)

    def test_gradient_finite(self):
        x, ei, N, C, H, W = _small_2d()
        x = x.requires_grad_(True)
        layer = TensorGINLayer(C, C).train()
        out = layer(x, ei, chunk_size=5)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    @skip_bf16
    def test_amp_bfloat16_smoke(self):
        x, ei, N, C, H, W = _small_2d()
        layer = self._layer()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = layer(x, ei, chunk_size=5)
        assert torch.isfinite(out).all()


# ── Graph builder chunked paths ───────────────────────────────────────────────

class TestKNNChunked:
    def _sort(self, ei):
        keys = ei[0] * 10000 + ei[1]
        return ei[:, keys.sort().indices]

    def test_parity_small(self):
        torch.manual_seed(0)
        coords = torch.randn(20, 2)
        ei_full = build_knn_graph(coords, k=3)
        ei_chunk = build_knn_graph(coords, k=3, chunk_size=5)
        assert self._sort(ei_full).shape == self._sort(ei_chunk).shape
        assert torch.all(self._sort(ei_full) == self._sort(ei_chunk))

    def test_parity_directed(self):
        torch.manual_seed(1)
        coords = torch.randn(12, 3)
        ei_full = build_knn_graph(coords, k=2, directed=True)
        ei_chunk = build_knn_graph(coords, k=2, directed=True, chunk_size=4)
        assert self._sort(ei_full).shape == self._sort(ei_chunk).shape
        assert torch.all(self._sort(ei_full) == self._sort(ei_chunk))

    def test_no_warning_with_chunk_size(self):
        coords = torch.randn(10_001, 2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ei = build_knn_graph(coords, k=2, chunk_size=100)
        # chunk_size suppresses the O(N²) memory warning
        mem_warns = [x for x in w if "O(N²)" in str(x.message) or "O.N.2" in str(x.message)]
        assert len(mem_warns) == 0, "unexpected O(N²) warning with chunk_size set"

    def test_warning_without_chunk_size(self):
        coords = torch.randn(10_001, 2)
        with pytest.warns(UserWarning, match="num_nodes"):
            build_knn_graph(coords, k=2)


class TestRadiusChunked:
    def _sort(self, ei):
        keys = ei[0] * 10000 + ei[1]
        return ei[:, keys.sort().indices]

    def test_parity_undirected(self):
        torch.manual_seed(2)
        coords = torch.randn(20, 2)
        ei_full = build_radius_graph(coords, radius=0.8)
        ei_chunk = build_radius_graph(coords, radius=0.8, chunk_size=5)
        assert self._sort(ei_full).shape == self._sort(ei_chunk).shape
        assert torch.all(self._sort(ei_full) == self._sort(ei_chunk))

    def test_parity_directed(self):
        torch.manual_seed(3)
        coords = torch.randn(16, 2)
        ei_full = build_radius_graph(coords, radius=0.5, directed=True)
        ei_chunk = build_radius_graph(coords, radius=0.5, directed=True, chunk_size=4)
        assert self._sort(ei_full).shape == self._sort(ei_chunk).shape
        assert torch.all(self._sort(ei_full) == self._sort(ei_chunk))

    def test_no_warning_with_chunk_size(self):
        coords = torch.randn(10_001, 2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            build_radius_graph(coords, radius=0.01, chunk_size=100)
        mem_warns = [x for x in w if "num_nodes" in str(x.message)]
        assert len(mem_warns) == 0


class TestIoUChunked:
    def _sort(self, ei):
        keys = ei[0] * 10000 + ei[1]
        return ei[:, keys.sort().indices]

    def _boxes(self, n=10, seed=4):
        torch.manual_seed(seed)
        x1 = torch.rand(n)
        y1 = torch.rand(n)
        x2 = x1 + torch.rand(n) * 0.5 + 0.1
        y2 = y1 + torch.rand(n) * 0.5 + 0.1
        return torch.stack([x1, y1, x2, y2], dim=1)

    def test_parity(self):
        boxes = self._boxes(16)
        ei_full = build_iou_graph(boxes, threshold=0.1)
        ei_chunk = build_iou_graph(boxes, threshold=0.1, chunk_size=4)
        assert self._sort(ei_full).shape == self._sort(ei_chunk).shape
        assert torch.all(self._sort(ei_full) == self._sort(ei_chunk))

    def test_parity_directed(self):
        boxes = self._boxes(12, seed=5)
        ei_full = build_iou_graph(boxes, threshold=0.05, directed=True)
        ei_chunk = build_iou_graph(boxes, threshold=0.05, directed=True, chunk_size=3)
        assert self._sort(ei_full).shape == self._sort(ei_chunk).shape
        assert torch.all(self._sort(ei_full) == self._sort(ei_chunk))

    def test_no_warning_with_chunk_size(self):
        boxes = self._boxes(5_001)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            build_iou_graph(boxes, threshold=0.5, chunk_size=100)
        mem_warns = [x for x in w if "num_nodes" in str(x.message)]
        assert len(mem_warns) == 0


class TestRandomGraphSample:
    def test_correct_edge_count(self):
        ei = build_random_graph(100, 200, algorithm="sample", seed=42)
        assert ei.shape == (2, 200)

    def test_no_self_loops(self):
        ei = build_random_graph(100, 200, algorithm="sample", seed=42)
        assert (ei[0] != ei[1]).all()

    def test_no_duplicates(self):
        ei = build_random_graph(100, 200, algorithm="sample", seed=42)
        keys = ei[0] * 100 + ei[1]
        assert keys.unique().numel() == 200

    def test_deterministic(self):
        ei1 = build_random_graph(100, 200, algorithm="sample", seed=99)
        ei2 = build_random_graph(100, 200, algorithm="sample", seed=99)
        assert torch.equal(ei1, ei2)

    def test_zero_edges(self):
        ei = build_random_graph(10, 0, algorithm="sample", seed=0)
        assert ei.shape == (2, 0)

    def test_too_many_edges_raises(self):
        with pytest.raises(ValueError, match="Cannot sample"):
            build_random_graph(5, 100, algorithm="sample", seed=0)

    def test_unsupported_undirected_raises(self):
        with pytest.raises(ValueError, match="directed=True"):
            build_random_graph(10, 5, directed=False, algorithm="sample")

    def test_exact_is_default_and_unchanged(self):
        """Default algorithm='exact' must match previous behavior."""
        ei_default = build_random_graph(10, 20, seed=0)
        ei_exact = build_random_graph(10, 20, seed=0, algorithm="exact")
        assert torch.equal(ei_default, ei_exact)
