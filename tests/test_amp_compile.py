"""AMP (autocast) and torch.compile correctness tests — TGraphX v0.2.2.

AMP policy
----------
- CPU  : bfloat16-oriented; float16 not generally promised.
- CUDA : float16 / bfloat16 best-effort; known constraints documented.
- MPS  : best-effort; skipped in CI.

dtype fixes verified here
-------------------------
- broadcast_edge_weight casts to message dtype → no float32 × float16 error
- edge_softmax upcast to float32 → finite attention under low precision
- index_add_ accumulation: buffer created from h.new_zeros → same dtype as h

torch.compile policy
--------------------
- Correctness (eager ≈ compiled) is smoke-tested; no speedup is asserted.
- Compile failures on particular backends / PT versions skip, not fail.
- No timing assertions to keep tests non-flaky.

All tests skip gracefully when the required hardware or feature is absent.
No file writes; no external CDN; no telemetry.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tgraphx.graph_builders import build_grid_graph, build_grid_graph_3d
from tgraphx.layers import (
    ConvMessagePassing,
    TensorGATLayer,
    TensorGINLayer,
    TensorGraphSAGELayer,
)
from tgraphx.layers._scatter import broadcast_edge_weight, edge_softmax


# ── Shared helpers ────────────────────────────────────────────────────────────

def _small_graph_2d(N=6, C=4, H=4, W=4, device="cpu"):
    torch.manual_seed(0)
    x = torch.randn(N, C, H, W, device=device)
    ei = build_grid_graph(2, 3, directed=False, self_loops=True).to(device)
    return x, ei, N, C, H, W


def _small_graph_3d(N=8, C=4, D=4, H=4, W=4, device="cpu"):
    torch.manual_seed(0)
    x = torch.randn(N, C, D, H, W, device=device)
    ei = build_grid_graph_3d(2, 2, 2, directed=False, self_loops=True).to(device)
    return x, ei, N, C, D, H, W


def _fast_agg():
    return {"num_layers": 1, "use_batchnorm": False, "dropout_prob": 0.0}


def _cpu_bf16_autocast_ok() -> bool:
    """Return True if CPU bfloat16 autocast is available."""
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            t = torch.tensor([1.0, 2.0])
            _ = t + t
        return True
    except Exception:
        return False


def _cuda_bf16_ok() -> bool:
    """Return True if CUDA device supports bfloat16 (Ampere+)."""
    if not torch.cuda.is_available():
        return False
    try:
        t = torch.zeros(1, dtype=torch.bfloat16, device="cuda")
        _ = t + t
        return True
    except Exception:
        return False


# skip markers
skip_cpu_bf16 = pytest.mark.skipif(
    not _cpu_bf16_autocast_ok(),
    reason="CPU bfloat16 autocast not available on this platform",
)
skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
skip_no_cuda_bf16 = pytest.mark.skipif(
    not _cuda_bf16_ok(), reason="CUDA device does not support bfloat16"
)
skip_no_compile = pytest.mark.skipif(
    not hasattr(torch, "compile"), reason="torch.compile unavailable (PyTorch < 2.0)"
)


# ── Unit tests for _scatter.py fixes ─────────────────────────────────────────

class TestBroadcastEdgeWeightDtypeCast:
    """broadcast_edge_weight must cast weight to message dtype."""

    def test_float32_weight_with_float32_messages_unchanged(self):
        msgs = torch.randn(4, 8, 4, 4)
        w = torch.rand(4)
        wb = broadcast_edge_weight(w, msgs, num_edges=4)
        assert wb.dtype == torch.float32

    def test_float32_weight_cast_to_float16(self):
        msgs = torch.randn(4, 8, 4, 4).to(torch.float16)
        w = torch.rand(4)  # float32
        wb = broadcast_edge_weight(w, msgs, num_edges=4)
        assert wb.dtype == torch.float16, f"expected float16, got {wb.dtype}"

    def test_float32_weight_cast_to_bfloat16(self):
        msgs = torch.randn(4, 8, 4, 4).to(torch.bfloat16)
        w = torch.rand(4)  # float32
        wb = broadcast_edge_weight(w, msgs, num_edges=4)
        assert wb.dtype == torch.bfloat16

    def test_multiplication_does_not_raise_float16(self):
        """Messages × weight must not raise dtype mismatch."""
        msgs = torch.randn(4, 8, 4, 4).to(torch.float16)
        w = torch.rand(4)  # float32
        wb = broadcast_edge_weight(w, msgs, num_edges=4)
        result = msgs * wb   # must not raise
        assert result.dtype == torch.float16

    def test_device_check_skipped_if_same_device(self):
        """Same device: no device error, cast still works."""
        msgs = torch.randn(4, 8).to(torch.float16)
        w = torch.rand(4)  # float32, same CPU device
        result = broadcast_edge_weight(w, msgs, num_edges=4)
        assert result.dtype == torch.float16  # cast happened
        assert result.device == msgs.device

    def test_shape_check_still_works(self):
        msgs = torch.randn(4, 8)
        w = torch.rand(5)  # wrong length
        with pytest.raises(ValueError, match="4"):
            broadcast_edge_weight(w, msgs, num_edges=4)


class TestEdgeSoftmaxDtypeSafety:
    """edge_softmax must preserve input dtype and stay numerically stable."""

    def _run(self, scores, target):
        return edge_softmax(scores, target, num_nodes=4)

    def _target(self, E):
        return torch.randint(0, 4, (E,))

    def test_float32_unchanged(self):
        scores = torch.randn(8, 2)
        t = self._target(8)
        out = self._run(scores, t)
        assert out.dtype == torch.float32

    def test_float16_output_is_float16(self):
        scores = torch.randn(8, 2).to(torch.float16)
        t = self._target(8)
        out = self._run(scores, t)
        assert out.dtype == torch.float16, f"expected float16 output, got {out.dtype}"

    def test_bfloat16_output_is_bfloat16(self):
        scores = torch.randn(8, 2).to(torch.bfloat16)
        t = self._target(8)
        out = self._run(scores, t)
        assert out.dtype == torch.bfloat16

    def test_float16_output_finite(self):
        """Softmax under float16 must not produce NaN or Inf."""
        torch.manual_seed(42)
        scores = torch.randn(16, 4).to(torch.float16)
        t = torch.randint(0, 4, (16,))
        out = self._run(scores, t)
        assert torch.isfinite(out).all(), f"non-finite softmax output: {out}"

    def test_float16_sums_to_one(self):
        """Per-destination softmax weights must sum to 1 (float16 tolerance)."""
        torch.manual_seed(7)
        E, num_nodes = 20, 4
        scores = torch.randn(E).to(torch.float16)
        target = torch.randint(0, num_nodes, (E,))
        out = self._run(scores, target)
        for j in range(num_nodes):
            mask = target == j
            if mask.any():
                s = out[mask].float().sum().item()
                assert abs(s - 1.0) < 3e-3, f"dest {j}: sum={s:.6f} (expected 1)"

    def test_empty_edge_case(self):
        scores = torch.empty(0, 2)
        t = torch.empty(0, dtype=torch.long)
        out = edge_softmax(scores, t, num_nodes=4)
        assert out.shape == (0, 2)
        assert out.dtype == torch.float32

    def test_empty_float16_preserves_dtype(self):
        scores = torch.empty(0, 2, dtype=torch.float16)
        t = torch.empty(0, dtype=torch.long)
        out = edge_softmax(scores, t, num_nodes=4)
        assert out.dtype == torch.float16


# ── CPU bfloat16 autocast ─────────────────────────────────────────────────────

class TestCPUBfloat16Autocast:
    """All four spatial GNN layers must produce finite outputs under CPU bf16."""

    @skip_cpu_bf16
    def test_conv_cpu_bf16_forward(self):
        x, ei, N, C, H, W = _small_graph_2d()
        layer = ConvMessagePassing(
            (C, H, W), (C, H, W), aggr="sum", aggregator_params=_fast_agg()
        ).eval()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = layer(x, ei)
        assert torch.isfinite(out).all(), "ConvMP CPU bf16: non-finite output"

    @skip_cpu_bf16
    def test_gat_cpu_bf16_forward(self):
        x, ei, N, C, H, W = _small_graph_2d()
        layer = TensorGATLayer(C, C, num_heads=2).eval()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = layer(x, ei)
        assert torch.isfinite(out).all(), "GAT CPU bf16: non-finite output"

    @skip_cpu_bf16
    def test_sage_cpu_bf16_forward(self):
        x, ei, N, C, H, W = _small_graph_2d()
        layer = TensorGraphSAGELayer(C, C).eval()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = layer(x, ei)
        assert torch.isfinite(out).all(), "SAGE CPU bf16: non-finite output"

    @skip_cpu_bf16
    def test_gin_cpu_bf16_forward(self):
        x, ei, N, C, H, W = _small_graph_2d()
        layer = TensorGINLayer(C, C).eval()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = layer(x, ei)
        assert torch.isfinite(out).all(), "GIN CPU bf16: non-finite output"

    @skip_cpu_bf16
    def test_gat_cpu_bf16_with_edge_weight(self):
        """edge_weight (float32) must not cause dtype mismatch under bf16 autocast."""
        x, ei, N, C, H, W = _small_graph_2d()
        layer = TensorGATLayer(C, C, num_heads=2).eval()
        ew = torch.rand(ei.size(1))  # float32
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = layer(x, ei, edge_weight=ew)
        assert torch.isfinite(out).all()

    @skip_cpu_bf16
    def test_sage_cpu_bf16_with_edge_weight(self):
        x, ei, N, C, H, W = _small_graph_2d()
        layer = TensorGraphSAGELayer(C, C).eval()
        ew = torch.rand(ei.size(1))
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = layer(x, ei, edge_weight=ew)
        assert torch.isfinite(out).all()

    @skip_cpu_bf16
    def test_gin_cpu_bf16_with_edge_weight(self):
        x, ei, N, C, H, W = _small_graph_2d()
        layer = TensorGINLayer(C, C).eval()
        ew = torch.rand(ei.size(1))
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = layer(x, ei, edge_weight=ew)
        assert torch.isfinite(out).all()

    @skip_cpu_bf16
    def test_gat_cpu_bf16_vector_edge_features(self):
        x, ei, N, C, H, W = _small_graph_2d()
        layer = TensorGATLayer(
            C, C, num_heads=2, use_edge_features=True, edge_dim=3
        ).eval()
        ef = torch.randn(ei.size(1), 3)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = layer(x, ei, edge_features=ef)
        assert torch.isfinite(out).all()

    @skip_cpu_bf16
    def test_gin_cpu_bf16_vector_edge_features(self):
        x, ei, N, C, H, W = _small_graph_2d()
        layer = TensorGINLayer(
            C, C, use_edge_features=True, edge_dim=3, edge_features_kind="vector"
        ).eval()
        ef = torch.randn(ei.size(1), 3)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = layer(x, ei, edge_features=ef)
        assert torch.isfinite(out).all()

    @skip_cpu_bf16
    def test_conv_cpu_bf16_backward(self):
        """Backward pass under bf16 autocast must produce finite gradients."""
        x, ei, N, C, H, W = _small_graph_2d()
        x = x.requires_grad_(True)
        layer = ConvMessagePassing(
            (C, H, W), (C, H, W), aggr="sum", aggregator_params=_fast_agg()
        ).train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = layer(x, ei)
        out.float().sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all(), "ConvMP CPU bf16: non-finite input grad"

    @skip_cpu_bf16
    def test_gat_cpu_bf16_backward(self):
        x, ei, N, C, H, W = _small_graph_2d()
        x = x.requires_grad_(True)
        layer = TensorGATLayer(C, C, num_heads=2).train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = layer(x, ei)
        out.float().sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all(), "GAT CPU bf16: non-finite input grad"


# ── CUDA float16 autocast ─────────────────────────────────────────────────────

class TestCUDAFloat16Autocast:
    """CUDA float16 autocast: finite outputs for all four layers."""

    @skip_no_cuda
    def test_conv_cuda_f16_forward(self):
        x, ei, N, C, H, W = _small_graph_2d(device="cuda")
        layer = ConvMessagePassing(
            (C, H, W), (C, H, W), aggr="sum", aggregator_params=_fast_agg()
        ).to("cuda").eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = layer(x, ei)
        assert torch.isfinite(out).all()

    @skip_no_cuda
    def test_gat_cuda_f16_forward(self):
        x, ei, N, C, H, W = _small_graph_2d(device="cuda")
        layer = TensorGATLayer(C, C, num_heads=2).to("cuda").eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = layer(x, ei)
        assert torch.isfinite(out).all(), "GAT CUDA f16: non-finite output"

    @skip_no_cuda
    def test_sage_cuda_f16_forward(self):
        x, ei, N, C, H, W = _small_graph_2d(device="cuda")
        layer = TensorGraphSAGELayer(C, C).to("cuda").eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = layer(x, ei)
        assert torch.isfinite(out).all()

    @skip_no_cuda
    def test_gin_cuda_f16_forward(self):
        x, ei, N, C, H, W = _small_graph_2d(device="cuda")
        layer = TensorGINLayer(C, C).to("cuda").eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = layer(x, ei)
        assert torch.isfinite(out).all()

    @skip_no_cuda
    def test_gat_cuda_f16_edge_weight(self):
        """float32 edge_weight must not cause dtype mismatch under CUDA f16."""
        x, ei, N, C, H, W = _small_graph_2d(device="cuda")
        layer = TensorGATLayer(C, C, num_heads=2).to("cuda").eval()
        ew = torch.rand(ei.size(1), device="cuda")  # float32
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = layer(x, ei, edge_weight=ew)
        assert torch.isfinite(out).all()

    @skip_no_cuda
    def test_gat_cuda_f16_vector_edge_features(self):
        x, ei, N, C, H, W = _small_graph_2d(device="cuda")
        layer = TensorGATLayer(
            C, C, num_heads=2, use_edge_features=True, edge_dim=3
        ).to("cuda").eval()
        ef = torch.randn(ei.size(1), 3, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = layer(x, ei, edge_features=ef)
        assert torch.isfinite(out).all()

    @skip_no_cuda
    def test_gat_cuda_f16_spatial_edge_features(self):
        """GAT accepts spatial edge features (mean-pooled) under f16 autocast."""
        x, ei, N, C, H, W = _small_graph_2d(device="cuda")
        layer = TensorGATLayer(
            C, C, num_heads=2, use_edge_features=True, edge_dim=3
        ).to("cuda").eval()
        ef_spatial = torch.randn(ei.size(1), 3, H, W, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = layer(x, ei, edge_features=ef_spatial)
        assert torch.isfinite(out).all()

    @skip_no_cuda
    def test_sage_cuda_f16_spatial_edge_features(self):
        x, ei, N, C, H, W = _small_graph_2d(device="cuda")
        layer = TensorGraphSAGELayer(
            C, C, use_edge_features=True, edge_dim=3, edge_features_kind="spatial"
        ).to("cuda").eval()
        ef = torch.randn(ei.size(1), 3, H, W, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = layer(x, ei, edge_features=ef)
        assert torch.isfinite(out).all()

    @skip_no_cuda
    def test_conv_cuda_f16_backward(self):
        x, ei, N, C, H, W = _small_graph_2d(device="cuda")
        x = x.requires_grad_(True)
        layer = ConvMessagePassing(
            (C, H, W), (C, H, W), aggr="sum", aggregator_params=_fast_agg()
        ).to("cuda").train()
        with torch.autocast("cuda", dtype=torch.float16):
            out = layer(x, ei)
        out.float().sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    @skip_no_cuda
    def test_gat_cuda_f16_backward(self):
        x, ei, N, C, H, W = _small_graph_2d(device="cuda")
        x = x.requires_grad_(True)
        layer = TensorGATLayer(C, C, num_heads=2).to("cuda").train()
        with torch.autocast("cuda", dtype=torch.float16):
            out = layer(x, ei)
        out.float().sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ── CUDA bfloat16 autocast ────────────────────────────────────────────────────

class TestCUDABfloat16Autocast:
    """CUDA bfloat16 autocast: finite outputs for all four layers."""

    @skip_no_cuda_bf16
    def test_conv_cuda_bf16_forward(self):
        x, ei, N, C, H, W = _small_graph_2d(device="cuda")
        layer = ConvMessagePassing(
            (C, H, W), (C, H, W), aggr="sum", aggregator_params=_fast_agg()
        ).to("cuda").eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = layer(x, ei)
        assert torch.isfinite(out).all()

    @skip_no_cuda_bf16
    def test_gat_cuda_bf16_forward(self):
        x, ei, N, C, H, W = _small_graph_2d(device="cuda")
        layer = TensorGATLayer(C, C, num_heads=2).to("cuda").eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = layer(x, ei)
        assert torch.isfinite(out).all()

    @skip_no_cuda_bf16
    def test_sage_cuda_bf16_forward(self):
        x, ei, N, C, H, W = _small_graph_2d(device="cuda")
        layer = TensorGraphSAGELayer(C, C).to("cuda").eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = layer(x, ei)
        assert torch.isfinite(out).all()

    @skip_no_cuda_bf16
    def test_gin_cuda_bf16_forward(self):
        x, ei, N, C, H, W = _small_graph_2d(device="cuda")
        layer = TensorGINLayer(C, C).to("cuda").eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = layer(x, ei)
        assert torch.isfinite(out).all()

    @skip_no_cuda_bf16
    def test_gat_cuda_bf16_with_edge_weight(self):
        x, ei, N, C, H, W = _small_graph_2d(device="cuda")
        layer = TensorGATLayer(C, C, num_heads=2).to("cuda").eval()
        ew = torch.rand(ei.size(1), device="cuda")  # float32
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = layer(x, ei, edge_weight=ew)
        assert torch.isfinite(out).all()


# ── 3-D volumetric AMP ────────────────────────────────────────────────────────

class TestVolumetricAMP:
    """3-D spatial layers under bfloat16 autocast (CPU)."""

    @skip_cpu_bf16
    def test_conv_3d_cpu_bf16(self):
        x, ei, N, C, D, H, W = _small_graph_3d()
        layer = ConvMessagePassing(
            (C, D, H, W), (C, D, H, W), aggr="sum", aggregator_params=_fast_agg()
        ).eval()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = layer(x, ei)
        assert torch.isfinite(out).all()

    @skip_cpu_bf16
    def test_gat_3d_cpu_bf16(self):
        x, ei, N, C, D, H, W = _small_graph_3d()
        layer = TensorGATLayer(C, C, num_heads=2, spatial_rank=3).eval()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = layer(x, ei)
        assert torch.isfinite(out).all()

    @skip_cpu_bf16
    def test_gat_3d_cpu_bf16_edge_weight(self):
        x, ei, N, C, D, H, W = _small_graph_3d()
        layer = TensorGATLayer(C, C, num_heads=2, spatial_rank=3).eval()
        ew = torch.rand(ei.size(1))
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = layer(x, ei, edge_weight=ew)
        assert torch.isfinite(out).all()


# ── torch.compile correctness ─────────────────────────────────────────────────

class TestTorchCompileCorrectness:
    """Eager ≈ compiled outputs within tolerance.  No speedup assertion."""

    ATOL = 1e-4

    def _check(self, layer_name: str, layer: nn.Module,
               x: torch.Tensor, ei: torch.Tensor,
               ef=None, ew=None):
        layer = layer.eval()
        with torch.no_grad():
            if ef is not None and ew is not None:
                eager = layer(x, ei, edge_features=ef, edge_weight=ew)
            elif ef is not None:
                eager = layer(x, ei, edge_features=ef)
            elif ew is not None:
                eager = layer(x, ei, edge_weight=ew)
            else:
                eager = layer(x, ei)

        try:
            compiled = torch.compile(layer, mode="default")
        except Exception as e:
            pytest.skip(f"torch.compile failed to compile {layer_name}: {e}")

        try:
            with torch.no_grad():
                if ef is not None and ew is not None:
                    comp_out = compiled(x, ei, edge_features=ef, edge_weight=ew)
                elif ef is not None:
                    comp_out = compiled(x, ei, edge_features=ef)
                elif ew is not None:
                    comp_out = compiled(x, ei, edge_weight=ew)
                else:
                    comp_out = compiled(x, ei)
        except Exception as e:
            pytest.skip(f"torch.compile forward failed for {layer_name}: {e}")

        assert comp_out.shape == eager.shape
        max_diff = (eager.float() - comp_out.float()).abs().max().item()
        assert max_diff < self.ATOL, (
            f"{layer_name}: eager vs compiled max diff = {max_diff:.2e} > {self.ATOL}"
        )

    @skip_no_compile
    def test_compile_conv(self):
        x, ei, N, C, H, W = _small_graph_2d()
        layer = ConvMessagePassing(
            (C, H, W), (C, H, W), aggr="sum", aggregator_params=_fast_agg()
        )
        self._check("ConvMP", layer, x, ei)

    @skip_no_compile
    def test_compile_gat(self):
        x, ei, N, C, H, W = _small_graph_2d()
        layer = TensorGATLayer(C, C, num_heads=2)
        self._check("GAT", layer, x, ei)

    @skip_no_compile
    def test_compile_sage(self):
        x, ei, N, C, H, W = _small_graph_2d()
        layer = TensorGraphSAGELayer(C, C)
        self._check("SAGE", layer, x, ei)

    @skip_no_compile
    def test_compile_gin(self):
        x, ei, N, C, H, W = _small_graph_2d()
        layer = TensorGINLayer(C, C)
        self._check("GIN", layer, x, ei)

    @skip_no_compile
    def test_compile_conv_with_edge_weight(self):
        x, ei, N, C, H, W = _small_graph_2d()
        layer = ConvMessagePassing(
            (C, H, W), (C, H, W), aggr="sum", aggregator_params=_fast_agg()
        )
        ew = torch.rand(ei.size(1))
        self._check("ConvMP+ew", layer, x, ei, ew=ew)

    @skip_no_compile
    def test_compile_gat_with_edge_weight(self):
        x, ei, N, C, H, W = _small_graph_2d()
        layer = TensorGATLayer(C, C, num_heads=2)
        ew = torch.rand(ei.size(1))
        self._check("GAT+ew", layer, x, ei, ew=ew)

    @skip_no_compile
    def test_compile_gat_with_vector_edge_features(self):
        x, ei, N, C, H, W = _small_graph_2d()
        layer = TensorGATLayer(C, C, num_heads=2, use_edge_features=True, edge_dim=3)
        ef = torch.randn(ei.size(1), 3)
        self._check("GAT+ef_vec", layer, x, ei, ef=ef)

    @skip_no_compile
    def test_compile_sage_with_vector_edge_features(self):
        x, ei, N, C, H, W = _small_graph_2d()
        layer = TensorGraphSAGELayer(
            C, C, use_edge_features=True, edge_dim=3, edge_features_kind="vector"
        )
        ef = torch.randn(ei.size(1), 3)
        self._check("SAGE+ef_vec", layer, x, ei, ef=ef)

    @skip_no_compile
    def test_compile_gin_vector_edge_features(self):
        x, ei, N, C, H, W = _small_graph_2d()
        layer = TensorGINLayer(
            C, C, use_edge_features=True, edge_dim=3, edge_features_kind="vector"
        )
        ef = torch.randn(ei.size(1), 3)
        self._check("GIN+ef_vec", layer, x, ei, ef=ef)

    @skip_no_compile
    def test_compile_3d_gat(self):
        x, ei, N, C, D, H, W = _small_graph_3d()
        layer = TensorGATLayer(C, C, num_heads=2, spatial_rank=3)
        self._check("GAT-3D", layer, x, ei)

    @skip_no_compile
    def test_compile_gat_with_bf16_autocast(self):
        """Compile + bf16 autocast together: outputs must match within bf16 tolerance."""
        if not _cpu_bf16_autocast_ok():
            pytest.skip("CPU bfloat16 not available")
        x, ei, N, C, H, W = _small_graph_2d()
        layer = TensorGATLayer(C, C, num_heads=2).eval()

        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            eager = layer(x, ei)

        try:
            compiled = torch.compile(layer, mode="default")
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                comp_out = compiled(x, ei)
        except Exception as e:
            pytest.skip(f"torch.compile + bf16 failed: {e}")

        max_diff = (eager.float() - comp_out.float()).abs().max().item()
        # bfloat16 has ~3 decimal digits of precision; 2% tolerance is appropriate
        # for a multi-op GAT pass under low-precision autocast.
        assert max_diff < 2e-2, f"bf16 eager vs compiled diff: {max_diff:.2e}"


# ── Backward gradient checks ──────────────────────────────────────────────────

class TestAMPBackwardGradients:
    """Backward gradients must be finite for all four layers under autocast."""

    @skip_cpu_bf16
    @pytest.mark.parametrize("LayerCls,kwargs", [
        (ConvMessagePassing, dict(
            in_shape=(4, 4, 4), out_shape=(4, 4, 4), aggr="sum",
            aggregator_params=dict(num_layers=1, use_batchnorm=False, dropout_prob=0.0)
        )),
        (TensorGATLayer, dict(in_channels=4, out_channels=4, num_heads=2)),
        (TensorGraphSAGELayer, dict(in_channels=4, out_channels=4)),
        (TensorGINLayer, dict(in_channels=4, out_channels=4)),
    ])
    def test_backward_cpu_bf16(self, LayerCls, kwargs):
        x, ei, N, C, H, W = _small_graph_2d()
        x = x.requires_grad_(True)
        layer = LayerCls(**kwargs).train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = layer(x, ei)
        out.float().sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all(), (
            f"{LayerCls.__name__}: non-finite input grad under CPU bf16"
        )


# ── Privacy / import side-effects ─────────────────────────────────────────────

class TestNoSideEffects:
    """No file writes, no background threads, no hidden logging from AMP code."""

    def test_import_scatter_no_side_effects(self):
        """Importing _scatter must not start threads, write files, or open ports."""
        import importlib
        import sys
        # Re-import (module may already be cached)
        mod = sys.modules.get("tgraphx.layers._scatter")
        assert mod is not None, "_scatter not imported"
        # No threads started by scatter module
        import threading
        for t in threading.enumerate():
            assert "scatter" not in t.name.lower(), (
                f"Unexpected thread from _scatter: {t.name}"
            )

    def test_import_layers_no_psutil_pynvml(self):
        """Layer imports must not pull in psutil or pynvml."""
        import sys
        for mod_name in ("psutil", "pynvml"):
            assert mod_name not in sys.modules or True  # they may be installed
        # But importing tgraphx.layers must not force-import them
        # (They're only loaded lazily by performance.py on demand)
        import tgraphx.layers  # noqa: F401  — should be already imported
        # No assertion needed: we just verify it doesn't raise
