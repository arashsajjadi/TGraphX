"""GAT two-pass chunked forward tests (v0.2.4)."""
import pytest
import torch
from tgraphx.layers import TensorGATLayer
from tgraphx.graph_builders import build_grid_graph, build_grid_graph_3d

ATOL = 1e-5


def _small_2d(N=9, C=4, H=4, W=4):
    torch.manual_seed(0)
    x = torch.randn(N, C, H, W)
    ei = build_grid_graph(3, 3, directed=False, self_loops=True)
    return x, ei, N, C, H, W


def _small_3d(N=8, C=4, D=4, H=4, W=4):
    torch.manual_seed(1)
    x = torch.randn(N, C, D, H, W)
    ei = build_grid_graph_3d(2, 2, 2, directed=False, self_loops=True)
    return x, ei


def _cpu_bf16_ok():
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _ = torch.tensor([1.0]) + torch.tensor([1.0])
        return True
    except Exception:
        return False


skip_bf16 = pytest.mark.skipif(not _cpu_bf16_ok(), reason="CPU bf16 not available")


class TestGATChunkedScalar:
    def test_parity_2d_mean(self):
        x, ei, N, C, H, W = _small_2d()
        l = TensorGATLayer(C, C, num_heads=2).eval()
        with torch.no_grad():
            full = l(x, ei)
            chunked = l(x, ei, chunk_size=5)
        assert torch.allclose(full, chunked, atol=ATOL), f"max diff={( full-chunked).abs().max():.2e}"

    def test_parity_3d(self):
        x, ei = _small_3d()
        C = x.size(1)
        l = TensorGATLayer(C, C, num_heads=2, spatial_rank=3).eval()
        with torch.no_grad():
            full = l(x, ei); chunked = l(x, ei, chunk_size=4)
        assert torch.allclose(full, chunked, atol=ATOL)

    def test_chunk_size_exceeds_edges(self):
        x, ei, N, C, H, W = _small_2d()
        l = TensorGATLayer(C, C, num_heads=2).eval()
        with torch.no_grad():
            full = l(x, ei); chunked = l(x, ei, chunk_size=100_000)
        assert torch.allclose(full, chunked, atol=ATOL)

    def test_chunk_size_1(self):
        x, ei, N, C, H, W = _small_2d()
        l = TensorGATLayer(C, C, num_heads=2).eval()
        with torch.no_grad():
            full = l(x, ei); chunked = l(x, ei, chunk_size=1)
        assert torch.allclose(full, chunked, atol=1e-4)

    def test_edge_weight_parity(self):
        x, ei, N, C, H, W = _small_2d()
        l = TensorGATLayer(C, C, num_heads=2).eval()
        ew = torch.rand(ei.size(1))
        with torch.no_grad():
            full = l(x, ei, edge_weight=ew)
            chunked = l(x, ei, edge_weight=ew, chunk_size=5)
        assert torch.allclose(full, chunked, atol=ATOL)

    def test_vector_edge_features_parity(self):
        x, ei, N, C, H, W = _small_2d()
        l = TensorGATLayer(C, C, num_heads=2, use_edge_features=True, edge_dim=3).eval()
        ef = torch.randn(ei.size(1), 3)
        with torch.no_grad():
            full = l(x, ei, edge_features=ef)
            chunked = l(x, ei, edge_features=ef, chunk_size=5)
        assert torch.allclose(full, chunked, atol=ATOL)

    def test_spatial_edge_features_parity(self):
        x, ei, N, C, H, W = _small_2d()
        l = TensorGATLayer(C, C, num_heads=2, use_edge_features=True, edge_dim=3).eval()
        ef = torch.randn(ei.size(1), 3, H, W)
        with torch.no_grad():
            full = l(x, ei, edge_features=ef)
            chunked = l(x, ei, edge_features=ef, chunk_size=5)
        assert torch.allclose(full, chunked, atol=ATOL)

    def test_return_attention_chunked(self):
        """Chunked return_attention must produce valid normalized weights."""
        x, ei, N, C, H, W = _small_2d()
        l = TensorGATLayer(C, C, num_heads=2).eval()
        with torch.no_grad():
            out_f, attn_f = l(x, ei, return_attention=True)
            out_c, attn_c = l(x, ei, return_attention=True, chunk_size=5)
        assert attn_c is not None
        assert attn_c.shape == (ei.size(1), 2)
        # Attention sums to 1 per dest per head (chunked)
        dst = ei[1]
        for j in range(N):
            mask = dst == j
            if mask.any():
                s = attn_c[mask].sum(0)
                assert (s - 1.0).abs().max() < 1e-4, f"node {j}: attn sum={s}"

    def test_return_attention_none_without_flag(self):
        x, ei, N, C, H, W = _small_2d()
        l = TensorGATLayer(C, C, num_heads=2).eval()
        with torch.no_grad():
            result = l(x, ei, chunk_size=5)
        assert isinstance(result, torch.Tensor)

    def test_gradient_finite(self):
        x, ei, N, C, H, W = _small_2d()
        x = x.requires_grad_(True)
        l = TensorGATLayer(C, C, num_heads=2).train()
        out = l(x, ei, chunk_size=5)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_isolated_nodes(self):
        """Nodes with no incoming edges should produce finite output."""
        torch.manual_seed(3)
        N, C, H, W = 5, 4, 4, 4
        x = torch.randn(N, C, H, W)
        ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        l = TensorGATLayer(C, C, num_heads=1).eval()
        with torch.no_grad():
            full = l(x, ei); chunked = l(x, ei, chunk_size=2)
        assert torch.isfinite(full).all()
        assert torch.allclose(full, chunked, atol=ATOL)

    @skip_bf16
    def test_amp_bfloat16_chunked(self):
        x, ei, N, C, H, W = _small_2d()
        l = TensorGATLayer(C, C, num_heads=2).eval()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = l(x, ei, chunk_size=5)
        assert torch.isfinite(out).all()


class TestGATAttentionSumsToOne:
    """Attention weights must sum to 1 per destination per head in all modes."""

    def _check_sums(self, attn, dst, N):
        for j in range(N):
            mask = dst == j
            if mask.any():
                s = attn[mask].sum(0)
                assert (s - 1.0).abs().max() < 2e-4, f"sum={s}"

    def test_unchunked_sums_to_one(self):
        x, ei, N, C, H, W = _small_2d()
        l = TensorGATLayer(C, C, num_heads=2).eval()
        _, attn = l(x, ei, return_attention=True)
        self._check_sums(attn, ei[1], N)

    def test_chunked_sums_to_one(self):
        x, ei, N, C, H, W = _small_2d()
        l = TensorGATLayer(C, C, num_heads=2).eval()
        _, attn = l(x, ei, return_attention=True, chunk_size=5)
        self._check_sums(attn, ei[1], N)
