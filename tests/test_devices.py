"""Device-specific tests.

CPU tests always run.
CUDA / MPS tests are skipped automatically when the hardware is absent
(see conftest.py).

The tests in this file are intentionally minimal: they verify that tensors
stay on the correct device through the entire forward→backward pipeline and
that Graph.to() / GraphBatch.to() work correctly.
"""

import pytest
import torch

from tgraphx import Graph, GraphBatch
from tgraphx.layers import ConvMessagePassing
from tgraphx.core.utils import get_device


# ──────────────────────────────────────────────────────────────────── #
# Shared logic (device-agnostic)                                        #
# ──────────────────────────────────────────────────────────────────── #

N, C, H, W = 4, 3, 8, 8


def _ei(n=N, device="cpu"):
    src = torch.arange(n, device=device)
    return torch.stack([src, (src + 1) % n])


def _fast_agg():
    return {"num_layers": 1, "use_batchnorm": False, "dropout_prob": 0.0}


def _run_forward_backward(device_str: str):
    """Forward + backward on a ConvMessagePassing layer on the given device."""
    x = torch.randn(N, C, H, W, device=device_str, requires_grad=True)
    ei = _ei(device=device_str)
    layer = ConvMessagePassing(
        (C, H, W), (8, H, W), aggregator_params=_fast_agg()
    ).to(device_str)
    out = layer(x, ei)
    assert out.device.type == device_str, f"output on {out.device}, expected {device_str}"
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def _run_mean_aggregation(device_str: str):
    """C-06: mean aggregation must work for spatial features on any device."""
    x = torch.randn(N, C, H, W, device=device_str)
    ei = _ei(device=device_str)
    layer = ConvMessagePassing(
        (C, H, W), (8, H, W), aggr="mean", aggregator_params=_fast_agg()
    ).to(device_str)
    out = layer(x, ei)
    assert out.shape == (N, 8, H, W)
    assert out.device.type == device_str
    assert torch.isfinite(out).all()


def _run_graph_batch(device_str: str):
    """GraphBatch correctly offsets and stays on the given device."""
    x1 = torch.randn(3, C, H, W, device=device_str)
    x2 = torch.randn(2, C, H, W, device=device_str)
    ei1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long, device=device_str)
    ei2 = torch.tensor([[0], [1]], dtype=torch.long, device=device_str)
    b = GraphBatch([Graph(x1, ei1), Graph(x2, ei2)])
    assert b.node_features.device.type == device_str
    assert b.batch.device.type == device_str
    # g2's only edge: 0→1 offset by 3 nodes  →  3→4
    assert b.edge_index[:, -1].tolist() == [3, 4]


# ──────────────────────────────────────────────────────────────────── #
# CPU (always run)                                                      #
# ──────────────────────────────────────────────────────────────────── #

class TestCPU:
    def test_forward_backward(self):
        _run_forward_backward("cpu")

    def test_mean_aggregation_spatial(self):
        _run_mean_aggregation("cpu")

    def test_graph_batch_offset(self):
        _run_graph_batch("cpu")

    def test_graph_construction_cpu(self):
        x = torch.randn(N, C, H, W)
        ei = _ei()
        g = Graph(x, ei)
        assert g.node_features.device.type == "cpu"

    def test_graph_to_cpu(self):
        g = Graph(torch.randn(N, C, H, W), _ei())
        g.to("cpu")
        assert g.node_features.device.type == "cpu"

    def test_graphbatch_to_cpu(self):
        g1 = Graph(torch.randn(3, C, H, W), None)
        g2 = Graph(torch.randn(2, C, H, W), None)
        b = GraphBatch([g1, g2]).to("cpu")
        assert b.node_features.device.type == "cpu"
        assert b.batch.device.type == "cpu"

    def test_get_device_returns_torch_device(self):
        d = get_device()
        assert isinstance(d, torch.device)

    def test_get_device_device_id_param(self):
        """get_device(device_id=0) should not raise even on CPU-only machines."""
        # On CPU-only machines, device_id is simply ignored.
        d = get_device(device_id=0)
        assert isinstance(d, torch.device)


# ──────────────────────────────────────────────────────────────────── #
# CUDA (skipped when unavailable)                                       #
# ──────────────────────────────────────────────────────────────────── #

@pytest.mark.cuda
class TestCUDA:
    def test_forward_backward(self):
        _run_forward_backward("cuda")

    def test_mean_aggregation_spatial(self):
        _run_mean_aggregation("cuda")

    def test_graph_batch_offset(self):
        _run_graph_batch("cuda")

    def test_graph_to_cuda(self):
        g = Graph(torch.randn(N, C, H, W), _ei()).to("cuda")
        assert g.node_features.device.type == "cuda"
        assert g.edge_index.device.type == "cuda"

    def test_graphbatch_to_cuda(self):
        g1 = Graph(torch.randn(3, C, H, W), None)
        g2 = Graph(torch.randn(2, C, H, W), None)
        b = GraphBatch([g1, g2]).to("cuda")
        assert b.node_features.device.type == "cuda"

    def test_graph_validation_both_on_cuda(self):
        """Graph must accept matching CUDA tensors."""
        x = torch.randn(N, C, H, W, device="cuda")
        ei = _ei(device="cuda")
        g = Graph(x, ei)
        assert g.node_features.device.type == "cuda"

    def test_graph_validation_device_mismatch_raises(self):
        """edge_index on CUDA with node_features on CPU must be rejected."""
        x = torch.randn(N, C, H, W)                    # CPU
        ei = _ei(device="cuda")                         # CUDA
        with pytest.raises(ValueError, match="device"):
            Graph(x, ei)

    def test_get_device_returns_cuda(self):
        assert get_device().type == "cuda"


# ──────────────────────────────────────────────────────────────────── #
# MPS (skipped when unavailable)                                        #
# ──────────────────────────────────────────────────────────────────── #

@pytest.mark.mps
class TestMPS:
    def test_forward_backward(self):
        _run_forward_backward("mps")

    def test_mean_aggregation_spatial(self):
        _run_mean_aggregation("mps")

    def test_graph_to_mps(self):
        g = Graph(torch.randn(N, C, H, W), _ei()).to("mps")
        assert g.node_features.device.type == "mps"

    def test_graphbatch_to_mps(self):
        g1 = Graph(torch.randn(3, C, H, W), None)
        g2 = Graph(torch.randn(2, C, H, W), None)
        b = GraphBatch([g1, g2]).to("mps")
        assert b.node_features.device.type == "mps"

    def test_get_device_returns_mps(self):
        assert get_device().type == "mps"
