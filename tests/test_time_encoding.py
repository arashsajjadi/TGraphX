"""Tests for tgraphx.temporal.time_encoding."""
from __future__ import annotations

import math

import pytest
import torch

from tgraphx.temporal import LearnableTimeEncoding, sinusoidal_time_encoding


# ── sinusoidal_time_encoding ─────────────────────────────────────────────────


class TestSinusoidalEncoding:
    def test_shape(self):
        t = torch.tensor([0.0, 1.0, 2.0, 3.0])
        enc = sinusoidal_time_encoding(t, dim=8)
        assert enc.shape == (4, 8)

    def test_batched_shape(self):
        t = torch.zeros(3, 5)
        enc = sinusoidal_time_encoding(t, dim=4)
        assert enc.shape == (3, 5, 4)

    def test_dtype_is_float32(self):
        t = torch.tensor([1.0, 2.0])
        enc = sinusoidal_time_encoding(t, dim=4)
        assert enc.dtype == torch.float32

    def test_integer_input_accepted(self):
        t = torch.tensor([0, 1, 2], dtype=torch.long)
        enc = sinusoidal_time_encoding(t, dim=4)
        assert enc.shape == (3, 4)
        assert enc.dtype == torch.float32

    def test_t_zero_columns(self):
        # At t=0: even cols are sin(0)=0, odd cols are cos(0)=1.
        enc = sinusoidal_time_encoding(torch.zeros(1), dim=8)
        assert torch.allclose(enc[0, 0::2], torch.zeros(4))
        assert torch.allclose(enc[0, 1::2], torch.ones(4))

    def test_finite(self):
        t = torch.linspace(0, 1e6, 100)
        enc = sinusoidal_time_encoding(t, dim=64)
        assert torch.isfinite(enc).all()

    def test_deterministic(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        a = sinusoidal_time_encoding(t, dim=8)
        b = sinusoidal_time_encoding(t, dim=8)
        assert torch.equal(a, b)

    def test_no_global_rng_pollution(self):
        torch.manual_seed(0)
        before = torch.rand(3)
        torch.manual_seed(0)
        sinusoidal_time_encoding(torch.tensor([1.0, 2.0]), dim=4)
        after = torch.rand(3)
        assert torch.equal(before, after)

    def test_rejects_odd_dim(self):
        with pytest.raises(ValueError, match="dim"):
            sinusoidal_time_encoding(torch.tensor([0.0]), dim=5)

    def test_rejects_zero_dim(self):
        with pytest.raises(ValueError, match="dim"):
            sinusoidal_time_encoding(torch.tensor([0.0]), dim=0)

    def test_norm_bounded(self):
        # Each pair sin²+cos²=1, so ||enc||² == dim/2.
        t = torch.tensor([0.5, 1.0, 100.0])
        enc = sinusoidal_time_encoding(t, dim=16)
        norms = enc.pow(2).sum(dim=-1)
        assert torch.allclose(norms, torch.full((3,), 8.0), atol=1e-4)


# ── LearnableTimeEncoding ────────────────────────────────────────────────────


class TestLearnableTimeEncoding:
    def test_shape(self):
        m = LearnableTimeEncoding(dim=8)
        out = m(torch.tensor([0.0, 1.0, 2.0]))
        assert out.shape == (3, 8)

    def test_batched_shape(self):
        m = LearnableTimeEncoding(dim=4)
        out = m(torch.zeros(2, 3))
        assert out.shape == (2, 3, 4)

    def test_finite_forward(self):
        m = LearnableTimeEncoding(dim=8)
        out = m(torch.linspace(0, 1e3, 50))
        assert torch.isfinite(out).all()

    def test_finite_backward(self):
        m = LearnableTimeEncoding(dim=8)
        t = torch.linspace(0, 10, 16)
        out = m(t).sum()
        out.backward()
        for p in m.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()

    def test_gradients_nonzero(self):
        m = LearnableTimeEncoding(dim=8)
        t = torch.tensor([0.5, 1.5, 2.5])
        m(t).sum().backward()
        # Linear params and at least one periodic channel must have non-zero grad.
        assert m.linear_w.grad.abs().item() > 0
        # Some periodic param has non-zero grad (sin' * t):
        assert m.periodic_w.grad.abs().sum().item() > 0

    def test_rejects_dim_lt_2(self):
        with pytest.raises(ValueError, match="dim"):
            LearnableTimeEncoding(dim=1)

    def test_repr(self):
        m = LearnableTimeEncoding(dim=4)
        assert "dim=4" in repr(m)

    def test_linear_channel_is_zero_at_init_when_b_zero(self):
        # Default init: bias = 0, so linear channel at t=0 should be 0.
        m = LearnableTimeEncoding(dim=4)
        out = m(torch.tensor([0.0]))
        assert abs(out[0, 0].detach().item()) < 1e-6


# ── Validation and edge cases ─────────────────────────────────────────────────


class TestValidationAndEdgeCases:
    def test_sinusoidal_base_zero_raises(self):
        with pytest.raises(ValueError, match="base"):
            sinusoidal_time_encoding(torch.tensor([1.0]), dim=4, base=0.0)

    def test_sinusoidal_negative_base_raises(self):
        with pytest.raises(ValueError, match="base"):
            sinusoidal_time_encoding(torch.tensor([1.0]), dim=4, base=-1.0)

    def test_sinusoidal_dim_2_works(self):
        enc = sinusoidal_time_encoding(torch.tensor([0.0, 1.0]), dim=2)
        assert enc.shape == (2, 2)

    def test_sinusoidal_large_timestamp_finite(self):
        enc = sinusoidal_time_encoding(torch.tensor([1e9, 1e12]), dim=8)
        assert torch.isfinite(enc).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sinusoidal_cuda_device_preserved(self):
        t = torch.tensor([1.0, 2.0, 3.0]).cuda()
        enc = sinusoidal_time_encoding(t, dim=8)
        assert enc.device.type == "cuda"
        assert torch.isfinite(enc).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_learnable_cuda_forward(self):
        m = LearnableTimeEncoding(dim=4).cuda()
        t = torch.tensor([0.5, 1.5]).cuda()
        out = m(t)
        assert out.device.type == "cuda"
        assert torch.isfinite(out).all()
