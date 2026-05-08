"""TemporalGraphBatch + readouts + models tests (v0.2.5)."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tgraphx import Graph
from tgraphx.core.temporal import TemporalGraphSequence
from tgraphx.core.temporal_batch import TemporalGraphBatch
from tgraphx.layers.temporal_readout import temporal_readout
from tgraphx.models.temporal_models import TemporalGraphClassifier, TemporalGraphRegressor


def _seq(n_snap=3, n_nodes=4, dim=8, seed=0):
    torch.manual_seed(seed)
    graphs = [Graph(torch.randn(n_nodes, dim), None) for _ in range(n_snap)]
    return TemporalGraphSequence(graphs, timestamps=[float(i) for i in range(n_snap)])


# ── TemporalGraphBatch ────────────────────────────────────────────────────────

class TestTemporalBatch:
    def test_equal_length(self):
        b = TemporalGraphBatch([_seq(3, 4, 8, 0), _seq(3, 4, 8, 1)])
        assert b.num_sequences == 2
        assert b.max_length == 3
        assert not b.is_variable_length

    def test_variable_length(self):
        b = TemporalGraphBatch([_seq(3, 4, 8, 0), _seq(2, 4, 8, 1)])
        assert b.is_variable_length
        assert b.max_length == 3

    def test_iteration_equal_length(self):
        b = TemporalGraphBatch([_seq(3, 4, 8, 0), _seq(3, 4, 8, 1)])
        snaps = list(b)
        assert len(snaps) == 3
        for t, gb, mask in snaps:
            assert mask.tolist() == [True, True]
            assert gb.num_graphs == 2

    def test_iteration_variable_length(self):
        b = TemporalGraphBatch([_seq(3, 4, 8, 0), _seq(2, 4, 8, 1)])
        snaps = list(b)
        assert len(snaps) == 3
        # Only sequence 0 has snapshot 2.
        t2, gb2, mask2 = snaps[2]
        assert mask2.tolist() == [True, False]
        assert gb2.num_graphs == 1

    def test_timestamps_padded(self):
        b = TemporalGraphBatch([_seq(3, 4, 8, 0), _seq(2, 4, 8, 1)])
        ts = b.timestamps_padded
        assert ts is not None
        assert ts.shape == (2, 3)
        assert ts[0, 2].item() == 2.0
        # Sequence 1 has only 2 snapshots; index 2 is NaN.
        assert torch.isnan(ts[1, 2])

    def test_force_equal_raises(self):
        with pytest.raises(ValueError, match="same length"):
            TemporalGraphBatch([_seq(3, 4, 8, 0), _seq(2, 4, 8, 1)], mode="equal")

    def test_to_cpu(self):
        b = TemporalGraphBatch([_seq(2, 3, 4, 0), _seq(2, 3, 4, 1)])
        b2 = b.to("cpu")
        assert b2.num_sequences == 2

    def test_snapshot_accessor(self):
        b = TemporalGraphBatch([_seq(3, 4, 8, 0), _seq(2, 4, 8, 1)])
        gb, mask = b.snapshot(2)
        assert gb.num_graphs == 1
        assert mask.tolist() == [True, False]
        with pytest.raises(IndexError):
            b.snapshot(99)


# ── temporal_readout ──────────────────────────────────────────────────────────

class TestTemporalReadout:
    def test_last_no_mask(self):
        emb = torch.randn(4, 3, 8)
        out = temporal_readout(emb, "last")
        assert out.shape == (3, 8)
        assert torch.equal(out, emb[-1])

    def test_mean_no_mask(self):
        emb = torch.randn(4, 3, 8)
        out = temporal_readout(emb, "mean")
        assert torch.allclose(out, emb.mean(0))

    def test_max_no_mask(self):
        emb = torch.randn(4, 3, 8)
        out = temporal_readout(emb, "max")
        assert torch.equal(out, emb.max(0).values)

    def test_last_with_mask(self):
        emb = torch.randn(3, 2, 4)
        mask = torch.tensor([[True, True], [True, False], [False, False]])
        out = temporal_readout(emb, "last", mask)
        # batch 0: last True at t=1 → emb[1, 0]
        # batch 1: last True at t=0 → emb[0, 1]
        assert torch.equal(out[0], emb[1, 0])
        assert torch.equal(out[1], emb[0, 1])

    def test_mean_with_mask(self):
        emb = torch.ones(3, 2, 4)
        mask = torch.tensor([[True, True], [True, False], [True, False]])
        out = temporal_readout(emb, "mean", mask)
        # batch 0: 3 valid → mean = 1.0
        # batch 1: 1 valid → mean = 1.0
        assert torch.allclose(out, torch.ones(2, 4))

    def test_max_with_mask(self):
        emb = torch.tensor([
            [[1.0], [10.0]],
            [[2.0], [-5.0]],
            [[3.0], [-1.0]],
        ])  # shape (3, 2, 1)
        mask = torch.tensor([[True, False], [True, False], [True, False]])
        out = temporal_readout(emb, "max", mask)
        assert out[0].item() == 3.0  # max of [1,2,3]
        # batch 1 all-masked → 0
        assert out[1].item() == 0.0

    def test_invalid_mode(self):
        emb = torch.randn(2, 1, 4)
        with pytest.raises(ValueError, match="mode"):
            temporal_readout(emb, "bad")

    def test_wrong_shape(self):
        with pytest.raises(ValueError, match="3-D"):
            temporal_readout(torch.randn(4, 3), "last")


# ── Temporal models ──────────────────────────────────────────────────────────

class _MeanPoolBase(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, gb):
        x = self.lin(gb.node_features)
        out = torch.zeros(gb.num_graphs, x.size(1))
        out = out.index_add(0, gb.batch, x)
        cnt = torch.zeros(gb.num_graphs).index_add(0, gb.batch, torch.ones(x.size(0)))
        return out / cnt.unsqueeze(1).clamp_min(1.0)


class TestTemporalModels:
    def test_classifier_equal_length(self):
        base = _MeanPoolBase(8, 16)
        clf = TemporalGraphClassifier(base, feature_dim=16, num_classes=3, readout="last")
        b = TemporalGraphBatch([_seq(3, 4, 8, 0), _seq(3, 4, 8, 1)])
        out = clf(b)
        assert out.shape == (2, 3)

    def test_classifier_variable_length(self):
        base = _MeanPoolBase(8, 16)
        clf = TemporalGraphClassifier(base, feature_dim=16, num_classes=3, readout="mean")
        b = TemporalGraphBatch([_seq(3, 4, 8, 0), _seq(2, 4, 8, 1)])
        out = clf(b)
        assert out.shape == (2, 3)
        assert torch.isfinite(out).all()

    def test_classifier_backward(self):
        base = _MeanPoolBase(8, 16)
        clf = TemporalGraphClassifier(base, feature_dim=16, num_classes=3, readout="last")
        b = TemporalGraphBatch([_seq(2, 3, 8, 0), _seq(2, 3, 8, 1)])
        out = clf(b)
        out.sum().backward()
        for p in clf.parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all()

    def test_regressor(self):
        base = _MeanPoolBase(8, 16)
        reg = TemporalGraphRegressor(base, feature_dim=16, out_dim=1, readout="mean")
        b = TemporalGraphBatch([_seq(3, 4, 8, 0), _seq(2, 4, 8, 1)])
        out = reg(b)
        assert out.shape == (2, 1)

    def test_invalid_readout(self):
        with pytest.raises(ValueError, match="readout"):
            TemporalGraphClassifier(
                _MeanPoolBase(4, 4), feature_dim=4, num_classes=2, readout="bogus",
            )

    def test_inconsistent_base_output_raises(self):
        class BadBase(nn.Module):
            def forward(self, gb):
                return torch.randn(gb.num_graphs)  # 1-D, should be 2-D
        clf = TemporalGraphClassifier(BadBase(), feature_dim=4, num_classes=2)
        b = TemporalGraphBatch([_seq(2, 3, 8, 0)])
        with pytest.raises(ValueError, match="2-D"):
            clf(b)
