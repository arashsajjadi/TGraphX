"""Tests for temporal sampling helpers (v0.2.8)."""
from __future__ import annotations

import pytest
import torch

from tgraphx import (
    Graph,
    TemporalGraphBatch,
    TemporalGraphSequence,
    temporal_window_sample,
    temporal_window_sample_batch,
)


def _snapshot(N=4, F=3, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(N, F)
    src = torch.arange(N, dtype=torch.long)
    dst = (src + 1) % N
    return Graph(x, torch.stack([src, dst], dim=0))


def _seq(T=5, N=4, F=3, seed=0):
    return TemporalGraphSequence(
        graphs=[_snapshot(N, F, seed=seed + t) for t in range(T)],
        timestamps=[float(t) for t in range(T)],
        metadata={"name": "demo"},
    )


# ── temporal_window_sample ────────────────────────────────────────────────────


class TestTemporalWindowSample:
    def test_basic_window(self):
        seq = _seq(T=6)
        sub = temporal_window_sample(seq, 1, 4)
        assert sub.num_snapshots == 3
        assert sub.timestamps == [1.0, 2.0, 3.0]

    def test_full_window(self):
        seq = _seq(T=4)
        sub = temporal_window_sample(seq, 0, 4)
        assert sub.num_snapshots == 4

    def test_single_snapshot(self):
        seq = _seq(T=4)
        sub = temporal_window_sample(seq, 2, 3)
        assert sub.num_snapshots == 1
        assert sub.timestamps == [2.0]

    def test_metadata_preserved_and_window_recorded(self):
        seq = _seq(T=5)
        sub = temporal_window_sample(seq, 1, 3)
        assert sub.metadata["name"] == "demo"
        w = sub.metadata["window"]
        assert w["t_start"] == 1
        assert w["t_end"] == 3
        assert w["source_num_snapshots"] == 5

    def test_no_timestamps(self):
        seq = TemporalGraphSequence(graphs=[_snapshot(seed=t) for t in range(3)])
        sub = temporal_window_sample(seq, 1, 3)
        assert sub.timestamps is None
        assert sub.num_snapshots == 2

    def test_invalid_t_start(self):
        seq = _seq(T=3)
        with pytest.raises(ValueError, match="t_start"):
            temporal_window_sample(seq, -1, 2)
        with pytest.raises(ValueError, match="t_start"):
            temporal_window_sample(seq, 5, 6)

    def test_invalid_t_end(self):
        seq = _seq(T=3)
        with pytest.raises(ValueError, match="t_end"):
            temporal_window_sample(seq, 1, 1)
        with pytest.raises(ValueError, match="t_end"):
            temporal_window_sample(seq, 1, 99)

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="TemporalGraphSequence"):
            temporal_window_sample([1, 2], 0, 1)


# ── temporal_window_sample_batch ──────────────────────────────────────────────


class TestTemporalWindowSampleBatch:
    def test_equal_length_batch(self):
        seqs = [_seq(T=5, seed=i) for i in range(3)]
        batch = TemporalGraphBatch(seqs)
        sub_batch = temporal_window_sample_batch(batch, 1, 4)
        assert sub_batch.num_sequences == 3
        assert sub_batch.max_length == 3
        assert not sub_batch.is_variable_length

    def test_variable_length_clipped(self):
        # Sequences of length 5, 3, 4 — window [1, 4) clipped to lengths.
        seq_5 = _seq(T=5, seed=0)
        seq_3 = _seq(T=3, seed=1)
        seq_4 = _seq(T=4, seed=2)
        batch = TemporalGraphBatch([seq_5, seq_3, seq_4])
        sub_batch = temporal_window_sample_batch(batch, 1, 4)
        assert sub_batch.num_sequences == 3
        # Lengths: min(4, 5) - 1 = 3; min(4, 3) - 1 = 2; min(4, 4) - 1 = 3.
        assert sub_batch.lengths == [3, 2, 3]
        assert sub_batch.is_variable_length

    def test_short_sequence_raises(self):
        # t_start exceeds shortest sequence length.
        seq_5 = _seq(T=5, seed=0)
        seq_2 = _seq(T=2, seed=1)
        batch = TemporalGraphBatch([seq_5, seq_2])
        with pytest.raises(ValueError, match="cannot start a window"):
            temporal_window_sample_batch(batch, 3, 5)

    def test_invalid_window(self):
        batch = TemporalGraphBatch([_seq(T=5, seed=i) for i in range(2)])
        with pytest.raises(ValueError, match="t_start"):
            temporal_window_sample_batch(batch, -1, 2)
        with pytest.raises(ValueError, match="t_end"):
            temporal_window_sample_batch(batch, 2, 2)

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="TemporalGraphBatch"):
            temporal_window_sample_batch([1, 2], 0, 1)
