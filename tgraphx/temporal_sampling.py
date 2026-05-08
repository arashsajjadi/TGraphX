"""Window sampling for :class:`TemporalGraphSequence` / batches (v0.2.8).

These helpers cut a contiguous time-window out of a temporal sequence
and return a new :class:`~tgraphx.TemporalGraphSequence`.  They are
shape-preserving and lossless: snapshot graphs are referenced (not
copied) where possible, so memory is minimal.

For batches, :func:`temporal_window_sample_batch` applies the same
window to every sequence and returns a new
:class:`~tgraphx.TemporalGraphBatch`.  Variable-length sequences are
respected — sequences shorter than the requested ``t_end`` end at their
own length.

These are the *temporal* counterparts of the homogeneous sampling
helpers in :mod:`tgraphx.sampling`.
"""
from __future__ import annotations

from typing import Optional

from .core.temporal import TemporalGraphSequence
from .core.temporal_batch import TemporalGraphBatch

__all__ = [
    "temporal_window_sample",
    "temporal_window_sample_batch",
]


def temporal_window_sample(
    seq: TemporalGraphSequence,
    t_start: int,
    t_end: int,
) -> TemporalGraphSequence:
    """Return the contiguous window ``[t_start, t_end)`` of ``seq``.

    Args:
        seq: Source :class:`TemporalGraphSequence`.
        t_start: Inclusive start index (>= 0).
        t_end: Exclusive end index (<= ``num_snapshots``); must be
            strictly greater than ``t_start``.

    Returns:
        New :class:`TemporalGraphSequence` containing the snapshots and
        timestamps in the requested range.  Metadata is copied verbatim;
        a ``window`` entry is added under ``metadata['window']``.

    Raises:
        ValueError: On invalid window bounds.
    """
    if not isinstance(seq, TemporalGraphSequence):
        raise TypeError(
            f"seq must be a TemporalGraphSequence; got {type(seq)}"
        )
    n = seq.num_snapshots
    if not (0 <= t_start < n):
        raise ValueError(
            f"t_start={t_start} out of range [0, {n})"
        )
    if not (t_start < t_end <= n):
        raise ValueError(
            f"t_end={t_end} must satisfy t_start < t_end <= num_snapshots ({n})"
        )

    graphs = [seq[t] for t in range(t_start, t_end)]
    ts = seq.timestamps
    sub_ts: Optional[list]
    if ts is None:
        sub_ts = None
    else:
        sub_ts = list(ts[t_start:t_end])

    base_meta = dict(seq.metadata) if isinstance(seq.metadata, dict) else {}
    base_meta = dict(base_meta)
    base_meta["window"] = {
        "t_start": int(t_start),
        "t_end": int(t_end),
        "source_num_snapshots": int(n),
    }
    return TemporalGraphSequence(
        graphs=graphs,
        timestamps=sub_ts,
        metadata=base_meta,
    )


def temporal_window_sample_batch(
    batch: TemporalGraphBatch,
    t_start: int,
    t_end: int,
) -> TemporalGraphBatch:
    """Apply :func:`temporal_window_sample` to every sequence in a batch.

    Sequences shorter than ``t_end`` end at their own length (the window
    is clipped per sequence).  Sequences whose length is ``<= t_start``
    contribute no snapshots and would invalidate the batch — they are
    rejected with a ``ValueError``.

    Args:
        batch: Source :class:`TemporalGraphBatch`.
        t_start: Inclusive start index applied to every sequence.
        t_end: Exclusive upper bound; clipped to each sequence's length.

    Returns:
        New :class:`TemporalGraphBatch`.  ``mode='auto'`` is used so the
        result is equal-length when all sub-sequences come out the same
        length and variable-length otherwise.
    """
    if not isinstance(batch, TemporalGraphBatch):
        raise TypeError(
            f"batch must be a TemporalGraphBatch; got {type(batch)}"
        )
    if t_start < 0:
        raise ValueError(f"t_start must be >= 0; got {t_start}")
    if t_end <= t_start:
        raise ValueError(
            f"t_end={t_end} must be strictly greater than t_start={t_start}"
        )

    sub_seqs = []
    for i, length in enumerate(batch.lengths):
        if length <= t_start:
            raise ValueError(
                f"Sequence {i} has length {length}; cannot start a window "
                f"at t_start={t_start}.  Filter sequences before windowing "
                f"or pick a smaller t_start."
            )
        local_end = min(t_end, length)
        # ``batch._sequences`` is the canonical source of truth.
        seq = batch._sequences[i]  # noqa: SLF001 — batch is built from a list
        sub_seqs.append(temporal_window_sample(seq, t_start, local_end))

    return TemporalGraphBatch(sub_seqs, mode="auto")
