"""Experimental TemporalGraphBatch — batching of temporal sequences.

.. experimental::
    🧪 Batches a list of :class:`TemporalGraphSequence` objects with two
    supported modes:

    * **Equal-length** sequences are stacked snapshot-wise: snapshot ``t``
      from each sequence becomes a single :class:`GraphBatch`.
    * **Variable-length** sequences are right-padded with a length mask.
      When iterating, you receive ``(t, GraphBatch_t, mask_t)`` tuples
      where ``mask_t[b]`` is ``False`` if sequence ``b`` has no snapshot
      at time ``t``.

The batch never copies snapshot tensors that are already on the same
device; it only adjusts the GraphBatch construction per time step.
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import torch

from .graph import GraphBatch
from .temporal import TemporalGraphSequence

__all__ = ["TemporalGraphBatch"]


class TemporalGraphBatch:
    """🧪 Experimental: a batch of temporal graph sequences.

    Args:
        sequences: List of :class:`TemporalGraphSequence`.  All sequences
            must use compatible per-snapshot ``Graph`` shapes
            (consistent with :class:`GraphBatch` requirements).
        mode: ``"auto"`` (default) — equal-length if all sequences have the
            same length, otherwise variable-length with mask.
            Force ``"equal"`` to raise on length mismatch, or
            ``"variable"`` to always use the masked variable-length path.

    Attributes:
        max_length: Maximum number of snapshots across all sequences.
        is_variable_length: ``True`` if sequences have differing lengths.
        timestamps: List of length-padded timestamp lists or ``None``.

    Iteration::

        batch = TemporalGraphBatch([seq_a, seq_b, seq_c])
        for t, graph_batch, mask in batch:
            # graph_batch: GraphBatch over the active sequences at time t
            # mask: BoolTensor [B] — True for sequences whose snapshot t exists
            ...

    Notes:
        Variable-length mode reduces the per-time-step ``GraphBatch`` to
        only those sequences that have a snapshot at that time step; the
        ``mask`` lets the user reconstruct per-sequence outputs.
    """

    def __init__(
        self,
        sequences: List[TemporalGraphSequence],
        mode: str = "auto",
    ) -> None:
        if not isinstance(sequences, list) or len(sequences) == 0:
            raise ValueError("TemporalGraphBatch requires a non-empty list of sequences.")
        for i, s in enumerate(sequences):
            if not isinstance(s, TemporalGraphSequence):
                raise TypeError(
                    f"sequences[{i}] is not a TemporalGraphSequence; got {type(s)}"
                )
        if mode not in ("auto", "equal", "variable"):
            raise ValueError(f"mode must be 'auto', 'equal', or 'variable'; got {mode!r}")

        self._sequences = list(sequences)
        lengths = [s.num_snapshots for s in sequences]
        max_len = max(lengths)
        is_variable = len(set(lengths)) > 1

        if mode == "equal" and is_variable:
            raise ValueError(
                f"mode='equal' requires all sequences to have the same length; "
                f"got lengths {lengths}"
            )

        self._lengths: List[int] = lengths
        self._max_length: int = max_len
        self._is_variable: bool = is_variable or (mode == "variable")

        # Collate timestamps into a [B, max_len] padded tensor (NaN where absent).
        timestamps: Optional[torch.Tensor] = None
        if all(s.timestamps is not None for s in sequences):
            ts = torch.full((len(sequences), max_len), float("nan"))
            for b, s in enumerate(sequences):
                ts_list = s.timestamps or []
                for t, val in enumerate(ts_list):
                    ts[b, t] = float(val)
            timestamps = ts
        self._timestamps_padded = timestamps

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def num_sequences(self) -> int:
        return len(self._sequences)

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def is_variable_length(self) -> bool:
        return self._is_variable

    @property
    def lengths(self) -> List[int]:
        return list(self._lengths)

    @property
    def timestamps_padded(self) -> Optional[torch.Tensor]:
        return self._timestamps_padded

    # ── Iteration ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._max_length

    def __iter__(self) -> Iterator[Tuple[int, GraphBatch, torch.Tensor]]:
        """Yield ``(time_index, GraphBatch_t, mask_t)`` per snapshot index."""
        for t in range(self._max_length):
            mask_list: List[bool] = []
            graphs_t = []
            for s, L in zip(self._sequences, self._lengths):
                if t < L:
                    graphs_t.append(s[t])
                    mask_list.append(True)
                else:
                    mask_list.append(False)
            mask = torch.tensor(mask_list, dtype=torch.bool)
            if not graphs_t:
                # Should not happen since max_length is at least 1.
                continue
            graph_batch = GraphBatch(graphs_t)
            yield t, graph_batch, mask

    def snapshot(self, t: int) -> Tuple[GraphBatch, torch.Tensor]:
        """Return ``(GraphBatch_t, mask_t)`` for a specific time index."""
        if not 0 <= t < self._max_length:
            raise IndexError(
                f"time index {t} out of range [0, {self._max_length})"
            )
        mask_list = [t < L for L in self._lengths]
        mask = torch.tensor(mask_list, dtype=torch.bool)
        graphs_t = [s[t] for s, L in zip(self._sequences, self._lengths) if t < L]
        return GraphBatch(graphs_t), mask

    # ── Device movement ───────────────────────────────────────────────────────

    def to(self, device, dtype: torch.dtype | None = None) -> "TemporalGraphBatch":
        moved = [s.to(device, dtype=dtype) if dtype is not None else s.to(device)
                 for s in self._sequences]
        return TemporalGraphBatch(moved, mode="variable" if self._is_variable else "auto")

    def cpu(self) -> "TemporalGraphBatch":
        return self.to("cpu")

    def cuda(self, device_id: int | None = None) -> "TemporalGraphBatch":
        d = "cuda" if device_id is None else f"cuda:{device_id}"
        return self.to(d)

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"TemporalGraphBatch("
            f"num_sequences={self.num_sequences}, "
            f"max_length={self.max_length}, "
            f"is_variable_length={self._is_variable}"
            f")  [🧪 Experimental]"
        )
