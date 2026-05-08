"""Experimental TemporalGraphSequence container.

.. experimental::
    This API is **🧪 Experimental**.  The container stores a sequence of
    graph snapshots with optional timestamps but does NOT implement any
    temporal GNN layers.  API may change in future releases.

Usage::

    from tgraphx.core.temporal import TemporalGraphSequence
    from tgraphx import Graph

    seq = TemporalGraphSequence(
        graphs=[g_t0, g_t1, g_t2],
        timestamps=[0.0, 1.0, 2.0],  # optional
    )
    seq.to("cuda")
    for t, g in seq:
        out = model(g.node_features, g.edge_index)
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Tuple, Union

import torch


class TemporalGraphSequence:
    """🧪 Experimental: container for a temporal sequence of graph snapshots.

    Stores a list of :class:`~tgraphx.Graph` objects representing the same
    graph (or graph topology) at different time steps, with optional
    scalar timestamps.

    This is a **data container only** — it does not implement any temporal
    GNN message-passing.  Typical usage: iterate over the sequence and feed
    each snapshot to a stateless GNN layer, optionally maintaining a hidden
    state externally (e.g. via an LSTM).

    Args:
        graphs: List of :class:`~tgraphx.Graph` snapshots.
        timestamps: Optional list / tensor of scalar timestamps, one per
            snapshot.  Must have the same length as ``graphs``.
        metadata: Optional dict of run/experiment metadata.
    """

    def __init__(
        self,
        graphs: list,
        timestamps: Optional[Union[list, torch.Tensor]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        if not isinstance(graphs, list) or len(graphs) == 0:
            raise ValueError("graphs must be a non-empty list of Graph instances.")
        for i, g in enumerate(graphs):
            if not hasattr(g, "node_features") or not hasattr(g, "edge_index"):
                raise ValueError(
                    f"graphs[{i}] does not appear to be a Graph instance "
                    f"(missing 'node_features' or 'edge_index')."
                )
        if timestamps is not None:
            if isinstance(timestamps, torch.Tensor):
                ts = timestamps.tolist()
            else:
                ts = list(timestamps)
            if len(ts) != len(graphs):
                raise ValueError(
                    f"timestamps length ({len(ts)}) must match "
                    f"graphs length ({len(graphs)})."
                )
            self._timestamps: Optional[list] = ts
        else:
            self._timestamps = None

        self._graphs: list = list(graphs)
        self.metadata = metadata

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def num_snapshots(self) -> int:
        return len(self._graphs)

    @property
    def timestamps(self) -> Optional[list]:
        return self._timestamps

    # ── Accessors ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.num_snapshots

    def __getitem__(self, idx: int):
        if not -len(self._graphs) <= idx < len(self._graphs):
            raise IndexError(
                f"Index {idx} out of range for sequence of length "
                f"{self.num_snapshots}."
            )
        return self._graphs[idx]

    def __iter__(self) -> Iterator[Tuple[Optional[float], object]]:
        """Iterate as ``(timestamp, graph)`` pairs.

        ``timestamp`` is ``None`` if no timestamps were provided.

        Example::

            for t, g in seq:
                out = model(g.node_features, g.edge_index)
        """
        ts = self._timestamps if self._timestamps is not None else [None] * len(self._graphs)
        return iter(zip(ts, self._graphs))

    # ── Device movement ───────────────────────────────────────────────────────

    def to(self, device, dtype: torch.dtype | None = None) -> "TemporalGraphSequence":
        """Move all graph tensors to ``device``."""
        moved = []
        for g in self._graphs:
            moved.append(g.to(device, dtype=dtype) if dtype is not None else g.to(device))
        return TemporalGraphSequence(moved, self._timestamps, self.metadata)

    def cpu(self) -> "TemporalGraphSequence":
        return self.to("cpu")

    def cuda(self, device_id: int | None = None) -> "TemporalGraphSequence":
        d = "cuda" if device_id is None else f"cuda:{device_id}"
        return self.to(d)

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        ts_str = (
            f"timestamps=[{self._timestamps[0]:.2f}..{self._timestamps[-1]:.2f}]"
            if self._timestamps else "timestamps=None"
        )
        return (
            f"TemporalGraphSequence("
            f"num_snapshots={self.num_snapshots}, "
            f"{ts_str}"
            f")  [🧪 Experimental]"
        )
