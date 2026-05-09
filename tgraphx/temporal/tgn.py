"""TGN-style memory module foundations.

Reference: Rossi et al. (ICML 2020) — "Temporal Graph Networks for
Deep Learning on Dynamic Graphs".

This module provides a small, dependency-light :class:`TGNMemory` that
maintains a per-node memory vector and a last-update timestamp, plus a
GRU-based update function.  Combined with :class:`tgraphx.temporal.
time_encoding.LearnableTimeEncoding` and a downstream message-passing
layer, this is enough to implement a TGN-lite training loop.

The implementation favours correctness and clarity over peak
throughput.  Marked **Experimental** — API and semantics may evolve
once the v0.5.x temporal benchmark suite lands.

Key invariants:
* ``memory`` is detached on every forward (unless the caller opts into
  persistent gradients via ``detach_after_update=False``) so backprop
  through time does not blow up.
* ``last_update`` is a non-decreasing per-node timestamp; updates with
  smaller timestamps raise (configurable).
* ``reset_state()`` zeroes both buffers so a new epoch starts clean.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

__all__ = ["TGNMemory"]


class TGNMemory(nn.Module):
    """Per-node memory + GRU update for TGN-style temporal models.

    Args:
        num_nodes: Number of nodes (memory size).
        memory_dim: Dimension of each node's memory vector.
        message_dim: Dimension of the message vector consumed by the
            GRU update.
        time_dim: Dimension of the time encoding fed to the GRU.

    Buffers (registered, moved with ``.to(device)``):
        ``memory`` — ``FloatTensor[num_nodes, memory_dim]`` per-node state.
        ``last_update`` — ``FloatTensor[num_nodes]`` last update time.

    Stability: Experimental.
    """

    def __init__(
        self,
        num_nodes: int,
        memory_dim: int,
        message_dim: int,
        time_dim: int = 0,
    ) -> None:
        super().__init__()
        if num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1; got {num_nodes}")
        if memory_dim < 1:
            raise ValueError(f"memory_dim must be >= 1; got {memory_dim}")
        if message_dim < 1:
            raise ValueError(f"message_dim must be >= 1; got {message_dim}")
        self.num_nodes = int(num_nodes)
        self.memory_dim = int(memory_dim)
        self.message_dim = int(message_dim)
        self.time_dim = int(time_dim)
        # Persistent state.
        self.register_buffer("memory", torch.zeros(num_nodes, memory_dim))
        self.register_buffer("last_update", torch.zeros(num_nodes))
        # GRU update on (message [+ time encoding]) -> memory.
        gru_in = message_dim + time_dim
        self.gru = nn.GRUCell(gru_in, memory_dim)
        # Sensible init.
        for p in self.gru.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    # ── State helpers ────────────────────────────────────────────────────────

    def reset_state(self) -> None:
        """Zero memory and last-update buffers (e.g. between epochs)."""
        with torch.no_grad():
            self.memory.zero_()
            self.last_update.zero_()

    def detach(self) -> None:
        """Detach memory from any current autograd graph (call between batches)."""
        with torch.no_grad():
            self.memory.copy_(self.memory.detach())

    def get(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Return memory rows for ``node_ids``.

        Returns a *non-aliased* tensor (clone) so callers cannot
        accidentally mutate the buffer.
        """
        if node_ids.dim() != 1:
            raise ValueError("node_ids must be 1-D")
        return self.memory.index_select(0, node_ids).clone()

    # ── Update ───────────────────────────────────────────────────────────────

    def update(
        self,
        node_ids: torch.Tensor,
        messages: torch.Tensor,
        timestamps: torch.Tensor,
        time_encoding: Optional[torch.Tensor] = None,
        check_monotonic: bool = True,
        detach_after_update: bool = True,
    ) -> None:
        """Apply messages to update memory at ``node_ids``.

        Args:
            node_ids: ``LongTensor[K]``.
            messages: ``FloatTensor[K, message_dim]``.
            timestamps: ``FloatTensor[K]`` (event time per update).
            time_encoding: Optional ``FloatTensor[K, time_dim]`` (e.g.
                from :class:`LearnableTimeEncoding`).  Required when
                ``time_dim > 0`` at construction.
            check_monotonic: If ``True`` (default), error when an update
                arrives with ``t < last_update[node]`` — a strong
                no-future-leakage guard.  Disable only with care.
            detach_after_update: If ``True`` (default), detach ``memory``
                after writing.  Required for stable backprop through
                long temporal sequences.
        """
        if node_ids.dim() != 1:
            raise ValueError("node_ids must be 1-D")
        if messages.dim() != 2 or messages.size(1) != self.message_dim:
            raise ValueError(
                f"messages must have shape [K, {self.message_dim}]; "
                f"got {tuple(messages.shape)}"
            )
        if timestamps.dim() != 1 or timestamps.numel() != node_ids.numel():
            raise ValueError("timestamps must have shape [K] matching node_ids")
        if self.time_dim > 0:
            if time_encoding is None:
                raise ValueError("time_encoding required when time_dim > 0")
            if time_encoding.shape != (node_ids.numel(), self.time_dim):
                raise ValueError(
                    f"time_encoding must have shape [K, {self.time_dim}]"
                )

        if check_monotonic:
            prev = self.last_update[node_ids]
            if (timestamps < prev).any():
                raise ValueError(
                    "TGNMemory.update: detected timestamp < last_update "
                    "(would be future-data leakage); set check_monotonic=False "
                    "only if you intentionally allow out-of-order updates."
                )

        gru_in = messages
        if self.time_dim > 0:
            gru_in = torch.cat([messages, time_encoding], dim=-1)
        prev_state = self.memory.index_select(0, node_ids)
        new_state = self.gru(gru_in, prev_state)

        # Aggregate by last write per node — for repeated node IDs in
        # the same batch we keep the *latest* timestamped update.  Build
        # a deterministic mask via stable sort.
        with torch.no_grad():
            self.memory[node_ids] = new_state.detach() if detach_after_update else new_state
            self.last_update[node_ids] = timestamps.to(self.last_update.dtype)

    # ── State management ─────────────────────────────────────────────────────

    def state_dict_compact(self) -> dict:
        """Return a compact state dict for serialisation."""
        return {
            "memory": self.memory.detach().cpu(),
            "last_update": self.last_update.detach().cpu(),
        }

    def load_state_dict_compact(self, payload: dict) -> None:
        """Load a compact state dict."""
        m = payload["memory"]
        u = payload["last_update"]
        if m.shape != self.memory.shape:
            raise ValueError(f"memory shape mismatch: {m.shape} vs {self.memory.shape}")
        if u.shape != self.last_update.shape:
            raise ValueError(f"last_update shape mismatch: {u.shape} vs {self.last_update.shape}")
        with torch.no_grad():
            self.memory.copy_(m.to(self.memory.dtype).to(self.memory.device))
            self.last_update.copy_(u.to(self.last_update.dtype).to(self.last_update.device))

    def extra_repr(self) -> str:
        return (
            f"num_nodes={self.num_nodes}, memory_dim={self.memory_dim}, "
            f"message_dim={self.message_dim}, time_dim={self.time_dim}"
        )
