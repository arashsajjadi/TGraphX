"""Doubling cube state.

The first PPO training loop ignores the cube entirely.  These classes
exist so the env API can be extended later without breaking changes.

CubeState.available = True  →  either player may double
CubeState.owner     = None  →  cube is centred (either may double)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CubeState:
    value: int            = 1
    owner: Optional[int]  = None   # None = centred, 0 or 1 = owns it
    available: bool       = True   # False in Crawford game

    def doubled_by(self, player: int) -> "CubeState":
        """Return a new CubeState after *player* doubles."""
        return CubeState(value=self.value * 2, owner=1 - player,
                         available=True)

    @property
    def is_centred(self) -> bool:
        return self.owner is None

    def can_double(self, player: int) -> bool:
        if not self.available:
            return False
        return self.is_centred or self.owner == player
