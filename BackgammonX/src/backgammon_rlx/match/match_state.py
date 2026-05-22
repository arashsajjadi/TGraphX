"""Match-level state (score, Crawford, post-Crawford).

Placeholder for proper match-play support.  The BackgammonEnv uses
raw game states; MatchState wraps it for multi-game match tracking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .cube import CubeState


@dataclass
class MatchState:
    match_length: int          = 1    # 0 = money game
    score: list                = field(default_factory=lambda: [0, 0])
    crawford_triggered: bool   = False
    crawford_game: bool        = False   # True in the one Crawford game
    post_crawford: bool        = False
    cube: CubeState            = field(default_factory=CubeState)

    def record_game(self, winner: int, game_score: int) -> None:
        """Update match score after a game."""
        self.score[winner] += self.cube.value * game_score
        self._update_crawford()

    def _update_crawford(self) -> None:
        if self.match_length <= 0:
            return
        target = self.match_length
        for p in range(2):
            if self.score[p] == target - 1 and not self.crawford_triggered:
                self.crawford_triggered = True
                self.crawford_game      = True
                return

        if self.crawford_game:
            self.crawford_game = False
            self.post_crawford = True
            self.cube.available = True

        if self.crawford_triggered and not self.crawford_game:
            self.post_crawford = True

    @property
    def is_over(self) -> bool:
        if self.match_length <= 0:
            return False
        return any(s >= self.match_length for s in self.score)

    def match_winner(self) -> Optional[int]:
        if not self.is_over:
            return None
        return 0 if self.score[0] >= self.match_length else 1
