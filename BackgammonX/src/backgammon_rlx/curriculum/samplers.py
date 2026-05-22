"""Curriculum learning position samplers.

Each sampler generates initial GameState configurations for a training stage.
Use in place of GameState.initial() when curriculum is enabled.
"""
from __future__ import annotations

import random
from typing import Optional

from ..env.state import GameState
from ..env.rules import home_board_range, player_sign, _sign


def _empty_board() -> GameState:
    return GameState(board=[0]*24, bar=[0,0], borne_off=[0,0],
                     current_player=0, dice=[])


class BearOffSampler:
    """Both players have 1-6 checkers in their home boards only (race to finish)."""

    def __init__(self, max_per_point: int = 5, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._max = max_per_point

    def sample(self) -> GameState:
        state = _empty_board()
        # Place checkers for both players in home boards
        for player in range(2):
            lo, hi = home_board_range(player)
            n_checkers = self._rng.randint(1, 6)
            placed = 0
            for _ in range(200):
                if placed >= n_checkers:
                    break
                pt = self._rng.randint(lo, hi)
                val = state.board[pt - 1]
                ps  = player_sign(player)
                if val == 0 or _sign(val) == ps:
                    if abs(val) < self._max:
                        state.board[pt - 1] += ps
                        placed += 1
            # Remaining checkers go to borne_off
            state.borne_off[player] = 15 - placed
        return state


class RacingSampler:
    """Pure racing: all checkers past the bar, no contact."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def sample(self) -> GameState:
        state = _empty_board()
        # Player 0 checkers in points 1-12, player 1 in 13-24
        for _ in range(15):
            while True:
                pt = self._rng.randint(1, 12)
                if abs(state.board[pt-1]) < 5:
                    state.board[pt-1] += 1
                    break
        for _ in range(15):
            while True:
                pt = self._rng.randint(13, 24)
                if abs(state.board[pt-1]) < 5:
                    state.board[pt-1] -= 1
                    break
        return state


class FullGameSampler:
    """Returns the standard initial position (no curriculum offset)."""

    def sample(self) -> GameState:
        return GameState.initial()
