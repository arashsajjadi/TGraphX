"""Heuristic baseline agents.

GreedyPipAgent   — minimises pip count after the move.
HeuristicAgent   — weighted combination of tactical features.
"""
from __future__ import annotations

from typing import List, Optional

from ..env.state import GameState, Turn, BAR, OFF
from ..env.rules import pip_count, home_board_range, player_sign, _sign
from ..env.movegen import (
    get_legal_turns, apply_full_turn, canonicalize_state_for_player
)


class GreedyPipAgent:
    """Chooses the legal action that minimises own pip count after the move."""

    def select_action(self, state: GameState,
                      legal_turns: Optional[List[Turn]] = None) -> Turn:
        turns  = legal_turns if legal_turns is not None else get_legal_turns(state)
        player = state.current_player
        best   = turns[0]
        best_v = float("inf")
        for t in turns:
            s = apply_full_turn(state, t)
            # evaluate from original player's perspective before turn switch
            v = pip_count(s, player)
            if v < best_v:
                best_v = v
                best   = t
        return best

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------

_WEIGHTS = {
    "pip_gain":    1.0,
    "hits":        2.0,
    "points_made": 3.0,
    "avoid_blots":-1.5,
    "bear_offs":   4.0,
    "bar_escape":  2.5,
}


class HeuristicAgent:
    """Simple but stronger heuristic agent.

    Score = Σ weight_i * feature_i  (higher = better for current player).
    """

    def __init__(self, weights: Optional[dict] = None) -> None:
        self._w = weights or _WEIGHTS

    def _score(self, state: GameState, turn: Turn, player: int) -> float:
        after  = apply_full_turn(state, turn)
        ps     = player_sign(player)
        opp    = 1 - player
        board  = after.board

        # pip gain (positive = improvement)
        pip_before = pip_count(state, player)
        pip_after  = pip_count(after, player)
        pip_gain   = pip_before - pip_after

        # hits
        hits = sum(1 for m in turn.moves if m.hit)

        # points made (2+ own checkers after move)
        points_made = sum(1 for v in board if _sign(v) == ps and abs(v) >= 2)

        # blots created (exactly 1 own checker on an exposed point)
        blots_created = sum(1 for v in board if _sign(v) == ps and abs(v) == 1)

        # bear-offs
        bear_offs = sum(1 for m in turn.moves if m.dst == OFF)

        # bar escape (checkers entering from bar)
        bar_escapes = sum(1 for m in turn.moves if m.src == BAR)

        score = (
            self._w["pip_gain"]    * pip_gain
            + self._w["hits"]      * hits
            + self._w["points_made"] * points_made
            + self._w["avoid_blots"] * blots_created
            + self._w["bear_offs"]   * bear_offs
            + self._w["bar_escape"]  * bar_escapes
        )
        return score

    def select_action(self, state: GameState,
                      legal_turns: Optional[List[Turn]] = None) -> Turn:
        turns  = legal_turns if legal_turns is not None else get_legal_turns(state)
        player = state.current_player
        best   = max(turns, key=lambda t: self._score(state, t, player))
        return best

    def reset(self) -> None:
        pass
