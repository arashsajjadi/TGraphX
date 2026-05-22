"""Tests for action canonicalization and deduplication."""
import pytest
from backgammon_rlx.env.state import GameState
from backgammon_rlx.env.movegen import (
    get_legal_turns, apply_atomic_move_inplace,
)


def _unique_final_states(s: GameState) -> int:
    """Return the number of distinct final board states across all legal turns."""
    turns = get_legal_turns(s)
    keys = set()
    for t in turns:
        tmp = s.clone()
        for m in t.moves:
            apply_atomic_move_inplace(tmp, m, s.current_player)
        keys.add(tmp.board_key())
    return len(keys)


class TestDeduplication:

    def test_identical_checkers_deduplicated(self):
        """Three identical checkers on one point; moves reaching same final state merged."""
        s = GameState(board=[0]*24)
        s.board[11] = 3   # 3 checkers on point 12
        s.dice = [2, 1]
        s.current_player = 0
        turns = get_legal_turns(s)
        n_unique = _unique_final_states(s)
        assert n_unique == len(turns)   # all turns are unique by construction

    def test_two_point_choices_not_merged(self):
        """Different source points → different final states → preserved."""
        s = GameState(board=[0]*24)
        s.board[11] = 1   # point 12
        s.board[7]  = 1   # point 8
        s.dice = [2, 1]
        s.current_player = 0
        turns = get_legal_turns(s)
        n_unique = _unique_final_states(s)
        assert n_unique == len(turns)

    def test_all_legal_turns_unique_final_state(self):
        """Invariant: every generated turn maps to a distinct final state."""
        for seed in range(10):
            import random
            rng = random.Random(seed)
            s = GameState.initial()
            s.dice = [rng.randint(1,6), rng.randint(1,6)]
            s.dice = [s.dice[0]] * 4 if s.dice[0] == s.dice[1] else s.dice
            turns = get_legal_turns(s)
            n_unique = _unique_final_states(s)
            assert n_unique == len(turns), (
                f"seed={seed}: {len(turns)} turns but only {n_unique} unique states"
            )
