"""Unit tests for legal move generation — the most critical module."""
import pytest
from backgammon_rlx.env.state import GameState, AtomicMove, Turn, BAR, OFF
from backgammon_rlx.env.movegen import (
    get_legal_atomic_moves, get_legal_turns,
    apply_atomic_move_inplace, apply_full_turn,
)
from backgammon_rlx.env.rules import total_checkers


def _state(**kwargs) -> GameState:
    s = GameState(board=[0]*24, bar=[0,0], borne_off=[0,0],
                  current_player=0, dice=[])
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


# ---------------------------------------------------------------------------
# Atomic move helpers
# ---------------------------------------------------------------------------

class TestAtomicMoves:

    def test_normal_move_player0(self):
        s = _state(bar=[0,0], dice=[3])
        s.board[7] = 1    # checker on point 8
        moves = get_legal_atomic_moves(s, 0, 3)
        dsts  = [m.dst for m in moves]
        assert 5 in dsts  # 8-3=5

    def test_normal_move_player1(self):
        s = _state(bar=[0,0], dice=[4], current_player=1)
        s.board[11] = -1  # player 1 checker on point 12
        moves = get_legal_atomic_moves(s, 1, 4)
        dsts  = [m.dst for m in moves]
        assert 16 in dsts  # 12+4=16

    def test_blocked_by_opponent_prime(self):
        s = _state(dice=[3])
        s.board[7] = 1    # player 0 on point 8
        s.board[4] = -2   # player 1 prime on point 5 (8-3=5)
        moves = get_legal_atomic_moves(s, 0, 3)
        assert all(m.dst != 5 for m in moves)

    def test_hit_detected(self):
        s = _state(dice=[3])
        s.board[7] = 1    # player 0 on 8
        s.board[4] = -1   # player 1 blot on 5
        moves = get_legal_atomic_moves(s, 0, 3)
        hits = [m for m in moves if m.dst == 5]
        assert hits and hits[0].hit

    def test_bar_priority(self):
        s = _state(bar=[1, 0], dice=[3])
        s.board[7] = 5    # player 0 has many checkers on board
        moves = get_legal_atomic_moves(s, 0, 3)
        # Only bar-entry moves should be returned
        assert all(m.src == BAR for m in moves)

    def test_bar_entry_player0_die3(self):
        s = _state(bar=[1, 0], dice=[3])
        # die 3 for player 0: enter on point 25-3=22
        moves = get_legal_atomic_moves(s, 0, 3)
        assert any(m.dst == 22 for m in moves)

    def test_bar_entry_player1_die4(self):
        s = _state(bar=[0, 1], dice=[4], current_player=1)
        # die 4 for player 1: enter on point 4
        moves = get_legal_atomic_moves(s, 1, 4)
        assert any(m.dst == 4 for m in moves)

    def test_blocked_bar_entry(self):
        s = _state(bar=[1, 0], dice=[3])
        s.board[21] = -2  # player 1 prime on point 22 (player 0's entry with die 3)
        moves = get_legal_atomic_moves(s, 0, 3)
        assert not moves   # cannot enter


# ---------------------------------------------------------------------------
# Full turn generation
# ---------------------------------------------------------------------------

class TestFullTurns:

    def test_two_dice_both_used(self):
        s = _state(dice=[3, 2])
        s.board[7] = 2    # checkers on point 8
        s.current_player = 0
        turns = get_legal_turns(s)
        assert all(len(t) == 2 for t in turns)

    def test_doubles_produce_four_moves(self):
        s = _state(dice=[3, 3, 3, 3])
        s.board[12] = 15  # many checkers on point 13
        turns = get_legal_turns(s)
        max_moves = max(len(t) for t in turns)
        assert max_moves == 4

    def test_forced_pass(self):
        # All entry points blocked; player on bar
        s = _state(bar=[1, 0], dice=[1, 2])
        # Block points 24 and 23 (entry for player 0 with die 1 and 2)
        s.board[23] = -2
        s.board[22] = -2
        turns = get_legal_turns(s)
        assert turns == [Turn()]   # forced pass

    def test_larger_die_rule(self):
        # Checker at point 20.  Point 12 is blocked by opponent.
        # die 5: 20→15 (open); then die 3: 15→12 blocked → only 1 move
        # die 3: 20→17 (open); then die 5: 17→12 blocked → only 1 move
        # Both dice individually give 1 move but NOT together → use larger (5).
        s = _state(dice=[5, 3])
        s.board[19] = 1   # player 0 checker on point 20
        s.board[11] = -2  # point 12 blocked by 2 player-1 checkers
        turns = get_legal_turns(s)
        assert turns
        assert all(len(t) == 1 for t in turns)
        assert all(t.moves[0].die == 5 for t in turns)

    def test_mandatory_dice_usage(self):
        # If both dice can be played, both must be
        s = _state(dice=[1, 2])
        s.board[5]  = 1  # player 0 checker on point 6
        s.board[11] = 1  # player 0 checker on point 12
        # 6→5 (die 1), 12→10 (die 2) — both playable
        turns = get_legal_turns(s)
        assert all(len(t) == 2 for t in turns)

    def test_deduplication(self):
        # Two identical checkers on the same point; same final state
        s = _state(dice=[3, 2])
        s.board[7] = 2   # two identical checkers on point 8
        turns = get_legal_turns(s)
        # Both 8→5→3 orderings reach same state; should deduplicate
        keys = set()
        for t in turns:
            tmp = s.clone()
            for m in t.moves:
                apply_atomic_move_inplace(tmp, m, 0)
            keys.add(tmp.board_key())
        assert len(keys) == len(turns)  # each turn maps to unique final state

    def test_checker_count_invariant_after_turn(self):
        s = GameState.initial()
        s.dice = [3, 5]
        turns = get_legal_turns(s)
        for t in turns:
            after = apply_full_turn(s, t)
            assert total_checkers(after, 0) == 15
            assert total_checkers(after, 1) == 15


# ---------------------------------------------------------------------------
# Bar re-entry
# ---------------------------------------------------------------------------

class TestBarReentry:

    def test_must_enter_from_bar_first(self):
        s = _state(bar=[1, 0], dice=[3, 5])
        s.board[7]  = 14  # other player 0 checkers
        turns = get_legal_turns(s)
        # All turns must have first move from BAR
        assert all(t.moves[0].src == BAR for t in turns if t.moves)

    def test_enter_then_move(self):
        s = _state(bar=[1, 0], dice=[3, 5])
        s.board[7] = 1   # another checker on board
        turns = get_legal_turns(s)
        # Some turns should use both dice (enter + regular move)
        has_two = any(len(t) == 2 for t in turns)
        assert has_two

    def test_all_entry_blocked_pass(self):
        s = _state(bar=[2, 0], dice=[1, 2])
        s.board[23] = -2   # point 24 blocked (entry die 1)
        s.board[22] = -2   # point 23 blocked (entry die 2)
        turns = get_legal_turns(s)
        assert turns == [Turn()]
