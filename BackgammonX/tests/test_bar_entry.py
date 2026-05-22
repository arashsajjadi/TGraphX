"""Dedicated tests for bar entry rules."""
import pytest
from backgammon_rlx.env.state import GameState, Turn, BAR
from backgammon_rlx.env.movegen import (
    get_legal_turns, get_legal_atomic_moves, apply_full_turn,
)
from backgammon_rlx.env.rules import total_checkers, has_checker_on_bar


def _state(**kw) -> GameState:
    s = GameState(board=[0] * 24, bar=[0, 0], borne_off=[0, 0],
                  current_player=0, dice=[])
    for k, v in kw.items():
        setattr(s, k, v)
    return s


class TestBarEntryPoints:

    def test_player0_entry_die1(self):
        """Player 0, die 1 → enter on point 24."""
        s = _state(bar=[1, 0], dice=[1])
        moves = get_legal_atomic_moves(s, 0, 1)
        assert any(m.src == BAR and m.dst == 24 for m in moves)

    def test_player0_entry_die6(self):
        """Player 0, die 6 → enter on point 19."""
        s = _state(bar=[1, 0], dice=[6])
        moves = get_legal_atomic_moves(s, 0, 6)
        assert any(m.src == BAR and m.dst == 19 for m in moves)

    def test_player1_entry_die1(self):
        """Player 1, die 1 → enter on point 1."""
        s = _state(bar=[0, 1], dice=[1], current_player=1)
        moves = get_legal_atomic_moves(s, 1, 1)
        assert any(m.src == BAR and m.dst == 1 for m in moves)

    def test_player1_entry_die6(self):
        """Player 1, die 6 → enter on point 6."""
        s = _state(bar=[0, 1], dice=[6], current_player=1)
        moves = get_legal_atomic_moves(s, 1, 6)
        assert any(m.src == BAR and m.dst == 6 for m in moves)

    def test_bar_entry_blocked(self):
        """Entry point blocked by 2 opponent checkers → no move."""
        s = _state(bar=[1, 0], dice=[3])
        s.board[21] = -2  # point 22 blocked (25-3=22)
        moves = get_legal_atomic_moves(s, 0, 3)
        assert not moves

    def test_bar_entry_can_hit_blot(self):
        """Entry point has exactly 1 opponent checker → hit is allowed."""
        s = _state(bar=[1, 0], dice=[3])
        s.board[21] = -1  # blot on point 22
        moves = get_legal_atomic_moves(s, 0, 3)
        assert any(m.dst == 22 and m.hit for m in moves)

    def test_bar_entry_own_point(self):
        """Entry point occupied by own checkers → allowed."""
        s = _state(bar=[1, 0], dice=[3])
        s.board[21] = 2  # own checkers on 22
        moves = get_legal_atomic_moves(s, 0, 3)
        assert any(m.dst == 22 and not m.hit for m in moves)


class TestBarPriority:

    def test_bar_forces_entry_before_other_moves(self):
        """With a checker on bar, ALL turns must start with a bar entry."""
        s = _state(bar=[1, 0], dice=[3, 2])
        s.board[7] = 10  # many checkers on board
        turns = get_legal_turns(s)
        assert all(t.moves and t.moves[0].src == BAR for t in turns if t.moves)

    def test_bar_prevents_board_moves_entirely(self):
        """Atomic board moves should return nothing when player is on bar."""
        s = _state(bar=[1, 0], dice=[5])
        s.board[12] = 5  # checkers on board, but bar takes priority
        moves = get_legal_atomic_moves(s, 0, 5)
        # All returned moves must be bar-entry moves
        assert all(m.src == BAR for m in moves)

    def test_multiple_bar_checkers_all_must_enter(self):
        """With 2 checkers on bar and doubles, must enter both before moving."""
        s = _state(bar=[2, 0], dice=[3, 3, 3, 3])
        # Entry on 22 open
        turns = get_legal_turns(s)
        # First 2 moves must be bar entries
        for t in turns:
            assert t.moves[0].src == BAR
            assert t.moves[1].src == BAR

    def test_enter_bar_checker_sends_opponent_to_bar(self):
        """Entering from bar onto opponent blot sends the blot to the bar."""
        s = _state(bar=[1, 0], dice=[3], current_player=0)
        s.board[21] = -1  # player 1 blot on 22 (entry for die 3)
        turns = get_legal_turns(s)
        assert turns
        after = apply_full_turn(s, turns[0])
        # Player 1 should now be on bar
        assert after.bar[1] == 1
        assert after.bar[0] == 0

    def test_bar_blocked_all_dice_pass(self):
        """All entry points blocked → forced pass regardless of board position."""
        s = _state(bar=[1, 0], dice=[1, 2])
        s.board[23] = -2  # point 24 blocked (die 1 entry)
        s.board[22] = -2  # point 23 blocked (die 2 entry)
        s.board[10] = 5   # many own checkers on board
        turns = get_legal_turns(s)
        assert turns == [Turn()]

    def test_bar_checker_count_invariant(self):
        """Entering from bar preserves total checker count."""
        s = _state(bar=[1, 0], dice=[3, 2])
        turns = get_legal_turns(s)
        for t in turns:
            after = apply_full_turn(s, t)
            assert total_checkers(after, 0) == 1  # only 1 checker total in this state
            assert after.bar[0] == 0 or (len(t.moves) < 2)
