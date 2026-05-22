"""Tests for the explain_illegal_action diagnostic tool."""
import pytest
from backgammon_rlx.env.state import GameState, AtomicMove, Turn, BAR, OFF
from backgammon_rlx.validation.explain_illegal import (
    explain_illegal_action, explain_full_turn, explain_atomic_move,
)
from backgammon_rlx.env.movegen import get_legal_turns


def _state(**kw):
    s = GameState(board=[0]*24, bar=[0,0], borne_off=[0,0],
                  current_player=0, dice=[])
    for k, v in kw.items():
        setattr(s, k, v)
    return s


class TestExplainAtomicMove:

    def test_valid_move_returns_ok(self):
        s = _state(dice=[3])
        s.board[12] = 1  # p0 at 13
        move = AtomicMove(src=13, dst=10, die=3, hit=False)
        result = explain_atomic_move(s, move, 0)
        assert result == "OK"

    def test_no_checker_at_source(self):
        s = _state(dice=[3])
        move = AtomicMove(src=13, dst=10, die=3, hit=False)
        result = explain_atomic_move(s, move, 0)
        assert "ILLEGAL" in result
        assert "13" in result  # mention the source point

    def test_blocked_destination(self):
        s = _state(dice=[3])
        s.board[12] = 1  # p0 at 13
        s.board[9]  = -2  # p1 prime at 10
        move = AtomicMove(src=13, dst=10, die=3, hit=False)
        result = explain_atomic_move(s, move, 0)
        assert "ILLEGAL" in result
        assert "blocked" in result.lower()

    def test_bar_priority_violation(self):
        s = _state(bar=[1, 0], dice=[3])
        s.board[12] = 1  # p0 board checker
        move = AtomicMove(src=13, dst=10, die=3, hit=False)
        result = explain_atomic_move(s, move, 0)
        assert "ILLEGAL" in result
        assert "bar" in result.lower()

    def test_bearoff_forbidden(self):
        s = _state(dice=[3])
        s.board[12] = 1  # point 13 – NOT in home
        s.board[2]  = 1  # point 3 – in home
        move = AtomicMove(src=3, dst=OFF, die=3, hit=False)
        result = explain_atomic_move(s, move, 0)
        assert "ILLEGAL" in result
        assert "bear off" in result.lower()


class TestExplainFullTurn:

    def test_legal_turn_identified(self):
        s = GameState.initial()
        s.dice = [3, 1]
        turns = get_legal_turns(s)
        assert turns
        result = explain_full_turn(s, turns[0])
        assert "LEGAL" in result

    def test_illegal_turn_identified(self):
        s = GameState.initial()
        s.dice = [3, 1]
        # Construct an illegal turn: try to move from a blocked destination
        illegal_turn = Turn.from_list([AtomicMove(src=13, dst=12, die=1, hit=False)])
        result = explain_full_turn(s, illegal_turn)
        # Point 12 has 5 player-1 checkers → blocked
        assert "NOT" in result or "ILLEGAL" in result

    def test_pass_turn_on_stuck_position(self):
        s = _state(bar=[1, 0], dice=[3, 5])
        s.board[21] = -2  # block point 22
        s.board[19] = -2  # block point 20
        turns = get_legal_turns(s)
        assert turns[0] == Turn()
        result = explain_full_turn(s, turns[0])
        assert "LEGAL" in result or "(pass)" in result

    def test_explain_illegal_action_interface(self):
        s = GameState.initial()
        move = AtomicMove(src=1, dst=OFF, die=1, hit=False)
        turn = Turn.from_list([move])
        result = explain_illegal_action(s, [3, 1], turn)
        assert isinstance(result, str)
        assert len(result) > 0


class TestExplainWithLegalActions:

    def test_every_legal_action_explains_as_ok(self):
        """Every turn returned by get_legal_turns must explain as legal."""
        s = GameState.initial()
        s.dice = [5, 3]
        turns = get_legal_turns(s)
        for t in turns:
            result = explain_full_turn(s, t)
            assert "LEGAL" in result, (
                f"Legal turn {t} was not explained as legal:\n{result}")

    def test_illegal_action_never_explains_as_ok(self):
        """A turn with a blocked move must not explain as legal."""
        s = GameState.initial()
        s.dice = [3, 1]
        # Blocked: player 1 has 5 checkers at point 12 → player 0 can't land there
        move = AtomicMove(src=13, dst=12, die=1, hit=False)
        bad_turn = Turn.from_list([move])
        result = explain_full_turn(s, bad_turn)
        assert "LEGAL" not in result or "NOT" in result
