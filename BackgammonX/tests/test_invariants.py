"""Tests for the strict invariant checker."""
import pytest
from backgammon_rlx.env.state import GameState
from backgammon_rlx.validation.invariants import check_state_invariants, InvariantError


def test_valid_initial():
    s = GameState.initial()
    s.dice = [3, 5]
    check_state_invariants(s)   # should not raise


def test_wrong_checker_count():
    s = GameState(board=[0]*24)
    s.board[0] = 14   # only 14 on board, bar=0, borne_off=0 → total=14
    with pytest.raises(InvariantError):
        check_state_invariants(s)


def test_negative_bar():
    s = GameState.initial()
    s.bar = [-1, 0]
    with pytest.raises(InvariantError):
        check_state_invariants(s)


def test_invalid_die():
    s = GameState.initial()
    s.dice = [7]
    with pytest.raises(InvariantError):
        check_state_invariants(s)


def test_borne_off_overflow():
    s = GameState(board=[0]*24)
    s.borne_off = [16, 0]
    with pytest.raises(InvariantError):
        check_state_invariants(s)
