"""Unit tests for backgammon rules and initial setup."""
import pytest
from backgammon_rlx.env.state import GameState, BAR, OFF
from backgammon_rlx.env.rules import (
    is_point_open, has_checker_on_bar, can_bear_off,
    all_checkers_in_home, total_checkers, home_board_range,
    pip_count, player_sign, _sign, score_value, winner, is_terminal,
)


def initial() -> GameState:
    return GameState.initial()


# ---------------------------------------------------------------------------
# Initial position
# ---------------------------------------------------------------------------

class TestInitialPosition:

    def test_total_checkers_player0(self):
        s = initial()
        assert total_checkers(s, 0) == 15

    def test_total_checkers_player1(self):
        s = initial()
        assert total_checkers(s, 1) == 15

    def test_player0_on_24(self):
        s = initial()
        assert s.board[23] == 2   # point 24

    def test_player0_on_13(self):
        s = initial()
        assert s.board[12] == 5   # point 13

    def test_player0_on_8(self):
        s = initial()
        assert s.board[7] == 3    # point 8

    def test_player0_on_6(self):
        s = initial()
        assert s.board[5] == 5    # point 6

    def test_player1_on_1(self):
        s = initial()
        assert s.board[0] == -2   # point 1

    def test_player1_on_12(self):
        s = initial()
        assert s.board[11] == -5  # point 12

    def test_player1_on_17(self):
        s = initial()
        assert s.board[16] == -3  # point 17

    def test_player1_on_19(self):
        s = initial()
        assert s.board[18] == -5  # point 19

    def test_no_bar_initially(self):
        s = initial()
        assert s.bar == [0, 0]

    def test_no_borne_off_initially(self):
        s = initial()
        assert s.borne_off == [0, 0]

    def test_not_terminal_initially(self):
        s = initial()
        assert not is_terminal(s)


# ---------------------------------------------------------------------------
# is_point_open
# ---------------------------------------------------------------------------

class TestIsPointOpen:

    def test_empty_point_open(self):
        s = GameState(board=[0]*24)
        assert is_point_open(s, 5, 0)
        assert is_point_open(s, 5, 1)

    def test_own_checker_open(self):
        s = GameState(board=[0]*24)
        s.board[4] = 3   # player 0 has 3 checkers on point 5
        assert is_point_open(s, 5, 0)   # own → open
        assert not is_point_open(s, 5, 1)  # 3 opp → blocked

    def test_opponent_blot_open(self):
        s = GameState(board=[0]*24)
        s.board[4] = -1   # player 1 blot on point 5
        assert is_point_open(s, 5, 0)   # one opp → can hit
        assert is_point_open(s, 5, 1)   # own blot → open for self

    def test_opponent_prime_blocked(self):
        s = GameState(board=[0]*24)
        s.board[4] = -2   # player 1 has 2 checkers on point 5
        assert not is_point_open(s, 5, 0)   # blocked
        assert is_point_open(s, 5, 1)       # own → open


# ---------------------------------------------------------------------------
# Home board detection
# ---------------------------------------------------------------------------

class TestHomeBoard:

    def test_player0_home_range(self):
        lo, hi = home_board_range(0)
        assert lo == 1 and hi == 6

    def test_player1_home_range(self):
        lo, hi = home_board_range(1)
        assert lo == 19 and hi == 24

    def test_not_all_in_home_initially(self):
        s = initial()
        assert not all_checkers_in_home(s, 0)
        assert not all_checkers_in_home(s, 1)

    def test_all_in_home_when_ready(self):
        s = GameState(board=[0]*24)
        s.board[0] = 5    # point 1, player 0's home
        s.board[1] = 5    # point 2
        s.board[2] = 5    # point 3
        assert all_checkers_in_home(s, 0)

    def test_bar_prevents_home(self):
        s = GameState(board=[0]*24)
        s.board[0] = 14
        s.bar[0]   = 1
        assert not all_checkers_in_home(s, 0)


# ---------------------------------------------------------------------------
# Pip count
# ---------------------------------------------------------------------------

class TestPipCount:

    def test_pip_count_initial_p0(self):
        s = initial()
        # 2*24 + 5*13 + 3*8 + 5*6 = 48 + 65 + 24 + 30 = 167
        assert pip_count(s, 0) == 167

    def test_pip_count_initial_p1(self):
        s = initial()
        # 2*(25-1) + 5*(25-12) + 3*(25-17) + 5*(25-19) = 48+65+24+30 = 167
        assert pip_count(s, 1) == 167

    def test_pip_count_borne_off(self):
        s = GameState(board=[0]*24)
        s.board[0] = 15   # all on point 1
        assert pip_count(s, 0) == 15

    def test_pip_count_bar(self):
        s = GameState(board=[0]*24)
        s.bar[0] = 2
        assert pip_count(s, 0) == 50   # 2 * 25


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestScoring:

    def test_normal_win(self):
        s = GameState(board=[0]*24)
        s.borne_off = [15, 3]   # player 0 wins, player 1 has borne off 3
        assert is_terminal(s)
        assert winner(s) == 0
        assert score_value(s) == 1

    def test_gammon(self):
        s = GameState(board=[0]*24)
        s.borne_off = [15, 0]
        s.board[11] = -15   # player 1 not in home of player 0
        assert score_value(s) == 2

    def test_backgammon_bar(self):
        s = GameState(board=[0]*24)
        s.borne_off = [15, 0]
        s.bar[1] = 1           # loser has checker on bar
        assert score_value(s) == 3

    def test_backgammon_in_winner_home(self):
        s = GameState(board=[0]*24)
        s.borne_off = [15, 0]
        s.board[0] = -1        # loser still on point 1 (player 0's home)
        assert score_value(s) == 3

    def test_not_terminal(self):
        s = initial()
        assert score_value(s) == 0
