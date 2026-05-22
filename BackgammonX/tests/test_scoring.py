"""Tests for game scoring (normal, gammon, backgammon)."""
import pytest
from backgammon_rlx.env.state import GameState
from backgammon_rlx.env.rules import score_value, winner, is_terminal


def w0_board(loser_board_val: int = 0, loser_bar: int = 0,
              loser_borne_off: int = 0) -> GameState:
    """Player 0 wins; set up loser (player 1) state."""
    s = GameState(board=[0]*24)
    s.borne_off = [15, loser_borne_off]
    s.bar = [0, loser_bar]
    if loser_board_val:
        s.board[loser_board_val - 1] = -(15 - loser_borne_off - loser_bar)
    return s


class TestScoring:

    def test_normal_win(self):
        s = w0_board(loser_borne_off=3, loser_board_val=12)
        # loser has borne off > 0 → normal win
        assert is_terminal(s)
        assert winner(s) == 0
        assert score_value(s) == 1

    def test_gammon_all_in_outer(self):
        # Loser has 0 borne off, checkers NOT in winner's home and NOT on bar
        s = GameState(board=[0]*24)
        s.borne_off = [15, 0]
        s.board[11] = -15   # loser's checkers on point 12 (player 1's outer board)
        assert score_value(s) == 2

    def test_backgammon_loser_on_bar(self):
        s = GameState(board=[0]*24)
        s.borne_off = [15, 0]
        s.bar = [0, 1]       # loser has checker on bar
        s.board[11] = -14   # remaining on outer
        assert score_value(s) == 3

    def test_backgammon_loser_in_winner_home(self):
        # Winner is player 0 (home = 1-6); loser has checker there
        s = GameState(board=[0]*24)
        s.borne_off = [15, 0]
        s.board[2] = -1      # loser on point 3 = player 0's home
        s.board[11] = -14
        assert score_value(s) == 3

    def test_gammon_not_backgammon(self):
        # Loser has 0 borne off but not in winner's home and not on bar
        s = GameState(board=[0]*24)
        s.borne_off = [15, 0]
        s.board[13] = -15   # loser at point 14 (player 0's outer board)
        assert score_value(s) == 2

    def test_player1_wins_normal(self):
        s = GameState(board=[0]*24)
        s.borne_off = [3, 15]
        assert winner(s) == 1
        assert score_value(s) == 1

    def test_player1_wins_backgammon(self):
        # Player 1 wins; player 0 has checker in player 1's home (19-24) with no borne-off
        s = GameState(board=[0]*24)
        s.borne_off = [0, 15]
        s.board[21] = 1      # player 0 still in player 1's home (point 22)
        assert winner(s) == 1
        assert score_value(s) == 3

    def test_zero_if_not_terminal(self):
        s = GameState.initial()
        assert score_value(s) == 0
