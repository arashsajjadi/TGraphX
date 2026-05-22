"""Tests for match state and doubling cube."""
import pytest
from backgammon_rlx.match.cube import CubeState
from backgammon_rlx.match.match_state import MatchState


class TestCubeState:

    def test_initial_state(self):
        c = CubeState()
        assert c.value == 1
        assert c.is_centred
        assert c.available

    def test_can_double_centred(self):
        c = CubeState()
        assert c.can_double(0)
        assert c.can_double(1)

    def test_double_by_player0(self):
        c = CubeState()
        c2 = c.doubled_by(0)
        assert c2.value == 2
        assert c2.owner == 1   # opponent now owns cube

    def test_owner_can_double(self):
        c = CubeState(value=2, owner=0)
        assert c.can_double(0)
        assert not c.can_double(1)

    def test_unavailable_cube(self):
        c = CubeState(available=False)
        assert not c.can_double(0)
        assert not c.can_double(1)


class TestMatchState:

    def test_initial(self):
        m = MatchState(match_length=5)
        assert m.score == [0, 0]
        assert not m.is_over

    def test_record_normal_win(self):
        m = MatchState(match_length=5)
        m.record_game(winner=0, game_score=1)
        assert m.score[0] == 1
        assert not m.is_over

    def test_game_over(self):
        m = MatchState(match_length=3)
        m.record_game(0, 1)
        m.record_game(0, 1)
        m.record_game(0, 1)
        assert m.is_over
        assert m.match_winner() == 0

    def test_crawford_triggers(self):
        m = MatchState(match_length=5)
        m.record_game(0, 4)   # player 0 needs 1 more
        assert m.crawford_triggered
        assert m.crawford_game

    def test_money_game_never_over(self):
        m = MatchState(match_length=0)
        for _ in range(100):
            m.record_game(0, 1)
        assert not m.is_over
