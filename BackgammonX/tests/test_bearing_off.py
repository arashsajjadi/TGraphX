"""Tests for bearing-off rules — particularly tricky edge cases."""
import pytest
from backgammon_rlx.env.state import GameState, OFF
from backgammon_rlx.env.movegen import get_legal_turns, get_legal_atomic_moves
from backgammon_rlx.env.rules import can_bear_off, total_checkers


def _home_only(player: int, *point_counts) -> GameState:
    """Build a state with player's checkers only in their home board.

    point_counts: pairs (point, count) from the player's perspective.
    Player 0 home = points 1-6.
    Player 1 home = points 19-24.
    """
    s = GameState(board=[0]*24, bar=[0,0], borne_off=[0,0],
                  current_player=player, dice=[])
    total = 0
    for pt, cnt in point_counts:
        if player == 0:
            s.board[pt - 1] = cnt
        else:
            s.board[pt - 1] = -cnt
        total += cnt
    s.borne_off[player] = 15 - total
    return s


class TestBearingOff:

    def test_cannot_bear_off_outside_home(self):
        s = GameState(board=[0]*24)
        s.board[7]  = 5   # point 8 — NOT home (home=1-6)
        s.board[4]  = 10  # point 5 — home
        assert not can_bear_off(s, 0)

    def test_can_bear_off_all_in_home(self):
        s = _home_only(0, (1, 5), (2, 5), (3, 5))
        assert can_bear_off(s, 0)

    def test_bar_prevents_bearing_off(self):
        s = _home_only(0, (1, 14))
        s.bar[0] = 1
        s.borne_off[0] = 0
        assert not can_bear_off(s, 0)

    def test_exact_bear_off(self):
        s = _home_only(0, (3, 1))  # 1 checker on point 3
        s.dice = [3]
        moves = get_legal_atomic_moves(s, 0, 3)
        bear_offs = [m for m in moves if m.dst == OFF]
        assert len(bear_offs) == 1
        assert bear_offs[0].src == 3

    def test_larger_die_bear_off(self):
        # Checker on point 3 only; die 5 > 3 → can bear off
        s = _home_only(0, (3, 1))
        s.dice = [5]
        moves = get_legal_atomic_moves(s, 0, 5)
        bear_offs = [m for m in moves if m.dst == OFF]
        assert len(bear_offs) == 1   # can bear off highest checker with die 5

    def test_larger_die_blocked_by_higher_checker(self):
        # Checker on 2 and 5; die 4: can bear off from 4 exactly? No checker on 4.
        # Can use die 4 > 2 to bear off from 2 only if no checker on 3, 4, 5, 6?
        # But there IS a checker on 5 (higher). So die 4 cannot bear off from 2.
        s = _home_only(0, (2, 1), (5, 1))
        s.dice = [4]
        moves = get_legal_atomic_moves(s, 0, 4)
        # Die 4 from point 2: dist=2, die=4>2, but point 5 (dist=5>2) exists → blocked
        # Die 4 from point 5: dist=5, die=4 < 5 → cannot use die to move off board
        # So: die 4 from point 5 → destination 5-4=1, normal move to point 1
        bear_offs = [m for m in moves if m.dst == OFF]
        assert not bear_offs   # no bear-off possible with die 4

    def test_exact_bear_off_player1(self):
        # Player 1 home: 19-24; die 1 → bear off from point 24 (25-1=24)
        s = _home_only(1, (24, 1))
        s.dice = [1]
        moves = get_legal_atomic_moves(s, 1, 1)
        bear_offs = [m for m in moves if m.dst == OFF]
        assert len(bear_offs) == 1
        assert bear_offs[0].src == 24

    def test_hit_during_bearoff_resets(self):
        # Player 0 is bearing off; opponent hits one of their checkers
        s = _home_only(0, (1, 5), (2, 5), (3, 4))
        assert can_bear_off(s, 0)
        # Simulate hit: one of player 0's checkers goes to bar
        s.board[0] -= 1
        s.bar[0]   += 1
        assert not can_bear_off(s, 0)   # bar prevents bearing off

    def test_bear_off_reduces_borne_off(self):
        s = _home_only(0, (3, 1))
        s.dice = [3]
        turns = get_legal_turns(s)
        assert turns
        from backgammon_rlx.env.movegen import apply_full_turn
        after = apply_full_turn(s, turns[0])
        assert total_checkers(after, 0) == 15   # total never changes

    def test_bear_off_full_game_15(self):
        # All checkers on point 6; 15 consecutive turns should clear them
        s = GameState(board=[0]*24, bar=[0,0], borne_off=[0,0])
        s.board[5] = 15   # all on point 6
        s.current_player = 0
        from backgammon_rlx.env.movegen import apply_full_turn
        from backgammon_rlx.env.rules import is_terminal

        for die in [6]*15:
            s.dice = [die]
            turns = get_legal_turns(s)
            assert turns, f"No turns at step, borne_off={s.borne_off}"
            s = apply_full_turn(s, turns[0])
            s.current_player = 0   # keep same player for test simplicity

        assert s.borne_off[0] == 15
