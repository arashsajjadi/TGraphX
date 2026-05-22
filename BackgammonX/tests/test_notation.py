"""Tests for move notation formatting and parsing."""
import pytest
from backgammon_rlx.env.state import AtomicMove, Turn, BAR, OFF
from backgammon_rlx.notation.move_notation import (
    format_atomic_move, format_full_turn, parse_move_notation,
)


class TestFormatAtomicMove:

    def test_normal(self):
        m = AtomicMove(src=13, dst=8, die=5, hit=False)
        assert format_atomic_move(m) == "13/8"

    def test_hit(self):
        m = AtomicMove(src=24, dst=18, die=6, hit=True)
        assert format_atomic_move(m) == "24/18*"

    def test_bar_entry(self):
        m = AtomicMove(src=BAR, dst=22, die=3, hit=False)
        assert format_atomic_move(m) == "bar/22"

    def test_bear_off(self):
        m = AtomicMove(src=6, dst=OFF, die=6, hit=False)
        assert format_atomic_move(m) == "6/off"


class TestFormatFullTurn:

    def test_two_moves(self):
        t = Turn.from_list([
            AtomicMove(13, 8, 5, False),
            AtomicMove(6, 4, 2, False),
        ])
        result = format_full_turn(t)
        assert "13/8" in result
        assert "6/4" in result

    def test_pass(self):
        assert format_full_turn(Turn()) == "(pass)"

    def test_compact_doubles(self):
        t = Turn.from_list([
            AtomicMove(6, 3, 3, False),
            AtomicMove(6, 3, 3, False),
        ])
        result = format_full_turn(t)
        assert "6/3(2)" in result


class TestParseMoveNotation:

    def test_normal(self):
        t = parse_move_notation("13/8", player=0)
        assert len(t.moves) == 1
        assert t.moves[0].src == 13
        assert t.moves[0].dst == 8

    def test_hit(self):
        t = parse_move_notation("24/18*", player=0)
        assert t.moves[0].hit

    def test_bar_entry(self):
        t = parse_move_notation("bar/22", player=0)
        assert t.moves[0].src == BAR
        assert t.moves[0].dst == 22

    def test_bear_off(self):
        t = parse_move_notation("6/off", player=0)
        assert t.moves[0].dst == OFF

    def test_multi_move(self):
        t = parse_move_notation("13/8 6/4", player=0)
        assert len(t.moves) == 2

    def test_compact(self):
        t = parse_move_notation("6/3(2)", player=0)
        assert len(t.moves) == 2
        assert all(m.src == 6 and m.dst == 3 for m in t.moves)
