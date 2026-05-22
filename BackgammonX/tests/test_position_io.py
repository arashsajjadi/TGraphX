"""Tests for position import/export."""
import json
import pytest
import tempfile
from pathlib import Path
from backgammon_rlx.env.state import GameState
from backgammon_rlx.notation.position_io import (
    state_to_dict, state_from_dict, state_to_json, state_from_json,
    save_position, load_position,
)


def test_roundtrip_dict():
    s = GameState.initial()
    d = state_to_dict(s)
    s2 = state_from_dict(d)
    assert s.board == s2.board
    assert s.bar == s2.bar
    assert s.borne_off == s2.borne_off
    assert s.current_player == s2.current_player


def test_roundtrip_json():
    s = GameState.initial()
    s.dice = [3, 5]
    j = state_to_json(s)
    s2 = state_from_json(j)
    assert s.board == s2.board
    assert s.dice == s2.dice


def test_roundtrip_file():
    s = GameState.initial()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pos.json"
        save_position(s, p, metadata={"test": True})
        s2 = load_position(p)
    assert s.board == s2.board


def test_metadata_preserved():
    s = GameState.initial()
    d = state_to_dict(s, metadata={"note": "test position"})
    assert d["metadata"]["note"] == "test position"


def test_non_standard_position():
    s = GameState(board=[0]*24)
    s.board[5]  = 5
    s.board[18] = -5
    s.bar = [2, 1]
    s.borne_off = [3, 2]
    d   = state_to_dict(s)
    s2  = state_from_dict(d)
    assert s2.bar == [2, 1]
    assert s2.borne_off == [3, 2]
