"""Deterministic position import / export.

File format: JSON with a stable schema.  The format is self-contained and
can be shared with external tools (e.g. GNU Backgammon adapters).

Schema
------
{
  "board":        list[int],   # length-24, +player0 / -player1
  "bar":          [int, int],
  "borne_off":    [int, int],
  "current_player": int,
  "dice":         list[int],   # remaining dice, empty between turns
  "turn_number":  int,
  "metadata":     {str: any}   # arbitrary user metadata
}
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Optional

from ..env.state import GameState


def state_to_dict(state: GameState,
                  metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "board":          state.board[:],
        "bar":            state.bar[:],
        "borne_off":      state.borne_off[:],
        "current_player": state.current_player,
        "dice":           state.dice[:],
        "turn_number":    state.turn_number,
        "metadata":       metadata or {},
    }


def state_from_dict(d: Dict[str, Any]) -> GameState:
    return GameState(
        board=d["board"],
        bar=d["bar"],
        borne_off=d["borne_off"],
        current_player=d["current_player"],
        dice=d.get("dice", []),
        turn_number=d.get("turn_number", 0),
    )


def state_to_json(state: GameState,
                  metadata: Optional[Dict[str, Any]] = None,
                  indent: int = 2) -> str:
    return json.dumps(state_to_dict(state, metadata), indent=indent)


def state_from_json(text: str) -> GameState:
    return state_from_dict(json.loads(text))


def save_position(state: GameState, path: str | pathlib.Path,
                  metadata: Optional[Dict[str, Any]] = None) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state_to_json(state, metadata))


def load_position(path: str | pathlib.Path) -> GameState:
    return state_from_json(pathlib.Path(path).read_text())
