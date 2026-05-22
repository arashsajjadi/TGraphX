"""Golden-test harness for legal move generation.

Fixtures live in tests/fixtures/legal_moves/*.json.
Schema documented in tests/fixtures/legal_moves/README.md.

Usage (in pytest):
    from backgammon_rlx.validation.golden_tests import load_fixtures, run_fixture
    for fix in load_fixtures():
        run_fixture(fix)
"""
from __future__ import annotations

import json
import pathlib
import textwrap
from typing import Any, Dict, List, Optional

from ..env.state import GameState
from ..env.movegen import get_legal_turns
from ..env.rules import is_terminal, winner, score_value
from ..notation.move_notation import format_full_turn


FIXTURE_DIR = pathlib.Path(__file__).parent.parent.parent.parent.parent / \
              "tests" / "fixtures" / "legal_moves"


def load_fixtures(directory: Optional[pathlib.Path] = None) -> List[Dict]:
    d = directory or FIXTURE_DIR
    if not d.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]


def _build_state(fix: Dict) -> GameState:
    return GameState(
        board=fix["board"],
        bar=fix.get("bar", [0, 0]),
        borne_off=fix.get("borne_off", [0, 0]),
        current_player=fix["current_player"],
        dice=fix.get("dice") or [],
    )


def _failure_msg(fix: Dict, heading: str, actual_strs: set) -> str:
    board_repr = fix.get("board", [])
    nonzero = [(i+1, v) for i, v in enumerate(board_repr) if v != 0]
    return textwrap.dedent(f"""
    Fixture: {fix.get('id', '?')}
    Heading: {heading}
    Board (non-zero): {nonzero}
    Bar: {fix.get('bar', [0,0])}  Borne-off: {fix.get('borne_off', [0,0])}
    Player: {fix.get('current_player', '?')}  Dice: {fix.get('dice', [])}
    Expected turns: {fix.get('expected_turns', [])}
    Forbidden turns: {fix.get('forbidden_turns', [])}
    Actual turns ({len(actual_strs)}): {sorted(actual_strs)}
    Missing: {[e for e in fix.get('expected_turns', []) if e not in actual_strs]}
    Extra/forbidden: {[f for f in fix.get('forbidden_turns', []) if f in actual_strs]}
    Notes: {fix.get('notes', '')}
    """).strip()


def run_fixture(fix: Dict[str, Any], verbose: bool = False) -> None:
    """Assert that legal turns match the fixture expectations.

    Skips move-generation checks for terminal-state-only fixtures
    (those that only verify scoring, marked with is_terminal=true and
    no expected_num_turns or expected_turns).
    """
    fix_id = fix.get("id", "?")
    state  = _build_state(fix)

    # --- Terminal state checks ---
    if fix.get("is_terminal"):
        assert is_terminal(state), (
            f"Fixture {fix_id}: expected terminal state but is_terminal=False")
        if fix.get("expected_winner") is not None:
            assert winner(state) == fix["expected_winner"], (
                f"Fixture {fix_id}: expected winner {fix['expected_winner']}, "
                f"got {winner(state)}")
        if fix.get("expected_score") is not None:
            assert score_value(state) == fix["expected_score"], (
                f"Fixture {fix_id}: expected score {fix['expected_score']}, "
                f"got {score_value(state)}")
        # Terminal fixtures may also include expected_turns for completeness
        # but only test them if dice are provided
        if not state.dice:
            return

    # --- Legal-move generation checks ---
    turns   = get_legal_turns(state)
    n_turns = len(turns)

    if verbose:
        print(f"\n=== {fix_id} ===  ({n_turns} turns)")
        for t in turns:
            print(" ", format_full_turn(t))

    expected_count = fix.get("expected_num_turns")
    if expected_count is not None:
        actual_strs = {format_full_turn(t) for t in turns}
        assert n_turns == expected_count, (
            _failure_msg(fix, f"Count mismatch: expected {expected_count} got {n_turns}",
                         actual_strs))

    actual_strs = {format_full_turn(t) for t in turns}

    for et in fix.get("expected_turns", []):
        assert et in actual_strs, _failure_msg(fix, f"Missing: '{et}'", actual_strs)

    for ft in fix.get("forbidden_turns", []):
        assert ft not in actual_strs, _failure_msg(fix, f"Forbidden present: '{ft}'", actual_strs)
