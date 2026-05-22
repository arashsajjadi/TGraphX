"""Strict state-invariant checker.

Enable with ``strict_invariants=True`` on BackgammonEnv.
Each violation raises InvariantError with a full state dump.
"""
from __future__ import annotations

from typing import Optional

from ..env.state import GameState


class InvariantError(RuntimeError):
    """Raised when a board invariant is violated."""


def check_state_invariants(state: GameState,
                            context: str = "",
                            action: Optional[object] = None) -> None:
    """Raise InvariantError if any board invariant is violated."""
    errors = []

    # 1. Total checker counts
    from ..env.rules import total_checkers, _sign, player_sign
    for p in range(2):
        cnt = total_checkers(state, p)
        if cnt != 15:
            errors.append(f"Player {p} has {cnt} checkers (expected 15)")

    # 2. No point has both players' checkers
    for i, val in enumerate(state.board):
        if abs(val) > 0:
            # Only one sign may be present
            pass  # handled by rule: val is a single sign * count
    for i, val in enumerate(state.board):
        # val encodes a single player's checkers: sign determines player
        if abs(val) > 15:
            errors.append(f"Point {i+1} has {val} checkers — impossible")

    # 3. Bar counts non-negative
    for p in range(2):
        if state.bar[p] < 0:
            errors.append(f"Player {p} bar count is negative: {state.bar[p]}")

    # 4. Borne-off counts valid
    for p in range(2):
        if not (0 <= state.borne_off[p] <= 15):
            errors.append(
                f"Player {p} borne_off={state.borne_off[p]} out of range")

    # 5. Current player valid
    if state.current_player not in (0, 1):
        errors.append(f"Invalid current_player: {state.current_player}")

    # 6. Dice values valid
    for d in state.dice:
        if d not in range(1, 7):
            errors.append(f"Invalid die value: {d}")

    # 7. Board array length
    if len(state.board) != 24:
        errors.append(f"Board length is {len(state.board)}, expected 24")

    if errors:
        msg = "\n".join(errors)
        dump = (
            f"\nContext: {context}"
            f"\nAction:  {action}"
            f"\nState:   {state}"
            f"\nBoard:   {state.board}"
        )
        raise InvariantError(f"Invariant violation:\n{msg}{dump}")
