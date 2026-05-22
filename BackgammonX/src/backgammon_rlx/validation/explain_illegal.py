"""Explain why an action is illegal or diagnose move-generation issues.

Used in tests, CLI debugging, and the strict-invariant mode.
"""
from __future__ import annotations

from typing import List, Optional

from ..env.state import GameState, AtomicMove, Turn, BAR, OFF
from ..env.rules import (
    player_sign, _sign, is_point_open, has_checker_on_bar,
    can_bear_off, can_bear_off_checker, bar_entry_point,
    home_board_range, total_checkers,
)
from ..env.movegen import get_legal_turns, apply_atomic_move_inplace
from ..notation.move_notation import format_atomic_move, format_full_turn


def explain_atomic_move(
    state: GameState,
    move: AtomicMove,
    player: int,
) -> str:
    """Return a human-readable explanation of why *move* is or is not legal."""
    ps = player_sign(player)

    # 1. Source validation
    if move.src == BAR:
        if state.bar[player] <= 0:
            return f"ILLEGAL: player {player} has no checkers on bar (bar={state.bar[player]})"
    else:
        val = state.board[move.src - 1]
        if _sign(val) != ps or abs(val) == 0:
            return (f"ILLEGAL: no player-{player} checker at point {move.src} "
                    f"(board value={val})")
        # Bar priority
        if state.bar[player] > 0:
            return (f"ILLEGAL: player {player} must enter bar checker first "
                    f"(bar[{player}]={state.bar[player]}), but move starts from point {move.src}")

    # 2. Destination validation
    if move.dst == OFF:
        if not can_bear_off(state, player):
            return (f"ILLEGAL: player {player} cannot bear off "
                    f"(bar={state.bar[player]}, not all in home or not in bearing-off phase)")
        if move.src != BAR:
            if not can_bear_off_checker(state, player, move.src, move.die):
                return (f"ILLEGAL: cannot bear off from point {move.src} with die {move.die} "
                        f"(exact dist={move.src if player==0 else 25-move.src}, "
                        f"and larger-die rule may not apply)")
    elif move.dst == BAR:
        return "ILLEGAL: destination BAR is not a valid move destination"
    else:
        if not (1 <= move.dst <= 24):
            return f"ILLEGAL: destination point {move.dst} out of range"
        if not is_point_open(state, move.dst, player):
            val = state.board[move.dst - 1]
            return (f"ILLEGAL: destination point {move.dst} blocked by opponent "
                    f"({abs(val)} checkers there)")
        # Check hit flag consistency
        opp_sign = -ps
        has_blot = (state.board[move.dst - 1] == opp_sign)
        if move.hit and not has_blot:
            return f"ILLEGAL: move.hit=True but no opponent blot at point {move.dst}"
        if not move.hit and has_blot:
            return f"WARNING: move.hit=False but opponent blot exists at point {move.dst}"

    # 3. Die value consistency
    if move.src == BAR:
        expected_dst = bar_entry_point(player, move.die)
        if move.dst != expected_dst and move.dst != OFF:
            return (f"ILLEGAL: die {move.die} from bar should enter on point "
                    f"{expected_dst}, not {move.dst}")
    elif move.dst != OFF:
        if player == 0:
            expected = move.src - move.die
        else:
            expected = move.src + move.die
        if move.dst != expected:
            return (f"ILLEGAL: die {move.die} from point {move.src} should go to "
                    f"point {expected}, not {move.dst}")

    return "OK"


def explain_full_turn(
    state: GameState,
    turn: Turn,
) -> str:
    """Return a multi-line explanation of the full turn legality."""
    player = state.current_player
    lines = [f"Checking Turn for player {player}: {format_full_turn(turn)}"]

    legal = get_legal_turns(state)
    legal_strs = {format_full_turn(t) for t in legal}
    turn_str = format_full_turn(turn)

    if turn_str in legal_strs:
        lines.append(f"  ✓ Turn is LEGAL ({len(legal)} legal turns total)")
        return "\n".join(lines)

    lines.append(f"  ✗ Turn NOT found in legal moves ({len(legal)} legal turns)")

    # Step through atomic moves to find failure point
    cur = state.clone()
    for i, move in enumerate(turn.moves):
        reason = explain_atomic_move(cur, move, player)
        lines.append(f"  Move {i+1}: {format_atomic_move(move)}  →  {reason}")
        if reason.startswith("ILLEGAL"):
            break
        apply_atomic_move_inplace(cur, move, player)

    # Show mandatory dice usage
    max_len = max((len(t) for t in legal), default=0)
    lines.append(f"  Max legal moves for this dice={max_len}, this turn has {len(turn)}")

    if turn not in legal and len(turn) > 0:
        lines.append(f"  Legal turns include: " +
                     ", ".join(list(legal_strs)[:5]) +
                     ("..." if len(legal_strs) > 5 else ""))

    return "\n".join(lines)


def explain_illegal_action(
    state: GameState,
    dice: Optional[List[int]],
    action: Turn,
) -> str:
    """Top-level: explain why *action* is illegal given *state* and *dice*.

    Sets state.dice = dice if provided (non-destructive clone).
    """
    s = state.clone()
    if dice is not None:
        s.dice = dice[:]

    return explain_full_turn(s, action)
