"""Human-readable backgammon move notation.

Supports the standard point/point(*) format:
    13/8        normal move from 13 to 8
    bar/22      bar entry onto point 22
    6/off       bearing off from point 6
    24/18*      move from 24 to 18, hitting a blot
    8/5*/3      two-move sequence: 8→5 (hit) then 5→3

parse_move_notation() is a best-effort parser for test fixtures and
interactive input.  It does NOT validate against the legal-move generator.
"""
from __future__ import annotations

import re
from typing import List, Optional

from ..env.state import AtomicMove, Turn, BAR, OFF


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_point(p: int) -> str:
    if p == BAR:
        return "bar"
    if p == OFF:
        return "off"
    return str(p)


def format_atomic_move(move: AtomicMove) -> str:
    src = _fmt_point(move.src)
    dst = _fmt_point(move.dst)
    hit = "*" if move.hit else ""
    return f"{src}/{dst}{hit}"


def format_full_turn(turn: Turn) -> str:
    """Format a complete turn as standard notation.

    Identical src→dst pairs from the same source point are collapsed:
        6/3 6/3  →  6/3(2)
    """
    if not turn.moves:
        return "(pass)"

    moves = list(turn.moves)

    # Count repeats for compact notation
    counts: dict = {}
    for m in moves:
        key = (m.src, m.dst, m.hit)
        counts[key] = counts.get(key, 0) + 1

    parts = []
    seen: set = set()
    for m in moves:
        key = (m.src, m.dst, m.hit)
        if key in seen:
            continue
        seen.add(key)
        text = format_atomic_move(m)
        cnt  = counts[key]
        if cnt > 1:
            text += f"({cnt})"
        parts.append(text)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_TOKEN = re.compile(
    r"(bar|\d+)/(off|\d+)(\*)?(?:\((\d+)\))?",
    re.IGNORECASE,
)


def _parse_src(tok: str) -> int:
    return BAR if tok.lower() == "bar" else int(tok)


def _parse_dst(tok: str) -> int:
    return OFF if tok.lower() == "off" else int(tok)


def parse_atomic_move(text: str, die: int = 0) -> Optional[AtomicMove]:
    """Parse a single atomic move like '13/8', 'bar/22', '6/off*'.

    The die value defaults to 0 when not inferrable from notation alone.
    """
    m = _TOKEN.match(text.strip())
    if not m:
        return None
    src = _parse_src(m.group(1))
    dst = _parse_dst(m.group(2))
    hit = m.group(3) == "*" if m.group(3) else False
    return AtomicMove(src=src, dst=dst, die=die, hit=hit)


def parse_move_notation(text: str, player: int = 0) -> Turn:
    """Parse a full-turn notation string into a Turn.

    The die values are inferred from the movement distance when the board
    direction is known (player 0 moves high→low, player 1 moves low→high).
    Compact duplicate notation like '6/3(2)' is expanded.
    """
    moves: List[AtomicMove] = []
    for m in _TOKEN.finditer(text):
        src   = _parse_src(m.group(1))
        dst   = _parse_dst(m.group(2))
        hit   = m.group(3) == "*" if m.group(3) else False
        count = int(m.group(4)) if m.group(4) else 1

        # Infer die from distance
        if src == BAR:
            die = (25 - dst) if player == 0 else dst
        elif dst == OFF:
            die = src if player == 0 else (25 - src)
        else:
            die = abs(src - dst)

        for _ in range(count):
            moves.append(AtomicMove(src=src, dst=dst, die=die, hit=hit))

    return Turn.from_list(moves)
