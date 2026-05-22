"""GNU Backgammon adapter.

Communicates with gnubg via subprocess in --tty / batch mode.
If gnubg is not installed, all methods degrade gracefully.

Discovery order:
  1. $GNUBG_BIN env variable
  2. gnubg on PATH
  3. gnubg-cli on PATH
  4. /usr/local/bin/gnubg, /usr/bin/gnubg

Usage:
    adapter = GnuBackgammonAdapter()
    if adapter.is_available():
        turn, equity = adapter.request_move(state)
    else:
        print(adapter.unavailable_reason())

CLI:
    python -m backgammon_rlx.engines.gnu_backgammon_check --status
    python -m backgammon_rlx.engines.gnu_backgammon_check --all-fixtures
    python -m backgammon_rlx.engines.gnu_backgammon_check --random-positions 100
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import textwrap
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..env.state import GameState, Turn, AtomicMove, BAR, OFF
from ..env.movegen import get_legal_turns
from ..env.rules import bar_entry_point, player_sign, home_board_range
from ..notation.move_notation import format_full_turn
from .external_engine import ExternalEngineAgent, ExternalEngineError


# ---------------------------------------------------------------------------
# Helpers: Position encoding
# ---------------------------------------------------------------------------

def _board_to_gnubg_id(state: GameState) -> str:
    """Export state as GNU Backgammon Position ID (base-64 encoded).

    GnuBG uses a 73-bit position encoding. This function produces the
    canonical position-ID string that gnubg --tty can import via
    `set board <id>`.

    The encoding follows the GnuBG documentation:
      https://www.gnu.org/software/gnubg/manual/gnubg.html#Position-ID
    """
    # Build checker counts from P0 (player) and P1 (opponent)
    # For player 0 perspective (GnuBG "player on roll"):
    #   Points 1-24 in player 0's moving direction
    player = state.current_player
    opp    = 1 - player
    ps     = player_sign(player)

    # gnubg point 1 = pip closest to bearing off for current player
    # For player 0: point 1 → board[0], point 24 → board[23]
    # For player 1: point 1 → board[23], point 24 → board[0] (mirrored)

    own_counts  = [0] * 26   # 0=bar, 1-24=board, 25=off
    opp_counts  = [0] * 26

    # Bar
    own_counts[0]  = state.bar[player]
    opp_counts[0]  = state.bar[opp]
    # Board
    for i in range(24):
        val = state.board[i]
        if player == 0:
            pt = i + 1
        else:
            pt = 24 - i
        if val > 0 and ps == 1:
            own_counts[pt] = val
        elif val < 0 and ps == 1:
            opp_counts[pt] = -val
        elif val < 0 and ps == -1:
            own_counts[pt] = -val
        elif val > 0 and ps == -1:
            opp_counts[pt] = val

    own_counts[25] = state.borne_off[player]
    opp_counts[25] = state.borne_off[opp]

    # Encode into 73 bits following GnuBG spec
    bits = []
    for counts in (own_counts, opp_counts):
        for pt in range(26):
            n = counts[pt]
            bits.extend([1] * n + [0])

    # Pad to 80 bits (10 bytes); only 73 meaningful
    while len(bits) < 80:
        bits.append(0)

    # Convert bits to bytes, then base64
    import base64
    byte_vals = []
    for i in range(0, 80, 8):
        byte_val = 0
        for j in range(8):
            if i + j < len(bits):
                byte_val |= bits[i + j] << j
        byte_vals.append(byte_val)

    raw = bytes(byte_vals)[:10]
    return base64.b64encode(raw).decode("ascii")


def _dice_to_gnubg(dice: List[int]) -> str:
    """Convert dice list to gnubg dice string."""
    if not dice:
        return ""
    if len(dice) == 4:
        return f"{dice[0]} {dice[0]}"
    if len(dice) >= 2:
        return f"{dice[0]} {dice[1]}"
    return str(dice[0])


# ---------------------------------------------------------------------------
# Move notation parser (gnubg output → local Turn)
# ---------------------------------------------------------------------------

_GNUBG_MOVE_RE = re.compile(
    r'(?:(?P<src>bar|\d+)/(?P<dst>off|\d+)(?P<hit>\*)?)',
    re.IGNORECASE,
)

def _parse_gnubg_move(line: str, state: GameState) -> Optional[Turn]:
    """Parse a line of gnubg move output into a local Turn.

    gnubg uses point numbers from the current player's perspective:
    point 1 = closest to bearing off, point 24 = furthest.
    We convert these to global board point numbers.
    """
    player = state.current_player
    raw_moves = []

    for m in _GNUBG_MOVE_RE.finditer(line):
        src_s = m.group("src").lower()
        dst_s = m.group("dst").lower()
        hit   = m.group("hit") == "*" if m.group("hit") else False

        # Convert gnubg perspective → global board points
        if src_s == "bar":
            src = BAR
        else:
            gnubg_pt = int(src_s)
            if player == 0:
                src = gnubg_pt
            else:
                src = 25 - gnubg_pt

        if dst_s == "off":
            dst = OFF
        else:
            gnubg_pt = int(dst_s)
            if player == 0:
                dst = gnubg_pt
            else:
                dst = 25 - gnubg_pt

        # Infer die value
        if src == BAR:
            die = bar_entry_point(player, 0)  # placeholder
            if player == 0:
                die = 25 - dst
            else:
                die = dst
        elif dst == OFF:
            from ..env.rules import _checker_distance
            die_candidate = _checker_distance(player, src)
            die = die_candidate  # best guess; larger die handled below
        else:
            die = abs(src - dst)

        raw_moves.append(AtomicMove(src=src, dst=dst, die=die, hit=hit))

    if not raw_moves:
        return None
    return Turn.from_list(raw_moves)


def _best_match(gnubg_turn: Turn, legal: List[Turn]) -> Optional[Turn]:
    """Find the best-matching legal turn for a parsed gnubg turn.

    Since die-value parsing from gnubg output is imperfect, we match
    by (src, dst) sequences only.
    """
    def turn_key(t: Turn) -> tuple:
        return tuple((m.src, m.dst) for m in t.moves)

    gnubg_key = turn_key(gnubg_turn)
    for t in legal:
        if turn_key(t) == gnubg_key:
            return t
    return None


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

_SEARCH_PATHS = [
    "gnubg",
    "gnubg-cli",
    "/usr/bin/gnubg",
    "/usr/local/bin/gnubg",
    "/opt/homebrew/bin/gnubg",
]


def _find_gnubg_binary() -> Optional[str]:
    envvar = os.environ.get("GNUBG_BIN")
    if envvar and shutil.which(envvar):
        return envvar
    for candidate in _SEARCH_PATHS:
        found = shutil.which(candidate)
        if found:
            return found
    return None


class GnuBackgammonAdapter(ExternalEngineAgent):
    """Adapter for GNU Backgammon via subprocess.

    Communication protocol:
    - spawn `gnubg --tty` (or `gnubg-cli`)
    - pipe a sequence of commands to stdin
    - read output from stdout
    - parse move/equity lines

    If gnubg is not installed, all operations fail gracefully.

    NOTE: Full GnuBG integration is fragile because GnuBG's --tty mode
    output format varies by version. This adapter implements best-effort
    parsing with validation against local legal moves.
    """

    def __init__(self, timeout: float = 10.0) -> None:
        self._binary  = _find_gnubg_binary()
        self._timeout = timeout
        self._version_cache: Optional[str] = None

    def is_available(self) -> bool:
        return self._binary is not None

    def unavailable_reason(self) -> str:
        if self.is_available():
            return "available"
        return (
            "gnubg not found on PATH or in common locations. "
            "Install with: sudo apt install gnubg  OR  brew install gnubg. "
            f"Searched: {_SEARCH_PATHS}. "
            "Set $GNUBG_BIN env var to override."
        )

    def version(self) -> Optional[str]:
        if not self.is_available():
            return None
        if self._version_cache is not None:
            return self._version_cache
        try:
            result = subprocess.run(
                [self._binary, "--version"],
                capture_output=True, text=True, timeout=5.0
            )
            first_line = result.stdout.strip().split("\n")[0]
            self._version_cache = first_line
            return first_line
        except Exception:
            return None

    def _run_gnubg_commands(self, commands: str) -> str:
        """Run a sequence of gnubg --tty commands and return stdout."""
        if not self.is_available():
            raise ExternalEngineError(self.unavailable_reason())

        try:
            result = subprocess.run(
                [self._binary, "--tty"],
                input=commands,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            raise ExternalEngineError(
                f"gnubg timed out after {self._timeout}s. "
                "Check that gnubg --tty mode is supported on your system."
            )
        except Exception as e:
            raise ExternalEngineError(f"gnubg subprocess error: {e}")

    def request_move(self, state: GameState) -> Tuple[Turn, Optional[float]]:
        """Request best move and equity from gnubg.

        Returns (turn, equity) where equity is from current player's perspective
        in range [-1, 1] (or None if not parsed).

        Raises ExternalEngineError if gnubg is unavailable or output cannot
        be parsed. The caller should catch this and fall back gracefully.
        """
        if not self.is_available():
            raise ExternalEngineError(self.unavailable_reason())

        legal = get_legal_turns(state)
        if not legal:
            return Turn(), None

        pos_id = _board_to_gnubg_id(state)
        dice_str = _dice_to_gnubg(state.dice)

        # GnuBG --tty command sequence
        commands = textwrap.dedent(f"""\
            set board {pos_id}
            set dice {dice_str}
            hint
            quit
        """)

        try:
            output = self._run_gnubg_commands(commands)
        except ExternalEngineError:
            raise

        # Parse output: look for the top-ranked move line
        equity = None
        best_line = None
        for line in output.split("\n"):
            # gnubg hint output contains lines like:
            #   1.  Rolled 31:        13/10 6/5  Eq.: +0.123
            # or:
            #   Move                   Eq.
            #    1 13/10 6/5          +0.123
            if re.search(r'\d+/(?:\d+|off)', line, re.IGNORECASE):
                best_line = line
                # Try to extract equity
                eq_match = re.search(r'[Ee]q\.?\s*:?\s*([+-]?\d+\.?\d*)', line)
                if eq_match:
                    try:
                        equity = float(eq_match.group(1))
                    except ValueError:
                        pass
                break

        if best_line is None:
            raise ExternalEngineError(
                f"gnubg produced no parseable move output.\n"
                f"State: {state}\nDice: {state.dice}\n"
                f"gnubg output (first 500 chars):\n{output[:500]}"
            )

        parsed_turn = _parse_gnubg_move(best_line, state)
        if parsed_turn is None:
            raise ExternalEngineError(
                f"Could not parse gnubg move from line: '{best_line}'"
            )

        matched = _best_match(parsed_turn, legal)
        if matched is None:
            legal_strs = [format_full_turn(t) for t in legal]
            raise ExternalEngineError(
                f"gnubg returned move not in local legal moves.\n"
                f"Parsed: {format_full_turn(parsed_turn)}\n"
                f"Legal:  {legal_strs[:10]}\n"
                f"State:  {state}\nDice:   {state.dice}\n"
                f"Line:   '{best_line}'"
            )

        return matched, equity

    def evaluate_position(self, state: GameState) -> Dict:
        """Return equity evaluation from gnubg (best-effort).

        Returns dict with keys: equity, win_prob, gammon_prob, backgammon_prob.
        All values are from current player's perspective.
        """
        if not self.is_available():
            raise ExternalEngineError(self.unavailable_reason())

        pos_id = _board_to_gnubg_id(state)
        commands = textwrap.dedent(f"""\
            set board {pos_id}
            eval
            quit
        """)

        try:
            output = self._run_gnubg_commands(commands)
        except ExternalEngineError:
            raise

        result = {"equity": None, "win_prob": None, "gammon_prob": None,
                  "backgammon_prob": None, "raw_output": output[:500]}

        for line in output.split("\n"):
            # Look for lines like: "Cubeful equity: +0.123"
            eq = re.search(r'equity.*?([+-]?\d+\.\d+)', line, re.IGNORECASE)
            if eq and result["equity"] is None:
                try:
                    result["equity"] = float(eq.group(1))
                except ValueError:
                    pass
            # Win probability: "Win: 0.612"
            wp = re.search(r'[Ww]in\s*:?\s*(\d+\.?\d*)', line)
            if wp and result["win_prob"] is None:
                try:
                    result["win_prob"] = float(wp.group(1))
                except ValueError:
                    pass

        return result

    def select_action(self, state: GameState, legal_turns=None) -> Turn:
        """Select action, falling back to first legal move if gnubg unavailable."""
        if not self.is_available():
            raise ExternalEngineError(self.unavailable_reason())
        turn, _ = self.request_move(state)
        return turn


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _run_cli() -> None:
    import argparse
    import json
    import random

    parser = argparse.ArgumentParser(
        prog="python -m backgammon_rlx.engines.gnu_backgammon_check",
        description="GNU Backgammon integration status and validation"
    )
    parser.add_argument("--status",          action="store_true",
                        help="Show gnubg availability and version")
    parser.add_argument("--all-fixtures",    action="store_true",
                        help="Validate gnubg against all golden fixtures")
    parser.add_argument("--random-positions",type=int, default=0, metavar="N",
                        help="Validate gnubg against N random game positions")
    parser.add_argument("--timeout",         type=float, default=10.0)
    args = parser.parse_args()

    adapter = GnuBackgammonAdapter(timeout=args.timeout)

    print("=" * 50)
    print("GNU Backgammon Integration Status")
    print("=" * 50)
    print(f"  Available:  {adapter.is_available()}")
    if adapter.is_available():
        v = adapter.version()
        print(f"  Binary:     {adapter._binary}")
        print(f"  Version:    {v or 'unknown'}")
    else:
        print(f"  Reason:     {adapter.unavailable_reason()}")
    print()

    if not adapter.is_available():
        print("Skipping validation: gnubg not available.")
        return

    if args.all_fixtures:
        from ..validation.golden_tests import load_fixtures
        fixtures = load_fixtures()
        print(f"Testing {len(fixtures)} golden fixtures against gnubg...")
        ok = fail = skip = 0
        for fix in fixtures:
            if fix.get("is_terminal") or not fix.get("dice"):
                skip += 1
                continue
            from ..validation.golden_tests import _build_state
            state = _build_state(fix)
            try:
                turn, equity = adapter.request_move(state)
                print(f"  ✅ {fix['id']}: {format_full_turn(turn)}"
                      f"  eq={equity:.3f}" if equity else "")
                ok += 1
            except ExternalEngineError as e:
                print(f"  ❌ {fix['id']}: {e}")
                fail += 1
        print(f"\nFixtures: {ok} OK, {fail} failed, {skip} skipped")

    if args.random_positions > 0:
        from ..env.env import BackgammonEnv
        from ..agents.random_agent import RandomLegalAgent

        env   = BackgammonEnv()
        agent = RandomLegalAgent(seed=0)
        env.reset(seed=0)
        ok = fail = 0
        attempts = 0

        print(f"Testing {args.random_positions} random positions against gnubg...")
        while ok + fail < args.random_positions and attempts < args.random_positions * 20:
            attempts += 1
            if env.is_terminal():
                env.reset(seed=attempts)
                continue
            state = env.state
            if not state.dice:
                env.step(agent.select_action(state, env.legal_actions()))
                continue
            try:
                turn, equity = adapter.request_move(state)
                ok += 1
            except ExternalEngineError as e:
                print(f"  Position {ok+fail+1}: {e}")
                fail += 1
            env.step(agent.select_action(state, env.legal_actions()))

        print(f"Random positions: {ok} OK, {fail} failed")


if __name__ == "__main__":
    _run_cli()
