"""Full-game validation: play complete games and verify every invariant.

This is the definitive answer to 'Is this really backgammon?'

    python -m backgammon_rlx.validation.full_game_check \\
      --games 1000 \\
      --agents random,heuristic \\
      --strict-invariants \\
      --seed 123
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..env.env import BackgammonEnv
from ..env.rules import (
    total_checkers, is_terminal, winner, score_value,
    home_board_range, player_sign, _sign,
)
from ..env.movegen import apply_atomic_move_inplace
from ..env.state import GameState
from ..notation.move_notation import format_full_turn
from ..notation.position_io import state_to_dict
from ..validation.invariants import check_state_invariants, InvariantError


def _make_agent(name: str, seed: int = 0):
    if name == "random":
        from ..agents.random_agent import RandomLegalAgent
        return RandomLegalAgent(seed=seed)
    if name == "heuristic":
        from ..agents.heuristic_agent import HeuristicAgent
        return HeuristicAgent()
    if name == "greedy":
        from ..agents.heuristic_agent import GreedyPipAgent
        return GreedyPipAgent()
    raise ValueError(f"Unknown agent: {name}")


def _check_position_invariants(state: GameState, context: str = "") -> List[str]:
    """Return list of violation strings (empty = OK)."""
    errors = []
    for p in range(2):
        cnt = total_checkers(state, p)
        if cnt != 15:
            errors.append(f"{context}: player {p} checker count={cnt} (expected 15)")
    for i, val in enumerate(state.board):
        if abs(val) > 15:
            errors.append(f"{context}: point {i+1} has impossible count {val}")
    for p in range(2):
        if state.bar[p] < 0:
            errors.append(f"{context}: bar[{p}]={state.bar[p]} < 0")
        if not 0 <= state.borne_off[p] <= 15:
            errors.append(f"{context}: borne_off[{p}]={state.borne_off[p]} invalid")
    return errors


def _check_scoring_consistency(state: GameState) -> List[str]:
    errors = []
    if not is_terminal(state):
        return errors
    w = winner(state)
    if w is None:
        errors.append("Terminal state but winner() returned None")
        return errors
    if state.borne_off[w] != 15:
        errors.append(f"Winner {w} borne_off={state.borne_off[w]} != 15")
    loser = 1 - w
    sv = score_value(state)
    if sv not in (1, 2, 3):
        errors.append(f"score_value={sv} not in (1,2,3)")
    if state.borne_off[loser] > 0 and sv != 1:
        errors.append(f"Loser has {state.borne_off[loser]} borne off but score={sv}!=1")
    if state.borne_off[loser] == 0 and sv == 1:
        errors.append(f"Loser has 0 borne off but score=1 (should be ≥2)")
    return errors


def run_game_check(
    agent_a_name: str,
    agent_b_name: str,
    seed: int,
    strict_invariants: bool,
    max_steps: int = 5000,
) -> Dict[str, Any]:
    agent_a = _make_agent(agent_a_name, seed)
    agent_b = _make_agent(agent_b_name, seed + 1)
    env     = BackgammonEnv()
    obs     = env.reset(seed=seed)
    violations: List[str] = []
    illegal_moves = 0
    steps = 0
    trace_sample: Optional[Dict] = None

    while not env.is_terminal() and steps < max_steps:
        state  = env.state
        turns  = env.legal_actions()

        # Check invariants before move
        errs = _check_position_invariants(state, f"step {steps} before")
        violations.extend(errs)

        if strict_invariants:
            try:
                check_state_invariants(state, context=f"step {steps}")
            except InvariantError as e:
                violations.append(str(e))

        # Agent selects action
        player = state.current_player
        agent  = agent_a if player == 0 else agent_b
        action = agent.select_action(state, turns)

        # Verify action is legal
        from ..notation.move_notation import format_full_turn
        action_str = format_full_turn(action)
        legal_strs = {format_full_turn(t) for t in turns}
        if action_str not in legal_strs:
            illegal_moves += 1
            violations.append(
                f"step {steps}: ILLEGAL action '{action_str}' not in legal set "
                f"({len(legal_strs)} actions)"
            )

        # Save one trace sample
        if steps == 0:
            obs_step, rew_step, done_step, info_step = env.step(action, trace=True)
            if "trace" in info_step:
                trace_sample = info_step["trace"]
        else:
            obs_step, rew_step, done_step, info_step = env.step(action)

        # Check invariants after move
        errs2 = _check_position_invariants(env.state, f"step {steps} after")
        violations.extend(errs2)

        steps += 1

    # Final checks
    state = env.state
    score_errors = _check_scoring_consistency(state)
    violations.extend(score_errors)

    return {
        "completed": env.is_terminal(),
        "steps": steps,
        "winner": int(winner(state)) if winner(state) is not None else None,
        "score": int(score_value(state)),
        "violations": violations,
        "illegal_moves": illegal_moves,
        "trace_sample": trace_sample,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Full game validation check")
    parser.add_argument("--games",           type=int,  default=1000)
    parser.add_argument("--agents",          default="random,heuristic")
    parser.add_argument("--strict-invariants", action="store_true")
    parser.add_argument("--seed",            type=int,  default=42)
    parser.add_argument("--max-steps",       type=int,  default=5000)
    parser.add_argument("--out",             default=None)
    args = parser.parse_args()

    agent_names = [a.strip() for a in args.agents.split(",")]
    if len(agent_names) < 2:
        agent_names = agent_names * 2
    agent_a, agent_b = agent_names[0], agent_names[1]

    print(f"[full_game_check] {args.games} games  "
          f"agents={agent_a} vs {agent_b}  strict={args.strict_invariants}")
    t0 = time.time()

    total_violations    = 0
    total_illegal       = 0
    total_crashes       = 0
    total_completed     = 0
    score_inconsistency = 0
    scores_seen         = {1: 0, 2: 0, 3: 0}
    sample_traces       = []

    for g in range(args.games):
        try:
            result = run_game_check(
                agent_a_name=agent_a,
                agent_b_name=agent_b,
                seed=args.seed + g,
                strict_invariants=args.strict_invariants,
                max_steps=args.max_steps,
            )
        except Exception as e:
            total_crashes += 1
            print(f"  Game {g}: CRASH: {e}")
            continue

        if result["completed"]:
            total_completed += 1
            sv = result.get("score", 0)
            if sv in scores_seen:
                scores_seen[sv] += 1

        total_violations += len(result["violations"])
        total_illegal    += result["illegal_moves"]

        if result["violations"] and g < 5:
            print(f"  Game {g}: VIOLATIONS: {result['violations'][:3]}")

        if result.get("trace_sample") and len(sample_traces) < 3:
            sample_traces.append(result["trace_sample"])

    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"FULL GAME CHECK RESULTS")
    print(f"{'='*50}")
    print(f"  Games played:         {args.games}")
    print(f"  Completed:            {total_completed}/{args.games}")
    print(f"  Illegal moves:        {total_illegal}")
    print(f"  Invariant violations: {total_violations}")
    print(f"  Crashes:              {total_crashes}")
    print(f"  Scoring: normal={scores_seen[1]} gammon={scores_seen[2]} backgammon={scores_seen[3]}")
    print(f"  Elapsed:              {elapsed:.1f}s")
    passed = (total_violations == 0 and total_illegal == 0 and total_crashes == 0)
    print(f"\n  VERDICT: {'✅ PASS' if passed else '❌ FAIL'}")

    summary = {
        "games":           args.games,
        "completed":       total_completed,
        "illegal_moves":   total_illegal,
        "violations":      total_violations,
        "crashes":         total_crashes,
        "scores":          scores_seen,
        "passed":          passed,
        "elapsed_s":       elapsed,
        "agents":          f"{agent_a} vs {agent_b}",
        "seed":            args.seed,
        "strict_invariants": args.strict_invariants,
    }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\nResults saved to {args.out}")

    return summary


if __name__ == "__main__":
    main()
