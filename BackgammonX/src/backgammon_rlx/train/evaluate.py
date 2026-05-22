"""Evaluation script with statistical confidence intervals.

    python -m backgammon_rlx.train.evaluate \
        --checkpoint runs/latest/checkpoints/latest.pt --games 1000
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from ..env.env import BackgammonEnv
from ..env.encoding import ObservationEncoder, ActionEncoder
from ..env.movegen import get_legal_turns
from ..env.rules import is_terminal, winner, score_value
from ..agents.random_agent import RandomLegalAgent
from ..agents.heuristic_agent import GreedyPipAgent, HeuristicAgent
from ..agents.neural_agent import NeuralAgent
from ..models.policy_value_net import BackgammonPolicyValueNet
from ..utils.checkpoint import load_checkpoint
from ..utils.device import get_device
from ..utils.seed import seed_everything


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def binomial_ci(n: int, k: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson confidence interval for win rate k/n."""
    if n == 0:
        return (0.0, 1.0)
    p  = k / n
    q  = 1 - p
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * math.sqrt(p*q/n + z**2/(4*n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_mean_ci(data: List[float], z: float = 1.96,
                       n_boot: int = 1000) -> tuple[float, float, float]:
    arr = np.array(data)
    mu  = arr.mean()
    boot_means = np.array([
        np.random.choice(arr, len(arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    se = boot_means.std()
    return mu, mu - z*se, mu + z*se


# ---------------------------------------------------------------------------
# Single match evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    agent_a,
    agent_b,
    n_games:   int,
    obs_enc:   ObservationEncoder,
    act_enc:   ActionEncoder,
    seed:      int = 0,
    deterministic: bool = True,
) -> Dict:
    """Play n_games between agent_a (player 0) and agent_b (player 1)."""
    rng  = np.random.default_rng(seed)
    env  = BackgammonEnv(obs_enc, act_enc)

    wins_a = wins_b = 0
    gammons = backgammons = 0
    scores:   List[float] = []
    lengths:  List[int]   = []

    for g in range(n_games):
        obs  = env.reset(seed=int(rng.integers(0, 2**31)))
        done = False

        while not done:
            state  = env.state
            player = state.current_player
            turns  = env.legal_actions()

            if player == 0:
                action = agent_a.select_action(state, turns)
            else:
                action = agent_b.select_action(state, turns)

            obs, reward, done, info = env.step(action)

        w     = info.get("winner", 0)
        score = info.get("score", 1)
        length = info.get("game_length", 0)

        if w == 0:
            wins_a += 1
        else:
            wins_b += 1

        if score == 2:
            gammons += 1
        elif score == 3:
            backgammons += 1

        scores.append(float(score))
        lengths.append(length)

    n = n_games
    win_rate  = wins_a / n
    ci_lo, ci_hi = binomial_ci(n, wins_a)
    mu_score, s_lo, s_hi = bootstrap_mean_ci(scores)
    mu_len, l_lo, l_hi   = bootstrap_mean_ci(lengths)

    return {
        "games":            n,
        "wins_a":           wins_a,
        "wins_b":           wins_b,
        "win_rate_a":       win_rate,
        "win_rate_ci":      [ci_lo, ci_hi],
        "gammon_rate":      gammons / n,
        "backgammon_rate":  backgammons / n,
        "mean_score":       mu_score,
        "mean_score_ci":    [s_lo, s_hi],
        "mean_length":      mu_len,
        "mean_length_ci":   [l_lo, l_hi],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games",      type=int, default=500)
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--out",        default=None,
                        help="JSON output file (optional)")
    args = parser.parse_args()

    seed_everything(args.seed)
    device   = get_device(args.device)
    obs_enc  = ObservationEncoder()
    act_enc  = ActionEncoder()

    # Peek at checkpoint config to build matching model architecture
    import torch as _torch
    _ckpt_peek = _torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _cfg = _ckpt_peek.get("config", {})
    model = BackgammonPolicyValueNet(
        state_dim=_cfg.get("state_dim", 256),
        act_dim=_cfg.get("act_dim", 256),
        n_point_res=_cfg.get("n_point_residual", 4),
        n_action_res=_cfg.get("n_action_residual", 3),
    )
    load_checkpoint(args.checkpoint, model, device=str(device))
    model.to(device).eval()

    neural = NeuralAgent(model, obs_enc, act_enc, device=str(device),
                          deterministic=True)

    baselines = [
        ("random",    RandomLegalAgent(seed=args.seed)),
        ("greedy_pip",GreedyPipAgent()),
        ("heuristic", HeuristicAgent()),
    ]

    all_results = {}
    for name, opp in baselines:
        print(f"\n--- vs {name} ---")
        res = run_evaluation(neural, opp, args.games,
                             obs_enc, act_enc, seed=args.seed)
        all_results[f"vs_{name}"] = res
        print(f"  win_rate={res['win_rate_a']:.3f} "
              f"[{res['win_rate_ci'][0]:.3f}, {res['win_rate_ci'][1]:.3f}]  "
              f"gammon={res['gammon_rate']:.3f}  "
              f"backgammon={res['backgammon_rate']:.3f}  "
              f"avg_len={res['mean_length']:.1f}")

    if args.out:
        Path(args.out).write_text(json.dumps(all_results, indent=2))
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
