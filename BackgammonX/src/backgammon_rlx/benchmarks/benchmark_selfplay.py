"""Benchmark neural-network self-play throughput."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from ..env.encoding import ObservationEncoder, ActionEncoder
from ..env.env import BackgammonEnv
from ..env.movegen import get_legal_turns, canonicalize_state_for_player
from ..models.policy_value_net import BackgammonPolicyValueNet
from ..agents.random_agent import RandomLegalAgent


def benchmark_neural_inference(
    model:      BackgammonPolicyValueNet,
    n_samples:  int   = 1000,
    n_actions:  int   = 20,
    device_str: str   = "cuda",
) -> dict:
    """Measure neural scoring throughput (actions per second)."""
    device  = torch.device(device_str if torch.cuda.is_available() else "cpu")
    obs_enc = ObservationEncoder()
    act_enc = ActionEncoder()
    model   = model.to(device).eval()

    from ..env.encoding import OBS_DIM, ACT_DIM
    obs_t  = torch.randn(n_samples, OBS_DIM,   device=device)
    act_t  = torch.randn(n_samples, n_actions, ACT_DIM, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            model(obs_t[:4], act_t[:4])

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        logits, values = model(obs_t, act_t)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_actions = n_samples * n_actions
    return {
        "device":          str(device),
        "n_samples":       n_samples,
        "n_actions":       n_actions,
        "elapsed_s":       elapsed,
        "samples_per_s":   n_samples / elapsed,
        "actions_per_s":   total_actions / elapsed,
    }


def benchmark_selfplay_throughput(
    model:      BackgammonPolicyValueNet,
    n_games:    int   = 50,
    device_str: str   = "cuda",
) -> dict:
    """Measure end-to-end self-play games per second."""
    from ..rl.buffer import RolloutBuffer
    from ..rl.rollout import collect_rollouts

    device   = torch.device(device_str if torch.cuda.is_available() else "cpu")
    obs_enc  = ObservationEncoder()
    act_enc  = ActionEncoder()
    model    = model.to(device).eval()
    buffer   = RolloutBuffer()

    t0 = time.perf_counter()
    stats = collect_rollouts(model, n_games, obs_enc, act_enc, device, buffer)
    elapsed = time.perf_counter() - t0

    return {
        "device":       str(device),
        "games":        n_games,
        "total_steps":  stats["total_steps"],
        "elapsed_s":    elapsed,
        "games_per_s":  n_games / elapsed,
        "steps_per_s":  stats["total_steps"] / elapsed,
    }
