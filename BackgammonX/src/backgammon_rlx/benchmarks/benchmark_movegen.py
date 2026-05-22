"""Benchmark legal move generation speed."""
from __future__ import annotations

import time
import random
from typing import List

from ..env.state import GameState
from ..env.movegen import get_legal_turns
from ..env.env import BackgammonEnv
from ..agents.random_agent import RandomLegalAgent


def _play_random_game(seed: int) -> int:
    env   = BackgammonEnv()
    agent = RandomLegalAgent(seed=seed)
    env.reset(seed=seed)
    steps = 0
    while not env.is_terminal():
        turns  = env.legal_actions()
        action = agent.select_action(env.state, turns)
        env.step(action)
        steps += 1
    return steps


def benchmark_movegen(n_positions: int = 10_000) -> dict:
    """Generate legal moves from N random mid-game positions."""
    # Collect positions by playing partial games
    positions: List[GameState] = []
    rng = random.Random(42)
    env = BackgammonEnv()
    agent = RandomLegalAgent(seed=42)

    env.reset(seed=0)
    while len(positions) < n_positions:
        if env.is_terminal():
            env.reset(seed=rng.randint(0, 2**31))
        turns = env.legal_actions()
        positions.append(env.state.clone())
        env.step(agent.select_action(env.state, turns))

    t0 = time.perf_counter()
    total_turns = 0
    for s in positions:
        turns = get_legal_turns(s)
        total_turns += len(turns)
    elapsed = time.perf_counter() - t0

    return {
        "positions":      n_positions,
        "total_turns":    total_turns,
        "avg_turns":      total_turns / n_positions,
        "elapsed_s":      elapsed,
        "positions_per_s": n_positions / elapsed,
    }


def benchmark_random_games(n_games: int = 500) -> dict:
    """Simulate N complete random games."""
    t0 = time.perf_counter()
    total_steps = 0
    for i in range(n_games):
        total_steps += _play_random_game(i)
    elapsed = time.perf_counter() - t0
    return {
        "games":        n_games,
        "total_steps":  total_steps,
        "avg_length":   total_steps / n_games,
        "elapsed_s":    elapsed,
        "games_per_s":  n_games / elapsed,
        "steps_per_s":  total_steps / elapsed,
    }
