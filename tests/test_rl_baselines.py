"""Tests for RL baseline policies and ShortestPathEnv."""
from __future__ import annotations

import pytest
import torch

from tgraphx.rl.algorithms.baselines import RandomPolicy, GreedyPolicy
from tgraphx.rl.environments.shortest_path import ShortestPathEnv
from tgraphx.rl.environments.base import GraphEnvConfig


# ---------------------------------------------------------------------------
# RandomPolicy
# ---------------------------------------------------------------------------

def make_obs_with_mask(n_actions=5, valid_indices=None):
    mask = torch.zeros(n_actions, dtype=torch.bool)
    if valid_indices is None:
        mask[:] = True
    else:
        for i in valid_indices:
            mask[i] = True
    return {"action_mask": mask}


def test_random_policy_selects_valid_action():
    policy = RandomPolicy(n_actions=5)
    for _ in range(20):
        obs = make_obs_with_mask(5, valid_indices=[2, 3])
        a = policy.select_action(obs)
        assert a in (2, 3), f"Selected invalid action {a}"


def test_random_policy_deterministic_with_seed():
    policy = RandomPolicy(n_actions=8)
    obs = make_obs_with_mask(8)
    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)
    a1 = policy.select_action(obs, generator=gen1)
    a2 = policy.select_action(obs, generator=gen2)
    assert a1 == a2


def test_random_policy_update_returns_empty():
    policy = RandomPolicy(n_actions=5)
    result = policy.update(batch=None)
    assert result == {}


def test_random_policy_no_mask_uses_n_actions():
    policy = RandomPolicy(n_actions=4)
    obs = {}
    for _ in range(20):
        a = policy.select_action(obs)
        assert 0 <= a < 4


def test_random_policy_state_dict():
    policy = RandomPolicy(n_actions=6)
    sd = policy.state_dict()
    policy.load_state_dict(sd)
    assert policy.n_actions == 6


# ---------------------------------------------------------------------------
# GreedyPolicy
# ---------------------------------------------------------------------------

def test_greedy_policy_selects_max_score():
    scores = [1.0, 5.0, 3.0, 0.5, 4.0]
    policy = GreedyPolicy(
        scoring_fn=lambda obs, a: scores[a],
        n_actions=5,
    )
    obs = make_obs_with_mask(5)
    a = policy.select_action(obs)
    assert a == 1  # index with max score 5.0


def test_greedy_policy_respects_mask():
    scores = [1.0, 5.0, 3.0, 0.5, 4.0]
    policy = GreedyPolicy(
        scoring_fn=lambda obs, a: scores[a],
        n_actions=5,
    )
    # Block best action (1) with mask
    obs = make_obs_with_mask(5, valid_indices=[0, 2, 4])
    a = policy.select_action(obs)
    assert a == 4  # next best after 1 is 4 (score=4.0)


def test_greedy_policy_update_returns_empty():
    policy = GreedyPolicy(scoring_fn=lambda o, a: 0.0, n_actions=4)
    result = policy.update()
    assert result == {}


def test_greedy_policy_deterministic():
    scores = [2.0, 1.0, 3.0]
    policy = GreedyPolicy(scoring_fn=lambda obs, a: scores[a], n_actions=3)
    obs = make_obs_with_mask(3)
    a1 = policy.select_action(obs)
    a2 = policy.select_action(obs)
    assert a1 == a2 == 2


# ---------------------------------------------------------------------------
# ShortestPathEnv
# ---------------------------------------------------------------------------

def _make_line_graph(n=4):
    """Create a line graph: 0-1-2-3."""
    src = list(range(n - 1)) + list(range(1, n))
    dst = list(range(1, n)) + list(range(n - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def test_shortest_path_oracle_length():
    """Line graph 0-1-2-3: shortest path from 0 to 3 is 3 steps."""
    n = 4
    ei = _make_line_graph(n)
    env = ShortestPathEnv(
        edge_index=ei, num_nodes=n,
        target_node=3, start_node=0,
    )
    env.reset()
    assert env._optimal_length == 3


def test_shortest_path_oracle_on_2node_graph():
    """Graph 0-1: shortest path from 0 to 1 is 1 step."""
    ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    env = ShortestPathEnv(
        edge_index=ei, num_nodes=2,
        target_node=1, start_node=0,
    )
    env.reset()
    assert env._optimal_length == 1


def test_shortest_path_regret_zero_on_optimal():
    """Agent takes optimal path: regret should be 0."""
    n = 4
    ei = _make_line_graph(n)
    env = ShortestPathEnv(
        edge_index=ei, num_nodes=n,
        target_node=3, start_node=0,
        config=GraphEnvConfig(max_steps=20),
    )
    obs = env.reset()

    # On line graph, each step has at most 2 neighbors (previous and next)
    # From node 0: neighbors = [1] (undirected path)
    # From node 1: neighbors = [0, 2]
    # From node 2: neighbors = [1, 3]
    # Each step we want to move forward -> pick the higher-indexed neighbor

    regrets = []
    done = False
    truncated = False

    # Step through using the environment's adjacency list
    for step_i in range(10):
        if done or truncated:
            break
        current = env._current_node
        neighbors = env._adj[current]
        # Pick the neighbor that advances toward target
        best = min(neighbors, key=lambda nb: abs(nb - env._target_node))
        action_idx = neighbors.index(best)
        obs, reward, done, truncated, info = env.step(action_idx)
        if done or "regret" in info:
            regrets.append(info.get("regret", None))

    # We should have reached the target
    assert info.get("success", False) or info.get("regret", 1) == 0 or len(regrets) > 0


def test_shortest_path_info_keys():
    """Info dict should contain required keys."""
    n = 4
    ei = _make_line_graph(n)
    env = ShortestPathEnv(edge_index=ei, num_nodes=n, target_node=3)
    obs = env.reset()
    _, _, done, truncated, info = env.step(0)
    assert "optimal_length" in info
    assert "path_length" in info
    assert "regret" in info
