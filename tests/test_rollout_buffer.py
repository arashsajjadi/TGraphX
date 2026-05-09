"""Tests for RolloutBuffer."""
from __future__ import annotations

import pytest
import torch

from tgraphx.rl.algorithms.replay_buffer import RolloutBuffer


def test_add_and_compute_returns_single_step():
    """R_T = r_T when single terminal step."""
    buf = RolloutBuffer(capacity=10, gamma=0.99, gae_lambda=1.0)
    buf.add(obs={}, action=0, reward=5.0, done=True, value=0.0, log_prob=0.0)
    buf.compute_returns_and_advantages(last_value=0.0)
    assert buf._returns is not None
    # With one step, done=True, last_value unused:
    # delta = r + gamma*0*0 - V = 5.0 - 0.0 = 5.0
    # A = delta, R = A + V = 5.0
    assert float(buf._returns[0].item()) == pytest.approx(5.0, abs=1e-5)


def test_gae_two_step_hand_computed():
    """Two-step GAE with gamma=1, lambda=1, values=0.

    delta_0 = r_0 + 1*V_1*(1-done_0) - V_0 = r_0 + r_1 - 0
    delta_1 = r_1 + 0 - 0 = r_1  (done at step 1)
    A_1 = delta_1 = r_1
    A_0 = delta_0 + gamma*lambda*A_1 = (r_0 + 0 - 0) + r_1 = r_0 + r_1
    """
    buf = RolloutBuffer(capacity=10, gamma=1.0, gae_lambda=1.0)
    r0, r1 = 2.0, 3.0
    buf.add(obs={}, action=0, reward=r0, done=False, value=0.0, log_prob=0.0)
    buf.add(obs={}, action=1, reward=r1, done=True, value=0.0, log_prob=0.0)
    buf.compute_returns_and_advantages(last_value=0.0)

    assert buf._advantages is not None
    # A_0 = r0 + r1, A_1 = r1
    assert float(buf._advantages[0].item()) == pytest.approx(r0 + r1, abs=1e-5)
    assert float(buf._advantages[1].item()) == pytest.approx(r1, abs=1e-5)


def test_compute_returns_changes_advantages():
    buf = RolloutBuffer(capacity=5, gamma=0.99, gae_lambda=0.95)
    for _ in range(3):
        buf.add(obs={}, action=0, reward=1.0, done=False, value=0.5, log_prob=-0.5)
    buf.compute_returns_and_advantages(last_value=0.0)
    assert buf._advantages is not None
    assert buf._returns is not None
    assert len(buf._advantages) == 3


def test_get_batches_shape():
    buf = RolloutBuffer(capacity=10, gamma=0.99, gae_lambda=0.95)
    for _ in range(8):
        buf.add(obs={}, action=0, reward=1.0, done=False, value=1.0, log_prob=-0.5)
    buf.compute_returns_and_advantages(last_value=0.0)

    batches = list(buf.get_batches(mini_batch_size=4))
    assert len(batches) == 2
    for b in batches:
        assert "advantages" in b
        assert "returns" in b
        assert "actions" in b


def test_clear_resets_buffer():
    buf = RolloutBuffer(capacity=10, gamma=0.99, gae_lambda=0.95)
    buf.add(obs={}, action=0, reward=1.0, done=False, value=1.0, log_prob=0.0)
    buf.clear()
    assert len(buf) == 0
    assert buf._advantages is None


def test_mini_batch_larger_than_rollout_yields_one_batch():
    buf = RolloutBuffer(capacity=10, gamma=0.99, gae_lambda=0.95)
    for _ in range(3):
        buf.add(obs={}, action=1, reward=2.0, done=False, value=0.5, log_prob=-1.0)
    buf.compute_returns_and_advantages(last_value=0.0)

    batches = list(buf.get_batches(mini_batch_size=100))
    assert len(batches) == 1
    assert len(batches[0]["actions"]) == 3


def test_rollout_buffer_length():
    buf = RolloutBuffer(capacity=5)
    for i in range(4):
        buf.add(obs={}, action=i, reward=float(i), done=False, value=0.0, log_prob=0.0)
    assert len(buf) == 4


def test_gae_discounting():
    """Verify gamma affects returns: higher gamma -> higher return."""
    def compute_return(gamma):
        buf = RolloutBuffer(capacity=5, gamma=gamma, gae_lambda=1.0)
        buf.add(obs={}, action=0, reward=1.0, done=False, value=0.0, log_prob=0.0)
        buf.add(obs={}, action=0, reward=1.0, done=True, value=0.0, log_prob=0.0)
        buf.compute_returns_and_advantages(last_value=0.0)
        return float(buf._returns[0].item())

    r_high = compute_return(0.99)
    r_low = compute_return(0.5)
    assert r_high > r_low
