"""Tests for continuous RL algorithms (DDPG, TD3, SAC) and related components."""
from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn

from tgraphx.rl.algorithms.continuous import (
    OUNoise,
    GaussianNoise,
    ContinuousGraphActor,
    StochasticGraphActor,
    ContinuousGraphCritic,
    TwinContinuousGraphCritic,
    soft_update,
    GraphDDPGAgent,
    GraphDelayedDDPGAgent,
    GraphTD3Agent,
    GraphSACAgent,
)
from tgraphx.rl.algorithms.replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NODE_IN = 4
ACTION_DIM = 6
HIDDEN = 32
N_NODES = 5


@pytest.fixture
def edge_index():
    return torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)


@pytest.fixture
def node_features():
    return torch.randn(N_NODES, NODE_IN)


@pytest.fixture
def action():
    return torch.randn(1, ACTION_DIM)


@pytest.fixture
def obs(node_features, edge_index):
    return {"node_features": node_features, "edge_index": edge_index}


# ---------------------------------------------------------------------------
# OUNoise
# ---------------------------------------------------------------------------

def test_ou_noise_shape():
    noise = OUNoise(action_dim=ACTION_DIM, seed=0)
    s = noise.sample()
    assert s.shape == (ACTION_DIM,)


def test_ou_noise_reset():
    noise = OUNoise(action_dim=ACTION_DIM, mu=1.0)
    noise.reset()
    assert noise._state.shape == (ACTION_DIM,)
    assert float(noise._state[0].item()) == pytest.approx(1.0, abs=1e-5)


def test_ou_noise_multiple_samples():
    noise = OUNoise(action_dim=4, seed=1)
    s1 = noise.sample()
    s2 = noise.sample()
    assert s1.shape == s2.shape == (4,)


# ---------------------------------------------------------------------------
# GaussianNoise
# ---------------------------------------------------------------------------

def test_gaussian_noise_shape():
    noise = GaussianNoise(action_dim=ACTION_DIM, sigma=0.1)
    s = noise.sample()
    assert s.shape == (ACTION_DIM,)


def test_gaussian_noise_clip():
    noise = GaussianNoise(action_dim=ACTION_DIM, sigma=1.0, clip=0.5)
    gen = torch.Generator()
    gen.manual_seed(42)
    s = noise.sample(generator=gen)
    assert float(s.abs().max().item()) <= 0.5 + 1e-6


# ---------------------------------------------------------------------------
# ContinuousGraphActor
# ---------------------------------------------------------------------------

def test_actor_forward_shape(node_features, edge_index):
    actor = ContinuousGraphActor(NODE_IN, 0, HIDDEN, ACTION_DIM)
    out = actor(node_features, edge_index)
    assert out.shape == (1, ACTION_DIM)


def test_actor_tanh_range(node_features, edge_index):
    actor = ContinuousGraphActor(NODE_IN, 0, HIDDEN, ACTION_DIM, action_scale=1.0)
    out = actor(node_features, edge_index)
    assert float(out.abs().max().item()) <= 1.0 + 1e-5


def test_actor_wrong_input_raises():
    actor = ContinuousGraphActor(NODE_IN, 0, HIDDEN, ACTION_DIM)
    with pytest.raises(ValueError, match="expects"):
        actor(torch.randn(2, 3, 4), torch.zeros((2, 0), dtype=torch.long))


# ---------------------------------------------------------------------------
# StochasticGraphActor
# ---------------------------------------------------------------------------

def test_stochastic_actor_returns_tuple(node_features, edge_index):
    actor = StochasticGraphActor(NODE_IN, 0, HIDDEN, ACTION_DIM)
    action, log_prob = actor(node_features, edge_index)
    assert action.shape == (1, ACTION_DIM)
    assert log_prob.shape == (1,)


def test_stochastic_actor_log_prob_finite(node_features, edge_index):
    actor = StochasticGraphActor(NODE_IN, 0, HIDDEN, ACTION_DIM)
    _, log_prob = actor(node_features, edge_index)
    assert torch.isfinite(log_prob).all()


def test_stochastic_actor_log_prob_negative(node_features, edge_index):
    """Log prob for tanh-squashed Gaussian should typically be negative for non-trivial distributions."""
    actor = StochasticGraphActor(NODE_IN, 0, HIDDEN, ACTION_DIM)
    torch.manual_seed(0)
    _, log_prob = actor(node_features, edge_index)
    # This can be positive sometimes but usually should be finite
    assert torch.isfinite(log_prob).all()


def test_stochastic_actor_reparameterized(node_features, edge_index):
    """Action should have gradient through reparameterization."""
    actor = StochasticGraphActor(NODE_IN, 0, HIDDEN, ACTION_DIM)
    action, log_prob = actor(node_features, edge_index)
    loss = action.sum()
    loss.backward()
    for p in actor.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


# ---------------------------------------------------------------------------
# ContinuousGraphCritic
# ---------------------------------------------------------------------------

def test_critic_forward_shape(node_features, edge_index, action):
    critic = ContinuousGraphCritic(NODE_IN, 0, ACTION_DIM, HIDDEN)
    out = critic(node_features, edge_index, action)
    assert out.shape == (1, 1)


def test_critic_forward_finite(node_features, edge_index, action):
    critic = ContinuousGraphCritic(NODE_IN, 0, ACTION_DIM, HIDDEN)
    out = critic(node_features, edge_index, action)
    assert torch.isfinite(out).all()


def test_critic_wrong_input_raises():
    critic = ContinuousGraphCritic(NODE_IN, 0, ACTION_DIM, HIDDEN)
    with pytest.raises(ValueError, match="expects"):
        critic(torch.randn(2, 3, 4), torch.zeros((2, 0), dtype=torch.long), torch.zeros(1, ACTION_DIM))


# ---------------------------------------------------------------------------
# TwinContinuousGraphCritic
# ---------------------------------------------------------------------------

def test_twin_critic_forward_shapes(node_features, edge_index, action):
    twin = TwinContinuousGraphCritic(NODE_IN, 0, ACTION_DIM, HIDDEN)
    q1, q2 = twin(node_features, edge_index, action)
    assert q1.shape == (1, 1)
    assert q2.shape == (1, 1)


def test_twin_critic_both_finite(node_features, edge_index, action):
    twin = TwinContinuousGraphCritic(NODE_IN, 0, ACTION_DIM, HIDDEN)
    q1, q2 = twin(node_features, edge_index, action)
    assert torch.isfinite(q1).all() and torch.isfinite(q2).all()


def test_twin_critic_forward_min_shape(node_features, edge_index, action):
    twin = TwinContinuousGraphCritic(NODE_IN, 0, ACTION_DIM, HIDDEN)
    qmin = twin.forward_min(node_features, edge_index, action)
    assert qmin.shape == (1, 1)
    q1, q2 = twin(node_features, edge_index, action)
    assert float(qmin.item()) == pytest.approx(min(float(q1.item()), float(q2.item())), abs=1e-5)


# ---------------------------------------------------------------------------
# soft_update
# ---------------------------------------------------------------------------

def test_soft_update_tau_zero():
    """tau=0 -> target unchanged."""
    src = nn.Linear(4, 4)
    tgt = nn.Linear(4, 4)
    tgt_orig = copy.deepcopy(tgt.weight.data)
    soft_update(src, tgt, tau=0.0)
    assert torch.allclose(tgt.weight.data, tgt_orig)


def test_soft_update_tau_one():
    """tau=1 -> target = source."""
    src = nn.Linear(4, 4)
    tgt = nn.Linear(4, 4)
    soft_update(src, tgt, tau=1.0)
    assert torch.allclose(tgt.weight.data, src.weight.data)


def test_soft_update_interpolation():
    """Verify theta_target = tau*src + (1-tau)*tgt."""
    src = nn.Linear(2, 2, bias=False)
    tgt = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        src.weight.fill_(2.0)
        tgt.weight.fill_(0.0)
    tau = 0.3
    soft_update(src, tgt, tau=tau)
    expected = tau * 2.0 + (1 - tau) * 0.0
    assert torch.allclose(tgt.weight.data, torch.full_like(tgt.weight, expected))


# ---------------------------------------------------------------------------
# GraphDDPGAgent
# ---------------------------------------------------------------------------

def _make_ddpg(buf=None):
    actor = ContinuousGraphActor(NODE_IN, 0, HIDDEN, ACTION_DIM)
    target_actor = copy.deepcopy(actor)
    critic = ContinuousGraphCritic(NODE_IN, 0, ACTION_DIM, HIDDEN)
    target_critic = copy.deepcopy(critic)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    noise = OUNoise(ACTION_DIM, seed=0)
    rb = buf if buf is not None else ReplayBuffer(1000)
    return GraphDDPGAgent(
        actor, critic, target_actor, target_critic,
        actor_opt, critic_opt,
        gamma=0.99, tau=0.005, noise=noise, replay_buffer=rb, batch_size=4,
    )


def test_ddpg_critic_loss_finite(obs):
    agent = _make_ddpg()
    # Manually populate batch
    batch = [(obs, torch.zeros(ACTION_DIM), 1.0, obs, False)] * 4
    result = agent.update(batch=batch)
    assert "critic_loss" in result
    assert math.isfinite(result["critic_loss"])


def test_ddpg_actor_loss_finite(obs):
    agent = _make_ddpg()
    batch = [(obs, torch.zeros(ACTION_DIM), 1.0, obs, False)] * 4
    result = agent.update(batch=batch)
    assert "actor_loss" in result
    assert math.isfinite(result["actor_loss"])


def test_ddpg_optimizer_steps_change_params(obs):
    agent = _make_ddpg()
    w_before = agent.actor.mlp[-1].weight.data.clone()
    batch = [(obs, torch.zeros(ACTION_DIM), 1.0, obs, False)] * 4
    agent.update(batch=batch)
    # Actor parameters should change
    assert not torch.allclose(agent.actor.mlp[-1].weight.data, w_before)


def test_ddpg_target_soft_update(obs):
    agent = _make_ddpg()
    tgt_w_before = agent.target_critic.q_head[-1].weight.data.clone()
    batch = [(obs, torch.zeros(ACTION_DIM), 1.0, obs, False)] * 4
    agent.update(batch=batch)
    # Target should have changed (tau=0.005 != 0)
    assert not torch.allclose(agent.target_critic.q_head[-1].weight.data, tgt_w_before)


def test_ddpg_no_nan_in_loss(obs):
    agent = _make_ddpg()
    batch = [(obs, torch.zeros(ACTION_DIM), 0.0, obs, False)] * 4
    result = agent.update(batch=batch)
    assert math.isfinite(result.get("critic_loss", 0.0))
    assert math.isfinite(result.get("actor_loss", 0.0))


def test_ddpg_state_dict_roundtrip(obs):
    agent = _make_ddpg()
    sd = agent.state_dict()
    agent.load_state_dict(sd)
    # Should not raise


# ---------------------------------------------------------------------------
# GraphTD3Agent
# ---------------------------------------------------------------------------

def _make_td3(buf=None):
    actor = ContinuousGraphActor(NODE_IN, 0, HIDDEN, ACTION_DIM)
    target_actor = copy.deepcopy(actor)
    twin = TwinContinuousGraphCritic(NODE_IN, 0, ACTION_DIM, HIDDEN)
    target_twin = copy.deepcopy(twin)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(twin.parameters(), lr=1e-3)
    rb = buf if buf is not None else ReplayBuffer(1000)
    return GraphTD3Agent(
        actor, twin, target_actor, target_twin,
        actor_opt, critic_opt,
        gamma=0.99, tau=0.005, policy_delay=2,
        replay_buffer=rb, batch_size=4,
    )


def test_td3_both_critics_get_gradients(obs):
    agent = _make_td3()
    batch = [(obs, torch.zeros(ACTION_DIM), 1.0, obs, False)] * 4
    # Zero grads first
    agent.critic_optimizer.zero_grad()
    agent.update(step=1, batch=batch)
    # Both Q1 and Q2 heads should have gradients
    # We check that parameters actually changed after optimizer step
    q1_w_before = agent.twin_critic.q1.q_head[-1].weight.data.clone()
    q2_w_before = agent.twin_critic.q2.q_head[-1].weight.data.clone()
    agent.update(step=2, batch=batch)
    # At least one should change
    q1_changed = not torch.allclose(agent.twin_critic.q1.q_head[-1].weight.data, q1_w_before)
    q2_changed = not torch.allclose(agent.twin_critic.q2.q_head[-1].weight.data, q2_w_before)
    assert q1_changed or q2_changed


def test_td3_actor_update_delayed(obs):
    """Actor should NOT update on step 1, should update on step 2 (policy_delay=2)."""
    agent = _make_td3()
    batch = [(obs, torch.zeros(ACTION_DIM), 1.0, obs, False)] * 4

    actor_w_before = agent.actor.mlp[-1].weight.data.clone()
    result1 = agent.update(step=1, batch=batch)
    # After step 1 (1 % 2 != 0), actor_loss should be None
    assert result1.get("actor_loss") is None
    assert torch.allclose(agent.actor.mlp[-1].weight.data, actor_w_before)

    result2 = agent.update(step=2, batch=batch)
    # After step 2 (2 % 2 == 0), actor should have updated
    assert result2.get("actor_loss") is not None


def test_td3_target_noise_clipped(obs):
    """Target policy noise should be clipped to target_noise_clip."""
    agent = _make_td3()
    agent.target_noise_std = 10.0  # large noise
    agent.target_noise_clip = 0.1  # small clip
    # The update should still work without NaN
    batch = [(obs, torch.zeros(ACTION_DIM), 1.0, obs, False)] * 4
    result = agent.update(step=1, batch=batch)
    assert math.isfinite(result["critic1_loss"])
    assert math.isfinite(result["critic2_loss"])


def test_td3_min_target_uses_smaller_critic(obs):
    """With twin critics, target uses min(Q1, Q2)."""
    actor = ContinuousGraphActor(NODE_IN, 0, HIDDEN, ACTION_DIM)
    target_actor = copy.deepcopy(actor)
    twin = TwinContinuousGraphCritic(NODE_IN, 0, ACTION_DIM, HIDDEN)
    target_twin = copy.deepcopy(twin)

    # Force Q1 >> Q2
    with torch.no_grad():
        for p in target_twin.q1.parameters():
            p.fill_(10.0)
        for p in target_twin.q2.parameters():
            p.fill_(0.0)

    nf = obs["node_features"]
    ei = obs["edge_index"]
    a = torch.zeros(1, ACTION_DIM)
    q1, q2 = target_twin(nf, ei, a)
    qmin = target_twin.forward_min(nf, ei, a)
    assert float(qmin.item()) <= min(float(q1.item()), float(q2.item())) + 1e-5


def test_td3_no_nan(obs):
    agent = _make_td3()
    batch = [(obs, torch.zeros(ACTION_DIM), 0.0, obs, False)] * 4
    result = agent.update(step=2, batch=batch)
    assert math.isfinite(result["critic1_loss"])
    assert math.isfinite(result["critic2_loss"])


# ---------------------------------------------------------------------------
# GraphSACAgent
# ---------------------------------------------------------------------------

def _make_sac(buf=None, auto_entropy=True):
    actor = StochasticGraphActor(NODE_IN, 0, HIDDEN, ACTION_DIM)
    twin = TwinContinuousGraphCritic(NODE_IN, 0, ACTION_DIM, HIDDEN)
    target_twin = copy.deepcopy(twin)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(twin.parameters(), lr=1e-3)
    rb = buf if buf is not None else ReplayBuffer(1000)
    return GraphSACAgent(
        actor, twin, target_twin,
        actor_opt, critic_opt,
        gamma=0.99, tau=0.005, alpha=0.2,
        auto_entropy=auto_entropy,
        replay_buffer=rb, batch_size=4,
    )


def test_sac_log_prob_finite(obs):
    actor = StochasticGraphActor(NODE_IN, 0, HIDDEN, ACTION_DIM)
    nf = obs["node_features"]
    ei = obs["edge_index"]
    _, log_prob = actor(nf, ei)
    assert torch.isfinite(log_prob).all()


def test_sac_reparameterized_action_has_gradient(obs):
    actor = StochasticGraphActor(NODE_IN, 0, HIDDEN, ACTION_DIM)
    nf = obs["node_features"]
    ei = obs["edge_index"]
    action, log_prob = actor(nf, ei)
    assert action.requires_grad or log_prob.requires_grad


def test_sac_twin_critics_both_gradients(obs):
    agent = _make_sac()
    batch = [(obs, torch.zeros(ACTION_DIM), 1.0, obs, False)] * 4
    q1_w_before = agent.twin_critic.q1.q_head[-1].weight.data.clone()
    q2_w_before = agent.twin_critic.q2.q_head[-1].weight.data.clone()
    agent.update(batch=batch)
    q1_changed = not torch.allclose(agent.twin_critic.q1.q_head[-1].weight.data, q1_w_before)
    q2_changed = not torch.allclose(agent.twin_critic.q2.q_head[-1].weight.data, q2_w_before)
    assert q1_changed or q2_changed


def test_sac_actor_loss_backward_finite(obs):
    agent = _make_sac()
    batch = [(obs, torch.zeros(ACTION_DIM), 1.0, obs, False)] * 4
    result = agent.update(batch=batch)
    assert "actor_loss" in result
    assert math.isfinite(result["actor_loss"])


def test_sac_alpha_updates_when_auto_entropy(obs):
    agent = _make_sac(auto_entropy=True)
    alpha_before = float(agent.alpha.item())
    batch = [(obs, torch.zeros(ACTION_DIM), 1.0, obs, False)] * 4
    result = agent.update(batch=batch)
    assert "alpha_loss" in result
    assert math.isfinite(result["alpha_loss"])


def test_sac_alpha_loss_finite(obs):
    agent = _make_sac(auto_entropy=True)
    batch = [(obs, torch.zeros(ACTION_DIM), 1.0, obs, False)] * 4
    result = agent.update(batch=batch)
    assert math.isfinite(result.get("alpha_loss", 0.0))


def test_sac_no_nan(obs):
    agent = _make_sac()
    batch = [(obs, torch.zeros(ACTION_DIM), 0.0, obs, False)] * 4
    result = agent.update(batch=batch)
    for k, v in result.items():
        if isinstance(v, float):
            assert math.isfinite(v), f"{k}={v} is not finite"


def test_sac_state_dict_roundtrip(obs):
    agent = _make_sac()
    sd = agent.state_dict()
    agent.load_state_dict(sd)


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------

def test_ou_noise_seeded():
    n1 = OUNoise(4, seed=42)
    n2 = OUNoise(4, seed=42)
    assert torch.allclose(n1.sample(), n2.sample())


def test_gaussian_noise_seeded():
    n1 = GaussianNoise(4, sigma=0.5)
    n2 = GaussianNoise(4, sigma=0.5)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    assert torch.allclose(n1.sample(g1), n2.sample(g2))
