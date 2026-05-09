# Graph RL: Continuous Action Embeddings

**Stability: Experimental (v0.7.0+)**

## Overview

Discrete-action algorithms (DQN, PPO) select integer node/edge indices. Continuous-action algorithms (DDPG, TD3, SAC) instead output a float vector in `[-1, 1]^action_dim` that a decoder maps to graph edits.

## ContinuousGraphActionSpace

```python
from tgraphx.rl.environments.continuous import ContinuousGraphActionSpace

space = ContinuousGraphActionSpace(action_dim=8, action_low=-1.0, action_high=1.0)
action = space.sample()   # Tensor [8]
clipped = space.clip(action)
```

## Algorithms

| Algorithm | Description | Key math |
|-----------|-------------|----------|
| DDPG | Deterministic policy gradient | Actor: `a = mu(s)`, Critic: `Q(s, a)` |
| TD3 | Twin delayed DDPG | Twin critics + target policy smoothing |
| SAC | Soft actor-critic | Entropy-regularized, stochastic actor |

## DDPG

```python
Actor: a_t = mu_theta(s_t)
Critic: Q_phi(s_t, a_t)
Target: y = r + gamma * Q_phi'(s', mu_theta'(s'))
Soft update: theta' <- tau*theta + (1-tau)*theta'
```

## TD3 Extensions

- **Twin critics**: `Q1(s,a)`, `Q2(s,a)` — target uses `min(Q1', Q2')`
- **Target policy smoothing**: `a' = clip(mu'(s') + clip(eps, -c, c), low, high)`
- **Delayed actor update**: actor updated every `policy_delay` critic steps

## SAC

```python
# Stochastic actor with reparameterization
action, log_prob = actor(state)
# Log-prob with tanh correction
log_prob -= sum(log(1 - tanh(x)^2 + 1e-6), dim=-1)

# Entropy-regularized target
y = r + gamma * (min_Q(s', a') - alpha * log_pi(a'|s'))

# Auto-tune alpha
alpha_loss = -log_alpha * (log_pi + target_entropy).detach()
```

## Usage

```python
from tgraphx.rl import run_graph_rl

result = run_graph_rl(
    env="continuous_navigation",
    algorithm="sac",
    episodes=50,
    seed=42,
)
print(f"Mean return: {result.metrics['mean_return']:.2f}")
```

## Action Embedding Decoder

In `ContinuousNavigationEnv`, the action vector is decoded to neighbor selection via cosine similarity:
1. Project action vector to node embedding space.
2. Rank neighbors by cosine similarity.
3. Select highest-similarity neighbor.

In `ContinuousGraphEditEnv`, the action vector is split:
- `sigmoid(action[0])` -> probability to add a node
- `sigmoid(action[1])` -> probability to add an edge
- `sigmoid(action[2])` -> probability to remove an edge
- `action[3:]` -> feature delta
