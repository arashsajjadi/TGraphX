# Graph RL Algorithms

**Status: Experimental (v0.7.0+)**

## REINFORCE

Policy gradient with Monte Carlo returns:

    ∇J ≈ Σ_t ∇log π_θ(a_t|s_t) * G_t
    G_t = Σ_{k≥t} γ^{k-t} r_k

Entropy bonus: `-entropy_coef * H(π)` where `H(π) = -Σ_a π(a|s) log π(a|s)`.

```python
from tgraphx.rl.algorithms.reinforce import REINFORCEAgent

agent = REINFORCEAgent(policy, optimizer, gamma=0.99, entropy_coef=0.01)
traj = agent.collect_episode(env, max_steps=50)
losses = agent.update(traj)
```

## A2C (Advantage Actor-Critic)

Generalized Advantage Estimation (GAE):

    A_t = Σ_{l=0}^T (γλ)^l δ_{t+l}
    δ_t = r_t + γ V(s_{t+1})(1-done_t) - V(s_t)

Policy loss: `L_pi = -mean(log π(a_t|s_t) * A_t)`
Value loss: `L_V = (V(s_t) - R_t)^2`

## DQN

Q-function Bellman target:

    y = r + γ * max_{a'} Q_target(s', a')  (masked by valid actions)

Double-DQN separates action selection and evaluation:

    a* = argmax_a Q_online(s', a)
    y  = r + γ * Q_target(s', a*)

## PPO

Clipped surrogate objective:

    ρ_t = π_θ(a_t|s_t) / π_old(a_t|s_t)
    L_clip = -mean(min(ρ_t A_t, clip(ρ_t, 1-ε, 1+ε) A_t))

Total loss: `L_clip + value_coef * L_V - entropy_coef * H(π)`

Tracked: approximate KL = mean(old_logprob - new_logprob), clip fraction.

## Code Examples

```python
# DQN
from tgraphx.rl.algorithms.dqn import DQNAgent
from tgraphx.rl.algorithms.replay_buffer import ReplayBuffer
from tgraphx.rl.networks.qnetwork import GraphQNetwork

net = GraphQNetwork(node_in_dim=4, hidden_dim=64, num_actions=3)
target = GraphQNetwork(node_in_dim=4, hidden_dim=64, num_actions=3)
buf = ReplayBuffer(capacity=10000)
agent = DQNAgent(net, target, optimizer, replay_buffer=buf)

# PPO
from tgraphx.rl.algorithms.ppo import PPOAgent

agent = PPOAgent(actor_critic, optimizer, clip_eps=0.2, n_epochs=4)
rollout = agent.collect_rollout(env, n_steps=64)
losses = agent.update(rollout)
```
