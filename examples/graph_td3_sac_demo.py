"""Demonstrate TD3 and SAC on ContinuousNavigationEnv.

Usage:
    python examples/graph_td3_sac_demo.py
"""
from __future__ import annotations

import copy
import torch
from tgraphx.rl import run_graph_rl
from tgraphx.rl.environments.continuous import ContinuousNavigationEnv
from tgraphx.rl.environments.base import GraphEnvConfig
from tgraphx.rl.algorithms.continuous import (
    ContinuousGraphActor, TwinContinuousGraphCritic,
    GraphTD3Agent, soft_update,
)
from tgraphx.rl.algorithms.replay_buffer import ReplayBuffer


def demo_via_high_level_api():
    print("TD3 and SAC via high-level API")
    print("=" * 40)

    for algo in ("td3", "sac"):
        result = run_graph_rl(
            env="continuous_navigation",
            algorithm=algo,
            episodes=10,
            seed=42,
            verbose=False,
        )
        print(f"  {algo.upper()}: mean_return={result.metrics['mean_return']:.2f}")

    print()


def demo_td3_direct():
    print("Direct TD3 usage")
    print("=" * 40)

    node_in = 8
    action_dim = 8
    hidden_dim = 32
    n_nodes = 6

    # Create env
    src = list(range(n_nodes - 1))
    dst = list(range(1, n_nodes))
    ei = torch.tensor([src + dst, dst + src], dtype=torch.long)
    nf = torch.randn(n_nodes, node_in)
    env = ContinuousNavigationEnv(
        edge_index=ei, num_nodes=n_nodes, node_features=nf,
        action_dim=action_dim, target_node=n_nodes - 1,
        config=GraphEnvConfig(max_steps=15),
    )

    # Build networks
    actor = ContinuousGraphActor(node_in, 0, hidden_dim, action_dim)
    target_actor = copy.deepcopy(actor)
    twin = TwinContinuousGraphCritic(node_in, 0, action_dim, hidden_dim)
    target_twin = copy.deepcopy(twin)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(twin.parameters(), lr=1e-3)
    buf = ReplayBuffer(1000)

    agent = GraphTD3Agent(
        actor, twin, target_actor, target_twin,
        actor_opt, critic_opt,
        gamma=0.99, tau=0.005, policy_delay=2,
        replay_buffer=buf, batch_size=8,
    )

    gen = torch.Generator().manual_seed(42)
    total_return = 0.0
    global_step = 0

    for ep in range(10):
        obs = env.reset(seed=ep)
        ep_return = 0.0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(obs, generator=gen)
            next_obs, reward, done, truncated, info = env.step(action)
            buf.push(obs, action.clone(), reward, next_obs, done or truncated)
            if buf.is_ready(8):
                agent.update(step=global_step)
            obs = next_obs
            ep_return += reward
            global_step += 1

        total_return += ep_return

    print(f"  TD3: mean_return over 10 episodes = {total_return / 10:.2f}")
    print()


if __name__ == "__main__":
    demo_via_high_level_api()
    demo_td3_direct()
    print("TD3/SAC demo complete.")
