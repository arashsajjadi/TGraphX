"""Demonstrate PPO on GraphNavigationEnv.

Usage:
    python examples/graph_ppo_demo.py
"""
from __future__ import annotations

import torch
from tgraphx.rl import run_graph_rl, GraphNavigationEnv
from tgraphx.rl.environments.base import GraphEnvConfig


def main():
    print("PPO on GraphNavigationEnv")
    print("=" * 40)

    # Run via high-level API
    result = run_graph_rl(
        env="graph_navigation",
        algorithm="ppo",
        episodes=15,
        seed=42,
        gamma=0.99,
        lr=3e-4,
        verbose=True,
    )

    print(f"\nFinal metrics:")
    print(f"  Mean return: {result.metrics['mean_return']:.2f}")
    print(f"  Success rate: {result.metrics['success_rate']:.2%}")

    # Also show direct usage with PPO
    print("\nDirect PPO usage:")
    from tgraphx.rl.networks.actor_critic import GraphActorCriticNetwork
    from tgraphx.rl.algorithms.ppo import PPOAgent

    # Create a small navigation env
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4], [1, 0, 3, 2], [2, 1, 4, 3]], dtype=torch.long)
    ei = torch.tensor([[0, 1, 2, 3, 1, 2, 3, 4], [1, 2, 3, 4, 0, 1, 2, 3]], dtype=torch.long)
    n = 5
    nf = torch.randn(n, 8)
    env = GraphNavigationEnv(
        edge_index=ei, num_nodes=n, node_features=nf,
        target_node=4,
        config=GraphEnvConfig(max_steps=20),
    )

    net = GraphActorCriticNetwork(node_in_dim=8, edge_in_dim=0, hidden_dim=32, num_actions=4)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    agent = PPOAgent(actor_critic=net, optimizer=optimizer, gamma=0.99)

    gen = torch.Generator().manual_seed(0)
    total_return = 0.0

    for ep in range(5):
        rollout = agent.collect_rollout(env, n_steps=20, generator=gen)
        ep_return = sum(rollout["rewards"])
        total_return += ep_return
        agent.update(rollout)

    print(f"  Mean return over 5 episodes: {total_return / 5:.2f}")
    print("\nPPO demo complete.")


if __name__ == "__main__":
    main()
