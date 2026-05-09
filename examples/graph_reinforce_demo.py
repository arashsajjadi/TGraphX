"""Train REINFORCE on GraphNavigationEnv and write RL report."""
import tempfile
import os
import torch

from tgraphx.rl.environments import GraphNavigationEnv, GraphEnvConfig
from tgraphx.rl.networks.policy import GraphPolicyNetwork
from tgraphx.rl.algorithms.reinforce import REINFORCEAgent
from tgraphx.rl.metrics import episodic_return_mean, episodic_return_std
from tgraphx.rl.reports import write_graph_rl_training_report


def main():
    print("=== REINFORCE on GraphNavigationEnv ===\n")

    torch.manual_seed(42)

    # Build graph: path 0-1-2-3-4
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    n = 5
    nf = torch.randn(n, 8)

    config = GraphEnvConfig(max_steps=15, seed=42)
    env = GraphNavigationEnv(ei, n, node_features=nf, target_node=4, config=config, start_node=0)

    policy = GraphPolicyNetwork(node_in_dim=8, hidden_dim=32, num_actions=4)
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)
    agent = REINFORCEAgent(policy, optimizer, gamma=0.99, entropy_coef=0.01)

    gen = torch.Generator()
    gen.manual_seed(0)

    n_episodes = 30
    episode_returns = []
    loss_curves = {"total_loss": [], "policy_loss": []}

    for ep in range(n_episodes):
        traj = agent.collect_episode(env, generator=gen, max_steps=15)
        losses = agent.update(traj)
        episode_returns.append(traj["total_return"])
        loss_curves["total_loss"].append(losses.get("total_loss", 0.0))
        loss_curves["policy_loss"].append(losses.get("policy_loss", 0.0))

        if (ep + 1) % 10 == 0:
            recent = episode_returns[-10:]
            print(f"  Ep {ep+1:3d}: mean_return={sum(recent)/len(recent):.2f}, "
                  f"loss={losses.get('total_loss', 0.0):.4f}")

    print(f"\nFinal mean return: {episodic_return_mean(episode_returns[-10:]):.2f} "
          f"± {episodic_return_std(episode_returns[-10:]):.2f}")

    out = os.path.join(tempfile.gettempdir(), "reinforce_report.json")
    write_graph_rl_training_report(
        out,
        algorithm="REINFORCE",
        loss_curves=loss_curves,
        return_curves=episode_returns,
        config={"gamma": 0.99, "entropy_coef": 0.01, "lr": 5e-3, "episodes": n_episodes},
    )
    print(f"RL training report written to: {out}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
