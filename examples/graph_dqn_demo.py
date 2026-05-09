"""Train DQN on GraphColoringEnv and write RL report."""
import tempfile
import os
import torch

from tgraphx.rl.environments import GraphColoringEnv, GraphEnvConfig
from tgraphx.rl.networks.qnetwork import GraphQNetwork
from tgraphx.rl.algorithms.dqn import DQNAgent
from tgraphx.rl.algorithms.replay_buffer import ReplayBuffer
from tgraphx.rl.exploration.strategies import EpsilonGreedy
from tgraphx.rl.metrics import episodic_return_mean
from tgraphx.rl.reports import write_graph_rl_training_report


def run_episode(env, agent, gen, step_count):
    obs = env.reset()
    total_reward = 0.0
    for _ in range(env.config.max_steps):
        action = agent.select_action(obs, step=step_count, generator=gen)
        next_obs, reward, done, _, _ = env.step(action)
        agent.buffer.push(obs, action, reward, next_obs, done)
        losses = agent.update()
        obs = next_obs
        total_reward += reward
        step_count += 1
        if done:
            break
    return total_reward, step_count


def main():
    print("=== DQN on GraphColoringEnv ===\n")

    torch.manual_seed(42)

    # Triangle graph
    tri_ei = torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype=torch.long)
    n = 3
    nf = torch.randn(n, 4)
    config = GraphEnvConfig(max_steps=10, seed=42)
    env = GraphColoringEnv(tri_ei, n, node_features=nf, num_colors=3, config=config)

    node_in_dim = 4
    num_actions = 3  # 3 colors
    q_net = GraphQNetwork(node_in_dim=node_in_dim, hidden_dim=32, num_actions=num_actions)
    target_net = GraphQNetwork(node_in_dim=node_in_dim, hidden_dim=32, num_actions=num_actions)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    buf = ReplayBuffer(capacity=1000)

    agent = DQNAgent(
        q_net, target_net, optimizer,
        gamma=0.99,
        eps_start=1.0, eps_end=0.05, eps_decay=50,
        target_update_freq=20,
        batch_size=8,
        replay_buffer=buf,
    )

    gen = torch.Generator()
    gen.manual_seed(0)

    n_episodes = 40
    episode_returns = []
    step_count = 0
    loss_curves = {"q_loss": []}

    for ep in range(n_episodes):
        ret, step_count = run_episode(env, agent, gen, step_count)
        episode_returns.append(ret)
        eps = agent.epsilon_schedule.get_epsilon(step_count)

        if (ep + 1) % 10 == 0:
            recent = episode_returns[-10:]
            print(f"  Ep {ep+1:3d}: mean_return={sum(recent)/len(recent):.2f}, "
                  f"epsilon={eps:.3f}, buffer={len(buf)}")

    print(f"\nFinal mean return: {episodic_return_mean(episode_returns[-10:]):.2f}")

    out = os.path.join(tempfile.gettempdir(), "dqn_report.json")
    write_graph_rl_training_report(
        out,
        algorithm="DQN",
        loss_curves={"q_loss": []},
        return_curves=episode_returns,
        config={"gamma": 0.99, "eps_start": 1.0, "eps_end": 0.05, "episodes": n_episodes},
    )
    print(f"RL training report written to: {out}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
