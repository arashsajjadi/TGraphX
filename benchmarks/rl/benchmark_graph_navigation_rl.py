"""Benchmark graph navigation RL: random policy and REINFORCE.

Usage:
    python benchmarks/rl/benchmark_graph_navigation_rl.py
    python benchmarks/rl/benchmark_graph_navigation_rl.py --small --json
"""
import argparse
import json
import time

import torch

from tgraphx.rl.environments import GraphNavigationEnv, GraphEnvConfig
from tgraphx.rl.networks.policy import GraphPolicyNetwork
from tgraphx.rl.algorithms.reinforce import REINFORCEAgent


def run_random_episodes(env, n_episodes=20):
    total = 0.0
    for ep in range(n_episodes):
        env.reset()
        for _ in range(env.config.max_steps):
            mask = env.valid_action_mask()
            valid = mask.nonzero(as_tuple=False).squeeze(1)
            if len(valid) == 0:
                break
            action = int(valid[torch.randint(len(valid), (1,)).item()].item())
            _, _, done, _, _ = env.step(action)
            if done:
                break
        total += 1
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    n = 6
    n_eps = 10 if args.small else 30
    ei = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    nf = torch.randn(n, 4)
    config = GraphEnvConfig(max_steps=15, seed=42)
    env = GraphNavigationEnv(ei, n, node_features=nf, target_node=5, config=config)

    results = []

    # Random policy benchmark
    t0 = time.perf_counter()
    run_random_episodes(env, n_eps)
    rand_time = time.perf_counter() - t0
    eps_per_sec_rand = n_eps / rand_time if rand_time > 0 else 0

    results.append({
        "policy": "random",
        "n_episodes": n_eps,
        "time_s": rand_time,
        "episodes_per_sec": eps_per_sec_rand,
    })

    if not args.json:
        print(f"Random: {eps_per_sec_rand:.1f} ep/s")

    # REINFORCE benchmark
    policy = GraphPolicyNetwork(node_in_dim=4, hidden_dim=16, num_actions=5)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    agent = REINFORCEAgent(policy, opt)
    gen = torch.Generator()
    gen.manual_seed(42)

    t0 = time.perf_counter()
    for _ in range(n_eps):
        traj = agent.collect_episode(env, generator=gen, max_steps=15)
        agent.update(traj)
    reinforce_time = time.perf_counter() - t0
    eps_per_sec_rl = n_eps / reinforce_time if reinforce_time > 0 else 0

    results.append({
        "policy": "REINFORCE",
        "n_episodes": n_eps,
        "time_s": reinforce_time,
        "episodes_per_sec": eps_per_sec_rl,
    })

    if not args.json:
        print(f"REINFORCE: {eps_per_sec_rl:.1f} ep/s")

    if args.json:
        import sys, tgraphx
        device = "cuda" if torch.cuda.is_available() else "cpu"
        output = {
            "package_version": tgraphx.__version__,
            "benchmark": "graph_navigation_rl",
            "seed": 42,
            "device": device,
            "status": "ok",
            "limitations": "CPU-only small-scale; Experimental stability",
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "torch": torch.__version__,
            "results": results,
        }
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
