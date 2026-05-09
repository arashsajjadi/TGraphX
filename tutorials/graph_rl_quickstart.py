"""Graph RL Quickstart — CPU runnable, deterministic, ~60 seconds.

This tutorial shows how to:
1. Compare random / greedy / DQN / PPO on a tiny graph navigation task.
2. Run TD3 (continuous) on the same environment via auto-routing.
3. Inspect reward and success metrics.
4. Write a dashboard artifact.

Stability: Experimental (v0.7.0+)
"""
import json
import os
import tempfile

from tgraphx.rl import run_graph_rl, list_graph_rl_algorithms

SEED = 42
EPISODES = 20
ENV = "graph_navigation"

# ---------------------------------------------------------------------------
# 1. List available algorithms
# ---------------------------------------------------------------------------
print("Available graph RL algorithms:")
for name, info in list_graph_rl_algorithms().items():
    print(f"  {name:12s}  [{info['stability']:12s}]  {info['action_type']:10s}  "
          f"{info['description']}")

print()

# ---------------------------------------------------------------------------
# 2. Compare discrete algorithms on graph navigation
# ---------------------------------------------------------------------------
discrete_algos = ["random", "greedy", "dqn", "ppo"]
print(f"Comparing {discrete_algos} on {ENV} ({EPISODES} episodes, seed={SEED})")
print(f"{'Algorithm':12s}  {'Mean Return':12s}  {'Success Rate':12s}")
print("-" * 40)

for algo in discrete_algos:
    result = run_graph_rl(
        env=ENV,
        algorithm=algo,
        episodes=EPISODES,
        seed=SEED,
        hidden_dim=32,
    )
    print(
        f"  {algo:10s}  {result.metrics['mean_return']:12.3f}  "
        f"{result.metrics['success_rate']:12.3f}"
    )

# ---------------------------------------------------------------------------
# 3. Continuous algorithms — auto-routed to continuous_navigation
# ---------------------------------------------------------------------------
print()
continuous_algos = ["td3", "sac"]
print(f"Continuous algorithms on {ENV} (auto-routed to continuous_navigation):")
print(f"{'Algorithm':12s}  {'Mean Return':12s}  {'Success Rate':12s}")
print("-" * 40)

for algo in continuous_algos:
    result = run_graph_rl(
        env=ENV,
        algorithm=algo,
        episodes=EPISODES,
        seed=SEED,
        hidden_dim=32,
    )
    print(
        f"  {algo:10s}  {result.metrics['mean_return']:12.3f}  "
        f"{result.metrics['success_rate']:12.3f}"
    )

# ---------------------------------------------------------------------------
# 4. Gradient metrics for a learning algorithm
# ---------------------------------------------------------------------------
print()
result_dqn = run_graph_rl(
    env=ENV,
    algorithm="dqn",
    episodes=30,
    seed=SEED,
    hidden_dim=32,
)
print("DQN metrics (30 episodes):")
for k, v in result_dqn.metrics.items():
    if k not in ("episode_returns",):
        print(f"  {k}: {v}")

# ---------------------------------------------------------------------------
# 5. Dashboard artifact
# ---------------------------------------------------------------------------
print()
with tempfile.TemporaryDirectory() as tmpdir:
    result_dash = run_graph_rl(
        env=ENV,
        algorithm="random",
        episodes=5,
        seed=SEED,
        dashboard_dir=tmpdir,
    )
    files = [f for f in os.listdir(tmpdir) if f.endswith(".json")]
    assert files, "No dashboard artifact written"
    artifact_path = os.path.join(tmpdir, files[0])
    with open(artifact_path) as fh:
        artifact = json.load(fh)
    assert "metrics" in artifact
    print(f"Dashboard artifact: {files[0]}")
    print(f"  algorithm: {artifact.get('config', {}).get('algorithm', artifact['metrics'].get('algorithm', 'random'))}")
    print(f"  mean_return: {artifact['metrics']['mean_return']:.3f}")

# ---------------------------------------------------------------------------
# 6. Error handling: invalid algorithm gives clear list
# ---------------------------------------------------------------------------
try:
    run_graph_rl(env=ENV, algorithm="bad_algo_xyz", episodes=1)
    assert False, "Expected ValueError"
except ValueError as e:
    assert "bad_algo_xyz" in str(e) and "Choose from" in str(e)
    print()
    print("Invalid algorithm error: OK")

# ---------------------------------------------------------------------------
# 7. Seed determinism
# ---------------------------------------------------------------------------
r_a = run_graph_rl(env=ENV, algorithm="random", episodes=5, seed=99)
r_b = run_graph_rl(env=ENV, algorithm="random", episodes=5, seed=99)
assert r_a.metrics["episode_returns"] == r_b.metrics["episode_returns"]
print("Seed determinism: OK")

print()
print("=== graph_rl_quickstart COMPLETE ===")
