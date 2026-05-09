"""Easy graph RL — minimal imports.

This example runs a graph RL workflow using the TGraphX high-level API.

Usage::

    python examples/easy_graph_rl_no_torch.py
"""

import tgraphx as tgx

# Discovery: what algorithms are available?
algos = tgx.list_graph_rl_algorithms()
print("Available graph RL algorithms:")
for name, desc in algos.items():
    print(f"  {name}: {desc}")

# Run a short RL episode.
result = tgx.run_graph_rl(
    algorithm="reinforce",
    env="graph_navigation",
    episodes=3,
    seed=42,
)
print(f"\nRL result metrics: {result.metrics}")
print(f"\nGraph RL PASSED")
