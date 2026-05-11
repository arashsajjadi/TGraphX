# Graph Reinforcement Learning

**Status: Experimental (v0.7.0+)**

## MDP Formulation

Graph RL environments share a common MDP structure:

- **State**: Graph structure (edge_index, node_features) + task-specific context
- **Action**: Discrete integer (masked by `action_mask`)
- **Reward**: Task-specific scalar
- **Done**: Task completed or max_steps reached

## Environments

| Environment | Task | Action | Reward |
|-------------|------|--------|--------|
| `GraphNavigationEnv` | Navigate to target node | Choose neighbor edge | +10 on reach, -0.1/step |
| `GraphColoringEnv` | Color graph with k colors | Assign color to next node | -conflicts, +completion bonus |
| `MaxCutEnv` / `GraphMaxCutEnv` | Maximize cut value | Assign partition 0/1 | Delta cut value |
| `VertexCoverEnv` | Find minimum vertex cover | Select node for cover | -1/node + coverage bonus |
| `GraphGenerationEnv` | Build a graph | ADD_NODE/ADD_EDGE/STOP | Validity + target score |
| `KGPathReasoningEnv` | KG path reasoning | Choose outgoing relation | +1 on target, -0.05/step |

## Quick start (high-level API)

```python
from tgraphx.rl import run_graph_rl, GraphMaxCutEnv

env = GraphMaxCutEnv(num_nodes=40, edge_density=0.1, seed=42)
result = run_graph_rl(algorithm="ppo", env=env, episodes=30, seed=42)
print("Final reward:", result.final_reward)
print("Mean return:", result.mean_return)
```

`GraphMaxCutEnv(num_nodes=..., edge_density=..., seed=...)` is a thin
LLM-friendly wrapper around `MaxCutEnv(edge_index, num_nodes, ...)` that
builds the underlying Erdos-Renyi graph for you. The canonical
`MaxCutEnv(edge_index, num_nodes, ...)` form remains supported and is
preferred when you already have an `edge_index`.

List supported algorithms with `tgraphx.rl.list_graph_rl_algorithms()`.

## Quick code example (low-level API)

```python
from tgraphx.rl.environments import GraphNavigationEnv, GraphEnvConfig
from tgraphx.rl.networks.policy import GraphPolicyNetwork
from tgraphx.rl.algorithms.reinforce import REINFORCEAgent
import torch

# Build graph
ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
nf = torch.randn(5, 8)
config = GraphEnvConfig(max_steps=20, seed=42)
env = GraphNavigationEnv(ei, 5, node_features=nf, target_node=4, config=config)

# Create agent
policy = GraphPolicyNetwork(node_in_dim=8, hidden_dim=32, num_actions=4)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
agent = REINFORCEAgent(policy, optimizer)

# Train
for ep in range(50):
    traj = agent.collect_episode(env, max_steps=20)
    agent.update(traj)
```

## Limitations

- These environments and algorithms are small-scale research tools.
- For production RL: use dedicated libraries (stable-baselines3, RLlib).
- All environments run on single-machine CPU only.
- No parallelization across environments.

See also: [graph_rl_algorithms.md](graph_rl_algorithms.md)
