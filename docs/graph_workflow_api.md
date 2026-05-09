# Graph Workflow API Reference

**Stability: Beta (v0.7.0+)**

The high-level API provides one-line functions for end-to-end graph RL, generation, and evolutionary optimization.

## Graph RL

### `run_graph_rl`

```python
from tgraphx.rl import run_graph_rl

result = run_graph_rl(
    env="graph_navigation",       # env name or GraphEnv instance
    algorithm="dqn",              # from list_graph_rl_algorithms()
    episodes=50,
    seed=42,
    device="cpu",
    hidden_dim=64,
    gamma=0.99,
    lr=1e-3,
    dashboard_dir=None,           # if set, writes JSON report here
    verbose=False,
)

print(result.metrics["mean_return"])
print(result.metrics["success_rate"])
```

### `list_graph_rl_algorithms`

```python
from tgraphx.rl import list_graph_rl_algorithms
algs = list_graph_rl_algorithms()
# {'dqn': {'action_type': 'discrete', 'stability': 'Experimental', ...}, ...}
```

### `make_graph_env`

```python
from tgraphx.rl import make_graph_env
env = make_graph_env("graph_navigation", num_nodes=10, seed=42)
```

Valid env names: `graph_navigation`, `shortest_path`, `graph_coloring`, `max_cut`, `vertex_cover`, `graph_generation`, `kg_reasoning`, `continuous_navigation`, `continuous_graph_edit`.

## Graph Generation

### `run_graph_generation`

```python
from tgraphx.generation import run_graph_generation

result = run_graph_generation(
    method="barabasi_albert",
    num_graphs=16,
    num_nodes=50,
    m=2,
    node_feature_dim=8,
    seed=42,
)
print(f"Validity: {result.metrics['validity']:.2f}")
print(f"Uniqueness: {result.metrics['uniqueness']:.2f}")
```

### `list_graph_generation_methods`

```python
from tgraphx.generation import list_graph_generation_methods
methods = list_graph_generation_methods()
```

Valid methods: `erdos_renyi`, `barabasi_albert`, `watts_strogatz`, `stochastic_block_model`, `grid`, `cycle`, `path`, `star`, `complete`, `motif_injected`, `anomaly_injected`, `temporal`, `typed`, `random_geometric`.

## Evolutionary Optimization

### `run_evolutionary_optimization`

```python
from tgraphx.evolutionary import run_evolutionary_optimization

result = run_evolutionary_optimization(
    algorithm="ga",
    objective="connectivity",     # or ['connectivity', 'density'] for nsga2
    population_size=20,
    generations=10,
    num_nodes=10,
    seed=42,
)
print(f"Best fitness: {result.best_fitness:.4f}")
```

### `list_evolutionary_optimizers`

```python
from tgraphx.evolutionary import list_evolutionary_optimizers
opts = list_evolutionary_optimizers()
```

Valid algorithms: `ga`, `sa`, `nsga2`, `hill_climbing`, `random_search`.

## Error Handling

Invalid algorithm/method/objective raises `ValueError` with a helpful message:

```
ValueError: Unknown algorithm 'xyz'. Choose from: ['a2c', 'actor_critic', ...]
```
