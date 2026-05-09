# Tutorial Gallery

CPU-runnable quickstart tutorials for TGraphX v1.0.

All tutorials:
- run on CPU without a GPU,
- use a fixed random seed (deterministic output),
- finish in under 60 seconds on typical hardware,
- write a JSON dashboard artifact to a temporary directory,
- require no hidden downloads or private paths.

## Quickstart tutorials

| Tutorial | Capability | Runtime | GPU required | Output |
|----------|-----------|---------|:---:|--------|
| [graph_generation_quickstart.py](../tutorials/graph_generation_quickstart.py) | ER / BA / SBM generation, validity/uniqueness/diversity metrics, dashboard artifact | ~5 s | No | `GenerationResult` + JSON |
| [evolutionary_optimization_quickstart.py](../tutorials/evolutionary_optimization_quickstart.py) | GA, SA, NSGA-II multi-objective, Pareto front, dashboard artifact | ~10 s | No | `OptimizationResult` + JSON |
| [graph_rl_quickstart.py](../tutorials/graph_rl_quickstart.py) | random / DQN / PPO comparison, TD3/SAC continuous RL, dashboard artifact | ~30 s | No | `RLResult` + JSON |

## Running all tutorials

```bash
python tutorials/graph_generation_quickstart.py
python tutorials/evolutionary_optimization_quickstart.py
python tutorials/graph_rl_quickstart.py
```

## Colab

Each tutorial is structured as a linear script that can be split into Colab cells.
The three-line one-liner APIs also run directly in Colab:

```python
from tgraphx.generation import run_graph_generation
from tgraphx.evolutionary import run_evolutionary_optimization
from tgraphx.rl import run_graph_rl

g = run_graph_generation(method="barabasi_albert", num_graphs=16, seed=42)
e = run_evolutionary_optimization(algorithm="nsga2", objective=["connectivity","density"], seed=42)
r = run_graph_rl(env="graph_navigation", algorithm="dqn", episodes=20, seed=42)

print(g.metrics)
print(e.best_fitness, e.metrics)
print(r.metrics)
```

## Stability

All tutorials are tested on Python 3.10–3.13, CPU and CUDA.
The generation/evolution/RL subsystems are labeled **Experimental** (v1.0.0).
Tutorial scripts themselves are stable and will not break across v1.x patch releases.
