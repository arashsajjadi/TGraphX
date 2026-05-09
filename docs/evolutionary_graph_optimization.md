# Evolutionary Graph Optimization

**Status: Experimental (v0.7.0+)**

## Mathematical Background

### Genetic Algorithm

Standard GA loop over graphs:
1. Evaluate fitness for each genome
2. Select parents (tournament, roulette, rank)
3. Apply crossover with probability `crossover_rate`
4. Apply mutation with probability `mutation_rate`
5. Preserve `elitism_k` best individuals

### Simulated Annealing

Metropolis criterion at temperature T:

    Accept candidate if f(new) > f(old)
    Accept with probability exp((f(new) - f(old)) / T) otherwise

Cooling: T ← T * alpha until T < T_min.

### NSGA-II

Multi-objective optimization:
1. Non-dominated sort: fronts F0, F1, F2, ...
   - a dominates b iff ∀i f_a[i] >= f_b[i] and ∃j f_a[j] > f_b[j]
2. Crowding distance for diversity preservation
3. Select by front rank, then crowding distance within front

## Mutation/Crossover Operators

| Operator | Description |
|----------|-------------|
| `mutate_add_node` | Add node with optional features |
| `mutate_remove_node` | Remove node, remap edge IDs |
| `mutate_add_edge` | Add random valid edge |
| `mutate_remove_edge` | Remove random edge |
| `mutate_rewire_edge` | Change one endpoint of an edge |
| `mutate_node_feature` | Gaussian noise on node feature |
| `edge_set_crossover` | Union edges, split randomly |
| `node_induced_crossover` | Subgraph crossover |
| `feature_crossover` | Swap features between parents |

## Fitness Functions

- `connectivity_fitness`: Fraction of reachable node pairs
- `density_fitness`: Closeness to target density
- `clustering_fitness`: Average clustering coefficient
- `motif_count_fitness`: Triangle/wedge count
- `constraint_penalty`: Penalty for constraint violations
- `composite_fitness`: Weighted sum of components

## Limitations

- Designed for small graphs (N < 100). Quadratic operations warn at N > 500.
- No GPU acceleration for fitness evaluation.
- For large-scale optimization, use dedicated graph optimization libraries.

## Code Example

```python
from tgraphx.evolutionary import (
    GraphGenome, GeneticAlgorithmOptimizer, GeneticAlgorithmConfig,
    connectivity_fitness, EvolutionConfig, write_evolution_report,
)
import torch

# Create initial population
def make_genome(n=6, seed=0):
    rng = torch.Generator().manual_seed(seed)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    return GraphGenome(edge_index=ei, num_nodes=n)

population = [make_genome(seed=i) for i in range(10)]

# Run GA
config = GeneticAlgorithmConfig(population_size=10, n_generations=20, seed=42)
optimizer = GeneticAlgorithmOptimizer(config, connectivity_fitness)
result = optimizer.optimize(population)
print(f"Best fitness: {result.best_fitness:.4f}")
```
