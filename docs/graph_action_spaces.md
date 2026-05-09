# Graph Action Spaces

## MDP Formulation

Graph generation is formalized as a finite-horizon MDP:

    M = (S, A, P, R, gamma)

- **S**: Set of `GraphEditState` objects (partial graphs)
- **A**: Set of `GraphAction` objects
- **P(s'|s,a)**: Deterministic transition for structural actions
- **R(s,a)**: Scalar reward (defined by environment)
- **gamma**: Discount factor in [0, 1]

## Action Types

| Action | Description |
|--------|-------------|
| `ADD_NODE` | Add a new node (with optional features) |
| `ADD_EDGE` | Add a directed edge (src, tgt) |
| `REMOVE_NODE` | Remove a node and all incident edges |
| `REMOVE_EDGE` | Remove a specific edge |
| `SET_NODE_FEATURE` | Update a node's feature tensor |
| `SET_EDGE_FEATURE` | Update an edge's feature tensor |
| `STOP_GENERATION` | End generation |

## Action Masks

Action masks enforce hard constraints before any action is sampled:

```python
from tgraphx.generation.actions import GraphActionSpace, batch_action_masks

space = GraphActionSpace(
    max_nodes=50,
    max_edges=500,
    no_self_loops=True,
    connected_required=False,
    acyclic_required=False,
)

# Get masks for a batch of states
masks = batch_action_masks(states, space)  # BoolTensor [B, max_actions]
```

## Code Example

```python
from tgraphx.generation.data_model import GeneratedGraph, GraphEditState
from tgraphx.generation.actions import (
    GraphAction, GraphActionType, GraphActionSpace,
    apply_graph_action, sample_valid_action,
)
import torch

# Start with an empty graph
g = GeneratedGraph(
    edge_index=torch.zeros((2, 0), dtype=torch.long),
    num_nodes=0,
)
state = GraphEditState(graph=g)
space = GraphActionSpace(max_nodes=10)

# Sample and apply actions
gen = torch.Generator()
gen.manual_seed(42)
for _ in range(5):
    action = sample_valid_action(state, space, generator=gen)
    state = apply_graph_action(state, action)
    print(f"n={state.graph.num_nodes}, e={state.graph.num_edges}")
```
