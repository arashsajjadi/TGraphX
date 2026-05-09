# GraphSAINT

TGraphX ships three GraphSAINT-style subgraph samplers and a
DataLoader-compatible loader.  The implementation follows Zeng et al.,
ICLR 2020 ("GraphSAINT: Graph Sampling Based Inductive Learning
Method"), with a Monte-Carlo normalisation estimator instead of an
exact closed form.

## Samplers

| Class | Strategy | Notes |
|---|---|---|
| `GraphSAINTNodeSampler` | uniform node subset | matches the paper's "node sampler"; cheapest variant |
| `GraphSAINTEdgeSampler` | edge sampling with `p_e ∝ 1/deg(u) + 1/deg(v)` | degree-aware, induces touched-node subgraph |
| `GraphSAINTRandomWalkSampler` | walk roots + walks of length `walk_length` | matches the paper's RW sampler |

All samplers accept an optional `seed` and use a per-call
`torch.Generator` (no global RNG side effects).  Sampled subgraphs
preserve `edge_weight`, `edge_features`, and `node_labels`.

## Loader

`GraphSAINTLoader(sampler, attach_norm=True)` wraps any of the above
and yields fresh subgraphs.  When `attach_norm=True` it runs an
estimator that draws `num_norm_samples` independent subgraphs and
records per-node and per-edge inclusion probabilities.  Each yielded
subgraph carries:

- `metadata['sampling']['original_node_ids']`
- `metadata['sampling']['original_edge_ids']`
- `metadata['graphsaint']['node_norm']` — `1 / α_v` clamped
- `metadata['graphsaint']['edge_norm']` — `1 / λ_e` clamped

These coefficients let downstream training reweight aggregations to
remain unbiased on the sampled subgraph (see the GraphSAINT paper for
the proof).

## Quickstart

```python
from tgraphx import Graph, GraphSAINTNodeSampler, GraphSAINTLoader

sampler = GraphSAINTNodeSampler(graph, budget=512, num_steps=50, seed=0)
loader = GraphSAINTLoader(sampler, attach_norm=True, num_norm_samples=20)
for sub in loader:
    out = model(sub)
    loss = (loss_per_node * sub.metadata['graphsaint']['node_norm']).mean()
    loss.backward()
```

## Stability

**Beta** in v0.5.0+. The Monte-Carlo normaliser is approximate;
`num_norm_samples >= 50` is recommended for stable estimates.
