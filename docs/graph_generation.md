# Graph Generation Subsystem

TGraphX provides a tensor-aware graph generation subsystem under `tgraphx.generation`.

**Status: Experimental (v0.7.0+)**

## Overview

The generation subsystem enables:

- **Classical generators** with tensor node/edge features
- **MDP-based generation** via graph edit actions
- **Neural generators** (VGAE, autoregressive, transformer)
- **Quality metrics** (validity, uniqueness, novelty, diversity, MMD)

## Quick Example

```python
from tgraphx.generation import (
    FeatureAwareERGraph,
    GraphGenerationConfig,
    validity_score,
    uniqueness_score,
)

# Generate graphs with tensor features
graphs = [FeatureAwareERGraph(n=20, p=0.3, node_feature_dim=8, seed=i) for i in range(10)]

# Measure quality
print("Uniqueness:", uniqueness_score(graphs))
print("Validity:", validity_score(graphs, lambda g: g.num_nodes >= 5))
```

## Classical vs neural generators

`run_graph_generation(method=..., ...)` is a high-level entry point for
**classical** generators only. Supported methods include `erdos_renyi`,
`barabasi_albert`, `watts_strogatz`, `stochastic_block_model`, `grid`,
`cycle`, `path`, `star`, `complete`, `motif_injected`, `anomaly_injected`,
`temporal`, `typed`, and `random_geometric`.

**Neural generators** (VGAE, autoregressive, transformer) are *not* exposed
through `run_graph_generation` because they have very different training and
inference contracts. Use them directly:

```python
from tgraphx.generation import (
    VGAEGraphGenerator,
    AutoregressiveEdgeGenerator,
    GraphTransformerGenerator,
)
```

Passing `method="vgae"` (or `"gae"`, `"autoregressive"`, `"transformer"`)
to `run_graph_generation` raises a helpful `ValueError` that points to the
correct class.

## Submodules

- `data_model.py` — `GeneratedGraph`, `GraphEditState`, `GraphGenerationBatch`
- `actions.py` — MDP action space for graph editing
- `classical.py` — Feature-aware classical generators
- `metrics.py` — Quality metrics (WL hash, MMD, spectral distance)
- `neural.py` — VGAE, autoregressive, and transformer generators
- `projectors.py` — Multi-modal feature projectors
- `config.py` — `GraphGenerationConfig`
- `reports.py` — JSON artifact writers

## Tensor Feature Shapes

- Vector: `[N, F]`
- Image: `[N, C, H, W]` (requires `ImageNodeEncoder`)
- Volume: `[N, C, D, H, W]` (requires `VolumeNodeEncoder`)

See also: [graph_action_spaces.md](graph_action_spaces.md)
