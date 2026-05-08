# Migration Notes: v0.2 → v0.3

This document summarises what changes (and what doesn't) when upgrading
from TGraphX v0.2.x to the upcoming v0.3.0 release.

## TL;DR

> If your code uses only stable APIs (no 🧪 marker), there is nothing
> to migrate.  v0.3.0 is fully backward-compatible with the v0.2 stable
> surface.  Experimental APIs may have minor signature refinements;
> migration recipes are below.

## Stable API contract preserved

Every stable v0.2.x API listed in `deprecation_policy.md` keeps its
signature and observable behaviour in v0.3.0:

- Graph / GraphBatch / GraphDataLoader / GraphDataset
- ConvMessagePassing, TensorGATLayer, TensorGraphSAGELayer, TensorGINLayer,
  LinearMessagePassing
- make_layer / build_model / build_model_from_config
- All graph builders and patch helpers
- train_epoch / evaluate / fit / set_seed / save_checkpoint / load_checkpoint
- CSVLogger / TensorBoardLogger
- env_report / recommended_device / estimate_message_memory
- Dashboard HTTP API and CLI flags

If you only use the above, **`pip install tgraphx==0.3.0` works without
code changes**.

## Experimental APIs — possible refinements

The following may evolve between v0.2.x and v0.3.0.  Pinning a specific
v0.2.x release in your project mitigates risk; testing your code against
v0.3.0 release candidates is recommended.

### Hetero (v0.2.5) — promotion candidates

- `HeteroGraph`, `HeteroGraphBatch` likely **promote to stable** in v0.2.8.
- `HeteroConv` may gain hetero-aware edge_features per relation in
  v0.2.7+ before promotion.
- `HeteroGraphClassifier`/`HeteroNodeClassifier` are likely to add a
  `tensor-aware spatial` variant before v0.3.0.

**Migration recipe:** No code change anticipated for the container APIs.
For `HeteroConv`, if you wrote a custom dispatch wrapper, see the new
``edge_features_dict`` parameter and pass it explicitly.

### Temporal (v0.2.5) — promotion candidates

- `TemporalGraphSequence`, `TemporalGraphBatch`, `temporal_readout`
  likely **promote to stable** in v0.2.8.
- `TemporalGraphClassifier` / `TemporalGraphRegressor` are likely to gain
  an explicit ``mask`` argument propagated to the readout.  The default
  behaviour will be unchanged.

### Graph Transformer (v0.2.4 + v0.2.7)

- `GraphTransformerLayer` gains positional encodings (`degree` /
  `laplacian`) and edge-bias support in v0.2.7.  Existing code that
  passed only ``x`` continues to work — both new arguments default to
  unused.
- Factory key ``"graph_transformer"`` added; vector-only.

### Sampling (v0.2.6) — stable

`tgraphx.sampling` and `tgraphx.sampling_loaders` are stable as of v0.2.6.

### Distributed helpers (v0.2.6) — stable

`tgraphx.distributed` helpers are stable as of v0.2.6.

### Optional integrations

- `MLflowLogger`, `tgraphx.interop` (PyG/DGL converters), and
  `tgraphx.learned_graph` are stable interfaces; their behaviour depends
  on the optional dependency installed.

## Removed in v0.3.0

None planned at the time of writing.

## How to test compatibility

```bash
pip install -e .
pytest -q
python examples/run_all_fast_examples.py
```

If you have a downstream project, the canonical compatibility check is:

```python
import tgraphx
# Stable surface:
from tgraphx import (
    Graph, GraphBatch, GraphDataLoader,
    build_model, make_layer,
    fit, train_epoch, evaluate, set_seed,
    CSVLogger, TensorBoardLogger,
    env_report, recommended_device,
    GraphClassifier, NodeClassifier, EdgePredictor,
    induced_subgraph, k_hop_subgraph, neighbor_sample,
    SubgraphDataLoader, NeighborSamplerLoader,
)
# Experimental surface (treat as unstable):
from tgraphx import (
    HeteroGraph, HeteroGraphBatch,
    TemporalGraphSequence, TemporalGraphBatch,
    MLflowLogger,
)
```

If any of the above fails to import, file a GitHub issue.
