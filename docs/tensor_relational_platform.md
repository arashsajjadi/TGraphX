# The tensor-relational platform: choosing a relation regime

TGraphX is a platform for models of the form

> **tensor-valued entities + relation/fusion operator + readout.**

Nodes keep their spatial layout (`[C, H, W]`, `[C, D, H, W]`, or vectors),
and the *relation operator* decides how entities exchange information.
Different problems call for different relation regimes, and the platform
supports several — a component qualifies as a TGraphX model family by
using the shared `Graph`/`GraphBatch` contracts, batching, configuration,
validation, checkpointing, and evaluation interfaces, not by being a
message-passing operator.

## Operating-regime map

| Your data | Recommended regime | Topology source | TGraphX components |
|---|---|---|---|
| Fixed entity count, identity, order, and alignment (e.g. co-registered modality stacks) | Fixed ordered fusion: stack tensors along channels and use CNN fusion | `fixed` | `CNNEncoder`, plain CNN heads |
| Variable entities **with a meaningful, trusted supplied graph** | Explicit tensor message passing over `edge_index` | `given` | `ConvMessagePassing`, `TensorGATLayer`, `TensorGraphSAGELayer`, `TensorGINLayer`, `build_model(layer="conv"/"gat"/...)` |
| Variable entities **without a trusted supplied graph** | Learned implicit relations: global set attention infers interactions from node content | `learned_implicit` | `SetTransformerModel`, `build_model(family="set_transformer")` |
| Relations should be *constructed* explicitly from content, then message-passed | Learned explicit topology | `learned_explicit` | `tgraphx.learned_graph` (`EdgeScorer`, `top_k_edges_from_scores`, `build_knn_graph_from_embeddings`) + any `given`-topology layer |
| Partially trusted or incomplete supplied graph | Hybrid given + learned relations | `hybrid` | `GraphTransformerLayer(edge_bias=True)` today; fuller hybrid families are on the roadmap |

The vocabulary is available programmatically:

```python
from tgraphx import TOPOLOGY_SOURCES, topology_source_of

topology_source_of("conv")             # "given"
topology_source_of("set_transformer")  # "learned_implicit"
model.topology_source                  # set on every build_model output
```

Conceptual distinctions that matter scientifically:

- **SetTransformer is not "non-relational."** Self-attention learns dense,
  content-dependent pairwise interactions — it is relation-aware but
  *explicit-input-topology-blind*: it never consumes `edge_index` (and
  warns when one is supplied).
- **SetTransformer is not TensorGAT.** TensorGAT attends only over
  supplied graph edges; SetTransformer attends over all node pairs.
- **SetTransformer is not learned explicit topology.** It never builds a
  discrete edge set; `tgraphx.learned_graph` does.

## What the revised PASTIS-R evidence shows

The regime map above is grounded in a controlled re-run of the PASTIS-R
experiments (parcel-level crop classification — a constructed evaluation
task, **not** the published PASTIS panoptic benchmark; numbers are not
comparable to published PASTIS results).  Protocol: 18-class macro-F1,
frozen tile/fold splits, 5 paired seeds, matched 10-epoch budgets,
best-validation-checkpoint selection.  Full artifacts and per-seed
results live in the `TGraphX_revised` workspace
(`01_frozen_base_revised/RESULTS.md`, `04_branch_b_revised/RESULTS.md`,
`06_synthesis/MASTER_REVISED_REPORT.md`); these results are **reused, not
re-run**, for this release.

### S2-only validation (macro-F1, mean over 5 seeds)

| Model (relation regime) | Validation macro-F1 |
|---|---|
| SetTransformer (learned implicit) | **0.7023** |
| TemporalTransformer (learned implicit, sequence) | 0.6914 |
| PairSet (supplied edges, pairwise) | 0.6520 |
| **Corrected explicit-topology TGraphX** (given) | 0.6326 |
| Fixed imputed-slot CNN (fixed fusion) | 0.6196 |
| DeepSets (no relations, pooling) | 0.6099 |
| Flatten-vector GNN (given, spatial layout destroyed) | 0.6012 |
| Old TGraphX config with silent 0.3 CNN dropout (bridge arm) | 0.5360 |

### Branch B multimodal validation (macro-F1, mean over 5 seeds)

| Arm | Validation macro-F1 |
|---|---|
| SetTransformer (learned implicit, blind to topology) | **0.6593** |
| Real-topology TGraphX (given) | 0.6306 |
| S2-only TGraphX (given, single modality) | 0.6232 |
| Matched-content blind baseline (DeepSets) | 0.5813 |
| Shuffled-topology TGraphX (control) | 0.5718 |

### How to read this — three separate levels of claim

- **Platform-level:** among the platform's tested relation modes,
  *learned global set attention performed best on this natural-label
  task, while supplied topology provided an additional measurable
  signal* (real vs shuffled topology **+0.059**, real vs matched-content
  blind **+0.049**, both significant).  This supports offering multiple
  relation regimes rather than forcing one graph operator everywhere.
- **Model-family-level:** the corrected explicit-topology TGraphX family
  is substantially stronger than the previous silently-configured
  version (**+0.097** validation macro-F1 — see the
  [dropout migration note](migration_v1_4_to_v1_5.md)), and beats
  fixed CNN fusion, DeepSets, and flattened GNN baselines on validation.
- **Operator-level:** these numbers do **not** show that
  `ConvMessagePassing` or `TensorGAT` is the best operator on every
  task, and a SetTransformer win is not a message-passing win.

Generalization caveat: on the frozen geographic test tile, all models
drop sharply (SetTransformer 0.4272, corrected TGraphX 0.3243), and the
validation-level correction gain did not transfer to the out-of-tile
test.  Relation-mode choice does not remove distribution shift.

### BatchNorm is density-dependent, not universally good or bad

The same experiments isolated aggregator BatchNorm: it **helped** on
dense temporal-chain graphs where nodes almost always have neighbors
(+0.017 validation macro-F1) and was **harmful** in sparse candidate
pools where 58.5% of rows had zero degree (it normalizes zero-message
rows into the batch statistics).  TGraphX therefore keeps
`use_batchnorm` explicit on `DeepCNNAggregator` /
`aggregator_params["use_batchnorm"]` — decide per graph density.

## Choosing in practice

```python
from tgraphx import build_model

# Trusted supplied topology → given
gnn = build_model(task="graph_classification", layer="conv",
                  in_shape=(13, 32, 32), hidden_shape=(32, 32, 32),
                  num_layers=2, num_classes=18, dropout=0.0)

# No trusted topology → learned implicit relations
set_model = build_model(task="graph_classification", family="set_transformer",
                        in_shape=(13, 32, 32), hidden_shape=(64,),
                        num_layers=2, num_classes=18, heads=4, dropout=0.0)
```

Both models train with the same `GraphDataset` / `GraphDataLoader` /
`fit` / `evaluate` pipeline; the SetTransformer simply ignores
`edge_index` (warning once) instead of consuming it.

See also: [set_transformer.md](set_transformer.md),
[factories.md](factories.md), [spatial_tensor_gnn.md](spatial_tensor_gnn.md),
[migration_v1_4_to_v1_5.md](migration_v1_4_to_v1_5.md).
