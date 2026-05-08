# Explainability

`tgraphx.explain` (new in v0.3.0) provides diagnostic explainability
helpers for TGraphX models.  These are **diagnostic tools**, not causal
proof — they reveal which inputs the model is sensitive to under a given
forward pass, not why the model makes predictions in general.

All methods:
- run on CPU unless you explicitly move the model and graph to another device,
- do not retain autograd graphs after returning (no `create_graph=True`),
- make no claims about causality or model correctness.

## Install

No extra dependencies beyond the base `tgraphx` install.

## Import

```python
from tgraphx.explain import (
    node_feature_saliency,
    integrated_gradients,
    edge_gradient_attribution,
    edge_perturbation_attribution,
    attention_to_edge_scores,
    patch_saliency_to_image_grid,
    patch_saliency_to_volume_projection,
    export_explanation_metadata,
    export_edge_scores_csv,
    export_patch_heatmap_json,
)
```

## Node-feature saliency

```python
sal = node_feature_saliency(model, graph, target=label)
# sal: tensor with the same shape as graph.node_features
```

| Argument | Type | Description |
|----------|------|-------------|
| `model` | `nn.Module` | Model with `forward(node_features, edge_index)` signature |
| `graph` | `Graph` | Any TGraphX graph (vector, spatial, or volumetric features) |
| `target` | `int` | Class index to backprop from; ignored for scalar outputs |
| `abs_value` | `bool` | Return `|∂y/∂x|` (default `True`); set `False` for signed gradients |

**Returns:** `Tensor` shaped like `graph.node_features`.

## Integrated gradients

Riemann-sum approximation of Integrated Gradients (Sundararajan et al., 2017).

```python
ig = integrated_gradients(model, graph, target=label, steps=16)
# ig: tensor with the same shape as graph.node_features
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `model` | `nn.Module` | — | Model |
| `graph` | `Graph` | — | Input graph |
| `target` | `int` | `0` | Target class |
| `baseline` | `Tensor \| None` | `None` | Zero baseline when omitted |
| `steps` | `int` | `16` | Integration steps (>=2); higher → more accurate |

**Returns:** `Tensor` shaped like `graph.node_features`.

## Edge gradient attribution

Gradient of the target logit with respect to `graph.edge_weight`.
When the model does not use edge weights, returns zeros and emits a warning.

```python
edge_imp = edge_gradient_attribution(model, graph, target=label)
# edge_imp: tensor of shape [num_edges]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `model` | `nn.Module` | — | Model |
| `graph` | `Graph` | — | Graph (edge_weight may be None; ones are used in that case) |
| `target` | `int` | `0` | Target class |
| `abs_value` | `bool` | `True` | Return absolute values |

## Edge perturbation attribution

Model-agnostic: drops each edge in turn and measures `Δlogit`.
Positive score means the edge supports the prediction.

```python
edge_imp = edge_perturbation_attribution(model, graph, target=label, max_edges=64)
# edge_imp: tensor of shape [min(max_edges, num_edges)]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `model` | `nn.Module` | — | Model |
| `graph` | `Graph` | — | Input graph |
| `target` | `int` | `0` | Target class |
| `max_edges` | `int` | `256` | Cap on edges to perturb; keeps this CI-safe for large graphs |

## Attention edge scores

Convert raw `TensorGATLayer` attention weights to per-edge scalar scores.

```python
out, attn = layer(x, edge_index, return_attention=True)
edge_scores = attention_to_edge_scores(attn, edge_index, head_reduce="mean")
# edge_scores: tensor of shape [num_edges]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `attn` | `Tensor` | — | Raw attention weights from `TensorGATLayer` |
| `edge_index` | `Tensor` | — | `[2, E]` edge index |
| `head_reduce` | `str` | `"mean"` | How to reduce across heads: `"mean"` or `"max"` |

## Patch heatmap helpers

For graphs built from image patches (`grid_shape` stored in
`graph.metadata["grid_shape"]`):

```python
heatmap = patch_saliency_to_image_grid(sal, grid_shape=(4, 4))
# heatmap: [H_full, W_full] float tensor
```

For 3-D volumetric patches:

```python
proj = patch_saliency_to_volume_projection(sal, grid_shape=(2, 4, 4), axis=0)
# proj: 2-D projection
```

## Exports (dashboard-compatible)

All export helpers write to **explicit paths** that you provide; they never
create files outside the paths you specify.

```python
# Write explanation_metadata.json (dashboard reads this)
export_explanation_metadata(
    "runs/demo/explanation_metadata.json",
    method="saliency",        # or "integrated_gradients", "edge_perturbation", etc.
    target=int(label),
)

# Write top-k edge scores as explanation_edges.csv
export_edge_scores_csv(
    "runs/demo/explanation_edges.csv",
    graph.edge_index,
    edge_scores,
    top_k=20,
)

# Write patch heatmap as explanation_patch_heatmap.json
export_patch_heatmap_json(
    "runs/demo/explanation_patch_heatmap.json",
    heatmap,
    grid_shape=(4, 4),
)
```

The dashboard renders `explanation_metadata.json`, `explanation_edges.csv`,
and `explanation_patch_heatmap.json` when they are present in the `--logdir`.

## Limitations

- Attribution methods assume a deterministic `forward(node_features, edge_index)`
  signature.  Dropout layers should be in `eval()` mode for consistent results
  (the helpers switch the model to `eval()` automatically and restore training
  mode afterward).
- `edge_perturbation_attribution` is O(max_edges) forward passes — it is slow
  for large graphs.  Use `max_edges` to limit the computation.
- Integrated gradients uses a zero baseline by default, which may not be
  meaningful for all feature spaces.  Pass a domain-appropriate `baseline`.
- Saliency and IG reflect sensitivity of the model's current weights; they
  change every time the model is retrained.

## Related examples and tests

- `examples/explainability_saliency_demo.py` — vanilla gradient saliency
- `examples/explainability_attention_demo.py` — attention edge scores
- `examples/explainability_end_to_end_validation.py` — full pipeline
  (train → saliency → IG → edge perturbation → export → dashboard snapshot)
- `tests/test_explainability.py` — shape / finiteness / no-autograd-retention /
  export round-trip tests
