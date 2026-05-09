# sklearn-like estimator API

`tgraphx.estimators` provides a small `BaseGraphEstimator` plus
concrete wrappers around TGraphX algorithms (label propagation,
Node2Vec, VGAE) so users can adopt a familiar `fit / predict / score`
workflow.

## Estimators

| Class | Wraps | Returns |
|---|---|---|
| `LabelPropagationEstimator` | iterative label propagation | `LongTensor[N]` predicted classes |
| `Node2VecEstimator` | `Node2VecEmbedding` skip-gram trainer | `FloatTensor[N, D]` embeddings |
| `VGAEEstimator` | `VGAE` link-prediction model | `FloatTensor[N, D]` embeddings |

All estimators implement `fit / predict / get_params / set_params`,
plus `transform` and `predict_proba` where applicable.

## Pipeline

```python
from tgraphx.estimators import GraphPipeline, Node2VecEstimator, LabelPropagationEstimator

pipe = GraphPipeline([
    ("emb", Node2VecEstimator(embedding_dim=64, walk_length=20)),
    ("lp",  LabelPropagationEstimator(num_iters=20)),
])
pipe.fit(graph, y)
preds = pipe.predict(graph)
```

## Splits

```python
from tgraphx.estimators import (
    node_train_val_test_split,
    edge_train_val_test_split,
    temporal_train_val_test_split,
)
```

`temporal_train_val_test_split` enforces strict no-leakage ordering on
edge timestamps.

## Early stopping

```python
from tgraphx.estimators import EarlyStopping

es = EarlyStopping(patience=5, mode="max")
for epoch in ...:
    if es.step(val_metric):
        break
```

## Stability

**Beta** in v0.5.0+. The estimator base class is intentionally a
*subset* of scikit-learn's contract; we do not aim for full sklearn
interoperability.
