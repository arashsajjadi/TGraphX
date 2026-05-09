# KG Training

`tgraphx.kg.trainer` provides `KGTrainer` and `KGTrainingConfig` for
reproducible, dashboard-aware KG embedding training.

**Stability: Experimental**

## Quick start

```python
from tgraphx.kg import (
    FamilyKG, DistMultModel, SoftplusKGLoss,
    UniformNegativeSampler, KGTrainer, KGTrainingConfig, KGEvaluator,
)

ds = FamilyKG(num_persons=50, seed=0)
model = DistMultModel(ds.kg.num_entities, ds.kg.num_relations, embedding_dim=64)
evaluator = KGEvaluator(
    ds.train.triples, ds.valid.triples, ds.test.triples, ds.kg.num_entities
)
cfg = KGTrainingConfig(
    num_epochs=100, batch_size=64, loss_type="softplus",
    lr=0.01, grad_clip_norm=1.0, valid_every=20, seed=0,
)
trainer = KGTrainer(model, cfg, ds.train.triples, evaluator=evaluator)
result = trainer.train()
print(f"Final loss: {result['final_loss']:.4f}")
print(f"Best valid MRR: {result['best_valid_mrr']}")
```

## KGTrainingConfig fields

| Field | Default | Description |
|-------|---------|-------------|
| `num_epochs` | 100 | Training epochs |
| `batch_size` | 256 | Positive triples per batch |
| `num_negatives` | 1 | Negatives per positive |
| `loss_type` | `"softplus"` | `"margin"`, `"bce"`, or `"softplus"` |
| `lr` | 1e-3 | Adam learning rate |
| `weight_decay` | 0.0 | L2 regularisation |
| `margin` | 1.0 | Margin (for `"margin"` loss only) |
| `grad_clip_norm` | None | Max gradient norm (None = no clip) |
| `valid_every` | 10 | Validation interval in epochs |
| `seed` | 0 | RNG seed |
| `device` | `"cpu"` | `"cpu"`, `"cuda"`, or `"auto"` |

## Losses

- `"margin"` → `max(0, γ - s_pos + s_neg)` — works well with TransE.
- `"bce"` → `BCEWithLogits` — works well with DistMult/ComplEx.
- `"softplus"` → `softplus(-s_pos) + softplus(s_neg)` — stable, works broadly.

## Negative samplers

Pass any `_BaseNegativeSampler` via `sampler=...`.  Default: `UniformNegativeSampler`.

```python
from tgraphx.kg import FilteredNegativeSampler, UniformNegativeSampler

sampler = FilteredNegativeSampler(
    N_e, num_negatives=2,
    positive_set=kg.positive_triple_set(),
    base_sampler=UniformNegativeSampler(N_e, 1),
)
trainer = KGTrainer(model, cfg, train_triples, sampler=sampler)
```

## Limitations

- No automatic LR scheduling (use callbacks or post-hoc).
- AMP (`torch.cuda.amp`) not yet wired; manual use is possible.
- Dashboard writing must be done by the caller from `result` dict.

## Dashboard artifacts

```python
from tgraphx.kg.reports import write_kg_training_report
write_kg_training_report("logs/kg_training_report.json", result)
```

See `examples/kg_training_pipeline_demo.py`.
