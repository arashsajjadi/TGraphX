# Temporal Knowledge Graphs

`tgraphx.kg.data.TemporalKnowledgeGraph` and `tgraphx.kg.temporal`
provide foundations for temporal KG learning.

**Stability: Experimental**

## Data model

A temporal KG triple is `(h, r, t, τ)` where `τ` is the event timestamp.

```python
import torch
from tgraphx.kg.data import TemporalKnowledgeGraph

triples = torch.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 0]], dtype=torch.long)
timestamps = torch.tensor([1.0, 2.5, 5.0])
tkg = TemporalKnowledgeGraph(triples, timestamps, num_entities=3, num_relations=2)
print(tkg.summary())
```

Or from `generate_synthetic_kg`:

```python
from tgraphx.kg import generate_synthetic_kg
tkg = generate_synthetic_kg(30, 4, 80, seed=0, with_timestamps=True)
```

## Chronological split (no future leakage)

```python
train, valid, test = tkg.chronological_split(0.7, 0.15, 0.15)
# Guaranteed: train.timestamp.max() <= valid.timestamp.min()
#             valid.timestamp.max() <= test.timestamp.min()
```

## Temporal negative sampling

```python
from tgraphx.kg import TemporalKGNegativeSampler

sampler = TemporalKGNegativeSampler(
    num_entities=N_e, num_negatives=2, temporal_kg=tkg,
)
gen = torch.Generator().manual_seed(0)
neg = sampler.sample(train.triples[:8], train.timestamp[:8], generator=gen, filtered=True)
# filtered=True: rejects triples that exist at or before the event timestamp.
```

## Time-aware filtered evaluation

```python
from tgraphx.kg.temporal import evaluate_temporal_filtered_ranking

result = evaluate_temporal_filtered_ranking(
    model=model, test_kg=test, train_kg=train,
    num_entities=N_e, hits_at=(1, 3, 10),
)
print(result.filt_mrr)
```

For evaluation at time τ, only train triples with `timestamp ≤ τ` are
used for filtering — future positives are never removed from ranking.

## Limitations

- `TemporalKGNegativeSampler` filtered mode is O(B·K·max_attempts) and may be slow for dense temporal KGs.
- Time-aware evaluation iterates per triple; vectorise externally for large test sets.
- TTransE / temporal DistMult models are not yet implemented — temporal scoring is an active research direction.
