# KG Datasets

`tgraphx.kg.datasets` provides synthetic KG datasets for testing and tutorials.
**No network access required.**

**Stability: Beta (synthetic datasets)**

## Available datasets

### FamilyKG

```python
from tgraphx.kg import FamilyKG
ds = FamilyKG(num_persons=50, seed=0)
print(ds.kg)            # KnowledgeGraph(entities=50, ...)
train, valid, test = ds.train, ds.valid, ds.test
```

Relations: `parentOf (0)`, `childOf (1)`, `siblingOf (2)`, `spouseOf (3)`, `grandparentOf (4)`.
Useful for testing rule mining (parentOf ∘ parentOf → grandparentOf).

### AcademicKG

```python
from tgraphx.kg import AcademicKG
ds = AcademicKG(num_authors=20, num_papers=30, num_venues=5, seed=0)
```

Relations: `wrote`, `publishedIn`, `cites`, `affiliatedWith`, `hostedBy`.

### MultimodalKG

```python
from tgraphx.kg import MultimodalKG
ds = MultimodalKG(num_entities=50, entity_feature_dim=32, relation_feature_dim=8, seed=0)
print(ds.kg.entity_features["x"].shape)   # [50, 32]
print(ds.kg.relation_features["r"].shape) # [num_relations, 8]
```

Use this to test feature-aware KG models.  Features are stored as-is — not flattened.

### generate_synthetic_kg

```python
from tgraphx.kg import generate_synthetic_kg
kg = generate_synthetic_kg(100, 5, 400, seed=0, with_timestamps=True)
# Returns TemporalKnowledgeGraph with random timestamps in [0, 100].
```

## Real datasets (optional, not bundled)

FB15k-237 and WN18RR can be loaded via optional dataset adapters in
`tgraphx.datasets` (requires files on disk + `--download` flag).
TGraphX does **not** bundle any third-party datasets.

## Splits

All dataset classes expose `.train`, `.valid`, `.test` attributes produced by
`train_valid_test_split(seed=...)`.  Splits are disjoint with no triple overlap.
