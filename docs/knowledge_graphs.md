# Knowledge Graphs

TGraphX provides a tensor-aware knowledge graph subsystem in `tgraphx.kg`.

## What it is

A knowledge graph (KG) is a directed multi-relational graph whose triples
``(h, r, t)`` encode typed relationships between entities.  TGraphX's KG
subsystem extends this with:

- **Tensor entity features** — vector, image-like `[N, C, H, W]`, volume
  `[N, C, D, H, W]`, or arbitrary tensor features per entity.
- **Typed/multimodal entities** — entity types, modality masks, and typed
  projectors (image, user, text) via `MultimodalKGModel`.
- **Tensor relation features** — embeddings or feature vectors per relation.
- **Triple features** — per-triple edge attributes, weights, confidence
  scores, and timestamps.
- **Filtered ranking evaluation** — correct head+tail MRR/Hits@K.
- **KG+GNN integration** — RGCN link prediction over KGs.
- **Temporal KG** — timestamped events with no-future-leakage splits.
- **KG reasoning** — path extraction and Horn-rule candidate generation.

## Quick start

```python
from tgraphx.kg import (
    KnowledgeGraph, FamilyKG,
    DistMultModel, SoftplusKGLoss,
    UniformNegativeSampler,
    KGTrainer, KGTrainingConfig, KGEvaluator,
)

# Load synthetic dataset.
ds = FamilyKG(num_persons=50, seed=0)
kg, tr, va, te = ds.kg, ds.train, ds.valid, ds.test

# Train DistMult.
model = DistMultModel(kg.num_entities, kg.num_relations, embedding_dim=64)
evaluator = KGEvaluator(tr.triples, va.triples, te.triples, kg.num_entities)
cfg = KGTrainingConfig(num_epochs=100, batch_size=64, loss_type="softplus", seed=0)
trainer = KGTrainer(model, cfg, tr.triples, evaluator=evaluator)
result = trainer.train()

# Evaluate.
test_result = evaluator.evaluate(model, te.triples)
print(f"Filtered MRR: {test_result.filt_mrr:.4f}")
```

## Components

| Component | Module | Stability |
|-----------|--------|-----------|
| KnowledgeGraph, TemporalKnowledgeGraph | `tgraphx.kg.data` | Beta |
| UniformNegativeSampler, BernoulliNegativeSampler, FilteredNegativeSampler | `tgraphx.kg.sampling` | Beta |
| KGEvaluator, evaluate_filtered_ranking | `tgraphx.kg.evaluation` | Beta |
| TransEModel, DistMultModel | `tgraphx.kg.models` | Beta |
| ComplExModel, RotatEModel | `tgraphx.kg.models` | Experimental |
| KGTrainer, KGTrainingConfig | `tgraphx.kg.trainer` | Experimental |
| KGRGCNModel | `tgraphx.kg.gnn` | Experimental |
| TemporalKGNegativeSampler | `tgraphx.kg.temporal` | Experimental |
| PathExtractor, HornRuleCandidate, LogicalConstraintChecker | `tgraphx.kg.reasoning` | Experimental |
| FamilyKG, AcademicKG, MultimodalKG | `tgraphx.kg.datasets` | Beta |

## Limitations

- No billion-scale KG support.
- Filtered ranking evaluation scans all N_e candidates per triple — use
  `chunk_size` to control memory.
- KG+GNN (RGCN) is Experimental and not benchmarked against reference
  implementations.
- Temporal KG and reasoning are Experimental foundations.

See also: `docs/kg_data_model.md`, `docs/kg_models.md`, `docs/kg_evaluation.md`.
