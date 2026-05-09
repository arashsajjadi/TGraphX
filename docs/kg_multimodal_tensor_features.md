# Multimodal Tensor-Aware KG Features

TGraphX supports KGs where different entities carry **different modality features**
— image-like tensors, user profile vectors, text embeddings — without any silent
flattening of spatial structure.

**Stability: Experimental**

## Motivation

A typical multimodal KG contains triples such as:

```
(image_001, viewedBy, user_123)
(user_123,  wrote,    text_doc_045)
(text_doc_045, describes, image_001)
```

where:
- `image_001` has image-like features `[N, C, H, W]`
- `user_123` has a profile vector `[N, F]`
- `text_doc_045` has a pre-computed text embedding `[N, D]`

TGraphX preserves these typed features throughout the pipeline — no silent flattening.

## Data model

```python
from tgraphx.kg import KnowledgeGraph
import torch

N_e = 26  # 10 images + 8 users + 8 texts
entity_types = torch.tensor([0]*10 + [1]*8 + [2]*8)

kg = KnowledgeGraph(
    triples,
    num_entities=N_e, num_relations=3,

    # Per-entity type label.
    entity_types=entity_types,
    entity_type_to_id={"image": 0, "user": 1, "text": 2},

    # Modality coverage masks: BoolTensor[N_e], True = entity has this feature.
    entity_feature_masks={
        "image": entity_types == 0,  # only image entities
        "user":  entity_types == 1,
        "text":  entity_types == 2,
    },

    # Typed feature tensors — stored as-is, NOT flattened.
    entity_features={
        "image": torch.randn(N_e, 3, 8, 8),  # [N_e, C, H, W]
        "user":  torch.randn(N_e, 16),        # [N_e, F]
        "text":  torch.randn(N_e, 8),         # [N_e, D]
    },
)

print(kg.entity_type_counts())
# {'image': 10, 'user': 8, 'text': 8}
print(kg.summary()["entity_feature_masks"])
# {'image': {'coverage': 10, 'total': 26}, ...}
```

## Projectors

Each modality has a dedicated projector that maps raw features to a common
``out_dim``-dimensional space:

| Projector | Input shape | Use case |
|-----------|-------------|---------|
| `VectorEntityProjector` | `[N, F]` | User/profile vectors, generic vectors |
| `ImageEntityProjector` | `[N, C, H, W]` | Image/spatial features via global avg pool |
| `TextEntityProjector` | `[N, D]` | Pre-computed text embeddings (alias of Vector) |
| `UserEntityProjector` | `[N, F]` | User features (alias of Vector) |
| `RelationFeatureProjector` | `[N_r, F]` | Relation features |
| `TripleFeatureProjector` | `[N_t, F]` | Triple/edge features |

All projectors are **differentiable**. Gradients flow back through projected features.

**`ImageEntityProjector` never silently flattens the input — passing a 2-D tensor raises a clear error.**

## MultimodalEntityFusion

The fusion module combines projections from all available modalities:

```python
from tgraphx.kg import ImageEntityProjector, VectorEntityProjector, MultimodalEntityFusion

fusion = MultimodalEntityFusion(
    projectors={
        "image": ImageEntityProjector(in_channels=3, out_dim=32),
        "user":  VectorEntityProjector(in_dim=16, out_dim=32),
        "text":  VectorEntityProjector(in_dim=8,  out_dim=32),
    },
    out_dim=32,
    num_entities=N_e,
    fusion_mode="gated",          # or "add" or "concat_project"
    add_learnable_bias=True,      # fallback embedding for entities without features
)
# Entities without a given modality have that modality masked to zero.
```

**Fusion modes:**
- `"add"` — sum all projected modalities.
- `"gated"` — per-modality sigmoid gate (the model learns how much to weight each modality).
- `"concat_project"` — concatenate then project down.

## MultimodalKGModel

Full model combining fusion + DistMult decoder:

```python
from tgraphx.kg import MultimodalKGModel

model = MultimodalKGModel(
    num_entities=N_e, num_relations=N_r, out_dim=32,
    projectors={
        "image": ImageEntityProjector(3, 32),
        "user":  VectorEntityProjector(16, 32),
        "text":  VectorEntityProjector(8, 32),
    },
    fusion_mode="gated",
)

# Score triples from a KG with typed features.
scores = model.score_from_kg(kg, triples)
loss = SoftplusKGLoss()(pos_scores, neg_scores)
loss.backward()
# Gradients flow through image, user, and text projectors.
```

## Backprop path

```
scores = <z_h, e_r, z_t>
z_e = gate_img * ImageProj(img[e]) + gate_usr * UserProj(user[e]) + fallback_emb[e]
     ↕                 ↕                       ↕
  trainable    image features              user features
  gates
```

All paths are differentiable. After `loss.backward()`:
- `ImageEntityProjector.proj.weight.grad` is finite and nonzero.
- `VectorEntityProjector.proj.weight.grad` (user/text) is finite and nonzero.
- `relation_emb.weight.grad` is finite and nonzero.

## Dashboard artifact

```python
from tgraphx.kg.reports import write_kg_summary
write_kg_summary("logs/kg_summary.json", kg.summary())
```

The artifact includes `entity_type_counts`, `entity_feature_masks` with coverage
statistics, and all feature shapes — without dumping raw tensors.

## Limitations

- **Not a full multimodal foundation model.** The model does not contain a
  text tokenizer, a CNN backbone, or a vision-language pre-training step.
- **Image features should be pre-processed.** `ImageEntityProjector` applies
  only global average pooling + linear. For rich spatial reasoning, pass
  CNN-extracted features as input.
- **Text features are pre-computed embeddings.** Passing raw strings is not
  supported — use an external model to compute embeddings first.
- **Absent modality = zero contribution.** Entities without features for a
  modality have that modality masked to zero (they still have a learned fallback).
- **Masked contributions are zero** — entities listed in `entity_feature_masks`
  as False do not contribute modality projections.

See `examples/kg_multimodal_tensor_features_demo.py` for a complete working example.
