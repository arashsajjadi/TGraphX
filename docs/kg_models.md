# KG Models

## TransEModel — Beta

Score: ``-||h + r - t||_p``

Head/tail entity embeddings are L2-normalised.
Entity embedding initialisation: U[-6/√D, 6/√D] (Bordes 2013).
Use `MarginRankingLoss` for training.

## DistMultModel — Beta

Score: ``<h, r, t> = Σ h_i · r_i · t_i``

Limitation: symmetric — scores (h,r,t) = (t,r,h) for fixed r.
Use `BCEKGLoss` or `SoftplusKGLoss` for training.

## ComplExModel — Experimental

Score: ``Re(<e_h, e_r, conj(e_t)>)``

Expanded real form:
  ``a_h·a_r·a_t - b_h·b_r·a_t + a_h·b_r·b_t + b_h·a_r·b_t``

Supports asymmetric relations (unlike DistMult).

## RotatEModel — Experimental

Relation as unit-complex rotation: ``r_i = exp(i·θ_i)``
Score: ``margin - ||h ∘ r - t||``

Entity embeddings are normalised to the unit complex circle.
Phase parameterisation is numerically stable.

## Feature-aware extensions

All models support optional `entity_feature_dim` / `relation_feature_dim`.
When provided, a linear projector blends pre-computed features into the embedding:

```python
z_e = embedding[e] + W_x · feature_e
```

**Non-vector features** (images, volumes) must be pre-projected to a vector
by the caller.  TGraphX never silently flattens spatial node features.

## Common interface

```python
model.score_triples(triples)   # LongTensor[B, 3] → FloatTensor[B]
model.score(h, r, t)           # LongTensor[B] × 3 → FloatTensor[B]
```
