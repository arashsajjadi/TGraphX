# Neural graph mining

`tgraphx.mining.neural` (Experimental, v0.4.0+) provides compact,
trainable neural mining models for prototype graph membership, anomaly
detection, and pattern classification.

These are **foundations**, not state-of-the-art systems.  They are:

- Differentiable end-to-end.
- Compatible with `tgraphx.Graph` objects and raw tensors.
- CPU-first; CUDA optional.
- Free of hidden training, hidden downloads, or telemetry.
- Honest about scope: not SOTA, not replacements for specialised libraries.

## Import

```python
from tgraphx.mining import (
    PrototypeMembershipScorer,
    GraphAutoencoderAnomalyDetector,
    GraphPatternClassifier,
    create_synthetic_pattern_dataset,
    train_prototype_membership_step,
    train_anomaly_autoencoder_step,
    train_graph_pattern_classifier_step,
)
```

---

## `PrototypeMembershipScorer`

Scores whether a candidate graph's query node "belongs" to the class
represented by the support graph.

**Architecture:** shared 2-layer GNN encoder → mean-pool support nodes
→ Siamese MLP scorer on `[support_emb; query_emb; |diff|; product]`.

```python
model = PrototypeMembershipScorer(
    in_dim=8, hidden_dim=64, out_dim=32, num_gnn_layers=2,
)

# Score a single candidate.
logit = model(node_features, edge_index, query_idx=N-1, num_nodes=N)
# Scalar logit; positive = belongs to class.

# Score multiple candidates (sequential).
logits = model.score_batch(candidates)    # [B]

# Score multiple candidates (single GNN pass — faster for large B).
logits = model.score_batch_fast(candidates)  # [B]
```

| Argument | Type | Description |
|----------|------|-------------|
| `in_dim` | `int` | Node feature dimension |
| `hidden_dim` | `int` | GNN hidden dimension (default 64) |
| `out_dim` | `int` | Encoder output dimension (default 32) |
| `num_gnn_layers` | `int` | GNN depth (default 2) |
| `dropout` | `float` | Dropout probability (default 0) |
| `flatten_spatial` | `bool` | Auto-flatten spatial `[N,C,H,W]` features (default False) |

**Forward arguments:** `(node_features, edge_index, query_idx, num_nodes=None)`.

**Training:**

```python
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss = train_prototype_membership_step(model, opt, candidates, targets)
# candidates: list of dicts with node_features, edge_index, query_idx
# targets: FloatTensor[B] of 0/1 labels
```

Uses `BCEWithLogitsLoss` internally.

**Batched forward** (`score_batch_fast`): concatenates all candidate graphs
into one disjoint graph, runs a single GNN pass, then extracts
per-graph embeddings.  Gradient-compatible.

---

## `GraphAutoencoderAnomalyDetector`

Reconstruction-based node anomaly scoring using a graph autoencoder.
Train on normal graphs; nodes with high reconstruction error are flagged.

```python
ae = GraphAutoencoderAnomalyDetector(
    in_dim=8, latent_dim=16, hidden_dim=32, num_gnn_layers=2,
)
# Train.
loss = ae.reconstruction_loss(x, edge_index)
# Inference (no_grad; no autograd retention).
node_scores = ae.node_anomaly_scores(x, edge_index)  # FloatTensor[N]
graph_score = ae.graph_anomaly_score(x, edge_index)  # float
```

| Argument | Type | Description |
|----------|------|-------------|
| `in_dim` | `int` | Input (and output) node feature dimension |
| `latent_dim` | `int` | Latent space dimension |
| `hidden_dim` | `int` | Encoder/decoder hidden dimension |

**Training:**

```python
opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
loss_val = train_anomaly_autoencoder_step(ae, opt, x, edge_index)
```

---

## `GraphPatternClassifier`

Classifies graphs into structural pattern families using a GNN +
mean-pooling + MLP classifier.

```python
clf = GraphPatternClassifier(
    in_dim=4, hidden_dim=32, enc_dim=16, num_classes=4,
)
# Graph-level logits.
logits = clf(node_features, edge_index, num_nodes=N)  # [num_classes]
```

**Training:**

```python
opt = torch.optim.Adam(clf.parameters(), lr=5e-3)
loss = train_graph_pattern_classifier_step(clf, opt, graphs, labels)
# graphs: list of dicts with node_features, edge_index, num_nodes
# labels: LongTensor[B]
```

Uses `CrossEntropyLoss` internally.

---

## Synthetic dataset

```python
from tgraphx.mining import create_synthetic_pattern_dataset

ds = create_synthetic_pattern_dataset(
    num_graphs_per_class=40, num_nodes=8, in_dim=4, seed=0, noise_std=0.05,
)
# Returns list of dicts: node_features, edge_index, num_nodes, label (0-3), pattern
# Pattern classes: 'path', 'star', 'cycle', 'complete'
```

---

## Backpropagation behavior

- All three models are end-to-end differentiable.
- `node_anomaly_scores` and `graph_anomaly_score` use `@torch.no_grad()`
  and return detached tensors — no autograd graph is retained.
- Score functions preserve the gradient graph for training.
- All three pass tiny-overfit tests on synthetic controlled data.

---

## Reproducibility

Set a seed before building models for reproducible weight initialisation:

```python
from tgraphx.reproducibility import set_seed
set_seed(42)
model = PrototypeMembershipScorer(in_dim=8, ...)
```

---

## Limitations

- Neural models are compact foundations, not production-hardened systems.
- `PrototypeMembershipScorer` uses a 2-layer mean-aggregation GNN (no
  attention, no edge features in the encoder by default).
- `GraphAutoencoderAnomalyDetector` uses node-feature MSE reconstruction;
  it does not model the graph topology directly in the loss.
- `GraphPatternClassifier` may not generalise beyond the structural
  pattern families it is trained on.
- No TGN/TGAT, no graph diffusion, no spatial message-passing support
  in the neural mining models (use `ConvMessagePassing` etc. for that).

## Stability

**Experimental.**  APIs may evolve in v0.4.x.

## Related

- Tests: `tests/test_neural_mining.py`, `tests/test_neural_mining_batched.py`
- Benchmarks: `benchmarks/mining/benchmark_neural_mining.py`
- Examples: `examples/neural_prototype_membership_demo.py`,
  `examples/neural_graph_anomaly_demo.py`,
  `examples/neural_graph_pattern_classifier_demo.py`
- Docs: `docs/prototype_graphs.md`, `docs/anomaly_detection.md`
