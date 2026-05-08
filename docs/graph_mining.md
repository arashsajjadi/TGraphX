# Graph mining and pattern recognition

`tgraphx.mining` (v0.3.2+) provides tensor-aware graph mining utilities
for GNN training workflows.  All algorithms are pure-PyTorch, require
no heavy optional dependencies, and produce dashboard-compatible JSON
artifacts.

**TGraphX is not a NetworkX, gSpan, or full graph-analytics library
replacement.**  These utilities are GNN-training-oriented building blocks.

## Import

```python
import tgraphx.mining as m
# or individual functions:
from tgraphx.mining import (
    graph_density, triangle_count, common_neighbors_score,
    wl_kernel_matrix, label_propagation_communities,
    ClassGraphBuilder, MembershipEvaluator,
)
```

---

## Level 1 — Structural features (Beta)

### `graph_density`

```python
from tgraphx.mining import graph_density
d = graph_density(edge_index, num_nodes, directed=True, exclude_self_loops=True)
# d ∈ [0, 1]
```

### `degree_statistics` / `graph_summary`

```python
from tgraphx.mining import degree_statistics, graph_summary
stats = degree_statistics(edge_index, num_nodes)
# returns min/max/mean in/out/total degree, isolated nodes, density
summary = graph_summary(edge_index, num_nodes)
# JSON-serializable; also includes connected component count
```

### `structural_features`

```python
from tgraphx.mining import structural_features
sf = structural_features(edge_index, num_nodes,
    features=("degree", "in_degree", "out_degree", "log_degree"))
# FloatTensor[N, F]
```

Available feature names: `"degree"`, `"in_degree"`, `"out_degree"`,
`"log_degree"`, `"log_in_degree"`, `"log_out_degree"`, `"norm_degree"`.

### `add_structural_features`

Appends structural features to a TGraphX `Graph`.  For **vector** node
features (`[N, D]`), the features are concatenated.  For **spatial/volumetric**
node features (`[N, C, H, W]` or `[N, C, D, H, W]`), they are stored
in `graph.metadata[key]` to preserve the spatial layout.

```python
from tgraphx.mining import add_structural_features
g2 = add_structural_features(graph, features=("log_degree",))
```

---

## Level 1 — Motifs (Beta)

```python
from tgraphx.mining import (
    triangle_count, wedge_count, local_clustering_coefficient,
    motif_counts, motif_features,
)
# Graph-level triangle count (K3 = 1 triangle):
t = triangle_count(edge_index, num_nodes, directed=False)
# Per-node clustering coefficients:
cc = local_clustering_coefficient(edge_index, num_nodes)  # FloatTensor[N]
# Full motif summary dict:
mc = motif_counts(edge_index, num_nodes)
# JSON-serializable
# Per-node feature matrix [N, 3] = [degree, triangle_count, clustering_coeff]:
feats = motif_features(edge_index, num_nodes)
```

**Complexity:** O(N × d²) where d is average degree.  A warning is
emitted for N > 10 000.

---

## Level 1 — Classical link prediction scores (Beta)

```python
from tgraphx.mining import (
    common_neighbors_score, jaccard_score, adamic_adar_score,
    resource_allocation_score, preferential_attachment_score,
)
# pairs: LongTensor[2, P] of candidate edges to score.
scores = common_neighbors_score(edge_index, pairs, num_nodes)
# FloatTensor[P]
```

All functions return `FloatTensor[P]`.  Zero denominators return 0.

---

## Level 2 — WL features and kernel (Beta)

```python
from tgraphx.mining import (
    weisfeiler_lehman_labels, wl_graph_features, wl_kernel_matrix,
    degree_histogram_features,
)
# WL kernel matrix [G, G] normalised:
K = wl_kernel_matrix(graphs, num_iterations=3, normalize=True)
# graphs: list of dicts {'edge_index': ..., 'num_nodes': ...}
# or TGraphX Graph objects
```

**Note:** WL label hashes are deterministic within a Python session.
Cross-session reproducibility requires a fixed `node_labels` vocabulary.

---

## Level 2 — Graph similarity (Beta)

```python
from tgraphx.mining import (
    wl_feature_similarity, pairwise_graph_similarity,
    degree_histogram_distance, graph_feature_cosine_similarity,
)
s = wl_feature_similarity(g1_ei, g1_n, g2_ei, g2_n)   # float ∈ [0,1]
S = pairwise_graph_similarity(graphs, method="wl")      # FloatTensor[G,G]
```

---

## Level 2 — Community detection (Beta)

```python
from tgraphx.mining import (
    label_propagation_communities, modularity, community_summary,
)
labels = label_propagation_communities(edge_index, num_nodes, max_iter=50, seed=0)
# LongTensor[N] of community labels in [0, K)
Q = modularity(edge_index, labels, num_nodes)  # float
```

This is a simple synchronous label-propagation baseline — **not** Louvain.

---

## Level 2 — Random walks (Beta)

```python
from tgraphx.mining import random_walks, generate_random_walks

walks = random_walks(edge_index, start_nodes, walk_length=20, seed=0)
# LongTensor[W, walk_length+1]
# Dead ends: node stays in place.
all_walks = generate_random_walks(
    edge_index, num_nodes=N, num_walks_per_node=10, walk_length=20, seed=0,
)
# LongTensor[N*10, 21]
```

Biased Node2Vec walks (p≠1 or q≠1) are supported but CPU-only.

---

## Level 2 — Anomaly detection (Experimental)

```python
from tgraphx.mining import (
    DegreeAnomalyScorer, EgoDensityAnomalyScorer, graph_level_anomaly_scores,
)

scorer = DegreeAnomalyScorer().fit(train_edge_index, num_nodes)
scores = scorer.score_nodes(test_edge_index, num_nodes)  # FloatTensor[N]

g_scores = graph_level_anomaly_scores(graphs, method="degree_histogram")  # FloatTensor[G]
```

These are **classical baseline anomaly scorers**, not SOTA anomaly
detection.

---

## Level 2 — Prototype graph membership (Experimental)

The **class-graph membership** paradigm is TGraphX's native approach
to pattern recognition for structured image/volume graph inputs.

```python
from tgraphx.mining import (
    ClassGraphBuilder, CandidateGraphBuilder,
    MembershipEvaluator, cosine_graph_membership_baseline,
)

# 1. Build one support graph per class from training embeddings.
builder = ClassGraphBuilder(k_support=5, max_neighbor_fraction=0.5).fit(
    node_features, labels, embeddings=embeddings,
)

# 2. For each query, build a candidate graph.
cand_builder = CandidateGraphBuilder(top_k_query=5)
cg = builder.get_class_graph(cls)
candidate, query_idx = cand_builder.build(cg, query_features, query_embedding)

# 3. Score and evaluate.
def score_fn(candidate): ...  # returns float; higher = more likely true class

result = MembershipEvaluator.evaluate(
    score_fn, query_features, query_labels, builder, cand_builder,
)
# result: {'accuracy', 'balanced_accuracy', 'classification_report',
#           'confusion_matrix', 'top_confusion_pairs', ...}
```

**Tensor-aware:** node features may be `[N, D]`, `[N, C, H, W]`, or
`[N, C, D, H, W]`.  Embeddings for topology construction are separate
from raw features and are always `[N, D_embed]`.

---

## Level 3 — Small patterns (Experimental)

```python
from tgraphx.mining import (
    path_pattern_count, star_pattern_count, contains_triangle, small_pattern_counts,
)
t = contains_triangle(edge_index, num_nodes)           # bool
p = path_pattern_count(edge_index, num_nodes, length=2) # int
s = star_pattern_count(edge_index, num_nodes, center_degree=3)  # int
```

---

## Level 3 — Temporal mining (Experimental)

```python
from tgraphx.mining import (
    temporal_degree, sliding_window_edges,
    temporal_chronological_split, burst_score,
)

tr, va, te = temporal_chronological_split(timestamps)
# Returns boolean masks; train timestamps are strictly ≤ val timestamps.

windows = sliding_window_edges(src, dst, timestamps, window_size=100.0, step=50.0)
```

---

## Reports and dashboard integration

```python
from tgraphx.mining import (
    write_graph_mining_summary,
    write_motif_summary,
    write_link_prediction_summary,
    write_anomaly_summary,
    write_prototype_membership_report,
)

write_graph_mining_summary("runs/demo/graph_mining_summary.json", summary)
write_anomaly_summary("runs/demo/anomaly_summary.json", "degree_zscore", scores)
```

The dashboard reads `graph_mining_summary.json` and `anomaly_summary.json`
when present in the `--logdir` directory.  No crash if files are absent.

---

## Stability labels

| Module / symbol | Stability |
|-----------------|-----------|
| `graph_density`, `degree_statistics`, `graph_summary`, `structural_features`, `add_structural_features` | Beta |
| `triangle_count`, `wedge_count`, `local_clustering_coefficient`, `motif_counts`, `motif_features` | Beta |
| `common_neighbors_score`, `jaccard_score`, `adamic_adar_score`, `resource_allocation_score`, `preferential_attachment_score` | Beta |
| `weisfeiler_lehman_labels`, `wl_graph_features`, `wl_kernel_matrix`, `degree_histogram_features` | Beta |
| `degree_histogram_distance`, `wl_feature_similarity`, `pairwise_graph_similarity`, `graph_feature_cosine_similarity` | Beta |
| `label_propagation_communities`, `modularity`, `community_summary` | Beta |
| `random_walks`, `generate_random_walks` | Beta |
| `DegreeAnomalyScorer`, `EgoDensityAnomalyScorer`, `graph_level_anomaly_scores` | Experimental |
| `ClassGraphBuilder`, `CandidateGraphBuilder`, `MembershipEvaluator`, `cosine_graph_membership_baseline` | Experimental |
| `path_pattern_count`, `star_pattern_count`, `contains_triangle`, `small_pattern_counts` | Experimental |
| `frequent_node_labels`, `frequent_degree_bins`, `support_count` | Experimental |
| `temporal_degree`, `sliding_window_edges`, `temporal_chronological_split`, `burst_score` | Experimental |
| `typed_degree_features`, `relation_frequency_features` | Experimental |
| Report writers | Beta |

---

## Limitations

- `triangle_count` and `local_clustering_coefficient` are O(N × d²); use with care on dense large graphs.
- Community detection uses synchronous label-propagation — not Louvain, not hierarchical.
- WL label hashes are session-deterministic only.
- Random walk biased mode (Node2Vec p/q) is CPU-only.
- `ClassGraphBuilder` is an experimental primitive — it does not train a model.
- No gSpan, no full subgraph isomorphism, no graph edit distance.
- `typed_degree_features` and `relation_frequency_features` require a TGraphX `HeteroGraph`.

---

## Related

- Tests: `tests/test_mining_structural.py`, `tests/test_mining_core.py`
- Examples: `examples/graph_mining_structural_demo.py`, `examples/prototype_graph_membership_demo.py`, …
- Architecture: `docs/architecture.md`
