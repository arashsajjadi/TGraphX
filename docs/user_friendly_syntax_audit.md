# User-friendly syntax audit (v1.4.0)

This document classifies every alias / shortcut that TGraphX **does or does not**
accept. The classification follows v1.4.0 hard rules:

- **IMPLEMENTED_AND_TESTED**: The alias works and has regression tests.
- **DOCUMENTED_ONLY**: The canonical name remains stable; no alias is added but
  docs explain how to use it.
- **REJECTED_WITH_HELPFUL_ERROR**: The alias is intentionally not added; an
  attempt raises an actionable error with the canonical form.
- **ROADMAP_ONLY**: Reserved for a future minor release; tests describe the
  shape of the intended API but do not run yet.

Compatibility promise: every API listed as "stable" in
[docs/api_stability.md](api_stability.md) preserves its signature within a
major version. v1.4.0 introduces new beta aliases without removing or renaming
any existing public API.

## Group summary table

| Group | Topic | Status | Canonical | Aliases | Tests |
| ----- | ----- | ------ | --------- | ------- | ----- |
| 71 | PyG-style Graph(x=, y=, edge_attr=) | IMPLEMENTED_AND_TESTED | `Graph(node_features=...)` | `x=`, `y=`, `labels=`, `edge_attr=` | `TestGroup71PyG` |
| 72 | DGL conversion | DOCUMENTED_ONLY | `from_dgl_graph`, `to_dgl_graph` | (optional dep) | `TestGroup72DGL` |
| 73 | NetworkX conversion | IMPLEMENTED_AND_TESTED | `Graph.from_networkx`, `Graph.to_networkx` | — | `TestGroup73NetworkX` |
| 74 | Adjacency conversion | IMPLEMENTED_AND_TESTED | `Graph.from_adjacency` (dense torch / sparse scipy) | — | `TestGroup74Adjacency` |
| 75 | Edge-list conversion | IMPLEMENTED_AND_TESTED | `Graph.from_edges(edge_list, num_nodes=...)` | `[E,2]`, `[2,E]`, list-of-tuples | `TestGroup75EdgeList` |
| 76 | Dataset loader registry | IMPLEMENTED_AND_TESTED | `tgx.load_dataset(name, ...)` | `mnist_graph`, `cifar10_patch`, `cora`, `mutag`, `movielens_kg` (planned), ... | `TestGroup76DatasetRegistry` |
| 77 | Model factory | DOCUMENTED_ONLY | `tgraphx.build_model` | — | `TestGroup77ModelFactory` |
| 78 | Unified workflow | IMPLEMENTED_AND_TESTED | `tgx.workflow(task=...)` | `run_workflow` | `TestGroup78Workflow` |
| 79 | Task-name aliases | IMPLEMENTED_AND_TESTED | `node_classification` | `node-classification`, `node_cls`, `kg_completion`, ... | `TestGroup79TaskAliases` |
| 80 | Tensor-native flag | IMPLEMENTED_AND_TESTED | `tgx.assert_tensor_native(g, min_rank=3)` | `validate_graph(allow_vector_features=False)` | `TestGroup80TensorMode` |
| 81 | Edge / relation mode | DOCUMENTED_ONLY | `edge_attr` tensor | — | `TestGroup81EdgeMode` |
| 82 | kNN graph | IMPLEMENTED_AND_TESTED | `tgx.knn_graph(x, k=...)` | `metric=`, `make_symmetric=`, `exclude_self=` | `TestGroup82KNN` |
| 83 | Prototype graph | IMPLEMENTED_AND_TESTED | `tgx.build_class_prototypes`, `tgx.build_prototype_graph` | — | `TestGroup83Prototype` |
| 84 | Patch graph from image | IMPLEMENTED_AND_TESTED | `tgx.image_to_patch_graph(image, patch_size=...)` | — | `TestGroup84PatchGraph` |
| 85 | Graph-level readout | IMPLEMENTED_AND_TESTED | `global_mean_pool`, `global_max_pool`, `global_sum_pool` | — | `TestGroup85Readout` |
| 86 | Negative sampling | IMPLEMENTED_AND_TESTED | `tgraphx.negative_sampling` | — | `TestGroup86NegSampling` |
| 87 | Metrics | IMPLEMENTED_AND_TESTED | `tgraphx.accuracy`, `mean_squared_error`, ... | — | `TestGroup87Metrics` |
| 88 | Config dict support | DOCUMENTED_ONLY | `tgx.workflow(**kwargs)` | — | `TestGroup88Config` |
| 89 | CLI ergonomics | IMPLEMENTED_AND_TESTED | `python -m tgraphx doctor`, `tgraphx-doctor` | `info`, `tasks`, `models` | `TestGroup89CLI` |
| 90 | One-liners | IMPLEMENTED_AND_TESTED | `tgx.workflow(task="...", fast_mode=True)` | — | `TestGroup90Quickstart` |
| 91 | WorkflowResult.to_dict | IMPLEMENTED_AND_TESTED | `result.to_dict()` | — | `TestGroup91WorkflowResult` |
| 92 | describe / summary | IMPLEMENTED_AND_TESTED | `tgx.describe(obj)`, `g.summary()` | `tgx.summary` | `TestGroup92Describe` |
| 93 | validate_graph | IMPLEMENTED_AND_TESTED | `tgx.validate_graph(g, strict=True)` | `assert_tensor_native`, `check_graph_invariants` | `TestGroup93Validate` |
| 94 | reproducible context | IMPLEMENTED_AND_TESTED | `with tgx.reproducible(seed=42):` | `seeded`, `reproducibility_state` | `TestGroup94Reproducible` |
| 95 | compare / benchmark | IMPLEMENTED_AND_TESTED | `tgx.compare(workflows=[...])` | — | `TestGroup95Compare` |
| 96 | Leakage guard | IMPLEMENTED_AND_TESTED | `tgx.check_leakage(train_mask, val_mask, test_mask)` | `leakage_report` (KG triples) | `TestGroup96Leakage` |
| 97 | Native save/load | IMPLEMENTED_AND_TESTED | `tgx.save(g, path)`, `tgx.load(path)`, `g.save(path)`, `Graph.load(path)` | `save_tgraphx`, `load_tgraphx` | `TestGroup97Serialization` |
| 98 | Dashboard audit | IMPLEMENTED_AND_TESTED | `tgx.audit_run_dir(path)` | `dashboard_audit` | `TestGroup98Dashboard` |
| 99 | Migration aliases | IMPLEMENTED_AND_TESTED | `g.x`, `g.y`, `g.edge_attr`, `g.number_of_nodes()`, `g.number_of_edges()` | (PyG/NetworkX/PyKEEN-style read-only) | `TestGroup99Migration` |
| 100 | Public API registry | IMPLEMENTED_AND_TESTED | `tgx.public_api()`, `tgx.api_status(name)`, `tgx.list_aliases(canonical)` | — | `TestGroup100PublicAPI` |

## Mathematical / tensor-native safety

The v1.4.0 audit enforces these properties through tests:

1. **No silent flattening.** `knn_graph` flattens features ONLY for similarity
   computation; the original tensor is unchanged. `validate_graph` reports the
   actual node-feature rank.
2. **Shape preservation.** `Graph(x=images_rank_4)` preserves rank-4 features
   end-to-end; `g.num_node_features` returns the **product** of per-node dims
   (consistent with PyG semantics for tensor features).
3. **Device preservation.** `Graph.to(device)`, `KnowledgeGraph.to(device)`,
   `GraphBatch.to(device)` all move node_features, edge_index, edge_attr, y,
   masks, and graph_label consistently.
4. **Autograd preservation.** `validate_graph(check_gradients=True)` reports
   `requires_grad`; the save/load round-trip detaches before serialization
   (load returns CPU detached tensors by default — opt in to GPU via
   `map_location="cuda"`).
5. **Leakage prevention.** `build_class_prototypes` requires `train_mask` and
   raises if absent. `check_leakage` detects overlapping splits.
   `leakage_report` detects identical triples across KG splits.

## Rejected shortcuts and helpful errors

| User wrote | Why we reject | What we suggest |
| ---------- | ------------- | --------------- |
| `Graph(x=x, y=y_graphlevel)` with mismatched shape | Could confuse node vs graph labels | "Provide at most one of y / labels / node_labels" |
| `Graph(node_features=x, x=x2)` with `x is not node_features` | Ambiguous which is correct | "Provide node_features or x, not both" |
| `tgx.workflow(task="nope")` | Unknown task | Lists supported tasks + closest match |
| `tgx.api_status("Graff")` | Typo of `Graph` | `KeyError` with `"Closest match: 'Graph'"` |
| `tgx.knn_graph(x, k=999)` for small N | Mathematically impossible | "k=N too large for N=N nodes" |
| `tgx.knn_graph(x, metric="nope")` | Unsupported metric | Lists supported metrics |
| `tgx.image_to_patch_graph(image, patch_size=8)` with H not divisible | Loss of info if silent-pad | "Image size HxW not divisible by patch_size; use image_to_patches with padding" |
| `Graph.from_edges("bad")` | Cannot parse | Suggests `[(src,dst), ...]` or `[E,2]` tensor |
| `Graph.from_adjacency(non_square)` | Cannot make sense of asymmetric adj | "Dense adjacency must be square [N, N]" |
| `tgx.save(42, "x.tgx")` | Unsupported type | "Supported: tgraphx.Graph, tgraphx.KnowledgeGraph" |
| `tgx.load("not_a_tgx_file")` | Corrupted bundle | `TGraphXSerializationError` with format hint |

## Compatibility promise

- All v1.3.x syntax remains valid in v1.4.0 (3262 tests pass, no regressions).
- New aliases are marked **beta** in [docs/api_stability.md](api_stability.md);
  signatures may evolve in v1.4.x patch releases but not be removed.
- Canonical APIs (Graph, GraphBatch, ConvMessagePassing, NeighborLoader,
  set_seed, ...) are **stable** and unchanged.
- No silent semantics changes. Every alias goes through the same input
  validation as its canonical name.
