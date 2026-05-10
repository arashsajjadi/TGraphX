# API Stability Contract — TGraphX v1.0.1+

This document defines the stability contract for all public TGraphX APIs.

**Policy summary:**
- **Stable** APIs will not have breaking changes in any v1.x release.
- **Beta** APIs are stable within a minor version (v1.0.x) and may only expand, not break.
- **Experimental** APIs may have surface changes between minor releases. Documented behavior is preserved; signatures and defaults may evolve.
- **Optional** APIs require an optional dependency or explicit `download=True`.

See [deprecation_policy.md](deprecation_policy.md) for the full deprecation procedure.

---

## Stable (v1.0+)

Breaking changes require a major version bump (v2.x) or explicit deprecation cycle.

### Core data objects
- `tgraphx.Graph`, `tgraphx.GraphBatch`
  - All existing constructor parameters remain valid.
  - New keyword-only aliases added in v1.0.1: `y=`, `labels=`, `edge_attr=`, `train_mask=`, `val_mask=`, `test_mask=`.
  - `graph_features=` added in v1.0.2 as a **distinct** graph-level input feature field (not aliased to `graph_label`).
  - Properties: `.x`, `.y`, `.labels`, `.edge_attr`, `.num_node_features`, `.num_classes`, `.train_mask`, `.val_mask`, `.test_mask`.
  - Methods: `.has_labels()`, `.get_labels()`, `.with_labels()`, `.to()`, `.clone()`, `.validate()`.
- `tgraphx.core.graph_utils` — all utility functions.
- `tgraphx.GraphDataset`, `tgraphx.GraphDataLoader`

### Tensor-aware GNN layers
- `ConvMessagePassing`, `TensorGATLayer`, `TensorGraphSAGELayer`, `TensorGINLayer`
- `LinearMessagePassing`, `TensorMessagePassingLayer`
- `AttentionMessagePassing` (legacy, preserved for backward compat)

### Vector model zoo
- `GCNConv`, `GATv2Conv`, `APPNP`
- `global_mean_pool`, `global_sum_pool`, `global_max_pool`

### Graph builders and patch helpers
- All functions in `tgraphx.graph_builders`

### Sampling utilities
- All functions in `tgraphx.sampling`
- `tgraphx.sampling_loaders.SubgraphDataLoader`, `NeighborSamplerLoader`

### Factories
- `make_layer`, `build_model`, `build_model_from_config`

### Model classes
- `GraphClassifier`, `NodeClassifier`, `EdgePredictor`, `NodeRegressor`, `GraphRegressor`

### Training utilities
- `train_epoch`, `evaluate`, `fit`, `set_seed`
- `save_checkpoint`, `load_checkpoint`, `count_parameters`
- `accuracy`, `mean_absolute_error`, `mean_squared_error`

### Logging / tracking
- `CSVLogger`, `TensorBoardLogger`, `MLflowLogger`
- All `write_*` metadata helpers in `tgraphx.tracking`

### Datasets
- `get_dataset`, `list_datasets`, `dataset_info`, `cache_summary`, `clear_cache`
- All synthetic dataset classes
- `ImageFolderPatchGraphDataset`, `VolumeFolderPatchGraphDataset`

### Transforms
- All classes in `tgraphx.transforms`

### Metrics
- All functions in `tgraphx.metrics`

### Dashboard
- Dashboard CLI (`tgraphx-dashboard`)
- `launch_dashboard`, `launch_dashboard_background`, `export_dashboard_html`

### Experiment manager
- `load_config`, `ExperimentConfig`, `Runner`, `GridRunner`
- `EarlyStopping`, `ModelCheckpoint`, `CSVLoggerCallback`, `LearningRateLogger`
- `tgraphx-train`, `tgraphx-grid`, `tgraphx-report` CLI scripts

### Tutorials (content stable)
- `tutorials/graph_generation_quickstart.py`
- `tutorials/evolutionary_optimization_quickstart.py`
- `tutorials/graph_rl_quickstart.py`
- `tutorials/tensor_node_classification_neighbor_loader.py` *(new in v1.0.1)*

---

## Beta (v1.0.1+)

Tested and documented. API may expand but not break within the v1.x series.

### Loaders & batch objects
- `NeighborLoader` — yields `GraphMiniBatch` objects. Legacy tuple unpacking preserved via `__iter__`.
- `GraphMiniBatch` *(new in v1.0.1)* — ergonomic batch wrapper; all listed attributes and methods are stable.
  - Attributes: `node_features`, `x`, `edge_index`, `edge_features`, `edge_attr`, `edge_weight`, `y`, `labels`, `seed_y`, `seed_labels`, `seed_node_ids`, `seed_local_indices`, `input_nodes`, `batch_size`, `num_nodes`, `num_edges`, `metadata`.
  - Methods: `seed_logits(logits)`, `all_logits(logits)`, `loss(logits)`, `to(device)`, `as_tuple()`.
- `LinkNeighborLoader`, `GraphLoader`
- `map_global_to_local(global_ids, sampled_ids)` *(new in v1.0.1)*
- `seed_logits(logits, batch)` *(new in v1.0.1)*
- `make_neighbor_loader`, `make_link_loader`, `make_graph_loader`
- `fetch_features_for_subgraph`
- `InMemoryFeatureStore`, `MemmapFeatureStore`

### Sampling & partitioning
- `hetero_induced_subgraph`, `hetero_neighbor_sample`
- `temporal_window_sample`, `temporal_window_sample_batch`
- `negative_sampling`, `structured_negative_sampling`, `batched_negative_sampling`, `hard_negative_sampling`
- `GraphSAINTNodeSampler`, `GraphSAINTEdgeSampler`, `GraphSAINTRandomWalkSampler`, `GraphSAINTLoader`, `estimate_norm_coefficients`
- `RandomBalancedPartitioner`, `BFSPartitioner`, `ConnectedComponentPartitioner`, `SpectralPartitioner`, `ClusterLoader`

### Graph algorithms
- All functions in `tgraphx.algorithms` (BFS/DFS, shortest paths, MST, max-flow, matching, coloring)

### Graph mining
- `tgraphx.mining.structural`, `.link_prediction`, `.motifs`, `.kernels`, `.similarity`, `.communities`, `.random_walk`, `.centrality`, `.spectral`, `.paths`
- `analyze_graph`, `graph_mining_report`

### Knowledge graphs (core)
- `KnowledgeGraph`, `TemporalKnowledgeGraph` data model
- `UniformNegativeSampler`, `BernoulliNegativeSampler`, `FilteredNegativeSampler`, `TypedNegativeSampler`
- `KGEvaluator`, `evaluate_filtered_ranking`
- `TransEModel`, `DistMultModel`, `ComplExModel`, `RotatEModel`
- `KGTrainer`, `KGTrainingConfig`
- `list_kg_models()` *(new in v1.0.1)*
- `KnowledgeGraph.from_hrt(heads, relations, tails, ...)` *(new in v1.0.3)* — classmethod for users with separate h/r/t tensors (the existing `from_triples` accepts tuple lists or `[N_t, 3]` tensors)

### Classical graph generation
- `FeatureAwareERGraph`, `FeatureAwareBAGraph`, `TemporalEvolvingGraph`, `TypedGeneratedGraph`, `AnomalyInjectedGraph`, `MotifInjectedGraph`
- Generation metrics: `validity_score`, `uniqueness_score`, `novelty_score`, `diversity_score`
- `run_graph_generation`, `list_graph_generation_methods`, `GenerationResult`

### Evolutionary optimization (core)
- `GraphGenome` — tensor-aware graph genome with full mutation/crossover support
- Mutation operators: `mutate_add_node`, `mutate_add_edge`, `mutate_remove_edge`, `mutate_node_feature`, `mutate_edge_weight`
- Crossover operators: `edge_set_crossover`, `node_crossover`, `uniform_crossover`
- Selection strategies: `tournament_selection`, `roulette_wheel_selection`, `rank_selection`, `elitist_selection`, `nsga2_selection`
- `GeneticAlgorithmOptimizer`, `SimulatedAnnealingOptimizer`, `HillClimbingOptimizer`, `RandomSearchOptimizer`
- Pareto utilities: `pareto_dominates`, `non_dominated_sort`, `crowding_distance`, `ParetoFront`, `hypervolume_2d`
- `EvolutionConfig`, `EvolutionResult`
- `run_evolutionary_optimization`, `list_evolutionary_optimizers`, `OptimizationResult`

### Graph RL — baselines
- `RandomPolicy`, `GreedyPolicy` — no-learn baselines with stable API

### Easy Mode *(new in v1.0.1)*
- `tgraphx.easy.synthetic_tensor_node_classification`
- `tgraphx.easy.synthetic_vector_node_classification`
- `tgraphx.easy.synthetic_link_prediction`
- `tgraphx.easy.synthetic_graph_classification`
- `tgraphx.easy.make_tensor_node_classifier`
- `tgraphx.easy.make_vector_node_classifier`
- `tgraphx.easy.train_node_classifier` / `fit_node_classifier`
- `tgraphx.easy.list_tasks`, `list_models`, `list_samplers`, `list_workflows`, `explain_workflow`
- `tgraphx.easy.doctor`, `check_install`, `show_capabilities`
- `tgraphx.easy.EasyConfig`, `EasyResult`
- `tgraphx.easy.TGraphXError`, `TGraphXConfigError`, `TGraphXLabelError`, `TGraphXShapeError`, `TGraphXUnknownNameError`

### CLI tools *(v1.0.1)*
- `python -m tgraphx [doctor|info|capabilities|tasks|models|samplers]`
- `tgraphx-doctor`, `tgraphx-info`

### Explainability
- `node_feature_saliency`, `integrated_gradients`, `edge_perturbation_attribution`, `edge_gradient_attribution`
- `attention_to_edge_scores`, `patch_saliency_to_image_grid`
- `export_explanation_metadata`, `export_edge_scores_csv`, `export_patch_heatmap_json`

### Representation learning
- `Node2Vec`, `DeepWalk` — embedding utilities

### Semi-supervised
- `LabelPropagation`, `RandomNodeSplit`, `RandomLinkSplit`

### Reproducibility
- `set_seed`, `make_generator`, `seed_worker`, `reproducibility_report`, `deterministic_mode`

### OGB / TGB wrappers (Optional)
- `OGBNodeEvaluatorWrapper`, `OGBLinkEvaluatorWrapper`, `OGBGraphEvaluatorWrapper`

---

## Experimental (v1.0.1+)

Correct foundations. API or semantics may evolve in future minor releases. Documented behavior is preserved.

### Neural graph generation
- `VGAEGraphGenerator`, `AutoregressiveEdgeGenerator`
- `GraphTransformerGenerator` (requires PyTorch >= 2.0)

### Graph RL — learning algorithms
- `REINFORCEAgent`, `ActorCriticAgent`, `A2CAgent`, `DQNAgent`, `DoubleDQNAgent`, `PPOAgent`
- `GraphDDPGAgent`, `GraphTD3Agent`, `GraphSACAgent`
- `GraphPolicyNetwork`, `GraphValueNetwork`, `GraphQNetwork`, `GraphActorCriticNetwork`, `MaskedCategoricalPolicy`
- `run_graph_rl`, `list_graph_rl_algorithms`, `make_graph_env`
- All RL environments (navigation, coloring, max-cut, vertex-cover, generation, KG reasoning, shortest path, continuous variants)

### Knowledge graphs — advanced
- `KGRGCNModel`, `kg_to_edge_index` — GNN-based KG completion
- `TemporalKGNegativeSampler`, `evaluate_temporal_filtered_ranking`
- `PathExtractor`, `HornRuleCandidate`, `LogicalConstraintChecker`
- `MultimodalKGModel` and all entity projectors

### Neural graph mining
- `PrototypeMembershipScorer`, `GraphAutoencoderAnomalyDetector`, `GraphPatternClassifier`

### Heterogeneous graphs
- `HeteroGraph`, `HeteroGraphBatch`, `HeteroConv`, `HGTConv`, `HANConv`, `RGCNConv`
- `HeteroNodeClassifier`, `HeteroGraphClassifier`
- `hetero_induced_subgraph`, `hetero_neighbor_sample`

### Temporal graphs
- `TemporalGraphSequence`, `TemporalGraphBatch`
- `TGNMemory`, `TGATConv`, `sinusoidal_time_encoding`, `LearnableTimeEncoding`
- `temporal_readout`, `TemporalGraphClassifier`, `TemporalGraphRegressor`

### Graph autoencoders
- `GAEModel`, `VGAEModel` and supporting utilities

### Hypergraphs
- Incidence matrix utilities, clique/star expansion

### Distributed helpers
- `tgraphx.distributed` — rank-zero utilities, DDP wrapping, shard helpers

### sklearn-like API
- `GraphPipeline`, estimator base classes
- `calibration` — ECE, temperature scaling

### Graph sequence models
- `GraphRNN` / `GraphLSTM` utilities

---

## Legacy / Deprecated

- `AttentionMessagePassing` — kept for backward compatibility; prefer `TensorGATLayer` for true multi-head GAT.

---

## Not yet shipped (roadmap items)

- Per-pixel / per-voxel GAT attention (memory-prohibitive naive form).
- Full automatic multi-GPU training framework.
- Universal arbitrary-rank tensor support across all layers.
- Raw text tokenization in multimodal KG (current: pre-computed embeddings only).

See [limitations.md](limitations.md) and [roadmap.md](roadmap.md).

---

## What justifies promotion from Experimental to Beta

For a component to move from Experimental → Beta:

1. Public API is stable and unlikely to change in v1.x.
2. Unit tests cover shape, gradient, and error paths.
3. Documentation with minimal example, API contract, shape contract, common mistakes.
4. At least one working example or tutorial.
5. No known semantic ambiguity in core behavior.
6. No mandatory optional dependency for basic usage.

For a component to move from Beta → Stable:

All Beta criteria plus:
7. Backward compatibility contract explicitly stated.
8. Breaking changes would require a deprecation cycle.
9. Real-world or broad benchmark validation.
