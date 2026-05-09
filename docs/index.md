# TGraphX documentation

TGraphX is a tensor-aware graph learning and graph mining framework for
PyTorch. It supports vector/spatial/volumetric node features, graph neural
networks, scalable samplers (GraphSAINT, Cluster-GCN), knowledge graphs,
hypergraphs, temporal and heterogeneous graph learning, a local dashboard,
sklearn-like estimators, and benchmark tooling — all in one package.

## Getting started

| Step | Doc |
|------|-----|
| Install + first steps | [getting_started.md](getting_started.md) |
| 10-minute tutorial path | [tutorials.md](tutorials.md) |

## Core graph data model

| Topic | Doc |
|-------|-----|
| Core graph objects | [graph_basics.md](graph_basics.md) |
| Tensor features | [edge_features.md](edge_features.md) |
| Graph builders | [graph_builders.md](graph_builders.md) |
| Batching | [batching.md](batching.md) |
| Weighted edges | [weighted_edges.md](weighted_edges.md) |

## GNN layers and neural mining

| Topic | Doc |
|-------|-----|
| Spatial tensor GNN layers | [spatial_tensor_gnn.md](spatial_tensor_gnn.md) |
| Vector GNN layers + model zoo | [vector_gnn.md](vector_gnn.md) |
| 3-D volumetric support | [volumetric_3d.md](volumetric_3d.md) |
| Heterogeneous graph layers (RGCN / HAN / HGT) | [hetero_gnns.md](hetero_gnns.md) |
| Neural graph mining (VGAE / GAE / prototypes) | [neural_graph_mining.md](neural_graph_mining.md) |
| Explainability | [explainability.md](explainability.md) |

## Sampling and loaders

| Topic | Doc |
|-------|-----|
| GraphSAINT samplers | [graphsaint.md](graphsaint.md) |
| Cluster-GCN | [cluster_gcn.md](cluster_gcn.md) |
| Negative sampling | [negative_sampling.md](negative_sampling.md) |
| Feature store | [feature_store.md](feature_store.md) |
| Sparse backend | [backends.md](backends.md) |

## Knowledge graphs, hypergraphs, temporal, heterogeneous

| Topic | Doc |
|-------|-----|
| Knowledge graph overview | [knowledge_graphs.md](knowledge_graphs.md) |
| KG data model | [kg_evaluation.md](kg_evaluation.md) |
| KG models (TransE, DistMult, ComplEx, RotatE) | [kg_models.md](kg_models.md) |
| Filtered ranking evaluation | [kg_evaluation.md](kg_evaluation.md) |
| KG training pipeline | [kg_training.md](kg_training.md) |
| KG + GNN integration | [kg_gnn_integration.md](kg_gnn_integration.md) |
| Temporal knowledge graphs | [temporal_knowledge_graphs.md](temporal_knowledge_graphs.md) |
| KG reasoning (paths, rules, constraints) | [kg_reasoning.md](kg_reasoning.md) |
| KG datasets (synthetic) | [kg_datasets.md](kg_datasets.md) |
| KG benchmarks | [kg_benchmarks.md](kg_benchmarks.md) |
| Multimodal tensor KG features | [kg_multimodal_tensor_features.md](kg_multimodal_tensor_features.md) |
| Temporal graph learning (TGN / TGAT) | [temporal_graph_learning.md](temporal_graph_learning.md) |
| Temporal utilities | [temporal.md](temporal.md) |
| Heterogeneous graphs | [hetero_gnns.md](hetero_gnns.md) |
| Graph algorithms | [graph_algorithms.md](graph_algorithms.md) |
| Graph mining and pattern recognition | [graph_mining.md](graph_mining.md) |

## Training, experiments, dashboard

| Topic | Doc |
|-------|-----|
| Training utilities | [training_utilities.md](training_utilities.md) |
| Factories (`make_layer`, `build_model`) | [factories.md](factories.md) |
| Experiment manager | [experiments.md](experiments.md) |
| Dashboard | [dashboard.md](dashboard.md) |
| Reproducibility | [reproducibility.md](reproducibility.md) |
| Distributed training | [distributed_training.md](distributed_training.md) |
| sklearn-like API | [sklearn_api.md](sklearn_api.md) |

## Datasets, transforms, benchmarks

| Topic | Doc |
|-------|-----|
| Datasets | [datasets.md](datasets.md) |
| Transforms | [transforms.md](transforms.md) |
| Metrics | [metrics.md](metrics.md) |
| Benchmarks | [benchmarks.md](benchmarks.md) |
| OGB / TGB integration | [ogb_tgb_integration.md](ogb_tgb_integration.md) |
| Public dataset validation | [public_dataset_validation.md](public_dataset_validation.md) |
| Device validation | [device_validation.md](device_validation.md) |
| Benchmark protocol | [benchmark_protocol.md](benchmark_protocol.md) |
| Plotting | [plotting.md](plotting.md) |

## API reference and policies

| Topic | Doc |
|-------|-----|
| API stability | [api_stability.md](api_stability.md) |
| Limitations | [limitations.md](limitations.md) |
| Roadmap | [roadmap.md](roadmap.md) |
| Performance | [performance.md](performance.md) |
| Interoperability (PyG/DGL/OGB) | [comparison.md](comparison.md) |
| Deprecation policy | [deprecation_policy.md](deprecation_policy.md) |
| Experimental policy | [experimental_policy.md](experimental_policy.md) |
| Release checklist | [release_checklist.md](release_checklist.md) |
| Dataset license policy | [dataset_license_policy.md](dataset_license_policy.md) |
| Architecture | [architecture.md](architecture.md) |
| Migration guide | [migration_v0_2_to_v0_3.md](migration_v0_2_to_v0_3.md) |

## Examples

See [examples/README.md](../examples/README.md) for a gallery of all
runnable demos and their requirements.
