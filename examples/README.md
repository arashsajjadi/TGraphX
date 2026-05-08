# Examples gallery

Every example in this directory is runnable on CPU in well under a minute unless
noted. The `examples/run_all_fast_examples.py` script runs them all.

## Core layer demos

| Script | What it shows |
|--------|--------------|
| `minimal_spatial_message_passing.py` | Spatial GNN forward + backward |
| `minimal_graph_classifier.py` | Short training loop |
| `tensor_gat_minimal.py` | Multi-head GAT with attention sums-to-1 check |
| `tensor_graphsage_minimal.py` | GraphSAGE mean/max/edge-feature variants |
| `custom_message_passing.py` | Subclassing `TensorMessagePassingLayer` |
| `gat_chunking_demo.py` | Chunked GAT parity check |
| `graph_transformer_demo.py` | Vector `GraphTransformerLayer` |

## Dataset + patch demos

| Script | What it shows |
|--------|--------------|
| `datasets_quickstart.py` | Registry `list_datasets` + get synthetic |
| `synthetic_datasets_demo.py` | All 7 native synthetic datasets |
| `image_patch_graph.py` | 2-D image → patches → GAT |
| `volume_patch_graph.py` | 3-D volume → patches → GAT |
| `gnn_family_with_graph_builders.py` | Grid/kNN/radius builders |
| `directed_vs_undirected_graphs.py` | Self-loops and directionality |
| `weighted_edges.py` | Edge-weight API |
| `tensor_edge_features.py` | Spatial edge features |

## Training and evaluation demos

| Script | What it shows |
|--------|--------------|
| `01_vector_node_classification.py` | Factory, node classification |
| `02_spatial_graph_classification.py` | Factory, graph classification |
| `03_volumetric_graph_classification.py` | Factory, 3-D volumetric |
| `04_config_based_model.py` | `build_model_from_config` |
| `05_edge_prediction.py` | `EdgePredictor` |
| `tiny_overfit_tensor_gat.py` | Trainability sanity |
| `tiny_overfit_edge_features.py` | Vector edge feature dependency |
| `gradient_sanity_stack.py` | 8-layer deep-stack gradients |
| `training_minimal_fit.py` | `fit()` helper |
| `training_with_csvlogger.py` | CSV logging |
| `training_with_tensorboard.py` | TensorBoard (optional) |
| `training_with_dashboard.py` | Dashboard-compatible run |
| `checkpoint_save_load.py` | Checkpoint roundtrip |
| `transforms_metrics_demo.py` | `Compose` + classification report |

## Model zoo (v0.3.0)

| Script | What it shows |
|--------|--------------|
| `model_zoo_demo.py` | `GCNConv`, `GATv2Conv`, `APPNP`, pooling |

## Sampling demos

| Script | What it shows |
|--------|--------------|
| `sampling_demo_v028.py` | Random walk + hetero + temporal sampling |
| `neighbor_sampling_demo.py` | `NeighborSamplerLoader` |

## Hetero and temporal demos

| Script | What it shows |
|--------|--------------|
| `hetero_graph_batch_demo.py` | `HeteroGraphBatch` |
| `hetero_graph_classifier_demo.py` | `HeteroGraphClassifier` |
| `temporal_graph_batch_demo.py` | `TemporalGraphBatch` |
| `temporal_graph_classifier_demo.py` | `TemporalGraphClassifier` |

## Performance and hardware

| Script | What it shows |
|--------|--------------|
| `memory_report.py` | `env_report` + memory estimates |
| `mixed_precision_inference.py` | AMP (bfloat16/float16) forward demo |
| `torch_compile_benchmark.py` | `torch.compile` comparison |
| `v024_new_features.py` | GAT chunked forward + attention modes |

## Experiment manager (v0.3.0)

| Script | What it shows |
|--------|--------------|
| `experiment_config_quickstart.py` | Run from a YAML config |
| `configs/synthetic_patch_graph.yaml` | Example patch-graph config |
| `configs/node_classification.yaml` | Node classification config |
| `configs/grid_sweep.yaml` | Grid + multi-seed sweep config |

## Explainability (v0.3.0)

| Script | What it shows |
|--------|--------------|
| `explainability_saliency_demo.py` | Saliency + IG + perturbation |
| `explainability_attention_demo.py` | Attention → per-edge scores |

## End-to-end validation scripts

| Script | Purpose |
|--------|---------|
| `dashboard_artifact_validation.py` | Writes all metadata files, exports HTML |
| `device_validation.py` | CPU/CUDA/MPS layer smoke with JSON report |
| `experiment_end_to_end_validation.py` | Train + checkpoint + resume |
| `explainability_end_to_end_validation.py` | Train + explain + export |

## Public dataset scripts (manual, opt-in)

These scripts require `--download` for any network access and skip cleanly when
optional dependencies are missing.  They are **not** run in default CI.

```
examples/public_datasets/
├── README.md
├── fake_torchvision_patch_smoke.py   ← CI-safe (FakeData, no download)
├── mnist_patch_smoke.py              ← requires --download
├── pyg_cora_smoke.py                 ← requires torch-geometric + --download
├── ogb_arxiv_smoke.py                ← requires ogb + --download
└── dgl_cora_smoke.py                 ← requires dgl + --download
```

See [docs/public_dataset_validation.md](../docs/public_dataset_validation.md).
