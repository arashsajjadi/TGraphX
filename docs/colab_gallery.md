# TGraphX Notebook Gallery

A curated index of CPU-runnable, educational TGraphX notebooks focused on
tensor-native graph intelligence workflows.

> **Note on links.** Notebooks 01–07 are committed to the repository and can be run locally.
> Expanded notebooks (08+) are provided as Google Drive files for download.
> Verified one-click "Open in Colab" links will be added after maintainer-side Colab testing.

> All notebooks use synthetic or toy data to keep examples reproducible and lightweight.

---

## Committed Notebooks (v1.3, in repository)

These 7 notebooks ship with TGraphX, can be run locally, and have been validated:

| # | File | Scenario | Subsystem |
|---|------|----------|-----------|
| 01 | [notebooks/01_easy_tensor_node_classification.ipynb](../notebooks/01_easy_tensor_node_classification.ipynb) | Easy Mode tensor node classification | `tgraphx.easy` |
| 02 | [notebooks/02_image_patch_tensor_graph.ipynb](../notebooks/02_image_patch_tensor_graph.ipynb) | Image-patch graph, tensor-vs-flatten comparison | `ConvMessagePassing` |
| 03 | [notebooks/03_kg_completion_rescal_simple_hpo.ipynb](../notebooks/03_kg_completion_rescal_simple_hpo.ipynb) | KG completion with RESCAL, TransE, SimplE + HPO | `tgraphx.kg` |
| 04 | [notebooks/04_graph_generation_and_optimization.ipynb](../notebooks/04_graph_generation_and_optimization.ipynb) | Graph generation + evolutionary optimization | `tgraphx.generation` |
| 05 | [notebooks/05_graph_rl_coloring_and_navigation.ipynb](../notebooks/05_graph_rl_coloring_and_navigation.ipynb) | Graph RL with callbacks and CSV logging | `tgraphx.rl` |
| 06 | [notebooks/06_graph_io_roundtrip.ipynb](../notebooks/06_graph_io_roundtrip.ipynb) | GraphML write/read round-trip | `tgraphx.io` |
| 07 | [notebooks/07_benchmark_suite_and_dashboard.ipynb](../notebooks/07_benchmark_suite_and_dashboard.ipynb) | v1.3 benchmark suite and dashboard artifacts | `benchmarks/` |

Regenerate locally: `python tools/generate_notebooks.py`

---

## Expanded Notebook Gallery (Google Drive)

These notebooks are available as uploaded Google Drive files.
Download and open in Jupyter or Google Colab.

### 1 — Quick Start / Easy Mode

| # | Notebook | What it demonstrates | Link |
|---|----------|----------------------|------|
| 01 | 01_easy_tensor_node_classification | Train on `[C,H,W]` node features with Easy Mode | [Google Drive notebook](https://drive.google.com/file/d/1C-vydQXnn9LrYhx5hZDQl6H601itnbGp/view?usp=sharing) |

### 2 — Tensor-Native Core Identity

| # | Notebook | What it demonstrates | Link |
|---|----------|----------------------|------|
| 02 | 02_image_patch_tensor_graph_core_identity | Tensor-aware model vs flatten baseline; shape preservation | [Google Drive notebook](https://drive.google.com/file/d/1uPXV1Ybmw0iR8-5A57Ig6HM_Y10vpXM5/view?usp=sharing) |
| 03 | 03_tensor_vs_flatten_benchmark_story | Runtime, parameter count, gradient health comparison | [Google Drive notebook](https://drive.google.com/file/d/1XsHLO1ktivQuKlr0uZFBA5l9wK9KjU9Y/view?usp=sharing) |
| 04 | 04_edge_tensor_features_message_passing | Edge scalar/vector attributes in aggregation | [Google Drive notebook](https://drive.google.com/file/d/1hfsFMNu891m22SLjZK3xCPDfp2AlG8mU/view?usp=sharing) |
| 05 | 05_graph_level_tensor_state_classification | `graph_features` (input) vs `graph_label` (target) | [Google Drive notebook](https://drive.google.com/file/d/1zbVydvlb3mFjRQ2iVjrYLPb6QMwRQ1It/view?usp=sharing) |

### 3 — Sampling and Mini-Batch Workflows

| # | Notebook | What it demonstrates | Link |
|---|----------|----------------------|------|
| 06 | 06_neighborloader_seed_node_loss | `batch.seed_y`, `batch.seed_logits()`; why `logits[:batch_size]` is unsafe | [Google Drive notebook](https://drive.google.com/file/d/16h2S_6tOJW_Z51wX167rARhgk8BaVBWQ/view?usp=sharing) |
| 07 | 07_sampling_benchmark_neighborloader | NeighborLoader throughput; scope and limitations | [Google Drive notebook](https://drive.google.com/file/d/1CPviNy3vx_lqjHQR-y6n4nyLLqc61nb7/view?usp=sharing) |
| 08 | 08_graphsaint_cluster_gcn_smoke | GraphSAINT node sampling and Cluster-GCN partitioning | [Google Drive notebook](https://drive.google.com/file/d/1RnUZxEbd6s5P9sSQiQ9aTn6-ZhC-Ceg3/view?usp=sharing) |

### 4 — Knowledge Graphs

| # | Notebook | What it demonstrates | Link |
|---|----------|----------------------|------|
| 09 | 09_kg_completion_transe_rescal_simple | KG embedding models; filtered MRR / Hits@K | [Google Drive notebook](https://drive.google.com/file/d/1QlCNZg2U0HJ6I6M4V8qXArKEwweZOjqn/view?usp=sharing) |
| 10 | 10_kg_hpo_grid_random_search | `run_kg_hpo()` across model/hyperparameter combinations | [Google Drive notebook](https://drive.google.com/file/d/12mqfQvcUGCm3UhfAh0PGbCMvpNBgktmF/view?usp=sharing) |
| 11 | 11_multimodal_kg_tensor_features | Feature-aware KG scoring with entity embeddings | [Google Drive notebook](https://drive.google.com/file/d/1WVm__OQyWItg3SRozTUWeULs8QaYBUu1/view?usp=sharing) |
| 12 | 12_kg_filtered_ranking_explained | Hand-checkable MRR / Hits@K on a tiny KG | [Google Drive notebook](https://drive.google.com/file/d/1LgyDOIXb7iH70a5PxMHO8j6bYiHNS1GK/view?usp=sharing) |

### 5 — Graph Generation

| # | Notebook | What it demonstrates | Link |
|---|----------|----------------------|------|
| 13 | 13_graph_generation_metrics | ER/BA/SBM generation; validity, uniqueness, statistics | [Google Drive notebook](https://drive.google.com/file/d/1Q8358qYmw80SBr-fXFmkTg1tcRhaUg16/view?usp=sharing) |

---

## Coming Next — Draft Queue

| Planned notebook | Theme |
|-----------------|-------|
| graph_generation_evolutionary_optimization | Evolve graph structure toward target properties |
| graph_rl_coloring_with_callbacks | Graph coloring with EarlyStoppingCallback |
| graph_rl_maxcut_or_navigation | MaxCut / navigation sequential decision-making |
| rl_callbacks_logging_dashboard_artifacts | RL training logs and dashboard artifacts |
| graph_io_roundtrip | GraphML round-trip and tensor-feature limitations |
| graph_mining_motifs_and_cliques | Motifs, cliques, structural summaries |
| graph_mining_kernels_wl_similarity | WL graph kernel similarity |
| dashboard_easy_mode_artifacts | Write dashboard artifacts from Easy Mode |
| benchmark_suite_v13 | Run v1.3 benchmark suite |
| reproducibility_and_seed_control | Deterministic workflows with `set_seed` |
| end_to_end_research_workflow | Complete pipeline: data → model → metrics → artifacts |
| limitations_and_roadmap_honest_demo | Honest capabilities and roadmap |

---

## Tutorial Scripts (no Jupyter required)

| Script | What it demonstrates |
|--------|---------------------|
| [tutorials/tensor_node_classification_neighbor_loader.py](../tutorials/tensor_node_classification_neighbor_loader.py) | Canonical tensor node classification with NeighborLoader |
| [tutorials/graph_generation_quickstart.py](../tutorials/graph_generation_quickstart.py) | Graph generation quickstart |
| [tutorials/evolutionary_optimization_quickstart.py](../tutorials/evolutionary_optimization_quickstart.py) | Evolutionary optimization (GA/SA/NSGA-II) |
| [tutorials/graph_rl_quickstart.py](../tutorials/graph_rl_quickstart.py) | Graph RL comparison |
| [tutorials/real_dataset_cora_node_classification.py](../tutorials/real_dataset_cora_node_classification.py) | Cora node classification (optional PyG) |
| [tutorials/image_patch_tensor_graph_demo.py](../tutorials/image_patch_tensor_graph_demo.py) | Image-patch tensor graph demo |
| [tutorials/kg_benchmark_quickstart.py](../tutorials/kg_benchmark_quickstart.py) | KG benchmark (TransE/DistMult/RESCAL/SimplE + MRR) |

---

## Running on Colab

```python
!pip install -q tgraphx
!python -m tgraphx doctor
```
