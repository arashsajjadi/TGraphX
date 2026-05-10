# TGraphX Notebook Gallery

A curated index of CPU-runnable, educational TGraphX notebooks focused on
tensor-native graph intelligence workflows.

> **Note on links.** Notebooks are provided as Google Drive files for download.
> Open the file in Google Drive and choose "Open with Google Colaboratory", or
> download and open with local Jupyter.
> All notebooks are CPU-friendly and use synthetic or toy data unless noted.

---

## Available Notebooks

### 1 — Quick Start / Easy Mode

| # | Notebook | What it demonstrates | Link |
|---|----------|----------------------|------|
| 01 | 01_easy_tensor_node_classification | Train on `[C,H,W]` node features with Easy Mode; no boilerplate | [Google Drive](https://drive.google.com/file/d/1C-vydQXnn9LrYhx5hZDQl6H601itnbGp/view?usp=sharing) |
| 26 | 26_low_level_pytorch_escape_hatch | From Easy Mode result objects to raw PyTorch control | [Google Drive](https://drive.google.com/file/d/1c8A1-_ZoImnmv4NgIGMDibD-x-Zm9nM_/view?usp=sharing) |
| 27 | 27_custom_tensor_projector_workflow | Custom GNN with spatial pooling and classifier head | [Google Drive](https://drive.google.com/file/d/1DAVJ-dm6uP4vmExYDOp9hKkrE6SGB7vl/view?usp=sharing) |
| 28 | 28_colab_install_and_doctor | Install TGraphX in Colab and verify with `tgraphx doctor` | [Google Drive](https://drive.google.com/file/d/1rHuhQKwNkZpH3_gj46xq1vwdvzutQd_9/view?usp=sharing) |
| 29 | 29_end_to_end_research_workflow | Complete pipeline: data → model → train → metrics → artifacts | [Google Drive](https://drive.google.com/file/d/14y8rxwk8ajepynjSV2wvEbs7QQLLqXjB/view?usp=sharing) |
| 23 | 23_dashboard_easy_mode_artifacts | Write dashboard artifacts from Easy Mode; view locally | [Google Drive](https://drive.google.com/file/d/1LgJ5JJNNmoAzJQUamAkFg8G009sLmS3L/view?usp=sharing) |

### 2 — Tensor-Native Core Identity

| # | Notebook | What it demonstrates | Link |
|---|----------|----------------------|------|
| 02 | 02_image_patch_tensor_graph_core_identity | Tensor-aware model vs flatten baseline; `[C,H,W]` preserved | [Google Drive](https://drive.google.com/file/d/1uPXV1Ybmw0iR8-5A57Ig6HM_Y10vpXM5/view?usp=sharing) |
| 03 | 03_tensor_vs_flatten_benchmark_story | Runtime, parameter count, gradient health comparison | [Google Drive](https://drive.google.com/file/d/1XsHLO1ktivQuKlr0uZFBA5l9wK9KjU9Y/view?usp=sharing) |
| 04 | 04_edge_tensor_features_message_passing | Edge scalar/vector attributes and their effect on aggregation | [Google Drive](https://drive.google.com/file/d/1hfsFMNu891m22SLjZK3xCPDfp2AlG8mU/view?usp=sharing) |
| 05 | 05_graph_level_tensor_state_classification | `graph_features` (input) vs `graph_label` (target) as distinct fields | [Google Drive](https://drive.google.com/file/d/1zbVydvlb3mFjRQ2iVjrYLPb6QMwRQ1It/view?usp=sharing) |

### 3 — Sampling and Mini-Batch Workflows

| # | Notebook | What it demonstrates | Link |
|---|----------|----------------------|------|
| 06 | 06_neighborloader_seed_node_loss | `batch.seed_y`, `batch.seed_logits()`; why `logits[:batch_size]` is unsafe | [Google Drive](https://drive.google.com/file/d/16h2S_6tOJW_Z51wX167rARhgk8BaVBWQ/view?usp=sharing) |
| 07 | 07_sampling_benchmark_neighborloader | NeighborLoader throughput; scope and honest limitations | [Google Drive](https://drive.google.com/file/d/1CPviNy3vx_lqjHQR-y6n4nyLLqc61nb7/view?usp=sharing) |
| 08 | 08_graphsaint_cluster_gcn_smoke | GraphSAINT node sampling and Cluster-GCN partitioning foundations | [Google Drive](https://drive.google.com/file/d/1RnUZxEbd6s5P9sSQiQ9aTn6-ZhC-Ceg3/view?usp=sharing) |

### 4 — Knowledge Graphs

| # | Notebook | What it demonstrates | Link |
|---|----------|----------------------|------|
| 09 | 09_kg_completion_transe_rescal_simple | KG embedding models; filtered MRR / Hits@K | [Google Drive](https://drive.google.com/file/d/1QlCNZg2U0HJ6I6M4V8qXArKEwweZOjqn/view?usp=sharing) |
| 10 | 10_kg_hpo_grid_random_search | `run_kg_hpo()` grid/random search across model/hyperparameter combinations | [Google Drive](https://drive.google.com/file/d/12mqfQvcUGCm3UhfAh0PGbCMvpNBgktmF/view?usp=sharing) |
| 11 | 11_multimodal_kg_tensor_features | Feature-aware KG scoring with entity embeddings | [Google Drive](https://drive.google.com/file/d/1WVm__OQyWItg3SRozTUWeULs8QaYBUu1/view?usp=sharing) |
| 12 | 12_kg_filtered_ranking_explained | Hand-checkable MRR / Hits@K on a tiny KG | [Google Drive](https://drive.google.com/file/d/1LgyDOIXb7iH70a5PxMHO8j6bYiHNS1GK/view?usp=sharing) |

### 5 — Graph Generation

| # | Notebook | What it demonstrates | Link |
|---|----------|----------------------|------|
| 13 | 13_graph_generation_metrics | ER/BA/SBM generation; validity, uniqueness, structural statistics | [Google Drive](https://drive.google.com/file/d/1Q8358qYmw80SBr-fXFmkTg1tcRhaUg16/view?usp=sharing) |

### 6 — Graph Reinforcement Learning

| # | Notebook | What it demonstrates | Link |
|---|----------|----------------------|------|
| 15 | 15_graph_rl_coloring_with_callbacks | Graph coloring with EarlyStoppingCallback and CSVLoggerCallback | [Google Drive](https://drive.google.com/file/d/1TNPIMbADQtglWfTz55MGHAWYLS3d8UvV/view?usp=sharing) |
| 16 | 16_graph_rl_maxcut_or_navigation | MaxCut / navigation sequential decision-making | [Google Drive](https://drive.google.com/file/d/1xsmgihWMTY07XMDIesAHHKfwoli_gDWT/view?usp=sharing) |
| 17 | 17_rl_callbacks_logging_dashboard_artifacts | RL training logs and dashboard-compatible artifacts | [Google Drive](https://drive.google.com/file/d/1RabWNOWvZVmg5PIT8t_cQ3UgRUlPckzZ/view?usp=sharing) |

### 7 — Graph IO

| # | Notebook | What it demonstrates | Link |
|---|----------|----------------------|------|
| 18 | 18_graphml_io_roundtrip | GraphML read/write round-trip; tensor-feature limitations explained | [Google Drive](https://drive.google.com/file/d/11Ul2v5KVYkrVFOhoeSZkE6Y8HG1qcgu5/view?usp=sharing) |

### 8 — Reproducibility and Workflows

| # | Notebook | What it demonstrates | Link |
|---|----------|----------------------|------|
| 30 | 30_limitations_and_roadmap_honest_demo | Honest capabilities and what is on the roadmap | Draft pending link |

---

## Generating Notebooks Locally

Notebook source files are **not** committed to this repository.
Generate them locally with:

```bash
python tools/generate_notebooks.py   # creates notebooks/ (gitignored)
python tools/validate_notebooks.py   # validates structure
```

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

---

## Generating Local Notebooks

```bash
python tools/generate_notebooks.py     # regenerates notebooks/ (tracked in git)
python tools/generate_colab_drafts.py  # creates colab_drafts/ (gitignored, for review)
```
