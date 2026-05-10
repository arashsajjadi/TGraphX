# TGraphX Notebook Gallery

A curated index of CPU-runnable, educational TGraphX notebooks focused on
tensor-native graph intelligence workflows.

> **Note on links.** The notebooks below are provided as Google Drive notebook files.
> Download or open them from Google Drive.
> Verified one-click "Open in Colab" links will be added after maintainer-side Colab testing.

> All notebooks use synthetic or toy data to keep examples reproducible and lightweight.

---

## Available Notebooks

### 1 — Quick Start / Easy Mode

| # | Notebook | What it demonstrates | Link | Status |
|---|----------|----------------------|------|--------|
| 01 | Easy Tensor Node Classification | Train on `[C,H,W]` node features with Easy Mode | [Google Drive notebook](https://drive.google.com/file/d/1C-vydQXnn9LrYhx5hZDQl6H601itnbGp/view?usp=sharing) | Available |

### 2 — Tensor-Native Core Identity

| # | Notebook | What it demonstrates | Link | Status |
|---|----------|----------------------|------|--------|
| 02 | Image-Patch Tensor Graph (Core Identity) | Tensor-aware model vs flatten baseline; shape preservation verified | [Google Drive notebook](https://drive.google.com/file/d/1uPXV1Ybmw0iR8-5A57Ig6HM_Y10vpXM5/view?usp=sharing) | Available |
| 03 | Tensor vs Flatten Benchmark Story | Runtime, parameter count, gradient health comparison | [Google Drive notebook](https://drive.google.com/file/d/1XsHLO1ktivQuKlr0uZFBA5l9wK9KjU9Y/view?usp=sharing) | Available |
| 04 | Edge Tensor Features in Message Passing | Edge scalar/vector attributes in aggregation | [Google Drive notebook](https://drive.google.com/file/d/1hfsFMNu891m22SLjZK3xCPDfp2AlG8mU/view?usp=sharing) | Available |
| 05 | Graph-Level Tensor State Classification | `graph_features` (input) vs `graph_label` (target) as distinct fields | [Google Drive notebook](https://drive.google.com/file/d/1zbVydvlb3mFjRQ2iVjrYLPb6QMwRQ1It/view?usp=sharing) | Available |

### 3 — Sampling and Mini-Batch Workflows

| # | Notebook | What it demonstrates | Link | Status |
|---|----------|----------------------|------|--------|
| 06 | NeighborLoader, GraphMiniBatch, Seed-Node Loss | `batch.seed_y`, `batch.seed_logits(logits)`; why `logits[:batch_size]` is unsafe | [Google Drive notebook](https://drive.google.com/file/d/16h2S_6tOJW_Z51wX167rARhgk8BaVBWQ/view?usp=sharing) | Available |
| 07 | NeighborLoader Throughput Benchmark | Sampling throughput measurement; scope and limitations stated | [Google Drive notebook](https://drive.google.com/file/d/1CPviNy3vx_lqjHQR-y6n4nyLLqc61nb7/view?usp=sharing) | Available |
| 08 | GraphSAINT and Cluster-GCN Foundations | Node sampling and balanced partitioning on small graphs | [Google Drive notebook](https://drive.google.com/file/d/1RnUZxEbd6s5P9sSQiQ9aTn6-ZhC-Ceg3/view?usp=sharing) | Available |

### 4 — Knowledge Graphs

| # | Notebook | What it demonstrates | Link | Status |
|---|----------|----------------------|------|--------|
| 09 | KG Completion: TransE, RESCAL, SimplE | Train and evaluate KG models; filtered MRR / Hits@K | [Google Drive notebook](https://drive.google.com/file/d/1QlCNZg2U0HJ6I6M4V8qXArKEwweZOjqn/view?usp=sharing) | Available |
| 10 | KG HPO: Grid and Random Search | `run_kg_hpo()` across model/hyperparameter combinations | [Google Drive notebook](https://drive.google.com/file/d/12mqfQvcUGCm3UhfAh0PGbCMvpNBgktmF/view?usp=sharing) | Available |
| 11 | Multimodal KG: Entity Tensor Features | Feature-aware KG scoring with visual/user entity embeddings | [Google Drive notebook](https://drive.google.com/file/d/1WVm__OQyWItg3SRozTUWeULs8QaYBUu1/view?usp=sharing) | Available |
| 12 | Filtered Ranking Explained | Hand-checkable MRR / Hits@K on a tiny KG | [Google Drive notebook](https://drive.google.com/file/d/1LgyDOIXb7iH70a5PxMHO8j6bYiHNS1GK/view?usp=sharing) | Available |

### 5 — Graph Generation

| # | Notebook | What it demonstrates | Link | Status |
|---|----------|----------------------|------|--------|
| 13 | Graph Generation Metrics | ER/BA/SBM generation; validity, uniqueness, structural statistics | [Google Drive notebook](https://drive.google.com/file/d/1Q8358qYmw80SBr-fXFmkTg1tcRhaUg16/view?usp=sharing) | Available |

---

## Coming Next — Draft Queue

The following notebooks are being prepared and will be linked once tested and uploaded:

| Planned notebook | Theme |
|-----------------|-------|
| graph_generation_evolutionary_optimization | Evolve graph structure toward target structural properties |
| graph_rl_coloring_with_callbacks | Graph coloring with EarlyStoppingCallback and CSVLoggerCallback |
| graph_rl_maxcut_or_navigation | MaxCut / navigation sequential decision-making |
| rl_callbacks_logging_dashboard_artifacts | RL training logs and dashboard-compatible artifacts |
| graph_io_roundtrip | GraphML read/write round-trip and tensor-feature limitations |
| io_tensor_semantics_warning | Why image-like tensors are not silently serialized |
| graph_mining_motifs_and_cliques | Motifs, cliques, and structural summaries |
| graph_mining_kernels_wl_similarity | WL graph kernel similarity |
| structural_roles_concept_demo | Structural role intuition |
| dashboard_easy_mode_artifacts | Write dashboard artifacts from Easy Mode |
| benchmark_suite_v13 | Run v1.3 benchmark suite and read the JSON results |
| reproducibility_and_seed_control | Deterministic workflows with `set_seed` |
| low_level_pytorch_escape_hatch | From Easy Mode to raw PyTorch control |
| custom_tensor_projector_workflow | Custom GNN with spatial pooling and classifier head |
| colab_install_and_doctor | Install TGraphX in Colab and verify the environment |
| end_to_end_research_workflow | Complete pipeline: data → model → train → metrics → artifacts |
| limitations_and_roadmap_honest_demo | Honest capabilities and roadmap |

---

## Tutorial Scripts (CPU-runnable, no Jupyter required)

| Script | What it demonstrates |
|--------|---------------------|
| [tutorials/tensor_node_classification_neighbor_loader.py](../tutorials/tensor_node_classification_neighbor_loader.py) | Canonical tensor node classification with NeighborLoader |
| [tutorials/graph_generation_quickstart.py](../tutorials/graph_generation_quickstart.py) | Graph generation quickstart |
| [tutorials/evolutionary_optimization_quickstart.py](../tutorials/evolutionary_optimization_quickstart.py) | Evolutionary optimization (GA/SA/NSGA-II) |
| [tutorials/graph_rl_quickstart.py](../tutorials/graph_rl_quickstart.py) | Graph RL comparison |
| [tutorials/real_dataset_cora_node_classification.py](../tutorials/real_dataset_cora_node_classification.py) | Cora node classification (optional PyG, graceful fallback) |
| [tutorials/image_patch_tensor_graph_demo.py](../tutorials/image_patch_tensor_graph_demo.py) | Image-patch tensor graph demo |
| [tutorials/kg_benchmark_quickstart.py](../tutorials/kg_benchmark_quickstart.py) | KG benchmark (TransE/DistMult/RESCAL/SimplE + MRR) |

A 60-second Easy Mode smoke (**no direct `import torch` required**):
[examples/easy_tensor_node_classification_no_torch.py](../examples/easy_tensor_node_classification_no_torch.py)

---

## Running on Colab

```python
# In a Colab cell:
!pip install -q tgraphx
!python -m tgraphx doctor
# Download a notebook from Google Drive, then open it.
```

---

## Generating Local Notebooks

```bash
python tools/generate_notebooks.py     # regenerates notebooks/ (tracked in git)
python tools/generate_colab_drafts.py  # creates colab_drafts/ (gitignored, for review)
python tools/validate_colab_drafts.py
```
