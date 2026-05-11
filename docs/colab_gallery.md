# Notebook Gallery

A curated set of CPU-runnable, scenario-driven notebooks for TGraphX. Each
notebook focuses on a single concept, uses synthetic or toy data, and runs in
under two minutes on a modest CPU.

> **About the links.** Notebooks are hosted on Google Drive as `.ipynb` files.
> Open the link, then choose **Open with → Google Colaboratory** at the top
> of Drive (or download the file and open it locally in Jupyter).

---

## Quick start

| Goal | Open notebook |
|------|---------------|
| Train your first tensor-aware GNN | [01 — Easy Mode node classification](https://drive.google.com/file/d/1C-vydQXnn9LrYhx5hZDQl6H601itnbGp/view?usp=sharing) |
| Install in Colab and verify | [28 — Install and `tgraphx doctor`](https://drive.google.com/file/d/1rHuhQKwNkZpH3_gj46xq1vwdvzutQd_9/view?usp=sharing) |
| End-to-end research workflow | [29 — Data → model → metrics](https://drive.google.com/file/d/14y8rxwk8ajepynjSV2wvEbs7QQLLqXjB/view?usp=sharing) |
| See current limitations honestly | [30 — Limitations and roadmap](https://drive.google.com/file/d/1hfkS1NF--8Q9Lpw26tEveVTCvZpiOIM2/view?usp=sharing) |

---

## Topics

### Easy Mode and onboarding

| # | Notebook | What you learn |
|---|----------|----------------|
| 01 | [Easy Mode node classification](https://drive.google.com/file/d/1C-vydQXnn9LrYhx5hZDQl6H601itnbGp/view?usp=sharing) | Train on `[C, H, W]` node features with zero boilerplate |
| 23 | [Dashboard artifacts from Easy Mode](https://drive.google.com/file/d/1LgJ5JJNNmoAzJQUamAkFg8G009sLmS3L/view?usp=sharing) | Write run metadata, view it locally with the dashboard |
| 25 | [Reproducibility and seed control](https://drive.google.com/file/d/1ihdOfq-_z9iH9n7s52mJ2Veyog8jqdoB/view?usp=sharing) | Strict CPU determinism, `set_seed`, CUDA caveats |
| 26 | [Low-level PyTorch escape hatch](https://drive.google.com/file/d/1c8A1-_ZoImnmv4NgIGMDibD-x-Zm9nM_/view?usp=sharing) | From `EasyResult` to raw PyTorch control |
| 27 | [Custom tensor projector workflow](https://drive.google.com/file/d/1DAVJ-dm6uP4vmExYDOp9hKkrE6SGB7vl/view?usp=sharing) | Build a custom GNN with spatial pooling and a classifier head |
| 28 | [Install and `tgraphx doctor`](https://drive.google.com/file/d/1rHuhQKwNkZpH3_gj46xq1vwdvzutQd_9/view?usp=sharing) | Verify an install in Colab in 30 seconds |
| 29 | [End-to-end research workflow](https://drive.google.com/file/d/14y8rxwk8ajepynjSV2wvEbs7QQLLqXjB/view?usp=sharing) | Data → model → train → metrics → artifacts |
| 30 | [Limitations and roadmap](https://drive.google.com/file/d/1hfkS1NF--8Q9Lpw26tEveVTCvZpiOIM2/view?usp=sharing) | Honest capability map and what is not yet implemented |

### Tensor-native message passing

| # | Notebook | What you learn |
|---|----------|----------------|
| 02 | [Image-patch graph: tensor vs flatten](https://drive.google.com/file/d/1uPXV1Ybmw0iR8-5A57Ig6HM_Y10vpXM5/view?usp=sharing) | Why `[C, H, W]` features beat naive flattening |
| 03 | [Tensor-vs-flatten benchmark story](https://drive.google.com/file/d/1XsHLO1ktivQuKlr0uZFBA5l9wK9KjU9Y/view?usp=sharing) | Runtime, parameter count, gradient health |
| 04 | [Edge tensor features and message passing](https://drive.google.com/file/d/1hfsFMNu891m22SLjZK3xCPDfp2AlG8mU/view?usp=sharing) | Scalar and vector edge attributes during aggregation |
| 05 | [Graph-level tensor state classification](https://drive.google.com/file/d/1zbVydvlb3mFjRQ2iVjrYLPb6QMwRQ1It/view?usp=sharing) | `graph_features` (input) vs `graph_label` (target) |

### Sampling and mini-batch workflows

| # | Notebook | What you learn |
|---|----------|----------------|
| 06 | [NeighborLoader seed-node loss](https://drive.google.com/file/d/16h2S_6tOJW_Z51wX167rARhgk8BaVBWQ/view?usp=sharing) | `batch.seed_y`, `batch.seed_logits()`, why `logits[:batch_size]` is unsafe |
| 07 | [Sampling benchmark with NeighborLoader](https://drive.google.com/file/d/1CPviNy3vx_lqjHQR-y6n4nyLLqc61nb7/view?usp=sharing) | Throughput; honest scope and limitations |
| 08 | [GraphSAINT and Cluster-GCN smoke](https://drive.google.com/file/d/1RnUZxEbd6s5P9sSQiQ9aTn6-ZhC-Ceg3/view?usp=sharing) | Node sampling and partitioning foundations |

### Knowledge graphs

| # | Notebook | What you learn |
|---|----------|----------------|
| 09 | [TransE / RESCAL / SimplE quickstart](https://drive.google.com/file/d/1QlCNZg2U0HJ6I6M4V8qXArKEwweZOjqn/view?usp=sharing) | KG embedding models with filtered MRR / Hits@K |
| 10 | [KG HPO: grid and random search](https://drive.google.com/file/d/12mqfQvcUGCm3UhfAh0PGbCMvpNBgktmF/view?usp=sharing) | `run_kg_hpo()` across model and hyperparameter combinations |
| 11 | [Multimodal KG with tensor features](https://drive.google.com/file/d/1WVm__OQyWItg3SRozTUWeULs8QaYBUu1/view?usp=sharing) | Feature-aware KG scoring with entity embeddings |
| 12 | [Filtered ranking explained](https://drive.google.com/file/d/1LgyDOIXb7iH70a5PxMHO8j6bYiHNS1GK/view?usp=sharing) | Hand-checkable MRR / Hits@K on a tiny KG |

### Graph generation and evolutionary optimization

| # | Notebook | What you learn |
|---|----------|----------------|
| 13 | [Classical generation and metrics](https://drive.google.com/file/d/1Q8358qYmw80SBr-fXFmkTg1tcRhaUg16/view?usp=sharing) | ER / BA / SBM generators; validity, uniqueness, structure |
| 14 | Evolutionary optimization (local draft) | GA and NSGA-II with explicit objective lists |

> Notebook 14 ships as a local draft (`colab_drafts/14_*.ipynb`). Regenerate with `python tools/generate_colab_drafts.py`.

### Graph reinforcement learning

| # | Notebook | What you learn |
|---|----------|----------------|
| 15 | [Coloring with callbacks](https://drive.google.com/file/d/1TNPIMbADQtglWfTz55MGHAWYLS3d8UvV/view?usp=sharing) | EarlyStopping and CSV logging during RL training |
| 16 | [MaxCut and navigation](https://drive.google.com/file/d/1xsmgihWMTY07XMDIesAHHKfwoli_gDWT/view?usp=sharing) | Sequential decision-making on graphs |
| 17 | [Callbacks, logging, dashboard artifacts](https://drive.google.com/file/d/1RabWNOWvZVmg5PIT8t_cQ3UgRUlPckzZ/view?usp=sharing) | RL training logs that the dashboard can render |

### Graph IO

| # | Notebook | What you learn |
|---|----------|----------------|
| 18 | [GraphML round-trip](https://drive.google.com/file/d/11Ul2v5KVYkrVFOhoeSZkE6Y8HG1qcgu5/view?usp=sharing) | Read / write GraphML; what tensor features cannot be encoded |
| 19 | GraphML tensor-semantics warning (local draft) | Why multi-dimensional features cannot be serialized to GraphML |

### Graph mining

| # | Notebook | What you learn |
|---|----------|----------------|
| 20 | [Motifs and cliques](https://drive.google.com/file/d/1ZbtFGqNuPxfqI8xt3FlxzgPozukzbvua/view?usp=sharing) | `motif_profile`, clique detection, centrality measures |
| 21 | [WL-subtree kernels and similarity](https://drive.google.com/file/d/1iiOsGb9Tb5bxUQYq0i-anpV4rOiTRlQk/view?usp=sharing) | WL kernel, graph-pair similarity scoring |
| 22 | Structural roles (local draft) | Reading `degree_statistics` correctly for bidirectional edge_index |

### Benchmarking

| # | Notebook | What you learn |
|---|----------|----------------|
| 24 | v1.3 benchmark suite (local draft) | Run `tgraphx.benchmarks.run_v13_benchmark_suite` from a pip install |

> Notebook 24 uses the package-level API (`from tgraphx.benchmarks import run_v13_benchmark_suite`) and requires no repository checkout. Regenerate with `python tools/generate_colab_drafts.py`.

### Advanced real-dataset projects (v1.3.8+)

End-to-end project notebooks that combine real datasets, scientific framing,
TGraphX-native APIs, an explicit leakage policy, baselines, dashboard
artifacts, and a "Scientific and methodological notes" section. Each one was
executed in FAST_MODE before publication.

| # | Notebook | What you learn |
|---|----------|----------------|
| 31 | [MNIST class-graph membership with tensor nodes](https://drive.google.com/file/d/1WrD3kS8T83rlclytUP7r7kIzgLTsjpF8/view?usp=sharing) | Tensor-valued image nodes `[N, 1, 28, 28]`, two-edge-type `edge_attr` (visual + prototype), train-only prototypes, seed-node loss, `FlattenMLP` baseline |
| 32 | [CIFAR-10 patch-graph classification](https://drive.google.com/file/d/1oDolpqZHpM5jEfFJDZj_rl2_CYJZ7mRj/view?usp=sharing) | True per-image patch graphs `[16, 3, 8, 8]`, grid-adjacency edges, `global_mean_pool + global_max_pool`, `GraphDataLoader` batching |
| 33 | [Cora citation network: sampling and dashboard](https://drive.google.com/file/d/19MnreOV41RgSLMyYb_9E2fSzaFQUMepT/view?usp=sharing) | Transductive setting, `PyGPlanetoidDataset` bridge, `NeighborLoader` seed-node loss, `FlattenMLP` baseline, sampling metadata |
| 34 | [MovieLens user–item KG recommendation](https://drive.google.com/file/d/1ulg9wr4w4foW397KSSJJvHo9WJVbTGS0/view?usp=sharing) | Multi-relational KG (`rated_high`, `rated_low`, `has_genre`, `has_occupation`), `entity_features`, `KGTrainer`, filtered MRR/Hits@K, `run_kg_hpo`, top-K titles, popularity baseline |
| 35 | [Molecular graph classification on MUTAG](https://drive.google.com/file/d/1HBPu52cnH60LVkYwIvaBzImU_psvavs5/view?usp=sharing) | Atom + bond features (`edge_attr`), edge-aware GNN, mean+max readout, motif/structural mining, degree-feature baseline |

> Notebooks 31–35 are Google Drive `.ipynb` files. Open the link, then choose
> **Open with → Google Colaboratory** at the top of Drive. They run in FAST_MODE
> in under five minutes; full-mode requires a GPU. No SOTA / parity claims are
> made; every notebook ships with a "Scientific and methodological notes"
> section, a leakage policy, baselines, and a FAST_MODE disclaimer.

---

## Running on Colab

```python
!pip install -q tgraphx
!python -m tgraphx doctor
```

## Regenerating notebooks locally

Notebook source files are not committed to the repository. Generate them with:

```bash
python tools/generate_notebooks.py      # creates notebooks/   (gitignored)
python tools/generate_colab_drafts.py   # creates colab_drafts/ (gitignored)
python tools/validate_notebooks.py      # structural validation
```

## Tutorial scripts (no Jupyter required)

| Script | What it covers |
|--------|----------------|
| [`tutorials/tensor_node_classification_neighbor_loader.py`](../tutorials/tensor_node_classification_neighbor_loader.py) | Canonical tensor node classification with NeighborLoader |
| [`tutorials/graph_generation_quickstart.py`](../tutorials/graph_generation_quickstart.py) | Classical graph generation quickstart |
| [`tutorials/evolutionary_optimization_quickstart.py`](../tutorials/evolutionary_optimization_quickstart.py) | GA, SA, NSGA-II |
| [`tutorials/graph_rl_quickstart.py`](../tutorials/graph_rl_quickstart.py) | Graph RL algorithm comparison |
| [`tutorials/real_dataset_cora_node_classification.py`](../tutorials/real_dataset_cora_node_classification.py) | Cora node classification (optional PyG) |
| [`tutorials/image_patch_tensor_graph_demo.py`](../tutorials/image_patch_tensor_graph_demo.py) | Image-patch tensor graph demo |
| [`tutorials/kg_benchmark_quickstart.py`](../tutorials/kg_benchmark_quickstart.py) | TransE / DistMult / RESCAL / SimplE + MRR |
