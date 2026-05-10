# TGraphX Colab Gallery

A curated index of CPU-runnable, deterministic TGraphX notebooks and scripts.

---

## Notebooks (v1.3, committed, CPU-runnable)

The following 7 notebooks are part of the v1.3 release and included in the repository.
Run them with local Jupyter or open in Google Colab.

> Colab links for notebooks 01–07 verified by maintainer (v1.3.1). Additional expanded notebooks (08–30) are available locally via `tools/generate_colab_drafts.py`; their Colab links will be added as they are tested and uploaded.

| # | File | Scenario | Subsystem | Colab |
|---|------|----------|-----------|-------|
| 01 | [notebooks/01_easy_tensor_node_classification.ipynb](../notebooks/01_easy_tensor_node_classification.ipynb) | Classify nodes with `[C,H,W]` features via Easy Mode | `tgraphx.easy` | [Open in Colab](https://drive.google.com/file/d/1C-vydQXnn9LrYhx5hZDQl6H601itnbGp/view?usp=sharing) |
| 02 | [notebooks/02_image_patch_tensor_graph.ipynb](../notebooks/02_image_patch_tensor_graph.ipynb) | **Core identity** — image-patch graph, tensor-vs-flatten comparison | `ConvMessagePassing` | [Open in Colab](https://drive.google.com/file/d/1uPXV1Ybmw0iR8-5A57Ig6HM_Y10vpXM5/view?usp=sharing) |
| 03 | [notebooks/03_kg_completion_rescal_simple_hpo.ipynb](../notebooks/03_kg_completion_rescal_simple_hpo.ipynb) | KG completion with RESCAL, TransE, SimplE + HPO | `tgraphx.kg` | [Open in Colab](https://drive.google.com/file/d/1XsHLO1ktivQuKlr0uZFBA5l9wK9KjU9Y/view?usp=sharing) |
| 04 | [notebooks/04_graph_generation_and_optimization.ipynb](../notebooks/04_graph_generation_and_optimization.ipynb) | Graph generation metrics + evolutionary optimization | `tgraphx.generation`, `tgraphx.evolutionary` | [Open in Colab](https://drive.google.com/file/d/1hfsFMNu891m22SLjZK3xCPDfp2AlG8mU/view?usp=sharing) |
| 05 | [notebooks/05_graph_rl_coloring_and_navigation.ipynb](../notebooks/05_graph_rl_coloring_and_navigation.ipynb) | Graph RL with callbacks and CSV logging | `tgraphx.rl` | [Open in Colab](https://drive.google.com/file/d/1zbVydvlb3mFjRQ2iVjrYLPb6QMwRQ1It/view?usp=sharing) |
| 06 | [notebooks/06_graph_io_roundtrip.ipynb](../notebooks/06_graph_io_roundtrip.ipynb) | GraphML write/read round-trip and tensor-feature limitations | `tgraphx.io` | [Open in Colab](https://drive.google.com/file/d/16h2S_6tOJW_Z51wX167rARhgk8BaVBWQ/view?usp=sharing) |
| 07 | [notebooks/07_benchmark_suite_and_dashboard.ipynb](../notebooks/07_benchmark_suite_and_dashboard.ipynb) | Run v1.3 benchmark suite and inspect dashboard artifacts | `benchmarks/` | [Open in Colab](https://drive.google.com/file/d/1CPviNy3vx_lqjHQR-y6n4nyLLqc61nb7/view?usp=sharing) |

To regenerate notebooks locally: `python tools/generate_notebooks.py`

---

## Expanded Colab Gallery — Preparation in Progress

An expanded gallery of **30 scenario-driven draft notebooks** has been prepared locally
covering a wider range of TGraphX capabilities.

**Status:** Notebooks are under maintainer review and will be uploaded to Google Colab.
Verified Colab links for notebooks 01–07 are in this file. Additional notebooks (08–30) will be added as they are uploaded and tested.

Planned scenarios include:

| Category | Planned notebooks |
|---|---|
| Easy Mode | Easy Mode intro, low-level escape hatch, reproducibility |
| Tensor-native | Image patches (core identity), tensor-vs-flatten benchmark, edge tensor features, graph-level states |
| KG | TransE/RESCAL/SimplE completion, HPO grid/random search, multimodal entity features, filtered ranking explained |
| Graph RL | Coloring with callbacks, MaxCut/navigation, logging and dashboard artifacts |
| Graph IO | GraphML round-trip, tensor serialization semantics |
| Graph mining | Motifs and cliques, WL kernel similarity, structural roles (conceptual) |
| Workflows | End-to-end research workflow, dashboard artifacts, custom projector, install/doctor |
| Limits | Honest capabilities and roadmap |

To generate the expanded draft notebooks locally (for review only — NOT for git):
```bash
python tools/generate_colab_drafts.py
# Creates colab_drafts/*.ipynb — not tracked in git
python tools/validate_colab_drafts.py
```

---

## Tutorial Scripts (CPU-runnable, no Jupyter required)

| # | Capability | Script | Status |
|---|------------|--------|--------|
| 1 | Tensor node classification with NeighborLoader (canonical) | [tutorials/tensor_node_classification_neighbor_loader.py](../tutorials/tensor_node_classification_neighbor_loader.py) | runnable |
| 2 | Graph generation quickstart (ER/BA/SBM + metrics + dashboard) | [tutorials/graph_generation_quickstart.py](../tutorials/graph_generation_quickstart.py) | runnable |
| 3 | Evolutionary optimization quickstart (GA/SA/NSGA-II) | [tutorials/evolutionary_optimization_quickstart.py](../tutorials/evolutionary_optimization_quickstart.py) | runnable |
| 4 | Graph RL quickstart (random/DQN/PPO/TD3/SAC comparison) | [tutorials/graph_rl_quickstart.py](../tutorials/graph_rl_quickstart.py) | runnable |
| 5 | Cora node classification (PyG-optional, graceful skip) | [tutorials/real_dataset_cora_node_classification.py](../tutorials/real_dataset_cora_node_classification.py) | runnable |
| 6 | Image-patch tensor graph demo | [tutorials/image_patch_tensor_graph_demo.py](../tutorials/image_patch_tensor_graph_demo.py) | runnable |
| 7 | KG benchmark quickstart (TransE/DistMult/RESCAL/SimplE + MRR) | [tutorials/kg_benchmark_quickstart.py](../tutorials/kg_benchmark_quickstart.py) | runnable |

A 60-second **Easy Mode** smoke (no direct `import torch` required):
[examples/easy_tensor_node_classification_no_torch.py](../examples/easy_tensor_node_classification_no_torch.py)

---

## Running on Colab

```python
# In a Colab cell — install TGraphX:
!pip install -q tgraphx

# Check installation:
!python -m tgraphx doctor

# Upload a notebook file or a tutorial script and run it.
# For Cora real data (requires PyG):
# !pip install -q torch_geometric
# !python tutorials/real_dataset_cora_node_classification.py --download
```

All tutorial scripts run unchanged on local CPU.
