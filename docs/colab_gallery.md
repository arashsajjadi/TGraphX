# TGraphX Colab Gallery

A curated index of CPU-runnable, deterministic TGraphX notebooks and scripts.

---

## Notebooks (v1.3, committed, CPU-runnable)

The following 7 notebooks are part of the v1.3 release and included in the repository.
Run them with local Jupyter or open in Google Colab.

> **Colab URLs will be added** by the maintainer after uploading to Google Drive and testing.
> Until then, upload or paste the notebook file directly into Colab.

| # | File | Scenario | Subsystem |
|---|------|----------|-----------|
| 01 | [notebooks/01_easy_tensor_node_classification.ipynb](../notebooks/01_easy_tensor_node_classification.ipynb) | Classify nodes with `[C,H,W]` features via Easy Mode | `tgraphx.easy` |
| 02 | [notebooks/02_image_patch_tensor_graph.ipynb](../notebooks/02_image_patch_tensor_graph.ipynb) | **Core identity** — image-patch graph, tensor-vs-flatten comparison | `ConvMessagePassing` |
| 03 | [notebooks/03_kg_completion_rescal_simple_hpo.ipynb](../notebooks/03_kg_completion_rescal_simple_hpo.ipynb) | KG completion with RESCAL, TransE, SimplE + HPO | `tgraphx.kg` |
| 04 | [notebooks/04_graph_generation_and_optimization.ipynb](../notebooks/04_graph_generation_and_optimization.ipynb) | Graph generation metrics + evolutionary optimization | `tgraphx.generation`, `tgraphx.evolutionary` |
| 05 | [notebooks/05_graph_rl_coloring_and_navigation.ipynb](../notebooks/05_graph_rl_coloring_and_navigation.ipynb) | Graph RL with callbacks and CSV logging | `tgraphx.rl` |
| 06 | [notebooks/06_graph_io_roundtrip.ipynb](../notebooks/06_graph_io_roundtrip.ipynb) | GraphML write/read round-trip and tensor-feature limitations | `tgraphx.io` |
| 07 | [notebooks/07_benchmark_suite_and_dashboard.ipynb](../notebooks/07_benchmark_suite_and_dashboard.ipynb) | Run v1.3 benchmark suite and inspect dashboard artifacts | `benchmarks/` |

To regenerate notebooks locally: `python tools/generate_notebooks.py`

---

## Expanded Colab Gallery — Preparation in Progress

An expanded gallery of **30 scenario-driven draft notebooks** has been prepared locally
covering a wider range of TGraphX capabilities.

**Status:** Notebooks are under maintainer review and will be uploaded to Google Colab.
Verified Colab links will be added in **v1.3.1** after testing.

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
