# TGraphX Colab Gallery

A curated index of CPU-runnable, deterministic TGraphX scripts.

> **Note on Colab links.** TGraphX is currently a script-first project.
> Where Colab notebooks exist (in the official Colab tutorial drive), they
> are linked.  Other entries are runnable as plain Python scripts on Colab
> by uploading the file or running it from a checkout.  Notebook versions
> for the v1.2 additions are planned but not yet uploaded — those entries
> are marked **(script)**.

---

## Tutorials by capability

| # | Capability | Script | Status |
|---|------------|--------|--------|
| 1 | Tensor node classification with NeighborLoader (canonical first tutorial) | [tutorials/tensor_node_classification_neighbor_loader.py](../tutorials/tensor_node_classification_neighbor_loader.py) | runnable |
| 2 | Graph generation quickstart (ER/BA/SBM + metrics + dashboard) | [tutorials/graph_generation_quickstart.py](../tutorials/graph_generation_quickstart.py) | runnable |
| 3 | Evolutionary optimization quickstart (GA/SA/NSGA-II) | [tutorials/evolutionary_optimization_quickstart.py](../tutorials/evolutionary_optimization_quickstart.py) | runnable |
| 4 | Graph RL quickstart (random/DQN/PPO/TD3/SAC comparison) | [tutorials/graph_rl_quickstart.py](../tutorials/graph_rl_quickstart.py) | runnable |
| 5 | **Cora node classification** (PyG-optional, graceful skip) | [tutorials/real_dataset_cora_node_classification.py](../tutorials/real_dataset_cora_node_classification.py) | runnable (script) |
| 6 | **Image-patch tensor graph** (synthetic image, tensor-vs-flatten comparison) | [tutorials/image_patch_tensor_graph_demo.py](../tutorials/image_patch_tensor_graph_demo.py) | runnable (script) |
| 7 | **KG benchmark quickstart** (TransE / DistMult / RESCAL + filtered MRR / Hits@K) | [tutorials/kg_benchmark_quickstart.py](../tutorials/kg_benchmark_quickstart.py) | runnable (script) |

A 60-second **Easy Mode** smoke is also available:
[examples/easy_tensor_node_classification_no_torch.py](../examples/easy_tensor_node_classification_no_torch.py)
— no direct `import torch` required.

---

## Running on Colab

Until dedicated notebooks are published, the recommended pattern is:

```python
# In a Colab cell:
!pip install -q tgraphx
!python -m tgraphx doctor
# Then upload or paste a tutorial script and run it.
```

For tutorials that require PyG (e.g. real Cora data), additionally:

```python
!pip install -q torch_geometric
!python tutorials/real_dataset_cora_node_classification.py --download
```

The same scripts run unchanged on a local CPU.

---

## What is **not** in the gallery yet

- Dedicated notebooks for the v1.2 tutorials.  Roadmapped — see [roadmap.md](roadmap.md).
- OGB benchmark adapters as a tutorial.  Roadmapped.
- Streamed/out-of-core graph training tutorial.  Roadmapped.

We deliberately avoid linking to broken or hypothetical notebook URLs.
