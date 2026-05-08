# Tutorials

A suggested 10-minute path to learn TGraphX:

1. **Install and quickstart** — `examples/minimal_spatial_message_passing.py`
   confirms that the package is installed and a spatial GNN forward + backward pass works.

2. **Synthetic patch-graph dataset** — `examples/datasets_quickstart.py`
   shows the dataset registry, synthetic datasets, and how to iterate over them.

3. **Transforms** — `examples/transforms_metrics_demo.py` demonstrates
   `Compose([NormalizeFeatures(), AddSelfLoops(), RandomNodeSplit(...)])` and computing
   `accuracy`, `classification_report`, and `regression_report`.

4. **Train from a config** — `examples/experiment_config_quickstart.py`
   reads `examples/configs/synthetic_patch_graph.yaml`, trains for a few epochs, and
   writes dashboard artefacts.

5. **Explainability** — `examples/explainability_saliency_demo.py` and
   `examples/explainability_attention_demo.py` show saliency, integrated gradients,
   edge attribution, and the `TensorGATLayer` attention-to-edge-scores path.

6. **Dashboard** — `tgraphx-dashboard --logdir runs/` shows live training metrics,
   dataset metadata, and explanation artefacts written by the previous demos.

## Colab tutorial

An interactive Colab notebook covers install, dataset loading, 2-D and 3-D spatial
GNNs, regression, edge prediction, and the dashboard:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1agls1xtqE5WxbWthcG0HEa3Gbk3fvoCD?usp=sharing)

All tasks in the notebook use synthetic data designed to verify installation and
correctness — they are not benchmarks.

## Full example gallery

See [examples/README.md](../examples/README.md) for every runnable demo
and what it tests.
