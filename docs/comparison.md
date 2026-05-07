# When to Use TGraphX

TGraphX occupies a specific niche in the GNN ecosystem.  This page describes
when it is the right choice and when other tools serve you better.

---

## TGraphX is a strong fit when…

- **Node features are multi-dimensional tensors** — image-patch feature maps
  `[C, H, W]`, volumetric patches `[C, D, H, W]`, or any spatial layout you
  want to preserve through message passing.
- **You want PyTorch-native GNN workflows** without installing a separate C++
  extension package.
- **You are building or studying patch-graph architectures** — e.g. treating an
  image as a graph of non-overlapping patches and running GNN layers across
  the spatial neighbourhoods.
- **You want a lightweight local dashboard** for monitoring training without
  setting up a TensorBoard server or external service.
- **You need educational clarity** — TGraphX is designed to be readable and
  verifiable; the Colab tutorial walks through every API feature on synthetic
  sanity-check tasks.
- **Ordinary vector-feature GNNs** — TGraphX supports `[N, D]` vector
  features via `LinearMessagePassing` and the `"linear"` factory key; it is
  usable for standard graph tasks, not only spatial ones.

---

## PyTorch Geometric (PyG) is a better fit when…

- You need a large library of GNN variants (GCN, APPNP, DimeNet, etc.) out
  of the box.
- You need scalable neighbor sampling (GraphSAGE mini-batch, ClusterGCN,
  SAINT).
- You need heterogeneous or temporal graph support.
- You need tight integration with the broader PyG ecosystem (datasets,
  transforms, benchmarks).

TGraphX is **not** a drop-in replacement for PyG.  The API, layer semantics,
and edge-feature conventions differ.

---

## Deep Graph Library (DGL) is a better fit when…

- You need a backend-agnostic, production-scale graph learning framework.
- You need multi-GPU or distributed graph training.
- You need DGL's built-in dataset zoo.

---

## NetworkX is a better fit when…

- You need classical graph algorithms (shortest paths, centrality, community
  detection).
- You are doing graph analysis or inspection rather than deep learning.
- CPU-only, no gradient, no tensor operations needed.

TGraphX graph builders (`build_grid_graph`, `build_knn_graph`, etc.) can
produce the `edge_index` tensors that feed into TGraphX models, but they are
not a substitute for full graph algorithm libraries.

---

## TensorBoard is a better fit when…

- You need histograms, embedding projectors, image grids, or hyperparameter
  parallel coordinates.
- You have an existing TensorBoard workflow.
- You need multi-user or remote team access to experiment logs.

The TGraphX dashboard is a **local-first, lightweight** training monitor for
the GNN-specific information most relevant during a single experiment.  It is
not a TensorBoard replacement.  You can use `TensorBoardLogger` from
`tgraphx.tracking` to write TensorBoard-compatible event files if you want
both.

---

## Side-by-side comparison

| Capability | TGraphX | PyG | DGL | NetworkX |
|---|:-:|:-:|:-:|:-:|
| Vector `[N,D]` node features | ✅ | ✅ | ✅ | N/A |
| Spatial/volumetric `[N,C,H,W]` | ✅ | ❌ native | ❌ native | N/A |
| GAT / SAGE / GIN | ✅ | ✅ | ✅ | N/A |
| Edge weights + edge features | ✅ | ✅ | ✅ | N/A |
| Heterogeneous graphs | ❌ | ✅ | ✅ | partial |
| Temporal graphs | ❌ | ✅ | ✅ | N/A |
| Graph Transformers | ❌ | ✅ | ✅ | N/A |
| Large-scale sampling | ❌ | ✅ | ✅ | N/A |
| CPU-only install | ✅ | ✅ | ✅ | ✅ |
| Local training dashboard | ✅ | ❌ | ❌ | N/A |
| No C++ extension required | ✅ | ❌ | ❌ | ✅ |
| Classical graph algorithms | ❌ | partial | partial | ✅ |

---

## Honest positioning

TGraphX does not claim to be:

- The fastest GNN library
- The most complete layer zoo
- A production-scale distributed training framework
- A TensorBoard replacement
- Compatible with PyG or DGL APIs

TGraphX's goal is to be the **most natural choice for tensor-aware patch-graph
GNN experiments** in pure PyTorch, with a pleasant local monitoring story and
honest educational documentation.

---

## See also

- [Limitations](limitations.md)
- [Dashboard](dashboard.md)
- [Training utilities](training_utilities.md)
