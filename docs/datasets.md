# Datasets

TGraphX ships a unified dataset registry, native synthetic
datasets, folder-backed datasets, and **optional** adapters for
torchvision, PyTorch Geometric, DGL, and OGB.

> **TGraphX does not bundle or redistribute third-party datasets.** Every
> external adapter delegates download/loading to the upstream library
> (which the user installs separately) and converts samples to TGraphX
> :class:`~tgraphx.Graph` / :class:`~tgraphx.HeteroGraph` /
> :class:`~tgraphx.TemporalGraphSequence` objects on the fly.  Downloads
> happen **only when the user passes** ``download=True``.

## Quickstart

```python
from tgraphx.datasets import list_datasets, dataset_info, get_dataset

print(list_datasets())            # registered names
print(dataset_info("synthetic:patch_graph"))

ds = get_dataset("synthetic:patch_graph", num_graphs=32, seed=0)
g = ds[0]                          # tgraphx.Graph
```

## Cache layout

By default datasets cache under:

| Priority | Location |
|----------|----------|
| 1 | The ``root`` argument explicitly passed to the dataset / `get_dataset` |
| 2 | `$TGRAPHX_DATA` (environment variable) |
| 3 | `~/.cache/tgraphx/datasets` |

Each dataset uses a `<root>/<slug>/raw/` and `<root>/<slug>/processed/`
layout.  Inspect or clear the cache with:

```python
from tgraphx.datasets import cache_summary, clear_cache

print(cache_summary())                       # info dict
print(clear_cache(dry_run=True))             # list candidates (default)
clear_cache(dataset_name="synthetic:patch_graph", dry_run=False)
```

## Download policy

* Downloads **never** happen at import time.
* Downloads **never** happen during tests.
* `DownloadableGraphDataset` raises `DatasetFilesNotFoundError` (a
  subclass of `Exception`) when the raw files are missing and
  `download=False`, with a clear hint.
* Downloads are atomic (writes via a sibling `.tmp` file) and verify
  SHA-256 checksums when one is supplied.
* Archive extraction blocks path-traversal entries and is tested for
  `.zip`, `.tar`, `.tar.gz`, `.tar.bz2`, `.tar.xz`.

## Synthetic datasets

| Name | Task | Graph shape |
|------|------|-------------|
| `synthetic:patch_graph` | graph_classification / graph_regression | `[P, C, ph, pw]` |
| `synthetic:volume_graph` | graph_classification | `[P, C, pd, ph, pw]` |
| `synthetic:node_classification` | node_classification | `[N, D]` |
| `synthetic:edge_prediction` | edge_classification | `[N, D]` |
| `synthetic:graph_regression` | graph_regression | `[P, C, ph, pw]` |
| `synthetic:hetero` | hetero_node_classification | typed |
| `synthetic:temporal` | temporal_graph_classification | sequences |

All synthetic datasets are deterministic with `seed=...`, are tiny by
default (CI-safe), and use *learnable* labels so tiny-overfit
trainability tests succeed in seconds.

## Folder datasets

* `ImageFolderPatchGraphDataset(root, ...)` — walks
  `root/class/*.png|.jpg|...`, converts each image into a patch graph.
* `VolumeFolderPatchGraphDataset(root, ...)` — walks
  `root/class/*.npy|.npz|.pt`, converts each volume into a 3-D patch graph.

PIL is required by the image variant (lazy-imported, install with
`pip install "tgraphx[pillow]"` or `pip install Pillow`).

## torchvision adapters

```python
from tgraphx.datasets import MNISTPatchGraphDataset, get_dataset

ds = MNISTPatchGraphDataset(
    root="data", download=True, patch_size=7, graph_builder="grid",
)
# or via the registry:
ds = get_dataset("torchvision:mnist_patch", root="data", download=True, patch_size=7)
```

| Registry name | Upstream class |
|---------------|-----------------|
| `torchvision:mnist_patch` | `MNIST` |
| `torchvision:fashion_mnist_patch` | `FashionMNIST` |
| `torchvision:kmnist_patch` | `KMNIST` |
| `torchvision:cifar10_patch` | `CIFAR10` |
| `torchvision:cifar100_patch` | `CIFAR100` |
| `torchvision:svhn_patch` | `SVHN` |
| `torchvision:stl10_patch` | `STL10` |
| `torchvision:fake_patch` | `FakeData` (no download) |
| `torchvision:image_folder_patch` | local `ImageFolderPatchGraphDataset` |

For any other torchvision dataset, use the generic
`TorchvisionPatchGraphDataset(dataset_class_or_name, root=..., ...)`.

## PyG adapters

```python
ds = get_dataset("pyg:planetoid/cora", root="data", download=True)
```

Curated:

| Registry name | Upstream class |
|---------------|-----------------|
| `pyg:planetoid/cora`, `.../citeseer`, `.../pubmed` | `Planetoid` |
| `pyg:tudataset/{mutag,proteins,enzymes,imdb-binary,reddit-binary}` | `TUDataset` |

Generic:
```python
from tgraphx.datasets import PyGDatasetAdapter

ds = PyGDatasetAdapter(
    dataset_cls="Planetoid",
    root="data",
    dataset_kwargs={"name": "Cora"},
    download=True,
)
```

PyG must be installed separately
(`pip install torch-geometric` — install instructions depend on your
PyTorch / CUDA build).

## DGL adapters

```python
ds = get_dataset("dgl:cora", root="data", download=True)
```

Curated:
* `dgl:cora`, `dgl:citeseer`, `dgl:pubmed` (citation graph datasets).
* Generic `dgl:generic` accepts any `dgl.data.<DGLDataset>` subclass.

DGL wheels are platform-specific; we deliberately do **not** add DGL
to a TGraphX optional extra.  Install with the upstream instructions
([www.dgl.ai/pages/start.html](https://www.dgl.ai/pages/start.html)).

## OGB adapters

```python
ds = get_dataset("ogb:ogbn-arxiv", root="data", download=True)
g = ds[0]
split = ds.get_idx_split()
ev = ds.get_evaluator()
```

* `ogb:ogbn-arxiv`, `ogb:ogbn-products` — node-property prediction.
* `ogb:ogbl-collab` — link-property prediction.
* `ogb:ogbg-molhiv` — graph-property prediction.
* `ogb:generic` — pass any name (`name="ogbn-..."`).

Install with `pip install "tgraphx[ogb]"`.  TGraphX makes **no
leaderboard claims** — we re-export the official OGB evaluators so
your training loop runs the standard protocol.

## Optional dependency policy

| Adapter family | Install |
|----------------|---------|
| Native synthetic / folder datasets | (no extra) |
| torchvision wrappers | already a TGraphX base dependency |
| PyG | `pip install "tgraphx[pyg]"` |
| DGL | follow upstream install (platform-sensitive) |
| OGB | `pip install "tgraphx[ogb]"` |
| Image folder dataset (PIL) | `pip install "tgraphx[pillow]"` |

Importing `tgraphx` or `tgraphx.datasets` does **not** import any of
torch_geometric, dgl, or ogb.

## License / citation policy

* **TGraphX does not redistribute third-party datasets.**  Adapters
  call into the upstream loader; on first use those loaders may
  download from the upstream's own URL/CDN.  Upstream license terms
  apply unchanged.
* Each adapter's `DatasetMetadata.citation` field points to the
  upstream paper / dataset card.  Cite the upstream dataset *and*
  TGraphX when publishing.
* Synthetic datasets are TGraphX-generated and released under the same
  MIT licence as the rest of the package.

## Test policy

* No test in this repository connects to the network.
* Download paths are exercised through monkey-patched `urlopen`.
* Optional-adapter tests skip cleanly when the upstream package is
  missing and explicitly assert that constructing the adapter without
  it raises `OptionalDependencyError` with a helpful install hint.
