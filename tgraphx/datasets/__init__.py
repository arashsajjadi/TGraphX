"""TGraphX dataset ecosystem (v0.2.9).

Importing this module is **lightweight**: it does not import
torchvision, torch_geometric, dgl, or ogb.  Optional adapters live in
sibling modules and import their upstream package only when
constructed.

User-facing API:

* :func:`get_dataset(name, **kwargs)` — registry lookup + instantiation.
* :func:`list_datasets()` / :func:`dataset_info(name)`.
* All synthetic and folder-backed dataset classes.
* Optional adapter classes (importing them does not import the
  upstream package).
"""
from __future__ import annotations

# ── Public surface ───────────────────────────────────────────────────────────

from .base import (
    BaseGraphDataset,
    DownloadableGraphDataset,
    ExternalDatasetAdapter,
    InMemoryGraphDataset,
)
from .cache import (
    cache_summary,
    clear_cache,
    get_default_cache_root,
    resolve_dataset_root,
)
from .converters import (
    from_dgl_graph,
    from_dgl_heterograph,
    from_pyg_data,
    from_pyg_heterodata,
    ogb_item_to_graph,
    to_dgl_graph,
    to_dgl_heterograph,
    to_pyg_data,
    to_pyg_heterodata,
    torchvision_image_to_patch_graph,
)
from .dgl_wrappers import DGLCitationDatasetAdapter, DGLDatasetAdapter
from .download import download_url, extract_archive, maybe_download, verify_checksum
from .errors import (
    DatasetError,
    DatasetExtractionError,
    DatasetFilesNotFoundError,
    DatasetIntegrityError,
    DatasetNotFoundError,
    OptionalDependencyError,
)
from .folder import ImageFolderPatchGraphDataset, VolumeFolderPatchGraphDataset
from .metadata import DatasetMetadata
from .ogb_wrappers import (
    OGBDatasetAdapter,
    OGBEvaluatorWrapper,
    OGBGraphPropertyDatasetAdapter,
    OGBLinkPropertyDatasetAdapter,
    OGBNodePropertyDatasetAdapter,
)
from .pyg_wrappers import (
    PyGDatasetAdapter,
    PyGPlanetoidDataset,
    PyGTUDatasetAdapter,
)
from .registry import (
    available_dataset_groups,
    dataset_info,
    get_dataset,
    has_dataset,
    list_datasets,
    list_dataset_aliases,
    load_dataset,
    normalize_dataset_name,
    register_dataset,
)
from .synthetic import (
    SyntheticEdgePredictionDataset,
    SyntheticGraphRegressionDataset,
    SyntheticHeteroGraphDataset,
    SyntheticNodeClassificationDataset,
    SyntheticPatchGraphDataset,
    SyntheticTemporalGraphDataset,
    SyntheticVolumeGraphDataset,
)
from .torchvision_wrappers import (
    CIFAR10PatchGraphDataset,
    CIFAR100PatchGraphDataset,
    FakeDataPatchGraphDataset,
    FashionMNISTPatchGraphDataset,
    KMNISTPatchGraphDataset,
    MNISTPatchGraphDataset,
    STL10PatchGraphDataset,
    SVHNPatchGraphDataset,
    TorchvisionPatchGraphDataset,
)


# ── Registry population ──────────────────────────────────────────────────────


def _register_synthetic() -> None:
    register_dataset(
        "synthetic:patch_graph",
        SyntheticPatchGraphDataset,
        tags=["synthetic", "vision", "graph_classification"],
        metadata={
            "description": "Synthetic 2-D image-patch graphs with learnable patterns.",
            "task": "graph_classification",
            "requires": "(none)",
        },
    )
    register_dataset(
        "synthetic:volume_graph",
        SyntheticVolumeGraphDataset,
        tags=["synthetic", "volume", "graph_classification"],
        metadata={
            "description": "Synthetic 3-D volume-patch graphs.",
            "task": "graph_classification",
            "requires": "(none)",
        },
    )
    register_dataset(
        "synthetic:node_classification",
        SyntheticNodeClassificationDataset,
        tags=["synthetic", "vector", "node_classification"],
        metadata={
            "description": "Single-graph SBM node classification with masks.",
            "task": "node_classification",
            "requires": "(none)",
        },
    )
    register_dataset(
        "synthetic:edge_prediction",
        SyntheticEdgePredictionDataset,
        tags=["synthetic", "vector", "edge_prediction"],
        metadata={
            "description": "Similarity-based edge prediction sanity dataset.",
            "task": "edge_prediction",
            "requires": "(none)",
        },
    )
    register_dataset(
        "synthetic:graph_regression",
        SyntheticGraphRegressionDataset,
        tags=["synthetic", "vision", "graph_regression"],
        metadata={
            "description": "Patch graph + scalar intensity regression.",
            "task": "graph_regression",
            "requires": "(none)",
        },
    )
    register_dataset(
        "synthetic:hetero",
        SyntheticHeteroGraphDataset,
        tags=["synthetic", "hetero"],
        metadata={
            "description": "Tiny paper/author/venue hetero graph.",
            "task": "hetero_node_classification",
            "requires": "(none)",
        },
    )
    register_dataset(
        "synthetic:temporal",
        SyntheticTemporalGraphDataset,
        tags=["synthetic", "temporal"],
        metadata={
            "description": "Tiny temporal graph sequence with trend labels.",
            "task": "temporal_graph_classification",
            "requires": "(none)",
        },
    )


def _register_torchvision() -> None:
    common_meta = {"requires": "torchvision (base dependency; reinstall with `pip install --force-reinstall torchvision` if missing)"}
    register_dataset(
        "torchvision:mnist_patch", MNISTPatchGraphDataset,
        tags=["torchvision", "vision"],
        metadata={"description": "MNIST split into patch graphs.", **common_meta},
    )
    register_dataset(
        "torchvision:fashion_mnist_patch", FashionMNISTPatchGraphDataset,
        tags=["torchvision", "vision"],
        metadata={"description": "FashionMNIST split into patch graphs.", **common_meta},
    )
    register_dataset(
        "torchvision:kmnist_patch", KMNISTPatchGraphDataset,
        tags=["torchvision", "vision"],
        metadata={"description": "KMNIST split into patch graphs.", **common_meta},
    )
    register_dataset(
        "torchvision:cifar10_patch", CIFAR10PatchGraphDataset,
        tags=["torchvision", "vision"],
        metadata={"description": "CIFAR-10 split into patch graphs.", **common_meta},
    )
    register_dataset(
        "torchvision:cifar100_patch", CIFAR100PatchGraphDataset,
        tags=["torchvision", "vision"],
        metadata={"description": "CIFAR-100 split into patch graphs.", **common_meta},
    )
    register_dataset(
        "torchvision:svhn_patch", SVHNPatchGraphDataset,
        tags=["torchvision", "vision"],
        metadata={"description": "SVHN split into patch graphs.", **common_meta},
    )
    register_dataset(
        "torchvision:stl10_patch", STL10PatchGraphDataset,
        tags=["torchvision", "vision"],
        metadata={"description": "STL-10 split into patch graphs.", **common_meta},
    )
    register_dataset(
        "torchvision:fake_patch", FakeDataPatchGraphDataset,
        tags=["torchvision", "vision", "fake"],
        metadata={
            "description": "torchvision.datasets.FakeData-backed patch graphs (no download).",
            **common_meta,
        },
    )
    # Generic factory registered too — users can pass any torchvision class.
    register_dataset(
        "torchvision:image_folder_patch", ImageFolderPatchGraphDataset,
        tags=["torchvision", "vision", "folder"],
        metadata={
            "description": "User-owned image folder converted to patch graphs.",
            "requires": "Pillow",
        },
    )


def _register_pyg() -> None:
    requires = "torch-geometric  (pip install torch-geometric)"

    def _make_planetoid(name: str):
        def _factory(**kwargs):
            return PyGPlanetoidDataset(name=name, **kwargs)
        return _factory

    register_dataset(
        "pyg:planetoid/cora", _make_planetoid("Cora"),
        tags=["pyg", "vector", "node_classification"],
        metadata={"description": "Planetoid/Cora via PyG.", "requires": requires},
    )
    register_dataset(
        "pyg:planetoid/citeseer", _make_planetoid("CiteSeer"),
        tags=["pyg", "vector", "node_classification"],
        metadata={"description": "Planetoid/CiteSeer via PyG.", "requires": requires},
    )
    register_dataset(
        "pyg:planetoid/pubmed", _make_planetoid("PubMed"),
        tags=["pyg", "vector", "node_classification"],
        metadata={"description": "Planetoid/PubMed via PyG.", "requires": requires},
    )

    def _make_tu(name: str):
        def _factory(**kwargs):
            return PyGTUDatasetAdapter(name=name, **kwargs)
        return _factory

    for tu in ("MUTAG", "PROTEINS", "ENZYMES", "IMDB-BINARY", "REDDIT-BINARY"):
        register_dataset(
            f"pyg:tudataset/{tu.lower()}", _make_tu(tu),
            tags=["pyg", "graph_classification"],
            metadata={
                "description": f"TUDataset/{tu} via PyG.",
                "requires": requires,
            },
        )

    register_dataset(
        "pyg:generic", PyGDatasetAdapter,
        tags=["pyg"],
        metadata={
            "description": "Generic PyG dataset adapter; pass dataset_cls=...",
            "requires": requires,
        },
    )


def _register_dgl() -> None:
    requires = "dgl  (see https://www.dgl.ai/pages/start.html)"

    def _make_citation(name: str):
        def _factory(**kwargs):
            return DGLCitationDatasetAdapter(name=name, **kwargs)
        return _factory

    for citation in ("cora", "citeseer", "pubmed"):
        register_dataset(
            f"dgl:{citation}", _make_citation(citation),
            tags=["dgl", "vector", "node_classification"],
            metadata={
                "description": f"DGL citation dataset: {citation}.",
                "requires": requires,
            },
        )
    register_dataset(
        "dgl:generic", DGLDatasetAdapter,
        tags=["dgl"],
        metadata={
            "description": "Generic DGL dataset adapter; pass dataset_cls=...",
            "requires": requires,
        },
    )


def _register_ogb() -> None:
    requires = "ogb  (pip install ogb)"

    def _make_ogb(name: str):
        def _factory(**kwargs):
            return OGBDatasetAdapter(name=name, **kwargs)
        return _factory

    for n in ("ogbn-arxiv", "ogbn-products"):
        register_dataset(
            f"ogb:{n}", _make_ogb(n),
            tags=["ogb", "node_classification"],
            metadata={"description": f"{n} via OGB.", "requires": requires},
        )
    register_dataset(
        "ogb:ogbl-collab", _make_ogb("ogbl-collab"),
        tags=["ogb", "link_prediction"],
        metadata={"description": "ogbl-collab via OGB.", "requires": requires},
    )
    register_dataset(
        "ogb:ogbg-molhiv", _make_ogb("ogbg-molhiv"),
        tags=["ogb", "graph_classification"],
        metadata={"description": "ogbg-molhiv via OGB.", "requires": requires},
    )
    register_dataset(
        "ogb:generic", OGBDatasetAdapter,
        tags=["ogb"],
        metadata={
            "description": "Generic OGB adapter; pass name='ogbn-arxiv' etc.",
            "requires": requires,
        },
    )


_register_synthetic()
_register_torchvision()
_register_pyg()
_register_dgl()
_register_ogb()


__all__ = [
    # Base classes
    "BaseGraphDataset",
    "InMemoryGraphDataset",
    "DownloadableGraphDataset",
    "ExternalDatasetAdapter",
    # Metadata + cache
    "DatasetMetadata",
    "get_default_cache_root",
    "resolve_dataset_root",
    "cache_summary",
    "clear_cache",
    # Download utilities
    "download_url",
    "verify_checksum",
    "extract_archive",
    "maybe_download",
    # Errors
    "DatasetError",
    "DatasetNotFoundError",
    "DatasetFilesNotFoundError",
    "DatasetIntegrityError",
    "DatasetExtractionError",
    "OptionalDependencyError",
    # Registry
    "register_dataset",
    "get_dataset",
    "list_datasets",
    "list_dataset_aliases",
    "load_dataset",
    "dataset_info",
    "available_dataset_groups",
    "normalize_dataset_name",
    "has_dataset",
    # Synthetic
    "SyntheticPatchGraphDataset",
    "SyntheticVolumeGraphDataset",
    "SyntheticNodeClassificationDataset",
    "SyntheticEdgePredictionDataset",
    "SyntheticGraphRegressionDataset",
    "SyntheticHeteroGraphDataset",
    "SyntheticTemporalGraphDataset",
    # Folder
    "ImageFolderPatchGraphDataset",
    "VolumeFolderPatchGraphDataset",
    # Torchvision adapters
    "TorchvisionPatchGraphDataset",
    "MNISTPatchGraphDataset",
    "FashionMNISTPatchGraphDataset",
    "KMNISTPatchGraphDataset",
    "CIFAR10PatchGraphDataset",
    "CIFAR100PatchGraphDataset",
    "SVHNPatchGraphDataset",
    "STL10PatchGraphDataset",
    "FakeDataPatchGraphDataset",
    # PyG adapters
    "PyGDatasetAdapter",
    "PyGPlanetoidDataset",
    "PyGTUDatasetAdapter",
    # DGL adapters
    "DGLDatasetAdapter",
    "DGLCitationDatasetAdapter",
    # OGB adapters
    "OGBDatasetAdapter",
    "OGBNodePropertyDatasetAdapter",
    "OGBLinkPropertyDatasetAdapter",
    "OGBGraphPropertyDatasetAdapter",
    "OGBEvaluatorWrapper",
    # Converters
    "from_pyg_data",
    "to_pyg_data",
    "from_pyg_heterodata",
    "to_pyg_heterodata",
    "from_dgl_graph",
    "to_dgl_graph",
    "from_dgl_heterograph",
    "to_dgl_heterograph",
    "ogb_item_to_graph",
    "torchvision_image_to_patch_graph",
]
