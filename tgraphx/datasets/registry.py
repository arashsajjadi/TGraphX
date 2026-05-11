"""Lightweight dataset registry — name → factory.

The registry is the user-facing API for ``get_dataset`` /
``list_datasets`` / ``dataset_info``.  It deliberately avoids importing
optional dependencies just to list names; lazy instantiation happens
only inside the registered factory functions.

A factory entry stores:

* ``factory``: callable returning a dataset instance (lazy).
* ``aliases``: alternate names the user can pass (case-insensitive).
* ``tags``: free-form labels (``"synthetic"``, ``"vision"``,
  ``"hetero"``, ``"temporal"``, ``"vector"``) used for filtering.
* ``metadata``: small dict shown by :func:`dataset_info`
  (``"description"``, ``"task"``, ``"requires"``, etc.) — all
  *string-only* so listing never imports anything heavy.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .errors import DatasetNotFoundError

# Type alias for clarity.
DatasetFactory = Callable[..., Any]

__all__ = [
    "register_dataset",
    "get_dataset",
    "list_datasets",
    "dataset_info",
    "available_dataset_groups",
    "normalize_dataset_name",
    "has_dataset",
    "RegistryEntry",
]


@dataclass
class RegistryEntry:
    name: str
    factory: DatasetFactory
    aliases: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


# Module-level registry.  Populated by :func:`register_dataset` and by
# ``tgraphx/datasets/__init__.py`` at import time.  Keys are lowercase
# canonical names.
_REGISTRY: Dict[str, RegistryEntry] = {}
_ALIASES: Dict[str, str] = {}  # alias → canonical


def normalize_dataset_name(name: str) -> str:
    """Normalise a name for registry lookup (lowercase, strip)."""
    return name.strip().lower()


def register_dataset(
    name: str,
    factory: DatasetFactory,
    aliases: Optional[Sequence[str]] = None,
    tags: Optional[Sequence[str]] = None,
    metadata: Optional[Dict[str, str]] = None,
    overwrite: bool = False,
) -> RegistryEntry:
    """Register ``factory`` under ``name``.

    Args:
        name: Canonical name (case-insensitive at lookup time).
        factory: Callable accepting keyword arguments and returning a
            dataset instance.
        aliases: Alternate names the user can pass.
        tags: Free-form labels used by :func:`list_datasets` filtering.
        metadata: Static info shown by :func:`dataset_info`; values
            should be plain strings (``"description"``, ``"task"``,
            ``"requires"``).
        overwrite: If ``True``, replace an existing entry with the same
            name; otherwise raise ``KeyError``.
    """
    canonical = normalize_dataset_name(name)
    if canonical in _REGISTRY and not overwrite:
        raise KeyError(
            f"Dataset name {name!r} is already registered. "
            f"Pass overwrite=True to replace."
        )
    entry = RegistryEntry(
        name=canonical,
        factory=factory,
        aliases=[normalize_dataset_name(a) for a in (aliases or [])],
        tags=[t.strip().lower() for t in (tags or [])],
        metadata=dict(metadata or {}),
    )
    _REGISTRY[canonical] = entry
    for alias in entry.aliases:
        _ALIASES[alias] = canonical
    return entry


def has_dataset(name: str) -> bool:
    canonical = normalize_dataset_name(name)
    return canonical in _REGISTRY or canonical in _ALIASES


def _resolve(name: str) -> RegistryEntry:
    canonical = normalize_dataset_name(name)
    if canonical in _REGISTRY:
        return _REGISTRY[canonical]
    if canonical in _ALIASES:
        return _REGISTRY[_ALIASES[canonical]]
    # Unknown — provide suggestions.
    pool = list(_REGISTRY.keys()) + list(_ALIASES.keys())
    suggestions = difflib.get_close_matches(canonical, pool, n=3, cutoff=0.5)
    hint = f"  Did you mean: {', '.join(suggestions)}?\n" if suggestions else ""
    raise DatasetNotFoundError(
        f"Unknown dataset {name!r}.\n{hint}"
        f"Use list_datasets() to see registered datasets."
    )


def get_dataset(name: str, **kwargs: Any) -> Any:
    """Construct the dataset registered under ``name``.

    Optional dependencies are imported only inside the factory — this
    call is the first place an adapter for PyG/DGL/OGB/torchvision
    will try to import its upstream package.
    """
    entry = _resolve(name)
    return entry.factory(**kwargs)


# User-friendly aliases (v1.4.0+)
# Maps short, LLM-predictable names to canonical registry keys.
_FRIENDLY_ALIASES: Dict[str, str] = {
    "mnist": "torchvision:mnist_patch",
    "mnist_graph": "torchvision:mnist_patch",
    "mnist_patch": "torchvision:mnist_patch",
    "mnist_class_graph": "torchvision:mnist_patch",
    "fashion_mnist": "torchvision:fashion_mnist_patch",
    "kmnist": "torchvision:kmnist_patch",
    "cifar10": "torchvision:cifar10_patch",
    "cifar10_patch": "torchvision:cifar10_patch",
    "cifar10_patch_graph": "torchvision:cifar10_patch",
    "cifar100": "torchvision:cifar100_patch",
    "svhn": "torchvision:svhn_patch",
    "stl10": "torchvision:stl10_patch",
    "cora": "pyg:planetoid/cora",
    "planetoid/cora": "pyg:planetoid/cora",
    "citeseer": "pyg:planetoid/citeseer",
    "pubmed": "pyg:planetoid/pubmed",
    "mutag": "pyg:tudataset/mutag",
    "tu/mutag": "pyg:tudataset/mutag",
    "proteins": "pyg:tudataset/proteins",
    "enzymes": "pyg:tudataset/enzymes",
    "imdb_binary": "pyg:tudataset/imdb-binary",
    "reddit_binary": "pyg:tudataset/reddit-binary",
    # Synthetic shortcuts
    "synthetic_patch": "synthetic:patch_graph",
    "synthetic_node": "synthetic:node_classification",
    "synthetic_hetero": "synthetic:hetero",
    "synthetic_temporal": "synthetic:temporal",
}


def load_dataset(name: str, **kwargs: Any) -> Any:
    """User-friendly dataset loader with short aliases (v1.4.0+).

    Resolves short names like ``"mnist_graph"``, ``"cifar10_patch"``,
    ``"cora"``, ``"mutag"`` to their canonical registry keys, then constructs
    the dataset. Falls back to :func:`get_dataset` for the canonical names.

    Args:
        name: Short or canonical dataset name. Use :func:`list_dataset_aliases`
            to discover.
        **kwargs: Forwarded to the dataset constructor (e.g. ``download=True``,
            ``train=True``, ``patch_size=8``).

    Returns:
        A TGraphX dataset wrapper.

    Raises:
        DatasetNotFoundError: If the name is unknown; the error message includes
            the closest match suggestions.
    """
    if name in _FRIENDLY_ALIASES:
        canonical = _FRIENDLY_ALIASES[name]
    elif name in _REGISTRY or normalize_dataset_name(name) in _REGISTRY:
        canonical = name
    else:
        pool = list(_FRIENDLY_ALIASES) + list(_REGISTRY)
        suggestion = difflib.get_close_matches(name, pool, n=3, cutoff=0.5)
        hint = f"  Did you mean: {', '.join(suggestion)}?\n" if suggestion else ""
        raise DatasetNotFoundError(
            f"Unknown dataset {name!r}.\n{hint}"
            f"Use list_dataset_aliases() for friendly names "
            "or list_datasets() for canonical names."
        )
    return get_dataset(canonical, **kwargs)


def list_dataset_aliases() -> Dict[str, str]:
    """Return the user-friendly dataset alias map (v1.4.0+)."""
    return dict(_FRIENDLY_ALIASES)


def dataset_info(name: str) -> Dict[str, Any]:
    """Return registry metadata for ``name`` *without* constructing it."""
    entry = _resolve(name)
    return {
        "name": entry.name,
        "aliases": list(entry.aliases),
        "tags": list(entry.tags),
        "metadata": dict(entry.metadata),
    }


def list_datasets(
    tags: Optional[Sequence[str]] = None,
    include_aliases: bool = False,
) -> List[str]:
    """List registered dataset names.

    Args:
        tags: When provided, return only entries whose ``tags`` include
            **all** of these labels.
        include_aliases: When ``True``, also list aliases.
    """
    if tags:
        tag_set = {t.strip().lower() for t in tags}
        names = [
            entry.name for entry in _REGISTRY.values()
            if tag_set.issubset(set(entry.tags))
        ]
    else:
        names = list(_REGISTRY.keys())
    if include_aliases:
        names = sorted(set(names) | set(_ALIASES.keys()))
    else:
        names = sorted(names)
    return names


def available_dataset_groups() -> Dict[str, List[str]]:
    """Group registered datasets by their first-level prefix.

    For ``"torchvision:mnist_patch"``, the group is ``"torchvision"``.
    """
    groups: Dict[str, List[str]] = {}
    for name in _REGISTRY:
        prefix = name.split(":", 1)[0] if ":" in name else "other"
        groups.setdefault(prefix, []).append(name)
    for v in groups.values():
        v.sort()
    return dict(sorted(groups.items()))
