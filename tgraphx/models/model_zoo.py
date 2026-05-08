"""TGraphX model-zoo registry.

Shipped layers (v0.3.0):

* :class:`tgraphx.layers.GCNConv` (vector, stable).
* :class:`tgraphx.layers.GATv2Conv` (vector, stable).
* :class:`tgraphx.layers.APPNP` (vector propagation, stable).
* :func:`tgraphx.layers.global_sum_pool` / :func:`global_mean_pool` /
  :func:`global_max_pool`.

Use :func:`list_layers()` to list registered layer classes;
:func:`make_zoo_layer(name, **kwargs)` to construct one by name.
"""
from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from ..layers.appnp import APPNP
from ..layers.gatv2 import GATv2Conv
from ..layers.pooling import global_max_pool, global_mean_pool, global_sum_pool
from ..layers.vector_gcn import GCNConv

_ZOO: Dict[str, Any] = {
    "gcn_conv": GCNConv,
    "gatv2": GATv2Conv,
    "appnp": APPNP,
    "global_sum_pool": global_sum_pool,
    "global_mean_pool": global_mean_pool,
    "global_max_pool": global_max_pool,
}


def list_layers() -> list[str]:
    """Return registered model-zoo layer / pooling names."""
    return sorted(_ZOO.keys())


def make_zoo_layer(name: str, **kwargs: Any):
    """Construct a layer by name.  Pooling helpers are returned as-is."""
    if name not in _ZOO:
        raise KeyError(
            f"Unknown zoo layer {name!r}. Available: {list_layers()}"
        )
    obj = _ZOO[name]
    if isinstance(obj, type):
        return obj(**kwargs)
    return obj  # pooling functions


__all__ = ["list_layers", "make_zoo_layer", "GCNConv", "GATv2Conv", "APPNP",
           "global_sum_pool", "global_mean_pool", "global_max_pool"]
