"""TGraphX Easy Mode — high-level, beginner-friendly API.

This package provides zero-boilerplate workflows for the most common TGraphX
tasks.  Advanced users can always access the low-level PyTorch objects through
the result objects.

Quick start::

    import tgraphx as tgx

    data = tgx.easy.synthetic_tensor_node_classification(
        num_nodes=1000, node_shape=(16, 8, 8), num_classes=10, seed=42,
    )
    result = tgx.easy.train_node_classifier(
        data, model="tensor_gcn", epochs=5, seed=42,
    )
    print(result.metrics)
    result.summary()

Design principles:
- Every inferred default is visible in ``result.config``.
- No hidden defaults.
- Every result exposes the underlying PyTorch objects (``result.model``,
  ``result.graph``, ``result.loader``, etc.) for advanced use.
- No additional imports are added to the top-level ``tgraphx`` namespace on
  import — this module is opt-in.

Stability: Beta (v1.0.1+).  Public function names are stable; return-object
fields may be extended in patch releases.
"""
from __future__ import annotations

# Exceptions
from ._exceptions import (
    TGraphXError,
    TGraphXConfigError,
    TGraphXLabelError,
    TGraphXShapeError,
    TGraphXUnknownNameError,
)

# Result and config objects
from ._results import EasyResult, EasyConfig

# Discovery
from ._discovery import (
    list_tasks,
    list_models,
    list_samplers,
    list_workflows,
    explain_workflow,
)

# Diagnostics
from ._diagnostics import check_install, doctor, show_capabilities

# Data creation
from ._data import (
    synthetic_tensor_node_classification,
    synthetic_vector_node_classification,
    synthetic_link_prediction,
    synthetic_graph_classification,
)

# Model creation
from ._models import make_tensor_node_classifier, make_vector_node_classifier

# Training workflows
from ._workflows import train_node_classifier, fit_node_classifier

__all__ = [
    # Exceptions
    "TGraphXError",
    "TGraphXConfigError",
    "TGraphXLabelError",
    "TGraphXShapeError",
    "TGraphXUnknownNameError",
    # Result / config
    "EasyResult",
    "EasyConfig",
    # Discovery
    "list_tasks",
    "list_models",
    "list_samplers",
    "list_workflows",
    "explain_workflow",
    # Diagnostics
    "check_install",
    "doctor",
    "show_capabilities",
    # Data creation
    "synthetic_tensor_node_classification",
    "synthetic_vector_node_classification",
    "synthetic_link_prediction",
    "synthetic_graph_classification",
    # Model creation
    "make_tensor_node_classifier",
    "make_vector_node_classifier",
    # Training
    "train_node_classifier",
    "fit_node_classifier",
]
