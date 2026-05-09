"""sklearn-like estimator/pipeline API for TGraphX.

Provides a small ``BaseGraphEstimator`` plus concrete wrappers around
TGraphX algorithms (label propagation, Node2Vec, VGAE) so users can
adopt a familiar ``fit / predict / score`` workflow.

The API intentionally mirrors a *subset* of scikit-learn's estimator
contract:

* ``fit(graph, y=None) -> self``
* ``predict(graph)``
* ``predict_proba(graph)`` (where applicable)
* ``score(graph, y) -> float``
* ``transform(graph)`` / ``fit_transform(graph)``
* ``get_params(deep=True) -> dict``
* ``set_params(**params) -> self``

Plus a tiny :class:`GraphPipeline` and dataset-split helpers.

Stability: Beta (v0.5.0+).
"""
from __future__ import annotations

from .base import BaseGraphEstimator, GraphPipeline
from .label_propagation import LabelPropagationEstimator
from .node2vec import Node2VecEstimator
from .vgae import VGAEEstimator
from .splits import (
    node_train_val_test_split,
    edge_train_val_test_split,
    temporal_train_val_test_split,
    graph_train_test_split,
)
from .early_stopping import EarlyStopping

__all__ = [
    "BaseGraphEstimator",
    "GraphPipeline",
    "LabelPropagationEstimator",
    "Node2VecEstimator",
    "VGAEEstimator",
    "node_train_val_test_split",
    "edge_train_val_test_split",
    "temporal_train_val_test_split",
    "graph_train_test_split",
    "EarlyStopping",
]
