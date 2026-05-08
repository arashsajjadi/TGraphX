"""TGraphX metrics (v0.2.9).

Pure-PyTorch metrics for classification, regression, and link
prediction.  Importing this module is lightweight and does not load
optional dependencies.

The metrics in :mod:`.ogb` re-export the lazy-import OGB evaluator
wrapper from :mod:`tgraphx.datasets.ogb_wrappers`.
"""
from __future__ import annotations

from .classification import (
    accuracy,
    classification_report,
    confusion_matrix,
    precision_recall_f1,
    top_k_accuracy,
)
from .link_prediction import (
    average_precision,
    link_prediction_report,
    roc_auc,
)
from .ranking import (
    hits_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
)
from .regression import (
    mae,
    mse,
    r2_score,
    regression_report,
    rmse,
)
from .reports import (
    edge_classification_report,
    graph_classification_report,
    graph_regression_report,
    node_classification_report,
)

# Conventional aliases — match the names already used in tgraphx.training.
mean_absolute_error = mae
mean_squared_error = mse

# OGB evaluator wrapper is re-exported via :mod:`.ogb`; importing it
# here would also be safe (it's a pure Python wrapper class), but we
# keep it under the `.ogb` submodule so users explicitly opt in.
from .ogb import OGBEvaluatorWrapper  # noqa: E402


__all__ = [
    # classification
    "accuracy",
    "top_k_accuracy",
    "confusion_matrix",
    "precision_recall_f1",
    "classification_report",
    # regression
    "mae",
    "mse",
    "rmse",
    "r2_score",
    "regression_report",
    "mean_absolute_error",
    "mean_squared_error",
    # ranking / link
    "hits_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "roc_auc",
    "average_precision",
    "link_prediction_report",
    # reports
    "graph_classification_report",
    "node_classification_report",
    "edge_classification_report",
    "graph_regression_report",
    # OGB (lazy)
    "OGBEvaluatorWrapper",
]
