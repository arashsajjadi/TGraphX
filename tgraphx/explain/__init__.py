"""TGraphX explainability foundation (v0.3.0).

Diagnostic helpers — *not* causal proof.  Every method runs on CPU
unless the user moves the model and the graph to a different device,
and none of them retain an autograd graph after returning.

Public surface:

* :func:`node_feature_saliency`
* :func:`integrated_gradients`
* :func:`edge_gradient_attribution`
* :func:`edge_perturbation_attribution`
* :func:`attention_to_edge_scores`
* :func:`patch_saliency_to_image_grid`
* :func:`patch_saliency_to_volume_projection`
* :func:`export_explanation_metadata`
* :func:`export_edge_scores_csv`
* :func:`export_patch_heatmap_json`
"""
from __future__ import annotations

from .attention import attention_to_edge_scores
from .edge_attribution import (
    edge_gradient_attribution,
    edge_perturbation_attribution,
)
from .export import (
    export_edge_scores_csv,
    export_explanation_metadata,
    export_patch_heatmap_json,
)
from .integrated_gradients import integrated_gradients
from .patch_heatmap import (
    patch_saliency_to_image_grid,
    patch_saliency_to_volume_projection,
)
from .saliency import node_feature_saliency

__all__ = [
    "node_feature_saliency",
    "integrated_gradients",
    "edge_gradient_attribution",
    "edge_perturbation_attribution",
    "attention_to_edge_scores",
    "patch_saliency_to_image_grid",
    "patch_saliency_to_volume_projection",
    "export_explanation_metadata",
    "export_edge_scores_csv",
    "export_patch_heatmap_json",
]
