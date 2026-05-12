"""TGraphX fusion inference: turn a trained model + detection graph
into final detection boxes/scores/labels for an image."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from .graph_builder import DetectionGraphMeta, NODE_TYPES


@torch.no_grad()
def fuse_with_model(
    model,
    graph,
    meta: DetectionGraphMeta,
    *,
    keep_threshold: float = 0.5,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Run the trained fusion model on one graph and return final detections.

    Returns a dict with ``boxes_xyxy``, ``scores``, ``labels``.
    """
    model.eval()
    if meta.num_clusters == 0:
        return {"boxes_xyxy": torch.zeros(0, 4),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.long)}

    g = graph.to(device)
    out = model(g)

    # Score = sigmoid(objectness logits); class = argmax(class logits); box = cluster + reg
    mask = (meta.node_types == NODE_TYPES["cluster"]).to(device)
    obj_logits = out["objectness_logits"][mask]
    cls_logits = out["class_logits"][mask]
    box_reg = out["box_reg"][mask]
    scores = torch.sigmoid(obj_logits).cpu()
    labels = cls_logits.argmax(dim=1).cpu()
    refined = meta.cluster_boxes.to(device) + box_reg
    refined = refined.cpu()

    keep = scores >= keep_threshold
    return {
        "boxes_xyxy": refined[keep],
        "scores": scores[keep],
        "labels": labels[keep],
    }
