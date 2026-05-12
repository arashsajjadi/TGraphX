"""TGraphX fusion inference.

v3 (guarded residual selector):

The previous v2 selector picked the candidate node with the highest
``sigmoid(objectness_logits)``. With a randomly-initialized or
under-trained model, those scores are noise, and TGraphX collapses
below "pick by detector confidence". The guarded residual selector
fixes this by scoring candidates as:

    final_score = base_score + alpha * (sigmoid(obj_logits) - 0.5)

where ``base_score`` is the detector's own confidence for that node
(or the cluster mean for cluster/consensus nodes). With ``alpha`` small,
the ranking is dominated by the well-calibrated base scores; the graph
residual can only push slightly up or down. As training proves the residual
helps, ``alpha`` can be increased on validation.

This guarantees: in the no-learning regime, TGraphX ≥ "pick by detector
confidence" baseline, instead of collapsing to noise.

Modes:
- ``score_mode="base_only"``: final_score = base_score (TGraphX picks the
  candidate but uses its detector score verbatim; useful for ablations).
- ``score_mode="residual"``: final_score = base + alpha * (sigmoid(logits) - 0.5).
- ``score_mode="logits"``: final_score = sigmoid(obj_logits) (legacy v2 behaviour).

Box modes:
- ``fusion_mode="selector"``: copy the chosen node's box verbatim (default).
- ``fusion_mode="refiner"``: add ``box_reg`` offsets to the chosen node's box
  (off by default; enable only after selector quality is validated).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .graph_builder import DetectionGraphMeta, NODE_TYPES


@torch.no_grad()
def fuse_with_model(
    model,
    graph,
    meta: DetectionGraphMeta,
    *,
    keep_threshold: float = 0.0,
    device: str = "cpu",
    fusion_mode: str = "selector",
    score_mode: str = "residual",
    residual_alpha: float = 0.1,
    apply_box_regression: bool = False,
) -> Dict[str, torch.Tensor]:
    """Run the trained fusion model on one graph and return final detections."""
    model.eval()
    g = graph.to(device)
    out = model(g)
    # TGraphXSourceRouter returns "quality_logits"; legacy returns "objectness_logits"
    obj_logits = out.get("quality_logits", out.get("objectness_logits"))
    resid = (torch.sigmoid(obj_logits) - 0.5).cpu()
    box_reg = out["box_reg"].cpu()

    node_types = meta.node_types
    node_box = graph.metadata.get("node_box")
    node_label = graph.metadata.get("node_label")
    node_score = graph.metadata.get("node_score")
    if node_box is None or node_label is None or node_score is None:
        return {"boxes_xyxy": torch.zeros(0, 4),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.long)}

    cand_mask = (
        (node_types == NODE_TYPES["proposal"])
        | (node_types == NODE_TYPES["cluster"])
        | (node_types == NODE_TYPES["consensus"]) \
        | (node_types == NODE_TYPES["nms_candidate"])
    )

    # Build the scoring tensor used for in-cluster ranking
    if score_mode == "base_only":
        ranking = node_score.clone()
    elif score_mode == "logits":
        ranking = torch.sigmoid(obj_logits).cpu()
    else:  # "residual"
        ranking = node_score + residual_alpha * resid

    cluster_of = meta.cluster_of_node
    final_boxes: List[torch.Tensor] = []
    final_scores: List[float] = []
    final_labels: List[int] = []

    for c in range(meta.num_clusters):
        eligible = (cluster_of == c) & cand_mask
        if not eligible.any():
            continue
        eligible_idx = eligible.nonzero(as_tuple=False).squeeze(-1)
        scores_in = ranking[eligible_idx]
        best_local = int(scores_in.argmax().item())
        chosen_idx = int(eligible_idx[best_local].item())
        chosen_score = float(scores_in[best_local].item())
        if chosen_score < keep_threshold:
            continue
        box = node_box[chosen_idx].clone()
        if fusion_mode == "refiner" and apply_box_regression:
            box = box + box_reg[chosen_idx]
        # Final reported score: keep base + residual (well-calibrated).
        # In base_only mode, scores are already detector confidences.
        final_score = chosen_score
        # Class: in selector mode prefer the candidate's known label.
        if fusion_mode == "selector":
            label = int(node_label[chosen_idx].item())
        else:
            cls_logits = out["class_logits"][chosen_idx].cpu()
            label = int(cls_logits.argmax().item())
        final_boxes.append(box)
        final_scores.append(final_score)
        final_labels.append(label)

    if not final_boxes:
        return {"boxes_xyxy": torch.zeros(0, 4),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.long)}

    return {
        "boxes_xyxy": torch.stack(final_boxes, dim=0),
        "scores": torch.tensor(final_scores, dtype=torch.float32),
        "labels": torch.tensor(final_labels, dtype=torch.long),
    }
