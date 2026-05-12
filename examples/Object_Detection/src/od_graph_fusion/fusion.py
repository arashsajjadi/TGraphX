"""TGraphX fusion inference.

v1.1 (faithfulness rebuild):
- ``fusion_mode="selector"`` (default): predict a selection score for every
  candidate node (proposal/cluster/consensus). For each cluster, pick the
  candidate node with the highest score. **Copy the box from the chosen
  node verbatim** — no box regression. This guarantees TGraphX is
  lower-bounded by the best candidate per cluster.
- ``fusion_mode="refiner"``: apply learned box offsets on top of the selected
  candidate box. **Disabled by default** until the selector is validated.
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
    apply_box_regression: bool = False,
) -> Dict[str, torch.Tensor]:
    """Run the trained fusion model on one graph and return final detections.

    Returns a dict with ``boxes_xyxy``, ``scores``, ``labels``.
    """
    model.eval()
    g = graph.to(device)
    out = model(g)
    obj_logits = out["objectness_logits"]
    scores_all = torch.sigmoid(obj_logits).cpu()
    box_reg = out["box_reg"].cpu()

    node_types = meta.node_types
    node_box = graph.metadata.get("node_box")
    node_label = graph.metadata.get("node_label")
    if node_box is None or node_label is None:
        # Fallback to cluster nodes if metadata is missing (very old graphs)
        cluster_mask = (node_types == NODE_TYPES["cluster"])
        keep = (scores_all > keep_threshold) & cluster_mask
        return {
            "boxes_xyxy": meta.cluster_boxes[keep[:meta.num_clusters]] if meta.cluster_boxes.numel() > 0 else torch.zeros(0, 4),
            "scores": scores_all[keep], "labels": meta.cluster_labels[keep[:meta.num_clusters]],
        }

    # Candidate eligibility: proposal | cluster | consensus
    cand_mask = (
        (node_types == NODE_TYPES["proposal"])
        | (node_types == NODE_TYPES["cluster"])
        | (node_types == NODE_TYPES["consensus"])
    )

    # For each cluster, pick the candidate with the highest score
    cluster_of = meta.cluster_of_node
    final_boxes: List[torch.Tensor] = []
    final_scores: List[float] = []
    final_labels: List[int] = []

    for c in range(meta.num_clusters):
        # Eligible nodes for this cluster
        eligible = (cluster_of == c) & cand_mask
        if not eligible.any():
            continue
        sc = scores_all[eligible]
        if sc.max().item() < keep_threshold:
            continue
        # Pick best
        eligible_idx = eligible.nonzero(as_tuple=False).squeeze(-1)
        best_local = int(sc.argmax().item())
        chosen_idx = int(eligible_idx[best_local].item())
        box = node_box[chosen_idx].clone()
        if fusion_mode == "refiner" and apply_box_regression:
            box = box + box_reg[chosen_idx]
        # Class: from class head (more flexible) — but in selector mode prefer
        # the candidate node's known label.
        cls_logits = out["class_logits"][chosen_idx].cpu()
        if fusion_mode == "selector":
            label = int(node_label[chosen_idx].item())
        else:
            label = int(cls_logits.argmax().item())
        final_boxes.append(box)
        final_scores.append(float(sc.max().item()))
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
