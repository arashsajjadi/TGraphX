"""Build object-level candidate node graphs for TGraphXCandidateNodeSelector.

One TGraphX Graph per object cluster (one object hypothesis).
Every candidate detection box for that SAME object is one node, carrying
its [3, H, W] crop tensor — the native TGraphX tensor feature.

Node sources (NODE_TYPES from graph_builder, reused for consistency):
  proposal              : one per detector that detected in this cluster
  cluster               : WBF / weighted-average box
  consensus             : union box
  nms_candidate         : highest-score proposal in cluster
  soft_nms_candidate    : Gaussian-decay soft-NMS pick
  best_proposal_candidate: highest-score proposal (distinct token)

Graph topology: fully connected (all-pairs edges within the cluster).
Edge features: pairwise IoU, center-distance, area-ratio, same-class,
               same-detector, score-diff, edge-type one-hot.

Returns a list of tuples:
  (graph, image_id, cluster_id, split, candidate_sources, gt_box, gt_label)

where gt_box / gt_label are the best-matched GT for this cluster
(present for train split only — None for val/test to prevent leakage).
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
from tgraphx import Graph

from .box_ops import box_iou, weighted_box_average, union_box
from .features import (
    crop_tensor_from_image, proposal_metadata, cluster_metadata,
    edge_feature_vector,
)
from .graph_builder import NODE_TYPES, NUM_EDGE_TYPES, EDGE_TYPES
from .matching import cluster_proposals


def build_object_candidate_graphs(
    image: torch.Tensor,                # [3, H, W] float in [0, 1]
    image_id: str,
    image_size: Tuple[int, int],        # (H, W)
    detector_results: List[Any],        # list of DetectionResult | None
    detector_names: List[str],
    class_names: List[str],
    *,
    gt_boxes: Optional[torch.Tensor] = None,   # [G, 4] ground-truth
    gt_labels: Optional[torch.Tensor] = None,  # [G] ground-truth class ids
    iou_cluster: float = 0.5,
    iou_match: float = 0.5,
    crop_size: int = 128,
    max_proposals_per_detector: int = 5,
    split: str = "train",
) -> List[Tuple]:
    """Build one small TGraphX Graph per object cluster in an image.

    Each returned tuple:
      (graph, image_id, cluster_id, split, candidate_sources, gt_box, gt_label)

    If an image has K object clusters, K tuples are returned.
    The model receives each graph independently and selects the best node.
    """
    num_detectors = len(detector_names)
    num_classes   = len(class_names)

    # ── Collect proposals from all detectors ─────────────────────────────
    boxes_list, scores_list, labels_list, det_ids_list = [], [], [], []
    for d_idx, res in enumerate(detector_results):
        if res is None or res.num_detections() == 0:
            continue
        keep = min(res.boxes_xyxy.shape[0], max_proposals_per_detector)
        for li in range(keep):
            name = res.labels[li]
            if name in class_names:
                label = class_names.index(name)
            else:
                label = int(res.label_ids[li].item()) if res.label_ids.numel() > li else 0
            boxes_list.append(res.boxes_xyxy[li])
            scores_list.append(float(res.scores[li].item()))
            labels_list.append(label)
            det_ids_list.append(d_idx)

    if not boxes_list:
        return []

    all_boxes   = torch.stack(boxes_list)                         # [P, 4]
    all_scores  = torch.tensor(scores_list, dtype=torch.float32)  # [P]
    all_labels  = torch.tensor(labels_list, dtype=torch.long)     # [P]
    all_det_ids = torch.tensor(det_ids_list, dtype=torch.long)    # [P]

    # ── Cluster proposals across detectors ───────────────────────────────
    cluster_id = cluster_proposals(
        all_boxes, all_labels, all_det_ids,
        iou_threshold=iou_cluster, require_same_class=False,
    )
    num_clusters = int(cluster_id.max().item()) + 1 if cluster_id.numel() > 0 else 0

    # ── One graph per cluster ─────────────────────────────────────────────
    result: List[Tuple] = []
    for c in range(num_clusters):
        cluster_mask = cluster_id == c
        if not cluster_mask.any():
            continue

        c_boxes  = all_boxes[cluster_mask]    # [K, 4]
        c_scores = all_scores[cluster_mask]   # [K]
        c_labels = all_labels[cluster_mask]   # [K]
        c_dets   = all_det_ids[cluster_mask]  # [K]

        # Cluster-level statistics
        wbf_box      = weighted_box_average(c_boxes, c_scores)
        union_b      = union_box(c_boxes)
        mean_score   = float(c_scores.mean().item())
        max_score    = float(c_scores.max().item())
        n_supporting = int(cluster_mask.sum().item())
        det_diversity = int(c_dets.unique().numel()) / max(num_detectors, 1)
        best_in_c    = int(c_scores.argmax().item())
        cluster_lbl  = int(c_labels[best_in_c].item())

        # WBF score: same formula as baselines.weighted_boxes_fusion
        # score = mean(contributors) * min(1.0, N / 3)  — support-weighted
        # This makes graph::cluster node score identical to external WBF score.
        wbf_score = mean_score * min(1.0, n_supporting / 3.0)

        # ── Candidate nodes ───────────────────────────────────────────────
        node_crops:   List[torch.Tensor] = []
        node_boxes:   List[torch.Tensor] = []
        node_scores:  List[float]        = []
        node_labels:  List[int]          = []
        node_types_l: List[int]          = []
        node_det_ids: List[int]          = []
        node_sources: List[str]          = []

        # One proposal node per detector (highest-score in this cluster)
        seen_dets = set()
        for ki in c_scores.argsort(descending=True).tolist():
            d = int(c_dets[ki].item())
            if d in seen_dets:
                continue
            seen_dets.add(d)
            box   = c_boxes[ki]
            score = float(c_scores[ki].item())
            lbl   = int(c_labels[ki].item())
            node_crops.append(crop_tensor_from_image(image, box, crop_size))
            node_boxes.append(box)
            node_scores.append(score)
            node_labels.append(lbl)
            node_types_l.append(NODE_TYPES["proposal"])
            node_det_ids.append(d)
            node_sources.append(detector_names[d])

        def _add_fusion(box, score, lbl, det_id, ntype, src_name):
            node_crops.append(crop_tensor_from_image(image, box, crop_size))
            node_boxes.append(box)
            node_scores.append(score)
            node_labels.append(lbl)
            node_types_l.append(ntype)
            node_det_ids.append(det_id)
            node_sources.append(src_name)

        # WBF / cluster node — score matches baselines.weighted_boxes_fusion formula
        _add_fusion(wbf_box, wbf_score, cluster_lbl, -1, NODE_TYPES["cluster"], "wbf")

        # Union / consensus node
        _add_fusion(union_b, wbf_score, cluster_lbl, -1, NODE_TYPES["consensus"], "union")

        # NMS node: highest-score proposal in cluster
        nms_ki    = int(c_scores.argmax().item())
        nms_box   = c_boxes[nms_ki]
        nms_score = float(c_scores[nms_ki].item())
        nms_lbl   = int(c_labels[nms_ki].item())
        nms_det   = int(c_dets[nms_ki].item())
        _add_fusion(nms_box, nms_score, nms_lbl, nms_det, NODE_TYPES["nms_candidate"], "nms")

        # Soft-NMS node: Gaussian decay by IoU with NMS box
        _sigma = 0.5
        _ix1 = torch.max(c_boxes[:, 0], nms_box[0])
        _iy1 = torch.max(c_boxes[:, 1], nms_box[1])
        _ix2 = torch.min(c_boxes[:, 2], nms_box[2])
        _iy2 = torch.min(c_boxes[:, 3], nms_box[3])
        _inter = (_ix2 - _ix1).clamp(0) * (_iy2 - _iy1).clamp(0)
        _area_c = (c_boxes[:, 2] - c_boxes[:, 0]) * (c_boxes[:, 3] - c_boxes[:, 1])
        _area_n = (nms_box[2] - nms_box[0]) * (nms_box[3] - nms_box[1])
        _iou_v  = (_inter / (_area_c + _area_n - _inter + 1e-6)).clamp(0, 1)
        soft_scores = c_scores * torch.exp(-(_iou_v ** 2) / _sigma)
        snms_ki     = int(soft_scores.argmax().item())
        snms_box    = c_boxes[snms_ki]
        snms_score  = float(soft_scores[snms_ki].item())
        snms_lbl    = int(c_labels[snms_ki].item())
        snms_det    = int(c_dets[snms_ki].item())
        _add_fusion(snms_box, snms_score, snms_lbl, snms_det,
                    NODE_TYPES["soft_nms_candidate"], "soft_nms")

        # BestProposal node: distinct token for the top-score box
        _add_fusion(nms_box, nms_score, nms_lbl, nms_det,
                    NODE_TYPES["best_proposal_candidate"], "best_proposal")

        # ── Stack node tensors ────────────────────────────────────────────
        N            = len(node_boxes)
        node_feats   = torch.stack(node_crops)                         # [N, 3, H, W]
        node_box_t   = torch.stack(node_boxes)                         # [N, 4]
        node_score_t = torch.tensor(node_scores, dtype=torch.float32)  # [N]
        node_label_t = torch.tensor(node_labels, dtype=torch.long)     # [N]
        node_type_t  = torch.tensor(node_types_l, dtype=torch.long)    # [N]
        node_det_t   = torch.tensor(node_det_ids, dtype=torch.long)    # [N]
        # All nodes belong to cluster 0 in this object-level graph
        cluster_of_t = torch.zeros(N, dtype=torch.long)                # [N]

        # Node metadata vectors (same layout as image-level builder)
        node_metas: List[torch.Tensor] = []
        for i in range(N):
            nt  = node_types_l[i]
            det = node_det_ids[i]
            if nt == NODE_TYPES["proposal"]:
                node_metas.append(proposal_metadata(
                    node_boxes[i], node_scores[i], node_labels[i], det,
                    num_detectors, num_classes, image_size))
            else:
                node_metas.append(cluster_metadata(
                    node_boxes[i], mean_score, max_score, n_supporting,
                    det_diversity, num_detectors, num_classes, node_labels[i], image_size))
        node_meta_t = torch.stack(node_metas)  # [N, D_meta]

        # ── Fully-connected edges ─────────────────────────────────────────
        src_l: List[int] = []
        dst_l: List[int] = []
        ef_l:  List[torch.Tensor] = []
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                is_prop_i = node_types_l[i] == NODE_TYPES["proposal"]
                is_prop_j = node_types_l[j] == NODE_TYPES["proposal"]
                if is_prop_i and is_prop_j:
                    et = EDGE_TYPES.get("detector_agreement", 2)
                elif is_prop_i or is_prop_j:
                    et = EDGE_TYPES.get("proposal_to_cluster", 0)
                else:
                    et = EDGE_TYPES.get("spatial_overlap", 3)
                ef_l.append(edge_feature_vector(
                    node_boxes[i], node_boxes[j],
                    node_scores[i], node_scores[j],
                    node_labels[i], node_labels[j],
                    node_det_ids[i], node_det_ids[j],
                    et, NUM_EDGE_TYPES,
                ))
                src_l.append(i); dst_l.append(j)

        if src_l:
            edge_index = torch.tensor([src_l, dst_l], dtype=torch.long)
            edge_attr  = torch.stack(ef_l)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr  = torch.zeros(0, 6 + NUM_EDGE_TYPES)

        # ── GT assignment: match cluster WBF box to GT ────────────────────
        gt_box_c = gt_label_c = None
        if gt_boxes is not None and gt_boxes.numel() > 0 and gt_labels is not None:
            ious_gt = box_iou(wbf_box.unsqueeze(0), gt_boxes)[0]  # [G]
            best_gt_iou, best_gt_idx = ious_gt.max(dim=0)
            if best_gt_iou.item() >= iou_match:
                gt_box_c   = gt_boxes[best_gt_idx].unsqueeze(0)   # [1, 4]
                gt_label_c = gt_labels[best_gt_idx].unsqueeze(0)  # [1]

        # ── Assemble TGraphX Graph ────────────────────────────────────────
        graph_meta: dict = {
            "node_metadata":     node_meta_t,
            "node_types":        node_type_t,
            "node_box":          node_box_t,
            "node_score":        node_score_t,
            "node_label":        node_label_t,
            "node_det_ids":      node_det_t,
            "cluster_of_raw":    cluster_of_t,   # all zeros — one cluster per graph
            "proposal_det_ids":  node_det_t,     # alias for V3 slot mapper
            "image_id":          image_id,
            "cluster_id":        c,
            "candidate_sources": node_sources,
        }
        # GT only stored for training (never inside inference features)
        if split == "train" and gt_box_c is not None:
            graph_meta["gt_boxes"]  = gt_box_c
            graph_meta["gt_labels"] = gt_label_c

        g = Graph(
            node_features=node_feats,
            edge_index=edge_index,
            edge_attr=edge_attr,
            metadata=graph_meta,
        )
        result.append((g, image_id, c, split, node_sources, gt_box_c, gt_label_c))

    return result
