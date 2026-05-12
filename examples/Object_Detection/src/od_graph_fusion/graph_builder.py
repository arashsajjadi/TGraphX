"""Build heterogeneous detection graphs for a single image and a dataset.

Each image becomes one graph with:
  - proposal nodes (one per detector prediction)
  - candidate cluster nodes (one per IoU-grouped consensus)
  - optional consensus / context nodes
Edges encode IoU, class agreement, detector agreement, and same-detector
suppression. Tensor crop features remain ``[3, H, W]`` and live in
``graph.node_features``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from tgraphx import Graph

from .box_ops import (
    box_iou, weighted_box_average, union_box, normalize_boxes,
)
from .features import (
    crop_tensor_from_image, proposal_metadata, cluster_metadata,
    edge_feature_vector,
)
from .matching import cluster_proposals, match_to_gt


# Edge type IDs (also encoded as one-hot inside edge feature vectors)
EDGE_TYPES = {
    "proposal_to_cluster":      0,
    "proposal_to_consensus":    1,
    "detector_agreement":       2,
    "spatial_overlap":          3,
    "class_agreement":          4,
    "same_detector_suppression":5,
    "cluster_to_context":       6,
    "proposal_to_context":      7,
}
NUM_EDGE_TYPES = len(EDGE_TYPES)


# Node type IDs (encoded inside metadata so a single TGraphX Graph can hold
# heterogeneous nodes).
NODE_TYPES = {"proposal": 0, "cluster": 1, "consensus": 2, "context": 3, "nms_candidate": 4}
NUM_NODE_TYPES = len(NODE_TYPES)


@dataclass
class DetectionGraphMeta:
    """Per-graph bookkeeping used by training/evaluation."""
    image_id: str
    image_size: Tuple[int, int]
    num_proposals: int
    num_clusters: int
    num_consensus: int
    has_context: bool
    detector_names: List[str]
    class_names: List[str]
    node_types: torch.Tensor          # [N] long
    node_to_proposal_index: torch.Tensor  # [N] long, -1 for non-proposal
    cluster_of_node: torch.Tensor     # [N] long, -1 if node not in a cluster
    cluster_boxes: torch.Tensor       # [num_clusters, 4]
    cluster_labels: torch.Tensor      # [num_clusters] long
    proposal_boxes: torch.Tensor      # [num_proposals, 4]
    proposal_scores: torch.Tensor     # [num_proposals]
    proposal_labels: torch.Tensor     # [num_proposals] long
    proposal_detector_ids: torch.Tensor   # [num_proposals] long
    cluster_score: torch.Tensor       # [num_clusters] mean confidence
    targets: Optional[Dict[str, torch.Tensor]] = None   # filled if GT available


def build_detection_graph(
    image: torch.Tensor,
    image_id: str,
    image_size: Tuple[int, int],
    detector_results: List[Any],
    detector_names: List[str],
    class_names: List[str],
    *,
    gt_boxes: Optional[torch.Tensor] = None,
    gt_labels: Optional[torch.Tensor] = None,
    iou_cluster: float = 0.5,
    iou_match: float = 0.5,
    crop_size: int = 64,
    max_proposals: int = 64,
    include_context_node: bool = True,
    include_consensus_nodes: bool = True,
    is_training: bool = False,
) -> Tuple[Graph, DetectionGraphMeta]:
    """Construct a TGraphX detection graph for one image.

    Args:
        image: ``[C, H, W]`` float in [0, 1].
        detector_results: list of ``DetectionResult`` objects.
        gt_boxes / gt_labels: optional ground-truth. Only used to build training
            targets — never used to construct edges or filter proposals.
        is_training: if True, populate per-node targets. If False, leave the
            graph leakage-free.
    """
    num_detectors = len(detector_names)
    num_classes = len(class_names)
    device = image.device

    # Collect all proposals
    boxes_list, scores_list, labels_list, det_ids_list = [], [], [], []
    for d_idx, res in enumerate(detector_results):
        if res is None or res.num_detections() == 0:
            continue
        keep = min(res.boxes_xyxy.shape[0], max_proposals)
        boxes_list.append(res.boxes_xyxy[:keep])
        scores_list.append(res.scores[:keep])
        # Map label name → class index using class_names; fall back to label_id
        for li in range(keep):
            name = res.labels[li]
            if name in class_names:
                labels_list.append(class_names.index(name))
            else:
                labels_list.append(int(res.label_ids[li].item()) if res.label_ids.numel() > li else 0)
        det_ids_list.extend([d_idx] * keep)

    if not boxes_list:
        return _empty_graph(image, image_id, image_size, detector_names,
                            class_names, crop_size, num_detectors, num_classes,
                            include_context_node)

    proposal_boxes = torch.cat(boxes_list, dim=0)
    proposal_scores = torch.cat(scores_list, dim=0)
    proposal_labels = torch.tensor(labels_list, dtype=torch.long)
    proposal_det_ids = torch.tensor(det_ids_list, dtype=torch.long)

    # Cluster proposals across detectors
    cluster_id = cluster_proposals(
        proposal_boxes, proposal_labels, proposal_det_ids,
        iou_threshold=iou_cluster, require_same_class=False,
    )
    num_clusters = int(cluster_id.max().item() + 1) if cluster_id.numel() > 0 else 0

    # Build per-cluster stats
    cluster_boxes_list = []
    cluster_label_list = []
    cluster_mean_scores = []
    cluster_max_scores = []
    cluster_n_supporting = []
    cluster_detector_diversity = []
    for c in range(num_clusters):
        m = cluster_id == c
        cb = weighted_box_average(proposal_boxes[m], proposal_scores[m])
        cluster_boxes_list.append(cb)
        # Mode label (highest-scoring proposal in cluster)
        scores_in_c = proposal_scores[m]
        labels_in_c = proposal_labels[m]
        if scores_in_c.numel() > 0:
            cluster_label_list.append(int(labels_in_c[scores_in_c.argmax()].item()))
            cluster_mean_scores.append(scores_in_c.mean().item())
            cluster_max_scores.append(scores_in_c.max().item())
        else:
            cluster_label_list.append(0)
            cluster_mean_scores.append(0.0)
            cluster_max_scores.append(0.0)
        cluster_n_supporting.append(int(m.sum().item()))
        cluster_detector_diversity.append(
            int(proposal_det_ids[m].unique().numel()) / max(num_detectors, 1)
        )

    if num_clusters > 0:
        cluster_boxes = torch.stack(cluster_boxes_list, dim=0)
    else:
        cluster_boxes = torch.zeros(0, 4)
    cluster_labels = torch.tensor(cluster_label_list, dtype=torch.long)
    cluster_mean = torch.tensor(cluster_mean_scores, dtype=torch.float32)

    # Build consensus nodes — one weighted-average box per cluster.
    # We deliberately keep these as separate nodes so the model can compare
    # raw proposals, the cluster centroid, and the WBF-like consensus.
    consensus_nodes_enabled = include_consensus_nodes and num_clusters > 0

    # --- Compose node features ---
    metadata_dim = 8 + num_detectors + num_classes
    proposal_tensor_features: List[torch.Tensor] = []
    proposal_metadata_list: List[torch.Tensor] = []
    for i in range(proposal_boxes.shape[0]):
        crop = crop_tensor_from_image(image, proposal_boxes[i], crop_size=crop_size)
        proposal_tensor_features.append(crop)
        proposal_metadata_list.append(proposal_metadata(
            proposal_boxes[i], float(proposal_scores[i].item()),
            int(proposal_labels[i].item()), int(proposal_det_ids[i].item()),
            num_detectors, num_classes, image_size,
        ))

    cluster_tensor_features: List[torch.Tensor] = []
    cluster_metadata_list: List[torch.Tensor] = []
    for c in range(num_clusters):
        crop = crop_tensor_from_image(image, cluster_boxes[c], crop_size=crop_size)
        cluster_tensor_features.append(crop)
        cluster_metadata_list.append(cluster_metadata(
            cluster_boxes[c], cluster_mean_scores[c], cluster_max_scores[c],
            cluster_n_supporting[c], cluster_detector_diversity[c],
            num_detectors, num_classes, cluster_label_list[c], image_size,
        ))

    consensus_tensor_features: List[torch.Tensor] = []
    consensus_metadata_list: List[torch.Tensor] = []
    if consensus_nodes_enabled:
        for c in range(num_clusters):
            # Use the union box as a complementary view
            m = cluster_id == c
            ub = union_box(proposal_boxes[m])
            crop = crop_tensor_from_image(image, ub, crop_size=crop_size)
            consensus_tensor_features.append(crop)
            consensus_metadata_list.append(cluster_metadata(
                ub, cluster_mean_scores[c], cluster_max_scores[c],
                cluster_n_supporting[c], cluster_detector_diversity[c],
                num_detectors, num_classes, cluster_label_list[c], image_size,
            ))

    # Context node — downsampled global image
    if include_context_node:
        ctx_crop = crop_tensor_from_image(
            image, torch.tensor([0, 0, image_size[1] - 1, image_size[0] - 1],
                                dtype=torch.float32),
            crop_size=crop_size,
        )
        ctx_meta = torch.zeros(metadata_dim)

    # Concatenate all nodes
    all_crops = []
    all_meta = []
    node_types = []
    node_to_proposal: List[int] = []
    node_cluster: List[int] = []

    for i in range(proposal_boxes.shape[0]):
        all_crops.append(proposal_tensor_features[i])
        all_meta.append(proposal_metadata_list[i])
        node_types.append(NODE_TYPES["proposal"])
        node_to_proposal.append(i)
        node_cluster.append(int(cluster_id[i].item()))

    cluster_node_offsets = []
    for c in range(num_clusters):
        cluster_node_offsets.append(len(all_crops))
        all_crops.append(cluster_tensor_features[c])
        all_meta.append(cluster_metadata_list[c])
        node_types.append(NODE_TYPES["cluster"])
        node_to_proposal.append(-1)
        node_cluster.append(c)

    consensus_node_offsets = []
    if consensus_nodes_enabled:
        for c in range(num_clusters):
            consensus_node_offsets.append(len(all_crops))
            all_crops.append(consensus_tensor_features[c])
            all_meta.append(consensus_metadata_list[c])
            node_types.append(NODE_TYPES["consensus"])
            node_to_proposal.append(-1)
            node_cluster.append(c)

    # ── NMS candidate nodes (one per cluster) ──────────────────────────
    # NMS node = the proposal with highest base confidence per cluster.
    # This lets TGraphX learn "keep NMS" vs "override NMS".
    nms_node_offsets = []
    if num_clusters > 0:
        for c in range(num_clusters):
            m = cluster_id == c
            if not m.any():
                nms_node_offsets.append(-1)
                continue
            # Pick the highest-score proposal in this cluster as NMS representative
            scores_in_c = proposal_scores[m]
            idx_in_c = m.nonzero(as_tuple=False).squeeze(-1)
            nms_local = int(scores_in_c.argmax().item())
            nms_global = int(idx_in_c[nms_local].item())
            nms_crop = crop_tensor_from_image(image, proposal_boxes[nms_global], crop_size=crop_size)
            nms_meta = proposal_metadata(
                proposal_boxes[nms_global], float(proposal_scores[nms_global].item()),
                int(proposal_labels[nms_global].item()),
                int(proposal_det_ids[nms_global].item()),
                num_detectors, num_classes, image_size,
            )
            nms_node_offsets.append(len(all_crops))
            all_crops.append(nms_crop)
            all_meta.append(nms_meta)
            node_types.append(NODE_TYPES["nms_candidate"])
            node_to_proposal.append(nms_global)   # points to the actual proposal node
            node_cluster.append(c)

    context_node_idx = -1
    if include_context_node:
        context_node_idx = len(all_crops)
        all_crops.append(ctx_crop)
        all_meta.append(ctx_meta)
        node_types.append(NODE_TYPES["context"])
        node_to_proposal.append(-1)
        node_cluster.append(-1)

    # Stack
    node_features = torch.stack(all_crops, dim=0)        # [N, 3, H, W]
    node_metadata = torch.stack(all_meta, dim=0)         # [N, D_meta]
    node_types_t = torch.tensor(node_types, dtype=torch.long)

    # ── Edges ────────────────────────────────────────────────────────────
    edges_src: List[int] = []
    edges_dst: List[int] = []
    edge_feats: List[torch.Tensor] = []

    def _add(s, d, et_id, box_a, box_b, sa, sb, la, lb, da, db):
        edges_src.append(s); edges_dst.append(d)
        edges_src.append(d); edges_dst.append(s)
        feat = edge_feature_vector(box_a, box_b, sa, sb, la, lb, da, db,
                                    et_id, NUM_EDGE_TYPES)
        edge_feats.append(feat); edge_feats.append(feat)

    # proposal ↔ cluster (only those that belong to the cluster)
    for i in range(proposal_boxes.shape[0]):
        c = int(cluster_id[i].item())
        if c < 0 or c >= num_clusters:
            continue
        cluster_node = cluster_node_offsets[c]
        _add(i, cluster_node, EDGE_TYPES["proposal_to_cluster"],
              proposal_boxes[i], cluster_boxes[c],
              float(proposal_scores[i].item()), float(cluster_mean[c].item()),
              int(proposal_labels[i].item()), int(cluster_labels[c].item()),
              int(proposal_det_ids[i].item()), -1)

    # proposal ↔ consensus
    if consensus_nodes_enabled:
        for i in range(proposal_boxes.shape[0]):
            c = int(cluster_id[i].item())
            if c < 0 or c >= num_clusters:
                continue
            cons_node = consensus_node_offsets[c]
            _add(i, cons_node, EDGE_TYPES["proposal_to_consensus"],
                  proposal_boxes[i], cluster_boxes[c],
                  float(proposal_scores[i].item()), float(cluster_mean[c].item()),
                  int(proposal_labels[i].item()), int(cluster_labels[c].item()),
                  int(proposal_det_ids[i].item()), -1)

    # detector_agreement / spatial_overlap / same_detector_suppression /
    # class_agreement among proposals
    P = proposal_boxes.shape[0]
    if P > 0:
        ious_all = box_iou(proposal_boxes, proposal_boxes)
        for i in range(P):
            for j in range(i + 1, P):
                iou_ij = ious_all[i, j].item()
                if iou_ij < 0.1:
                    continue
                same_det = proposal_det_ids[i].item() == proposal_det_ids[j].item()
                same_class = proposal_labels[i].item() == proposal_labels[j].item()
                if same_det:
                    et = EDGE_TYPES["same_detector_suppression"]
                elif iou_ij >= iou_cluster and same_class:
                    et = EDGE_TYPES["detector_agreement"]
                elif iou_ij >= 0.3:
                    et = EDGE_TYPES["spatial_overlap"]
                else:
                    continue
                _add(i, j, et,
                      proposal_boxes[i], proposal_boxes[j],
                      float(proposal_scores[i].item()),
                      float(proposal_scores[j].item()),
                      int(proposal_labels[i].item()),
                      int(proposal_labels[j].item()),
                      int(proposal_det_ids[i].item()),
                      int(proposal_det_ids[j].item()))
                if same_class and not same_det:
                    _add(i, j, EDGE_TYPES["class_agreement"],
                          proposal_boxes[i], proposal_boxes[j],
                          float(proposal_scores[i].item()),
                          float(proposal_scores[j].item()),
                          int(proposal_labels[i].item()),
                          int(proposal_labels[j].item()),
                          int(proposal_det_ids[i].item()),
                          int(proposal_det_ids[j].item()))

    # context edges
    if include_context_node and context_node_idx >= 0:
        # cluster_to_context for each cluster, plus proposal_to_context for each proposal
        for c in range(num_clusters):
            cn = cluster_node_offsets[c]
            edges_src.append(cn); edges_dst.append(context_node_idx)
            edges_src.append(context_node_idx); edges_dst.append(cn)
            feat = torch.zeros(6 + NUM_EDGE_TYPES)
            feat[5 + EDGE_TYPES["cluster_to_context"]] = 1.0  # one-hot type
            edge_feats.append(feat); edge_feats.append(feat)
        for i in range(P):
            edges_src.append(i); edges_dst.append(context_node_idx)
            edges_src.append(context_node_idx); edges_dst.append(i)
            feat = torch.zeros(6 + NUM_EDGE_TYPES)
            feat[5 + EDGE_TYPES["proposal_to_context"]] = 1.0
            edge_feats.append(feat); edge_feats.append(feat)

    edge_index = (torch.tensor([edges_src, edges_dst], dtype=torch.long)
                  if edges_src else torch.zeros(2, 0, dtype=torch.long))
    edge_attr = (torch.stack(edge_feats, dim=0) if edge_feats
                 else torch.zeros(0, 6 + NUM_EDGE_TYPES))

    # Build Graph. node_metadata stored in graph.metadata to keep tensor crops
    # as the primary tensor-native feature.
    metadata: Dict[str, Any] = {
        "node_metadata": node_metadata,
        "node_types": node_types_t,
        "image_id": image_id,
    }

    # Build a per-node "candidate box": the box the selector should pick if it
    # picks this node. For proposal nodes this is the proposal box; for
    # cluster nodes the WBF box; for consensus the union box; for context
    # the full image.
    node_box = torch.zeros(node_features.shape[0], 4, dtype=torch.float32)
    node_label = torch.zeros(node_features.shape[0], dtype=torch.long)
    node_score = torch.zeros(node_features.shape[0], dtype=torch.float32)
    for i in range(proposal_boxes.shape[0]):
        node_box[i] = proposal_boxes[i]
        node_label[i] = proposal_labels[i]
        node_score[i] = proposal_scores[i]
    for c, idx in enumerate(cluster_node_offsets):
        node_box[idx] = cluster_boxes[c]
        node_label[idx] = cluster_labels[c]
        node_score[idx] = cluster_mean[c]
    for c, idx in enumerate(consensus_node_offsets):
        node_box[idx] = cluster_boxes[c]
        node_label[idx] = cluster_labels[c]
        node_score[idx] = cluster_mean[c]
    # NMS candidate nodes: copy from the highest-score proposal in each cluster
    for c, idx in enumerate(nms_node_offsets):
        if idx < 0:
            continue
        src_prop = node_to_proposal[idx]  # points to the NMS-selected proposal
        if 0 <= src_prop < proposal_boxes.shape[0]:
            node_box[idx] = proposal_boxes[src_prop]
            node_label[idx] = proposal_labels[src_prop]
            node_score[idx] = proposal_scores[src_prop]
    if include_context_node and context_node_idx >= 0:
        node_box[context_node_idx] = torch.tensor(
            [0, 0, image_size[1] - 1, image_size[0] - 1], dtype=torch.float32)

    targets = None
    if is_training and gt_boxes is not None and gt_labels is not None and gt_boxes.numel() > 0:
        targets = _build_targets_full(
            node_types_t, node_box, node_label,
            cluster_node_offsets, consensus_node_offsets if consensus_nodes_enabled else [],
            gt_boxes, gt_labels, iou_match,
        )

    g = Graph(node_features=node_features, edge_index=edge_index,
              edge_attr=edge_attr, metadata=metadata)
    meta = DetectionGraphMeta(
        image_id=image_id, image_size=image_size,
        num_proposals=P, num_clusters=num_clusters,
        num_consensus=num_clusters if consensus_nodes_enabled else 0,
        has_context=include_context_node,
        detector_names=detector_names, class_names=class_names,
        node_types=node_types_t,
        node_to_proposal_index=torch.tensor(node_to_proposal, dtype=torch.long),
        cluster_of_node=torch.tensor(node_cluster, dtype=torch.long),
        cluster_boxes=cluster_boxes,
        cluster_labels=cluster_labels,
        proposal_boxes=proposal_boxes,
        proposal_scores=proposal_scores,
        proposal_labels=proposal_labels,
        proposal_detector_ids=proposal_det_ids,
        cluster_score=cluster_mean,
        targets=targets,
    )
    # Attach per-node candidate box (used for selector-mode decoding)
    g.metadata["node_box"] = node_box
    g.metadata["node_label"] = node_label
    g.metadata["node_score"] = node_score
    # V3 source-slot router metadata
    g.metadata["cluster_of_raw"] = torch.tensor(node_cluster, dtype=torch.long)
    g.metadata["nms_node_offsets"] = nms_node_offsets  # for override router
    g.metadata["proposal_det_ids"] = torch.cat([
        proposal_det_ids,                          # proposals
        torch.full((num_clusters,), -1, dtype=torch.long),  # cluster nodes
        torch.full((num_clusters if consensus_nodes_enabled else 0,), -1, dtype=torch.long),
        torch.tensor([-1] * (1 if include_context_node else 0), dtype=torch.long),
    ]) if proposal_det_ids.numel() > 0 else torch.full((node_features.shape[0],), -1, dtype=torch.long)
    return g, meta


def _empty_graph(image, image_id, image_size, detector_names, class_names,
                  crop_size, num_detectors, num_classes, include_context):
    """Return a 1-node placeholder graph (context only)."""
    if include_context:
        ctx_crop = crop_tensor_from_image(
            image, torch.tensor([0, 0, image_size[1] - 1, image_size[0] - 1],
                                dtype=torch.float32),
            crop_size=crop_size,
        )
        nf = ctx_crop.unsqueeze(0)
        node_types = torch.tensor([NODE_TYPES["context"]], dtype=torch.long)
    else:
        nf = torch.zeros(1, 3, crop_size, crop_size)
        node_types = torch.tensor([0], dtype=torch.long)
    metadata_dim = 8 + num_detectors + num_classes
    meta_t = torch.zeros(1, metadata_dim)
    g = Graph(node_features=nf,
              edge_index=torch.zeros(2, 0, dtype=torch.long),
              edge_attr=torch.zeros(0, 6 + NUM_EDGE_TYPES),
              metadata={"node_metadata": meta_t, "node_types": node_types,
                        "image_id": image_id})
    return g, DetectionGraphMeta(
        image_id=image_id, image_size=image_size,
        num_proposals=0, num_clusters=0, num_consensus=0,
        has_context=include_context,
        detector_names=detector_names, class_names=class_names,
        node_types=node_types,
        node_to_proposal_index=torch.full((1,), -1, dtype=torch.long),
        cluster_of_node=torch.full((1,), -1, dtype=torch.long),
        cluster_boxes=torch.zeros(0, 4),
        cluster_labels=torch.zeros(0, dtype=torch.long),
        proposal_boxes=torch.zeros(0, 4),
        proposal_scores=torch.zeros(0),
        proposal_labels=torch.zeros(0, dtype=torch.long),
        proposal_detector_ids=torch.zeros(0, dtype=torch.long),
        cluster_score=torch.zeros(0),
        targets=None,
    )


def _build_targets_full(
    node_types: torch.Tensor,
    node_box: torch.Tensor,            # [N, 4] per-node candidate box
    node_label: torch.Tensor,          # [N]
    cluster_offsets: List[int],
    consensus_offsets: List[int],
    gt_boxes: torch.Tensor, gt_labels: torch.Tensor,
    iou_match: float,
) -> Dict[str, torch.Tensor]:
    """Target assignment for the *selector*.

    A candidate node (proposal, cluster, or consensus) is positive if its
    candidate box has IoU >= ``iou_match`` with any GT AND the predicted
    class agrees with that GT's class (or the candidate has no class info
    yet, in which case we trust IoU).

    Positives also get a per-GT ranking: among all nodes that overlap the
    same GT, the one with the highest IoU is marked ``is_best_source=1``.
    The selector loss prefers the best-source node and discourages other
    positives, while still allowing the model to pick a runner-up when the
    best-source choice is wrong.

    Targets:
        objectness:        [N] 1.0 for positives else 0.0
        class:             [N] gt class id for positives else -1
        box_reg:           [N, 4] gt - candidate_box (offsets; only used if
                           refiner mode is enabled)
        iou:               [N] IoU with assigned GT (0 if no match)
        is_best_source:    [N] 1.0 for the single best-IoU node per GT
        candidate_mask:    [N] True for proposal/cluster/consensus (selector
                           eligibility)
    """
    N = node_types.shape[0]
    objectness = torch.zeros(N, dtype=torch.float32)
    cls_targets = torch.full((N,), -1, dtype=torch.long)
    box_reg = torch.zeros(N, 4, dtype=torch.float32)
    iou_targets = torch.zeros(N, dtype=torch.float32)
    is_best_source = torch.zeros(N, dtype=torch.float32)
    candidate_mask = (node_types == NODE_TYPES["proposal"]) \
        | (node_types == NODE_TYPES["cluster"]) \
        | (node_types == NODE_TYPES["consensus"]) \
        | (node_types == NODE_TYPES["nms_candidate"])

    if gt_boxes.numel() == 0 or not candidate_mask.any():
        return {"objectness": objectness, "class": cls_targets,
                "box_reg": box_reg, "iou": iou_targets,
                "is_best_source": is_best_source,
                "candidate_mask": candidate_mask}

    cand_idx = candidate_mask.nonzero(as_tuple=False).squeeze(-1)
    cand_boxes = node_box[cand_idx]
    cand_labels = node_label[cand_idx]
    ious = box_iou(cand_boxes, gt_boxes)  # [Nc, G]
    best_iou, best_gt = ious.max(dim=1)
    for k, ni in enumerate(cand_idx.tolist()):
        if best_iou[k].item() >= iou_match:
            objectness[ni] = 1.0
            cls_targets[ni] = int(gt_labels[best_gt[k]].item())
            box_reg[ni] = gt_boxes[best_gt[k]] - cand_boxes[k]
            iou_targets[ni] = float(best_iou[k].item())

    # Mark best-source per GT (only among candidates assigned to that GT)
    for g_idx in range(gt_boxes.shape[0]):
        # Candidates that picked this GT
        picks = []
        for k, ni in enumerate(cand_idx.tolist()):
            if best_iou[k].item() >= iou_match and int(best_gt[k].item()) == g_idx:
                picks.append((float(best_iou[k].item()), ni))
        if picks:
            picks.sort(reverse=True)
            is_best_source[picks[0][1]] = 1.0

    return {"objectness": objectness, "class": cls_targets,
            "box_reg": box_reg, "iou": iou_targets,
            "is_best_source": is_best_source,
            "candidate_mask": candidate_mask}


def _build_targets(*args, **kwargs):
    """Backward-compatible shim for old signature."""
    return _build_targets_full(*args, **kwargs)
