"""Paper-faithful candidate-node-selection oracle audit.

Reads runs/<run>/graphs.pt and asks:

  Q1. Does selecting ALL graph WBF nodes reproduce external WBF AP?
      (graph-node baseline equivalence — required by Part 4)
  Q2. Does the per-cluster best-node oracle have AP50 / AP75 / mIoU
      headroom over the strongest classical baseline?
      (required by Part 5)

If Q1 fails grossly, graph construction is broken.
If Q2 shows no headroom at any IoU threshold, no node-selection model
can win and we stop.

Outputs:
  runs/<run>/candidate_node_oracle_audit.json
  reports/CANDIDATE_NODE_ORACLE_AUDIT.md
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from od_graph_fusion.baselines import nms, weighted_boxes_fusion, soft_nms
from od_graph_fusion.box_ops import box_iou
from od_graph_fusion.evaluation import (
    DetectionPrediction, GroundTruth, evaluate_predictions,
)
from od_graph_fusion.graph_builder import NODE_TYPES
from od_graph_fusion.multi_seed_v2 import _attach_slot_metadata, _build_util_and_labels


def _filter_by_split(graphs, src_labels, split):
    return [(e[0], e[1], e[2]) for e in graphs
             if src_labels.get(e[2], {}).get("split") == split]


def _gts(data):
    return [GroundTruth(image_id=iid,
                          boxes_xyxy=g.metadata.get("gt_boxes", torch.zeros(0, 4)),
                          labels=g.metadata.get("gt_labels", torch.zeros(0, dtype=torch.long)))
             for g, meta, iid in data]


def _select_nodes_by_type(data, type_name):
    """Output one prediction per graph node of the given type (raw type name).

    `type_name`: a NODE_TYPES key (e.g. "cluster" for WBF, "nms_candidate" for NMS).
    Score = node_score; label = node_label.
    """
    type_id = NODE_TYPES[type_name]
    out = []
    for g, meta, iid in data:
        md = g.metadata
        if "node_box" not in md:
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0, 4),
                                             scores=torch.zeros(0),
                                             labels=torch.zeros(0, dtype=torch.long)))
            continue
        nt = md["node_types"]; nb = md["node_box"]; ns = md["node_score"]
        nl = md.get("node_label", torch.zeros(nb.shape[0], dtype=torch.long))
        mask = (nt == type_id)
        out.append(DetectionPrediction(image_id=iid, boxes_xyxy=nb[mask],
                                         scores=ns[mask], labels=nl[mask]))
    return out


def _select_per_detector(data, det_idx):
    out = []
    for g, meta, iid in data:
        md = g.metadata
        if "node_box" not in md:
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0, 4),
                                             scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
            continue
        nt = md["node_types"]; nb = md["node_box"]; ns = md["node_score"]
        nl = md.get("node_label", torch.zeros(nb.shape[0], dtype=torch.long))
        n2p = meta.node_to_proposal_index
        mask = (nt == NODE_TYPES["proposal"])
        keep = []
        for gp in mask.nonzero(as_tuple=False).squeeze(-1).tolist():
            pi = int(n2p[gp].item()) if gp < n2p.shape[0] else -1
            if 0 <= pi < meta.proposal_detector_ids.shape[0] \
                    and int(meta.proposal_detector_ids[pi].item()) == det_idx:
                keep.append(gp)
        if not keep:
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0, 4),
                                             scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
        else:
            k = torch.tensor(keep, dtype=torch.long)
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=nb[k], scores=ns[k], labels=nl[k]))
    return out


def _external_classical(data, fusion: str, iou_thr: float):
    """Recompute external WBF / NMS / Soft-NMS / BestProposal on the proposal pool."""
    out = []
    for g, meta, iid in data:
        md = g.metadata
        if "node_box" not in md:
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0, 4),
                                             scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
            continue
        nt = md["node_types"]; nb = md["node_box"]; ns = md["node_score"]
        nl = md.get("node_label", torch.zeros(nb.shape[0], dtype=torch.long))
        mask = (nt == NODE_TYPES["proposal"])
        b, s, l = nb[mask], ns[mask], nl[mask]
        if b.numel() == 0:
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=b, scores=s, labels=l))
            continue
        if fusion == "nms":
            k = nms(b, s, iou_threshold=iou_thr)
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=b[k], scores=s[k], labels=l[k]))
        elif fusion == "soft_nms":
            k, decayed = soft_nms(b, s)
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=b[k], scores=decayed, labels=l[k]))
        elif fusion == "wbf":
            fb, fs, fl = weighted_boxes_fusion(b, s, l, iou_threshold=iou_thr)
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=fb, scores=fs, labels=fl))
        elif fusion == "best_proposal":
            k = nms(b, s, iou_threshold=iou_thr)[:1]
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=b[k], scores=s[k], labels=l[k]))
    return out


def _per_cluster_oracle(data, util_mode="ap50", class_agnostic=True,
                          score_mode="picked_node"):
    """For each cluster, pick the best node by AP-style utility.

    score_mode:
      "picked_node"     — score = picked node's own score (default)
      "cluster_max"     — score = max source score in the cluster
                          (simulates a TP-aware score head that picks
                          the right score for the chosen box)
      "perfect_tp"      — score = 1.0 if TP@0.5 else 0.0
                          (the AP-optimal score given known TP labels —
                          the true upper bound)
    """
    out = []
    for g, meta, iid in data:
        md = g.metadata
        if "node_box" not in md:
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0, 4),
                                             scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
            continue
        gt_b = md.get("gt_boxes"); gt_l = md.get("gt_labels")
        if gt_b is None or gt_l is None or gt_b.numel() == 0:
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0, 4),
                                             scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
            continue
        labels = _build_util_and_labels(g, meta, gt_b, gt_l, class_agnostic=class_agnostic,
                                          utility_mode=util_mode)
        if labels is None:
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0, 4),
                                             scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
            continue
        node_util, best_slot, _, util_per_slot, slot_avail = labels
        nb = md["node_box"]; ns = md["node_score"]; nl = md.get("node_label")
        slot_assignments = md["slot_assignments"]
        cluster_of = meta.cluster_of_node
        from od_graph_fusion.candidate_mask import candidate_node_mask
        cand_mask = candidate_node_mask(meta.node_types, NODE_TYPES)
        n_clusters = int(cluster_of.max().item()) + 1 if cluster_of.numel() > 0 else 0
        boxes, scores, lbls = [], [], []
        # IoU lookup against GT for "perfect_tp" scoring
        for c in range(n_clusters):
            in_c = (cluster_of == c) & cand_mask
            if not in_c.any():
                continue
            idx_c = in_c.nonzero(as_tuple=False).squeeze(-1)
            # pick highest utility in cluster
            best_local = int(node_util[idx_c].argmax().item())
            best_ni = int(idx_c[best_local].item())
            picked_box = nb[best_ni]
            if score_mode == "picked_node":
                picked_score = ns[best_ni].clone()
            elif score_mode == "cluster_max":
                picked_score = ns[idx_c].max().clone()
            elif score_mode == "perfect_tp":
                # TP if IoU(picked_box, any gt) >= 0.5 (and class matches if not class_agnostic)
                iou_max = float(box_iou(picked_box.unsqueeze(0), gt_b)[0].max().item())
                picked_score = torch.tensor(1.0 if iou_max >= 0.5 else 0.0)
            else:
                picked_score = ns[best_ni].clone()
            boxes.append(picked_box)
            scores.append(picked_score)
            lbls.append(nl[best_ni] if nl is not None else torch.tensor(0, dtype=torch.long))
        out.append(DetectionPrediction(
            image_id=iid,
            boxes_xyxy=torch.stack(boxes) if boxes else torch.zeros(0, 4),
            scores=torch.stack(scores).float() if scores else torch.zeros(0),
            labels=torch.stack(lbls) if lbls else torch.zeros(0, dtype=torch.long),
        ))
    return out


def _mean_iou(preds, gts):
    gts_by_id = {g.image_id: g for g in gts}
    ious = []
    for p in preds:
        gt = gts_by_id.get(p.image_id)
        if gt is None or p.boxes_xyxy.numel() == 0 or gt.boxes_xyxy.numel() == 0:
            continue
        m = box_iou(p.boxes_xyxy, gt.boxes_xyxy)
        if m.numel() > 0:
            ious.append(float(m.max(dim=1)[0].mean().item()))
    return float(sum(ious) / max(1, len(ious)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="YAML config (preferred). Derives run_dir automatically.")
    ap.add_argument("--run-dir", default=None,
                    help="Run directory (overrides config-derived path).")
    ap.add_argument("--iou-match", type=float, default=None)
    ap.add_argument("--iou-cluster", type=float, default=None)
    args = ap.parse_args()

    cfg = {}
    if args.config:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from od_graph_fusion.config import load_config
        cfg = load_config(args.config)

    if args.run_dir:
        rd = Path(args.run_dir)
    elif cfg:
        rd = Path(f"runs/{cfg.get('run_name', 'exp')}")
    else:
        ap.error("Provide --config or --run-dir.")
    rd = Path(rd)

    iou_match   = args.iou_match   or float(cfg.get("evaluation", {}).get("iou_match_threshold", 0.5))
    iou_cluster = args.iou_cluster or float(cfg.get("graph", {}).get("cluster_iou_threshold", 0.5))

    # Prefer object_graphs.pt (new format); fall back to graphs.pt (image-level)
    obj_graphs_path = rd / "object_graphs.pt"
    img_graphs_path = rd / "graphs.pt"

    if obj_graphs_path.exists():
        # New object-level format: each entry is per-cluster; aggregate to image-level for audit
        print(f"[oracle-audit] Using object_graphs.pt (object-level format)")
        all_obj_graphs = torch.load(obj_graphs_path, weights_only=False)
        obj_labels = torch.load(rd / "object_labels.pt", weights_only=False) \
            if (rd / "object_labels.pt").exists() else {}
        manifest = json.loads((rd / "object_manifest.json").read_text()) \
            if (rd / "object_manifest.json").exists() else {}
        detector_names = manifest.get("detector_names", [])
        num_classes = manifest.get("num_classes", 1)
        # Build image-level graph list by aggregating object-level entries
        # (attach slot metadata from image-level graphs.pt if available for full audit)
        if img_graphs_path.exists():
            graphs = torch.load(img_graphs_path, weights_only=False)
            src_labels_path = rd / "source_labels.pt"
            src_labels = torch.load(src_labels_path, weights_only=False) if src_labels_path.exists() else {}
            if not detector_names:
                manifest2 = json.loads((rd / "split_manifest.json").read_text()) \
                    if (rd / "split_manifest.json").exists() else {}
                detector_names = manifest2.get("detector_names", [])
                num_classes = manifest2.get("num_classes", 1)
        else:
            graphs = None; src_labels = {}
    else:
        print(f"[oracle-audit] Using graphs.pt (image-level format)")
        graphs = torch.load(img_graphs_path, weights_only=False)
        src_labels_path = rd / "source_labels.pt"
        src_labels = torch.load(src_labels_path, weights_only=False) if src_labels_path.exists() else {}
        manifest_path = rd / "split_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        detector_names = manifest.get("detector_names", [])
        num_classes = manifest.get("num_classes", 1)

    if graphs is None:
        print(f"[oracle-audit] No image-level graphs found — skipping image-level audit.")
        return

    is_mc = num_classes > 2

    for e in graphs:
        if "slot_assignments" not in e[0].metadata:
            _attach_slot_metadata(e[0], e[1], detector_names)

    out: dict = {"detector_names": detector_names}
    for split in ("val", "test"):
        data = _filter_by_split(graphs, src_labels, split)
        gts = _gts(data)
        rows = {}

        def _eval(preds, iou_t, agn=not is_mc):
            return evaluate_predictions(preds, gts, iou_threshold=iou_t,
                                          num_classes=num_classes, class_agnostic=agn)["AP"]

        def _record(name, preds):
            rows[name] = {
                "AP50": _eval(preds, iou_match),
                "AP75": _eval(preds, 0.75),
                "mIoU": _mean_iou(preds, gts),
            }

        # External classical baselines
        for fusion in ("nms", "soft_nms", "wbf", "best_proposal"):
            _record(f"external::{fusion}", _external_classical(data, fusion, iou_cluster))
        # Per-detector raw baselines
        for di, dn in enumerate(detector_names):
            _record(f"raw::{dn}", _select_per_detector(data, di))
        # Graph-node baselines
        graph_to_external = {
            "graph::nms_candidate": ("nms_candidate", "external::nms"),
            "graph::soft_nms_candidate": ("soft_nms_candidate", "external::soft_nms"),
            "graph::cluster_wbf": ("cluster", "external::wbf"),
            "graph::best_proposal_candidate": ("best_proposal_candidate", "external::best_proposal"),
            "graph::consensus_union": ("consensus", None),
        }
        for label, (tname, _) in graph_to_external.items():
            _record(label, _select_nodes_by_type(data, tname))
        # Oracles — three scoring modes
        _record("oracle::best_node_picked_score", _per_cluster_oracle(
            data, util_mode="ap50", class_agnostic=not is_mc, score_mode="picked_node"))
        _record("oracle::best_node_cluster_max_score", _per_cluster_oracle(
            data, util_mode="ap50", class_agnostic=not is_mc, score_mode="cluster_max"))
        _record("oracle::best_node_perfect_tp_score", _per_cluster_oracle(
            data, util_mode="ap50", class_agnostic=not is_mc, score_mode="perfect_tp"))

        # Equivalence deltas
        equiv = {}
        for label, (tname, ext_key) in graph_to_external.items():
            if ext_key is None or ext_key not in rows or label not in rows:
                continue
            for m in ("AP50", "AP75"):
                equiv[f"{label}_vs_{ext_key}_{m}"] = rows[label][m] - rows[ext_key][m]
        out[split] = {"rows": rows, "equivalence_deltas": equiv}

        print(f"\n=== {split} ===")
        for n, v in sorted(rows.items(), key=lambda kv: -kv[1]["AP50"]):
            print(f"  {n:38s}  AP50={v['AP50']:.4f}  AP75={v['AP75']:.4f}  mIoU={v['mIoU']:.4f}")
        print("  --- equivalence (graph_node − external) ---")
        for k, v in sorted(equiv.items()):
            print(f"    {k:60s}  Δ={v:+.4f}")

    (rd / "candidate_node_oracle_audit.json").write_text(json.dumps(out, indent=2))
    print(f"\n→ {rd/'candidate_node_oracle_audit.json'}")


if __name__ == "__main__":
    main()
