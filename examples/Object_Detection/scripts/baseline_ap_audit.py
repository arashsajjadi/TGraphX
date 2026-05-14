"""Compute actual AP50 of every baseline + per-cluster Oracle on the
already-saved graphs.pt. No detector run, no training.

Answers:
  - What is the TRUE on-disk AP50 of NMS / WBF / rt_detr / retinanet / yolo_modern
    on this 200-image car-only test split?
  - What is the AP50 of a per-cluster Oracle that picks the best available source?
  - What is the AP50 of "always anchor=rt_detr" vs "anchor + oracle override"?

This grounds the user-supplied AP numbers in measurement, not assertion.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from od_graph_fusion.baselines import nms, weighted_boxes_fusion
from od_graph_fusion.box_ops import box_iou
from od_graph_fusion.evaluation import evaluate_predictions, DetectionPrediction, GroundTruth
from od_graph_fusion.graph_builder import NODE_TYPES
from od_graph_fusion.multi_seed_v2 import _attach_slot_metadata, _build_util_and_labels
from od_graph_fusion.source_router_v3 import NUM_SOURCES, SOURCE_SLOTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--iou-cluster", type=float, default=0.5)
    ap.add_argument("--iou-match", type=float, default=0.5)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    graphs = torch.load(run_dir / "graphs.pt", weights_only=False)
    src_labels = torch.load(run_dir / "source_labels.pt", weights_only=False)
    manifest = json.loads((run_dir / "split_manifest.json").read_text())
    detector_names = manifest["detector_names"]
    num_classes = manifest.get("num_classes", 1)
    is_mc = num_classes > 2

    for entry in graphs:
        g, meta = entry[0], entry[1]
        if "slot_assignments" not in g.metadata:
            _attach_slot_metadata(g, meta, detector_names)

    def _per_det_preds(split, di):
        out = []
        for entry in graphs:
            g, meta, iid = entry[0], entry[1], entry[2]
            sp = src_labels.get(iid, {}).get("split")
            if sp != split:
                continue
            md = g.metadata
            if "node_box" not in md or "node_score" not in md:
                out.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0,4),
                                                 scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
                continue
            nt = md["node_types"]; nb = md["node_box"]; ns = md["node_score"]
            nl = md.get("node_label")
            n2p = meta.node_to_proposal_index
            mask = (nt == NODE_TYPES["proposal"])
            keep = []
            for gp in mask.nonzero(as_tuple=False).squeeze(-1).tolist():
                pi = int(n2p[gp].item()) if gp < n2p.shape[0] else -1
                if 0 <= pi < meta.proposal_detector_ids.shape[0]:
                    if int(meta.proposal_detector_ids[pi].item()) == di:
                        keep.append(gp)
            if not keep:
                out.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0,4),
                                                 scores=torch.zeros(0),
                                                 labels=torch.zeros(0, dtype=torch.long)))
                continue
            k = torch.tensor(keep, dtype=torch.long)
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=nb[k], scores=ns[k],
                                             labels=nl[k] if nl is not None else torch.zeros(len(k), dtype=torch.long)))
        return out

    def _pool_pred(split, fusion):
        out = []
        for entry in graphs:
            g, meta, iid = entry[0], entry[1], entry[2]
            if src_labels.get(iid, {}).get("split") != split: continue
            md = g.metadata
            if "node_box" not in md or "node_score" not in md:
                out.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0,4),
                                                 scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
                continue
            nt = md["node_types"]; nb = md["node_box"]; ns = md["node_score"]; nl = md.get("node_label")
            mask = (nt == NODE_TYPES["proposal"])
            b, s, l = nb[mask], ns[mask], (nl[mask] if nl is not None else torch.zeros(mask.sum(), dtype=torch.long))
            if b.numel() == 0:
                out.append(DetectionPrediction(image_id=iid, boxes_xyxy=b, scores=s, labels=l)); continue
            if fusion == "nms":
                k = nms(b, s, iou_threshold=args.iou_cluster)
                out.append(DetectionPrediction(image_id=iid, boxes_xyxy=b[k], scores=s[k], labels=l[k]))
            elif fusion == "wbf":
                fb, fs, fl = weighted_boxes_fusion(b, s, l, iou_threshold=args.iou_cluster)
                out.append(DetectionPrediction(image_id=iid, boxes_xyxy=fb, scores=fs, labels=fl))
            elif fusion == "best_proposal":
                # one box per cluster = top-1 NMS
                k = nms(b, s, iou_threshold=args.iou_cluster)[:1] if b.numel() else torch.zeros(0, dtype=torch.long)
                out.append(DetectionPrediction(image_id=iid, boxes_xyxy=b[k], scores=s[k], labels=l[k]))
        return out

    def _gts(split):
        out = []
        for entry in graphs:
            g, meta, iid = entry[0], entry[1], entry[2]
            if src_labels.get(iid, {}).get("split") != split: continue
            md = g.metadata
            out.append(GroundTruth(image_id=iid, boxes_xyxy=md.get("gt_boxes", torch.zeros(0,4)),
                                     labels=md.get("gt_labels", torch.zeros(0, dtype=torch.long))))
        return out

    # Build per-cluster Oracle predictions where each cluster contributes the
    # best-available source (max utility) — same selection that bounds the
    # source-router family.
    def _oracle_preds(split, anchor_slot):
        out = []
        for entry in graphs:
            g, meta, iid = entry[0], entry[1], entry[2]
            if src_labels.get(iid, {}).get("split") != split: continue
            md = g.metadata
            if "node_box" not in md:
                out.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0,4),
                                                 scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
                continue
            gt_b = md.get("gt_boxes"); gt_l = md.get("gt_labels")
            if gt_b is None or gt_l is None or gt_b.numel() == 0:
                out.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0,4),
                                                 scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
                continue
            labels = _build_util_and_labels(g, meta, gt_b, gt_l, class_agnostic=not is_mc,
                                              utility_mode="ap50")
            if labels is None:
                out.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0,4),
                                                 scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
                continue
            _u, _bs, _bl, util_per_slot, slot_avail = labels
            nb = md["node_box"]; ns = md["node_score"]; nl = md["node_label"]
            slot_assignments = md["slot_assignments"]
            cluster_of = meta.cluster_of_node
            C = util_per_slot.shape[0]; S = util_per_slot.shape[1]
            slot_node_idx = torch.full((C, S), -1, dtype=torch.long)
            for ni in range(slot_assignments.shape[0]):
                s = int(slot_assignments[ni].item())
                c = int(cluster_of[ni].item()) if ni < cluster_of.shape[0] else -1
                if c < 0 or s < 0: continue
                cur = int(slot_node_idx[c, s].item())
                if cur < 0 or float(ns[ni].item()) > float(ns[cur].item()):
                    slot_node_idx[c, s] = ni
            boxes, scores, labels_ = [], [], []
            for c in range(C):
                u = util_per_slot[c].clone()
                u[~slot_avail[c]] = float("-inf")
                if not torch.isfinite(u).any(): continue
                # oracle: pick argmax(util) cluster-wise. If anchor_slot is
                # set, simulate the anchor-preserving policy: take best slot
                # only if delta > 0; else take anchor.
                best_slot = int(u.argmax().item())
                if anchor_slot is not None and bool(slot_avail[c, anchor_slot].item()):
                    if float(util_per_slot[c, best_slot].item()) - float(util_per_slot[c, anchor_slot].item()) <= 0:
                        best_slot = anchor_slot
                n = int(slot_node_idx[c, best_slot].item())
                if n < 0: continue
                boxes.append(nb[n]); scores.append(ns[n].clone())
                labels_.append(nl[n] if nl is not None else torch.tensor(0, dtype=torch.long))
            out.append(DetectionPrediction(image_id=iid,
                boxes_xyxy=torch.stack(boxes) if boxes else torch.zeros(0,4),
                scores=torch.stack(scores).float() if scores else torch.zeros(0),
                labels=torch.stack(labels_) if labels_ else torch.zeros(0, dtype=torch.long)))
        return out

    out_summary = {}
    for split in ("val", "test"):
        gts = _gts(split)
        methods = {}
        for di, dn in enumerate(detector_names):
            preds = _per_det_preds(split, di)
            methods[f"det::{dn}"] = preds
        for fusion in ("nms", "wbf", "best_proposal"):
            methods[f"fusion::{fusion}"] = _pool_pred(split, fusion)
        # Oracles
        methods["oracle::per_cluster_best_available"] = _oracle_preds(split, anchor_slot=None)
        methods["oracle::rtdetr_anchor_with_oracle_override"] = _oracle_preds(
            split, anchor_slot=SOURCE_SLOTS["rt_detr"])
        methods["oracle::nms_anchor_with_oracle_override"] = _oracle_preds(
            split, anchor_slot=SOURCE_SLOTS["nms_candidate"])

        rows = {}
        for name, preds in methods.items():
            ap_agn = evaluate_predictions(preds, gts, iou_threshold=args.iou_match,
                                            num_classes=num_classes, class_agnostic=True)["AP"]
            rows[name] = ap_agn
        out_summary[split] = rows
        print(f"\n=== {split} ===")
        for name, ap in sorted(rows.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<55s}  AP50={ap:.4f}")

    (run_dir / "baseline_ap_audit.json").write_text(json.dumps(out_summary, indent=2))
    print(f"\n[audit] → {run_dir/'baseline_ap_audit.json'}")


if __name__ == "__main__":
    main()
