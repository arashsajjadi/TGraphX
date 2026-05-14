"""Oracle audit for the LEARNED BOX FUSION formulation.

Question: assuming a perfect cluster-level box decoder, what is the
*highest possible* AP50 we can reach on this dataset, given the existing
cluster set?

We compute several oracles, in order of increasing power:

  1. GT_oracle      — emit the matched GT box per cluster (absolute
                       ceiling). Tells us the *recall ceiling* of the
                       current cluster construction.
  2. Convex_oracle  — best convex combination over the cluster's source
                       boxes (gradient search over Δ^S maximizing IoU
                       with matched GT). Bounded to within the convex
                       hull of source corners.
  3. WBFΔ_oracle    — WBF box + free residual Δ, chosen to maximize
                       IoU(WBF+Δ, matched GT) per cluster. Bounded
                       version: residual capped at 0.1·diag(WBF_box).
  4. AnyBox_oracle  — any axis-aligned box. Equivalent to GT_oracle
                       when a GT is matched; reported for completeness.

A cluster's "matched GT" is the GT with maximum IoU against any source
box in the cluster (≥ iou_match_threshold).

Scoring: every oracle emits one box per cluster scored with the cluster's
maximum source-detector confidence. Score *calibration* is not what
we're testing here; box quality is.

Reads:  runs/<run>/graphs.pt, source_labels.pt, split_manifest.json
Writes: runs/<run>/box_fusion_oracle_audit.json
        reports/BOX_FUSION_ORACLE_AUDIT.md (also written by this script)

Run:
  python scripts/box_fusion_oracle_audit.py --run-dir runs/real_voc_car_v2
"""
import argparse, json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from od_graph_fusion.baselines import nms, weighted_boxes_fusion
from od_graph_fusion.box_ops import box_iou
from od_graph_fusion.evaluation import (
    DetectionPrediction, GroundTruth, evaluate_predictions,
)
from od_graph_fusion.graph_builder import NODE_TYPES
from od_graph_fusion.multi_seed_v2 import _attach_slot_metadata
from od_graph_fusion.source_router_v3 import SOURCE_SLOTS


def _build_cluster_source_table(g, meta, detector_names):
    """Return per-cluster lists of (slot, node_idx) for available sources."""
    md = g.metadata
    if "node_box" not in md:
        return None
    slot_assignments = md["slot_assignments"]
    cluster_of = meta.cluster_of_node
    ns = md["node_score"]
    C = int(cluster_of.max().item()) + 1 if cluster_of.numel() > 0 else 0
    # For each (cluster, slot) keep the highest-score node.
    per_cluster: dict = {c: {} for c in range(C)}
    for ni in range(slot_assignments.shape[0]):
        s = int(slot_assignments[ni].item())
        c = int(cluster_of[ni].item()) if ni < cluster_of.shape[0] else -1
        if c < 0 or s < 0:
            continue
        cur = per_cluster[c].get(s)
        if cur is None or float(ns[ni].item()) > float(ns[cur].item()):
            per_cluster[c][s] = ni
    return per_cluster


def _convex_oracle_box(source_boxes: torch.Tensor, gt_box: torch.Tensor,
                        n_steps: int = 80, lr: float = 0.5) -> torch.Tensor:
    """Find convex combination weights w∈Δ^S maximizing IoU(Σ w_i b_i, gt).

    `source_boxes`: [S, 4] in xyxy. Returns the optimized fused box [4].
    """
    S = source_boxes.shape[0]
    if S == 0:
        return torch.zeros(4)
    if S == 1:
        return source_boxes[0]
    # Parameterize w via softmax over unconstrained z.
    z = torch.zeros(S, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    sb = source_boxes.detach()
    gt = gt_box.detach()
    for _ in range(n_steps):
        opt.zero_grad()
        w = torch.softmax(z, dim=0).unsqueeze(-1)
        fused = (w * sb).sum(dim=0, keepdim=True)
        iou = box_iou(fused, gt.unsqueeze(0))[0, 0]
        loss = 1.0 - iou
        if loss.requires_grad:
            loss.backward()
            opt.step()
    with torch.no_grad():
        w = torch.softmax(z, dim=0).unsqueeze(-1)
        fused = (w * sb).sum(dim=0)
    return fused


def _wbf_residual_oracle(wbf_box: torch.Tensor, gt_box: torch.Tensor,
                          *, cap_frac: float = 0.1,
                          n_steps: int = 80, lr: float = 0.5) -> torch.Tensor:
    """WBF + Δ with ‖Δ‖∞ ≤ cap_frac * diag(wbf_box), Δ maximizes IoU vs GT.

    Returns refined box.
    """
    w = float(max(1e-6, wbf_box[2] - wbf_box[0]))
    h = float(max(1e-6, wbf_box[3] - wbf_box[1]))
    cap = cap_frac * math.sqrt(w * w + h * h)
    delta = torch.zeros(4, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        d_clamped = torch.tanh(delta) * cap  # ‖d‖_∞ ≤ cap
        fused = (wbf_box.detach() + d_clamped).unsqueeze(0)
        iou = box_iou(fused, gt_box.unsqueeze(0))[0, 0]
        loss = 1.0 - iou
        if loss.requires_grad:
            loss.backward()
            opt.step()
    with torch.no_grad():
        d_clamped = torch.tanh(delta) * cap
        return wbf_box + d_clamped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--iou-cluster", type=float, default=0.5)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    print(f"[oracle] loading graphs from {run_dir}/graphs.pt")
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

    def _gts(split):
        out = []
        for entry in graphs:
            g, meta, iid = entry[0], entry[1], entry[2]
            if src_labels.get(iid, {}).get("split") != split:
                continue
            md = g.metadata
            out.append(GroundTruth(
                image_id=iid,
                boxes_xyxy=md.get("gt_boxes", torch.zeros(0, 4)),
                labels=md.get("gt_labels", torch.zeros(0, dtype=torch.long)),
            ))
        return out

    def _wbf_predictions(split):
        out = []
        for entry in graphs:
            g, meta, iid = entry[0], entry[1], entry[2]
            if src_labels.get(iid, {}).get("split") != split:
                continue
            md = g.metadata
            if "node_box" not in md:
                out.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0, 4),
                                                 scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
                continue
            nt = md["node_types"]
            mask = (nt == NODE_TYPES["proposal"])
            b = md["node_box"][mask]; s = md["node_score"][mask]
            l = md.get("node_label", torch.zeros(b.shape[0], dtype=torch.long))[mask]
            if b.numel() == 0:
                out.append(DetectionPrediction(image_id=iid, boxes_xyxy=b, scores=s, labels=l))
                continue
            fb, fs, fl = weighted_boxes_fusion(b, s, l, iou_threshold=args.iou_cluster)
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=fb, scores=fs, labels=fl))
        return out

    def _oracle_preds(split, mode: str):
        out = []
        for entry in graphs:
            g, meta, iid = entry[0], entry[1], entry[2]
            if src_labels.get(iid, {}).get("split") != split:
                continue
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
            tbl = _build_cluster_source_table(g, meta, detector_names)
            if tbl is None:
                out.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0, 4),
                                                 scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
                continue
            nb = md["node_box"]; ns = md["node_score"]; nl = md.get("node_label")
            boxes, scores, labels = [], [], []
            # For each cluster pick one (box, score, label)
            for c, sd in tbl.items():
                if not sd:
                    continue
                # Score: max source confidence in cluster
                node_idxs = list(sd.values())
                max_score = float(max(ns[ni].item() for ni in node_idxs))
                # Cluster source boxes
                slots = list(sd.keys())
                src_b = torch.stack([nb[sd[s]] for s in slots])  # [S, 4]
                src_lbls = torch.tensor([int(nl[sd[s]].item()) if nl is not None else 0
                                          for s in slots])
                # Match to nearest GT (by max IoU across source boxes)
                ious = box_iou(src_b, gt_b)  # [S, G]
                # Per source, best GT it could match
                best_gt_iou, best_gt_idx = ious.max(dim=1)  # [S]
                # Cluster matches the GT whose max over sources is largest
                cluster_iou, src_pick = best_gt_iou.max(dim=0)
                gt_idx = int(best_gt_idx[src_pick].item())
                # Default: cluster has no qualifying GT → emit WBF box (no oracle help)
                if float(cluster_iou.item()) < args.iou_match:
                    if "wbf" in [SOURCE_SLOTS_by_name(s) for s in slots] or SOURCE_SLOTS["wbf"] in slots:
                        wbf_node = sd.get(SOURCE_SLOTS["wbf"])
                        fused_box = nb[wbf_node]
                    else:
                        fused_box = src_b[int(ns[torch.tensor(node_idxs)].argmax().item())]
                    boxes.append(fused_box); scores.append(torch.tensor(max_score))
                    labels.append(torch.tensor(int(src_lbls[0].item())))
                    continue
                matched_gt = gt_b[gt_idx]
                matched_lbl = int(gt_l[gt_idx].item())

                if mode == "gt_oracle" or mode == "any_box_oracle":
                    fused_box = matched_gt
                elif mode == "convex_oracle":
                    fused_box = _convex_oracle_box(src_b, matched_gt)
                elif mode == "wbf_residual_oracle":
                    if SOURCE_SLOTS["wbf"] in slots:
                        anchor = nb[sd[SOURCE_SLOTS["wbf"]]]
                    elif SOURCE_SLOTS["nms_candidate"] in slots:
                        anchor = nb[sd[SOURCE_SLOTS["nms_candidate"]]]
                    else:
                        anchor = src_b[int(ns[torch.tensor(node_idxs)].argmax().item())]
                    fused_box = _wbf_residual_oracle(anchor, matched_gt, cap_frac=0.1)
                elif mode == "wbf_residual_oracle_unconstrained":
                    # No cap — effectively any box. Useful as sanity vs gt_oracle.
                    if SOURCE_SLOTS["wbf"] in slots:
                        anchor = nb[sd[SOURCE_SLOTS["wbf"]]]
                    elif SOURCE_SLOTS["nms_candidate"] in slots:
                        anchor = nb[sd[SOURCE_SLOTS["nms_candidate"]]]
                    else:
                        anchor = src_b[int(ns[torch.tensor(node_idxs)].argmax().item())]
                    fused_box = _wbf_residual_oracle(anchor, matched_gt, cap_frac=10.0)
                else:
                    raise ValueError(mode)

                boxes.append(fused_box)
                scores.append(torch.tensor(max_score))
                # Predict the matched GT's class so class-aware AP is fair.
                labels.append(torch.tensor(matched_lbl))
            out.append(DetectionPrediction(
                image_id=iid,
                boxes_xyxy=torch.stack(boxes) if boxes else torch.zeros(0, 4),
                scores=torch.stack(scores).float() if scores else torch.zeros(0),
                labels=torch.stack(labels) if labels else torch.zeros(0, dtype=torch.long),
            ))
        return out

    def _eval(preds, gts, class_agnostic=True):
        r = evaluate_predictions(preds, gts, iou_threshold=args.iou_match,
                                  num_classes=num_classes, class_agnostic=class_agnostic)
        return r["AP"]

    def _eval75(preds, gts, class_agnostic=True):
        r = evaluate_predictions(preds, gts, iou_threshold=0.75,
                                  num_classes=num_classes, class_agnostic=class_agnostic)
        return r["AP"]

    def _mean_iou(preds, gts):
        gts_by_id = {g.image_id: g for g in gts}
        ious = []
        for p in preds:
            gt = gts_by_id.get(p.image_id)
            if gt is None or p.boxes_xyxy.numel() == 0 or gt.boxes_xyxy.numel() == 0:
                continue
            ious_m = box_iou(p.boxes_xyxy, gt.boxes_xyxy)
            if ious_m.numel() > 0:
                ious.append(float(ious_m.max(dim=1)[0].mean().item()))
        return float(sum(ious) / max(1, len(ious)))

    summary = {}
    for split in ("val", "test"):
        print(f"\n=== {split} ===")
        gts = _gts(split)
        wbf = _wbf_predictions(split)
        wbf_ap50 = _eval(wbf, gts)
        wbf_ap75 = _eval75(wbf, gts)
        wbf_miou = _mean_iou(wbf, gts)
        print(f"  WBF baseline:                  AP50={wbf_ap50:.4f} AP75={wbf_ap75:.4f} mIoU={wbf_miou:.4f}")
        modes = {
            "gt_oracle":                   "Output matched-GT box per cluster",
            "convex_oracle":               "Best convex combo of source boxes",
            "wbf_residual_oracle":         "WBF + capped Δ (‖Δ‖∞ ≤ 0.1·diag)",
            "wbf_residual_oracle_unconstrained": "WBF + unbounded Δ",
        }
        rows = {"fusion::wbf": {"AP50": wbf_ap50, "AP75": wbf_ap75, "mIoU": wbf_miou, "gap_vs_wbf": 0.0}}
        for m, desc in modes.items():
            preds = _oracle_preds(split, m)
            ap50 = _eval(preds, gts); ap75 = _eval75(preds, gts); miou = _mean_iou(preds, gts)
            rows[f"oracle::{m}"] = {
                "AP50": ap50, "AP75": ap75, "mIoU": miou,
                "gap_vs_wbf": ap50 - wbf_ap50, "description": desc,
            }
            print(f"  {m:<35s}  AP50={ap50:.4f} AP75={ap75:.4f} mIoU={miou:.4f}  gap_vs_WBF={ap50 - wbf_ap50:+.4f}")
        summary[split] = rows

    out_path = run_dir / "box_fusion_oracle_audit.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[oracle] → {out_path}")

    # ── Write markdown report ───────────────────────────────────────
    md_path = Path("reports") / "BOX_FUSION_ORACLE_AUDIT.md"
    md = ["# BOX_FUSION_ORACLE_AUDIT — empirical on real VOC car",
          "", f"**Generated:** by `scripts/box_fusion_oracle_audit.py` on `{run_dir.name}`.",
          "", "## Oracle table"]
    for split in ("val", "test"):
        md.append(f"\n### {split}\n")
        md.append("| Policy | AP50 | AP75 | mIoU | Δ AP50 vs WBF |")
        md.append("|--------|-----:|-----:|-----:|--------------:|")
        for name, r in sorted(summary[split].items(), key=lambda kv: -kv[1]["AP50"]):
            md.append(f"| `{name}` | {r['AP50']:.4f} | {r['AP75']:.4f} | {r['mIoU']:.4f} | {r['gap_vs_wbf']:+.4f} |")
    # ── Verdict line ────────────────────────────────────────────────
    test = summary["test"]
    wbf_test = test["fusion::wbf"]["AP50"]
    headroom = max([test[k]["AP50"] - wbf_test for k in test
                     if k.startswith("oracle::") and k != "oracle::gt_oracle"
                     and k != "oracle::wbf_residual_oracle_unconstrained"])
    md.append("\n## Verdict\n")
    if headroom >= 0.005:
        md.append(f"**`BOX_FUSION_ORACLE_HAS_HEADROOM`** — max non-trivial oracle "
                   f"is +{headroom:.4f} AP above WBF on test. Learned box fusion has a "
                   "target above the strongest classical baseline. Proceed with Parts 3–9.")
    else:
        md.append(f"**`BOX_FUSION_ORACLE_NO_HEADROOM`** — max non-trivial oracle gain "
                   f"is {headroom:+.4f} AP on test. The graph + cluster construction does "
                   "not contain enough information to fuse a box better than WBF, even "
                   "with perfect cluster-level oracle decisions. STOP — do not train.")
    md_path.write_text("\n".join(md) + "\n")
    print(f"[oracle] → {md_path}")

    # Print a one-liner that the next step can read.
    print(f"\nverdict_test_headroom={headroom:+.4f}")


# Small helper for slot-by-name lookup (used once above)
def SOURCE_SLOTS_by_name(slot_int: int) -> str:
    inv = {v: k for k, v in SOURCE_SLOTS.items()}
    return inv.get(slot_int, str(slot_int))


if __name__ == "__main__":
    main()
