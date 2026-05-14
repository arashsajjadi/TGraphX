"""Multi-seed runner for TGraphXLearnedBoxFusion.

Trains and evaluates the learned box-fusion model on the existing
graphs.pt (no detector re-run). Per seed it writes:

  metrics_seedN.json (Step-05 compatible: baseline_methods +
                       paired_bootstrap_vs_baselines + tgraphx headline)

and at the end:

  summary.json   per-method means + paired bootstrap aggregate.

The model trains on TRAIN, validates score-mode and AP75 on VAL, and is
evaluated once on TEST against all baselines from `baseline_ap_audit`.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from .baselines import nms, weighted_boxes_fusion
from .box_ops import box_iou
from .evaluation import (
    DetectionPrediction, GroundTruth, evaluate_predictions,
)
from .graph_builder import NODE_TYPES
from .learned_box_fusion import (
    LearnedBoxFusionConfig, TGraphXLearnedBoxFusion,
    FusionLossWeights, learned_fusion_loss,
)
from .multi_seed_v2 import _attach_slot_metadata
from .paired_bootstrap import paired_bootstrap, per_image_aps
from .reproducibility import set_global_seed
from .source_router_v3 import SOURCE_SLOTS, NUM_SOURCES


def _build_anchor_box(g, meta, anchor_slots):
    """For each cluster, build the WBF (or fallback) anchor box. Returns
    [C, 4] anchor_box tensor, [C] anchor_slot. Done in plain Python here
    so the model can stay GPU-pure during training."""
    md = g.metadata
    if "node_box" not in md:
        return None, None
    cluster_of = meta.cluster_of_node
    C = int(cluster_of.max().item()) + 1 if cluster_of.numel() > 0 else 0
    slot_assignments = md["slot_assignments"]
    ns = md["node_score"]
    nb = md["node_box"]
    per_cluster_best = [{} for _ in range(C)]
    for ni in range(slot_assignments.shape[0]):
        s = int(slot_assignments[ni].item())
        c = int(cluster_of[ni].item()) if ni < cluster_of.shape[0] else -1
        if c < 0 or s < 0:
            continue
        cur = per_cluster_best[c].get(s)
        if cur is None or float(ns[ni].item()) > float(ns[cur].item()):
            per_cluster_best[c][s] = ni
    anchor_box = torch.zeros(C, 4)
    anchor_slot = torch.full((C,), -1, dtype=torch.long)
    for c in range(C):
        avail = per_cluster_best[c]
        if not avail:
            continue
        for s in anchor_slots:
            if s in avail:
                anchor_box[c] = nb[avail[s]]
                anchor_slot[c] = s
                break
        if int(anchor_slot[c].item()) < 0:
            # First available
            s0 = sorted(avail.keys())[0]
            anchor_slot[c] = s0
            anchor_box[c] = nb[avail[s0]]
    return anchor_box, anchor_slot


def _match_clusters_to_gt(g, meta, anchor_box, iou_thresh: float = 0.5,
                            class_agnostic: bool = True):
    """For each cluster, find the matched GT (or none).

    Matching rule: the GT with maximum IoU against the cluster's anchor box,
    if that IoU is >= iou_thresh. (We could match against the best of any
    source box; the anchor-IoU rule is conservative and avoids label noise.)
    """
    md = g.metadata
    gt_b = md.get("gt_boxes"); gt_l = md.get("gt_labels")
    C = anchor_box.shape[0]
    matched = torch.zeros(C, 4)
    has_gt = torch.zeros(C, dtype=torch.bool)
    matched_lbl = torch.zeros(C, dtype=torch.long)
    if gt_b is None or gt_l is None or gt_b.numel() == 0:
        return matched, has_gt, matched_lbl
    ious = box_iou(anchor_box, gt_b)  # [C, G]
    best_iou, best_idx = ious.max(dim=1)
    for c in range(C):
        if float(best_iou[c].item()) >= iou_thresh:
            matched[c] = gt_b[int(best_idx[c].item())]
            has_gt[c] = True
            matched_lbl[c] = int(gt_l[int(best_idx[c].item())].item())
    return matched, has_gt, matched_lbl


def _clamp_to_image(boxes: torch.Tensor, image_size) -> torch.Tensor:
    """Clamp xyxy boxes to image extents — autograd-safe (no in-place ops).

    image_size = (H, W).
    """
    if boxes.numel() == 0:
        return boxes
    H, W = image_size
    x1 = boxes[:, 0].clamp(min=0, max=float(W - 1))
    y1 = boxes[:, 1].clamp(min=0, max=float(H - 1))
    x2 = boxes[:, 2].clamp(min=0, max=float(W - 1))
    y2 = boxes[:, 3].clamp(min=0, max=float(H - 1))
    x2 = torch.maximum(x2, x1 + 1.0)
    y2 = torch.maximum(y2, y1 + 1.0)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _baseline_predictions(graphs, src_labels, split, detector_names, iou_cluster):
    """Compute baseline predictions on a split (re-using saved graphs)."""
    out: Dict[str, List[DetectionPrediction]] = {
        f"det::{d}": [] for d in detector_names
    }
    out["fusion::nms"] = []
    out["fusion::wbf"] = []
    out["fusion::best_proposal"] = []
    gts: List[GroundTruth] = []
    for entry in graphs:
        g, meta, iid = entry[0], entry[1], entry[2]
        if src_labels.get(iid, {}).get("split") != split:
            continue
        md = g.metadata
        gt_b = md.get("gt_boxes", torch.zeros(0, 4))
        gt_l = md.get("gt_labels", torch.zeros(0, dtype=torch.long))
        gts.append(GroundTruth(image_id=iid, boxes_xyxy=gt_b, labels=gt_l))
        if "node_box" not in md:
            for k in out:
                out[k].append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0, 4),
                                                    scores=torch.zeros(0),
                                                    labels=torch.zeros(0, dtype=torch.long)))
            continue
        nt = md["node_types"]; nb = md["node_box"]; ns = md["node_score"]
        nl = md.get("node_label", torch.zeros(nb.shape[0], dtype=torch.long))
        mask = (nt == NODE_TYPES["proposal"])
        b, s, l = nb[mask], ns[mask], nl[mask]
        for di, dn in enumerate(detector_names):
            n2p = meta.node_to_proposal_index
            global_props = mask.nonzero(as_tuple=False).squeeze(-1)
            keep = []
            for gp in global_props.tolist():
                pi = int(n2p[gp].item()) if gp < n2p.shape[0] else -1
                if 0 <= pi < meta.proposal_detector_ids.shape[0] \
                        and int(meta.proposal_detector_ids[pi].item()) == di:
                    keep.append(gp)
            if not keep:
                out[f"det::{dn}"].append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0, 4),
                                                              scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
            else:
                k = torch.tensor(keep, dtype=torch.long)
                out[f"det::{dn}"].append(DetectionPrediction(image_id=iid, boxes_xyxy=nb[k], scores=ns[k],
                                                              labels=(nl[k] if nl is not None else torch.zeros(len(k), dtype=torch.long))))
        if b.numel() == 0:
            out["fusion::nms"].append(DetectionPrediction(image_id=iid, boxes_xyxy=b, scores=s, labels=l))
            out["fusion::wbf"].append(DetectionPrediction(image_id=iid, boxes_xyxy=b, scores=s, labels=l))
            out["fusion::best_proposal"].append(DetectionPrediction(image_id=iid, boxes_xyxy=b, scores=s, labels=l))
            continue
        keep = nms(b, s, iou_threshold=iou_cluster)
        out["fusion::nms"].append(DetectionPrediction(image_id=iid, boxes_xyxy=b[keep], scores=s[keep], labels=l[keep]))
        fb, fs, fl = weighted_boxes_fusion(b, s, l, iou_threshold=iou_cluster)
        out["fusion::wbf"].append(DetectionPrediction(image_id=iid, boxes_xyxy=fb, scores=fs, labels=fl))
        bp_keep = keep[:1] if keep.numel() else torch.zeros(0, dtype=torch.long)
        out["fusion::best_proposal"].append(DetectionPrediction(image_id=iid, boxes_xyxy=b[bp_keep], scores=s[bp_keep], labels=l[bp_keep]))
    return out, gts


def _ece_brier(preds, gts, iou_thresh, class_agnostic):
    from .evaluation import _match_predictions
    all_s, all_t = [], []
    gt_by_id = {g.image_id: g for g in gts}
    for p in preds:
        gt = gt_by_id.get(p.image_id)
        if gt is None or p.boxes_xyxy.numel() == 0:
            for s in p.scores.tolist():
                all_s.append(s); all_t.append(0)
            continue
        tp_flags, _ = _match_predictions(
            p.boxes_xyxy, p.scores, p.labels, gt.boxes_xyxy, gt.labels,
            iou_thresh, class_agnostic=class_agnostic,
        )
        for s, t in zip(p.scores.tolist(), tp_flags.tolist()):
            all_s.append(s); all_t.append(t)
    if not all_s:
        return 0.0, 0.0
    sc = torch.tensor(all_s); tp = torch.tensor(all_t, dtype=torch.float32)
    brier = float(((sc - tp) ** 2).mean())
    ece = 0.0; n = len(sc)
    for b in range(10):
        lo, hi = b/10, (b+1)/10
        mask = (sc >= lo) & (sc < hi)
        if mask.sum() == 0: continue
        conf = sc[mask].mean(); acc = tp[mask].mean()
        ece += (mask.sum().item()/n) * abs(float(conf) - float(acc))
    return ece, brier


def run_learned_fusion_seed(
    cfg: Dict[str, Any],
    seed: int,
    base_dir: Path,
    graphs,
    src_labels,
    detector_names: List[str],
    class_names: List[str],
    *,
    epochs: int = 30,
    lr: float = 5e-4,
    fusion_mode: str = "residual",
    device: str = "cpu",
) -> Dict[str, Any]:
    set_global_seed(seed, deterministic=False)
    num_classes = len(class_names)
    is_mc = num_classes > 2
    class_agnostic = bool(cfg.get("evaluation", {}).get("class_agnostic", not is_mc))
    iou_match = float(cfg.get("evaluation", {}).get("iou_match_threshold", 0.5))
    iou_cluster = float(cfg.get("graph", {}).get("cluster_iou_threshold", 0.5))
    crop_size = int(cfg.get("graph", {}).get("crop_size", 64))

    # ── Build split lists from manifest/src_labels ─────────────────
    train_idx, val_idx, test_idx = [], [], []
    for i, entry in enumerate(graphs):
        sp = src_labels.get(entry[2], {}).get("split")
        if sp == "train": train_idx.append(i)
        elif sp == "val": val_idx.append(i)
        elif sp == "test": test_idx.append(i)

    # Re-shuffle train order per seed for SGD.
    g_train = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_idx), generator=g_train).tolist()
    train_idx = [train_idx[i] for i in perm]

    # ── Model ──────────────────────────────────────────────────────
    g0, m0 = graphs[train_idx[0]][0], graphs[train_idx[0]][1]
    md = g0.metadata.get("node_metadata")
    metadata_dim = md.shape[1] if md is not None else None
    ea = g0.edge_features
    edge_feat_dim = ea.shape[1] if ea is not None and ea.numel() > 0 else 14

    model_cfg = LearnedBoxFusionConfig(
        num_classes=num_classes, num_detectors=len(detector_names),
        crop_size=crop_size,
        crop_channels=cfg.get("model", {}).get("crop_channels", 16),
        hidden_dim=cfg.get("model", {}).get("hidden_dim", 64),
        metadata_dim=metadata_dim, edge_feat_dim=edge_feat_dim,
        num_message_passing=cfg.get("model", {}).get("num_message_passing", 2),
        fusion_mode=fusion_mode,
        delta_cap_frac=float(cfg.get("model", {}).get("delta_cap_frac", 0.1)),
    )
    model = TGraphXLearnedBoxFusion(model_cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    weights = FusionLossWeights(
        box=cfg.get("training", {}).get("lambda_box", 1.0),
        giou=cfg.get("training", {}).get("lambda_giou", 1.0),
        tp50=cfg.get("training", {}).get("lambda_tp50", 1.0),
        tp75=cfg.get("training", {}).get("lambda_tp75", 2.0),
        iou=cfg.get("training", {}).get("lambda_iou", 0.5),
        delta_reg=cfg.get("training", {}).get("lambda_delta_reg", 0.05),
    )
    anchor_pref = [
        SOURCE_SLOTS["wbf"], SOURCE_SLOTS["nms_candidate"],
        SOURCE_SLOTS["best_proposal"], SOURCE_SLOTS["rt_detr"],
    ]

    # ── Train ──────────────────────────────────────────────────────
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0; n = 0
        for i in train_idx:
            g, meta, iid, *_ = graphs[i]
            md = g.metadata
            if "node_box" not in md:
                continue
            anchor_box, anchor_slot = _build_anchor_box(g, meta, anchor_pref)
            if anchor_box is None or anchor_box.shape[0] == 0:
                continue
            matched, has_gt, _ = _match_clusters_to_gt(g, meta, anchor_box,
                                                         iou_thresh=iou_match,
                                                         class_agnostic=class_agnostic)
            if not has_gt.any():
                continue
            gg = g.to(device)
            out = model(gg, detector_names=detector_names)
            if out["final_box_xyxy"].numel() == 0:
                continue
            # Clamp final_box to image extents.
            fb = _clamp_to_image(out["final_box_xyxy"], meta.image_size)
            out["final_box_xyxy"] = fb
            # IoU(final, gt) used as label for TP heads.
            iou_at_final = torch.zeros(fb.shape[0])
            if matched.numel() > 0:
                ious = box_iou(fb.detach().cpu(), matched.cpu())
                # IoU(final[c], matched[c]) is along the diagonal
                for c in range(fb.shape[0]):
                    iou_at_final[c] = float(ious[c, c].item()) if has_gt[c] else 0.0
            losses = learned_fusion_loss(
                out, gt_box=matched.to(device), has_gt=has_gt.to(device),
                iou_at_final=iou_at_final.to(device), weights=weights,
            )
            loss = losses["total"]
            if not loss.requires_grad:
                continue
            optim.zero_grad(); loss.backward(); optim.step()
            ep_loss += float(loss.item()); n += 1
        avg = ep_loss / max(1, n)
        history.append(avg)
        if ep % max(1, epochs // 5) == 0 or ep == epochs or ep == 1:
            print(f"  [seed {seed}] ep {ep}/{epochs}  loss={avg:.4f}")

    # ── Eval helper ─────────────────────────────────────────────────
    def _model_predictions(split):
        out_preds = []
        gts = []
        model.eval()
        with torch.no_grad():
            for i, entry in enumerate(graphs):
                if src_labels.get(entry[2], {}).get("split") != split:
                    continue
                g, meta, iid = entry[0], entry[1], entry[2]
                md = g.metadata
                gt_b = md.get("gt_boxes", torch.zeros(0, 4))
                gt_l = md.get("gt_labels", torch.zeros(0, dtype=torch.long))
                gts.append(GroundTruth(image_id=iid, boxes_xyxy=gt_b, labels=gt_l))
                if "node_box" not in md:
                    out_preds.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0, 4),
                                                          scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
                    continue
                gg = g.to(device)
                out = model(gg, detector_names=detector_names)
                C = out["final_box_xyxy"].shape[0]
                if C == 0:
                    out_preds.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0, 4),
                                                          scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
                    continue
                fb = _clamp_to_image(out["final_box_xyxy"].cpu(), meta.image_size)
                # Score = sigmoid(tp75_logit) — primary objective is AP75.
                score = torch.sigmoid(out["tp75_logit"].cpu())
                # Class: car (label 0 for single-class). For multi-class, pick
                # the anchor node's label.
                nl = md.get("node_label")
                slot_node_idx = out["slot_node_idx"].cpu()
                anchor_slot = out["anchor_slot"].cpu()
                labels = torch.zeros(C, dtype=torch.long)
                if nl is not None:
                    for c in range(C):
                        a = int(anchor_slot[c].item())
                        if a < 0:
                            continue
                        ni = int(slot_node_idx[c, a].item())
                        if 0 <= ni < nl.shape[0]:
                            labels[c] = int(nl[ni].item())
                out_preds.append(DetectionPrediction(image_id=iid, boxes_xyxy=fb,
                                                      scores=score, labels=labels))
        return out_preds, gts

    # Baselines (cached in this seed's eval)
    baseline_val, val_gts = _baseline_predictions(graphs, src_labels, "val",
                                                    detector_names, iou_cluster)
    baseline_test, test_gts = _baseline_predictions(graphs, src_labels, "test",
                                                      detector_names, iou_cluster)

    # Model predictions
    tgx_val, _ = _model_predictions("val")
    tgx_test, _ = _model_predictions("test")
    baseline_val["fusion::tgraphx_learned_fusion"] = tgx_val
    baseline_test["fusion::tgraphx_learned_fusion"] = tgx_test

    def _eval(preds, gts, iou_t, agn=class_agnostic):
        return evaluate_predictions(preds, gts, iou_threshold=iou_t,
                                      num_classes=num_classes, class_agnostic=agn)["AP"]

    def _miou(preds, gts):
        gt_by_id = {g.image_id: g for g in gts}
        ious = []
        for p in preds:
            gt = gt_by_id.get(p.image_id)
            if gt is None or p.boxes_xyxy.numel() == 0 or gt.boxes_xyxy.numel() == 0:
                continue
            m = box_iou(p.boxes_xyxy, gt.boxes_xyxy)
            if m.numel() > 0:
                ious.append(float(m.max(dim=1)[0].mean().item()))
        return float(sum(ious) / max(1, len(ious)))

    method_test = {}
    for name, preds in baseline_test.items():
        ap50 = _eval(preds, test_gts, iou_t=iou_match, agn=True)
        ap75 = _eval(preds, test_gts, iou_t=0.75, agn=True)
        miou = _miou(preds, test_gts)
        method_test[name] = {"AP50": ap50, "AP75": ap75, "mIoU": miou}

    method_val = {}
    for name, preds in baseline_val.items():
        ap50 = _eval(preds, val_gts, iou_t=iou_match, agn=True)
        ap75 = _eval(preds, val_gts, iou_t=0.75, agn=True)
        miou = _miou(preds, val_gts)
        method_val[name] = {"AP50": ap50, "AP75": ap75, "mIoU": miou}

    # Paired bootstrap of TGraphX vs every baseline (AP50 AND AP75)
    bootstraps: Dict[str, Dict[str, Any]] = {}
    for iou_t, tag in [(iou_match, "ap50"), (0.75, "ap75")]:
        _, tgx_aps = per_image_aps(tgx_test, test_gts, iou_threshold=iou_t,
                                      class_agnostic=class_agnostic)
        for name, preds in baseline_test.items():
            if name == "fusion::tgraphx_learned_fusion":
                continue
            _, b_aps = per_image_aps(preds, test_gts, iou_threshold=iou_t,
                                       class_agnostic=class_agnostic)
            if tgx_aps.shape != b_aps.shape:
                continue
            bootstraps.setdefault(name, {})[tag] = paired_bootstrap(tgx_aps, b_aps, seed=seed)

    # ECE/Brier on TGraphX test preds at IoU=0.5 and 0.75.
    ece50, brier50 = _ece_brier(tgx_test, test_gts, iou_thresh=iou_match, class_agnostic=class_agnostic)
    ece75, brier75 = _ece_brier(tgx_test, test_gts, iou_thresh=0.75, class_agnostic=class_agnostic)

    # Diagnostics: mean Δ magnitude, out-of-bounds rate
    delta_norms = []
    oob_count = 0; total_boxes = 0
    model.eval()
    with torch.no_grad():
        for i, entry in enumerate(graphs):
            if src_labels.get(entry[2], {}).get("split") != "test":
                continue
            g, meta, iid = entry[0], entry[1], entry[2]
            md = g.metadata
            if "node_box" not in md:
                continue
            gg = g.to(device)
            out = model(gg, detector_names=detector_names)
            if out["final_box_xyxy"].numel() == 0:
                continue
            delta = out["delta_box"].cpu()
            fb_raw = out["final_box_xyxy"].cpu()
            delta_norms.extend(delta.norm(dim=-1).tolist())
            H, W = meta.image_size
            oob = ((fb_raw[:, 0] < 0) | (fb_raw[:, 1] < 0)
                   | (fb_raw[:, 2] > W) | (fb_raw[:, 3] > H))
            oob_count += int(oob.sum().item())
            total_boxes += int(fb_raw.shape[0])

    metrics = {
        "seed": seed, "fusion_mode": fusion_mode, "device": device,
        "detector_names": detector_names, "num_classes": num_classes,
        "test_metrics_selected_mode": {
            "test_ap50": method_test["fusion::tgraphx_learned_fusion"]["AP50"],
            "test_ap75": method_test["fusion::tgraphx_learned_fusion"]["AP75"],
            "test_miou": method_test["fusion::tgraphx_learned_fusion"]["mIoU"],
            "headline_ap": method_test["fusion::tgraphx_learned_fusion"]["AP75"],
            "ece_at50": ece50, "brier_at50": brier50,
            "ece_at75": ece75, "brier_at75": brier75,
        },
        "val_methods": method_val,
        "test_methods": method_test,
        "baseline_methods": {  # Step-05 compatibility for Step 06 report
            n: {"headline_ap": v["AP75"], "test_ap_class_agnostic": v["AP50"],
                "test_ap_class_aware": v["AP50"]} for n, v in method_test.items()
        },
        "paired_bootstrap_vs_baselines": {n: b.get("ap50", {}) for n, b in bootstraps.items()},
        "paired_bootstrap_ap75_vs_baselines": {n: b.get("ap75", {}) for n, b in bootstraps.items()},
        "training_history": history,
        "delta_diagnostics": {
            "mean_delta_norm": float(sum(delta_norms) / max(1, len(delta_norms))),
            "max_delta_norm": float(max(delta_norms)) if delta_norms else 0.0,
            "oob_box_count": oob_count,
            "total_test_boxes": total_boxes,
            "oob_rate": oob_count / max(1, total_boxes),
        },
        "is_multiclass": is_mc,
        "uses_learned_box_fusion": True,
        "selected_score_mode": "p_tp75",
        "score_mode_selection_metric": "AP75",
    }
    (base_dir / f"metrics_seed{seed}.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"  [seed {seed}] tgx AP50={metrics['test_metrics_selected_mode']['test_ap50']:.4f}  "
          f"AP75={metrics['test_metrics_selected_mode']['test_ap75']:.4f}  "
          f"WBF AP75={method_test['fusion::wbf']['AP75']:.4f}")
    return metrics


def run_multi_seed_learned_fusion(
    config_path: str,
    seeds: Optional[Sequence[int]] = None,
    out_dir: Optional[str] = None,
    *,
    run_dir_with_graphs: Optional[str] = None,
    fusion_mode: str = "residual",
    epochs: int = 30,
    device: str = "cpu",
) -> Dict[str, Any]:
    from .config import load_config, resolve_device
    cfg = load_config(config_path)
    device = resolve_device(device or cfg.get("device", "auto"))
    cfg["device"] = device
    if seeds is None:
        seeds = cfg.get("seeds", [0, 1, 2, 3, 4])
    base_dir = Path(out_dir or "runs") / (cfg.get("run_name", "learned_fusion") + f"_{fusion_mode}")
    base_dir.mkdir(parents=True, exist_ok=True)

    # Re-use on-disk graphs if a source run dir is given (skip Step 03).
    rd = Path(run_dir_with_graphs) if run_dir_with_graphs else Path(f"runs/{cfg.get('run_name','exp')}")
    print(f"[learned_fusion] reading graphs from {rd}/graphs.pt …")
    graphs = torch.load(rd / "graphs.pt", weights_only=False)
    src_labels = torch.load(rd / "source_labels.pt", weights_only=False)
    manifest = json.loads((rd / "split_manifest.json").read_text())
    detector_names = manifest["detector_names"]
    class_names = manifest.get("class_names", ["car"])

    for entry in graphs:
        g, meta = entry[0], entry[1]
        if "slot_assignments" not in g.metadata:
            _attach_slot_metadata(g, meta, detector_names)

    all_seeds = []
    for seed in seeds:
        t0 = time.time()
        r = run_learned_fusion_seed(cfg, seed, base_dir, graphs, src_labels,
                                       detector_names, class_names,
                                       epochs=epochs, fusion_mode=fusion_mode, device=device)
        r["elapsed_s"] = time.time() - t0
        all_seeds.append(r)

    # Aggregate summary
    summary: Dict[str, Any] = {"seeds": list(seeds), "fusion_mode": fusion_mode,
                                "detector_names": detector_names,
                                "class_names": class_names, "n_seeds": len(all_seeds)}
    means: Dict[str, Dict[str, float]] = {}
    boot_means: Dict[str, Dict[str, Any]] = {}
    if all_seeds:
        method_names = sorted(all_seeds[0]["test_methods"].keys())
        for name in method_names:
            ap50s = [s["test_methods"][name]["AP50"] for s in all_seeds]
            ap75s = [s["test_methods"][name]["AP75"] for s in all_seeds]
            mious = [s["test_methods"][name]["mIoU"] for s in all_seeds]
            means[name] = {
                "AP50_mean": statistics.mean(ap50s),
                "AP50_std": statistics.stdev(ap50s) if len(ap50s) > 1 else 0.0,
                "AP75_mean": statistics.mean(ap75s),
                "AP75_std": statistics.stdev(ap75s) if len(ap75s) > 1 else 0.0,
                "mIoU_mean": statistics.mean(mious),
            }
        # Bootstrap means (AP50 + AP75 separately)
        for bk in ("paired_bootstrap_vs_baselines", "paired_bootstrap_ap75_vs_baselines"):
            d: Dict[str, Any] = {}
            for s in all_seeds:
                for b_name, b in s.get(bk, {}).items():
                    if not b: continue
                    d.setdefault(b_name, []).append(b["p_a_gt_b"])
            boot_means[bk] = {n: (statistics.mean(v), min(v), max(v), len(v))
                                for n, v in d.items()}
    summary["method_means"] = means
    summary["paired_bootstrap_means"] = boot_means
    (base_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[learned_fusion] summary → {base_dir/'summary.json'}")
    return summary
