"""Step 05: Evaluate checkpoint only.

Reads:  {run_dir}/checkpoint_seed{seed}.pt   (model_config must be present)
        {run_dir}/graphs.pt
        {run_dir}/source_labels.pt            (GT labels — not in graph features)
        {run_dir}/split_manifest.json         (split IDs + detector_names)
Writes: {run_dir}/metrics_seed{seed}.json

Invariants enforced:
- detector_names loaded from split_manifest or graph.metadata, NEVER empty.
- model reconstructed from checkpoint["model_config"], NOT from config/hard-coded values.
- score_mode selected on VALIDATION only, frozen before TEST.
- ECE, Brier score, FP/image computed.
- class-aware AP and class-agnostic AP both reported.
- Test split is evaluated ONCE with frozen score mode.

Does NOT train. Does NOT call run_pipeline.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description="Step 05: evaluate TGraphX")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import torch
    from od_graph_fusion.config import load_config, resolve_device
    from od_graph_fusion.source_router_v3 import TGraphXSourceRouterV3, fuse_v3, SOURCE_SLOTS
    from od_graph_fusion.multi_seed_v2 import _attach_slot_metadata, _build_util_and_labels
    from od_graph_fusion.evaluation import evaluate_predictions, DetectionPrediction, GroundTruth

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir or f"runs/{cfg.get('run_name', 'exp')}")
    ckpt_path = run_dir / f"checkpoint_seed{args.seed}.pt"
    graphs_path = run_dir / "graphs.pt"
    out_metrics = run_dir / f"metrics_seed{args.seed}.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"[05] Missing checkpoint — run step 04 first: {ckpt_path}")
    if not graphs_path.exists():
        raise FileNotFoundError(f"[05] Missing graphs.pt — run step 03 first: {graphs_path}")
    if out_metrics.exists() and not args.force:
        print(f"[05] Metrics exist: {out_metrics}  (--force to rerun)")
        return

    device = resolve_device(args.device or cfg.get("device", "auto"))
    all_graphs = torch.load(graphs_path, weights_only=False)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Require model_config in checkpoint — no hard-coded fallbacks
    if "model_config" not in ckpt:
        raise RuntimeError(
            f"[05] checkpoint missing 'model_config' — re-run step 04 with --force. "
            f"Keys found: {list(ckpt.keys())}"
        )
    mc = ckpt["model_config"]
    detector_names = mc["detector_names"]
    num_classes = mc["num_classes"]
    num_detectors = mc["num_detectors"]
    if num_detectors == 0 or not detector_names:
        raise RuntimeError(
            f"[05] model_config has num_detectors={num_detectors}, detector_names={detector_names}. "
            "Passing empty detector_names to V3.forward makes proposal nodes invisible. "
            "Re-run step 03 and 04 with --force."
        )
    print(f"[05] Model: {num_classes} classes, {num_detectors} detectors: {detector_names}")

    # Load source_labels (GT boxes/labels stored separately from graph features)
    labels_path = run_dir / "source_labels.pt"
    source_labels = torch.load(labels_path, weights_only=False) if labels_path.exists() else {}

    # Use split manifest for deterministic splits
    manifest_path = run_dir / "split_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        val_ids  = set(manifest["split_ids"]["val"])
        test_ids = set(manifest["split_ids"]["test"])
        val_data  = [(g, meta, iid) for g, meta, iid, *_ in all_graphs if iid in val_ids]
        test_data = [(g, meta, iid) for g, meta, iid, *_ in all_graphs if iid in test_ids]
    else:
        # Fallback
        n = len(all_graphs); n_train = int(n*0.75); n_val = int(n*0.10)
        val_data  = [(g, meta, iid) for g, meta, iid, *_ in all_graphs[n_train:n_train+n_val]]
        test_data = [(g, meta, iid) for g, meta, iid, *_ in all_graphs[n_train+n_val:]]

    print(f"[05] Val: {len(val_data)} | Test: {len(test_data)}")
    iou_match = float(cfg.get("evaluation", {}).get("iou_match_threshold", 0.5))
    is_multiclass = (num_classes > 2)

    # Reconstruct model from checkpoint — not from config
    model = TGraphXSourceRouterV3(
        num_classes=num_classes,
        num_detectors=num_detectors,
        crop_size=mc["crop_size"],
        crop_channels=mc["crop_channels"],
        hidden_dim=mc["hidden_dim"],
        num_message_passing=mc["num_message_passing"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    def make_gts(data):
        gts = []
        for entry in data:
            g, meta, iid = entry[0], entry[1], entry[2]
            sl = source_labels.get(iid, {})
            gt_b = sl.get("gt_boxes",
                   g.metadata.get("gt_boxes", torch.zeros(0,4)) if isinstance(g.metadata, dict)
                   else torch.zeros(0,4))
            gt_l = sl.get("gt_labels",
                   g.metadata.get("gt_labels", torch.zeros(0,dtype=torch.long)) if isinstance(g.metadata, dict)
                   else torch.zeros(0,dtype=torch.long))
            gts.append(GroundTruth(image_id=iid, boxes_xyxy=gt_b, labels=gt_l))
        return gts

    def _ece_brier(preds, gts, iou_thresh=0.5):
        """Compute ECE and Brier score from predictions vs GT."""
        from od_graph_fusion.evaluation import _match_predictions
        all_scores, all_tp = [], []
        gt_by_id = {g.image_id: g for g in gts}
        for pred in preds:
            gt = gt_by_id.get(pred.image_id)
            if gt is None or pred.boxes_xyxy.numel() == 0:
                for s in pred.scores.tolist():
                    all_scores.append(s); all_tp.append(0)
                continue
            tp_flags, _ = _match_predictions(
                pred.boxes_xyxy, pred.scores, pred.labels,
                gt.boxes_xyxy, gt.labels, iou_thresh, class_agnostic=True)
            for s, t in zip(pred.scores.tolist(), tp_flags.tolist()):
                all_scores.append(s); all_tp.append(t)
        if not all_scores:
            return 0.0, 0.0
        import torch as _t
        sc = _t.tensor(all_scores); tp = _t.tensor(all_tp, dtype=_t.float32)
        # Brier
        brier = float(((sc - tp)**2).mean())
        # ECE with 10 bins
        ece = 0.0; n = len(sc)
        for b in range(10):
            lo, hi = b/10, (b+1)/10
            mask = (sc >= lo) & (sc < hi)
            if mask.sum() == 0: continue
            conf = sc[mask].mean(); acc = tp[mask].mean()
            ece += (mask.sum().item()/n) * abs(float(conf) - float(acc))
        return ece, brier

    def run_score_mode(data, score_fn):
        preds = []
        with torch.no_grad():
            for g, meta, iid in data:
                # CRITICAL: pass real detector_names, not []
                out = model(g.to(device), detector_names=detector_names)
                src_logits = out.get("source_logits"); src_mask = out.get("source_mask")
                nb = g.metadata.get("node_box"); ns = g.metadata.get("node_score")
                nl = g.metadata.get("node_label"); sni = g.metadata.get("_slot_node_idx")
                if src_logits is None:
                    preds.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0,4),
                                                      scores=torch.zeros(0), labels=torch.zeros(0,dtype=torch.long)))
                    continue
                # Check proposals are visible
                prop_count = (src_mask[:, :4].any(dim=1)).sum() if src_mask.shape[1] >= 4 else 0
                boxes, scores, labels = [], [], []
                for c in range(src_logits.shape[0]):
                    sm2 = src_mask[c]
                    if not sm2.any(): continue
                    avail = sm2.nonzero().squeeze(-1)
                    best_local = int(src_logits[c][avail].argmax())
                    chosen_slot = int(avail[best_local])
                    raw = float(src_logits[c, chosen_slot])
                    rp = float(torch.sigmoid(torch.tensor(raw)))
                    ni = -1
                    if sni is not None and c < sni.shape[0] and chosen_slot < sni.shape[1]:
                        ni = int(sni[c, chosen_slot])
                    if ni < 0: continue
                    base_s = float(ns[ni]) if ns is not None else 0.5
                    boxes.append(nb[ni].cpu()); scores.append(torch.tensor(score_fn(base_s, rp)))
                    labels.append(nl[ni].cpu() if nl is not None else torch.tensor(0, dtype=torch.long))
                preds.append(DetectionPrediction(image_id=iid,
                    boxes_xyxy=torch.stack(boxes) if boxes else torch.zeros(0,4),
                    scores=torch.stack(scores) if scores else torch.zeros(0),
                    labels=torch.stack(labels) if labels else torch.zeros(0,dtype=torch.long)))
        return preds

    score_modes = {
        "routing_prob": lambda b, p: float(p),
        "routing_prob*max_base": lambda b, p: float(p * max(b, 0.1)),
        "base_score": lambda b, p: float(b),
    }

    val_gts = make_gts(val_data)
    test_gts = make_gts(test_data)

    # Select score mode on VALIDATION only.
    # Multi-class: use class-AWARE AP for selection (the operational metric).
    # Single-class car-only: class-agnostic = class-aware, use class-agnostic for back-compat.
    selection_class_agnostic = not is_multiclass
    best_mode = None; best_val_ap = -1.0; val_score_modes = {}
    for mode_name, score_fn in score_modes.items():
        vp = run_score_mode(val_data, score_fn)
        v_ap_agn = evaluate_predictions(vp, val_gts, iou_threshold=iou_match,
                                          num_classes=num_classes, class_agnostic=True)["AP"]
        v_ap_aware = evaluate_predictions(vp, val_gts, iou_threshold=iou_match,
                                            num_classes=num_classes, class_agnostic=False)["AP"]
        v_ap_for_selection = v_ap_agn if selection_class_agnostic else v_ap_aware
        val_score_modes[mode_name] = {
            "val_ap_agnostic": v_ap_agn,
            "val_ap_aware": v_ap_aware,
            "val_ap_used_for_selection": v_ap_for_selection,
            "selection_metric": "class_agnostic_AP" if selection_class_agnostic else "class_aware_AP",
        }
        print(f"  [val] {mode_name}: AP_agn={v_ap_agn:.4f} AP_aware={v_ap_aware:.4f} "
              f"(selecting on {val_score_modes[mode_name]['selection_metric']})")
        if v_ap_for_selection > best_val_ap:
            best_val_ap = v_ap_for_selection; best_mode = mode_name
    print(f"  → Score mode selected on VAL: {best_mode} (val AP={best_val_ap:.4f})")

    # Evaluate on TEST once — frozen score mode
    test_preds = run_score_mode(test_data, score_modes[best_mode])
    test_ap_agnostic = evaluate_predictions(test_preds, test_gts, iou_threshold=iou_match,
                                             num_classes=num_classes, class_agnostic=True)["AP"]
    test_ap_aware = evaluate_predictions(test_preds, test_gts, iou_threshold=iou_match,
                                          num_classes=num_classes, class_agnostic=False)["AP"]
    ece, brier = _ece_brier(test_preds, test_gts, iou_match)
    fp_per_image = sum(max(0, len(p.scores) - max(1, len(g.boxes_xyxy)))
                       for p, g in zip(test_preds, test_gts)) / max(1, len(test_preds))

    headline_ap = test_ap_aware if is_multiclass else test_ap_agnostic

    # ── Baselines on the SAME test split (Part 9 requirement) ──────────
    # Reconstruct per-image detector boxes by reading them off the test
    # graphs themselves. Each proposal node has its source detector_id
    # in g.metadata['proposal_det_ids'] and its box in 'node_box'.
    from od_graph_fusion.baselines import nms, weighted_boxes_fusion
    from od_graph_fusion.paired_bootstrap import per_image_aps, paired_bootstrap

    def _per_detector_preds(test_data, det_name, det_idx):
        out = []
        for g, meta, iid in test_data:
            md = g.metadata if isinstance(g.metadata, dict) else {}
            pdet_ids = md.get("proposal_det_ids")
            nb = md.get("node_box"); ns = md.get("node_score"); nl = md.get("node_label")
            from od_graph_fusion.graph_builder import NODE_TYPES
            nt = md.get("node_types")
            if pdet_ids is None or nb is None or ns is None or nt is None:
                out.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0,4),
                                                scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
                continue
            mask = (nt == NODE_TYPES["proposal"]) & (pdet_ids == det_idx)
            if not mask.any():
                out.append(DetectionPrediction(image_id=iid, boxes_xyxy=torch.zeros(0,4),
                                                scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
                continue
            b = nb[mask]; s = ns[mask]; lbl = nl[mask] if nl is not None else torch.zeros(mask.sum(), dtype=torch.long)
            out.append(DetectionPrediction(image_id=iid, boxes_xyxy=b, scores=s, labels=lbl))
        return out

    def _pool_all(test_data):
        from od_graph_fusion.graph_builder import NODE_TYPES
        out = []
        for g, meta, iid in test_data:
            md = g.metadata if isinstance(g.metadata, dict) else {}
            nb = md.get("node_box"); ns = md.get("node_score"); nl = md.get("node_label")
            nt = md.get("node_types")
            if nb is None or ns is None or nt is None:
                out.append((iid, torch.zeros(0,4), torch.zeros(0), torch.zeros(0, dtype=torch.long)))
                continue
            mask = (nt == NODE_TYPES["proposal"])
            b = nb[mask]; s = ns[mask]; lbl = nl[mask] if nl is not None else torch.zeros(mask.sum(), dtype=torch.long)
            out.append((iid, b, s, lbl))
        return out

    iou_cluster = float(cfg.get("graph", {}).get("cluster_iou_threshold", 0.5))
    baseline_methods: Dict[str, Any] = {}

    # Raw detectors
    for di, dn in enumerate(detector_names):
        baseline_methods[f"det::{dn}"] = _per_detector_preds(test_data, dn, di)

    # NMS / WBF / BestProposal — pool from per-detector boxes
    pooled = _pool_all(test_data)
    nms_preds, wbf_preds, bp_preds = [], [], []
    for iid, b, s, lbl in pooled:
        if b.numel() == 0:
            nms_preds.append(DetectionPrediction(image_id=iid, boxes_xyxy=b, scores=s, labels=lbl))
            wbf_preds.append(DetectionPrediction(image_id=iid, boxes_xyxy=b, scores=s, labels=lbl))
            bp_preds.append(DetectionPrediction(image_id=iid, boxes_xyxy=b, scores=s, labels=lbl))
            continue
        keep = nms(b, s, iou_threshold=iou_cluster)
        nms_preds.append(DetectionPrediction(image_id=iid, boxes_xyxy=b[keep], scores=s[keep], labels=lbl[keep]))
        fb, fs, fl = weighted_boxes_fusion(b, s, lbl, iou_threshold=iou_cluster)
        wbf_preds.append(DetectionPrediction(image_id=iid, boxes_xyxy=fb, scores=fs, labels=fl))
        # best_proposal: highest-score proposal per cluster (approximate by NMS top-1)
        bp_preds.append(DetectionPrediction(image_id=iid, boxes_xyxy=b[keep[:1]] if keep.numel() else b,
                                              scores=s[keep[:1]] if keep.numel() else s,
                                              labels=lbl[keep[:1]] if keep.numel() else lbl))
    baseline_methods["fusion::nms"] = nms_preds
    baseline_methods["fusion::wbf"] = wbf_preds
    baseline_methods["fusion::best_proposal"] = bp_preds
    baseline_methods["fusion::tgraphx"] = test_preds

    # Score each method (class-agnostic + class-aware)
    method_test_ap: Dict[str, Dict[str, float]] = {}
    for name, preds in baseline_methods.items():
        ap_agn = evaluate_predictions(preds, test_gts, iou_threshold=iou_match,
                                        num_classes=num_classes, class_agnostic=True)["AP"]
        ap_aware = evaluate_predictions(preds, test_gts, iou_threshold=iou_match,
                                          num_classes=num_classes, class_agnostic=False)["AP"]
        method_test_ap[name] = {
            "test_ap_class_agnostic": ap_agn,
            "test_ap_class_aware": ap_aware,
            "headline_ap": ap_aware if is_multiclass else ap_agn,
        }

    # Paired bootstrap of TGraphX vs each baseline.
    pair_class_agnostic = not is_multiclass
    _, tgx_aps = per_image_aps(test_preds, test_gts, iou_threshold=iou_match,
                                 class_agnostic=pair_class_agnostic)
    bootstraps: Dict[str, Dict[str, float]] = {}
    for name, preds in baseline_methods.items():
        if name == "fusion::tgraphx":
            continue
        _, b_aps = per_image_aps(preds, test_gts, iou_threshold=iou_match,
                                   class_agnostic=pair_class_agnostic)
        if tgx_aps.shape != b_aps.shape:
            continue
        bootstraps[name] = paired_bootstrap(tgx_aps, b_aps, seed=args.seed)

    metrics = {
        "seed": args.seed, "device": device,
        "detector_names": detector_names,
        "num_classes": num_classes,
        "num_detectors": num_detectors,
        "val_score_modes": val_score_modes,
        "selected_score_mode": best_mode,
        "score_mode_selection_metric": "class_aware_AP" if is_multiclass else "class_agnostic_AP",
        "test_metrics_selected_mode": {
            "test_ap_class_agnostic": test_ap_agnostic,
            "test_ap_class_aware": test_ap_aware,
            "headline_ap": headline_ap,
            "ece": ece,
            "brier": brier,
            "fp_per_image": fp_per_image,
        },
        "baseline_methods": method_test_ap,
        "paired_bootstrap_vs_baselines": bootstraps,
        "num_val": len(val_data), "num_test": len(test_data),
        "is_multiclass": is_multiclass,
    }
    out_metrics.write_text(json.dumps(metrics, indent=2))
    print(f"[05] Test headline AP ({best_mode}) = {headline_ap:.4f} "
          f"(class_agnostic={test_ap_agnostic:.4f}, class_aware={test_ap_aware:.4f}) "
          f"ECE={ece:.4f} Brier={brier:.4f} FP/img={fp_per_image:.2f}")
    print(f"     → {out_metrics}")


if __name__ == "__main__":
    main()
