"""Evaluate TGraphXCandidateNodeSelector on object-level candidate graphs.

Reads:  {run_dir}/object_graphs.pt
        {run_dir}/object_labels.pt
        {run_dir}/object_manifest.json
        {run_dir}/candidate_checkpoint_seed{N}.pt

Writes: {run_dir}/candidate_eval_seed{N}.json
        {run_dir}/candidate_eval_summary.json

Invariants:
- Score head selected on VALIDATION only, frozen before TEST.
- selected_box = exactly node_box[argmax(selection_logit)].
- No GT leakage: gt_boxes not accessed during inference.
- Both AP50 and AP75 reported.
- Paired bootstrap vs every baseline.
"""
import argparse, json, sys, statistics, time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    ap = argparse.ArgumentParser(description="Evaluate TGraphXCandidateNodeSelector")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch
    from od_graph_fusion.config import load_config, resolve_device
    from od_graph_fusion.candidate_node_selector import (
        CandidateSelectorConfig, TGraphXCandidateNodeSelector, select_per_cluster,
    )
    from od_graph_fusion.candidate_mask import candidate_node_mask
    from od_graph_fusion.graph_builder import NODE_TYPES
    from od_graph_fusion.evaluation import DetectionPrediction, GroundTruth, evaluate_predictions
    from od_graph_fusion.baselines import nms, weighted_boxes_fusion, soft_nms
    from od_graph_fusion.box_ops import box_iou
    from od_graph_fusion.paired_bootstrap import paired_bootstrap, per_image_aps

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir or f"runs/{cfg.get('run_name', 'exp')}")

    obj_graphs_path = run_dir / "object_graphs.pt"
    obj_labels_path = run_dir / "object_labels.pt"
    manifest_path   = run_dir / "object_manifest.json"

    if not obj_graphs_path.exists():
        raise FileNotFoundError(f"Missing object_graphs.pt: {obj_graphs_path}")

    obj_graphs  = torch.load(obj_graphs_path, weights_only=False)
    obj_labels  = torch.load(obj_labels_path, weights_only=False) if obj_labels_path.exists() else {}
    manifest    = json.loads(manifest_path.read_text())
    detector_names = manifest["detector_names"]
    class_names    = manifest.get("class_names", ["car"])
    num_classes    = manifest.get("num_classes", len(class_names))
    is_mc          = num_classes > 2
    class_agnostic = bool(cfg.get("evaluation", {}).get("class_agnostic", not is_mc))
    iou_match      = float(cfg.get("evaluation", {}).get("iou_match_threshold", 0.5))
    iou_cluster    = float(cfg.get("graph", {}).get("cluster_iou_threshold", 0.5))
    device         = resolve_device(args.device or cfg.get("device", "auto"))

    # Discover seeds from available checkpoints if not specified
    if args.seeds is None:
        found = sorted(run_dir.glob("candidate_checkpoint_seed*.pt"))
        seeds = [int(p.stem.split("seed")[1]) for p in found]
        if not seeds:
            raise FileNotFoundError(
                f"No candidate_checkpoint_seed*.pt found in {run_dir}. "
                "Run train_candidate_node_selector.py first.")
    else:
        seeds = args.seeds

    # Split obj_graphs
    val_data  = [(g, img_id, cid, cand_src) for g, img_id, cid, sp, cand_src, *_ in obj_graphs
                 if obj_labels.get(f"{img_id}_{cid}", {}).get("split", sp) == "val"]
    test_data = [(g, img_id, cid, cand_src) for g, img_id, cid, sp, cand_src, *_ in obj_graphs
                 if obj_labels.get(f"{img_id}_{cid}", {}).get("split", sp) == "test"]
    print(f"[eval-cns] val={len(val_data)} test={len(test_data)} graphs | device={device}")

    def _make_gts(data):
        gts_by_img = {}
        for g, img_id, cid, _ in data:
            if img_id in gts_by_img:
                continue
            key = f"{img_id}_{cid}"
            lbl = obj_labels.get(key, {})
            gt_b = lbl.get("gt_image_boxes", torch.zeros(0, 4))
            gt_l = lbl.get("gt_image_labels", torch.zeros(0, dtype=torch.long))
            gts_by_img[img_id] = GroundTruth(image_id=img_id, boxes_xyxy=gt_b, labels=gt_l)
        return list(gts_by_img.values())

    val_gts  = _make_gts(val_data)
    test_gts = _make_gts(test_data)

    def _eval(preds, gts, iou_t):
        return evaluate_predictions(preds, gts, iou_threshold=iou_t,
                                      num_classes=num_classes,
                                      class_agnostic=class_agnostic)["AP"]

    def _miou(preds, gts):
        gts_by_id = {g.image_id: g for g in gts}
        ious = []
        for p in preds:
            gt = gts_by_id.get(p.image_id)
            if gt is None or p.boxes_xyxy.numel() == 0 or gt.boxes_xyxy.numel() == 0:
                continue
            m = box_iou(p.boxes_xyxy, gt.boxes_xyxy)
            if m.numel():
                ious.append(float(m.max(dim=1)[0].mean()))
        return float(sum(ious) / max(1, len(ious)))

    def _model_predict(model, data, score_head):
        model.eval()
        preds_by_img = defaultdict(lambda: {"b": [], "s": [], "l": []})
        with torch.no_grad():
            for g, img_id, cid, cand_src in data:
                nb = g.metadata.get("node_box")
                nl = g.metadata.get("node_label")
                nt = g.metadata.get("node_types")
                if nb is None:
                    continue
                gg = g.to(device)
                out = model(gg, detector_names=detector_names)
                cand_m = candidate_node_mask(nt, NODE_TYPES) if nt is not None \
                         else torch.ones(nb.shape[0], dtype=torch.bool)
                cluster_of = torch.zeros(nb.shape[0], dtype=torch.long)
                picked = select_per_cluster(
                    out, cluster_of=cluster_of, cand_mask=cand_m,
                    node_box=nb,
                    node_label=nl if nl is not None else torch.zeros(nb.shape[0], dtype=torch.long),
                    score_head=score_head,
                )
                if picked["boxes_xyxy"].numel() > 0:
                    preds_by_img[img_id]["b"].append(picked["boxes_xyxy"])
                    preds_by_img[img_id]["s"].append(picked["scores"])
                    preds_by_img[img_id]["l"].append(picked["labels"])
        return [DetectionPrediction(
            image_id=img_id,
            boxes_xyxy=torch.cat(d["b"]) if d["b"] else torch.zeros(0, 4),
            scores=torch.cat(d["s"]) if d["s"] else torch.zeros(0),
            labels=torch.cat(d["l"]) if d["l"] else torch.zeros(0, dtype=torch.long),
        ) for img_id, d in preds_by_img.items()]

    # Build external / graph-node baselines (same for all seeds)
    def _external_fusion(data, fusion):
        pool = defaultdict(lambda: {"b": [], "s": [], "l": []})
        for g, img_id, cid, _ in data:
            nb = g.metadata.get("node_box"); ns = g.metadata.get("node_score")
            nl = g.metadata.get("node_label"); nt_t = g.metadata.get("node_types")
            if nb is None or nt_t is None:
                continue
            mask = nt_t == NODE_TYPES["proposal"]
            if not mask.any():
                continue
            pool[img_id]["b"].append(nb[mask]); pool[img_id]["s"].append(ns[mask])
            pool[img_id]["l"].append(nl[mask] if nl is not None else torch.zeros(mask.sum(), dtype=torch.long))
        result = []
        for img_id, d in pool.items():
            b = torch.cat(d["b"]); s = torch.cat(d["s"]); l = torch.cat(d["l"])
            if fusion == "nms":
                k = nms(b, s, iou_threshold=iou_cluster)
                result.append(DetectionPrediction(image_id=img_id, boxes_xyxy=b[k], scores=s[k], labels=l[k]))
            elif fusion == "soft_nms":
                k, dec = soft_nms(b, s)
                result.append(DetectionPrediction(image_id=img_id, boxes_xyxy=b[k], scores=dec, labels=l[k]))
            elif fusion == "wbf":
                fb, fs, fl = weighted_boxes_fusion(b, s, l, iou_threshold=iou_cluster)
                result.append(DetectionPrediction(image_id=img_id, boxes_xyxy=fb, scores=fs, labels=fl))
            elif fusion == "best_proposal":
                k = nms(b, s, iou_threshold=iou_cluster)[:1]
                result.append(DetectionPrediction(image_id=img_id, boxes_xyxy=b[k], scores=s[k], labels=l[k]))
        return result

    def _select_by_type(data, type_name):
        type_id = NODE_TYPES[type_name]
        preds_by_img = defaultdict(lambda: {"b": [], "s": [], "l": []})
        for g, img_id, cid, _ in data:
            nb = g.metadata.get("node_box"); ns = g.metadata.get("node_score")
            nl = g.metadata.get("node_label"); nt_t = g.metadata.get("node_types")
            if nb is None or nt_t is None:
                continue
            mask = nt_t == type_id
            if not mask.any():
                continue
            preds_by_img[img_id]["b"].append(nb[mask])
            preds_by_img[img_id]["s"].append(ns[mask] if ns is not None else torch.ones(mask.sum()))
            preds_by_img[img_id]["l"].append(nl[mask] if nl is not None else torch.zeros(mask.sum(), dtype=torch.long))
        return [DetectionPrediction(
            image_id=img_id,
            boxes_xyxy=torch.cat(d["b"]) if d["b"] else torch.zeros(0, 4),
            scores=torch.cat(d["s"]) if d["s"] else torch.zeros(0),
            labels=torch.cat(d["l"]) if d["l"] else torch.zeros(0, dtype=torch.long),
        ) for img_id, d in preds_by_img.items()]

    baselines = {}
    for fusion in ("nms", "wbf", "soft_nms", "best_proposal"):
        baselines[f"external::{fusion}"] = _external_fusion(test_data, fusion)
    for tn in ("cluster", "consensus", "nms_candidate", "soft_nms_candidate", "best_proposal_candidate"):
        baselines[f"graph::{tn}"] = _select_by_type(test_data, tn)

    all_seed_results = []
    for seed in seeds:
        ckpt_path = run_dir / f"candidate_checkpoint_seed{seed}.pt"
        out_path  = run_dir / f"candidate_eval_seed{seed}.json"
        if out_path.exists() and not args.force:
            print(f"[eval-cns] seed {seed}: exists ({out_path}), skipping. Use --force to rerun.")
            r = json.loads(out_path.read_text())
            all_seed_results.append(r)
            continue
        if not ckpt_path.exists():
            print(f"[eval-cns] WARNING: checkpoint not found: {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        mc   = ckpt["model_config"]
        model_cfg = CandidateSelectorConfig(
            num_classes=mc["num_classes"],
            num_detectors=mc["num_detectors"],
            crop_size=mc["crop_size"],
            crop_channels=mc.get("crop_channels", 16),
            hidden_dim=mc.get("hidden_dim", 64),
            metadata_dim=mc.get("metadata_dim"),
            edge_feat_dim=mc.get("edge_feat_dim", 14),
            num_message_passing=mc.get("num_message_passing", 2),
            feature_mode=mc.get("feature_mode", "crop_metadata_mp"),
        )
        model = TGraphXCandidateNodeSelector(model_cfg).to(device)
        model.load_state_dict(ckpt["model_state"])

        # Select score head on val
        best_head = "p_tp50"; best_val_ap = -1.0
        val_score_modes = {}
        for sh in ("p_tp50", "p_tp75", "selection"):
            vp = _model_predict(model, val_data, sh)
            ap50 = _eval(vp, val_gts, iou_match)
            ap75 = _eval(vp, val_gts, 0.75)
            val_score_modes[sh] = {"val_ap50": ap50, "val_ap75": ap75}
            if ap75 > best_val_ap:
                best_val_ap = ap75; best_head = sh
        print(f"  [seed {seed}] score_head={best_head} val_ap75={best_val_ap:.4f}")

        # Test predictions with frozen score head
        tgx_test = _model_predict(model, test_data, best_head)
        test_methods = {"tgraphx_candidate_selector": tgx_test, **baselines}

        method_results = {}
        for n_m, preds in test_methods.items():
            method_results[n_m] = {
                "AP50": _eval(preds, test_gts, iou_match),
                "AP75": _eval(preds, test_gts, 0.75),
                "mIoU": _miou(preds, test_gts),
            }

        _, tgx_a50 = per_image_aps(tgx_test, test_gts, iou_threshold=iou_match, class_agnostic=class_agnostic)
        _, tgx_a75 = per_image_aps(tgx_test, test_gts, iou_threshold=0.75, class_agnostic=class_agnostic)
        bootstrap_ap50 = {}; bootstrap_ap75 = {}
        for n_m, preds in baselines.items():
            _, a50 = per_image_aps(preds, test_gts, iou_threshold=iou_match, class_agnostic=class_agnostic)
            _, a75 = per_image_aps(preds, test_gts, iou_threshold=0.75, class_agnostic=class_agnostic)
            if tgx_a50.shape == a50.shape:
                bootstrap_ap50[n_m] = paired_bootstrap(tgx_a50, a50, seed=seed)
            if tgx_a75.shape == a75.shape:
                bootstrap_ap75[n_m] = paired_bootstrap(tgx_a75, a75, seed=seed)

        tgx_r = method_results["tgraphx_candidate_selector"]
        r = {
            "seed": seed, "selected_score_head": best_head,
            "val_score_modes": val_score_modes,
            "test_methods": method_results,
            "paired_bootstrap_ap50": bootstrap_ap50,
            "paired_bootstrap_ap75": bootstrap_ap75,
            "headline_ap50": tgx_r["AP50"],
            "headline_ap75": tgx_r["AP75"],
        }
        out_path.write_text(json.dumps(r, indent=2, default=str))
        all_seed_results.append(r)
        print(f"  [seed {seed}] AP50={tgx_r['AP50']:.4f}  AP75={tgx_r['AP75']:.4f}  mIoU={tgx_r['mIoU']:.4f}")

    # Aggregate summary
    if all_seed_results:
        method_names = sorted(all_seed_results[0]["test_methods"].keys())
        means = {}
        for n_m in method_names:
            a50s = [s["test_methods"][n_m]["AP50"] for s in all_seed_results if n_m in s.get("test_methods", {})]
            a75s = [s["test_methods"][n_m]["AP75"] for s in all_seed_results if n_m in s.get("test_methods", {})]
            mious = [s["test_methods"][n_m]["mIoU"] for s in all_seed_results if n_m in s.get("test_methods", {})]
            if a50s:
                means[n_m] = {
                    "AP50_mean": statistics.mean(a50s), "AP50_std": statistics.stdev(a50s) if len(a50s) > 1 else 0.0,
                    "AP75_mean": statistics.mean(a75s), "AP75_std": statistics.stdev(a75s) if len(a75s) > 1 else 0.0,
                    "mIoU_mean": statistics.mean(mious),
                }
        summary = {"seeds": seeds, "method_means": means,
                    "detector_names": detector_names, "num_classes": num_classes}
        (run_dir / "candidate_eval_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print(f"[eval-cns] → {run_dir/'candidate_eval_summary.json'}")


if __name__ == "__main__":
    main()
