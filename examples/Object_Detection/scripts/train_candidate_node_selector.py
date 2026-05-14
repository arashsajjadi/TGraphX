"""Train TGraphXCandidateNodeSelector on object-level candidate graphs.

Reads:  {run_dir}/object_graphs.pt    (built by 03_build_object_candidate_graphs.py)
        {run_dir}/object_labels.pt
        {run_dir}/object_manifest.json

Writes: {run_dir}/candidate_checkpoint_seed{N}.pt
        {run_dir}/candidate_summary.json

Each entry in object_graphs.pt is already ONE object cluster.
The model receives the small per-cluster graph and selects the best candidate node.

Ablation modes (--feature-mode):
  crop_metadata_mp  : full TGraphX — spatial crops through tensor-aware ConvMP [default]
  flat_crop_mp      : crops flattened BEFORE MP (no spatial preservation)
  crop_no_mp        : CNN + metadata, no message passing
  metadata_only     : metadata MLP only, no crop tensors
"""
import argparse, json, statistics, sys, time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _compute_node_utility(
    node_box, node_score, node_label,
    gt_box, gt_label,
    class_agnostic: bool,
    mode: str = "deployable",
):
    """AP-style utility per candidate node against a GT box.

    AP50_utility = [TP@0.5 and cls_correct] + 0.05*IoU + 0.02*normalized_score
    AP75_utility = [TP@0.75 and cls_correct] + 0.05*IoU
    IoU_utility  = IoU if cls_correct else 0
    deployable   = 0.60*AP50 + 0.20*AP75 + 0.20*IoU

    Returns [N] utility tensor and [N] best_iou tensor.
    """
    from od_graph_fusion.box_ops import box_iou
    N = node_box.shape[0]
    G = gt_box.shape[0]
    if G == 0:
        return None, None

    ious = box_iou(node_box, gt_box)          # [N, G]
    best_iou, best_gt = ious.max(dim=1)       # [N]

    if node_label is not None and not class_agnostic:
        match_lbl = gt_label[best_gt]         # [N]
        cls_correct = (node_label == match_lbl).float()
    else:
        cls_correct = torch.ones(N, dtype=torch.float32)

    ns_norm = node_score.clamp(0, 1)
    tp50 = ((best_iou >= 0.50).float() * cls_correct)
    tp75 = ((best_iou >= 0.75).float() * cls_correct)
    iou_u = best_iou * cls_correct

    ap50_u = tp50 + 0.05 * best_iou + 0.02 * ns_norm
    ap75_u = tp75 + 0.05 * best_iou
    if mode == "ap50":
        util = ap50_u
    elif mode == "ap75":
        util = ap75_u
    elif mode == "iou":
        util = iou_u
    elif mode == "deployable":
        util = 0.60 * ap50_u + 0.20 * ap75_u + 0.20 * iou_u
    else:
        util = ap50_u
    return util, best_iou


import torch
import torch.nn.functional as F


def _train_one_seed(
    cfg: dict,
    seed: int,
    base_dir: Path,
    obj_graphs: list,
    obj_labels: dict,
    detector_names: list,
    class_names: list,
    *,
    epochs: int = 30,
    lr: float = 5e-4,
    device: str = "cpu",
    feature_mode: str = "crop_metadata_mp",
) -> dict:
    from od_graph_fusion.reproducibility import set_global_seed
    from od_graph_fusion.candidate_node_selector import (
        CandidateSelectorConfig, TGraphXCandidateNodeSelector,
        CandidateLossWeights, candidate_selector_loss, select_per_cluster,
    )
    from od_graph_fusion.candidate_mask import candidate_node_mask
    from od_graph_fusion.graph_builder import NODE_TYPES
    from od_graph_fusion.evaluation import DetectionPrediction, GroundTruth, evaluate_predictions
    from od_graph_fusion.baselines import nms, weighted_boxes_fusion, soft_nms
    from od_graph_fusion.box_ops import box_iou
    from od_graph_fusion.paired_bootstrap import paired_bootstrap, per_image_aps

    set_global_seed(seed, deterministic=False)
    num_classes = len(class_names)
    is_mc = num_classes > 2
    class_agnostic = bool(cfg.get("evaluation", {}).get("class_agnostic", not is_mc))
    iou_match = float(cfg.get("evaluation", {}).get("iou_match_threshold", 0.5))
    iou_cluster = float(cfg.get("graph", {}).get("cluster_iou_threshold", 0.5))
    crop_size = int(cfg.get("graph", {}).get("crop_size", 128))
    utility_mode = cfg.get("training", {}).get("utility_mode", "deployable")

    # ── Split data ────────────────────────────────────────────────────────
    train_data, val_data, test_data = [], [], []
    for entry in obj_graphs:
        g, img_id, cid, split, cand_src, gt_box, gt_lbl = entry
        key = f"{img_id}_{cid}"
        lbl_entry = obj_labels.get(key, {})
        sp = lbl_entry.get("split", split)
        if sp == "train":
            train_data.append(entry)
        elif sp == "val":
            val_data.append(entry)
        elif sp == "test":
            test_data.append(entry)

    # Shuffle training data deterministically
    g_rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_data), generator=g_rng).tolist()
    train_data = [train_data[i] for i in perm]
    print(f"  [seed {seed}] train={len(train_data)} val={len(val_data)} test={len(test_data)}")

    # ── Model config ──────────────────────────────────────────────────────
    g0 = obj_graphs[0][0] if obj_graphs else None
    metadata_dim = None
    edge_feat_dim = 14
    if g0 is not None:
        md = g0.metadata.get("node_metadata")
        metadata_dim = md.shape[1] if md is not None else None
        ea = g0.edge_features
        edge_feat_dim = ea.shape[1] if ea is not None and ea.numel() > 0 else 14

    model_cfg = CandidateSelectorConfig(
        num_classes=num_classes,
        num_detectors=len(detector_names),
        crop_size=crop_size,
        crop_channels=cfg.get("model", {}).get("crop_channels", 16),
        hidden_dim=cfg.get("model", {}).get("hidden_dim", 64),
        metadata_dim=metadata_dim,
        edge_feat_dim=edge_feat_dim,
        num_message_passing=cfg.get("model", {}).get("num_message_passing", 2),
        use_message_passing=cfg.get("model", {}).get("use_message_passing", True),
        use_metadata=cfg.get("model", {}).get("use_metadata", True),
        feature_mode=feature_mode,
    )
    # Use the attention factory for tgx_* variants
    attention_variants = {"tgx_edge_attention", "tgx_spatial_attention",
                          "tgx_hybrid_attention", "tgx_convmp_small", "tgx_convmp_full"}
    if feature_mode in attention_variants:
        from od_graph_fusion.attention_selector import build_selector
        model = build_selector(model_cfg, feature_mode).to(device)
    else:
        model = TGraphXCandidateNodeSelector(model_cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_weights = CandidateLossWeights(
        selection_ce=cfg.get("training", {}).get("lambda_selection", 1.0),
        tp50_bce=cfg.get("training", {}).get("lambda_tp50", 1.0),
        tp75_bce=cfg.get("training", {}).get("lambda_tp75", 2.0),
        iou_reg=cfg.get("training", {}).get("lambda_iou", 0.5),
        pairwise_rank=cfg.get("training", {}).get("lambda_rank", 0.5),
    )

    # ── Pre-compute training targets ──────────────────────────────────────
    # Each object graph is already ONE cluster, so:
    #   cluster_of = all zeros, cand_mask = all True for candidate nodes,
    #   best_node_per_cluster[0] = argmax(utility)
    print(f"  [seed {seed}] computing train targets …")
    train_targets = {}
    for entry in train_data:
        g, img_id, cid, split, cand_src, gt_box, gt_lbl = entry
        if gt_box is None or gt_box.numel() == 0:
            continue
        nb = g.metadata.get("node_box")
        ns = g.metadata.get("node_score")
        nl = g.metadata.get("node_label")
        nt = g.metadata.get("node_types")
        if nb is None:
            continue

        util, best_iou = _compute_node_utility(
            nb, ns if ns is not None else torch.zeros(nb.shape[0]),
            nl, gt_box, gt_lbl, class_agnostic, utility_mode)
        if util is None:
            continue

        cand_mask = candidate_node_mask(nt, NODE_TYPES) if nt is not None \
                    else torch.ones(nb.shape[0], dtype=torch.bool)

        # Zero utility for non-candidate nodes
        util_cand = util.clone()
        util_cand[~cand_mask] = -1e9
        best_node = util_cand.argmax()

        cluster_of = torch.zeros(nb.shape[0], dtype=torch.long)  # all in cluster 0
        best_node_per_cluster = torch.tensor([int(best_node.item())], dtype=torch.long)

        if nl is not None and not class_agnostic:
            match_lbl = gt_lbl[box_iou(nb, gt_box).max(dim=1)[1]]
            cls_correct = (nl == match_lbl)
        else:
            cls_correct = torch.ones(nb.shape[0], dtype=torch.bool)

        key = f"{img_id}_{cid}"
        train_targets[key] = {
            "cluster_of":            cluster_of,
            "cand_mask":             cand_mask,
            "best_node_per_cluster": best_node_per_cluster,
            "node_iou_with_gt":      best_iou,
            "node_class_correct":    cls_correct,
        }

    print(f"  [seed {seed}] valid train targets: {len(train_targets)}/{len(train_data)}")

    # ── Training loop ─────────────────────────────────────────────────────
    history = []
    best_model_state = model.state_dict()
    best_val_ap75 = -1.0

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0; n = 0
        for entry in train_data:
            g, img_id, cid, split, cand_src, _, _ = entry
            key = f"{img_id}_{cid}"
            t = train_targets.get(key)
            if t is None:
                continue
            gg = g.to(device)
            out = model(gg, detector_names=detector_names)
            losses = candidate_selector_loss(
                out,
                cluster_of=t["cluster_of"].to(device),
                cand_mask=t["cand_mask"].to(device),
                best_node_per_cluster=t["best_node_per_cluster"].to(device),
                node_iou_with_gt=t["node_iou_with_gt"].to(device),
                node_class_correct=t["node_class_correct"].to(device),
                weights=loss_weights,
            )
            loss = losses["total"]
            if not loss.requires_grad or losses.get("n_clusters", 0) == 0:
                continue
            optim.zero_grad(); loss.backward(); optim.step()
            total += float(loss.item()); n += 1
        avg = total / max(1, n)
        history.append(avg)

        if ep % max(1, epochs // 4) == 0 or ep == epochs or ep == 1:
            print(f"  [seed {seed}] ep {ep}/{epochs}  loss={avg:.4f}")

    # ── Score-head selection on val ───────────────────────────────────────
    def _predict_split(data, score_head):
        model.eval()
        preds_by_image: dict = defaultdict(lambda: {"boxes": [], "scores": [], "labels": []})
        with torch.no_grad():
            for entry in data:
                g, img_id, cid, sp, cand_src, _, _ = entry
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
                    preds_by_image[img_id]["boxes"].append(picked["boxes_xyxy"])
                    preds_by_image[img_id]["scores"].append(picked["scores"])
                    preds_by_image[img_id]["labels"].append(picked["labels"])

        result_preds = []
        for img_id, d in preds_by_image.items():
            if d["boxes"]:
                result_preds.append(DetectionPrediction(
                    image_id=img_id,
                    boxes_xyxy=torch.cat(d["boxes"]),
                    scores=torch.cat(d["scores"]),
                    labels=torch.cat(d["labels"]),
                ))
            else:
                result_preds.append(DetectionPrediction(
                    image_id=img_id, boxes_xyxy=torch.zeros(0, 4),
                    scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long),
                ))
        return result_preds

    def _make_gts(data):
        gts_by_image: dict = {}
        for entry in data:
            g, img_id, cid, sp, _, _, _ = entry
            if img_id in gts_by_image:
                continue
            key = f"{img_id}_{cid}"
            lbl = obj_labels.get(key, {})
            gt_b = lbl.get("gt_image_boxes", g.metadata.get("gt_image_boxes", torch.zeros(0, 4)))
            gt_l = lbl.get("gt_image_labels", g.metadata.get("gt_image_labels", torch.zeros(0, dtype=torch.long)))
            gts_by_image[img_id] = GroundTruth(image_id=img_id, boxes_xyxy=gt_b, labels=gt_l)
        return list(gts_by_image.values())

    val_gts = _make_gts(val_data)

    def _eval(preds, gts, iou_t):
        return evaluate_predictions(preds, gts, iou_threshold=iou_t,
                                      num_classes=num_classes, class_agnostic=class_agnostic)["AP"]

    val_score_modes = {}
    best_head = "p_tp50"; best_val_ap = -1.0
    for sh in ("p_tp50", "p_tp75", "selection"):
        vp = _predict_split(val_data, sh)
        ap50 = _eval(vp, val_gts, iou_match)
        ap75 = _eval(vp, val_gts, 0.75)
        val_score_modes[sh] = {"val_ap50": ap50, "val_ap75": ap75}
        sel_metric = ap75  # select on AP75 (main headroom metric)
        if sel_metric > best_val_ap:
            best_val_ap = sel_metric; best_head = sh
    print(f"  [seed {seed}] score_head={best_head} (val_ap75={best_val_ap:.4f})")

    # ── Final test evaluation ─────────────────────────────────────────────
    tgx_test = _predict_split(test_data, best_head)
    test_gts  = _make_gts(test_data)

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

    # Baseline predictions from graph node types (already in test data)
    def _select_by_type(data, type_name):
        from od_graph_fusion.graph_builder import NODE_TYPES as NT
        type_id = NT[type_name]
        preds_by_image = defaultdict(lambda: {"boxes": [], "scores": [], "labels": []})
        for entry in data:
            g, img_id, cid, sp, _, _, _ = entry
            nb = g.metadata.get("node_box"); ns = g.metadata.get("node_score")
            nl = g.metadata.get("node_label"); nt_t = g.metadata.get("node_types")
            if nb is None or nt_t is None:
                continue
            mask = nt_t == type_id
            if not mask.any():
                continue
            preds_by_image[img_id]["boxes"].append(nb[mask])
            preds_by_image[img_id]["scores"].append(ns[mask] if ns is not None else torch.ones(mask.sum()))
            preds_by_image[img_id]["labels"].append(nl[mask] if nl is not None else torch.zeros(mask.sum(), dtype=torch.long))
        result = []
        for img_id, d in preds_by_image.items():
            result.append(DetectionPrediction(
                image_id=img_id,
                boxes_xyxy=torch.cat(d["boxes"]) if d["boxes"] else torch.zeros(0, 4),
                scores=torch.cat(d["scores"]) if d["scores"] else torch.zeros(0),
                labels=torch.cat(d["labels"]) if d["labels"] else torch.zeros(0, dtype=torch.long),
            ))
        return result

    def _external_fusion(data, fusion):
        pool = defaultdict(lambda: {"b": [], "s": [], "l": []})
        for entry in data:
            g, img_id, cid, sp, _, _, _ = entry
            nb = g.metadata.get("node_box"); ns = g.metadata.get("node_score")
            nl = g.metadata.get("node_label"); nt_t = g.metadata.get("node_types")
            if nb is None or nt_t is None:
                continue
            mask = nt_t == NODE_TYPES_proposal_id
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

    from od_graph_fusion.graph_builder import NODE_TYPES as NT_ALL
    NODE_TYPES_proposal_id = NT_ALL["proposal"]

    methods = {"tgraphx_candidate_selector": tgx_test}
    for fusion in ("nms", "wbf", "soft_nms", "best_proposal"):
        methods[f"external::{fusion}"] = _external_fusion(test_data, fusion)
    for tn in ("cluster", "consensus", "nms_candidate", "soft_nms_candidate", "best_proposal_candidate"):
        methods[f"graph::{tn}"] = _select_by_type(test_data, tn)

    method_results = {}
    for n_m, preds in methods.items():
        method_results[n_m] = {
            "AP50": _eval(preds, test_gts, iou_match),
            "AP75": _eval(preds, test_gts, 0.75),
            "mIoU": _miou(preds, test_gts),
        }

    _, tgx_a50 = per_image_aps(tgx_test, test_gts, iou_threshold=iou_match, class_agnostic=class_agnostic)
    _, tgx_a75 = per_image_aps(tgx_test, test_gts, iou_threshold=0.75, class_agnostic=class_agnostic)
    bootstrap_ap50 = {}; bootstrap_ap75 = {}
    for n_m, preds in methods.items():
        if n_m == "tgraphx_candidate_selector":
            continue
        _, a50 = per_image_aps(preds, test_gts, iou_threshold=iou_match, class_agnostic=class_agnostic)
        _, a75 = per_image_aps(preds, test_gts, iou_threshold=0.75, class_agnostic=class_agnostic)
        if tgx_a50.shape == a50.shape:
            bootstrap_ap50[n_m] = paired_bootstrap(tgx_a50, a50, seed=seed)
        if tgx_a75.shape == a75.shape:
            bootstrap_ap75[n_m] = paired_bootstrap(tgx_a75, a75, seed=seed)

    tgx_res = method_results["tgraphx_candidate_selector"]
    metrics = {
        "seed": seed, "device": device,
        "feature_mode": feature_mode,
        "detector_names": detector_names, "num_classes": num_classes,
        "selected_score_head": best_head,
        "val_score_modes": val_score_modes,
        "test_metrics": {
            "AP50": tgx_res["AP50"], "AP75": tgx_res["AP75"], "mIoU": tgx_res["mIoU"],
            "headline_ap": tgx_res["AP75"],
        },
        "test_methods": method_results,
        "paired_bootstrap_ap50": bootstrap_ap50,
        "paired_bootstrap_ap75": bootstrap_ap75,
        "training_history": history,
        "num_train_graphs": len(train_data),
        "num_valid_targets": len(train_targets),
        "uses_object_level_graphs": True,
        "uses_candidate_node_selector": True,
    }
    ckpt_path = base_dir / f"candidate_checkpoint_seed{seed}.pt"
    torch.save({
        "model_state": model.state_dict(),
        "model_config": {
            "num_classes": num_classes, "num_detectors": len(detector_names),
            "crop_size": crop_size, "feature_mode": feature_mode,
            "crop_channels": model_cfg.crop_channels, "hidden_dim": model_cfg.hidden_dim,
            "num_message_passing": model_cfg.num_message_passing,
            "metadata_dim": metadata_dim, "edge_feat_dim": edge_feat_dim,
        },
        "metrics": metrics, "seed": seed,
    }, ckpt_path)
    (base_dir / f"candidate_metrics_seed{seed}.json").write_text(
        json.dumps(metrics, indent=2, default=str))

    print(f"  [seed {seed}] TGX AP50={tgx_res['AP50']:.4f}  AP75={tgx_res['AP75']:.4f}  "
          f"mIoU={tgx_res['mIoU']:.4f}  "
          f"NMS={method_results.get('external::nms', {}).get('AP50', 0):.4f} "
          f"WBF={method_results.get('external::wbf', {}).get('AP50', 0):.4f}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Train TGraphXCandidateNodeSelector")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=None)
    _ALL_VARIANTS = [
        "crop_metadata_mp", "flat_crop_mp", "crop_no_mp", "metadata_only",
        "tgx_convmp_small", "tgx_convmp_full",
        "tgx_edge_attention", "tgx_spatial_attention", "tgx_hybrid_attention",
    ]
    ap.add_argument("--feature-mode", default=None, choices=_ALL_VARIANTS)
    args = ap.parse_args()

    import torch
    from od_graph_fusion.config import load_config, resolve_device

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir or f"runs/{cfg.get('run_name', 'exp')}")

    obj_graphs_path = run_dir / "object_graphs.pt"
    obj_labels_path = run_dir / "object_labels.pt"
    manifest_path   = run_dir / "object_manifest.json"

    if not obj_graphs_path.exists():
        raise FileNotFoundError(
            f"[train-cns] Missing object_graphs.pt — run 03_build_object_candidate_graphs.py first: {obj_graphs_path}")

    print(f"[train-cns] loading object_graphs.pt …")
    obj_graphs  = torch.load(obj_graphs_path, weights_only=False)
    obj_labels  = torch.load(obj_labels_path, weights_only=False) if obj_labels_path.exists() else {}
    manifest    = json.loads(manifest_path.read_text())
    detector_names = manifest["detector_names"]
    class_names    = manifest.get("class_names", ["car"])

    device = resolve_device(args.device or cfg.get("device", "auto"))
    epochs = args.epochs or int(cfg.get("training", {}).get("epochs", 30))
    lr     = float(cfg.get("training", {}).get("lr", 5e-4))
    fm     = args.feature_mode or cfg.get("model", {}).get("feature_mode", "crop_metadata_mp")

    print(f"[train-cns] {len(obj_graphs)} object graphs | device={device} | epochs={epochs} | feature_mode={fm}")

    all_metrics = []
    for seed in args.seeds:
        t0 = time.time()
        print(f"\n[train-cns] ── seed {seed} ────────────────────────────────")
        m = _train_one_seed(
            cfg, seed, run_dir, obj_graphs, obj_labels,
            detector_names, class_names,
            epochs=epochs, lr=lr, device=device, feature_mode=fm,
        )
        m["elapsed_s"] = time.time() - t0
        all_metrics.append(m)

    # Summary across seeds
    summary = {"seeds": list(args.seeds), "detector_names": detector_names,
                "feature_mode": fm, "device": device, "epochs": epochs}
    if all_metrics:
        method_names = sorted(all_metrics[0]["test_methods"].keys())
        means = {}
        for n_m in method_names:
            a50s = [s["test_methods"][n_m]["AP50"] for s in all_metrics if n_m in s["test_methods"]]
            a75s = [s["test_methods"][n_m]["AP75"] for s in all_metrics if n_m in s["test_methods"]]
            mious = [s["test_methods"][n_m]["mIoU"] for s in all_metrics if n_m in s["test_methods"]]
            if a50s:
                means[n_m] = {
                    "AP50_mean": statistics.mean(a50s),
                    "AP50_std":  statistics.stdev(a50s) if len(a50s) > 1 else 0.0,
                    "AP75_mean": statistics.mean(a75s),
                    "AP75_std":  statistics.stdev(a75s) if len(a75s) > 1 else 0.0,
                    "mIoU_mean": statistics.mean(mious),
                }
        summary["method_means"] = means
    (run_dir / "candidate_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[train-cns] → {run_dir/'candidate_summary.json'}")


if __name__ == "__main__":
    main()
