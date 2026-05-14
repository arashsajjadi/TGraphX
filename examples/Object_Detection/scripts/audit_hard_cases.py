"""Audit hard-case counts from the training graphs.

Reads:
  - graphs.pt or builds them on the fly via the multi-seed pipeline
  - split_manifest.json for train/val/test split assignment

Writes:
  - {run_dir}/hard_case_audit.json — counts per split, per type

Usage:
  python scripts/audit_hard_cases.py --config configs/real_voc2007_car_anchor_router.yaml --device auto
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from od_graph_fusion.config import load_config, resolve_device
    from od_graph_fusion.datasets import load_dataset
    from od_graph_fusion.detectors import build_detectors
    from od_graph_fusion.graph_builder import build_detection_graph
    from od_graph_fusion.multi_seed_v2 import _attach_slot_metadata, _split_seeded, _build_util_and_labels
    from od_graph_fusion.hard_cases import build_descriptors, hard_case_counts
    from od_graph_fusion.source_priors import select_anchor_on_validation
    from od_graph_fusion.multi_seed_anchor import _cluster_meta, _compute_baseline_aps_val

    cfg = load_config(args.config)
    device = resolve_device(args.device or cfg.get("device", "auto"))
    cfg["device"] = device
    seed = cfg.get("seeds", [42])[0]

    out_path = Path(args.run_dir or f"runs/{cfg.get('run_name', 'anchor_audit')}_audit")
    out_path.mkdir(parents=True, exist_ok=True)

    records = load_dataset(cfg)
    class_names = records[0].class_names if records else ["car"]
    is_multiclass = len(class_names) > 2
    class_agnostic = bool(cfg.get("evaluation", {}).get("class_agnostic", not is_multiclass))
    detectors = build_detectors(dict(cfg, device=device), class_names)
    detector_names = list(detectors.keys())

    # Detector outputs
    det_outputs = {n: [] for n in detector_names}
    for rec in records:
        for name, det in detectors.items():
            try:
                if "synthetic" in det.model_identifier():
                    res = det.predict(rec.image, rec.image_id, class_filter=class_names,
                                       gt_boxes=rec.gt_boxes, gt_labels=rec.gt_labels)
                else:
                    res = det.predict(rec.image, rec.image_id, class_filter=class_names)
            except Exception as e:
                res = det.empty_result(rec.image_id, rec.image_size, error=str(e))
            det_outputs[name].append(res)

    by_split = _split_seeded(records, seed)
    idx_by_id = {r.image_id: i for i, r in enumerate(records)}
    cfg_graph = cfg.get("graph", {})
    iou_cluster = float(cfg_graph.get("cluster_iou_threshold", 0.5))
    crop_size = int(cfg_graph.get("crop_size", 64))
    max_props = int(cfg_graph.get("max_proposals_per_image", 48))
    iou_match = float(cfg.get("evaluation", {}).get("iou_match_threshold", 0.5))
    utility_mode = cfg.get("training", {}).get("utility_mode", "ap50")

    def _build_graphs(recs, is_training):
        gs = []
        for rec in recs:
            ri = idx_by_id[rec.image_id]
            det_res = [det_outputs[n][ri] for n in detector_names]
            g, meta = build_detection_graph(
                rec.image, rec.image_id, rec.image_size, det_res,
                detector_names, class_names,
                gt_boxes=rec.gt_boxes, gt_labels=rec.gt_labels,
                iou_cluster=iou_cluster, iou_match=iou_match,
                crop_size=crop_size, max_proposals=max_props,
                include_context_node=True, include_consensus_nodes=True,
                is_training=is_training,
            )
            _attach_slot_metadata(g, meta, detector_names)
            gs.append((g, meta, rec))
        return gs

    train_data = _build_graphs(by_split["train"], True)
    val_data   = _build_graphs(by_split["val"], False)
    test_data  = _build_graphs(by_split["test"], False)

    # Anchor from val baselines
    _val_preds, val_gts, val_method_ap = _compute_baseline_aps_val(
        val_data, detector_names, len(class_names), class_agnostic, iou_match, iou_cluster,
    )
    anchor_mode = cfg.get("model", {}).get("anchor_mode", "validation_best_global_source")
    from od_graph_fusion.source_priors import select_anchor_on_validation
    anchor_slot, anchor_label = select_anchor_on_validation(
        val_method_ap, detector_names=detector_names, anchor_mode=anchor_mode,
    )
    print(f"[audit] anchor={anchor_label} slot={anchor_slot}")

    def _descriptor_input_for(split_data):
        di = []
        for gi, (g, meta, rec) in enumerate(split_data):
            lbls = _build_util_and_labels(g, meta, rec.gt_boxes, rec.gt_labels,
                                            class_agnostic=class_agnostic,
                                            utility_mode=utility_mode)
            if lbls is None:
                continue
            _, _bs, _bl, util_per_slot, slot_avail = lbls
            cmeta = _cluster_meta(g, meta, anchor_slot)
            if not cmeta:
                continue
            for c in range(util_per_slot.shape[0]):
                di.append({
                    "graph_idx": gi, "cluster_id": c,
                    "slot_utility": util_per_slot[c],
                    "slot_avail": slot_avail[c],
                    "anchor_score": float(cmeta["cluster_anchor_score"][c].item()),
                    "iou_disagreement": float(cmeta["pairwise"][c, :, 0].abs().mean().item()),
                })
        return di

    out = {"anchor_label": anchor_label, "anchor_slot": anchor_slot}
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        di = _descriptor_input_for(data)
        descs = build_descriptors(di, anchor_slot=anchor_slot)
        out[name] = hard_case_counts(descs)
        print(f"[audit] {name}: total_clusters={out[name].get('_total_clusters', 0)}  "
              f"any_hard={out[name].get('_any_hard', 0)}")
    (out_path / "hard_case_audit.json").write_text(json.dumps(out, indent=2))
    print(f"[audit] → {out_path/'hard_case_audit.json'}")


if __name__ == "__main__":
    main()
