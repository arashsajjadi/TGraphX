"""Step 03 (object-level): Build per-cluster candidate node graphs.

Reads:  {run_dir}/detector_outputs.pt
        {run_dir}/detector_manifest.json
Writes: {run_dir}/object_graphs.pt    — list of per-cluster TGraphX Graphs
        {run_dir}/object_labels.pt    — GT labels keyed by (image_id, cluster_id)
        {run_dir}/object_manifest.json — dataset metadata + split IDs
        {run_dir}/object_graph_audit.json

Graph format: each entry is
  (graph, image_id, cluster_id, split, candidate_sources, gt_box, gt_label)

If image X has K car clusters, object_graphs.pt contains K entries for image X.
gt_box / gt_label are the best-matched GT box for that cluster.
For val/test entries gt_box is stored in object_labels.pt only (not in the
inference graph), preventing GT leakage.

Invariants:
- node_features is [N, 3, crop_size, crop_size]  (rank 4, no context padding)
- gt_boxes NOT in val/test graph metadata
- split determined from dataset record.split
- object_manifest records detector_names + split IDs
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description="Step 03: build object-level candidate graphs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import torch
    from od_graph_fusion.config import load_config
    from od_graph_fusion.datasets import load_dataset
    from od_graph_fusion.object_candidate_graphs import build_object_candidate_graphs

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir or f"runs/{cfg.get('run_name', 'exp')}")
    run_dir.mkdir(parents=True, exist_ok=True)

    out_path      = run_dir / "object_graphs.pt"
    labels_path   = run_dir / "object_labels.pt"
    manifest_path = run_dir / "object_manifest.json"
    cache_path    = run_dir / "detector_outputs.pt"

    if not cache_path.exists():
        raise FileNotFoundError(f"[03-obj] Missing detector_outputs.pt — run step 02 first: {cache_path}")
    if out_path.exists() and not args.force:
        print(f"[03-obj] object_graphs.pt exists: {out_path}  (--force to rerun)")
        return

    records = load_dataset(cfg)
    class_names = records[0].class_names if records else ["object"]
    detector_outputs = torch.load(cache_path, weights_only=False)
    detector_names = list(detector_outputs.keys())

    cfg_g = cfg.get("graph", {})
    iou_cluster = float(cfg_g.get("cluster_iou_threshold", 0.5))
    crop_size = args.crop_size or int(cfg_g.get("crop_size", 128))
    iou_match = float(cfg.get("evaluation", {}).get("iou_match_threshold", 0.5))
    max_props = int(cfg_g.get("max_proposals_per_image", 64)) // max(len(detector_names), 1)
    max_props = max(max_props, 5)

    t0 = time.time()
    all_obj_graphs = []       # (graph, image_id, cluster_id, split, candidate_sources, gt_box, gt_label)
    obj_labels = {}           # (image_id, cluster_id) → {gt_box, gt_label, split, image_id}
    split_ids  = {"train": [], "val": [], "test": []}
    n_images_with_clusters = 0

    for rec in records:
        split = getattr(rec, "split", "train")
        is_training = (split == "train")
        ii = {r.image_id: i for i, r in enumerate(records)}[rec.image_id]
        det_res = [detector_outputs[n][ii] for n in detector_names]

        obj_graphs = build_object_candidate_graphs(
            rec.image, rec.image_id, rec.image_size,
            det_res, detector_names, class_names,
            gt_boxes=rec.gt_boxes,
            gt_labels=rec.gt_labels,
            iou_cluster=iou_cluster,
            iou_match=iou_match,
            crop_size=crop_size,
            max_proposals_per_detector=max_props,
            split=split,
        )

        if obj_graphs:
            n_images_with_clusters += 1

        for g, img_id, cluster_id, sp, cand_src, gt_box, gt_lbl in obj_graphs:
            all_obj_graphs.append((g, img_id, cluster_id, sp, cand_src, gt_box, gt_lbl))
            key = f"{img_id}_{cluster_id}"
            obj_labels[key] = {
                "image_id":   img_id,
                "cluster_id": cluster_id,
                "split":      sp,
                "gt_box":    gt_box,
                "gt_label":  gt_lbl,
                "gt_image_boxes":  rec.gt_boxes.clone() if rec.gt_boxes is not None else torch.zeros(0, 4),
                "gt_image_labels": rec.gt_labels.clone() if rec.gt_labels is not None else torch.zeros(0, dtype=torch.long),
            }
            cluster_key = f"{img_id}_{cluster_id}"
            if cluster_key not in split_ids[sp]:
                split_ids[sp].append(cluster_key)

    torch.save(all_obj_graphs, out_path)
    torch.save(obj_labels, labels_path)

    manifest = {
        "detector_names": detector_names,
        "class_names":    class_names,
        "num_classes":    len(class_names),
        "crop_size":      crop_size,
        "split_ids":      split_ids,
        "num_train":      len(split_ids["train"]),
        "num_val":        len(split_ids["val"]),
        "num_test":       len(split_ids["test"]),
        "total_object_graphs": len(all_obj_graphs),
        "n_images_with_clusters": n_images_with_clusters,
        "config": str(args.config),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    elapsed = round(time.time() - t0, 1)
    audit = {
        "total_object_graphs": len(all_obj_graphs),
        "n_images_with_clusters": n_images_with_clusters,
        "split_counts": {k: len(v) for k, v in split_ids.items()},
        "detector_names": detector_names,
        "crop_size": crop_size,
        "elapsed_s": elapsed,
    }
    (run_dir / "object_graph_audit.json").write_text(json.dumps(audit, indent=2))

    print(f"[03-obj] Done. {len(all_obj_graphs)} object graphs → {out_path}  ({elapsed}s)")
    print(f"  Split: train={len(split_ids['train'])} val={len(split_ids['val'])} test={len(split_ids['test'])}")
    print(f"  Detectors: {detector_names}")
    print(f"  crop_size: {crop_size}")


if __name__ == "__main__":
    main()
