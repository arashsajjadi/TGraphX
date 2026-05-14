"""Step 03: Build detection graphs and source labels. Write split manifest.

Reads:  {run_dir}/detector_outputs.pt
        {run_dir}/detector_manifest.json
Writes: {run_dir}/graphs.pt           — inference-safe, split-tagged
        {run_dir}/source_labels.pt    — training-only GT labels
        {run_dir}/split_manifest.json — deterministic train/val/test image IDs
        {run_dir}/graph_audit.json

Invariants enforced:
- graphs.pt stores (graph, meta, image_id, split) — no GT in inference graph
- is_training=True only for train graphs (controls whether meta.targets is filled)
- GT boxes/labels stored in source_labels.pt, not inside graph node features
- split determined from record.split field (dataset-assigned) or config seed
- detector_names stored in manifest and attached to graph.metadata

Does NOT run detectors, train, or evaluate.
Does NOT call run_pipeline.
Skips if graphs.pt exists unless --force.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description="Step 03: build detection graphs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import torch
    from od_graph_fusion.config import load_config
    from od_graph_fusion.datasets import load_dataset
    from od_graph_fusion.graph_builder import build_detection_graph
    from od_graph_fusion.multi_seed_v2 import _attach_slot_metadata

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir or f"runs/{cfg.get('run_name', 'exp')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "graphs.pt"
    labels_path = run_dir / "source_labels.pt"
    manifest_path = run_dir / "split_manifest.json"
    cache_path = run_dir / "detector_outputs.pt"

    if not cache_path.exists():
        raise FileNotFoundError(f"[03] Missing detector_outputs.pt — run step 02 first: {cache_path}")
    if out_path.exists() and not args.force:
        print(f"[03] graphs.pt exists: {out_path}  (--force to rerun)")
        return

    records = load_dataset(cfg)
    class_names = records[0].class_names if records else ["object"]
    detector_outputs = torch.load(cache_path, weights_only=False)
    detector_manifest = json.loads((run_dir / "detector_manifest.json").read_text()) \
        if (run_dir / "detector_manifest.json").exists() else {}
    detector_names = list(detector_outputs.keys())
    idx_map = {r.image_id: i for i, r in enumerate(records)}

    cfg_g = cfg.get("graph", {})
    iou_cluster = float(cfg_g.get("cluster_iou_threshold", 0.5))
    crop_size = int(cfg_g.get("crop_size", 64))
    max_props = int(cfg_g.get("max_proposals_per_image", 48))
    iou_match = float(cfg.get("evaluation", {}).get("iou_match_threshold", 0.5))

    t0 = time.time()
    all_graphs = []           # (graph, meta, image_id, split)
    source_labels = {}        # image_id → {gt_boxes, gt_labels}
    split_ids = {"train": [], "val": [], "test": []}
    n_nodes_total = 0; n_edges_total = 0

    for rec in records:
        split = getattr(rec, "split", "train")  # use dataset-assigned split
        is_training = (split == "train")
        ii = idx_map[rec.image_id]
        det_res = [detector_outputs[n][ii] for n in detector_names]

        # Guard: real experiments must not pass GT to detectors
        g, meta = build_detection_graph(
            rec.image, rec.image_id, rec.image_size, det_res,
            detector_names, class_names,
            gt_boxes=rec.gt_boxes, gt_labels=rec.gt_labels,
            iou_cluster=iou_cluster, iou_match=iou_match,
            crop_size=crop_size, max_proposals=max_props,
            include_context_node=cfg_g.get("include_context_node", False),
            include_consensus_nodes=cfg_g.get("include_consensus_nodes", True),
            is_training=is_training,
        )
        _attach_slot_metadata(g, meta, detector_names)
        # Attach detector_names and GT to graph metadata
        if isinstance(g.metadata, dict):
            g.metadata["detector_names"] = detector_names
            # GT stored here only for training loss (Step 04 V3 path).
            # These are NOT used by inference (fuse_v3 ignores them).
            g.metadata["gt_boxes"]  = rec.gt_boxes.clone()
            g.metadata["gt_labels"] = rec.gt_labels.clone()

        all_graphs.append((g, meta, rec.image_id, split))
        # GT labels stored separately, not in inference graph
        source_labels[rec.image_id] = {
            "gt_boxes": rec.gt_boxes.clone(),
            "gt_labels": rec.gt_labels.clone(),
            "split": split,
        }
        split_ids[split].append(rec.image_id)
        n_nodes_total += g.num_nodes
        n_edges_total += g.edge_index.shape[1]

    torch.save(all_graphs, out_path)
    torch.save(source_labels, labels_path)
    manifest = {
        "detector_names": detector_names,
        "class_names": class_names,
        "num_classes": len(class_names),
        "split_ids": split_ids,
        "num_train": len(split_ids["train"]),
        "num_val": len(split_ids["val"]),
        "num_test": len(split_ids["test"]),
        "config": str(args.config),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    audit = {
        "num_graphs": len(all_graphs),
        "avg_nodes": n_nodes_total / max(1, len(all_graphs)),
        "avg_edges": n_edges_total / max(1, len(all_graphs)),
        "detector_names": detector_names,
        "split_counts": {k: len(v) for k, v in split_ids.items()},
        "elapsed_s": round(time.time()-t0, 1),
    }
    (run_dir / "graph_audit.json").write_text(json.dumps(audit, indent=2))
    print(f"[03] Done. {len(all_graphs)} graphs → {out_path}  ({audit['elapsed_s']}s)")
    print(f"     Split: train={len(split_ids['train'])} val={len(split_ids['val'])} test={len(split_ids['test'])}")
    print(f"     avg nodes/graph={audit['avg_nodes']:.1f}  avg edges/graph={audit['avg_edges']:.1f}")
    print(f"     split_manifest → {manifest_path}")


if __name__ == "__main__":
    main()
