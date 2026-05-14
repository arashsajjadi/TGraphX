# Object-Level Node Classification — Code Audit

**Date:** 2026-05-13  
**Auditor:** Claude Opus (TGraphX implementation lead)  
**Branch:** main  
**Experiment:** `universal_candidate_voc_car`

---

## Summary

The existing codebase contained a well-designed `TGraphXCandidateNodeSelector`
class and supporting infrastructure (`candidate_node_selector.py`,
`multi_seed_candidate_selector.py`), but **Step 04 and Step 05 of the main
pipeline still routed through `TGraphXSourceRouterV3`** (source-slot routing),
and **Step 03 produced image-level graphs** (one per image) rather than the
required per-cluster object-level graphs (one per object hypothesis).

This audit documents all 10 required invariants, their status before and after
the fixes applied in this session.

---

## 10-Question Audit

### Q1. Does the current main train path use TGraphXCandidateNodeSelector?

| Before | After |
|--------|-------|
| **NO** — `04_train_tgraphx_fusion.py` called `train_fusion_model()` which trained `TGraphXSourceRouterV3`. `train_candidate_node_selector.py` existed but required `--run-dir-with-graphs` (not `--config`). | **YES** — `train_candidate_node_selector.py` rewritten to accept `--config` directly and reads `object_graphs.pt` (per-cluster graphs). `TGraphXCandidateNodeSelector` is the only model trained on the main path. |

**Fix applied:** Rewrote `scripts/train_candidate_node_selector.py` to:
- Accept `--config` (derives `run_dir` automatically)
- Load `object_graphs.pt` (per-cluster format)
- Train `TGraphXCandidateNodeSelector` with the paper-faithful 5-component loss
- Select score head on VAL, evaluate on TEST

### Q2. Does Step 04 train candidate-node selection, or does it still train TGraphXSourceRouterV3?

| Before | After |
|--------|-------|
| **NO** — `04_train_tgraphx_fusion.py` called `train_fusion_model()` → `TGraphXSourceRouterV3` with source-slot CE loss. | `04_train_tgraphx_fusion.py` is **retained as an ablation/comparison path** only. The new main training entry point is `train_candidate_node_selector.py`. |

**Verdict:** Step 04 is now ablation infrastructure. The main experiment pipeline (Section 12 run order) does **not** go through Step 04.

### Q3. Does Step 05 evaluate TGraphXCandidateNodeSelector, or does it still reconstruct TGraphXSourceRouterV3?

| Before | After |
|--------|-------|
| **NO** — `05_evaluate.py` explicitly reconstructed `TGraphXSourceRouterV3` and used `source_logits` for inference. | New `scripts/evaluate_candidate_node_selector.py` reconstructs `TGraphXCandidateNodeSelector` from checkpoint `model_config`, uses `selection_logit` → `argmax` → `node_box[selected_node]`. |

**Fix applied:** Created `scripts/evaluate_candidate_node_selector.py`.

### Q4. Does Step 03 produce one graph per object cluster?

| Before | After |
|--------|-------|
| **NO** — `03_build_detection_graphs.py` called `build_detection_graph()` producing one graph per image. Cluster separation was done inside training loops. | **YES** — New `scripts/03_build_object_candidate_graphs.py` calls `build_object_candidate_graphs()` (new module) to produce one `Graph` per detection cluster. If image X has K car clusters, `object_graphs.pt` contains K entries for image X. |

**Fix applied:** Created:
- `src/od_graph_fusion/object_candidate_graphs.py` — new graph builder
- `scripts/03_build_object_candidate_graphs.py` — new Step 03 variant

### Q5. Are crop tensors present for every detector node and every fusion node?

| Before | After |
|--------|-------|
| **YES** (image-level graphs). Proposal, cluster, consensus, NMS, Soft-NMS, BestProposal nodes all had crops. WBF was represented as the cluster node. | **YES** — Object-level graphs carry the same node types with crops: detector proposals (one per detector in cluster), WBF/cluster node, union/consensus node, NMS node, Soft-NMS node, BestProposal node. All tested in `test_object_candidate_graphs.py`. |

**Node types with crops in the new format:**

| Node type             | Source              | Crop present |
|-----------------------|---------------------|:------------:|
| `proposal`            | Detector (one per)  | ✓            |
| `cluster`             | WBF box             | ✓            |
| `consensus`           | Union box           | ✓            |
| `nms_candidate`       | NMS top-1           | ✓            |
| `soft_nms_candidate`  | Soft-NMS decay pick | ✓            |
| `best_proposal_candidate` | Top-score box  | ✓            |

### Q6. Is the main output selected_node_id, not selected_source_id?

| Before | After |
|--------|-------|
| **NO for the main path** — Step 05 used `source_logits [C, S]` from V3, choosing a source slot (not a node index). | **YES** — `TGraphXCandidateNodeSelector` outputs `selection_logit [N]` per node. Inference: `selected_node = argmax(selection_logit)`. No source-slot indirection. |

### Q7. Is selected_box exactly node_box[selected_node]?

| Before | After |
|--------|-------|
| YES in `select_per_cluster` (CandidateNodeSelector path). NO for V3 path (used `sni` slot-node-idx lookup). | **YES** — `select_per_cluster` does `node_box[pick]` exactly. Verified by `test_selected_box_equals_node_box_entry`. |

No box regression. No generated boxes. The selected box is always exactly one of the precomputed candidate node boxes.

### Q8. Are source-router, anchor-router, and learned-box-fusion only ablations?

| Before | After |
|--------|-------|
| **NO** — Source router was the main Step 04/05 path. | **YES** — `TGraphXSourceRouterV3` is now reused **only as an internal tensor-aware encoder** inside `TGraphXCandidateNodeSelector` (the `_v3` attribute). It is not the final model, loss, or report subject. `04_train_tgraphx_fusion.py` and `05_evaluate.py` remain for comparison/ablation but are not part of the main run order. |

### Q9. Does the graph preserve [C,H,W] tensor features through message passing before pooling?

| Status |
|--------|
| **YES** — `TGraphXSourceRouterV3` uses `ConvMessagePassing` (edge-conditioned) over [N, C, H, W] crop tensors before `AdaptiveAvgPool2d`. The spatial dimensions are preserved during message passing; only pooled at the end. This is the paper-faithful `feature_mode="crop_metadata_mp"` ablation. |

The `flat_crop_mp` ablation deliberately flattens before MP to measure the value of spatial preservation.

### Q10. Are detector/fusion candidates represented as visual crop nodes, not only metadata?

| Status |
|--------|
| **YES** — `graph.node_features` is `[N, 3, crop_size, crop_size]` for every node. The metadata vector is stored in `graph.metadata["node_metadata"]` as an auxiliary feature, not as the primary feature. Tests verify `node_features.ndim == 4`. |

---

## Fixes Applied

| File | Action |
|------|--------|
| `src/od_graph_fusion/object_candidate_graphs.py` | **Created** — per-cluster graph builder |
| `scripts/03_build_object_candidate_graphs.py` | **Created** — builds `object_graphs.pt` |
| `scripts/train_candidate_node_selector.py` | **Rewritten** — `--config` arg, object-level graphs, CandidateNodeSelector only |
| `scripts/evaluate_candidate_node_selector.py` | **Created** — evaluates CandidateNodeSelector |
| `scripts/report_candidate_node_selector.py` | **Created** — generates final report |
| `scripts/universal_candidate_oracle_audit.py` | **Updated** — accepts `--config` arg |
| `tests/test_object_candidate_graphs.py` | **Created** — 13 tests, all passing |

---

## Tests Added

All 13 new tests pass. The full suite (227 tests) has zero failures.

Key invariants now tested:
1. K clusters → K object graphs for that image
2. Overlapping boxes → single cluster
3. `node_features` rank 4: [N, 3, H, W]
4. No all-zero crop tensors
5. Fusion nodes (WBF, union, NMS, Soft-NMS, BestProposal) have crops
6. `node_box` shape matches node count
7. `selected_box = node_box[argmax(logit)]` exactly
8. GT NOT in val/test graph metadata
9. GT IS in train graph metadata
10. All nodes in cluster 0 (object-level graph = one cluster)
11. Empty detections → empty list
12. `candidate_sources` contains detector and fusion source names
13. Tuple format: `(Graph, image_id, cluster_id, split, sources, gt_box, gt_label)`

---

## Run Order (Section 12)

```bash
python -m pytest tests -q

python scripts/audit_detector_availability.py \
  --config configs/universal_candidate_voc_car.yaml

python scripts/02_run_detectors.py \
  --config configs/universal_candidate_voc_car.yaml \
  --device auto --force

python scripts/03_build_object_candidate_graphs.py \
  --config configs/universal_candidate_voc_car.yaml \
  --crop-size 128 --force

python scripts/universal_candidate_oracle_audit.py \
  --config configs/universal_candidate_voc_car.yaml

python scripts/train_candidate_node_selector.py \
  --config configs/universal_candidate_voc_car.yaml \
  --device auto --seeds 0 1 2 3 4

python scripts/evaluate_candidate_node_selector.py \
  --config configs/universal_candidate_voc_car.yaml

python scripts/report_candidate_node_selector.py \
  --config configs/universal_candidate_voc_car.yaml
```

---

## Conclusion

All 10 required invariants are now satisfied. The main experiment pipeline
trains and evaluates `TGraphXCandidateNodeSelector` on per-cluster
object-level graphs. The source-router, anchor-router, and learned-box-fusion
paths are ablation infrastructure only.

**OBJECT_LEVEL_NODE_CLASSIFICATION_IMPLEMENTED**
