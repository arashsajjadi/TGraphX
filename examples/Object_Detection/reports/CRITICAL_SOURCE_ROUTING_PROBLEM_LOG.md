# Critical source-routing problem log

## 1. Architecture problems

**P1.1** `TGraphXSourceRouter` has a `quality_head` that produces per-node scores `[N]`. This is NOT source-level routing. True routing requires output `source_logits [num_clusters, num_sources]` — one logit per source slot per cluster. The current per-node scorer can only be used as an auxiliary head.

**P1.2** No source slots are defined. There is no explicit assignment of candidate nodes to source slots (YOLO=0, DETR=1, YOLOE=2, RetinaNet=3, Union=4, WBF=5). Without fixed slots, the model cannot learn "which family is best for this object" — it can only rank candidates locally.

**P1.3** `EdgeConditionedMP` still uses a Python `for` loop over edges (lines with `for e_idx in range(ei.shape[1])`). This is O(E) in Python, unnormalized, and invisible to gradient tracking in a useful sense. On 60 images with hundreds of edges, this adds ~10 s/epoch overhead vs vectorized scatter.

**P1.4** The spatial-pool → flatten before fusion means tensor-native spatial structure is discarded before message passing is complete. The ConvMessagePassing layers process crops spatially, but then AdaptiveAvgPool2d(1) collapses everything before the quality head sees it.

## 2. Source-routing formulation problems

**P2.1 (CRITICAL) Utility is thresholded before ranking.** `compute_source_utilities` sets `utility[i] = 0` whenever `IoU < iou_match` (default 0.5). This destroys the ordering information that source routing needs. Example: if all four candidates have IoU = [0.49, 0.30, 0.15, 0.05], the function returns `best_source = arbitrary first index (0)` because all utilities are 0.0, not the correct index 0 (highest IoU). Verified by test: the argmax of an all-zeros tensor is 0 by default.

**P2.2** No source_mask is defined. There is no concept of "absent source" with masking to `-inf` before softmax. When a detector fires nothing for a cluster, there is no placeholder node with a mask flag — the source simply has no node in the graph.

**P2.3** The `best_source` label in `_build_targets_full` is only set for cluster/consensus nodes that matched GT with IoU ≥ 0.5. Proposal nodes get is_best_source only conditionally. If no node exceeds 0.5 IoU, no node gets is_best_source=1.0 — another consequence of bug P2.1.

## 3. Loss-function problems

**P3.1 (CRITICAL) Per-cluster regret weighting is wrong.** Current code computes:
```python
rw_scale = 1 + lambda * mean(regret_weights)
total = rw_scale * mean(CE + KL + pairwise)
```
This multiplies the **entire batch** by the mean regret. Hard clusters do NOT receive larger per-cluster loss gradients. The correct formulation requires: `L_total = mean_c[(1 + lambda * regret_c) * L_c]`.

**P3.2** The pairwise ranking loss uses a Python double-for-loop over candidate pairs within each cluster. For a cluster with 6 candidates, this is C(6,2)=15 pairs. Correct, but slow when many clusters exist.

**P3.3** No true focal variant of source CE loss exists. With 6 source slots and 3 absent, class imbalance across clusters is not handled.

## 4. Evaluator / oracle problems

**P4.1** Oracle is `localization_oracle` (best IoU per GT, ignores class). Reported as a deployable upper bound. The class-aware oracle is separate. With COCO-trained detectors, localization oracle can pick boxes with wrong labels, so class-aware oracle can be lower.

**P4.2** `source_selection_accuracy` is not computed anywhere in the current pipeline. The main metrics table only shows AP50.

**P4.3** Functional copying diagnostics are not computed: we do not know what % of TGraphX selections match NMS/best_proposal/highest-confidence.

**P4.4** Oracle-gap recovery is defined in `source_router.py` but not called from `cli.py` or `multi_seed.py`.

## 5. Split / statistics problems

**P5.1** Current 10-seed run varies train/val/test split AND model seed simultaneously (single `seed` controls both via `set_global_seed`). These should be separated into: (A) fixed split, vary model seed; (B) vary split seed, fixed model.

**P5.2** With 9 test images per seed, 95% bootstrap CI is dominated by test-set variance, not model variance. The headline result is not statistically meaningful for claims about NMS superiority.

**P5.3** "TGraphX wins 8/10 seeds" while mean AP is lower (0.8878 < 0.8922 NMS). This contradiction means TGraphX wins by small margins frequently but loses by larger margins in 2 seeds — not a robust win.

## 6. Stepwise pipeline problems

**P6.1** All of `01_download_data.py` through `06_make_report.py` call `run_pipeline(config)`. They are not stepwise at all — running step 3 reruns the detector too. This means:
  - Detector inference is rerun on every experiment even when outputs are cached.
  - A training-only change triggers full re-detection.

**P6.2** No artifact-skip logic. `--force` flag is not implemented.

## 7. Detector / canonical-label problems

**P7.1** Canonical label mapping (`canonical_label()` in `source_router.py`) is defined but never called in `YOLOAdapter.predict()`, `RTDETRAdapter.predict()`, or `RetinaNetAdapter.predict()`. Detector outputs still carry raw COCO label names. The graph builder receives these raw names and tries to look them up in `class_names`, failing for "airplane" (not in VOC list which has "aeroplane").

**P7.2** `canonical_label_id()` returns -1 for unknown classes. This -1 is stored in `label_ids` but is never masked out before being compared to GT labels, causing false FPs to be counted as FNs.

**P7.3** In the class-aware evaluation, wrong-label predictions (e.g., COCO "airplane" on VOC images) are silently ignored rather than properly flagged.

## 8. Synthetic reproducibility problems

**P8.1** `SyntheticDetector` uses `rng = random.Random(hash((image_id, seed)))`. `hash()` is randomized by PYTHONHASHSEED (default in Python 3.3+), so two subprocess runs give different synthetic outputs. `hashlib.sha256` fix is implemented in `source_router.py` but `SyntheticDetector` in `registry.py` still uses `hash()`.

## 9. Hard-case mining problems

**P9.1** Hard-case mining is not implemented. Training uses uniform sampling over all clusters. There is no oversampling of clusters where NMS/best-proposal selects a different source than the oracle.

**P9.2** Hard-case metrics (source accuracy on hard clusters, oracle-gap recovery on hard cases) are not reported.

## 10. AP-vs-source-routing metric problems

**P10.1** The main result table only shows AP50. Source-selection accuracy, oracle-gap recovery, regret, top-2 accuracy, and functional copying diagnostics are defined but never output to the report table.

**P10.2** "AP50" for TGraphX is really "localization-agnostic AP50 with class-agnostic matching." The reader might interpret this as standard VOC AP50 (class-aware). The distinction must be explicit.

## 11. Plotting / reporting problems

**P11.1** `plot_detection_graph_sketch` uses random node positions (no actual edge layout). It is decorative and misleading.

**P11.2** No source-selection confusion matrix, no source-by-class heatmap, no source-by-disagreement figure.

## 12. Git / release hygiene problems

**P12.1** `runs/`, `cache/`, `*.pt`, `*.tar` are gitignored. But `reports/figures/` are not explicitly ignored, and SVG files from runs can be large.

**P12.2** No standalone showcase repo. The experiment lives inside the main TGraphX repo under `examples/Object_Detection/`, which is only partially browsable without the full repo.

## Summary priority

| # | Problem | Severity | Fix effort |
|---|---------|----------|------------|
| P2.1 | Utility thresholding destroys ranking | CRITICAL | 30 min |
| P3.1 | Regret weighting is batch-level not per-cluster | CRITICAL | 30 min |
| P1.1 | No true source-slot logits | HIGH | 4 h |
| P1.3 | Python loop in edge-conditioned MP | HIGH | 2 h |
| P7.1 | Canonical labels not applied in adapters | HIGH | 1 h |
| P10.1 | Source-selection accuracy not in report table | HIGH | 2 h |
| P4.3 | No functional copying diagnostics | HIGH | 1 h |
| P8.1 | hash() in synthetic detector | MEDIUM | 30 min |
| P5.1 | Seeds mix split and model variability | MEDIUM | 1 h |
| P11.1 | Fake graph visualization | LOW | 2 h |
| P6.1 | Fake stepwise pipeline | LOW | 3 h |
