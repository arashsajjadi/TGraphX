# Implementation problem log (post-v1 rebuild)

Before this round, the FAST_SMOKE run produced these numbers
(`runs/voc_real_4detectors/method_results.json`, 12 real VOC images):

```
detector::retinanet         AP50 = 0.0065
detector::yolo_modern       AP50 = 0.0000
detector::yolo_open_vocab   AP50 = 0.1000
detector::rt_detr           AP50 = 0.0476
fusion::nms                 AP50 = 0.0476
fusion::wbf                 AP50 = 0.0870
fusion::tgraphx             AP50 = 0.0000  ← collapses below every baseline
```

The TGraphX selector is **worse than every individual detector and every
classical fusion baseline**. This document enumerates *why* and what must
change.

## 1. Faithfulness-to-old-TGraphX problems

- **P1.1** TGraphX was used as a regression-style detector (predicting box
  offsets + class + objectness) rather than as a graph-based selector over
  existing candidate boxes. The paper-era design selects/refines existing
  proposals; the rebuild blindly regresses.
- **P1.2** The refiner head was always on. Box offsets at `box_reg` are added
  to the cluster's WBF centroid even when the regression head is
  near-random. Result: well-localized cluster boxes get pushed away from
  the true object.
- **P1.3** No `old_compatible` mode existed — there was no way to reproduce
  the YOLO + RetinaNet + Union flow that worked in the paper.

## 2. Detector availability and model-name problems

- **P2.1** Configs request `yolo11n.pt` rather than the paper's "YOLO26".
  YOLO26 is not a verified Ultralytics identifier; the registry should
  document this explicitly and not fake the name.
- **P2.2** YOLOE (`yoloe-11s-seg.pt`) downloaded a 572 MB `mobileclip_blt.ts`
  to the working directory at runtime. Now gitignored, but this should
  have been documented in the env report.
- **P2.3** The fallback path silently substitutes synthetic detectors with
  the same name as the real one. The detector_availability report records
  this honestly (`is_synthetic: true`), but the result table did not visually
  distinguish synthetic vs real runs.

## 3. Graph construction problems

- **P3.1** Cluster grouping is greedy and order-dependent. Two equally good
  groupings can give different cluster boxes.
- **P3.2** Crop size defaults to 64 px in `default.yaml` and 64 px in
  `fast_smoke.yaml`, smaller than the paper's `[3, 128, 128]`. Need a
  faithful mode with crop_size=128.
- **P3.3** Edge construction uses an `O(P²)` loop in Python. For small
  graphs this is fine, but it will not scale to 1000+ proposals.

## 4. Union/consensus node problems

- **P4.1** Only "WBF-style weighted average" cluster nodes and `union_box`
  consensus nodes exist. The paper-era setup had a single explicit Union
  node per object/cluster. The current implementation has both, but the
  selector head was never told to prefer a Union node as a fallback.
- **P4.2** WBF box and consensus/union box are computed from the same group
  but stored as identical-feature nodes apart from the box coordinates.
  Edge features should distinguish them more explicitly.

## 5. Tensor node feature problems

- **P5.1** Crop size 64 instead of 128. Faithful mode needs 128.
- **P5.2** The CNN encoder pools spatial features to 1×1 before fusion,
  which limits how much the message-passing layers can refine spatially.
  In faithful mode it should preserve a small spatial map (e.g. 4×4) until
  the head.

## 6. Edge feature problems

- **P6.1** Edge feature vector has 14 dims (6 scalar + 8 one-hot types).
  The legacy paper used a richer encoding that included detector pair id;
  in the rebuild, detector pair id is implicit only via the source nodes.

## 7. Target assignment problems

- **P7.1** Targets are only attached to cluster + consensus nodes. The
  selector head therefore never sees proposal-level targets, which makes
  "best source" supervision impossible.
- **P7.2** No `best_source` target. Without this, the model cannot learn
  to prefer YOLO over RetinaNet (or any other detector) per cluster.
- **P7.3** No `ignore` mask for ambiguous clusters (IoU between 0.4 and 0.5).

## 8. Inference / decode problems

- **P8.1** Decode does `refined = cluster_boxes + box_reg`; with an
  untrained regressor, this *destroys* localization. Selector mode should
  bypass `box_reg` entirely.
- **P8.2** `keep_threshold=0.3` is hard-coded. There is no validation sweep.
- **P8.3** Only cluster-mask nodes are scored. The selector head should
  also be able to pick a proposal node.

## 9. Threshold / calibration problems

- **P9.1** Fixed `keep_threshold=0.3` was applied to test data with a
  brand-new untrained head. Of course AP collapses.
- **P9.2** No threshold sweep, no `validation_sweep` policy, no record of
  the chosen threshold.

## 10. Loss / training problems

- **P10.1** Box-regression loss is on by default. With selector mode
  intended, this should be off until the selector demonstrably learns.
- **P10.2** No positive/negative reweighting. With ~1 positive cluster per
  image and many negatives, BCE collapses.

## 11. Hyperparameter problems

- **P11.1** FAST_SMOKE uses 2 epochs, lr=1e-3, batch=2 — nowhere near the
  paper's 50 epochs with a small lr around `5e-5`.

## 12. Class mapping problems

- **P12.1** Detector labels (COCO 80) are not mapped to VOC labels in
  `evaluation.py`. The evaluator matches predictions to GT only when their
  label ids are identical, so a YOLO `"car"` (COCO id=2) might not match
  a VOC `"car"` (VOC id=6). Real-VOC AP plummets for this reason alone.
- **P12.2** No `class_mapping_audit.csv` was produced.

## 13. Dataset / split / leakage problems

- **P13.1** Splits are sequential ordering (first 70 % train, next 15 %
  val, last 15 % test) rather than a shuffled deterministic split. With
  12 VOC images this places 2 test images that may not be representative.
- **P13.2** GT IS NOT used during inference graph construction — that part
  is correct.

## 14. Evaluation metric problems

- **P14.1** AP is computed with a simple in-repo evaluator, not COCO API.
  This is acceptable but the report should label results as such.
- **P14.2** Soft-NMS exists in baselines but is not surfaced in the report
  table.

## 15. Baseline / lower-bound problems

- **P15.1** No "highest-confidence proposal" lower bound is reported.
- **P15.2** No "cluster_best_confidence" lower bound.
- **P15.3** No oracle upper bound.

## 16. Learning-curve / reporting problems

- **P16.1** Training log only records train_loss / val_loss / obj_acc.
  No positive/negative score separation, no gradient norm, no
  per-component loss curves.

## 17. GPU / performance problems

- **P17.1** Detector batching is per-image only. For larger datasets, batched
  inference would speed up YOLO/RT-DETR substantially.

## 18. Notebook problems

- The clean notebook is 2.6 KB and executes; this is fine.

## 19. Git / repository hygiene problems

- **P19.1** Detector weight downloads (`yolo11n.pt`, `yoloe-11s-seg.pt`,
  `rtdetr-l.pt`, `mobileclip_blt.ts`) ended up in the working tree on the
  first real run. Now gitignored, but `scripts/00_create_env.sh` should
  prefetch weights into a cache dir to avoid polluting the repo root.

## 20. Standalone showcase problems

- **P20.1** No standalone repo created. The example lives under
  `examples/Object_Detection/`. This is fine for now — the structure is
  copy-pasteable for a later standalone repo if the showcase becomes a
  paper/demo.

---

## Summary of fixes applied in this round

The next round of code changes addresses the highest-impact problems above:

- **P1.1, P1.2, P8.1, P10.1** — Add explicit `fusion_mode` field with
  `"selector"` default. In selector mode the decoder copies the box from
  the candidate node and does NOT apply `box_reg`.
- **P8.3, P7.1, P7.2** — Selector head scores **all** candidate nodes
  (proposals + clusters + consensus). Best-source targets attached to all
  candidate-eligible nodes.
- **P9.1, P9.2** — Add a `validation_sweep` policy: evaluate at thresholds
  {0.0, 0.005, …, 0.5} on validation, pick the threshold with the best F1,
  freeze it, then run on test.
- **P15.1, P15.2, P15.3** — Report two new baselines:
  `best_proposal_per_cluster` and `oracle_best_iou_per_gt`.
- **P12.1** — Add a small COCO↔VOC class-name mapping fallback in the
  evaluator, so a real YOLO/RT-DETR `"car"` is matched to a VOC `"car"`.
- **P16.1** — Extend training log with positive_obj_mean, negative_obj_mean,
  positive_count, selected_count.

The fixes below are scoped to **interpretability and lower-bound behaviour**,
not "make TGraphX win". If TGraphX still does not beat WBF after these
fixes, the report says so honestly.
