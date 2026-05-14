# Opus Reformulation — Delivery Summary

This document accompanies the audit (`OPUS_FAILURE_AUDIT.md`) and the
implementation that followed it. The implementation is complete and
test-passing; the empirical verdict (Part 12) cannot be filled in until
you run the 10-seed GPU experiment.

## 1. Executive verdict

**Implementation: COMPLETE — 24 new tests pass, full Object_Detection suite
210/210 green.**

**Empirical verdict: PENDING — awaits Part 10.4 (10-seed real VOC car run).**
Do NOT claim a win. Do NOT run VOC500. Do NOT touch VOC200 multi-class
until Part 10.3 sanity overfit shows union/yolo learnable and Part 10.4
clears the paired-bootstrap bar.

## 2. What changed architecturally

| Old (v9 free router)                              | New (anchor router)                                            | Why                                                                                          |
|---------------------------------------------------|---------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| Free softmax over 10 source slots                 | Anchor-preserving keep/override with delta heads               | Free CE collapsed to majority sources; oracle headroom is small (+0.028 AP) — guarded gain is the only viable formulation |
| `routing_prob`/`base_score` score modes           | `p_tp50`, `p_tp50*base`, `base` — TP50 head + temperature scaling | AP depends on ranking; routing prob doesn't predict TP                                       |
| One source-CE loss + KL                           | L_delta + L_pairwise + L_keep_override + L_specialist + L_tp50 + false-override penalty (7×) | CE doesn't tell the model "override only when alt is provably better" |
| Slot embeddings only                              | + source-id emb, source-pair emb, val-only priors, pairwise features | Current crop embedding can't distinguish "union wins" from "rt_detr wins" |
| No specialist heads                               | Union / yolo_modern / rt_detr / retinanet specialist heads (BCE) | Suppressed sources need privileged features |
| Hard-case awareness = balanced CE weights         | Train-only hard-case sampler with 40/20/20/20 mix             | Reweighting did nothing; sampling changes the decision boundary |
| `_build_util_and_labels(class_agnostic=True)` hardcoded in `training.py` | `utility_mode` and `class_agnostic` are first-class kwargs   | Multi-class needs class-aware AP for validation selection |
| Step 05 selected score mode on `class_agnostic=True` for multi-class | `selection_class_agnostic = not is_multiclass`             | Selection metric must match the operational metric |
| Step 06 verdict used fixed AP thresholds (`>0.85` etc.) | Verdict compares vs strongest VAL-selected baseline + paired bootstrap | Absolute thresholds don't mean "win" — they mean "above a magic number" |
| `train_fusion_model` silently used `meta.detector_names` (could be empty) | `detector_names` is a required kwarg; empty raises `RuntimeError` | Empty detector_names was the v8/v9 source-of-truth bug |
| Override router (`NMSOverrideRouter`) — 1% override rate, 0% success, IoU-only utility | Replaced (kept on disk for back-compat) by `AnchorRouter` over V3 encoder | The override head learned "never override"; the new path uses delta heads with an explicit false-override penalty |

## 3. New / modified files

### New source modules
- `src/od_graph_fusion/anchor_router.py` — `AnchorRouter`, `AnchorRouterConfig`, `SPECIALIST_SLOTS`, `calibrate_temperature`
- `src/od_graph_fusion/anchor_training.py` — `anchor_router_loss`, `build_anchor_targets`, `specialist_targets`, `AnchorLossWeights`
- `src/od_graph_fusion/pairwise_features.py` — generic pairwise + union/yolo specialist features
- `src/od_graph_fusion/source_priors.py` — `compute_priors`, `select_anchor_on_validation`, `PriorTable`
- `src/od_graph_fusion/hard_cases.py` — `classify_hard_case`, `HardCaseSampler`
- `src/od_graph_fusion/paired_bootstrap.py` — per-image AP, paired bootstrap, verdict labels
- `src/od_graph_fusion/multi_seed_anchor.py` — end-to-end multi-seed runner

### New scripts
- `scripts/audit_hard_cases.py`
- `scripts/sanity_overfit_anchor.py`
- `scripts/run_anchor_multi_seed.py`

### Modified scripts
- `scripts/04_train_tgraphx_fusion.py` — passes `detector_names`, `utility_mode`, `class_agnostic`, `strict_source_router`
- `scripts/05_evaluate.py` — class-aware AP for multi-class selection; baseline AP for NMS/WBF/raw/best_proposal on same test split; paired bootstrap vs each baseline
- `scripts/06_make_report.py` — verdict vs strongest baseline + paired bootstrap; no fixed AP thresholds

### Modified source
- `src/od_graph_fusion/training.py` — `detector_names` is now required for V3; legacy objectness branch disabled under `strict_source_router=True`; `utility_mode` + `class_agnostic` kwargs

### New tests (24 added, all green)
- `tests/test_anchor_router.py` — 4
- `tests/test_anchor_training.py` — 3
- `tests/test_hard_cases.py` — 3
- `tests/test_paired_bootstrap.py` — 3
- `tests/test_priors_and_features.py` — 3
- `tests/test_multiclass_score_mode_selected_by_class_aware_ap.py` — 2
- `tests/test_step06_verdict_compares_to_baseline_not_absolute_threshold.py` — 3
- `tests/test_training_v3_raises_if_detector_names_missing.py` — 3

### New config
- `configs/real_voc2007_car_anchor_router.yaml`

## 4. How to run the experiment sequence

### 1. Unit tests
```bash
cd examples/Object_Detection
python -m pytest tests -q
```
Expected: **210 passed**.

### 2. Hard-case audit (no training)
```bash
python scripts/audit_hard_cases.py \
    --config configs/real_voc2007_car_anchor_router.yaml --device auto
```
Writes `runs/<run_name>_audit/hard_case_audit.json`. Inspect to verify
union/yolo cases exist on train.

### 3. Sanity overfit (≈30 epochs on the 200-image train slice)
```bash
python scripts/sanity_overfit_anchor.py \
    --config configs/real_voc2007_car_anchor_router.yaml --device auto
```
Pass criteria (printed at end):
- `false_override_rate <= 0.20`
- `deployed_source_acc >= 0.55`

If those fail, do NOT proceed to step 4.

### 4. Real VOC car-only, 10 seeds, GPU
```bash
python scripts/run_anchor_multi_seed.py \
    --config configs/real_voc2007_car_anchor_router.yaml --device auto
python scripts/06_make_report.py --run-dir runs/real_voc_car_anchor_anchor
```
Time budget: car-only 200-image × 10 seeds × 30 epochs on a 5080 is
typically 30–90 minutes total.

### 5. (Only if step 4 produces `REAL_VOC_CAR_WIN` or `SAFE_TIE`)
VOC200 multi-class with class-aware val selection:
```bash
# Edit a copy of the config: num_classes=20, class_filter=null, class_agnostic=false
python scripts/run_anchor_multi_seed.py \
    --config configs/real_voc200_anchor_router.yaml --device auto
python scripts/06_make_report.py --run-dir runs/real_voc200_anchor_anchor
```

## 5. Anchor selection (Part 10 §1)

| Dataset    | Anchor mode                            | Default anchor source on this data    |
|------------|----------------------------------------|---------------------------------------|
| VOC car    | `validation_best_global_source`        | rt_detr (val AP50 highest)            |
| VOC200     | `validation_best_global_source`        | NMS (on-disk: VOC200 NMS=0.8837 mean) |

`validation_best_class_size_source` is supported by `select_anchor_per_class`
in `source_priors.py`. The car-only experiment has one class, so per-class
≡ global. For multi-class, supply per-class val AP and call
`select_anchor_per_class`.

## 6. Hard-case audit (Part 12 §4 — to be filled)

Run `scripts/audit_hard_cases.py` and paste the JSON output here. Compare
counts before / after the new architecture once Step 4 completes; the
"after" column for `union recall`, `yolo recall`, etc. comes from
`source_routing_metrics.json` per seed.

## 7. Source confusion matrix (Part 12 §5 — to be filled)

Per seed in `seed_NN/results.json` the new runner records:
- `routing.deployed_source_acc`
- `routing.successful_overrides` / `routing.failed_overrides`
- `routing.false_override_rate`
- `routing.override_rate`

Aggregate across seeds to fill the confusion matrix.

## 8. Override metrics (Part 12 §7 — to be filled)

Read from `summary.json` / `metrics_seedN.json`. Step 06 emits the
top-level summary in the report.

## 9. AP table (Part 12 §8 — to be filled)

Step 06 emits this table automatically from the per-seed
`baseline_methods` field, with the paired bootstrap as a separate line.

## 10. Score calibration (Part 12 §9 — to be filled)

Per seed `metrics_seedN.json` records `val_score_modes`, including AP under
each (threshold, score_mode) pair. The selected mode is saved as
`selected_score_mode`.

## 11. Failure examples (Part 12 §10 — to be filled)

To produce the per-cluster trace table, post-process `source_routing_metrics.json`
+ the in-memory `chosen, chose_anchor` outputs from `AnchorRouter.decide`.
A small CLI helper is not yet provided; if the verdict comes out clean,
add one.

## 12. Failure-mode bailouts (Part 11)

The verdict labels emitted by Step 06 are exhaustive:
- `REAL_VOC_CAR_WIN`
- `REAL_VOC_CAR_SAFE_TIE`
- `REAL_VOC_CAR_NOT_YET_WIN`
- `REAL_VOC_MULTI_CLASS_WIN` / `_SAFE_TIE` / `_NOT_YET_WIN`
- `STILL_NOT_READY_FOR_REAL_CLAIM` (fallback when Step 05's paired bootstrap is missing)

Diagnostic labels from the user spec (`FALSE_OVERRIDE_BOTTLENECK`,
`UNION_SPECIALIST_FAILED`, `YOLO_SPECIALIST_FAILED`, `ANCHOR_ROUTER_FAILED`)
are not yet auto-derived. They map to inspecting the per-seed
`source_routing_metrics.json`:
- `false_override_rate > 0.3` AND `mean_iou_gain_per_override < 0` → `FALSE_OVERRIDE_BOTTLENECK`
- `union recall` (from hard_case_counts.A vs routing source-acc) `== 0` after training → `UNION_SPECIALIST_FAILED`
- `yolo recall` (from hard_case_counts.B vs routing source-acc) `== 0` → `YOLO_SPECIALIST_FAILED`
- `n_overrides > 0` but `deployed_source_acc < 0.5` → `ANCHOR_ROUTER_FAILED`

## 13. Strict scientific conclusion (Part 12 §11)

Until Part 10.4 runs and Step 06 produces a verdict against a paired
bootstrap on the same 200 images, no scientific conclusion is available
beyond what the audit already states:

- The previous formulation does not beat the strongest baseline (NMS),
  which is on-disk-confirmed by 60 overrides / 5660 clusters with negative
  IoU gain.
- Oracle headroom on car is +0.028 AP. The new formulation is designed
  to capture a fraction of that headroom under a strict false-override
  penalty; whether the captured fraction beats NMS is an empirical
  question and will not be answered by inspection.
- A win, if it materializes, will be small. A tie will be honest. A loss
  will require either (a) more data, (b) a different anchor (e.g. WBF or
  best-proposal), or (c) acknowledgment that the per-cluster IoU oracle
  ≈ 0.95 is not reachable in AP50 because raw detectors already saturate
  most of the easy cases.

End of delivery summary.
