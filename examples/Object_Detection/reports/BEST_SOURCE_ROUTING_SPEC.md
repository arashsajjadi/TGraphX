# TGraphX Object Detection: Best-Source Routing Specification

## 1. Problem statement

TGraphX is a **best-source router** for object detection. Given multiple detector
predictions for the same object instance (one from each of several detectors), it
learns which source is most likely to be the best candidate and selects it. It is
**not** a detector from scratch — it never generates novel box coordinates from raw
image content alone.

## 2. Concepts

### 2.1 Candidate cluster C

A candidate cluster is a set of detection proposals that are spatially collocated
(IoU-grouped) across all available detectors. For each image, we form clusters
by greedily grouping proposals whose pairwise IoU exceeds a threshold (0.5).

### 2.2 Candidate source S

For cluster C, the set of available sources is:

```
S = { YOLO26/fallback-YOLO, DETR/RT-DETR, Open-Vocab-YOLO, RetinaNet, Union, WBF }
```

Each source s ∈ S produces:
- box b_s ∈ ℝ⁴ (xyxy)
- raw score c_s ∈ [0, 1]
- predicted class y_s
- canonical class ŷ_s (after VOC/COCO label mapping)
- crop tensor x_s ∈ ℝ^{3×H×W}
- detector identity d_s ∈ {0, …, num_detectors-1}

A source may be **absent** for a given cluster (the detector fired nowhere nearby).
Absent sources carry a boolean mask flag and are never eligible for selection.

### 2.3 Utility function

For a source s matched to ground-truth box b_GT with class y_GT:

```
utility(s) = IoU(b_s, b_GT) × is_correct_class(ŷ_s, y_GT) × (1 − duplicate_penalty)
```

- In class-agnostic mode: `is_correct_class = 1` always.
- `duplicate_penalty` = 1 if another source already matched b_GT with higher IoU.
- Clusters with no GT match: utility = 0 for all sources.

### 2.4 Best-source label

```
best_source(C) = argmax_{s ∈ available(C)} utility(s)
```

Used **only** during training to supervise the source-routing head.

### 2.5 Source-selection accuracy

```
source_acc = mean over clusters [selected_source(C) == best_source(C)]
```

Measured on test data using the trained model.

### 2.6 Oracle definitions

| Oracle type | Definition | Deployable? |
|---|---|---|
| **localization_oracle** | Picks the proposal with highest IoU to GT; ignores class label | No (uses GT) |
| **class_aware_candidate_oracle** | Picks the correctly-labelled proposal with highest IoU | No (uses GT) |
| **source_routing_oracle** | Picks argmax utility(s); the deployment-target upper bound | No (uses GT) |
| **TGraphX router** | Predicts source from graph evidence; no GT at inference | **Yes** |

### 2.7 Oracle-gap recovery

```
gap_recovery(T, B, O) = [metric(T) − metric(B)] / max(ε, metric(O) − metric(B))
```

- T = TGraphX, B = baseline, O = oracle.
- = 0 if TGraphX equals baseline; = 1 if TGraphX equals oracle; < 0 if TGraphX is worse.

## 3. Why GT is training-only

The utility function and best-source label require knowing which box overlaps GT best.
This information is unavailable at inference (no GT exists). TGraphX must learn to
proxy the oracle from graph evidence: crop textures, detector scores, spatial relations,
and cross-detector agreement.

## 4. Faithfulness to old TGraphX

| Component | Old TGraphX | New TGraphX |
|---|---|---|
| Candidate sources | YOLO, RetinaNet, Union | YOLO26, DETR, Open-Vocab YOLO, RetinaNet, Union, WBF |
| Main task | Choose/refine best candidate | Learn best-source routing |
| Node feature | Crop tensor | Crop tensor + detector metadata |
| Edge feature | Source relation | IoU, agreement, detector pair, class agreement |
| Output | Better candidate choice | Predicted best source + optional score calibration |
| Minimum success | Better IoU than detector nodes | Source-selection accuracy above trivial baselines |

## 5. Minimum success criterion

TGraphX is successful as a router if:

1. `source_selection_accuracy(TGraphX) > source_selection_accuracy(highest_confidence)`.
2. `oracle_gap_recovery > 0` on validation (TGraphX partially closes the gap to oracle).
3. TGraphX shows non-trivial source switching — it does not always pick the same detector.
4. `AP50(TGraphX) >= AP50(best_proposal)` on test (no regression vs trivial lower bound).
