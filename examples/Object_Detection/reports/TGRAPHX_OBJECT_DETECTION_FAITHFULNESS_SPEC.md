# Faithfulness spec — TGraphX as a graph-based detection selector/refiner

This document defines the **correct implementation target** for the rebuild,
derived from the original TGraphX paper-era logic and from a strict reading of
what the architecture can and cannot do.

## Core principle

**TGraphX is not a detector.** TGraphX is a tensor-native **graph reasoning
layer** that takes existing detector proposals as input and produces a more
accurate *selection* — and optionally a *refinement* — of a final detection
per candidate object.

The pipeline is therefore:

```
images
   │
   ▼
[ YOLO26 ] [ RT-DETR ] [ open-vocab YOLO ] [ RetinaNet ]    ← detectors
   │            │              │                  │
   └────────────┴──────────────┴──────────────────┘
                      │  proposal boxes + scores + classes
                      ▼
              candidate clusters
            (IoU-grouped proposals)
                      │
                      ▼
       ┌──────────────┴──────────────┐
       │ For each cluster, create:   │
       │   - one node per proposal   │
       │   - one Union node          │
       │   - one WBF node            │
       │   - one cluster-summary node│
       └──────────────┬──────────────┘
                      ▼
              TGraphX graph fusion
              (selector / ranker)
                      │
                      ▼
       Final detection = candidate node with
       highest predicted selection score
       (box copied from that node verbatim)
```

The TGraphX model does **not** regress new boxes by default. It learns which
existing candidate node is the best per cluster.

## Implementation rules

1. **TGraphX is a selector, not a detector from scratch.** All output boxes
   must come from existing candidate nodes (proposal / union / WBF).
2. **Each candidate cluster becomes one or more graph nodes.** Nodes within a
   cluster share supervised signal (objectness, best-source).
3. **Detector outputs become proposal nodes.** One node per detector
   prediction, even when several detectors fire on the same object.
4. **Union / consensus / WBF boxes become candidate nodes** that the model can
   choose between alongside individual proposals.
5. **Node features include spatial crop tensors**, ideally `[3, 128, 128]` in
   faithful mode. They are extracted from the original image at the node's
   candidate box and never silently flattened.
6. **Edge features encode detector identity, IoU, geometry, confidence, and
   semantic agreement.** Edges are typed: detector_to_union,
   detector_to_wbf, detector_agreement, spatial_overlap, class_agreement,
   confidence_support, same_detector_suppression, context_support.
7. **TGraphX is lower-bounded by best_proposal / WBF / cluster_confidence.**
   If the selector cannot help, it must at least pick one of these candidates.
8. **Box regression is optional and disabled until the selector works.** This
   was the silent failure of the previous run: the refiner head pushed boxes
   far from the well-localized candidates, collapsing AP.
9. **The old YOLO + RetinaNet + Union setup must be reproducible** as a
   compatibility mode (`old_compatible`) without changing the core logic.
10. **Modern extension** = swap in or add YOLO26 / DETR / RT-DETR / YOLO
    open-vocab proposal nodes, plus WBF/consensus, without modifying the
    selector head.

## Old paper logic ⟶ required modern implementation

| Component | Old paper logic | Required modern implementation |
| --- | --- | --- |
| Detector nodes | YOLO + RetinaNet | YOLO26 + DETR/RT-DETR + YOLO open-vocab + RetinaNet (with graceful fallback) |
| Union/consensus | YOLO/Retina union box | Union, WBF and consensus boxes as separate candidate nodes |
| Node feature | `[3, 128, 128]` tensor crop | tensor crop, default `[3, 128, 128]` in faithful mode; configurable smaller in FAST_SMOKE |
| Vector metadata | detector id, score, geometry | + class id, score rank, score percentile, detector family, detector prior AP |
| Edge feature | detector pair / source relation | IoU, GIoU, center distance, area ratio, score difference, same-class flag, same-detector flag, edge-type one-hot |
| Prediction | choose / refine best candidate | selector → ranker → refiner (refiner disabled by default) |
| Lower bound | detector/union choice | best proposal, WBF, cluster confidence, oracle |
| Threshold | hand-picked | validation sweep over {0.0, 0.005, 0.01, …, 0.5}, frozen before test |
| Final box | from a selected node | **always** copied from the selected node verbatim (no regression by default) |

## What the rebuild must *not* claim

- TGraphX has not been trained as a detector. It cannot beat YOLO/DETR/RetinaNet
  at producing arbitrary boxes from raw pixels.
- TGraphX can only be as good as the best candidate node available — its job
  is to *pick* well, not to *invent* boxes.
- A FAST_MODE smoke run on 8 training graphs cannot serve as evidence for or
  against the architecture.

## What "success" looks like

A run is considered **scientifically interpretable** when:

1. **TGraphX ≥ trivial selector lower bound.** In selector mode with a frozen
   validation threshold, the final TGraphX AP50 must be **at least as high
   as the "pick the highest-confidence proposal in each cluster" baseline**.
   If it is below, the model has not learned to select, and the configuration
   should be flagged as broken.
2. **Test-set thresholds are frozen from validation.** No threshold is tuned
   on test.
3. **Per-method runtime, model identifiers, and detector availability are
   reported honestly.**

A run is considered a **success showcase** only if at least one of:

- TGraphX selector ≥ WBF on the same validation+test data, or
- TGraphX selector ≥ best individual detector on the same data,

with seed-level evidence in `runs/<name>/method_results.json`.

A run that falls below the trivial selector lower bound is reported as
**not yet evidence of superiority** and is *not* showcased.
