# LEARNED_BOX_FUSION_FORMULATION — formulation change for TGraphX OD

**Author:** Claude Opus (principal-scientist / implementation lead)
**Date:** 2026-05-13
**Predecessor reports:** `OPUS_FAILURE_AUDIT.md`, `LEARNABILITY_AUDIT.md`,
`OPUS_FINAL_VERDICT_2026_05_13.md`.

---

## 1. Why source routing cannot beat WBF on this data

Empirically (`runs/real_voc_car_v2/baseline_ap_audit.json`):

| Method                                          | Val AP50 | Test AP50 |
|-------------------------------------------------|---------:|----------:|
| `fusion::wbf`                                   | 0.8989   | **0.8834** |
| `fusion::nms`                                   | **0.9032** | 0.8687 |
| `oracle::per_cluster_best_available_source`     | 0.8748   | **0.8452** |
| `oracle::rtdetr_anchor + oracle_override`       | 0.8539   | 0.8452 |

A *source-routing* policy emits, per cluster, one of the existing source
boxes. The set of admissible outputs is therefore
`{box from rt_detr, box from retinanet, box from yolo, union box, wbf box,
nms box, soft_nms box, best_proposal box}`. The best-possible such
policy has AP50 = 0.8452 on test — 3.8 AP below WBF, 2.4 below NMS, 2.1
below rt_detr. Even an *omniscient* router loses; no model class can win.

## 2. Why WBF beats every per-source oracle

WBF emits, per cluster, a **synthesized** box that is a score-weighted
average of overlapping proposals across all detectors. The output box is
not required to be any single proposal. WBF therefore exits the
source-router family by construction.

Two mechanisms make WBF higher-AP on this car-only data:

1. **Box averaging is variance-reduction.** When 2–3 detectors agree on
   a region with slightly different boxes, the score-weighted average
   has lower IoU error than any individual proposal.
2. **WBF preserves per-detector evidence in scoring.** The synthesized
   box keeps the WBF-style score, which is a function of multiple
   detectors' confidences — empirically better-calibrated than picking
   one detector's score.

A source-routing model has access to the WBF box (as a `cluster` /
`wbf` node) but, by design, picks one source rather than synthesizing.
This is the structural cap.

## 3. Why TGraphX must learn a new box, not only select one

A learned model can do strictly more than WBF if and only if it can
**emit a box that depends on cluster context but is not constrained to
the convex hull of (existing source boxes, weighted by detector
confidence)**. Concretely:

- It can **regress a small correction** to WBF using cluster-level
  signal (detector agreement entropy, box variance, class agreement)
  that WBF ignores by formula.
- It can **predict a score that is better-calibrated than WBF's
  formula** — empirically the FP/image is what hurts AP at low-confidence
  clusters.
- It can **drop the cluster entirely** (low confidence) where WBF
  produces a low-but-nonzero score for a false-positive cluster.

The hypothesis is: *the IoU residual `WBF_box → GT_box` is non-trivially
predictable from the cluster's graph features*. If true, a learned
regression head over WBF beats WBF. The Part-2 audit measures this
hypothesis BEFORE training. If false, the project should stop.

## 4. Relation to the original TGraphX idea

TGraphX has always been a graph-based fusion architecture. The original
claim is "a graph over detectors + candidate sources + crops yields a
better detection than any single component." This claim is *not*
contingent on output type. Source routing is one decoder for that
graph; learned box fusion is another. The graph construction, crop
encoder, edge-conditioned message passing, and source nodes are reused
identically. Only the *decoder head* changes:

- **Source-router decoder:** per-cluster softmax over source slots, output box = chosen source's box.
- **Learned-fusion decoder:** per-cluster regression head that predicts (refined box, TP@0.5 score, TP@0.75 score, optional class), conditioned on cluster context, source-slot embeddings, and pairwise geometric features.

## 5. What the new model predicts

For each cluster `c`:

- `final_box_xyxy[c]`  ∈ ℝ⁴
- `tp50_logit[c]`      → sigmoid → predicted P(IoU ≥ 0.5 ∧ class correct)
- `tp75_logit[c]`      → sigmoid → predicted P(IoU ≥ 0.75 ∧ class correct)
- `expected_iou[c]`    → sigmoid → predicted IoU(final_box, GT)
- *(optional)* `class_logits[c]` ∈ ℝ^K for multi-class regimes
- *(optional)* `source_weights[c]` ∈ Δ^S (softmax over available source slots)
- *(optional)* `uncertainty[c]` — log-variance for the box regression

The final box is computed via one of three heads (Part 3 ablation):

- **A. Residual:** `final_box = anchor_box + Δ` with anchor = WBF.
- **B. Weighted fusion:** `final_box = Σᵢ softmax(weights_i)·box_i`.
- **C. Hybrid:** `final_box = (Σᵢ wᵢ·box_i) + Δ`.

For AP scoring at inference, the score is `sigmoid(tp50_logit)`
(or a learned linear combination of tp50 and base scores; see Part 7
score-mode ablation).

## 6. What stays from the current framework

Identical to v9 / anchor-router:

- Step 03 graph construction (proposals, clusters, consensus, NMS / WBF / BestProposal nodes).
- `TGraphXSourceRouterV3` encoder (crop CNN + metadata MLP + edge-conditioned message passing + slot aggregator).
- `SOURCE_SLOTS` indexing, slot mask handling, `_attach_slot_metadata`.
- Pairwise feature extractor (`src/od_graph_fusion/pairwise_features.py`).
- Source priors (`src/od_graph_fusion/source_priors.py`).
- Hard-case mining (`src/od_graph_fusion/hard_cases.py`) — used to oversample clusters where WBF needs the largest correction.
- Paired bootstrap (`src/od_graph_fusion/paired_bootstrap.py`).
- Step 06 verdict logic (verdict-vs-strongest-baseline + paired bootstrap is already implemented).

## 7. What changes

- New model class: `TGraphXLearnedBoxFusion` in
  `src/od_graph_fusion/learned_box_fusion.py`.
- New head over the shared encoder: residual / weighted / hybrid box.
- New loss: SmoothL1 + GIoU + BCE(TP50) + BCE(TP75) + IoU regression +
  score ranking + delta regularization (with ablation weights).
- New runner: `src/od_graph_fusion/multi_seed_learned_fusion.py`.
- New config: `configs/real_voc2007_car_learned_box_fusion.yaml`.
- New audit script (run BEFORE training):
  `scripts/box_fusion_oracle_audit.py`.

Source-router code stays in place as a baseline / ablation. Step 05
already reports the per-detector + classical baselines, so the
comparison table comes "for free" once `multi_seed_learned_fusion.py`
writes a compatible `metrics_seedN.json`.

## 8. Decision rule for proceeding

The Part-2 audit must show that **at least one** of the following
oracles beats WBF by ≥ +0.005 AP on val:

- Oracle box = matched-GT box (per-cluster). Trivial upper bound; if
  this is ≪ WBF, the cluster set has a *recall* problem and no learned
  box-fusion can save it.
- Oracle convex combination over source boxes (per-cluster best `w` on
  Δ^S that maximizes IoU with matched GT, scored at max source score).
- Oracle small residual: WBF + Δ where ‖Δ‖∞ ≤ 0.1·diag(WBF_box) and Δ
  is chosen to maximize IoU(WBF+Δ, GT). Measures whether *local* box
  refinement helps.

If none of these beat WBF, **the formulation change is also dead** and
the verdict becomes `BOX_FUSION_ORACLE_NO_HEADROOM`. No GPU training
will be launched in that case.

If any one beats WBF, the corresponding output mode (weighted / residual
/ hybrid) is the head to train first.

End of formulation note.
