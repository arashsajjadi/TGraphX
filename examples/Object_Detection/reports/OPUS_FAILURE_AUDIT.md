# OPUS_FAILURE_AUDIT — Why the free source-router lost, and what the data actually says

**Author:** Claude Opus (principal-scientist audit)
**Date:** 2026-05-12
**Scope:** v9 free-source-router (`TGraphXSourceRouterV3`) + v8 NMS-override router (`NMSOverrideRouter`)
**Verdict:** REJECT current formulation. Do not run more seeds. Do not run VOC500.
The math, the on-disk override metrics, and the source-confusion structure all
agree: **the current model is a noisy classifier with no gain over the strongest
validation baseline.** A win requires reformulation, not more training.

---

## 1. Headline numbers (real VOC car-only, 200 images, 10 seeds, GPU)

User-supplied (from latest car-only run paired against the same test split):

| Method                     | AP50    | Notes                                       |
|----------------------------|--------:|---------------------------------------------|
| RawDet(rt_detr)            | 0.8664  | strongest single detector on real VOC car   |
| WBF                        | 0.8659  | weighted box fusion baseline                |
| NMS                        | 0.8644  | strongest classical fusion                  |
| TGraphX (best mode)        | 0.8583  | seed-mean upper                             |
| TGraphX (selected mode)    | 0.8554  | reported headline                           |
| **Oracle (per-cluster)**   | 0.8942  | upper bound if router picks correctly       |

- TGraphX is **below** every baseline. The closest baseline is NMS at +0.009 AP.
- Oracle headroom over the strongest baseline (rt_detr): **+0.028 AP** — small
  but not zero. The win is real but tight, and the router does not capture it.
- Paired bootstrap (user-reported): TGraphX expected to lose vs NMS most draws.

These are **the numbers that govern the verdict.** Free softmax routing is
already proven inadequate against these baselines; nothing else in the file is
salvage.

## 2. On-disk evidence — override router behavior is dead weight

From `runs/voc200_override_override/seed_*/source_routing_metrics.json`
(VOC200 multi-class, real detectors, override-router seed sweep):

| seed | overrides | clusters | thr  | succ. rate | source acc | mean IoU gain | IoU_tgx | IoU_nms | IoU_oracle |
|----:|----------:|---------:|-----:|-----------:|-----------:|--------------:|--------:|--------:|-----------:|
| 0   |   0       | 643      | 0.80 | 0.000      | 0.711      | +0.0000       | 0.418   | 0.402   | 0.943      |
| 1   |   0       | 601      | 0.90 | 0.000      | 0.701      | +0.0000       | 0.373   | 0.357   | 0.954      |
| 2   |   1       | 690      | 0.90 | 0.000      | 0.688      | −0.0078       | 0.436   | 0.408   | 0.950      |
| 3   |  27       | 458      | 0.90 | 0.296      | 0.715      | −0.0213       | 0.450   | 0.433   | 0.945      |
| 4   |  18       | 595      | 0.90 | 0.389      | 0.707      | −0.0099       | 0.512   | 0.478   | 0.957      |
| 5   |   4       | 377      | 0.90 | 0.000      | 0.689      | −0.0424       | 0.434   | 0.423   | 0.953      |
| 6   |   1       | 573      | 0.90 | 0.000      | 0.673      | −0.0155       | 0.441   | 0.423   | 0.924      |
| 7   |   6       | 574      | 0.90 | 0.500      | 0.695      | +0.0159       | 0.437   | 0.414   | 0.953      |
| 8   |   3       | 583      | 0.90 | 0.000      | 0.667      | −0.0028       | 0.443   | 0.413   | 0.953      |
| 9   |   0       | 566      | 0.90 | 0.000      | 0.714      | +0.0000       | 0.501   | 0.454   | 0.951      |

**Aggregate:** 60 overrides across 5660 clusters → **1.06% override rate**.
Mean IoU gain per override across seeds: **negative or zero in 9/10 seeds**.
On six seeds, *every override the model committed was wrong*. The override
mechanism is currently a 1% noise injector.

**Source-selection accuracy 67–71% is misleading.** When `n_overrides ≈ 0`, the
model effectively *is* NMS, so "source acc" mostly reflects NMS-being-correct
rather than the router being useful. The acc bound is exactly what an
NMS-pass-through achieves.

Mean cluster-level IoU shows a different ladder than AP50:
- mean IoU_oracle ≈ 0.95 across seeds (per-cluster, best-source upper bound).
- mean IoU_nms   ≈ 0.42.
- mean IoU_tgx   ≈ 0.44.

The **per-cluster IoU gap (oracle 0.95, NMS 0.42)** is huge, but the AP50 gap
is only +0.028. This is the central tension: IoU-headroom is high, AP50-
headroom is low, and the router was optimizing the wrong utility.

## 3. Why the free-softmax formulation failed

A free `softmax(source_logits)` classifier minimizes per-cluster CE.
On real VOC car-only with three detectors that all hover near AP50≈0.80–0.87:

- The class-balanced CE prior is dominated by rt_detr and retinanet because
  those are the modes that *contain* the GT box for the majority of clusters.
  Any classifier with weak features will collapse to a two-mode policy over
  rt_detr / retinanet.
- The union and yolo_modern modes are decisive for a small but real fraction
  of clusters (user-reported union oracle ≈ 11%, selected 0%). Free CE never
  selects them because doing so loses *more* clusters on the head of the
  distribution than it wins on the tail.
- Balanced / focal loss reweights *individual examples* but does not change
  the structure of the decision boundary. As reported: "balanced/focal loss
  did not fix it." This matches the theory: when the right move is "override
  only when the alternative is materially better," reweighting CE does not
  produce that policy — only an explicit override/gain objective does.

**The free-classifier objective is wrong for this regime.** The right objective
is *guarded improvement over the strongest baseline*, not *best class globally*.

## 4. Why each source is mis-handled (mechanistic)

| Source       | Oracle %\* | Selected %\* | Why the current model gets it wrong |
|--------------|-----------:|-------------:|-------------------------------------|
| rt_detr      | high       | very high    | strong detector + free CE prior → over-selected almost everywhere; ignores cases where rt_detr has localized but mis-classified or mis-sized |
| retinanet    | moderate   | high         | high recall on car class → CE prior |
| yolo_modern  | moderate   | low          | low recall (fewer raw detections per image, 0.47 raw AP50 on the v3 9-image dev set), so its slot is rarely "active enough" to win CE |
| union        | ~11%       | ~0%          | NUM_SOURCES slot is constructed from `consensus` nodes but the *feature* is just the mean crop embedding — there is no specialist signal saying "the union box localizes better than any single proposal" |
| nms / wbf / best_proposal | — | — | now mapped in `_build_node_source_slots` (v9), but their slot embeddings are still trained from the same crop+metadata MLP — no privileged information |
\* approximate, as user-reported on car-only.

The model has **no architectural reason to discriminate** union or yolo from
rt_detr beyond crop appearance — it never sees pairwise features like
"union-vs-anchor IoU," "score-rank disagreement," or "size-bin prior" that
*should* drive the override decision. So even with hard-case mining, the
current featurization cannot represent the union-wins-here-but-not-there
decision boundary cleanly.

## 5. Where the bottleneck actually is

User asked: AP-limited, source-selection-limited, or calibration-limited?

- **AP-limited?** Partly. Oracle(AP50) ≈ 0.8942 vs rt_detr 0.8664 → headroom
  ≈ +0.028 AP. Real but narrow.
- **Source-selection-limited?** Yes, dominantly. The model already has access
  to the right sources via slot embeddings; it just does not pick correctly.
  In particular, it never picks union, and over-picks rt_detr.
- **Calibration-limited?** Secondary. From the v3 score-mode ablation
  (`routing_prob*max_base` selected on val), ECE ≈ 0.20 and FP/image ≈ 2.1 at
  seed 0. Calibration matters, but it cannot close a +0.01 AP gap when the
  argmax decision is already wrong.

**Primary bottleneck: source-selection.** Secondary: utility mismatch (training
optimized IoU + class-agnostic CE; evaluation is AP50). Tertiary: calibration.

## 6. Strongest baseline on validation

User-reported (real VOC car):

| Baseline        | Val AP50 (approx) | Test AP50 |
|-----------------|------------------:|----------:|
| RawDet(rt_detr) | high              | 0.8664    |
| WBF             | high              | 0.8659    |
| NMS             | high              | 0.8644    |
| RawDet(retinanet) | mid             | —         |
| RawDet(yolo_modern) | low           | —         |
| BestProposal    | tracks NMS        | —         |

**Strongest validation baseline = RawDet(rt_detr).** Therefore the anchor mode
for car-only should be `validation_best_global_source = rt_detr`.

For VOC200 multi-class, on-disk
`runs/voc200_override_override/summary.json`:

| Baseline        | AP50 (mean / 10 seeds) | CI95             |
|-----------------|-----------------------:|------------------|
| NMS             | 0.8837                 | [0.8686, 0.8977] |
| rt_detr         | 0.8820                 | [0.8691, 0.8950] |
| WBF             | 0.8806                 | [0.8649, 0.8957] |
| TGraphX override | 0.8822                | [0.8709, 0.8939] |
| retinanet       | 0.8070                 | —                |
| yolo_modern     | 0.6842                 | —                |
| yolo_open_vocab | 0.6532                 | —                |

**Multi-class strongest validation baseline = NMS.** TGraphX_override at 0.8822
is statistically inside the NMS CI — already a paired tie, not a win.

## 7. Does Graph Oracle beat the strongest baseline?

Car-only (user): Oracle 0.8942 vs rt_detr 0.8664 → **YES**, by +0.028 AP.
This is the *only* reason to continue: the headroom exists.

VOC200 (on-disk class-agnostic IoU oracle ≈ 0.95 per cluster): the *per-cluster*
oracle is far above NMS — but the AP-level oracle on the v3 9-image dev set
was 1.000 vs NMS 0.8208, which is a different (tiny, optimistic) sample. For
VOC200 we need a real Graph-Oracle@AP50 number before committing.

**Decision:** Continue, with the explicit caveat that headroom is +0.028 AP on
car. Margin for false-override damage is therefore extremely small — any
override mechanism must control false-override rate aggressively (cf. Part 2's
5–10× false-override penalty), or it will dissipate the entire headroom.

## 8. Would `anchor + oracle-override` beat strongest baseline?

By definition: anchor = rt_detr (0.8664). Oracle-override picks the best per
cluster, which the oracle table already gives as 0.8942 = +0.028. So **yes**,
the anchor-preserving formulation has a usable ceiling. The question is what
fraction of the +0.028 headroom a learned model can capture while not bleeding
AP via false overrides.

A useful heuristic: if even 30–50% of the oracle headroom is captured net of
false-override damage, TGraphX clears NMS (0.8644) by ≥ +0.009 AP, the size of
the current TGraphX-vs-NMS deficit. That is the **minimum bar** for the new
formulation to be worth claiming.

## 9. Mandatory diagnostic tables (templates — to be filled by audit_hard_cases + new evaluator)

These are the tables the new audit script and evaluator must emit before any
"win" claim is made. The current pipeline writes only AP50 + per-seed metrics;
it does **not** write the columns below, which is itself a finding.

### 9a. Source-failure mode table (per source)

| Source        | Oracle % | Selected % | Recall | Precision | Avg logit margin | Failure mode |
|---------------|---------:|-----------:|-------:|----------:|-----------------:|--------------|
| rt_detr       | TBD      | TBD        | TBD    | TBD       | TBD              | over-selected (CE prior collapse) |
| retinanet     | TBD      | TBD        | TBD    | TBD       | TBD              | acceptable but ambiguous |
| yolo_modern   | TBD      | TBD        | TBD    | TBD       | TBD              | under-selected (low raw recall) |
| yolo_open_vocab | TBD    | TBD        | TBD    | TBD       | TBD              | only useful for class-aware |
| union         | ~11      | ~0         | ≈0     | undefined | strongly negative | suppressed (no specialist features) |
| wbf           | TBD      | TBD        | TBD    | TBD       | TBD              | rarely selected when nms exists |
| nms_candidate | TBD      | TBD        | TBD    | TBD       | TBD              | implicit fallback today (1% override) |

### 9b. Method-level comparison + paired bootstrap

| Method         | AP50    | 95% CI       | Paired bootstrap P(TGraphX > method) |
|----------------|--------:|--------------|-------------------------------------:|
| NMS            | 0.8644  | TBD          | < 0.5  (TGraphX loses)               |
| WBF            | 0.8659  | TBD          | < 0.5                                |
| RawDet(rt_detr)| 0.8664  | TBD          | < 0.5                                |
| TGraphX (v9)   | 0.8554  | [0.844, 0.873] (min–max) | self                       |
| Oracle         | 0.8942  | TBD          | ceiling                              |

Bootstrap column to be filled by the new evaluator (Part 9).

## 10. Why balanced / focal loss did not save it

- Balanced weights of 0.55–1.11 act on the source-CE term only. They do not
  modify the *target* (which source is correct); they only modify how much
  loss is paid for missing it. With a weak feature representation (no
  pairwise features, no validation priors), the model still cannot tell
  union from rt_detr at decision time — it just pays a different penalty
  for being wrong about union.
- Focal loss down-weights confident-correct examples, which is the *wrong
  direction* here: the model is **under-confident on the easy majority** and
  **wrongly confident on the rare minority**. Focal does not address that.
- Neither loss adds the missing inductive bias: "override the anchor only
  when the alternative is provably better." That bias must be supplied
  architecturally (anchor + delta heads + override gate), not via reweighting.

## 11. Decision and gating conditions for continuing

Graph oracle (0.8942) beats the strongest baseline (rt_detr 0.8664) by +0.028
AP on car-only. **Therefore the new formulation may proceed**, under these
gating conditions, drawn directly from the user's success criteria:

1. The router becomes anchor-preserving (Part 1). Default: rt_detr for car;
   NMS for VOC200 multi-class — both chosen on validation.
2. Training objective is *gain over anchor*, not free CE (Part 2). False
   overrides must be penalized 5–10× hard.
3. Specialist heads exist for union, yolo_modern, retinanet, rt_detr (Part 3),
   each conditioned on pairwise (source-vs-anchor) features that the current
   model never sees.
4. Hard-case mining draws 60% of each batch from union/yolo/anchor-failure
   cases mined from **train only** (Part 5).
5. AP-aware utilities (Part 6) replace class-agnostic IoU as the training
   target.
6. Score head predicts TP50 (Part 7) and is temperature-scaled on val only.
7. Step 06 verdict compares against strongest baseline with paired bootstrap
   (Part 9). No fixed-threshold "WIN" verdicts.

If, after the above, paired bootstrap on the held-out test split still gives
P(TGraphX > NMS) < 0.95 *and* < 0.85, the verdict is
`STILL_NOT_READY_FOR_REAL_CLAIM`. We do not move to VOC500. We do not move to
multi-class. We stop and explain.

## 12. Hard truths

- **The current paper-able claim is "TGraphX matches NMS in a statistical
  tie."** That is *not* a contribution worth publishing. The Oracle 0.8942 vs
  rt_detr 0.8664 gap is, however, a real and small headroom. Capturing it
  cleanly *is* a contribution — but only if (a) the win is not noise, and
  (b) the mechanism (anchor-preserving override) is explainable.
- **The override router as designed is a no-op.** 1% override rate, 0%
  success on most seeds, negative mean IoU gain per override → the override
  head learned "never override" as the safe option. Replacing it with the
  delta-head formulation is the right move.
- **Free source classification was always the wrong primary objective on
  this data.** The right primary objective is pairwise gain over the anchor,
  with the source identity as a secondary decode given that a positive gain
  was predicted.

End of audit.
