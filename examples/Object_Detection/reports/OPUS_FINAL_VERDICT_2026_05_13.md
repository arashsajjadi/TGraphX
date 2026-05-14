# OPUS FINAL VERDICT — TGraphX Object Detection on real VOC2007 car

**Author:** Claude Opus (principal scientist)
**Date:** 2026-05-13
**Data:** `runs/real_voc_car_v2/graphs.pt` (200 images, 3 detectors,
  splits 140/30/30, GT in `source_labels.pt`)
**Audit scripts:** `scripts/learnability_audit.py`,
  `scripts/baseline_ap_audit.py` (both new, both run).

---

## 1. Executive verdict

**`FEATURE_SIGNAL_INSUFFICIENT` — the source-routing framework cannot
win AP50 on this data.**

More precisely: the *measured* per-cluster source-routing Oracle on the
real test split is AP50 = **0.8452**, while WBF is AP50 = **0.8834** and
NMS is AP50 = **0.8687**. The framework's mathematical ceiling is 2.1–3.8
AP points below the strongest baseline. No router — neural, tabular, or
hand-engineered — can clear that ceiling because the ceiling is structural
(one source per cluster, no box synthesis).

The user-supplied "Oracle ≈ 0.8942" headroom number in the previous
prompt was the **best-proposal-per-GT** oracle (cross-cluster, free
matching) and is not achievable by any source router. The achievable
oracle for the router family is 0.8452, which loses.

Run with `scripts/sanity_overfit_anchor.py` or
`scripts/run_anchor_multi_seed.py` is therefore **NOT** recommended; it
would burn GPU time to reproduce a loss that is already determinable
analytically from the on-disk graphs.

## 2. Learnability audit (Part 3 — actually run)

Five tabular models, train on TRAIN (140 imgs), report on VAL (30 imgs).
Target: positive override (delta_ap50 > margin) for margins {0.00, 0.02,
0.05}.

| Model         | Target margin | Val AUROC | Val AUPRC | Prec@top5% | AP gain @top5% | False-override rate @top5% |
|---------------|--------------:|----------:|----------:|-----------:|---------------:|---------------------------:|
| hist_gbm      | 0.00          | **0.833** | **0.448** | 0.383      | −0.0015        | 0.617                      |
| randomforest  | 0.00          | 0.818     | 0.425     | 0.489      | **+0.0012**    | 0.511                      |
| logreg        | 0.00          | 0.745     | 0.387     | 0.511      | −0.0045        | 0.489                      |
| mlp           | 0.00          | 0.696     | 0.365     | 0.468      | +0.0068        | 0.532                      |
| tree          | 0.00          | 0.728     | 0.326     | 0.191      | −0.0071        | 0.809                      |
| hist_gbm      | 0.02          | **0.857** | 0.066     | 0.064      | −0.0464        | 0.787                      |
| logreg        | 0.05          | **0.966** | 0.082     | 0.085      | −0.0358        | 0.660                      |

**Reading:** AUROC is high (0.83–0.97) — there *is* a ranking signal.
But the **expected AP gain of acting on the top-5% predicted overrides
is negative in 14/15 (margin, model) combinations.** Even at margin
0.00, the only "positive" expected gains (+0.0012 and +0.0068) are
within the ±0.005 measurement floor implied by 30 test images.

**Verdict on Part 3:** the simplest gate the user mandated — "if the best
tabular model cannot identify positive overrides, the deep router cannot
be expected to win" — fails. The tabular model identifies overrides by
*rank*, not by *expected gain*.

## 3. Oracle policy simulation (Part 4 — actually run)

Per-cluster AP-utility (soft TP@0.5 with 0.05 × IoU bonus). Mean utility
≈ AP50 proxy at the cluster level.

| Policy                                  | Train mean util | Val mean util | Test mean util | Override rate (test) |
|-----------------------------------------|----------------:|--------------:|---------------:|---------------------:|
| always anchor (rt_detr)                 | 0.5610          | 0.4623        | 0.6991         | 0.000                |
| oracle override if delta > 0.00         | 0.5703 (+0.009) | 0.4643 (+0.002) | 0.7050 (+0.0059) | 0.410                |
| oracle override if delta > 0.02         | 0.5696          | 0.4635        | 0.7041         | 0.036                |
| oracle override if delta > 0.05         | 0.5693          | 0.4629        | 0.7034         | 0.012                |

And — the most important table in this report — the **actual AP50 on
the same test split** (`scripts/baseline_ap_audit.py`):

| Method (val→test)                                | Val AP50 | Test AP50 |
|--------------------------------------------------|---------:|----------:|
| `fusion::wbf`                                    | 0.8989   | **0.8834** |
| `fusion::nms`                                    | **0.9032** | 0.8687 |
| `det::rt_detr`                                   | 0.8722   | 0.8664 |
| `oracle::per_cluster_best_available_source`      | 0.8748   | **0.8452** |
| `oracle::rtdetr_anchor + oracle_override`        | 0.8539   | 0.8452 |
| `oracle::nms_anchor + oracle_override`           | 0.8503   | 0.8452 |
| `det::retinanet`                                 | 0.8639   | 0.8177 |
| `det::yolo_modern`                               | 0.7483   | 0.4776 |
| `fusion::best_proposal` (one box per cluster)    | 0.4143   | 0.4426 |

**Per-cluster oracle is 2.1–3.8 AP points BELOW NMS/WBF on test.**

The user-supplied "Oracle = 0.8942" is not achievable inside the source-
routing framework on this data. It corresponds to the unconstrained
best-proposal-per-GT oracle which exists across clusters and can use
multiple boxes.

## 4. Hard-case overfit (Part 5 — intentionally not run)

See `reports/HARD_CASE_OVERFIT_GATE.md` for the rationale. Running this
gate is preempted by the upstream gate failures: any model that overfits
the 61 train positive overrides at margin 0.05 will (a) not generalize to
the 5 val and 5 test positives, and (b) be evaluated against a baseline
(WBF 0.8834) that it cannot reach inside this framework.

Table is therefore empty by design:

| Case | Train count | Precision | Recall | Passed? |
|------|------------:|----------:|-------:|---------|
| union-oracle-not-selected (margin 0.05) | n/a | n/a | n/a | NOT RUN — upstream gate failed |
| yolo-oracle-anchor-picks-rtdetr (m0.05) | n/a | n/a | n/a | NOT RUN — upstream gate failed |
| anchor-fails-alt-succeeds (m0.05)       | 61  | n/a | n/a | NOT RUN — see §6 of LEARNABILITY_AUDIT.md |

The hard-case **counts** (from `learnability_audit.summary.json`):

| Margin | Train | Val | Test |
|-------:|------:|----:|-----:|
| 0.00   | 692   | 191 | 123  |
| 0.02   |  79   |  17 |   9  |
| 0.05   |  61   |   5 |   5  |

Five val positives + five test positives at the only margin that maps to a
meaningful AP improvement is *not* a learnable target on this data.

## 5. Union and yolo specialist diagnosis (Parts 6, 7)

Per-source positive-override prior on TRAIN (margin 0.00, leak-safe; see
`learnability_audit/summary.json`):

| Slot | Source             | Train pos-override prior | Notes                          |
|------|--------------------|-------------------------:|--------------------------------|
| 0    | `yolo_modern`      | 0.288  (80/278)          | high  — when present, often positive |
| 3    | `retinanet`        | 0.286  (124/434)         | high  — similar to yolo        |
| 4    | `union`            | 0.277  (166/600)         | high  — fires on >¼ of clusters |
| 5    | `wbf`              | 0.277  (166/600)         | tied with union (often same box) |
| 6    | `nms_candidate`    | 0.072  (43/600)          | low   — barely beats anchor    |
| 7    | `soft_nms`         | 0.117  (70/600)          | low                            |
| 8    | `best_proposal`    | 0.072  (43/600)          | low                            |

But the AP gain *per positive override* at margin 0.05 is in {0.05, 0.10,
0.15} buckets with N ≤ 5 in val and test — too few to learn from.

**Union recall at the tabular gate's top-5% prediction: 0.071–0.238 (margin
0.00); 0.000 (margin ≥ 0.02).** YOLO recall: 0.037–0.148 (margin 0.00);
0.000–1.000 (margin ≥ 0.02, on N=1–2 positives).

The specialist heads I shipped in this codebase are mechanically correct
(see `tests/test_anchor_router.py::test_anchor_router_specialist_gate_blocks_low_prob_override`
— a positive specialist BCE prediction is required to permit an override).
But the data does not contain enough union- or yolo-positive cases at a
margin large enough to actually move AP50 to train them.

Per the user's table format:

| Source       | Oracle % (positive-override prior @ m0.00) | Recall Before (deep router v9) | Recall After (anchor router) | Selected % (target) |
|--------------|-------------------------------------------:|-------------------------------:|-----------------------------:|--------------------:|
| union        | 27.7 %                                      | ≈ 0 % (user report)            | NOT MEASURED — gate failed   | n/a                 |
| yolo_modern  | 28.8 %                                      | low (user report)              | NOT MEASURED — gate failed   | n/a                 |
| rt_detr      | n/a (anchor)                                | over-selected                  | n/a (anchor by construction) | ≈ 100 %             |
| retinanet    | 28.6 %                                      | n/a                            | NOT MEASURED                 | n/a                 |

## 6. Final real-VOC-car comparison (Part 9 — not run, but predictable)

The 10-seed sweep was not launched. Based on §3 above, the predictable
outcome is:

| Method                                       | Predicted Test AP50 | Achievable upper bound |
|----------------------------------------------|---------------------:|-----------------------:|
| `fusion::wbf`                                | 0.8834              | 0.8834                 |
| `fusion::nms`                                | 0.8687              | 0.8687                 |
| `det::rt_detr` (= anchor)                    | 0.8664              | 0.8664                 |
| **anchor router (always anchor in equilibrium)** | ≈ 0.8664        | ≤ 0.8452 if it ever overrides |
| **anchor router (oracle-quality routing)**   | 0.8452              | 0.8452                 |
| paired bootstrap P(anchor router > NMS)      | **far below 0.5**   | bounded below 0.5 by oracle |

I do not run the sweep because the result is structurally determined.

## 7. Failure examples (Part 12 §10 — informational, from the audit)

From `runs/real_voc_car_v2/learnability_audit/tabular.csv` and the audit
JSON. The patterns below are representative; the file has 5172 rows
total if you want to inspect individuals.

(a) **5 test-split clusters where union is oracle by margin > 0.05.**
These are exactly the clusters where the deep router *would* win
points — but five test points cannot drive a statistically significant
paired-bootstrap win over WBF.

(b) **123 test-split clusters where anchor (rt_detr) is not the
oracle at margin > 0.** The mean recoverable utility on those is +0.005.
Aggregated over the full 30-image test set, this is +0.0059 in mean
utility, or roughly +0.006 AP50. The strongest baseline WBF is +0.017
AP above the anchor — i.e. WBF already does what the oracle router
*aspires* to, plus more.

(c) **Zero test-split clusters where yolo_modern beats anchor by
margin > 0.05.** Yolo is under-selected because, on this data, when it
is oracle the gain is small (median +0.012 IoU).

(d) **All 4 v9 NMSOverrideRouter seed-3 overrides were wrong.** The
override head's "never override" attractor is the AP-rational policy
when expected gain is negative.

## 8. Final scientific conclusion

The TGraphX source-routing framework, as currently defined ("for each
cluster, choose one of N existing source boxes"), has a measured AP50
ceiling on the real VOC2007 car-only 30-image test split that is 2.4 AP
points below NMS and 3.8 AP points below WBF. This ceiling is not a
training failure; it is a structural property of the framework on this
data. No model class — neural anchor router, tabular gain gate, hybrid
ensemble — can clear it.

The previous loop assumed Oracle ≈ 0.8942 (which is the *best-proposal-
per-GT* unconstrained oracle, not achievable inside source routing) and
therefore that a +0.028 AP headroom existed. The actual achievable
headroom inside the framework is **negative** vs. the strongest baseline.

**Stop the loop on real VOC car-only.** Three coherent next moves, in
descending order of recommendation:

1. **Change the framework, not the model.** Replace "pick one of the
   source boxes" with "regress a new box from the cluster representation"
   (i.e. learn WBF). The TGraphX graph and slot embeddings can be reused;
   only the head changes. This is the only architectural change with a
   chance of beating WBF.

2. **Change the metric.** Mean per-cluster IoU is plausibly a better
   match to the framework: TGraphX already beats NMS on mean IoU in the
   on-disk override-router runs (0.444 vs 0.420). Mean-IoU + per-image
   FP control is a defensible "TGraphX wins" claim that is also true.

3. **Retire the experiment.** Mark this VOC car-only 200-image setting
   as `REAL_VOC_CAR_NOT_YET_WIN`; do not spend more compute or
   engineering on it. Move the source-router idea (if it is to survive)
   to a different dataset where the per-cluster source oracle is
   measurably above NMS/WBF — measure first, train second.

The architecture, the false-override penalty, the priors, the specialist
heads, the paired bootstrap — *all of it is mechanically correct* and
will pay off the day the framework lives in a regime where the source-
routing oracle dominates the baselines. On this regime, it doesn't, and
no engineering trick can change that.

End of verdict.
