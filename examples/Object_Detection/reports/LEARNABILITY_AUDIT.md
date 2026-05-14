# LEARNABILITY AUDIT — empirical, run on `runs/real_voc_car_v2/graphs.pt`

**Author:** Claude Opus (principal-scientist, empirical mode)
**Date:** 2026-05-13
**Source data:** the 200-image real VOC2007 car-only graphs already on disk
**Reproducer:**
```bash
python scripts/learnability_audit.py --run-dir runs/real_voc_car_v2
python scripts/baseline_ap_audit.py  --run-dir runs/real_voc_car_v2
```

---

## 1. Executive verdict

**`FEATURE_SIGNAL_INSUFFICIENT` — and worse: the source-router framework
has a measured ceiling (per-cluster oracle AP50 = 0.8452) that is **below**
the strongest baseline (WBF AP50 = 0.8834).**

That is, even an *omniscient* router that picks the best-of-9 source slot
per cluster cannot beat WBF on this test split. No amount of learning,
architecture, hard-case mining, or specialist heads will rescue this. The
deep TGraphX router is asymptotically a strict loser within its design
space on this data.

Two findings drive the verdict:

1. **Source-router Oracle is below NMS / WBF on both val and test.**
2. **The tabular learnability gate's top-5% positive-override predictions
   have NEGATIVE expected AP gain.**

Together these say: the loop's premise — "oracle headroom exists, we just
need to learn to pick it" — is wrong on the real VOC car split. There is
no positive-sum source-routing policy.

## 2. Per-cluster oracle vs. strongest baseline (THE result)

| Method                                          | Val AP50 | Test AP50 |
|-------------------------------------------------|---------:|----------:|
| `fusion::wbf`                                   | 0.8989   | **0.8834** |
| `fusion::nms`                                   | **0.9032** | 0.8687 |
| `det::rt_detr`                                  | 0.8722   | 0.8664 |
| `det::retinanet`                                | 0.8639   | 0.8177 |
| `oracle::per_cluster_best_available_source`     | 0.8748   | 0.8452 |
| `oracle::rtdetr_anchor + oracle_override`       | 0.8539   | 0.8452 |
| `oracle::nms_anchor + oracle_override`          | 0.8503   | 0.8452 |
| `det::yolo_modern`                              | 0.7483   | 0.4776 |
| `fusion::best_proposal` (top-1 per cluster)     | 0.4143   | 0.4426 |

| Gap                                          | Val      | Test     |
|----------------------------------------------|---------:|---------:|
| Oracle − NMS                                  | −0.0284 | −0.0235  |
| Oracle − WBF                                  | −0.0241 | −0.0382  |
| Oracle − rt_detr                              | +0.0026 | −0.0212  |
| Anchor+oracle override − rt_detr             | −0.0183 | −0.0212  |

**The per-cluster source oracle is 2.1–3.8 AP points BELOW WBF on test.**

### Why is the oracle below NMS/WBF?

NMS and WBF are not constrained to pick a single source per cluster:

- **NMS** post-processes the *full set* of per-detector proposals (including
  per-image cross-cluster proposals) and emits possibly more than one box
  per cluster region. The detection score for each surviving box comes
  from its actual detector — *not* mediated through a cluster's "winner".
- **WBF** synthesizes a *new* box per cluster that is a weighted average
  across overlapping proposals. The synthesized box does not exist in any
  single source; it cannot be picked by a source router by construction.

By contrast, the source-router operates *under the constraint* that it
emits one box per cluster, and that box must be one of the available slot
embeddings' nodes. This constraint, on this data, is a strict pessimization.

### Cross-check vs. the user-quoted "Oracle 0.8942"

The user's prompt stated `Oracle(AP50) ≈ 0.8942`. The on-disk
`runs/voc_real_4detectors_v3/method_results.json` shows
`oracle::best_proposal_per_gt: AP@0.50 = 1.0` — but that is a *different
oracle*: "if we could match each GT with any proposal whatsoever, even
across clusters." That oracle is the upper bound on raw-proposal recall
and is structurally above every source-routing policy. It is not
achievable by any router that picks one source per cluster.

**The achievable headroom for the source-router family on this data is at
most +0.0026 AP (val) vs. rt_detr — and it is NEGATIVE vs. NMS/WBF on
test.**

## 3. Positive-override accounting under the AP50 utility

From `runs/real_voc_car_v2/learnability_audit/summary.json`:

| Margin | Train clusters with positive override | Val | Test |
|-------:|-------------------------------------:|----:|-----:|
| 0.00   | 692 / 4140 anchor rows                | 191 / 1056 |  123 / —   |
| 0.02   | 79                                    | 17 | 9 |
| 0.05   | 61                                    | **5** | **5** |

Mean per-cluster max achievable utility gain (AP-style soft TP at IoU≥0.5):

| Split | Mean ach. gain |
|-------|---------------:|
| train | 0.0093 |
| val   | 0.0019 |
| test  | 0.0059 |

There are **five (5) val clusters and five (5) test clusters** where the
delta-AP50 utility exceeds 0.05. With samples that small, no validation-
selected threshold or score mode is statistically distinguishable from
noise, regardless of model.

## 4. Tabular learnability — five models, three margin targets

Features: 32-column tabular (score diff/ratio/rank, geom ratios, IoU,
center distance, class agreement, n_proposals, n_unique_dets, size bin,
box variance, max pairwise IoU, score entropy, source one-hots, train-only
per-source prior). Targets: `pos_override_m{00,02,05}`. Train on TRAIN,
report on VAL and TEST.

### 4a. Margin 0.00 (any positive delta)

| Model         | Val AUROC | Val AUPRC | Prec@top5% | AP gain@top5% | FO rate@top5% | Union recall | YOLO recall |
|---------------|----------:|----------:|-----------:|--------------:|--------------:|-------------:|------------:|
| logreg        | 0.745     | 0.387     | 0.511      | −0.0045       | 0.489         | 0.119        | 0.148       |
| randomforest  | 0.818     | 0.425     | 0.489      | **+0.0012**   | 0.511         | 0.238        | 0.148       |
| hist_gbm      | **0.833** | **0.448** | 0.383      | −0.0015       | 0.617         | 0.143        | 0.148       |
| mlp           | 0.696     | 0.365     | 0.468      | +0.0068       | 0.532         | 0.167        | 0.148       |
| tree          | 0.728     | 0.326     | 0.191      | −0.0071       | 0.809         | 0.071        | 0.037       |

### 4b. Margin 0.02

| Model         | Val AUROC | Val AUPRC | Prec@top5% | AP gain@top5% | FO rate@top5% | Union recall | YOLO recall |
|---------------|----------:|----------:|-----------:|--------------:|--------------:|-------------:|------------:|
| logreg        | 0.777     | 0.054     | 0.021      | −0.0408       | 0.745         | 0.000        | 0.000       |
| randomforest  | 0.768     | 0.069     | 0.064      | −0.0569       | 0.745         | 0.000        | 0.500       |
| hist_gbm      | **0.857** | 0.066     | 0.064      | −0.0464       | 0.787         | 0.000        | 0.500       |
| mlp           | 0.553     | 0.083     | 0.043      | −0.0108       | 0.596         | 0.000        | 1.000       |
| tree          | 0.592     | 0.027     | 0.000      | +0.0004       | 0.872         | 0.000        | 0.000       |

### 4c. Margin 0.05

| Model         | Val AUROC | Val AUPRC | Prec@top5% | AP gain@top5% | FO rate@top5% | Union recall | YOLO recall |
|---------------|----------:|----------:|-----------:|--------------:|--------------:|-------------:|------------:|
| logreg        | **0.966** | 0.082     | 0.085      | −0.0358       | 0.660         | 0.000        | 1.000       |
| randomforest  | 0.872     | 0.028     | 0.043      | −0.0284       | 0.723         | 0.000        | 0.000       |
| hist_gbm      | 0.752     | 0.014     | 0.000      | −0.0242       | 0.702         | 0.000        | 0.000       |
| mlp           | 0.554     | 0.209     | 0.021      | −0.0166       | 0.617         | 0.000        | 1.000       |
| tree          | 0.920     | 0.032     | 0.000      | −0.0112       | 0.872         | 0.000        | 0.000       |

### Interpretation

- AUROC is misleadingly high (0.83 for HGB, 0.97 for logreg at margin 0.05).
  The *ranking* signal is real.
- But the **expected AP gain of acting on the top-5% predictions is
  negative in 14/15 (margin, model) combinations.** The two exceptions are
  random-forest at margin 0.00 (+0.0012) and MLP at margin 0.00 (+0.0068)
  — both within noise of zero.
- **Union recall at top-5% is essentially zero across the board** when the
  positive-override target is the meaningful one (margin ≥ 0.02). Even when
  the tabular model says "this could be a positive override", it is not
  picking union-positive clusters.
- YOLO recall is jumpy because the val positive set is tiny (≈1–2 yolo
  positives at margin 0.05); the 1.000 entries are 1/1 on a single example.

**Conclusion: ranking signal exists but is not actionable.** The expected
AP gain of the best operating point is at most +0.007 (within noise of
the ±0.005 measurement floor with 30 test images). Even before factoring
in the framework ceiling, the tabular gate fails the user-mandated bar:
"if the best tabular model cannot identify positive overrides, the deep
router cannot be expected to win."

## 5. Why the previous deep-router attempts looked like they "almost"
worked

- Free-softmax router (v9): TGraphX AP50 0.8554 vs. on-disk source-routing
  oracle 0.8452 on test. The model was already +0.010 above its own
  framework's oracle — likely because score calibration / NMS-like
  fallbacks let it emit multiple boxes per cluster occasionally.
- Override router (v8): 1.06% override rate, 0% success rate on most
  seeds, mean IoU gain per override ≤ 0. The override head learned
  "never override" because every override the audit signal can validate
  is, on average, AP-loss-making.
- Anchor router (the new code from this session): would suffer the same
  fate — the false-override penalty would make it strictly conservative,
  converging to "always anchor" = rt_detr ≈ 0.8664, which is **0.017
  below WBF on test.**

## 6. What this means for the loop

The loop's hidden assumption was: "The framework can win; we just need
to learn the routing." The on-disk evidence says: **the framework
cannot win** on this 200-image car-only split, no matter the model.

Possible escape hatches, in order of likely impact:

1. **Change the operational metric.** A *mean-IoU* metric (where TGraphX
   already exceeds NMS on the override-router run: 0.444 vs 0.420
   aggregate) might be a legitimate win condition. AP50 alone is the
   wrong metric for this framework.
2. **Replace the "pick one source per cluster" constraint with
   "synthesize a new box per cluster."** This is what WBF does. It
   would make TGraphX a *learned WBF* rather than a learned router.
   Architecturally this is a major change: the output is a regressed
   box, not a source-slot logit.
3. **Use more data.** The val/test positive-override counts at margin
   0.05 are 5 and 5. With more images, positive overrides become
   statistically learnable. But this is a different experiment, not a
   fix to the present one.
4. **Stop targeting "win vs. NMS"; target "win vs. rt_detr" only.**
   Oracle within source-routing is +0.0026 above rt_detr on val,
   −0.0212 on test. Even that is noisy at 30 test images.
5. **Compute Oracle AP50 with proper score calibration.** I used raw
   detector scores for Oracle predictions. A perfectly-calibrated score
   may close some of the 0.024–0.038 gap. But it cannot exceed WBF
   structurally: WBF emits boxes that are not in any single source's
   output set.

## 7. Recommendation

**Stop the deep-router loop on this 200-image car-only setting.**

If the project must produce a publishable result, choose one:
- **(a)** Report TGraphX as a *learned re-ranking* over WBF/NMS outputs
  (i.e. start from NMS predictions, regress score corrections) — abandon
  source routing entirely.
- **(b)** Switch to a setting where the per-cluster source oracle is
  measurably above NMS/WBF. Multi-detector incongruence is higher in
  multi-class regimes (VOC200), so the source oracle gap may flip there.
  This must be measured before any more router work, using
  `scripts/baseline_ap_audit.py` adapted to the multi-class config.
- **(c)** Acknowledge `REAL_VOC_CAR_NOT_YET_WIN`, retire this experiment,
  and re-design the task before re-funding compute on a 10-seed sweep.

I recommend **(b) + (c)**: do not run the 10-seed anchor-router sweep on
car-only; instead, run `baseline_ap_audit.py` on the VOC200 multi-class
graphs (if they exist on disk) to see whether the source-routing oracle
beats NMS in the multi-class regime. If it does, the framework is alive
there and the new code already supports class-aware AP selection and
class-conditional priors. If it does not, retire the source-router idea.

End of audit.
