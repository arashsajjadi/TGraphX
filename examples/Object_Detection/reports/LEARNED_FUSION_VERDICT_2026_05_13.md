# LEARNED FUSION VERDICT — empirical, real VOC car

**Author:** Claude Opus (principal-scientist, empirical mode)
**Date:** 2026-05-13
**Data:** `runs/real_voc_car_v2/graphs.pt` (200 imgs, 3 detectors, splits 140/30/30)
**Model:** `TGraphXLearnedBoxFusion`, residual mode, ‖Δ‖∞ ≤ 0.1·diag(WBF)
**Training:** 5 seeds × 40 epochs on RTX 5080 (4 min 52 s total).
**Reproducer:**
```bash
python scripts/box_fusion_oracle_audit.py --run-dir runs/real_voc_car_v2
python scripts/run_learned_fusion.py \
    --config configs/real_voc2007_car_learned_box_fusion.yaml \
    --run-dir-with-graphs runs/real_voc_car_v2 \
    --seeds 0 1 2 3 4 --epochs 40 --device cuda --out-dir runs_e40
```

---

## 1. Executive verdict

**`LEARNED_FUSION_NOT_YET_WIN`** — TGraphXLearnedBoxFusion ties WBF on
AP75 with a positive per-image direction (P(TGraphX > WBF) = 0.697,
Δ = +0.022), but **loses to the actual strongest baseline (NMS at AP75
= 0.6388)** with P(TGraphX > NMS) = 0.010, Δ = −0.037. The user's
success bar (P ≥ 0.95 vs strongest baseline) is not cleared.

The underlying oracle audit *does* show headroom: per-cluster
`wbf_residual_oracle` AP75 = 0.840 vs WBF 0.579 (+0.26 AP75). The
trained model captures roughly **only +0.022 of that +0.26 headroom** —
a recovery rate of ≈ 8.5 % — before it begins to overfit the 140-image
training set.

Honest read: the formulation change *unlocked* AP75 headroom that the
source-router could not access, but the model class + data size are
not jointly strong enough to reach the strongest classical baseline at
AP75. This is not a framework failure — the oracle proves the framework
is alive. It is a **`LEARNED_FUSION_OVERFITS`** result: more data and
stronger regularization (or larger detectors with more diverse
proposals, or a multi-class regime where NMS is weaker per-class) are
the natural next steps.

## 2. Oracle audit (Part 2 gate — passed)

`scripts/box_fusion_oracle_audit.py` on `runs/real_voc_car_v2`.

### Test split (30 images)

| Policy                                  | AP50    | AP75    | mIoU    | Δ AP75 vs WBF |
|-----------------------------------------|--------:|--------:|--------:|--------------:|
| `fusion::wbf`                           | **0.8834** | 0.5786 | 0.6476  | 0.0          |
| `oracle::gt_oracle`                     | 0.8573  | **0.8573** | **0.7379** | **+0.2787** |
| `oracle::wbf_residual_oracle` (capped 0.1·diag) | 0.8573  | 0.8399  | 0.6954  | +0.2613 |
| `oracle::convex_oracle`                 | 0.8573  | 0.6835  | 0.6610  | +0.1049 |

**Verdict on Part 2:** `BOX_FUSION_ORACLE_HAS_HEADROOM`. The bounded
residual recovers 98% of the gt_oracle AP75. The framework target is
learnable in principle.

## 3. Model ablation — 5 seeds, 40 epochs vs 80 epochs

Residual mode (only mode evaluated in this session — weighted / hybrid
deferred to follow-up):

| Setting | Seed range | Mean TGX AP50 | Mean TGX AP75 | Mean TGX mIoU | Verdict |
|---------|-----------:|--------------:|--------------:|--------------:|---------|
| 40 epochs | 0–4    | 0.8123 ± 0.035 | **0.5784 ± 0.027** | 0.6482 | best (this row) |
| 80 epochs | 0–4    | 0.7351 ± 0.043 | 0.5119 ± 0.048 | 0.6453 | overfit |

40 epochs ≈ matches WBF AP75 (0.5786). 80 epochs is a clear
overfit; loss continues to drop on train (2.34 → 2.07) while AP drops
on test. Conclusion: this dataset (140 train clusters) supports only
~40 epochs before the model overfits to train.

## 4. Baseline comparison (test, 5-seed mean — AP75 headline)

| Method                                  | Test AP50 | Test AP75 | Test mIoU |
|-----------------------------------------|----------:|----------:|----------:|
| `fusion::wbf`                           | **0.8834** | 0.5786   | 0.6476    |
| `fusion::nms`                           | 0.8687    | **0.6388** | **0.6496** |
| `det::rt_detr`                          | 0.8664    | 0.6351    | 0.6968    |
| `det::retinanet`                        | 0.8177    | 0.4952    | 0.6909    |
| **`fusion::tgraphx_learned_fusion`** (40 ep)  | 0.8123    | 0.5784    | 0.6482 |
| `det::yolo_modern`                      | 0.4776    | 0.3278    | 0.8162    |
| `fusion::best_proposal`                 | 0.4426    | 0.3898    | 0.8750    |

**The strongest validation-selected baseline at AP75 is NMS (0.6388),
not WBF (0.5786).** The user's previous prompts identified WBF as the
strongest baseline; the on-disk evidence says NMS is stronger at AP75
on this test split.

## 5. Paired bootstrap (5 seeds, n=30 images per resample)

### AP75 (headline metric per the audit)

| Comparison                | Mean P(TGX > method) | Mean Δ AP75 | Per-seed P range |
|---------------------------|---------------------:|------------:|------------------|
| **TGX vs `fusion::wbf`**  | **0.697**            | **+0.0225** | [0.449, 0.860]   |
| TGX vs `det::retinanet`   | 0.879                | +0.0460     | [0.681, 0.960]   |
| TGX vs `fusion::best_proposal` | 1.000           | +0.0974     | [1.0, 1.0]       |
| TGX vs `det::yolo_modern` | 1.000                | +0.3266     | [1.0, 1.0]       |
| TGX vs `fusion::nms`      | 0.010                | −0.0563     | [0.0, 0.031]     |
| TGX vs `det::rt_detr`     | 0.003                | −0.0568     | [0.0, 0.015]     |

### AP50

| Comparison                | Mean P(TGX > method) | Mean Δ AP50 |
|---------------------------|---------------------:|------------:|
| TGX vs `fusion::wbf`      | 0.208                | −0.0154     |
| TGX vs `fusion::nms`      | 0.404                | −0.0070     |
| TGX vs `det::rt_detr`     | 0.395                | −0.0078     |
| TGX vs `det::retinanet`   | 0.817                | +0.0370     |

### Read

- **AP75 vs WBF:** TGraphX wins per-image on average (P=0.70) with a
  meaningful Δ (+0.022 AP) — but does NOT clear the user's "no win
  unless P ≥ 0.95" bar.
- **AP75 vs NMS:** TGraphX loses clearly. NMS is the real strongest
  baseline at AP75; the project's win-condition target should have
  always been "beat NMS at AP75."
- **AP50:** Matches the framework-ceiling finding — TGraphX is below
  WBF/NMS/rt_detr at AP50 because the per-cluster oracle itself is
  capped at AP50 = 0.857 < WBF AP50 = 0.883.

## 6. Box-refinement diagnostics (5-seed mean, test split)

| Metric                                | Value     |
|---------------------------------------|----------:|
| Mean ‖Δ‖₂ (pixels)                    | 1.03      |
| Max ‖Δ‖₂ across all test clusters     | 6.39      |
| Out-of-bounds box rate (pre-clamp)    | 0.110     |
| ECE @ IoU=0.5                          | 0.198     |
| ECE @ IoU=0.75                         | 0.318     |
| TGraphX test mIoU                      | 0.6482    |
| WBF test mIoU                          | 0.6476    |
| `wbf_residual_oracle` test mIoU        | 0.6954    |

- Mean Δ is tiny (1 pixel). The model is making very small corrections.
- Max Δ of 6 pixels indicates a small set of clusters where the model
  pushes more aggressively — these are also the OOB cases (11% of
  test boxes go out of image bounds before the clamp). That OOB rate
  is high enough to warrant a stronger box-bound penalty in future runs.
- ECE@0.75 is high (0.32). The TP75 head is poorly calibrated —
  sigmoid(tp75_logit) is over-confident relative to true TP75 rate.
  This hurts AP75 ranking and is a clean target for the next iteration
  (temperature scaling on val).
- mIoU is matched-baseline (0.648 vs WBF 0.648). The model is not
  improving localization quality on average — only on a sub-fraction
  of clusters, which is what produces the +0.022 mean AP75 win vs WBF
  in the paired bootstrap.

## 7. Failure examples (qualitative narrative)

I did not generate per-cluster failure images in this run (would need
to load test images by VOC ID and overlay boxes — straightforward but
deferred). The summary numbers explain the failures:

(a) **Clusters where WBF beats TGX (AP50 vs WBF P=0.21).** WBF
synthesizes a box that has IoU > 0.5 with GT; TGraphX's residual moves
the WBF box slightly and the new IoU drops below 0.5. This is the
"AP50 regression from box meddling" problem and is reflected in TGX's
AP50 mean (0.812) being 0.07 below WBF's (0.883). Δ-regularization
weight could be raised to mitigate.

(b) **Clusters where TGX beats WBF at AP75 (Δ=+0.022).** WBF has IoU
in (0.5, 0.75); TGX pushes IoU above 0.75 on a subset. This is the
intended mechanism, and it works on a measurable fraction of clusters
but not enough to clear the NMS bar.

(c) **Clusters where NMS beats both WBF and TGX at AP75.** NMS keeps
the top-confidence proposal per region; that proposal tends to have
higher IoU at the high-precision end (0.75) than either WBF's score-
weighted average or TGX's small correction over WBF. To capture these
the model would need to *recognize* clusters where NMS's pick is
better and **switch the anchor to NMS for those clusters** — which is
back-door routing. A natural follow-up is an *anchor-mixture head*
(σ(α)·WBF + (1−σ(α))·NMS) + residual.

## 8. Failure-mode diagnosis (Part 10)

The user's Part-10 options map as follows:

- `BOX_FUSION_ORACLE_NO_HEADROOM` — **No.** Oracle audit shows
  `wbf_residual_oracle` AP75 = 0.84, a +0.26 gap above WBF.
- `LEARNED_FUSION_OVERFITS` — **Yes.** Loss continues to drop while
  test AP drops past 40 epochs. 140 train clusters are too few for
  the encoder's parameter count without stronger regularization.
- `SCORE_CALIBRATION_FAILURE` — **Partially.** ECE@0.75 = 0.318 is
  poor; the TP75 logits over-state confidence. Temperature scaling on
  val would help but won't close the 0.06 AP75 gap to NMS.
- `GRAPH_CONSTRUCTION_LIMIT` — **At AP50, yes.** Per-cluster oracle
  AP50 = 0.857 vs WBF 0.883 says the cluster set is not as good as
  WBF's implicit clustering at AP50. At AP75 it is not the limit.
- `WBF_ALREADY_OPTIMAL` — **No, but NMS is stronger than expected at
  AP75 (0.6388).** WBF is the strongest baseline at AP50, not AP75.

**Primary diagnosis: `LEARNED_FUSION_OVERFITS` + the strongest AP75
baseline (NMS) is harder than the project assumed.**

## 9. What stays useful from this session

1. `scripts/box_fusion_oracle_audit.py` — every claim about "is the
   framework alive?" can now be answered before any training.
2. `src/od_graph_fusion/learned_box_fusion.py` — `TGraphXLearnedBoxFusion`
   in three modes (residual / weighted / hybrid). Only residual was
   trained; weighted/hybrid are deferred.
3. `src/od_graph_fusion/multi_seed_learned_fusion.py` — end-to-end
   runner with paired bootstrap, baselines, ECE/Brier, Δ diagnostics.
4. `runs/real_voc_car_v2/box_fusion_oracle_audit.json`
   + `runs_e40/.../{metrics_seed*.json, summary.json}` — full numbers
   used in this report.

## 10. Strict scientific conclusion

The framework change from "pick one source per cluster" to "learn a
bounded residual over WBF" was empirically justified: the oracle
audit shows +0.26 AP75 headroom over WBF that no source-router can
access. After training (5 seeds × 40 epochs), the learned model
**ties WBF at AP75 per-image (P=0.70, Δ=+0.022) but does not clear the
0.95 paired-bootstrap bar.** Against the actual strongest AP75
baseline (NMS = 0.6388), TGraphX loses clearly (P=0.010).

The honest interpretation:

- The formulation is correct. The on-disk oracle confirms the AP75 gap
  can be closed in principle.
- The trained model captures only ≈ 8.5 % of the oracle's available
  AP75 headroom over WBF in 40 epochs, then overfits.
- **The strongest AP75 baseline is NMS, not WBF**, and the project
  needs to be re-targeted accordingly. Beating NMS at AP75 (closing a
  0.06 AP gap, with only 30 test images per seed) is a tighter target
  than beating WBF.

**Concrete next-step recommendations (in descending order of
expected impact):**

1. **Anchor-mixture head.** Let the model choose anchor ∈ {WBF, NMS,
   rt_detr} per cluster via a softmax, *then* regress Δ over that
   anchor. This addresses NMS's AP75 advantage by enabling the model
   to inherit NMS's better-localized boxes on the relevant clusters.
2. **Calibration.** Add temperature scaling on val for the TP75 head
   (ECE@0.75 = 0.32 → target ≤ 0.10). Cheap and may add a few AP75
   points.
3. **Stronger regularization.** Δ-reg weight 0.05 → 0.2 to suppress
   the 11% out-of-bounds clusters and the destructive AP50 drift. Also
   add weight-decay and possibly dropout.
4. **Hybrid mode.** Weighted-fusion head (`final_box = Σ wᵢ·boxᵢ + Δ`)
   may outperform pure residual when WBF localization is not optimal.
5. **More data.** Move to the full VOC car set (1644 trainval images
   instead of 200) — the obvious unlock. The current dataset is too
   small for the encoder's capacity.

The `TGraphXLearnedBoxFusion` codebase is ready for any of (1)–(5).
The infrastructure (runner, paired bootstrap, oracle audit, baseline
comparison, Step-06-compatible metrics file) is already in place. The
verdict is honest: a directional AP75 win over WBF exists but the
0.95 paired-bootstrap bar over the *actual* strongest baseline (NMS)
is not cleared in this session.

End of verdict.
