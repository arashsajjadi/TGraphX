# HARD_CASE_OVERFIT_GATE — gate not entered

**Author:** Claude Opus (principal-scientist, empirical mode)
**Date:** 2026-05-13

**Verdict at top: `HARD_CASE_OVERFIT_FAILED` — preempted by prior gate.**

The user's directive (Part 5) requires a hard-case overfit gate before
running any 10-seed sweep. I did not enter that gate. The reason is
upstream: the **Oracle Policy Simulation gate (Part 4)** and the
**Learnability Audit gate (Part 3)** failed first, and the user's
directive is explicit: *"If any gate fails, stop and diagnose. Do not
continue blindly."*

See `reports/LEARNABILITY_AUDIT.md` for the upstream evidence. The
findings, in one line each:

1. **The per-cluster source-routing Oracle (AP50 = 0.8452) is below the
   strongest baseline WBF (AP50 = 0.8834) on test.** No router can beat
   WBF inside this framework.
2. **Tabular models reach AUROC 0.83 on positive-override prediction
   but their top-5% predictions have NEGATIVE expected AP gain.** Even
   the strongest tabular gate would lose AP if deployed.
3. **There are 5 (five) val and 5 (five) test clusters with delta-AP50
   > 0.05.** No threshold-selection or temperature-scaling protocol
   can be statistically distinguished from noise on a positive set
   that small.

Running hard-case overfit at this point would be a circus: the deep
router would (best case) memorize the 61 train positive-override
clusters at margin 0.05, fail to generalize to the 5 val clusters, and
be evaluated against a strongest baseline (WBF) it cannot beat in
principle.

**Therefore: HARD_CASE_OVERFIT not run.** The compute that would have
gone into Part 5 has instead gone into Parts 3 and 4 where the
empirical answer was decided.

## What would unlock the hard-case overfit

The hard-case overfit becomes scientifically meaningful when ONE of the
following holds:

1. **On a different data slice, the source-routing Oracle is above
   NMS/WBF.** Suggested: VOC200 multi-class (its graphs are not yet on
   disk — Step 03 needs to be re-run). The on-disk VOC200 override-
   router results (TGraphX_override 0.8822 vs NMS 0.8837) suggest the
   same ceiling holds there, but this should be measured by running
   `scripts/baseline_ap_audit.py` against multi-class graphs.

2. **The framework is changed so the router can synthesize new boxes**
   (box regression head on top of the chosen-source embedding), letting
   the output set escape the "one of the existing source boxes per
   cluster" constraint that makes source routing strictly below WBF.

3. **A different operational metric is chosen.** Mean per-cluster IoU
   *is* above NMS in the on-disk override-router runs (0.444 vs 0.420).
   That is a defensible win on a different metric. AP50 is the wrong
   metric for this framework.

Until at least one of those is true, hard-case overfit is dead engineering
time.

## What I did instead

- Built `scripts/learnability_audit.py` (tabular gain gate, all 5
  sklearn models, oracle policy simulation, leak-safe priors).
- Built `scripts/baseline_ap_audit.py` (AP50 of every baseline + per-
  cluster Oracle policies on already-saved graphs).
- Ran both on `runs/real_voc_car_v2/graphs.pt` and wrote
  `reports/LEARNABILITY_AUDIT.md`.
- Wrote this file to record why the hard-case overfit step is the
  *wrong next step*.

## What NOT to do next

- Do NOT run `scripts/sanity_overfit_anchor.py` or
  `scripts/run_anchor_multi_seed.py` on `real_voc2007_car_anchor_router.yaml`.
  Those scripts will execute and produce numbers, but the numbers are
  predestined to lose to WBF.
- Do NOT add another specialist head, prior table, or pairwise feature.
  The bottleneck is the framework, not the model class.

End.
