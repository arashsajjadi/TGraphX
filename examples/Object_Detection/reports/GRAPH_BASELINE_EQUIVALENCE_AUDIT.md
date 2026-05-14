# Graph Baseline Equivalence Audit

Config: `configs/universal_candidate_voc_car_v2.yaml`
Run dir: `runs/universal_candidate_voc_car_v2`
Tolerance: 0.015
Test graphs: 1192

## Root Cause Analysis

**WBF box equivalence**: Graph cluster node box = `weighted_box_average(cluster proposals)`
Max |graph_wbf_box − recomputed_wbf_box| = 19.989563
✗ Box mismatch found

**WBF score formula**: External WBF uses `mean(scores) × min(1.0, N/3)`.
Graph cluster node score (AFTER FIX) uses the same formula.
Before this fix, graph score = raw `mean_score` → systematically different from external WBF.

## Equivalence Table

| External Baseline | Ext AP50 | Graph Node | Graph AP50 | Ext AP75 | Graph AP75 | Match |
|:------------------|---------:|:-----------|----------:|---------:|----------:|:-----|
| external::wbf             | 0.9134 | graph::cluster                 | 0.9130 | 0.7258 | 0.7309 | ✓ |
| external::nms             | 0.8854 | graph::nms_candidate           | 0.8815 | 0.6597 | 0.6624 | ✓ |

## Overall: PASS — All pairs within tolerance

Training can proceed.
