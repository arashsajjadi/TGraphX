# BOX_FUSION_ORACLE_AUDIT — empirical, real VOC car

**Source:** `scripts/box_fusion_oracle_audit.py` on `runs/real_voc_car_v2/graphs.pt`.
**Outputs:** `runs/real_voc_car_v2/box_fusion_oracle_audit.json` (raw).

## Oracle table (val)

| Policy                                  | AP50    | AP75    | mIoU    | Δ AP50 vs WBF | Δ AP75 vs WBF | Δ mIoU vs WBF |
|-----------------------------------------|--------:|--------:|--------:|--------------:|--------------:|--------------:|
| `oracle::gt_oracle`                     | 0.9015  | **0.9015** | 0.5912 | +0.0026 | **+0.2003** | +0.0561 |
| `oracle::wbf_residual_oracle` (capped)  | 0.9015  | 0.9003  | 0.5642  | +0.0026  | +0.1992  | +0.0292  |
| `oracle::convex_oracle`                 | 0.9015  | 0.8167  | 0.5330  | +0.0026  | +0.1155  | −0.0021  |
| `fusion::wbf`                           | **0.8989** | 0.7012 | 0.5351 | 0.0      | 0.0      | 0.0      |
| `oracle::wbf_residual_oracle` (uncapped — param bug, ignore) | 0.0 | 0.0 | 0.041 | — | — | — |

## Oracle table (test)

| Policy                                  | AP50    | AP75    | mIoU    | Δ AP50 vs WBF | Δ AP75 vs WBF | Δ mIoU vs WBF |
|-----------------------------------------|--------:|--------:|--------:|--------------:|--------------:|--------------:|
| `fusion::wbf`                           | **0.8834** | 0.5786 | 0.6476  | 0.0      | 0.0      | 0.0      |
| `oracle::gt_oracle`                     | 0.8573  | **0.8573** | **0.7379** | −0.0261 | **+0.2787** | **+0.0903** |
| `oracle::wbf_residual_oracle` (capped)  | 0.8573  | 0.8399  | 0.6954  | −0.0261  | +0.2613  | +0.0478  |
| `oracle::convex_oracle`                 | 0.8573  | 0.6835  | 0.6610  | −0.0261  | +0.1049  | +0.0134  |
| `oracle::wbf_residual_oracle` (uncapped — param bug, ignore) | 0.0 | 0.0 | 0.052 | — | — | — |

## Reading the numbers

### AP50 is saturated by WBF — no headroom at this threshold

At IoU≥0.5, WBF (0.8834) is **above** every per-cluster oracle, including
`gt_oracle` (0.8573). Per-cluster oracles produce *one box per cluster*,
while WBF's score-weighted box synthesis effectively packs more
information per output. At AP50 the framework is capped here.

### AP75 is massively under-served by WBF — +0.26 AP oracle headroom

At IoU≥0.75, WBF (0.5786) is **far below** the per-cluster oracle. The
WBF box's mean IoU with matched GT is ≈ 0.65 (test), so a huge fraction
of WBF boxes sit in the (0.5, 0.75) IoU band — counted as TP at AP50,
but counted as **FN at AP75**. A learned refinement model that pushes
those IoUs above 0.75 captures the AP75 headroom.

The capped residual oracle (WBF + Δ, ‖Δ‖∞ ≤ 0.1·diag(WBF)) recovers
**0.84 / 0.86 = 98 %** of the gt-oracle AP75 with only a small, bounded
correction. This means the regression target is *learnable in
principle*: small local box corrections recover almost all the
available AP75.

### mIoU also has headroom

Both `gt_oracle` (+0.09 mIoU on test) and `wbf_residual_oracle` (+0.05
mIoU). Mean IoU is the natural metric for a *localization* refinement
head, and the headroom is sizable.

## Verdict

**`BOX_FUSION_ORACLE_HAS_HEADROOM` — but the headroom is in AP75 and
mIoU, NOT in AP50.**

This is a substantive correction to the project's framing:

- The source-router project chased AP50 wins. AP50 is saturated by WBF
  on this data; no further work there.
- The learned-fusion project should chase **AP75 and mIoU**. The oracle
  ceiling at AP75 is 0.86 vs WBF 0.58 on test, a +0.28 gap that is
  reachable in principle by a bounded local box regression.

Recommended primary metric for `TGraphXLearnedBoxFusion`:

> **AP75** with `mIoU` as a secondary continuous indicator.
> AP50 reported as a guardrail — the model must not *fall below* WBF at
> AP50 while improving AP75.

## Verified facts going into Part 3

1. The graph + cluster construction can represent boxes good enough for
   high AP75 (gt-oracle reaches 0.86 on test).
2. A small bounded residual (‖Δ‖∞ ≤ 0.1·diag) over WBF already captures
   98 % of that ceiling. The residual is the right output head.
3. Convex combinations are strictly weaker (test AP75 0.68) — they
   can't escape the convex hull of source corners, which is not tight
   enough for AP75.
4. The unbounded-residual numbers above are a parameterization bug in
   the audit (tanh·10.0·diag drives the optimizer past valid boxes);
   ignored, not a real result.

## Proceed criteria for Part 3

- Build `TGraphXLearnedBoxFusion` with **residual head as default**
  (option A: `final_box = wbf_box + Δ`, Δ from a learned regression
  head conditioned on the cluster's slot embeddings + pairwise features).
- Train with SmoothL1 + GIoU on Δ targets; BCE on TP75 logits; IoU
  regression as auxiliary.
- Loss weights default to `λ_box=1.0, λ_giou=1.0, λ_tp75=2.0,
  λ_iou=0.5, λ_delta_reg=0.1`.
- Headline metric for verdict: **AP75 vs WBF on test, paired bootstrap
  on the same images**. AP50 is a guardrail.

End of audit.
