"""Generate final report for TGraphXCandidateNodeSelector experiment.

Reads:  {run_dir}/candidate_eval_summary.json   (multi-seed evaluation)
        {run_dir}/candidate_summary.json         (training summary, fallback)
        {run_dir}/object_manifest.json
        {run_dir}/candidate_eval_seed*.json      (per-seed details)

Writes: reports/CANDIDATE_NODE_SELECTOR_REPORT.md
        {run_dir}/candidate_report.json

The final report describes the method as:
  "TGraphX performs object-level candidate-node classification.
   For each object hypothesis, candidate boxes from multiple detectors and
   fusion methods are represented as visual crop-tensor nodes in a graph.
   TGraphX uses tensor-aware graph message passing to select the best
   candidate node for that object."
"""
import argparse, json, sys, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    ap = argparse.ArgumentParser(description="Generate candidate-node-selector report")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--reports-dir", default="reports")
    args = ap.parse_args()

    from od_graph_fusion.config import load_config
    cfg = load_config(args.config)
    run_dir = Path(args.run_dir or f"runs/{cfg.get('run_name', 'exp')}")
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest   = _load_json(run_dir / "object_manifest.json")
    eval_summ  = _load_json(run_dir / "candidate_eval_summary.json")
    train_summ = _load_json(run_dir / "candidate_summary.json")

    detector_names = manifest.get("detector_names", [])
    class_names    = manifest.get("class_names", ["car"])
    num_classes    = manifest.get("num_classes", len(class_names))
    n_train        = manifest.get("num_train", "?")
    n_val          = manifest.get("num_val", "?")
    n_test         = manifest.get("num_test", "?")
    total_graphs   = manifest.get("total_object_graphs", "?")
    crop_size      = manifest.get("crop_size", "?")

    # Load per-seed details
    seed_results = []
    for p in sorted(run_dir.glob("candidate_eval_seed*.pt")):
        seed = int(p.stem.split("seed")[1])
        seed_results.append(_load_json(run_dir / f"candidate_eval_seed{seed}.json"))
    if not seed_results:
        for p in sorted(run_dir.glob("candidate_metrics_seed*.json")):
            seed_results.append(_load_json(p))

    # Prefer eval_summary; fall back to train_summary
    method_means = eval_summ.get("method_means", train_summ.get("method_means", {}))

    # ── Build table ───────────────────────────────────────────────────────
    tgx_key = "tgraphx_candidate_selector"
    rows = []
    for n_m, v in sorted(method_means.items(), key=lambda x: -x[1].get("AP50_mean", 0)):
        ap50_m = v.get("AP50_mean", 0)
        ap50_s = v.get("AP50_std", 0)
        ap75_m = v.get("AP75_mean", 0)
        ap75_s = v.get("AP75_std", 0)
        miou_m = v.get("mIoU_mean", 0)
        marker = " ← **TGraphX**" if n_m == tgx_key else ""
        rows.append(f"| {n_m:<42} | {ap50_m:.4f} ± {ap50_s:.4f} | {ap75_m:.4f} ± {ap75_s:.4f} | {miou_m:.4f} |{marker}")

    table_header = (
        "| Method                                     | AP50 (mean ± std)      | AP75 (mean ± std)      | mIoU   |\n"
        "|:-------------------------------------------|:-----------------------|:-----------------------|:-------|\n"
    )
    table = table_header + "\n".join(rows)

    # ── Bootstrap summary ─────────────────────────────────────────────────
    boot_lines = []
    if seed_results:
        for n_m, v in eval_summ.get("method_means", {}).items():
            if n_m == tgx_key:
                continue
        # Try to pull bootstrap from the first seed result
        sr = seed_results[0]
        for n_m, bv in sr.get("paired_bootstrap_ap50", {}).items():
            p_val = bv.get("p_a_gt_b", 0)
            delta = bv.get("mean_diff", 0)
            sig = "✓ p≥0.95" if p_val >= 0.95 else ("○ p≥0.80" if p_val >= 0.80 else "✗")
            boot_lines.append(f"  - TGX vs {n_m}: Δ AP50={delta:+.4f}  p(TGX>baseline)={p_val:.3f}  {sig}")

    # ── Verdicts ──────────────────────────────────────────────────────────
    tgx_means = method_means.get(tgx_key, {})
    nms_means  = method_means.get("external::nms", {})
    wbf_means  = method_means.get("external::wbf", {})

    tgx_ap50 = tgx_means.get("AP50_mean", 0)
    nms_ap50 = nms_means.get("AP50_mean", 0)
    wbf_ap50 = wbf_means.get("AP50_mean", 0)
    tgx_ap75 = tgx_means.get("AP75_mean", 0)
    nms_ap75 = nms_means.get("AP75_mean", 0)
    wbf_ap75 = wbf_means.get("AP75_mean", 0)

    best_baseline_ap50 = max(nms_ap50, wbf_ap50, 1e-9)
    delta_ap50 = tgx_ap50 - best_baseline_ap50
    delta_ap75 = tgx_ap75 - max(nms_ap75, wbf_ap75, 1e-9)

    if delta_ap50 > 0.005 or delta_ap75 > 0.005:
        verdict = "TGRAPHX_CANDIDATE_SELECTOR_WIN"
        verdict_text = "TGraphX outperforms the strongest classical baseline at AP50 or AP75."
    elif delta_ap50 > -0.005 or delta_ap75 > -0.005:
        verdict = "TGRAPHX_CANDIDATE_SELECTOR_PARTIAL_WIN"
        verdict_text = "TGraphX matches or slightly trails the best classical baseline."
    else:
        verdict = "TGRAPHX_NOT_YET_WIN"
        verdict_text = "TGraphX does not yet outperform classical baselines. See failure analysis."

    # ── Report text ───────────────────────────────────────────────────────
    boot_text = "\n".join(boot_lines) if boot_lines else "  (Bootstrap results in per-seed JSON files.)"

    report = f"""# TGraphX Candidate Node Selector — Final Report

## 1. Problem Statement

**TGraphX performs object-level candidate-node classification.**
For each object hypothesis (detection cluster), candidate boxes produced
by multiple detectors and fusion methods are represented as visual
crop-tensor nodes in a graph. TGraphX uses tensor-aware graph message
passing to select the best candidate node for that object.

This is NOT source routing, learned WBF, box regression, anchor override,
segmentation, or image-level detection from scratch.

## 2. Dataset and Setup

| Item               | Value                              |
|--------------------|-------------------------------------|
| Dataset            | {cfg.get("dataset", {}).get("name", "voc2007")} (class filter: {class_names}) |
| Images             | {n_train} train / {n_val} val / {n_test} test object graphs |
| Total object graphs| {total_graphs} (one per detection cluster) |
| Detectors          | {", ".join(detector_names)} |
| Crop size          | {crop_size}×{crop_size} |
| Class agnostic     | {cfg.get("evaluation", {}).get("class_agnostic", True)} |

## 3. Method: TGraphXCandidateNodeSelector

For each detection cluster, one small graph is built:
- **Proposal nodes**: one per detector (highest-score proposal in this cluster),
  each carrying a [3, {crop_size}, {crop_size}] crop tensor.
- **WBF node**: weighted-box-average box crop (cluster node).
- **Union node**: union-box crop (consensus node).
- **NMS node**: highest-score proposal crop.
- **Soft-NMS node**: Gaussian-decay pick crop.
- **BestProposal node**: highest-score distinct token.

TGraphX applies tensor-aware ConvMP over these crop nodes (crop tensors
preserved through message passing, NOT flattened before MP).
Per-node heads output `selection_logit`. At inference:
```
selected_node = argmax(selection_logit)
selected_box  = node_box[selected_node]   # exactly one candidate box
```

## 4. Ablation Table

| Variant                    | Description                                   |
|----------------------------|-----------------------------------------------|
| `crop_metadata_mp`         | Full TGraphX: spatial crop ConvMP + metadata  |
| `flat_crop_mp`             | Crops flattened BEFORE MP (no spatial in MP)  |
| `crop_no_mp`               | CNN + metadata, no message passing at all     |
| `metadata_only`            | Metadata MLP only, no crop tensors            |

## 5. Results (multi-seed, test split)

{table}

## 6. Statistical Comparison (Paired Bootstrap, AP50)

{boot_text}

## 7. Baseline Comparison

TGraphX AP50: **{tgx_ap50:.4f}** | NMS baseline: {nms_ap50:.4f} | WBF baseline: {wbf_ap50:.4f}
TGraphX AP75: **{tgx_ap75:.4f}** | NMS baseline: {nms_ap75:.4f} | WBF baseline: {wbf_ap75:.4f}

Δ AP50 vs best classical baseline: {delta_ap50:+.4f}
Δ AP75 vs best classical baseline: {delta_ap75:+.4f}

## 8. Verdict

**{verdict}**

{verdict_text}

## 9. Scientific Conclusion

The experiment tests whether TGraphX tensor-aware message passing over
visual crop nodes — representing candidate detections for the same object —
can select a better detection box than classical fusion methods (NMS, WBF).

The main model is `TGraphXCandidateNodeSelector` with `feature_mode=crop_metadata_mp`.
The selected box is always exactly one of the pre-computed candidate boxes;
no box regression is performed.

## 10. Code Audit Status

See `reports/OBJECT_LEVEL_NODE_CLASSIFICATION_AUDIT.md` for the full
10-question audit and fixes applied.
"""

    out_path = reports_dir / "CANDIDATE_NODE_SELECTOR_REPORT.md"
    out_path.write_text(report)
    summary_out = {
        "verdict": verdict,
        "method_means": method_means,
        "tgx_ap50": tgx_ap50,
        "tgx_ap75": tgx_ap75,
        "delta_ap50_vs_best_baseline": delta_ap50,
        "delta_ap75_vs_best_baseline": delta_ap75,
        "detector_names": detector_names,
        "class_names": class_names,
    }
    (run_dir / "candidate_report.json").write_text(json.dumps(summary_out, indent=2, default=str))
    print(f"[report-cns] Verdict: {verdict}")
    print(f"  TGX AP50={tgx_ap50:.4f}  AP75={tgx_ap75:.4f}  Δ={delta_ap50:+.4f}")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
