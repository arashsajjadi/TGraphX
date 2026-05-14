"""Step 06: Generate report from metrics/traces only.

Reads:  {run_dir}/metrics_seed*.json
        {run_dir}/graph_audit.json
        {run_dir}/dataset_inventory.json
Writes: {run_dir}/report.md

Verdict logic (Part 9):
  - Compares TGraphX vs the STRONGEST validation-selected baseline on the
    same test split.
  - Uses the paired bootstrap stored in each seed's metrics_seedN.json
    (paired_bootstrap_vs_baselines).
  - Never uses fixed AP thresholds for the verdict label.

Does NOT train or evaluate. Does NOT call run_pipeline.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


WIN_P = 0.95
TIE_P = 0.85


def _aggregate_method_aps(seeds_data):
    """Return {method_name: [per-seed headline AP]}."""
    out = {}
    for d in seeds_data:
        bm = d.get("baseline_methods", {})
        for name, vals in bm.items():
            out.setdefault(name, []).append(vals.get("headline_ap",
                vals.get("test_ap_class_agnostic", 0.0)))
    return out


def _strongest_baseline(method_aps, exclude=("fusion::tgraphx",)):
    """Pick the baseline with the highest mean AP across seeds."""
    cand = [(n, statistics.mean(v)) for n, v in method_aps.items()
            if n not in exclude and v]
    if not cand:
        return None, 0.0
    cand.sort(key=lambda x: -x[1])
    return cand[0]


def _aggregate_bootstrap(seeds_data, baseline_name):
    """Combine per-seed paired bootstraps into a summary.

    We do NOT pool the per-image AP vectors across seeds (different image
    subsets per seed via different test splits). Instead we report the
    fraction-of-seeds where p_a_gt_b >= WIN_P, and average mean_diff.
    """
    p_vals, mds, los, his = [], [], [], []
    for d in seeds_data:
        b = d.get("paired_bootstrap_vs_baselines", {}).get(baseline_name)
        if not b:
            continue
        p_vals.append(b.get("p_a_gt_b", 0.5))
        mds.append(b.get("mean_diff", 0.0))
        los.append(b.get("ci95_low", 0.0))
        his.append(b.get("ci95_high", 0.0))
    if not p_vals:
        return None
    return {
        "n_seeds": len(p_vals),
        "p_a_gt_b_mean": statistics.mean(p_vals),
        "p_a_gt_b_min": min(p_vals),
        "p_a_gt_b_max": max(p_vals),
        "mean_diff_mean": statistics.mean(mds),
        "ci95_low_mean": statistics.mean(los),
        "ci95_high_mean": statistics.mean(his),
        "seeds_pass_win_threshold": sum(1 for p in p_vals if p >= WIN_P),
    }


def _verdict_label(boot_summary, mean_tgx, mean_strongest, is_synthetic, is_multiclass):
    if is_synthetic:
        return f"**SYNTHETIC_CONTROLLED_ROUTING_WIN** — synthetic jitter benchmark, mean AP={mean_tgx:.4f}. Not a real detection claim."
    if boot_summary is None:
        # No baselines on disk — fall back to "not yet" with explanation.
        return (f"**STILL_NOT_READY_FOR_REAL_CLAIM** — no paired bootstrap "
                f"available (rerun Step 05 with --force after the new evaluator).")
    p_mean = boot_summary["p_a_gt_b_mean"]
    p_min = boot_summary["p_a_gt_b_min"]
    md_mean = boot_summary["mean_diff_mean"]
    lo = boot_summary["ci95_low_mean"]
    hi = boot_summary["ci95_high_mean"]
    fraction_passing = boot_summary["seeds_pass_win_threshold"] / max(1, boot_summary["n_seeds"])

    head = "REAL_VOC_MULTI_CLASS" if is_multiclass else "REAL_VOC_CAR"
    if p_mean >= WIN_P and md_mean > 0.0 and fraction_passing >= 0.5:
        return (f"**{head}_WIN** — mean AP={mean_tgx:.4f} > strongest baseline {mean_strongest:.4f} "
                f"(mean_diff={md_mean:+.4f}, mean P(TGraphX>baseline)={p_mean:.3f}, "
                f"{boot_summary['seeds_pass_win_threshold']}/{boot_summary['n_seeds']} seeds clear p≥{WIN_P}).")
    if (TIE_P <= p_mean < WIN_P) or (lo <= 0 <= hi):
        return (f"**{head}_SAFE_TIE** — mean AP={mean_tgx:.4f} vs strongest baseline {mean_strongest:.4f} "
                f"(mean_diff={md_mean:+.4f}, mean P={p_mean:.3f} in tie band [{TIE_P},{WIN_P})).")
    return (f"**{head}_NOT_YET_WIN** — mean AP={mean_tgx:.4f} vs strongest baseline "
            f"{mean_strongest:.4f}; mean P(TGraphX>baseline)={p_mean:.3f} below tie band.")


def main():
    parser = argparse.ArgumentParser(description="Step 06: generate report")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_report = run_dir / "report.md"
    if out_report.exists() and not args.force:
        print(f"[06] Report exists: {out_report}  (--force to rerun)")
        return

    metric_files = sorted(run_dir.glob("metrics_seed*.json"))
    if not metric_files:
        raise FileNotFoundError(f"[06] No metrics_seed*.json found in {run_dir}")
    seeds_data = [json.loads(f.read_text()) for f in metric_files]
    audit = json.loads((run_dir / "graph_audit.json").read_text()) if (run_dir / "graph_audit.json").exists() else {}
    inv   = json.loads((run_dir / "dataset_inventory.json").read_text()) if (run_dir / "dataset_inventory.json").exists() else {}
    manifest = json.loads((run_dir / "split_manifest.json").read_text()) if (run_dir / "split_manifest.json").exists() else {}

    dataset_name = inv.get("source", manifest.get("config", "unknown"))
    is_synthetic = "synthetic" in str(dataset_name).lower() or "jitter" in str(dataset_name).lower()
    n_classes = manifest.get("num_classes", len(inv.get("class_names", [])))
    is_multiclass = n_classes > 2

    headline_aps = [d.get("test_metrics_selected_mode", {}).get("headline_ap", d.get("test_ap", 0))
                    for d in seeds_data]
    mean_ap = statistics.mean(headline_aps) if headline_aps else 0.0
    std_ap = statistics.stdev(headline_aps) if len(headline_aps) > 1 else 0.0

    method_aps = _aggregate_method_aps(seeds_data)
    method_aps["fusion::tgraphx"] = headline_aps   # ensure present
    strongest_name, strongest_mean = _strongest_baseline(method_aps)
    boot = _aggregate_bootstrap(seeds_data, strongest_name) if strongest_name else None

    verdict = _verdict_label(boot, mean_ap, strongest_mean, is_synthetic, is_multiclass)

    lines = [
        "# TGraphX Run Report",
        "",
        f"**Run dir:** `{run_dir}`",
        f"**Dataset:** {inv.get('num_records', '?')} images, {n_classes} classes",
        f"**Experiment type:** {'SYNTHETIC' if is_synthetic else 'REAL VOC'}",
        f"**Seeds:** {len(seeds_data)}",
        "",
        "## Results — TGraphX",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Mean AP (headline) | {mean_ap:.4f} |",
        f"| Std AP             | {std_ap:.4f} |",
        f"| Min / Max          | {min(headline_aps):.4f} / {max(headline_aps):.4f} |",
        "",
        "Headline = class-aware AP for multi-class, class-agnostic AP for single-class.",
        "",
        "## Baseline comparison (same test split)",
        "",
        "| Method | Mean headline AP |",
        "|--------|-----------------:|",
    ]
    for name, aps in sorted(method_aps.items(), key=lambda kv: -statistics.mean(kv[1])):
        if not aps:
            continue
        lines.append(f"| {name} | {statistics.mean(aps):.4f} |")
    lines += [
        "",
        f"**Strongest baseline:** {strongest_name or '—'} (mean AP {strongest_mean:.4f}).",
        "",
        "## Paired bootstrap (TGraphX vs strongest baseline)",
        "",
    ]
    if boot:
        lines += [
            "| Statistic | Value |",
            "|-----------|------:|",
            f"| n seeds in summary | {boot['n_seeds']} |",
            f"| mean P(TGraphX > baseline) | {boot['p_a_gt_b_mean']:.3f} |",
            f"| min / max per-seed P       | {boot['p_a_gt_b_min']:.3f} / {boot['p_a_gt_b_max']:.3f} |",
            f"| mean Δ AP                  | {boot['mean_diff_mean']:+.4f} |",
            f"| mean 95% CI (Δ AP)         | [{boot['ci95_low_mean']:+.4f}, {boot['ci95_high_mean']:+.4f}] |",
            f"| seeds clearing P≥{WIN_P}    | {boot['seeds_pass_win_threshold']} / {boot['n_seeds']} |",
            "",
        ]
    else:
        lines += ["_(paired bootstrap not available — re-run Step 05 with --force)_", ""]

    lines += [
        "## Score mode",
        "",
        f"Selected on **validation split only** ({seeds_data[0].get('score_mode_selection_metric','?')}).",
        f"Modes used across seeds: {sorted(set(d.get('selected_score_mode', '?') for d in seeds_data))}",
        "",
        "## Graph audit",
        "",
        f"- Avg nodes/graph: {audit.get('avg_nodes', '?')}",
        f"- Avg edges/graph: {audit.get('avg_edges', '?')}",
        f"- Detector names: {audit.get('detector_names', '?')}",
        "",
        "## Verdict",
        "",
        verdict,
    ]
    out_report.write_text("\n".join(lines) + "\n")
    print(f"[06] Report → {out_report}")
    print(f"  TGraphX mean AP {mean_ap:.4f}  vs  strongest baseline {strongest_mean:.4f}")
    if boot:
        print(f"  paired bootstrap: P(TGraphX>{strongest_name})={boot['p_a_gt_b_mean']:.3f} "
              f"Δ={boot['mean_diff_mean']:+.4f}")


if __name__ == "__main__":
    main()
