"""training_with_dashboard.py

Simulates a complete training run that writes files compatible with
the TGraphX dashboard.  Does NOT auto-start the dashboard.

After running this script, launch the dashboard with:
    tgraphx-dashboard --logdir runs/demo

Or from Python:
    from tgraphx.dashboard import launch_dashboard
    launch_dashboard("runs/demo")
"""
from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    logdir = "runs/demo"
    os.makedirs(logdir, exist_ok=True)

    # ── Write run metadata ────────────────────────────────────────────────────
    run_meta = {
        "run_name":     "demo",
        "task":         "graph_classification",
        "model":        "TGraphX GAT (2-head, 3×3 grid graph)",
        "layer":        "gat",
        "device":       "cpu",
        "total_epochs": 20,
        "start_time":   utc_now(),
        "status":       "running",
    }
    with open(os.path.join(logdir, "run_metadata.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    # ── Write graph metadata (3×3 grid from build_grid_graph) ────────────────
    # Build a small 3×3 grid edge_index in Python (no torch needed in example)
    srcs, dsts = [], []
    rows, cols = 3, 3
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                u, v = r*cols+c, r*cols+c+1
                srcs += [u, v]; dsts += [v, u]
            if r + 1 < rows:
                u, v = r*cols+c, (r+1)*cols+c
                srcs += [u, v]; dsts += [v, u]
    for i in range(rows*cols):
        srcs.append(i); dsts.append(i)

    deg = {}
    for d in dsts:
        deg[d] = deg.get(d, 0) + 1
    all_deg = list(deg.values())

    graph_meta = {
        "num_nodes":    rows * cols,
        "num_edges":    len(srcs),
        "directed":     False,
        "self_loops":   True,
        "builder":      "build_grid_graph",
        "builder_params": {"rows": rows, "cols": cols},
        "degree_stats": {
            "mean":      sum(all_deg) / len(all_deg),
            "min":       min(all_deg),
            "max":       max(all_deg),
            "histogram": [0, 0, 4, 4, 1],  # counts at degree 0,1,2,3,4
        },
        "edge_index": [srcs, dsts],  # small enough for full render
    }
    with open(os.path.join(logdir, "graph_metadata.json"), "w") as f:
        json.dump(graph_meta, f, indent=2)

    # ── Simulate training epochs ──────────────────────────────────────────────
    fields = ["epoch", "step", "train_loss", "val_loss", "accuracy",
              "learning_rate", "grad_norm", "timestamp"]

    print(f"Simulating 20-epoch training run → {logdir}/")
    print()

    with open(os.path.join(logdir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for epoch in range(1, 21):
            t = epoch / 20
            train_loss = 1.4 * math.exp(-3.2 * t) + 0.08 + 0.04 * math.sin(t * 9)
            val_loss   = 1.4 * math.exp(-2.7 * t) + 0.10 + 0.03 * math.sin(t * 7)
            accuracy   = 0.48 + 0.48 * (1 - math.exp(-4.2 * t)) + 0.015 * math.sin(t * 6)
            lr         = 0.001 * (0.97 ** epoch)
            grad_norm  = 1.2 * math.exp(-1.5 * t) + 0.1 + 0.05 * math.sin(t * 15)

            row = {
                "epoch":         epoch,
                "step":          epoch * 50,
                "train_loss":    round(train_loss, 5),
                "val_loss":      round(val_loss,   5),
                "accuracy":      round(accuracy,   4),
                "learning_rate": round(lr,          7),
                "grad_norm":     round(grad_norm,   4),
                "timestamp":     utc_now(),
            }
            writer.writerow(row)
            f.flush()

            print(f"  Epoch {epoch:2d}/20  loss={train_loss:.4f}  "
                  f"val={val_loss:.4f}  acc={accuracy:.3f}  lr={lr:.6f}")

    # ── Update status to completed ────────────────────────────────────────────
    run_meta["status"]   = "completed"
    run_meta["end_time"] = utc_now()
    with open(os.path.join(logdir, "run_metadata.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    print()
    print("─" * 60)
    print("Training complete!")
    print()
    print("Launch the dashboard:")
    print(f"  tgraphx-dashboard --logdir {logdir}")
    print()
    print("Or from Python:")
    print(f"  from tgraphx.dashboard import launch_dashboard")
    print(f"  launch_dashboard('{logdir}')")
    print()
    print("For LAN access (requires a token):")
    print(f"  tgraphx-dashboard --logdir {logdir} \\")
    print(f"    --host 0.0.0.0 --port 8765 --token MY_SECRET_TOKEN")
    print()
    print("Files written:")
    for name in ("metrics.csv", "run_metadata.json", "graph_metadata.json"):
        path = os.path.join(logdir, name)
        size = os.path.getsize(path)
        print(f"  {path}  ({size} bytes)")


if __name__ == "__main__":
    main()
