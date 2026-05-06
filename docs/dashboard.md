# Dashboard

TGraphX ships a local, read-only monitoring dashboard — **off by default**.
No files are written, no ports are opened, and no background threads are
started unless you explicitly call `launch_dashboard` or run the CLI.

## Quickstart

```bash
# After a training run that wrote runs/demo/metrics.csv
tgraphx-dashboard --logdir runs/demo
# → http://127.0.0.1:8765
```

```python
# Python API — blocking
from tgraphx.dashboard import launch_dashboard
launch_dashboard("runs/demo")

# Non-blocking background thread (use during training)
from tgraphx.dashboard import launch_dashboard_background
server = launch_dashboard_background("runs/demo", port=8765)
# ... training loop (dashboard updates automatically) ...
server.shutdown()
```

## LAN / multi-device access

```bash
# Requires an explicit token — refused at startup without one
tgraphx-dashboard --logdir runs/demo \
  --host 0.0.0.0 --port 8765 --token MY_SECRET_TOKEN
```

Each browser shows times in its own local timezone (UTC stored, JS converts).

## What the dashboard reads

| File | Content |
|---|---|
| `{logdir}/metrics.csv` | Training metrics (epoch, loss, accuracy, timestamp…) |
| `{logdir}/run_metadata.json` | Run name, status, total epochs, device, task |
| `{logdir}/graph_metadata.json` | Graph summary + optional edge_index for preview |

**Nothing is written by the dashboard itself.** Write these files from your
training loop using `CSVLogger` or manually.

## metrics.csv schema

```
epoch,step,train_loss,val_loss,accuracy,learning_rate,timestamp
1,50,0.82,0.91,0.56,0.001,2025-01-01T12:00:00+00:00
```

Timestamps must be ISO-8601 UTC. All columns are optional except at least
one x-axis column (`epoch` or `step`). The dashboard auto-detects columns.

## Dashboard sections

| Section | Contents |
|---|---|
| **Overview** | Status chip, epoch progress, live loss, elapsed / ETA, device |
| **Metrics** | SVG line charts for every logged column; optional EMA smoothing |
| **Graph** | Graph summary, degree distribution, SVG preview for small graphs |
| **Hardware** | CPU/RAM/GPU (requires `psutil` / `pynvml`) |
| **Logs** | Last 50 metric rows as a table |
| **Config** | `run_metadata.json` as formatted JSON |
| **TV mode** | Full-screen large-font view for passive monitoring |

## Graph visualization

- Full SVG preview: ≤ 200 nodes AND ≤ 1 000 edges.
  Include `edge_index` in `graph_metadata.json` to enable.
- Larger graphs: summary + degree histogram only; `edge_index` is stripped
  server-side before sending.
- Grid graph layout rendered from `builder_params` metadata (no `edge_index` needed).
- **Full `edge_index` logging is opt-in** — include it in `graph_metadata.json`
  only when the graph is small enough.

## API endpoints

All endpoints return JSON and are read-only:

| Endpoint | Returns |
|---|---|
| `GET /api/status` | Run name, status, epoch, timestamps, device |
| `GET /api/metrics` | `{headers, rows}` from `metrics.csv` (mtime-cached) |
| `GET /api/hardware` | Python/PyTorch versions, CPU/RAM/GPU stats |
| `GET /api/metadata` | Raw `run_metadata.json` dict |
| `GET /api/graph` | Graph summary ± edge_index |

## Performance

- The dashboard polls every 2 seconds from the browser (not the training process).
- `/api/metrics` uses mtime/size caching — the CSV is re-parsed only when it changes.
- Incremental / tail-read for very large metrics files is not yet implemented;
  the full file is read on cache miss.
- No hardware polling happens in the training hot path.

## Security

| Scenario | Token required |
|---|---|
| `--host 127.0.0.1` (default) | No |
| `--host 0.0.0.0`, connecting from localhost | No |
| `--host 0.0.0.0`, connecting from another device | **Yes** |
| Starting `--host 0.0.0.0` without `--token` | **Refused at startup** |

- Read-only: no write endpoints.
- Path traversal: all file reads validated against `--logdir` via `realpath`.
- No external scripts, fonts, or CDN assets; fully self-contained.

## Hardware monitoring (optional)

```bash
pip install "tgraphx[monitoring]"   # psutil + pynvml
```

Missing packages → those sensor cards are hidden gracefully.
Hardware monitoring never runs in the training process.

## See also

- [Training utilities](training_utilities.md)
- [Example: training_with_dashboard.py](../examples/training_with_dashboard.py)
