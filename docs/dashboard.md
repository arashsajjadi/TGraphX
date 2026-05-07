# Dashboard

TGraphX ships a local-first, read-only monitoring dashboard — **off by default**.
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

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--logdir` | required | Run log directory |
| `--host` | `127.0.0.1` | Bind host (`0.0.0.0` for LAN; requires `--token`) |
| `--port` | `8765` | HTTP port |
| `--token` | none | Auth token for LAN mode; pass `auto` to generate one |
| `--max-metric-rows` | `5000` | Cap on rows returned by `/api/metrics` |
| `--refresh-interval` | `2.0` | Browser auto-refresh interval (seconds, clamped to [0.5, 60]) |
| `--open-browser` | off | Open the local URL in the default browser after start |
| `--quiet` | off | Suppress request access logs |

## LAN / multi-device access

```bash
# Explicit LAN binding — refused at startup if --token is missing.
tgraphx-dashboard --logdir runs/demo \
  --host 0.0.0.0 --port 8765 --token MY_SECRET_TOKEN

# Auto-generated token (printed once on stdout)
tgraphx-dashboard --logdir runs/demo --host 0.0.0.0 --token auto
```

When `--host 0.0.0.0` is used the CLI also prints a best-effort LAN URL
(detected via the kernel routing table without sending any packet). If
detection fails the CLI prints an instruction instead of crashing.

The dashboard's **Tools** section also shows copyable Local / LAN URLs.

## Activation contract

| Phase | Side effect |
|---|---|
| `import tgraphx` | None |
| `import tgraphx.dashboard` | Loads two small modules; **no server, no files, no hardware probes** |
| `launch_dashboard(...)` / CLI | Starts the HTTP server, blocks |
| `launch_dashboard_background(...)` | Starts the server in a daemon thread; call `.shutdown()` to stop |
| Training without dashboard kwarg | Zero overhead — no polling, no file watchers |

The dashboard is intentionally **not** auto-started by `fit()` or other training
utilities.

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
| **Metrics** | SVG line charts for every logged column; window selector; smoothing toggle; per-chart CSV/SVG export |
| **Graph** | Graph summary, degree distribution, SVG preview for small graphs |
| **Hardware** | CPU/RAM/GPU/MPS panels with explicit reasons when sensors are unavailable |
| **Logs** | Last 50 metric rows as a table; CSV / Print buttons |
| **Config** | `run_metadata.json` rendered as plain text (never as HTML) |
| **Tools** | Copy URL, export buttons, refresh-interval status |
| **About** | TGraphX/Python/PyTorch versions and copy-pasteable usage block |
| **TV mode** | Full-screen large-font view for passive monitoring |

## Charts

- Each metric gets its own SVG line chart with the latest value displayed
  in a pill next to the title.
- A **window selector** in the Metrics toolbar limits charts to the latest
  100 / 500 / 1000 rows or "All" — useful for very long runs.
- A **smoothing toggle** (off by default) applies an exponential moving
  average; the α value is shown next to the toggle.
- A **stale-data banner** appears below the top bar when the dashboard
  has not received fresh data for a configurable threshold (default 30 s).
- Missing / NaN / Inf values are filtered out and disclosed (the chart
  shows "No finite values" rather than producing a misleading curve).

## Color palette and accessibility

| Feature | How |
|---|---|
| Color-blind safe palette (Okabe-Ito) | Top-bar **palette** button (`●` icon) — persists in `localStorage` |
| Dark / light theme | Top-bar moon/sun button — persists; honours OS `prefers-color-scheme` |
| Keyboard navigation | Tab through topbar, sidebar, toolbars, exports |
| Focus indicator | High-contrast ring via `:focus-visible` |
| Skip-to-content link | Visible on first Tab; jumps past the sidebar |
| Reduced motion | Honours OS `prefers-reduced-motion: reduce` |
| Series identification | Charts always include a label; color is never the only signal |
| ARIA | Icon-only buttons carry `aria-label`; nav uses `aria-current="page"`; status chips use `aria-live` |
| Print stylesheet | Hides nav/toolbars, switches to ink-conservative greys, page-break per section |

## Exports

All exports are **client-side**. Nothing is uploaded; no server-side files
are created.

| Action | Where | File |
|---|---|---|
| Download metrics.csv | Metrics / Logs / Tools toolbar | `tgraphx_metrics.csv` |
| Download per-chart data | "CSV" button next to each chart | `tgraphx_chart_<metric>.csv` |
| Download per-chart SVG | "SVG" button next to each chart | `tgraphx_chart_<metric>.svg` |
| Save / Print as PDF | "Print / PDF" button → browser print dialog | (browser file picker) |
| Copy local / LAN URL | Tools section | (clipboard) |

## Performance and overhead

- **Dashboard disabled (default):** zero overhead. No timers, no file
  watchers, no hardware probes, no threads.
- **Dashboard enabled:** the server uses one daemon thread (or the main
  thread in blocking mode). The browser polls every `--refresh-interval`
  seconds (default 2 s).
- `/api/metrics` uses an mtime/size cache: the CSV is re-parsed only when
  it changes.
- `/api/hardware` is cached for ~1.5 s; `pynvml.nvmlInit()` is called at
  most once per process; `psutil.cpu_percent` is non-blocking after the
  first warm-up call.
- `--max-metric-rows` caps the response so the browser DOM never explodes
  on multi-million-row logs (the latest N rows are returned with a
  truncation disclosure).
- The Pause button in the top bar suspends polling immediately; visibility
  changes (browser tab not focused) also suspend the timer.

## Incremental metrics updates

After the first full metrics load the browser uses
`GET /api/metrics?since_row=<n>` to request only rows added since the last
poll.  New rows are appended to the in-memory buffer without replacing the
existing chart data.  This dramatically reduces payload size during long runs.

If the file is replaced or truncated (log rotation), the server returns
`reset_required: true` and the browser reloads from scratch.

Note: the server still parses the full CSV internally on cache miss; the
incremental response is a slice of that parse.  True byte-seek tail-reading
remains a future improvement.

## Multi-run mode

If `--logdir` points to a **parent directory** containing immediate child
directories each with their own `metrics.csv`, the dashboard detects
multi-run mode automatically:

- A **run-selector dropdown** appears below the top bar.
- The active run's metrics are loaded by name via `?run=<name>`.
- Run names are strictly validated (no path traversal, no slash, basename
  only).  Invalid names return HTTP 400.
- Up to 50 runs are listed (configurable in source).

Single-run mode (logdir has its own `metrics.csv`) is always preferred over
multi-run.

## Graph statistics from `graph_stats.json`

Write precomputed graph statistics alongside your log files and the
dashboard will display them in the Graph section even if you have no
`graph_metadata.json`:

```python
from tgraphx import write_graph_stats, Graph

g = Graph(node_features, edge_index)
write_graph_stats(g, "runs/demo/graph_stats.json")
# or with a pre-computed dict:
write_graph_stats(
    {"num_nodes": 100, "num_edges": 400, "directed": False,
     "avg_degree": 4.0, "density": 0.04, "connected_components": 1},
    "runs/demo/graph_stats.json",
)
```

Supported fields (all optional): `num_nodes`, `num_edges`, `directed`,
`self_loops`, `avg_degree`, `min_degree`, `max_degree`, `density`,
`connected_components`, `isolated_nodes`.

Missing fields are shown as "not reported" — no invented values.

## Offline HTML snapshot export

Export a complete, self-contained dashboard snapshot that opens in any
browser without a running server:

```bash
# Via CLI (no server started)
tgraphx-dashboard --logdir runs/demo --export-html snapshot.html

# Via Python API
from tgraphx.dashboard.app import export_dashboard_html
export_dashboard_html("runs/demo", "snapshot.html")
```

The exported file:

- Inlines CSS and JS from the installed package.
- Embeds metrics, metadata, and graph data as a JSON literal
  (`window.__TGXSNAP`).
- Does **not** include any token or credential.
- Does **not** reference external CDN or URLs.
- Does **not** use `eval()` or `new Function()`.
- Uses `<\/script>` escaping to prevent premature script-block termination.
- Displays a warning comment if the file exceeds 512 KB.
- Produces a read-only snapshot — no live polling in the offline file.

## API endpoints

All endpoints return JSON and are read-only:

| Endpoint | Returns |
|---|---|
| `GET /api/status` | Run name, status, epoch, timestamps, device |
| `GET /api/metrics` | Full: `{headers, rows, total_row_count, truncated, max_rows}` |
| `GET /api/metrics?since_row=N` | Incremental: rows after index N; includes `reset_required` |
| `GET /api/metrics?run=<name>` | Metrics for a named child run (multi-run mode) |
| `GET /api/runs` | `{mode: "single"\|"multi", runs: [...], capped: bool}` |
| `GET /api/hardware` | Versions + sensors + `cached_age_s`, `collected_at`, `unavailable_reason_*`, `gpu_power_w`, `gpu_power_limit_w`, `gpu_thermal_status` |
| `GET /api/metadata` | Raw `run_metadata.json` dict |
| `GET /api/graph` | Graph summary ± edge_index |
| `GET /api/graph_stats` | Precomputed stats from `graph_stats.json`; `{available: false}` if missing |
| `GET /api/config` | Public client-facing config — **never the token value** |

## Hardware monitoring (optional)

```bash
pip install "tgraphx[monitoring]"   # psutil + pynvml
```

Missing packages → those sensor cards show a compact "unavailable" row
with a reason that distinguishes:

- **`unavailable_reason_psutil`** — psutil not installed (no CPU/RAM)
- **`unavailable_reason_pynvml`** — pynvml not installed (no GPU sensors)
- **`unavailable_reason_cuda`** — no CUDA-capable device or query failed
- **`unavailable_reason_gpu_temp`** / `_gpu_fan` / `_gpu_util` — GPU
  doesn't report this particular sensor

Hardware monitoring never runs in the training process.

## Security

| Scenario | Token required |
|---|---|
| `--host 127.0.0.1` (default) | No |
| `--host 0.0.0.0`, connecting from localhost | No |
| `--host 0.0.0.0`, connecting from another device | **Yes** |
| Starting `--host 0.0.0.0` without `--token` | **Refused at startup** |

- Read-only: no write endpoints.
- Path traversal: all file reads validated against `--logdir` via
  `realpath`.
- No external scripts, fonts, or CDN assets; fully self-contained.
- The token value is **never** echoed in JSON responses, exported files,
  or HTML — `/api/config` exposes only `has_token: true|false`.
- All user-controlled strings (run name, builder name, metric column
  names, run_metadata values) are HTML-escaped or rendered via
  `textContent` before display.
- The HTML shell sets `<meta name="referrer" content="no-referrer">` so
  the dashboard URL (and any token query string) is never leaked via the
  Referer header.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Dashboard pages are blank | Static CSS/JS missing — verify wheel install: `python -c "import tgraphx, os, pathlib; print(pathlib.Path(tgraphx.__file__).parent / 'dashboard/static')"` |
| "No metric data yet" persists | Your training loop has not written `metrics.csv` yet. Check `--logdir` points at the right directory |
| GPU panel says "unavailable" | Install monitoring extras: `pip install "tgraphx[monitoring]"`. If pynvml is installed but the GPU is non-NVIDIA, the dashboard cannot read its sensors |
| Fan / temperature missing on a CUDA GPU | Datacenter cards and many laptops do not expose those sensors via NVML — the dashboard reflects this honestly with a per-row reason |
| LAN URL not reachable | Firewall is blocking the port; or your machine is on a network that isolates clients (some hotel/coffee-shop Wi-Fi). Try wired/hotspot |
| Token rejected | Confirm you copied the token without trailing whitespace; the server expects either `?token=…` or `Authorization: Bearer …` |
| Large metrics.csv slow to load | Lower `--max-metric-rows` or use the Window selector in the Metrics page |
| Stale-data banner stays on | Server crashed or process killed — check the terminal where you launched `tgraphx-dashboard` |
| Colab dashboard URL broken | Colab does not allow direct port binding from a hosted notebook; run TGraphX dashboard locally and point it at downloaded log files |

## Manual visual QA checklist

The dashboard cannot fully verify visual presentation in unit tests. Use
this checklist before each release:

- [ ] **Mobile (375 × 667 px)** — sidebar collapsed; topbar fits; cards
      stack vertically; no horizontal page scroll.
- [ ] **Tablet (768 × 1024 px)** — two-column card grid; toolbar wraps cleanly.
- [ ] **Desktop (1366 × 768 px)** — four-column overview; charts side-by-side.
- [ ] **Large monitor (1920 × 1080 +)** — increased card padding; auto-fill metrics grid uses 520 px columns.
- [ ] **TV mode** — large numbers, no clipping at 1080p / 4K.
- [ ] **Light mode** — readable contrast; chart axis ticks not too pale.
- [ ] **Dark mode** — readable contrast; chart polylines distinct.
- [ ] **Color-blind palette** — toggle activates Okabe-Ito; series remain distinguishable in deuteranopia simulation.
- [ ] **Keyboard only** — Tab order traverses topbar → nav → main; first Tab reveals "Skip to main content".
- [ ] **Print preview** — sidebar/toolbars hidden; charts visible; one section per page where reasonable.
- [ ] **Hardware panel** — without monitoring extras: no empty boxes, all rows show a reason.
- [ ] **Empty logdir** — "No metric data yet" shown; no crash; no broken charts.
- [ ] **Stale data** — kill the server with Ctrl-C while a browser is open; banner appears within ~30 s.
- [ ] **Long run name / file path** — overflows wrap, no horizontal page scroll.
- [ ] **Tooltip-free hover** — cards and chart points should still feel responsive without hover tooltips (deferred feature).
- [ ] **Copy URL** — copy button gives visible "Copied!" feedback.
- [ ] **CSV / SVG export** — files appear in Downloads with safe `tgraphx_*` names.

## Hardware GPU power and thermal fields

When `pynvml` is installed, the Hardware panel also shows:

- **Power draw** (watts) with a progress bar relative to the configured power limit.
- **Power limit** (watts, enforced).
- **Thermal status** — a compact chip with a text label (not color-only):
  - `✓ normal` — below 70 °C
  - `▲ warm` — 70–85 °C
  - `⚠ near-throttle` — above 85 °C
  - `unknown` — sensor not reported

If a field is unavailable (GPU doesn't report it, or pynvml is absent) a
compact reason is shown per-row, not an empty card.

## Chart hover tooltip

Hovering over a chart area shows a small tooltip with the epoch/step value
and the metric value at the nearest data point.  The tooltip is:

- Dependency-free (pure JS pointer events).
- Positioned to avoid viewport overflow.
- Visual-only (not keyboard accessible — the latest-value pill on each
  card provides the same information).
- Not shown on touch-primary devices (touch `pointermove` is unreliable).

## Intentionally deferred features

The following ideas were considered and deferred to keep the dashboard
lightweight, dependency-free, and training-safe:

| Feature | Why deferred |
|---|---|
| Sigma.js / D3 / uPlot | External CDN dependency — against the local-first principle |
| Gradient-flow monitor | Requires training hooks; would slow training; separate module |
| Dead-neuron monitor | Same as above |
| Adversarial robustness testing | Large scope; separate library |
| PyTorch profiler flamegraphs | Heavy infra; not dashboard-appropriate |
| Custom formula overlay | Complex, risk of eval-based injection |
| umap-js latent-space | External JS library |
| Memorization tracker | Requires training hooks + large data |
| Feature correlation heatmap | Would need scipy/numpy on server |
| OOD detection monitor | Requires model hooks |
| PCIe / page-fault profiler | OS-dependent; psutil does not expose this portably |
| Git/pip freeze tracking | Out of scope for a logging dashboard |
| Hyperparameter parallel coordinates | No standard schema yet |
| Run replay animation | Complex; low priority |
| Multi-run overlay charts | Implemented only as selector; overlay is complex and noisy |
| True byte-seek tail-read | Safe to add later but not necessary with mtime cache |
| Training-side `fit(dashboard=True)` | Lifecycle/threading risk; use `launch_dashboard_background` instead |

## Limitations

- **Not a TensorBoard replacement** for histograms, embeddings, or hyperparameter sweeps.
- **No remote/cloud monitoring.** Local-first by design.
- **Incremental parsing** still reads the full CSV on cache miss; the response is a slice.  True seek-based tail-read is future work.
- **No automatic dashboard launch** from `fit()` — start explicitly.
- **Hover tooltip** is visual-only (mouse); the latest-value pill covers the keyboard/screen-reader case.
- **Multi-run comparison overlay** is deferred; the run selector shows one run at a time.

## See also

- [Training utilities](training_utilities.md)
- [Example: training_with_dashboard.py](../examples/training_with_dashboard.py)
- [Performance overhead model](performance.md)
