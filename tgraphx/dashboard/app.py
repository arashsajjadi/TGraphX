"""TGraphX Dashboard HTTP server.

Provides a read-only, local-first HTTP server that serves
- Static assets (HTML shell, CSS, JS)
- JSON API endpoints (/api/*)

Security
--------
* Localhost clients (127.0.0.1, ::1) are always allowed without a token.
* Non-localhost clients are only accepted in LAN mode (host != loopback).
* LAN mode without a token raises ValueError at startup.
* All logdir file reads are validated against path-traversal attacks.
* No external URLs are referenced by the served assets.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
import urllib.parse

# Default upper bound on rows returned by /api/metrics.
# Limits response size for long runs; raw metrics.csv is never modified.
_DEFAULT_MAX_METRIC_ROWS: int = 5000

# ─────────────────────────────────────────────────────────────────────────────
# Loopback detection
# ─────────────────────────────────────────────────────────────────────────────
_LOOPBACK = frozenset(("127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost"))

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# ─────────────────────────────────────────────────────────────────────────────
# HTML shell (all dynamic content is injected by dashboard.js)
# ─────────────────────────────────────────────────────────────────────────────
_HTML = b"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark light">
<meta name="referrer" content="no-referrer">
<title>TGraphX Dashboard</title>
<link rel="stylesheet" href="/static/dashboard.css">
</head>
<body>
<a class="skip-link" href="#content">Skip to main content</a>
<div id="app">
  <nav id="sidebar" aria-label="Dashboard sections">
    <div class="brand"><span class="brand-hex" aria-hidden="true">&#x2B23;</span><span class="brand-name">TGraphX</span></div>
    <ul id="nav-list" role="list"></ul>
    <button id="sidebar-close" type="button" aria-label="Close navigation menu">&#x2715;</button>
  </nav>
  <div id="main-wrap">
    <header id="topbar" role="banner">
      <button id="hamburger" type="button" aria-label="Open navigation menu" aria-controls="sidebar">&#x2630;</button>
      <span id="run-title" aria-live="polite">TGraphX Dashboard</span>
      <div id="topbar-right">
        <button id="pause-btn" type="button" class="icon-btn"
                aria-label="Pause auto-refresh"
                aria-pressed="false" title="Pause auto-refresh">&#x23F8;</button>
        <button id="refresh-btn" type="button" class="icon-btn"
                aria-label="Refresh now" title="Refresh now">&#x21BB;</button>
        <span id="status-chip" class="chip chip-unknown" role="status" aria-live="polite">&#x2014;</span>
        <span id="viewer-clock" title="Your local time" aria-label="Current local time">&#x2014;</span>
        <button id="palette-btn" type="button" class="icon-btn"
                aria-label="Toggle color-blind safe palette"
                aria-pressed="false" title="Color-blind safe palette">&#x25CF;</button>
        <button id="theme-btn" type="button" class="icon-btn"
                aria-label="Toggle dark/light theme" title="Toggle theme">&#x263D;</button>
      </div>
    </header>
    <div id="stale-banner" class="stale-banner" hidden role="alert">
      <span id="stale-text"></span>
    </div>
    <main id="content" tabindex="-1">
      <section id="sec-overview"  class="sec active" aria-labelledby="sec-overview-title"></section>
      <section id="sec-metrics"   class="sec" aria-labelledby="sec-metrics-title"></section>
      <section id="sec-graph"     class="sec" aria-labelledby="sec-graph-title"></section>
      <section id="sec-mining"    class="sec" aria-labelledby="sec-mining-title"></section>
      <section id="sec-hardware"  class="sec" aria-labelledby="sec-hardware-title"></section>
      <section id="sec-logs"      class="sec" aria-labelledby="sec-logs-title"></section>
      <section id="sec-config"    class="sec" aria-labelledby="sec-config-title"></section>
      <section id="sec-tools"     class="sec" aria-labelledby="sec-tools-title"></section>
      <section id="sec-about"     class="sec" aria-labelledby="sec-about-title"></section>
    </main>
    <footer id="footer" role="contentinfo">
      <span id="update-txt" aria-live="polite">&#x2014;</span>
      <span class="sep" aria-hidden="true">&#xB7;</span>
      <button type="button" id="tv-btn" class="link-btn"
              aria-label="Enter full-screen TV mode">TV&nbsp;mode</button>
    </footer>
  </div>
</div>
<div id="tv-overlay" hidden role="dialog" aria-modal="true" aria-label="TV mode">
  <button id="tv-exit" type="button" aria-label="Exit TV mode">&#x2715; Exit TV mode</button>
  <div id="tv-body"></div>
</div>
<script src="/static/dashboard.js"></script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Path security
# ─────────────────────────────────────────────────────────────────────────────

def _safe_path(logdir: str, filename: str) -> Optional[str]:
    """Return realpath of logdir/filename if it stays inside logdir, else None."""
    base = os.path.realpath(logdir)
    target = os.path.realpath(os.path.join(base, filename))
    if target == base or target.startswith(base + os.sep):
        return target
    return None


def _read_logfile(logdir: str, filename: str) -> Optional[str]:
    """Read a file within logdir, returning None if absent or path-unsafe."""
    path = _safe_path(logdir, filename)
    if path is None or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Metrics CSV parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_metrics(content: str) -> Dict[str, Any]:
    reader = csv.reader(io.StringIO(content))
    rows = list(reader)
    if not rows:
        return {"headers": [], "rows": []}
    headers = rows[0]
    data: list[list] = []
    for row in rows[1:]:
        parsed: list = []
        for val in row:
            try:
                parsed.append(float(val))
            except ValueError:
                parsed.append(val)
        if parsed:
            data.append(parsed)
    return {"headers": headers, "rows": data}


def _parse_metrics_bounded(content: str, max_rows: int) -> Dict[str, Any]:
    """Parse metrics CSV and bound the response to ``max_rows`` recent rows.

    The raw ``metrics.csv`` file is never modified.  If the file has more
    rows than ``max_rows``, the response includes only the most recent rows
    plus truncation metadata so the dashboard can inform the user.

    Response keys
    -------------
    ``headers``         — column names
    ``rows``            — data rows (last ``max_rows`` if truncated)
    ``total_row_count`` — total number of data rows in the file
    ``truncated``       — ``True`` if rows were dropped from the front
    ``max_rows``        — effective row limit
    """
    base = _parse_metrics(content)
    total = len(base["rows"])

    if total > max_rows:
        return {
            "headers": base["headers"],
            "rows": base["rows"][-max_rows:],
            "total_row_count": total,
            "truncated": True,
            "max_rows": max_rows,
        }
    return {
        "headers": base["headers"],
        "rows": base["rows"],
        "total_row_count": total,
        "truncated": False,
        "max_rows": max_rows,
    }


# Maximum child runs returned by /api/runs.  Avoids accidentally scanning a
# huge parent directory.
_MAX_RUNS: int = 50


def _parse_metrics_incremental(
    content: str,
    since_row: int,
    max_rows: int,
) -> Dict[str, Any]:
    """Return only rows with data-row-index > since_row.

    Internally always parses the full CSV (the mtime cache handles re-parse
    cost).  Returns a lightweight slice together with metadata so the browser
    can append without replacing its in-memory copy.

    Response keys
    -------------
    ``headers``          — column names (always included so client can verify)
    ``rows``             — only the new rows (may be empty)
    ``latest_row_index`` — index (0-based) of the last data row in the file
    ``total_row_count``  — total number of data rows in the file
    ``reset_required``   — True when the file has shrunk and the client must
                           reload from scratch
    ``truncated``        — True when total_row_count > max_rows and we are
                           not returning all data rows
    ``max_rows``         — effective row cap
    """
    base = _parse_metrics(content)
    all_rows = base["rows"]
    total = len(all_rows)
    latest = total - 1  # -1 when empty

    # Detect file truncation / log rotation: if since_row is past the end,
    # the file must have been replaced.
    reset_required = total > 0 and since_row > latest

    if reset_required or total == 0:
        new_rows: list = []
    else:
        # Slice rows strictly after since_row (0-based indexing).
        new_rows = all_rows[max(0, since_row + 1):]

    # Apply max_rows cap to the new slice only; the client trims its own buffer.
    if len(new_rows) > max_rows:
        new_rows = new_rows[-max_rows:]

    return {
        "headers": base["headers"],
        "rows": new_rows,
        "latest_row_index": latest,
        "total_row_count": total,
        "reset_required": reset_required,
        "truncated": total > max_rows,
        "max_rows": max_rows,
    }


def _list_runs(logdir: str) -> Dict[str, Any]:
    """Scan logdir for child directories that contain metrics.csv.

    Returns a dict with:
    ``mode``  — ``"single"`` if logdir/metrics.csv exists, else ``"multi"``
    ``runs``  — list of run-name strings (basenames, safe, ≤ _MAX_RUNS)
    ``capped``— True if there were more than _MAX_RUNS runs found
    """
    base = os.path.realpath(logdir)
    # Single-run mode takes priority when logdir itself has metrics.csv.
    if os.path.isfile(os.path.join(base, "metrics.csv")):
        return {"mode": "single", "runs": [], "capped": False}

    try:
        entries = sorted(os.listdir(base))
    except OSError:
        return {"mode": "single", "runs": [], "capped": False}

    runs = []
    for name in entries:
        # Allow only safe basenames: no path separators, no dots in positions
        # that would allow traversal.
        if "/" in name or "\\" in name or name in (".", ".."):
            continue
        child = os.path.join(base, name)
        if os.path.isdir(child) and os.path.isfile(os.path.join(child, "metrics.csv")):
            runs.append(name)
        if len(runs) >= _MAX_RUNS + 1:  # +1 to detect capping
            break

    capped = len(runs) > _MAX_RUNS
    return {"mode": "multi", "runs": runs[:_MAX_RUNS], "capped": capped}


def _safe_run_path(logdir: str, run_name: str) -> Optional[str]:
    """Validate a run name and return its absolute path inside logdir, or None.

    Rejects any name that would escape ``logdir`` via path traversal, empty
    names, or names containing path separators.
    """
    if not run_name or "/" in run_name or "\\" in run_name:
        return None
    # Extra guard: only allow the exact basename to reach the child.
    if run_name != os.path.basename(run_name):
        return None
    base = os.path.realpath(logdir)
    candidate = os.path.realpath(os.path.join(base, run_name))
    # Must be a direct child — not base itself and must start with base + sep.
    if candidate == base or not candidate.startswith(base + os.sep):
        return None
    return candidate


# ─────────────────────────────────────────────────────────────────────────────
# Status helpers
# ─────────────────────────────────────────────────────────────────────────────

def _status_epoch(row_dict: Dict[str, Any]) -> Optional[Any]:
    """Extract the current epoch/step from the last metrics row.

    _parse_metrics converts parseable cells to ``float``, so ``epoch=0``
    becomes ``0.0`` — falsy in Python.  A plain ``or``-chain therefore
    skips valid zero values.  This helper uses explicit ``None`` checks so
    that epoch 0 and step 0 are reported correctly.

    Priority: ``"epoch"`` column > ``"step"`` column > ``None``.
    An empty string (unparseable cell) is treated as absent.
    """
    for key in ("epoch", "step"):
        val = row_dict.get(key)
        if val is None:
            continue
        if isinstance(val, str) and not val.strip():
            continue
        return val
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Hardware / system information (best-effort)
# ─────────────────────────────────────────────────────────────────────────────

# Process-wide cache for hardware probes.  /api/hardware can be polled at
# 0.5–1 Hz from a browser; running pynvml.nvmlInit() and a 100 ms blocking
# psutil.cpu_percent() on every request would otherwise be wasteful and could
# perceptibly slow down a busy CPU.  We cache the JSON payload for a short
# window and the pynvml handle/init-state for the lifetime of the process.
_HARDWARE_CACHE: Dict[str, Any] = {
    "ts": 0.0,        # last collection wall time
    "data": None,     # last collected dict
    "pynvml_inited": False,
    "pynvml_handle": None,
    "psutil_warmed": False,
}

# How long a /api/hardware payload is reused before recollecting.  Browser
# polls every ~2 s by default, so a 1.5 s window keeps the UI fresh while
# avoiding a probe per request.
_HARDWARE_CACHE_TTL_S: float = 1.5


def _collect_hardware(force: bool = False) -> Dict[str, Any]:
    """Best-effort hardware/version snapshot.

    Cached for ``_HARDWARE_CACHE_TTL_S`` seconds in the parent process so that
    repeated /api/hardware requests do not re-init pynvml or block on
    psutil.cpu_percent.  Pass ``force=True`` to bypass the cache (e.g. tests).

    Honesty contract
    ----------------
    Every "missing" field gets an explicit ``unavailable_reason_*`` key so the
    UI can distinguish "optional dep not installed" from "sensor not reported"
    from "no GPU".  Stale values are never invented.
    """
    now = time.monotonic()
    cached = _HARDWARE_CACHE.get("data")
    if not force and cached is not None and (now - _HARDWARE_CACHE["ts"]) < _HARDWARE_CACHE_TTL_S:
        # Return the cached snapshot, but stamp it with a fresh ``cached_age_s``
        # so the UI can display "X.Y s old" without recollecting.
        snapshot = dict(cached)
        snapshot["cached_age_s"] = round(now - _HARDWARE_CACHE["ts"], 2)
        return snapshot

    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "psutil_available": False,
        "pynvml_available": False,
        "cuda_available": False,
        "mps_available": False,
        "cpu_count": os.cpu_count(),
        "collected_at": _utc_iso(),
        "cached_age_s": 0.0,
    }

    try:
        import tgraphx
        info["tgraphx"] = tgraphx.__version__
    except Exception:
        info["tgraphx"] = "unknown"

    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            try:
                dev = torch.device("cuda:0")
                info["cuda_device_name"] = torch.cuda.get_device_name(0)
                info["cuda_mem_allocated_mb"] = round(torch.cuda.memory_allocated(dev) / 1024**2, 1)
                info["cuda_mem_reserved_mb"] = round(torch.cuda.memory_reserved(dev) / 1024**2, 1)
                info["cuda_mem_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1)
            except Exception as exc:
                info["unavailable_reason_cuda"] = f"CUDA query failed: {exc}"
        else:
            info["unavailable_reason_cuda"] = "no CUDA-capable device detected"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None:
            info["mps_available"] = bool(getattr(mps, "is_available", lambda: False)())
    except Exception:
        info["torch"] = "not installed"
        info["unavailable_reason_torch"] = "PyTorch is not importable"

    try:
        import psutil
        info["psutil_available"] = True
        # First call needs interval>0 to establish a baseline; subsequent calls
        # use interval=None (non-blocking, returns the delta since last call).
        if not _HARDWARE_CACHE["psutil_warmed"]:
            psutil.cpu_percent(interval=0.1)
            _HARDWARE_CACHE["psutil_warmed"] = True
        info["cpu_percent"] = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1024**3, 2)
        info["ram_used_gb"] = round(vm.used / 1024**3, 2)
        info["ram_percent"] = round(vm.percent, 1)
        try:
            proc = psutil.Process()
            info["process_ram_mb"] = round(proc.memory_info().rss / 1024**2, 1)
        except Exception:
            pass
    except ImportError:
        info["unavailable_reason_psutil"] = (
            "psutil not installed; CPU/RAM metrics unavailable. "
            "Install with: pip install 'tgraphx[monitoring]'"
        )

    # pynvml: initialise once per process; keep the device handle around.
    try:
        import pynvml  # noqa: F401  (module-level import for type/availability)
        try:
            if not _HARDWARE_CACHE["pynvml_inited"]:
                pynvml.nvmlInit()
                _HARDWARE_CACHE["pynvml_inited"] = True
            if _HARDWARE_CACHE["pynvml_handle"] is None:
                _HARDWARE_CACHE["pynvml_handle"] = pynvml.nvmlDeviceGetHandleByIndex(0)
            handle = _HARDWARE_CACHE["pynvml_handle"]
            info["pynvml_available"] = True
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info["gpu_mem_used_mb"] = round(mem.used / 1024**2, 1)
            info["gpu_mem_total_mb"] = round(mem.total / 1024**2, 1)
            try:
                info["gpu_util_pct"] = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            except Exception:
                info["unavailable_reason_gpu_util"] = "GPU does not report utilization"
            try:
                info["gpu_temp_c"] = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                # Simple thermal status label — calms users vs. alarming colors.
                t = info["gpu_temp_c"]
                if t < 70:
                    info["gpu_thermal_status"] = "normal"
                elif t < 85:
                    info["gpu_thermal_status"] = "warm"
                else:
                    info["gpu_thermal_status"] = "near-throttle"
            except Exception:
                info["unavailable_reason_gpu_temp"] = "GPU does not report temperature"
                info["gpu_thermal_status"] = "unknown"
            try:
                info["gpu_fan_pct"] = pynvml.nvmlDeviceGetFanSpeed(handle)
            except Exception:
                # Many GPUs (datacenter cards, some laptops) don't expose fan speed.
                info["unavailable_reason_gpu_fan"] = "GPU does not report fan speed"
            try:
                # pynvml returns milliwatts; convert to watts.
                info["gpu_power_w"] = round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000, 1)
            except Exception:
                info["unavailable_reason_gpu_power"] = "GPU does not report power usage"
            try:
                info["gpu_power_limit_w"] = round(
                    pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000, 1
                )
            except Exception:
                info["unavailable_reason_gpu_power_limit"] = "GPU does not report power limit"
        except Exception as exc:
            info["unavailable_reason_pynvml"] = f"pynvml runtime error: {exc}"
    except ImportError:
        info["unavailable_reason_pynvml"] = (
            "pynvml not installed; GPU sensor metrics unavailable. "
            "Install with: pip install 'tgraphx[monitoring]'"
        )

    _HARDWARE_CACHE["ts"] = now
    _HARDWARE_CACHE["data"] = info
    return info


def _utc_iso() -> str:
    """Return current UTC time as ISO-8601 (seconds resolution)."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ─────────────────────────────────────────────────────────────────────────────
# HTTP request handler
# ─────────────────────────────────────────────────────────────────────────────

class DashboardHandler(BaseHTTPRequestHandler):
    """Read-only HTTP handler for the TGraphX dashboard."""

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _send_json(self, data: Any, code: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data: bytes, content_type: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _extract_token(self) -> Optional[str]:
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        if "token" in qs:
            return qs["token"][0]
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:].strip()
        return None

    def _check_auth(self) -> bool:
        """Return True if the request is authorized to proceed."""
        client_ip = self.client_address[0]
        # Localhost is always allowed
        if client_ip in _LOOPBACK:
            return True
        # Non-localhost clients need a token (LAN mode)
        if not self.server.token:
            self._send_json(
                {"error": "LAN access requires a token. Start the dashboard with --token."},
                403,
            )
            return False
        if self._extract_token() != self.server.token:
            self._send_json({"error": "Invalid or missing token."}, 401)
            return False
        return True

    def log_message(self, fmt: str, *args: Any) -> None:  # type: ignore[override]
        if self.server.verbose:
            super().log_message(fmt, *args)

    # ── Routing ──────────────────────────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/":
            self._send_bytes(_HTML, "text/html; charset=utf-8")
        elif path.startswith("/static/"):
            self._serve_static(path)
        elif path.startswith("/api/"):
            if not self._check_auth():
                return
            self._handle_api(path)
        else:
            self._send_json({"error": "Not found"}, 404)

    # ── Static serving ───────────────────────────────────────────────────────

    def _serve_static(self, path: str) -> None:
        fname = os.path.basename(path)
        # Whitelist only known filenames
        if fname not in ("dashboard.css", "dashboard.js"):
            self._send_json({"error": "Not found"}, 404)
            return
        fpath = os.path.join(STATIC_DIR, fname)
        if not os.path.isfile(fpath):
            self._send_json({"error": f"Static file missing: {fname}"}, 404)
            return
        ct = "text/css" if fname.endswith(".css") else "application/javascript"
        with open(fpath, "rb") as f:
            data = f.read()
        self._send_bytes(data, ct)

    # ── API handlers ─────────────────────────────────────────────────────────

    def _handle_api(self, path: str) -> None:
        logdir = self.server.logdir
        endpoint = path[len("/api/"):]

        if endpoint == "status":
            self._api_status(logdir)
        elif endpoint == "metrics":
            self._api_metrics(logdir)
        elif endpoint == "hardware":
            self._send_json(_collect_hardware())
        elif endpoint == "metadata":
            self._api_json_file(logdir, "run_metadata.json")
        elif endpoint == "graph":
            self._api_graph(logdir)
        elif endpoint == "graph_stats":
            self._api_graph_stats(logdir)
        elif endpoint == "runs":
            self._send_json(_list_runs(logdir))
        elif endpoint == "config":
            self._api_config()
        # ── Mining artifact endpoints (v0.4.2+) ────────────────────────────
        elif endpoint == "mining_summary":
            self._api_json_file(logdir, "graph_mining_summary.json")
        elif endpoint == "motif_summary":
            self._api_json_file(logdir, "motif_summary.json")
        elif endpoint == "anomaly_summary":
            self._api_json_file_capped(logdir, "anomaly_summary.json", max_list_rows=50)
        elif endpoint == "community_summary":
            self._api_json_file(logdir, "community_summary.json")
        elif endpoint == "prototype_membership":
            self._api_json_file_capped(logdir, "prototype_membership_report.json", max_list_rows=30)
        elif endpoint == "neural_mining":
            self._api_json_file(logdir, "neural_mining_report.json")
        elif endpoint == "reproducibility":
            self._api_json_file(logdir, "reproducibility_report.json")
        elif endpoint == "mining_benchmark":
            self._api_json_file_capped(logdir, "mining_benchmark_results.json", max_list_rows=50)
        elif endpoint == "link_prediction_summary":
            self._api_json_file_capped(logdir, "link_prediction_summary.json", max_list_rows=30)
        # ── v0.5.0 artifact endpoints ────────────────────────────────────────
        elif endpoint == "kg_summary":
            self._api_json_file_capped(logdir, "kg_summary.json", max_list_rows=50)
        elif endpoint == "kg_metrics":
            self._api_json_file(logdir, "kg_metrics_report.json")
        elif endpoint == "hypergraph_summary":
            self._api_json_file(logdir, "hypergraph_summary.json")
        elif endpoint == "vgae_report":
            self._api_json_file(logdir, "vgae_report.json")
        elif endpoint == "loader_summary":
            self._api_json_file(logdir, "loader_summary.json")
        elif endpoint == "feature_store_summary":
            self._api_json_file(logdir, "feature_store_summary.json")
        elif endpoint == "sparse_backend_report":
            self._api_json_file(logdir, "sparse_backend_report.json")
        elif endpoint == "distributed_run_summary":
            self._api_json_file(logdir, "distributed_run_summary.json")
        # ── v0.5.0 additions ────────────────────────────────────────────────
        elif endpoint == "graphsaint_sampler_report":
            self._api_json_file(logdir, "graphsaint_sampler_report.json")
        elif endpoint == "cluster_partition_report":
            self._api_json_file_capped(logdir, "cluster_partition_report.json", max_list_rows=200)
        elif endpoint == "hetero_summary":
            self._api_json_file(logdir, "hetero_summary.json")
        elif endpoint == "temporal_summary":
            self._api_json_file(logdir, "temporal_summary.json")
        elif endpoint == "ogb_tgb_report":
            self._api_json_file(logdir, "ogb_tgb_report.json")
        elif endpoint == "estimator_report":
            self._api_json_file(logdir, "estimator_report.json")
        elif endpoint == "pipeline_report":
            self._api_json_file(logdir, "pipeline_report.json")
        elif endpoint == "graphsaint_benchmark":
            self._api_json_file_capped(logdir, "graphsaint_sampler_report.json", max_list_rows=50)
        elif endpoint == "cluster_gcn_benchmark":
            self._api_json_file_capped(logdir, "cluster_partition_report.json", max_list_rows=50)
        elif endpoint == "hetero_model_benchmark":
            self._api_json_file(logdir, "hetero_summary.json")
        elif endpoint == "temporal_model_benchmark":
            self._api_json_file(logdir, "temporal_summary.json")
        elif endpoint == "public_dataset_smoke":
            self._api_json_file(logdir, "public_dataset_smoke_report.json")
        elif endpoint == "distributed_smoke":
            self._api_json_file(logdir, "distributed_run_summary.json")
        elif endpoint == "calibration_report":
            self._api_json_file(logdir, "calibration_report.json")
        # ── v0.6.0 KG endpoints ──────────────────────────────────────────────
        elif endpoint == "kg_summary":
            self._api_json_file(logdir, "kg_summary.json")
        elif endpoint == "kg_evaluation_report":
            self._api_json_file(logdir, "kg_evaluation_report.json")
        elif endpoint == "kg_training_report":
            self._api_json_file(logdir, "kg_training_report.json")
        elif endpoint == "kg_model_report":
            self._api_json_file(logdir, "kg_model_report.json")
        elif endpoint == "kg_gnn_report":
            self._api_json_file(logdir, "kg_gnn_report.json")
        elif endpoint == "temporal_kg_report":
            self._api_json_file(logdir, "temporal_kg_report.json")
        elif endpoint == "kg_reasoning_report":
            self._api_json_file_capped(logdir, "kg_reasoning_report.json", max_list_rows=50)
        elif endpoint == "kg_benchmark_report":
            self._api_json_file(logdir, "kg_benchmark_report.json")
        elif endpoint == "kg_multimodal_feature_report":
            self._api_json_file(logdir, "kg_multimodal_feature_report.json")
        else:
            self._send_json({"error": f"Unknown endpoint: {endpoint}"}, 404)

    def _api_config(self) -> None:
        """Public client-facing config: poll interval, limits, mode flags.

        Never includes the actual token — only a boolean ``has_token`` so the
        UI can show an advisory when applicable.
        """
        srv = self.server
        host_is_loopback = (
            srv.server_address[0] in _LOOPBACK
            or str(srv.server_address[0]).startswith("127.")
        )
        self._send_json({
            "poll_ms": int(srv.refresh_interval_s * 1000),
            "refresh_interval_s": srv.refresh_interval_s,
            "max_metric_rows": srv.max_metric_rows,
            "host": srv.server_address[0],
            "port": srv.server_address[1],
            "host_is_loopback": host_is_loopback,
            "lan_mode": (not host_is_loopback),
            "has_token": bool(srv.token),
            "logdir_basename": os.path.basename(srv.logdir),
            "stale_after_s": int(max(srv.refresh_interval_s * 6, 30)),
        })

    def _api_status(self, logdir: str) -> None:
        meta_raw = _read_logfile(logdir, "run_metadata.json")
        meta = json.loads(meta_raw) if meta_raw else {}
        metrics_raw = _read_logfile(logdir, "metrics.csv")
        metrics = _parse_metrics(metrics_raw) if metrics_raw else {"headers": [], "rows": []}

        epoch = None
        last_ts = None
        if metrics["rows"] and metrics["headers"]:
            hdrs = metrics["headers"]
            last = metrics["rows"][-1]
            row_dict = dict(zip(hdrs, last))
            # Use explicit None-aware resolution so that epoch=0 / step=0 are
            # reported correctly.  _parse_metrics converts numeric cells to
            # float, so a falsy `or` chain would wrongly skip 0.0.
            epoch = _status_epoch(row_dict)
            last_ts = row_dict.get("timestamp")

        self._send_json({
            "run_name": meta.get("run_name", os.path.basename(logdir)),
            "status": meta.get("status", "unknown"),
            "epoch": epoch,
            "total_epochs": meta.get("total_epochs"),
            "step": meta.get("step"),
            "start_time": meta.get("start_time"),
            "end_time": meta.get("end_time"),
            "last_update": last_ts,
            "device": meta.get("device"),
            "task": meta.get("task"),
        })

    def _api_metrics(self, logdir: str) -> None:
        # Parse query string for optional ?since_row=<int> and ?run=<name>.
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        since_row_raw = qs.get("since_row", [None])[0]
        run_name = qs.get("run", [None])[0]

        # Resolve logdir: multi-run mode uses a child directory.
        effective_logdir = logdir
        if run_name is not None:
            child = _safe_run_path(logdir, run_name)
            if child is None or not os.path.isdir(child):
                self._send_json({"error": "Invalid or unknown run name."}, 400)
                return
            effective_logdir = child

        path = _safe_path(effective_logdir, "metrics.csv")
        empty_response = {
            "headers": [], "rows": [], "total_row_count": 0,
            "truncated": False, "max_rows": self.server.max_metric_rows,
        }
        if path is None or not os.path.isfile(path):
            if since_row_raw is not None:
                empty_response["latest_row_index"] = -1
                empty_response["reset_required"] = False
            self._send_json(empty_response)
            return

        try:
            st = os.stat(path)
            st_inode = getattr(st, "st_ino", None)
            cache_key = (st.st_mtime, st.st_size, self.server.max_metric_rows)
            cached = self.server._metrics_cache

            if cached is not None and cached[0] == cache_key:
                # Cache hit: both the bounded data and full parsed rows are fresh.
                cached_data = cached[1]
                full_rows_cache = cached[2]  # {"headers": ..., "rows": [...all...]}
                content = None  # no disk read needed
            else:
                # Cache miss: determine if we can do a byte-seek (file only grew).
                tail_state = self.server._metrics_tail_state
                content = None
                full_rows_cache = None

                if (
                    tail_state is not None
                    and tail_state.get("inode") == st_inode
                    and st.st_size > tail_state.get("size", 0)
                ):
                    # File grew since last parse: read only new bytes.
                    try:
                        with open(path, "rb") as fbin:
                            fbin.seek(tail_state["byte_pos"])
                            new_bytes = fbin.read()
                        new_text = tail_state.get("partial_buf", "") + new_bytes.decode("utf-8", errors="replace")
                        lines = new_text.split("\n")
                        # Last element may be a partial line (no trailing newline yet).
                        partial = lines[-1]
                        complete_lines = lines[:-1]
                        new_rows_parsed = []
                        for line in complete_lines:
                            line = line.rstrip("\r")
                            if not line:
                                continue
                            parsed: list = []
                            for val in next(csv.reader([line])):
                                try:
                                    parsed.append(float(val))
                                except ValueError:
                                    parsed.append(val)
                            if parsed:
                                new_rows_parsed.append(parsed)
                        existing = tail_state["all_rows"]
                        all_rows_combined = existing + new_rows_parsed
                        headers = tail_state["headers"]
                        new_byte_pos = st.st_size - len(partial.encode("utf-8"))
                        self.server._metrics_tail_state = {
                            "inode": st_inode,
                            "size": st.st_size,
                            "byte_pos": new_byte_pos,
                            "partial_buf": partial,
                            "headers": headers,
                            "all_rows": all_rows_combined,
                        }
                        full_rows_cache = {"headers": headers, "rows": all_rows_combined}
                        total = len(all_rows_combined)
                        if total > self.server.max_metric_rows:
                            cached_data = {
                                "headers": headers,
                                "rows": all_rows_combined[-self.server.max_metric_rows:],
                                "total_row_count": total,
                                "truncated": True,
                                "max_rows": self.server.max_metric_rows,
                            }
                        else:
                            cached_data = {
                                "headers": headers,
                                "rows": all_rows_combined,
                                "total_row_count": total,
                                "truncated": False,
                                "max_rows": self.server.max_metric_rows,
                            }
                        self.server._metrics_cache = (cache_key, cached_data, full_rows_cache)
                    except (OSError, UnicodeDecodeError):
                        content = None  # fall through to full reparse

                if full_rows_cache is None:
                    # Full reparse (inode changed, file shrank, or first read).
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    full_rows_cache = _parse_metrics(content)
                    cached_data = _parse_metrics_bounded(content, self.server.max_metric_rows)
                    self.server._metrics_cache = (cache_key, cached_data, full_rows_cache)
                    # Initialise tail state for next append.
                    self.server._metrics_tail_state = {
                        "inode": st_inode,
                        "size": st.st_size,
                        "byte_pos": st.st_size,
                        "partial_buf": "",
                        "headers": full_rows_cache["headers"],
                        "all_rows": list(full_rows_cache["rows"]),
                    }

            if since_row_raw is not None:
                try:
                    since_row = int(since_row_raw)
                except (ValueError, TypeError):
                    self._send_json({"error": "since_row must be an integer."}, 400)
                    return
                # Use the cached full rows — no second disk read.
                all_rows = full_rows_cache["rows"]
                total = len(all_rows)
                latest = total - 1
                reset_required = total > 0 and since_row > latest
                if reset_required or total == 0:
                    new_rows_out: list = []
                else:
                    new_rows_out = all_rows[max(0, since_row + 1):]
                if len(new_rows_out) > self.server.max_metric_rows:
                    new_rows_out = new_rows_out[-self.server.max_metric_rows:]
                self._send_json({
                    "headers": full_rows_cache["headers"],
                    "rows": new_rows_out,
                    "latest_row_index": latest,
                    "total_row_count": total,
                    "reset_required": reset_required,
                    "truncated": total > self.server.max_metric_rows,
                    "max_rows": self.server.max_metric_rows,
                })
            else:
                self._send_json(cached_data)
        except OSError:
            self._send_json(empty_response)

    def _api_graph_stats(self, logdir: str) -> None:
        """Serve precomputed graph statistics from graph_stats.json."""
        content = _read_logfile(logdir, "graph_stats.json")
        if content is None:
            self._send_json({"available": False})
            return
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"JSON parse error: {exc}", "available": False}, 500)
            return
        data["available"] = True
        self._send_json(data)

    def _api_json_file(self, logdir: str, filename: str) -> None:
        content = _read_logfile(logdir, filename)
        if content is None:
            self._send_json({})
            return
        try:
            self._send_json(json.loads(content))
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"JSON parse error in {filename}: {exc}"}, 500)

    def _api_json_file_capped(
        self,
        logdir: str,
        filename: str,
        max_list_rows: int = 50,
    ) -> None:
        """Read a JSON file and cap any top-level lists to max_list_rows.

        This prevents the dashboard from receiving unmanageably large
        datasets (e.g. hundreds of top-k anomaly nodes).  Adds
        ``'_truncated': True`` and the original length to capped lists.
        """
        content = _read_logfile(logdir, filename)
        if content is None:
            self._send_json({"_available": False})
            return
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"JSON parse error in {filename}: {exc}"}, 500)
            return
        if isinstance(data, dict):
            for k, v in list(data.items()):
                if isinstance(v, list) and len(v) > max_list_rows:
                    data[k] = v[:max_list_rows]
                    data[f"_{k}_total"] = len(v)
                    data[f"_{k}_truncated"] = True
        self._send_json(data)

    def _api_graph(self, logdir: str) -> None:
        content = _read_logfile(logdir, "graph_metadata.json")
        if content is None:
            self._send_json({"available": False})
            return
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            self._send_json({"error": str(exc)}, 500)
            return

        num_nodes = data.get("num_nodes", 0)
        num_edges = data.get("num_edges", 0)
        has_edge_index = "edge_index" in data

        # Decide render mode
        if has_edge_index and num_nodes <= 200 and num_edges <= 1000:
            render_mode = "full"
        elif has_edge_index:
            render_mode = "summary"
            data.pop("edge_index", None)  # don't send large array
        else:
            render_mode = "summary"

        data["render_mode"] = render_mode
        data["available"] = True
        self._send_json(data)


# ─────────────────────────────────────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────────────────────────────────────

class DashboardServer(ThreadingHTTPServer):
    """Thread-per-request HTTP server for the TGraphX dashboard."""

    def __init__(
        self,
        logdir: str,
        host: str,
        port: int,
        token: Optional[str] = None,
        verbose: bool = True,
        max_metric_rows: int = _DEFAULT_MAX_METRIC_ROWS,
        refresh_interval_s: float = 2.0,
    ) -> None:
        if host not in _LOOPBACK and not token:
            raise ValueError(
                f"Starting the dashboard on {host!r} (LAN mode) without a token "
                f"would expose it to your local network without authentication.\n"
                f"Provide --token <secret> or use --host 127.0.0.1 for local-only access."
            )
        self.logdir = os.path.realpath(logdir)
        self.token = token
        self.verbose = verbose
        self.max_metric_rows = max_metric_rows
        # Clamp refresh interval to a sane window so users can't accidentally
        # hammer the server or starve the UI.
        self.refresh_interval_s = max(0.5, min(60.0, float(refresh_interval_s)))
        # (mtime, size, max_rows) → (bounded_data, full_rows) cache; replaced atomically.
        # Tuple is 3-element: (key, bounded_payload, full_parsed_rows).
        self._metrics_cache = None
        # Byte-offset tail-read state for append-only metrics files.
        # Keys: inode, size, byte_pos, partial_buf, headers, all_rows.
        self._metrics_tail_state = None
        super().__init__((host, port), DashboardHandler)


# ─────────────────────────────────────────────────────────────────────────────
# Offline standalone HTML export
# ─────────────────────────────────────────────────────────────────────────────

def _load_export_assets() -> tuple:
    """Load dashboard CSS and JS from the static directory.

    Returns:
        (css_content, js_content) as strings.

    Raises:
        FileNotFoundError: If static assets are missing.
    """
    css_path = os.path.join(STATIC_DIR, "dashboard.css")
    js_path = os.path.join(STATIC_DIR, "dashboard.js")
    if not os.path.isfile(css_path) or not os.path.isfile(js_path):
        raise FileNotFoundError(f"Dashboard static assets not found in {STATIC_DIR!r}")
    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()
    with open(js_path, "r", encoding="utf-8") as f:
        js = f.read()
    return css, js


def _collect_run_snapshot(logdir_real: str) -> dict:
    """Read and parse run log files from logdir into a snapshot dict.

    Args:
        logdir_real: Realpath of the run directory.

    Returns:
        Dict with keys: metrics, metadata, graph, graph_stats, logdir_basename.
    """
    def _read_file(fname: str) -> Optional[str]:
        p = os.path.join(logdir_real, fname)
        if not os.path.isfile(p):
            return None
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError:
            return None

    metrics_raw = _read_file("metrics.csv")
    metadata_raw = _read_file("run_metadata.json")
    graph_raw = _read_file("graph_metadata.json")
    stats_raw = _read_file("graph_stats.json")

    metrics_data: dict = (
        _parse_metrics_bounded(metrics_raw, _DEFAULT_MAX_METRIC_ROWS)
        if metrics_raw
        else {"headers": [], "rows": [], "total_row_count": 0, "truncated": False,
              "max_rows": _DEFAULT_MAX_METRIC_ROWS}
    )
    metadata_data: dict = {}
    graph_data: dict = {"available": False}
    stats_data: dict = {"available": False}

    if metadata_raw:
        try:
            metadata_data = json.loads(metadata_raw)
        except json.JSONDecodeError:
            pass

    if graph_raw:
        try:
            graph_data = json.loads(graph_raw)
            graph_data["available"] = True
            if graph_data.get("num_nodes", 0) > 200 or graph_data.get("num_edges", 0) > 1000:
                graph_data.pop("edge_index", None)
                graph_data["render_mode"] = "summary"
            elif "edge_index" in graph_data:
                graph_data["render_mode"] = "full"
            else:
                graph_data["render_mode"] = "summary"
        except json.JSONDecodeError:
            pass

    if stats_raw:
        try:
            stats_data = json.loads(stats_raw)
            stats_data["available"] = True
        except json.JSONDecodeError:
            pass

    return {
        "metrics": metrics_data,
        "metadata": metadata_data,
        "graph": graph_data,
        "graph_stats": stats_data,
        "logdir_basename": os.path.basename(logdir_real),
    }


_HTML_SHELL = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark light">
<meta name="referrer" content="no-referrer">
<title>TGraphX Snapshot — {title}</title>
{size_comment}<style>
{css}
</style>
</head>
<body>
<a class="skip-link" href="#content">Skip to main content</a>
<div id="app">
  <nav id="sidebar" aria-label="Dashboard sections">
    <div class="brand"><span class="brand-hex" aria-hidden="true">&#x2B23;</span><span class="brand-name">TGraphX</span></div>
    <ul id="nav-list" role="list"></ul>
    <button id="sidebar-close" type="button" aria-label="Close navigation menu">&#x2715;</button>
  </nav>
  <div id="main-wrap">
    <header id="topbar" role="banner">
      <button id="hamburger" type="button" aria-label="Open navigation menu" aria-controls="sidebar">&#x2630;</button>
      <span id="run-title" aria-live="polite">TGraphX Snapshot</span>
      <div id="topbar-right">
        <button id="pause-btn" type="button" class="icon-btn" aria-label="Pause auto-refresh" aria-pressed="false" title="Pause auto-refresh" hidden>&#x23F8;</button>
        <button id="refresh-btn" type="button" class="icon-btn" aria-label="Refresh now" title="Refresh now" hidden>&#x21BB;</button>
        <span id="status-chip" class="chip chip-unknown" role="status" aria-live="polite">snapshot</span>
        <span id="viewer-clock" title="Your local time" aria-label="Current local time"></span>
        <button id="palette-btn" type="button" class="icon-btn" aria-label="Toggle color-blind safe palette" aria-pressed="false" title="Color-blind safe palette">&#x25CF;</button>
        <button id="theme-btn" type="button" class="icon-btn" aria-label="Toggle dark/light theme" title="Toggle theme">&#x263D;</button>
      </div>
    </header>
    <div id="stale-banner" class="stale-banner" hidden role="alert">
      <span id="stale-text">Offline snapshot — data is fixed at export time.</span>
    </div>
    <main id="content" tabindex="-1">
      <section id="sec-overview"  class="sec active" aria-labelledby="sec-overview-title"></section>
      <section id="sec-metrics"   class="sec" aria-labelledby="sec-metrics-title"></section>
      <section id="sec-graph"     class="sec" aria-labelledby="sec-graph-title"></section>
      <section id="sec-mining"    class="sec" aria-labelledby="sec-mining-title"></section>
      <section id="sec-hardware"  class="sec" aria-labelledby="sec-hardware-title"></section>
      <section id="sec-logs"      class="sec" aria-labelledby="sec-logs-title"></section>
      <section id="sec-config"    class="sec" aria-labelledby="sec-config-title"></section>
      <section id="sec-tools"     class="sec" aria-labelledby="sec-tools-title"></section>
      <section id="sec-about"     class="sec" aria-labelledby="sec-about-title"></section>
    </main>
    <footer id="footer" role="contentinfo">
      <span id="update-txt" aria-live="polite">Offline snapshot</span>
      <span class="sep" aria-hidden="true">&#xB7;</span>
      <button type="button" id="tv-btn" class="link-btn" aria-label="Enter full-screen TV mode">TV&nbsp;mode</button>
    </footer>
  </div>
</div>
<div id="tv-overlay" hidden role="dialog" aria-modal="true" aria-label="TV mode">
  <button id="tv-exit" type="button" aria-label="Exit TV mode">&#x2715; Exit TV mode</button>
  <div id="tv-body"></div>
</div>
<script>
/* TGraphX offline snapshot — data embedded at export time, no server needed.
   This script block embeds preloaded data ONLY.  No eval(), no Function().  */
window.__TGXSNAP = {snapshot_js};
</script>
<script>
{js}
</script>
</body>
</html>"""


def _render_snapshot_html(
    css_content: str,
    js_content: str,
    snapshot: dict,
    logdir_basename: str,
) -> bytes:
    """Render the offline HTML snapshot and return it as UTF-8 bytes.

    Args:
        css_content: Dashboard CSS string.
        js_content: Dashboard JS string.
        snapshot: Run data dict from _collect_run_snapshot.
        logdir_basename: Display name for the run directory.

    Returns:
        HTML as UTF-8 bytes.
    """
    snapshot_js = json.dumps(snapshot, ensure_ascii=True).replace(
        "</script>", r"<\/script>"
    )
    approx_kb = (len(css_content) + len(js_content) + len(snapshot_js)) // 1024
    size_comment = f"<!-- Exported snapshot ~{approx_kb} KB -->\n" if approx_kb > 512 else ""

    html = _HTML_SHELL.format(
        title=logdir_basename,
        size_comment=size_comment,
        css=css_content,
        snapshot_js=snapshot_js,
        js=js_content,
    )
    out_bytes = html.encode("utf-8")

    if approx_kb > 10240:
        import warnings
        warnings.warn(
            f"Snapshot HTML is large (~{approx_kb} KB). "
            "Consider lowering --max-metric-rows before exporting.",
            stacklevel=3,
        )

    return out_bytes


def export_dashboard_html(logdir: str, out_path: str) -> None:
    """Produce a self-contained offline HTML snapshot of a run.

    Reads the packaged dashboard CSS/JS and the run log files, embeds
    everything into one HTML file.  No server required to open the result.

    Security contract
    -----------------
    * Token is never embedded.
    * Embedded JSON is serialized safely and ``</script>`` occurrences
      inside values are escaped so the browser parser never closes the
      ``<script>`` block prematurely.
    * No external URLs or CDN references.
    * No ``eval`` or ``new Function``.

    Args:
        logdir:   Directory containing run log files.
        out_path: Destination HTML file path.  Parent directory must exist.
    """
    logdir_real = os.path.realpath(logdir)
    if not os.path.isdir(logdir_real):
        raise ValueError(f"logdir does not exist or is not a directory: {logdir!r}")

    out_real = os.path.realpath(out_path)
    parent = os.path.dirname(out_real)
    if not os.path.isdir(parent):
        raise ValueError(f"Output directory does not exist: {parent!r}")

    css_content, js_content = _load_export_assets()
    snapshot = _collect_run_snapshot(logdir_real)
    out_bytes = _render_snapshot_html(
        css_content, js_content, snapshot, os.path.basename(logdir_real)
    )

    with open(out_real, "wb") as f:
        f.write(out_bytes)


# ─────────────────────────────────────────────────────────────────────────────
# Best-effort LAN IP detection (no external network calls)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_lan_ip() -> Optional[str]:
    """Return this machine's primary LAN IP without hitting the network.

    Uses the standard "connect a UDP socket to a non-routable address" trick:
    no packet is actually sent, but the kernel populates the source address
    based on the routing table.  Returns ``None`` if detection fails so the
    caller can fall back to a printed instruction.
    """
    import socket
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 10.255.255.255 / port 1 is a non-routable reserved address; setting
        # it as the connect target tells the kernel which interface would be
        # used without sending anything.
        s.settimeout(0.2)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        return ip if ip and not ip.startswith("127.") else None
    except OSError:
        return None
    finally:
        if s is not None:
            try:
                s.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point: ``tgraphx-dashboard`` / ``python -m tgraphx.dashboard``."""
    parser = argparse.ArgumentParser(
        prog="tgraphx-dashboard",
        description="TGraphX local training dashboard (read-only, off by default).",
    )
    parser.add_argument("--logdir", required=True, help="Run log directory")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind host (default: 127.0.0.1; use 0.0.0.0 for LAN)")
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    parser.add_argument("--token", default=None,
                        help="Auth token for LAN mode (use 'auto' to generate one)")
    parser.add_argument("--quiet", action="store_true", help="Suppress request logs")
    parser.add_argument(
        "--max-metric-rows", type=int, default=_DEFAULT_MAX_METRIC_ROWS,
        help=f"Max rows returned by /api/metrics (default {_DEFAULT_MAX_METRIC_ROWS}). "
             "Older rows are omitted when the log exceeds this limit.",
    )
    parser.add_argument(
        "--refresh-interval", type=float, default=2.0,
        help="Browser auto-refresh interval in seconds (default: 2.0; clamped to [0.5, 60]).",
    )
    parser.add_argument(
        "--open-browser", action="store_true",
        help="Open the dashboard URL in the default browser after start (off by default).",
    )
    parser.add_argument(
        "--export-html", default=None, metavar="OUT_PATH",
        help=(
            "Export a self-contained offline HTML snapshot to OUT_PATH and exit. "
            "No server is started.  The file embeds inlined CSS/JS and log data. "
            "Token is never included in the exported file."
        ),
    )
    args = parser.parse_args()

    if not os.path.isdir(args.logdir):
        parser.error(f"logdir does not exist: {args.logdir}")

    # ── Offline HTML export mode ────────────────────────────────────────────
    if args.export_html is not None:
        out_path = args.export_html
        try:
            export_dashboard_html(args.logdir, out_path)
        except (ValueError, FileNotFoundError, OSError) as exc:
            parser.error(str(exc))
        import os as _os
        size_kb = _os.path.getsize(out_path) // 1024
        print("\n  TGraphX Dashboard snapshot exported.")
        print(f"  → {out_path}  ({size_kb} KB)")
        print("  Open the file in any browser — no server needed.\n")
        return

    # 'auto' token: generate a short URL-safe token rather than echo it twice.
    token = args.token
    if token == "auto":
        import secrets
        token = secrets.token_urlsafe(16)

    try:
        server = DashboardServer(
            args.logdir, args.host, args.port,
            token=token, verbose=not args.quiet,
            max_metric_rows=args.max_metric_rows,
            refresh_interval_s=args.refresh_interval,
        )
    except ValueError as exc:
        parser.error(str(exc))

    bound_host, bound_port = server.server_address

    print("\n  TGraphX Dashboard")
    print(f"  Local  → http://127.0.0.1:{bound_port}")
    if args.host == "0.0.0.0":
        lan_ip = _detect_lan_ip()
        if lan_ip:
            tok_qs = f"?token={token}" if token else ""
            print(f"  LAN    → http://{lan_ip}:{bound_port}{tok_qs}")
        else:
            print(f"  LAN    → use this machine's LAN IP, port {bound_port}")
        if token:
            print("  Token  → required for non-localhost clients (printed once).")
        else:
            # The DashboardServer constructor would have refused this, but
            # guard against future refactors.
            print("  Warning: LAN mode without a token; this should never happen.")
    elif args.host not in _LOOPBACK:
        print(f"  → bound to {args.host}:{bound_port}  (token protected)")
    print(f"  logdir: {args.logdir}")
    print(f"  Refresh: {server.refresh_interval_s}s  ·  Max rows: {server.max_metric_rows}")
    print("  Press Ctrl-C to stop.\n")

    if args.open_browser:
        try:
            import webbrowser
            url = f"http://127.0.0.1:{bound_port}"
            webbrowser.open(url)
        except Exception:
            pass  # never block startup on browser launch

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")
    finally:
        server.shutdown()
