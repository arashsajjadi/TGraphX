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
<title>TGraphX Dashboard</title>
<link rel="stylesheet" href="/static/dashboard.css">
</head>
<body>
<div id="app">
  <nav id="sidebar">
    <div class="brand"><span class="brand-hex">&#x2B23;</span><span class="brand-name">TGraphX</span></div>
    <ul id="nav-list"></ul>
    <button id="sidebar-close" aria-label="Close menu">&#x2715;</button>
  </nav>
  <div id="main-wrap">
    <header id="topbar">
      <button id="hamburger" aria-label="Open menu">&#x2630;</button>
      <span id="run-title">TGraphX Dashboard</span>
      <div id="topbar-right">
        <span id="status-chip" class="chip chip-unknown">&#x2014;</span>
        <span id="viewer-clock" title="Your local time">&#x2014;</span>
        <button id="theme-btn" title="Toggle theme">&#x263D;</button>
      </div>
    </header>
    <main id="content">
      <section id="sec-overview"  class="sec active"></section>
      <section id="sec-metrics"   class="sec"></section>
      <section id="sec-graph"     class="sec"></section>
      <section id="sec-hardware"  class="sec"></section>
      <section id="sec-logs"      class="sec"></section>
      <section id="sec-config"    class="sec"></section>
      <section id="sec-about"     class="sec"></section>
    </main>
    <footer id="footer">
      <span id="update-txt">&#x2014;</span>
      <span class="sep">&#xB7;</span>
      <a href="#" id="tv-btn">TV&nbsp;mode</a>
    </footer>
  </div>
</div>
<div id="tv-overlay" hidden>
  <button id="tv-exit">&#x2715; Exit TV mode</button>
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


# ─────────────────────────────────────────────────────────────────────────────
# Hardware / system information (best-effort)
# ─────────────────────────────────────────────────────────────────────────────

def _collect_hardware() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "psutil_available": False,
        "pynvml_available": False,
        "cuda_available": False,
        "mps_available": False,
        "cpu_count": os.cpu_count(),
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
            dev = torch.device("cuda:0")
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_mem_allocated_mb"] = round(torch.cuda.memory_allocated(dev) / 1024**2, 1)
            info["cuda_mem_reserved_mb"] = round(torch.cuda.memory_reserved(dev) / 1024**2, 1)
            info["cuda_mem_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1)
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None:
            info["mps_available"] = bool(getattr(mps, "is_available", lambda: False)())
    except Exception:
        info["torch"] = "not installed"

    try:
        import psutil
        info["psutil_available"] = True
        info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
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
        pass

    try:
        import pynvml
        pynvml.nvmlInit()
        info["pynvml_available"] = True
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        info["gpu_mem_used_mb"] = round(mem.used / 1024**2, 1)
        info["gpu_mem_total_mb"] = round(mem.total / 1024**2, 1)
        try:
            info["gpu_util_pct"] = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            info["gpu_temp_c"] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            pass
    except Exception:
        pass

    return info


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
        else:
            self._send_json({"error": f"Unknown endpoint: {endpoint}"}, 404)

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
            epoch = row_dict.get("epoch") or row_dict.get("step")
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
        path = _safe_path(logdir, "metrics.csv")
        if path is None or not os.path.isfile(path):
            self._send_json({"headers": [], "rows": [], "total_row_count": 0,
                             "truncated": False, "max_rows": self.server.max_metric_rows})
            return
        try:
            st = os.stat(path)
            cache_key = (st.st_mtime, st.st_size, self.server.max_metric_rows)
            cached = self.server._metrics_cache
            if cached is not None and cached[0] == cache_key:
                self._send_json(cached[1])
                return
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            parsed = _parse_metrics_bounded(content, self.server.max_metric_rows)
            self.server._metrics_cache = (cache_key, parsed)
            self._send_json(parsed)
        except OSError:
            self._send_json({"headers": [], "rows": [], "total_row_count": 0,
                             "truncated": False, "max_rows": self.server.max_metric_rows})

    def _api_json_file(self, logdir: str, filename: str) -> None:
        content = _read_logfile(logdir, filename)
        if content is None:
            self._send_json({})
            return
        try:
            self._send_json(json.loads(content))
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"JSON parse error in {filename}: {exc}"}, 500)

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
        # (mtime, size, max_rows) → parsed metrics cache; replaced atomically
        self._metrics_cache = None
        super().__init__((host, port), DashboardHandler)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point: ``tgraphx-dashboard`` / ``python -m tgraphx.dashboard``."""
    parser = argparse.ArgumentParser(
        prog="tgraphx-dashboard",
        description="TGraphX local training dashboard",
    )
    parser.add_argument("--logdir", required=True, help="Run log directory")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    parser.add_argument("--token", default=None, help="Auth token for LAN mode")
    parser.add_argument("--quiet", action="store_true", help="Suppress request logs")
    parser.add_argument(
        "--max-metric-rows", type=int, default=_DEFAULT_MAX_METRIC_ROWS,
        help=f"Max rows returned by /api/metrics (default {_DEFAULT_MAX_METRIC_ROWS}). "
             "Older rows are omitted when the log exceeds this limit.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.logdir):
        parser.error(f"logdir does not exist: {args.logdir}")

    try:
        server = DashboardServer(
            args.logdir, args.host, args.port,
            token=args.token, verbose=not args.quiet,
            max_metric_rows=args.max_metric_rows,
        )
    except ValueError as exc:
        parser.error(str(exc))

    display = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print(f"\n  TGraphX Dashboard")
    print(f"  → http://{display}:{args.port}")
    if args.host not in _LOOPBACK:
        print(f"  LAN mode  (token protected)")
    print(f"  logdir: {args.logdir}")
    print(f"  Press Ctrl-C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")
    finally:
        server.shutdown()
