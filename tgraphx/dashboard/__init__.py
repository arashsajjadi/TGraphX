"""TGraphX local training dashboard.

A lightweight, local-first monitoring dashboard for TGraphX training runs.
Serves on localhost by default (no external access).  LAN mode requires
an explicit ``--token`` for security.

Quick start::

    # CLI
    tgraphx-dashboard --logdir runs/demo

    # Python
    from tgraphx.dashboard import launch_dashboard
    launch_dashboard("runs/demo")

    # Non-blocking (background thread for use during training)
    from tgraphx.dashboard import launch_dashboard_background
    server = launch_dashboard_background("runs/demo", port=8765)
    # ... training loop ...
    server.shutdown()

LAN mode (requires token)::

    tgraphx-dashboard --logdir runs/demo --host 0.0.0.0 --token MY_SECRET

Security notes
--------------
* localhost is always allowed without a token.
* Binding to 0.0.0.0 (LAN mode) without a token raises ``ValueError``
  to prevent accidental network exposure.
* All file reads are restricted to ``logdir``.
* No external scripts, fonts, or CDN assets are loaded.
* The dashboard is **read-only** — it never writes to ``logdir``.
"""
from __future__ import annotations

import threading
from typing import Optional

from .app import DashboardServer, _DEFAULT_MAX_METRIC_ROWS


def launch_dashboard(
    logdir: str,
    host: str = "127.0.0.1",
    port: int = 8765,
    token: Optional[str] = None,
    verbose: bool = True,
    max_metric_rows: int = _DEFAULT_MAX_METRIC_ROWS,
    refresh_interval_s: float = 2.0,
) -> None:
    """Start the dashboard and block until interrupted (Ctrl-C).

    Args:
        logdir: Directory containing ``metrics.csv`` and optional JSON files.
        host:   Binding host.  Use ``"127.0.0.1"`` (default) for local-only
                or ``"0.0.0.0"`` for LAN access (requires ``token``).
        port:   HTTP port (default 8765).
        token:  Required when ``host`` is not a loopback address.
        verbose: Print startup message and access URL.
        max_metric_rows: Maximum rows returned by ``/api/metrics``.
            Older rows are omitted when the log exceeds this limit.
        refresh_interval_s: Browser auto-refresh interval (seconds, clamped
            to [0.5, 60]).
    """
    server = DashboardServer(logdir, host, port, token=token, verbose=verbose,
                             max_metric_rows=max_metric_rows,
                             refresh_interval_s=refresh_interval_s)
    if verbose:
        _print_banner(host, port, token)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


def launch_dashboard_background(
    logdir: str,
    host: str = "127.0.0.1",
    port: int = 8765,
    token: Optional[str] = None,
    verbose: bool = True,
    max_metric_rows: int = _DEFAULT_MAX_METRIC_ROWS,
    refresh_interval_s: float = 2.0,
) -> DashboardServer:
    """Start the dashboard in a daemon background thread and return the server.

    The server stops automatically when the main process exits.  Call
    ``server.shutdown()`` to stop it explicitly.

    Returns:
        The running :class:`DashboardServer` instance.
    """
    server = DashboardServer(logdir, host, port, token=token, verbose=verbose,
                             max_metric_rows=max_metric_rows,
                             refresh_interval_s=refresh_interval_s)
    if verbose:
        _print_banner(host, port, token)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="tgraphx-dashboard")
    thread.start()
    return server


def _print_banner(host: str, port: int, token: Optional[str]) -> None:
    if host == "0.0.0.0":
        print("\n  TGraphX Dashboard")
        print(f"  Local  → http://127.0.0.1:{port}")
        print(f"  LAN    → use this machine's LAN IP, port {port}  (token required)")
    else:
        print(f"\n  TGraphX Dashboard → http://{host}:{port}")
    print()


__all__ = ["launch_dashboard", "launch_dashboard_background", "DashboardServer"]
