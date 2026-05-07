"""Tests for tgraphx.dashboard — security, API, and rendering."""
from __future__ import annotations

import csv
import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request

import pytest

from tgraphx.dashboard.app import (
    DashboardServer,
    _parse_metrics,
    _safe_path,
    _read_logfile,
    _collect_hardware,
    _status_epoch,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def logdir():
    with tempfile.TemporaryDirectory() as d:
        # metrics.csv
        with open(os.path.join(d, "metrics.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_loss", "timestamp"])
            w.writerow([1, 0.8, 0.9, "2025-01-01T12:00:00Z"])
            w.writerow([2, 0.6, 0.75, "2025-01-01T12:01:00Z"])

        # run_metadata.json
        with open(os.path.join(d, "run_metadata.json"), "w") as f:
            json.dump({
                "run_name": "test_run",
                "status": "running",
                "total_epochs": 10,
                "epoch": 2,
                "task": "graph_classification",
                "device": "cpu",
                "start_time": "2025-01-01T12:00:00Z",
            }, f)

        # graph_metadata.json (small graph — should render)
        with open(os.path.join(d, "graph_metadata.json"), "w") as f:
            json.dump({
                "num_nodes": 4,
                "num_edges": 8,
                "directed": False,
                "self_loops": True,
                "builder": "build_grid_graph",
                "builder_params": {"rows": 2, "cols": 2},
                "degree_stats": {"mean": 3.0, "min": 2, "max": 4},
                "edge_index": [[0,1,2,3,0,1,2,3], [1,0,3,2,0,1,2,3]],
            }, f)

        yield d


@pytest.fixture(scope="module")
def server(logdir):
    """Start a real local server on a random port."""
    srv = DashboardServer(logdir, "127.0.0.1", 0, token=None, verbose=False)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)  # give the server a moment to bind
    yield srv
    srv.shutdown()


@pytest.fixture(scope="module")
def base_url(server):
    host, port = server.server_address
    return f"http://127.0.0.1:{port}"


def get(url: str, timeout: int = 5):
    """GET url and return (status_code, body_text)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


# ─────────────────────────────────────────────────────────────────────────────
# Security: path traversal
# ─────────────────────────────────────────────────────────────────────────────

class TestPathSecurity:

    def test_safe_path_allows_valid(self, logdir):
        p = _safe_path(logdir, "metrics.csv")
        assert p is not None
        assert p.endswith("metrics.csv")

    def test_safe_path_blocks_traversal(self, logdir):
        assert _safe_path(logdir, "../../../etc/passwd") is None

    def test_safe_path_blocks_double_dot(self, logdir):
        assert _safe_path(logdir, "../../secret") is None

    def test_safe_path_blocks_absolute(self, logdir):
        assert _safe_path(logdir, "/etc/passwd") is None

    def test_read_logfile_returns_none_for_absent(self, logdir):
        assert _read_logfile(logdir, "does_not_exist.csv") is None

    def test_read_logfile_returns_none_for_traversal(self, logdir):
        assert _read_logfile(logdir, "../../../etc/passwd") is None


# ─────────────────────────────────────────────────────────────────────────────
# Security: localhost always allowed, LAN mode requires token
# ─────────────────────────────────────────────────────────────────────────────

class TestAuthSecurity:

    def test_localhost_no_token_required(self, server):
        """Server in localhost-only mode: no token needed."""
        assert server.token is None  # started without token

    def test_lan_mode_without_token_raises(self, logdir):
        with pytest.raises(ValueError, match="token"):
            DashboardServer(logdir, "0.0.0.0", 0, token=None)

    def test_lan_mode_with_token_starts(self, logdir):
        srv = DashboardServer(logdir, "0.0.0.0", 0, token="test-secret", verbose=False)
        srv.server_close()

    def test_localhost_server_requires_no_token(self, base_url):
        code, _ = get(f"{base_url}/api/status")
        assert code == 200

    def test_api_no_token_header_needed_on_localhost(self, base_url):
        code, body = get(f"{base_url}/api/metrics")
        assert code == 200
        data = json.loads(body)
        assert "headers" in data


# ─────────────────────────────────────────────────────────────────────────────
# Static file serving
# ─────────────────────────────────────────────────────────────────────────────

class TestStaticFiles:

    def test_html_served_at_root(self, base_url):
        code, body = get(base_url + "/")
        assert code == 200
        assert "TGraphX" in body
        assert "dashboard.css" in body
        assert "dashboard.js" in body

    def test_css_served(self, base_url):
        code, body = get(base_url + "/static/dashboard.css")
        assert code == 200
        assert "sidebar" in body or "--bg" in body

    def test_js_served(self, base_url):
        code, body = get(base_url + "/static/dashboard.js")
        assert code == 200
        assert "TGraphX" in body or "SvgChart" in body

    def test_unknown_static_404(self, base_url):
        code, _ = get(base_url + "/static/evil.php")
        assert code == 404

    def test_no_external_asset_refs(self, base_url):
        _, css = get(base_url + "/static/dashboard.css")
        _, js  = get(base_url + "/static/dashboard.js")
        for content in (css, js):
            assert "cdn.jsdelivr" not in content
            assert "googleapis.com" not in content
            assert "cloudflare" not in content

    def test_unknown_route_404(self, base_url):
        code, body = get(base_url + "/totally-unknown")
        assert code == 404


# ─────────────────────────────────────────────────────────────────────────────
# API: /api/metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsEndpoint:

    def test_returns_200(self, base_url):
        code, _ = get(f"{base_url}/api/metrics")
        assert code == 200

    def test_returns_headers_and_rows(self, base_url):
        _, body = get(f"{base_url}/api/metrics")
        data = json.loads(body)
        assert "headers" in data
        assert "rows" in data

    def test_headers_correct(self, base_url):
        _, body = get(f"{base_url}/api/metrics")
        data = json.loads(body)
        assert "epoch" in data["headers"]
        assert "train_loss" in data["headers"]

    def test_rows_parsed_as_numbers(self, base_url):
        _, body = get(f"{base_url}/api/metrics")
        data = json.loads(body)
        assert len(data["rows"]) >= 2
        assert isinstance(data["rows"][0][0], (int, float))  # epoch

    def test_empty_metrics_graceful(self):
        """parse_metrics on empty string returns empty dict."""
        result = _parse_metrics("")
        assert result["headers"] == []
        assert result["rows"] == []


# ─────────────────────────────────────────────────────────────────────────────
# API: /api/status
# ─────────────────────────────────────────────────────────────────────────────

class TestStatusEndpoint:

    def test_returns_200(self, base_url):
        code, _ = get(f"{base_url}/api/status")
        assert code == 200

    def test_has_run_name(self, base_url):
        _, body = get(f"{base_url}/api/status")
        data = json.loads(body)
        assert "run_name" in data
        assert data["run_name"] == "test_run"

    def test_has_status_field(self, base_url):
        _, body = get(f"{base_url}/api/status")
        data = json.loads(body)
        assert "status" in data

    def test_without_metadata_json_returns_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            # Only metrics.csv, no run_metadata.json
            with open(os.path.join(d, "metrics.csv"), "w") as f:
                f.write("epoch,train_loss\n1,0.5\n")
            srv = DashboardServer(d, "127.0.0.1", 0, verbose=False)
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            time.sleep(0.05)
            try:
                h, p = srv.server_address
                _, body = get(f"http://127.0.0.1:{p}/api/status")
                data = json.loads(body)
                assert data["status"] == "unknown"
            finally:
                srv.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# API: /api/hardware
# ─────────────────────────────────────────────────────────────────────────────

class TestHardwareEndpoint:

    def test_returns_200(self, base_url):
        code, _ = get(f"{base_url}/api/hardware")
        assert code == 200

    def test_has_python_version(self, base_url):
        _, body = get(f"{base_url}/api/hardware")
        data = json.loads(body)
        assert "python" in data
        assert data["python"].startswith("3")

    def test_works_without_psutil(self):
        """_collect_hardware() should not raise if psutil is absent."""
        info = _collect_hardware()
        assert "python" in info
        assert "platform" in info

    def test_has_torch_field(self, base_url):
        _, body = get(f"{base_url}/api/hardware")
        data = json.loads(body)
        assert "torch" in data

    def test_has_psutil_flag(self, base_url):
        _, body = get(f"{base_url}/api/hardware")
        data = json.loads(body)
        assert "psutil_available" in data


# ─────────────────────────────────────────────────────────────────────────────
# API: /api/metadata
# ─────────────────────────────────────────────────────────────────────────────

class TestMetadataEndpoint:

    def test_returns_200(self, base_url):
        code, _ = get(f"{base_url}/api/metadata")
        assert code == 200

    def test_returns_metadata_fields(self, base_url):
        _, body = get(f"{base_url}/api/metadata")
        data = json.loads(body)
        assert data.get("run_name") == "test_run"
        assert data.get("task") == "graph_classification"

    def test_missing_metadata_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            srv = DashboardServer(d, "127.0.0.1", 0, verbose=False)
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            time.sleep(0.05)
            try:
                _, p = srv.server_address
                _, body = get(f"http://127.0.0.1:{p}/api/metadata")
                data = json.loads(body)
                assert data == {}
            finally:
                srv.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# API: /api/graph
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphEndpoint:

    def test_returns_200(self, base_url):
        code, _ = get(f"{base_url}/api/graph")
        assert code == 200

    def test_small_graph_has_edge_index(self, base_url):
        _, body = get(f"{base_url}/api/graph")
        data = json.loads(body)
        assert data.get("available") is True
        assert data.get("render_mode") == "full"
        assert "edge_index" in data

    def test_graph_summary_fields(self, base_url):
        _, body = get(f"{base_url}/api/graph")
        data = json.loads(body)
        assert "num_nodes" in data
        assert "num_edges" in data
        assert "directed" in data

    def test_large_graph_summary_only(self):
        with tempfile.TemporaryDirectory() as d:
            # Create a large graph metadata (over the node limit)
            big_nodes = 300
            big_edges = 1500
            with open(os.path.join(d, "graph_metadata.json"), "w") as f:
                json.dump({
                    "num_nodes": big_nodes,
                    "num_edges": big_edges,
                    "directed": False,
                    "self_loops": False,
                    "edge_index": [list(range(big_edges)), list(range(big_edges))],
                }, f)

            srv = DashboardServer(d, "127.0.0.1", 0, verbose=False)
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            time.sleep(0.05)
            try:
                _, p = srv.server_address
                _, body = get(f"http://127.0.0.1:{p}/api/graph")
                data = json.loads(body)
                assert data["render_mode"] == "summary"
                assert "edge_index" not in data, "edge_index must be stripped for large graphs"
            finally:
                srv.shutdown()

    def test_no_graph_metadata_returns_unavailable(self):
        with tempfile.TemporaryDirectory() as d:
            srv = DashboardServer(d, "127.0.0.1", 0, verbose=False)
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            time.sleep(0.05)
            try:
                _, p = srv.server_address
                _, body = get(f"http://127.0.0.1:{p}/api/graph")
                data = json.loads(body)
                assert data.get("available") is False
            finally:
                srv.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# Empty logdir (graceful degradation)
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyLogdir:

    @pytest.fixture(scope="class")
    def empty_server(self):
        with tempfile.TemporaryDirectory() as d:
            srv = DashboardServer(d, "127.0.0.1", 0, verbose=False)
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            time.sleep(0.05)
            yield srv
            srv.shutdown()

    def test_metrics_empty(self, empty_server):
        _, p = empty_server.server_address
        _, body = get(f"http://127.0.0.1:{p}/api/metrics")
        data = json.loads(body)
        assert data["headers"] == []
        assert data["rows"] == []

    def test_html_served(self, empty_server):
        _, p = empty_server.server_address
        code, body = get(f"http://127.0.0.1:{p}/")
        assert code == 200
        assert "TGraphX" in body

    def test_hardware_served(self, empty_server):
        _, p = empty_server.server_address
        code, _ = get(f"http://127.0.0.1:{p}/api/hardware")
        assert code == 200


# ─────────────────────────────────────────────────────────────────────────────
# Metrics CSV parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestCsvParsing:

    def test_basic_parse(self):
        csv_txt = "epoch,train_loss\n1,0.5\n2,0.4\n"
        result = _parse_metrics(csv_txt)
        assert result["headers"] == ["epoch", "train_loss"]
        assert result["rows"][0] == [1.0, 0.5]

    def test_string_values_preserved(self):
        csv_txt = "epoch,timestamp\n1,2025-01-01T00:00:00Z\n"
        result = _parse_metrics(csv_txt)
        assert isinstance(result["rows"][0][1], str)
        assert "2025" in result["rows"][0][1]

    def test_empty_content(self):
        result = _parse_metrics("")
        assert result == {"headers": [], "rows": []}

    def test_header_only(self):
        result = _parse_metrics("epoch,loss\n")
        assert result["headers"] == ["epoch", "loss"]
        assert result["rows"] == []


# ─────────────────────────────────────────────────────────────────────────────
# LAN mode token validation (unit test, no real network)
# ─────────────────────────────────────────────────────────────────────────────

class TestLanModeToken:

    def test_lan_with_token_api_reachable_from_localhost(self, logdir):
        """Even in LAN mode, localhost clients bypass token check."""
        srv = DashboardServer(logdir, "127.0.0.1", 0, token="secret", verbose=False)
        # Note: token is set but host is loopback, so localhost clients are always OK
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        time.sleep(0.05)
        try:
            _, p = srv.server_address
            # Localhost client — should be allowed regardless of token
            code, _ = get(f"http://127.0.0.1:{p}/api/status")
            assert code == 200
        finally:
            srv.shutdown()


# =========================================================================== #
# Bounded metrics loading                                                       #
# =========================================================================== #

class TestBoundedMetrics:

    def _make_big_csv(self, d: str, n_rows: int) -> str:
        """Write a metrics.csv with n_rows data rows."""
        path = os.path.join(d, "metrics.csv")
        with open(path, "w", newline="") as f:
            import csv as _csv
            w = _csv.writer(f)
            w.writerow(["epoch", "train_loss"])
            for i in range(1, n_rows + 1):
                w.writerow([i, round(1.0 / i, 4)])
        return path

    def _start_server(self, d: str, max_rows: int = 5000):
        import threading, time
        from tgraphx.dashboard.app import DashboardServer
        srv = DashboardServer(d, "127.0.0.1", 0, verbose=False, max_metric_rows=max_rows)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        time.sleep(0.05)
        return srv

    def _get(self, port: int, endpoint: str):
        import urllib.request as _req
        r = _req.urlopen(f"http://127.0.0.1:{port}{endpoint}", timeout=3)
        return json.loads(r.read())

    def test_small_file_not_truncated(self):
        with tempfile.TemporaryDirectory() as d:
            self._make_big_csv(d, n_rows=100)
            srv = self._start_server(d, max_rows=5000)
            try:
                _, port = srv.server_address
                data = self._get(port, "/api/metrics")
                assert data["truncated"] is False
                assert data["total_row_count"] == 100
                assert len(data["rows"]) == 100
            finally:
                srv.shutdown()

    def test_large_file_truncated(self):
        with tempfile.TemporaryDirectory() as d:
            n = 200
            self._make_big_csv(d, n_rows=n)
            max_rows = 50
            srv = self._start_server(d, max_rows=max_rows)
            try:
                _, port = srv.server_address
                data = self._get(port, "/api/metrics")
                assert data["truncated"] is True
                assert data["total_row_count"] == n
                assert len(data["rows"]) == max_rows
                assert data["max_rows"] == max_rows
                # Should be the LAST 50 rows (epoch 151..200)
                last_epoch = data["rows"][-1][0]  # epoch column
                assert last_epoch == pytest.approx(n)
            finally:
                srv.shutdown()

    def test_response_has_required_keys(self):
        with tempfile.TemporaryDirectory() as d:
            self._make_big_csv(d, n_rows=10)
            srv = self._start_server(d, max_rows=5000)
            try:
                _, port = srv.server_address
                data = self._get(port, "/api/metrics")
                for key in ("headers", "rows", "total_row_count", "truncated", "max_rows"):
                    assert key in data, f"Missing key: {key}"
            finally:
                srv.shutdown()

    def test_empty_file_returns_zero(self):
        with tempfile.TemporaryDirectory() as d:
            srv = self._start_server(d, max_rows=5000)
            try:
                _, port = srv.server_address
                data = self._get(port, "/api/metrics")
                assert data["total_row_count"] == 0
                assert data["truncated"] is False
            finally:
                srv.shutdown()

    def test_existing_api_keys_backward_compatible(self, server, base_url):
        """Existing tests rely on 'headers' and 'rows' keys."""
        code, body = get(f"{base_url}/api/metrics")
        assert code == 200
        data = json.loads(body)
        assert "headers" in data
        assert "rows" in data


# =========================================================================== #
# DASH-01: _status_epoch zero-value handling                                    #
# =========================================================================== #

class TestStatusEpochHelper:
    """Unit tests for _status_epoch — zero must not be treated as absent."""

    def test_epoch_nonzero(self):
        assert _status_epoch({"epoch": 5.0}) == 5.0

    def test_epoch_zero_is_valid(self):
        """epoch=0 (parsed as 0.0) must be returned, not skipped."""
        assert _status_epoch({"epoch": 0.0}) == 0.0

    def test_step_zero_is_valid(self):
        """step=0 with no epoch key must return 0, not None."""
        assert _status_epoch({"step": 0.0}) == 0.0

    def test_epoch_preferred_over_step(self):
        """When both keys exist, epoch wins regardless of value."""
        assert _status_epoch({"epoch": 0.0, "step": 99.0}) == 0.0

    def test_epoch_empty_string_falls_back_to_step(self):
        """An empty-string epoch cell is treated as absent."""
        assert _status_epoch({"epoch": "", "step": 3.0}) == 3.0

    def test_both_absent_returns_none(self):
        assert _status_epoch({"train_loss": 0.5}) is None

    def test_epoch_none_falls_back_to_step(self):
        assert _status_epoch({"epoch": None, "step": 7.0}) == 7.0

    def test_step_empty_string_returns_none(self):
        assert _status_epoch({"step": ""}) is None


class TestStatusEndpointZeroEpoch:
    """/api/status must report epoch=0 correctly, not as None."""

    def _make_csv(self, d: str, epoch_val, step_val=None) -> None:
        """Write a metrics.csv whose last row has the given epoch/step."""
        cols = ["timestamp", "epoch"]
        row_prev = ["2025-01-01T00:00:00Z", 99]
        row_zero = ["2025-01-01T00:01:00Z", epoch_val]
        if step_val is not None:
            cols.append("step")
            row_prev.append(0)
            row_zero.append(step_val)
        cols.append("train_loss")
        row_prev.append(0.9)
        row_zero.append(0.5)
        path = os.path.join(d, "metrics.csv")
        import csv as _csv
        with open(path, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(cols)
            w.writerow(row_prev)
            w.writerow(row_zero)

    def _start_server(self, d: str):
        srv = DashboardServer(d, "127.0.0.1", 0, verbose=False)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        time.sleep(0.05)
        return srv

    def test_epoch_zero_reported_as_zero(self):
        with tempfile.TemporaryDirectory() as d:
            self._make_csv(d, epoch_val=0)
            srv = self._start_server(d)
            try:
                _, p = srv.server_address
                _, body = get(f"http://127.0.0.1:{p}/api/status")
                data = json.loads(body)
                # epoch must be 0.0, not None
                assert data["epoch"] is not None, (
                    "/api/status returned epoch=None for a CSV row with epoch=0"
                )
                assert float(data["epoch"]) == pytest.approx(0.0)
            finally:
                srv.shutdown()

    def test_step_zero_reported_as_zero_when_no_epoch_col(self):
        """When CSV has only 'step' column and last row is step=0, report 0."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "metrics.csv")
            import csv as _csv
            with open(path, "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["timestamp", "step", "train_loss"])
                w.writerow(["2025-01-01T00:00:00Z", 0, 0.9])
            srv = self._start_server(d)
            try:
                _, p = srv.server_address
                _, body = get(f"http://127.0.0.1:{p}/api/status")
                data = json.loads(body)
                assert data["epoch"] is not None, (
                    "/api/status returned epoch=None for a CSV row with step=0"
                )
                assert float(data["epoch"]) == pytest.approx(0.0)
            finally:
                srv.shutdown()

    def test_epoch_nonzero_still_works(self, logdir, server, base_url):
        """Regression guard: normal non-zero epoch still reported correctly."""
        _, body = get(f"{base_url}/api/status")
        data = json.loads(body)
        # The module-scoped fixture has epoch=2 in the last CSV row
        assert data["epoch"] is not None
        assert float(data["epoch"]) == pytest.approx(2.0)


# =========================================================================== #
# /api/config endpoint (Prompt 6)                                              #
# =========================================================================== #

class TestConfigEndpoint:
    """The /api/config endpoint exposes safe, non-secret server settings to the UI."""

    def test_returns_200(self, base_url):
        code, _ = get(f"{base_url}/api/config")
        assert code == 200

    def test_shape(self, base_url):
        _, body = get(f"{base_url}/api/config")
        data = json.loads(body)
        for key in ("poll_ms", "refresh_interval_s", "max_metric_rows",
                    "host", "port", "host_is_loopback", "lan_mode",
                    "has_token", "logdir_basename", "stale_after_s"):
            assert key in data, f"missing key: {key}"
        # poll_ms must be a positive integer
        assert isinstance(data["poll_ms"], int)
        assert data["poll_ms"] > 0

    def test_does_not_leak_token(self):
        """has_token must be a bool, never the actual token value."""
        with tempfile.TemporaryDirectory() as d:
            srv = DashboardServer(d, "127.0.0.1", 0,
                                  token="SECRET_DO_NOT_LEAK", verbose=False)
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            time.sleep(0.05)
            try:
                _, p = srv.server_address
                _, body = get(f"http://127.0.0.1:{p}/api/config")
                data = json.loads(body)
                assert data["has_token"] is True
                assert "SECRET_DO_NOT_LEAK" not in body
                # No key whose value is the literal token
                for v in data.values():
                    assert v != "SECRET_DO_NOT_LEAK"
            finally:
                srv.shutdown()

    def test_refresh_interval_clamped(self):
        """refresh_interval is clamped to a sane range [0.5, 60]."""
        with tempfile.TemporaryDirectory() as d:
            srv = DashboardServer(d, "127.0.0.1", 0, verbose=False,
                                  refresh_interval_s=0.01)  # absurdly low
            assert srv.refresh_interval_s >= 0.5
            srv.server_close()
            srv2 = DashboardServer(d, "127.0.0.1", 0, verbose=False,
                                   refresh_interval_s=9999)
            assert srv2.refresh_interval_s <= 60.0
            srv2.server_close()


# =========================================================================== #
# Hardware caching (Prompt 6)                                                  #
# =========================================================================== #

class TestHardwareCaching:
    """Repeated /api/hardware calls must hit a cache, not re-init pynvml."""

    def test_repeated_calls_are_fast(self, base_url):
        """Second call within 1.5s should return ~immediately due to cache."""
        # Warm up
        get(f"{base_url}/api/hardware")
        t0 = time.time()
        for _ in range(5):
            code, _ = get(f"{base_url}/api/hardware")
            assert code == 200
        elapsed = time.time() - t0
        # Five cached calls should complete well under the original
        # ~100ms-per-call cost (psutil cpu_percent interval=0.1).
        assert elapsed < 0.5, f"5 cached calls took {elapsed:.2f}s — caching may be broken"

    def test_cached_age_field(self, base_url):
        """Cached responses include a `cached_age_s` float so the UI can show staleness."""
        get(f"{base_url}/api/hardware")  # warm
        _, body = get(f"{base_url}/api/hardware")
        data = json.loads(body)
        assert "cached_age_s" in data
        assert isinstance(data["cached_age_s"], (int, float))
        assert data["cached_age_s"] >= 0

    def test_collected_at_is_iso(self, base_url):
        _, body = get(f"{base_url}/api/hardware")
        data = json.loads(body)
        assert "collected_at" in data
        assert "T" in data["collected_at"]
        assert ("+" in data["collected_at"]
                or data["collected_at"].endswith("Z")
                or "+00:00" in data["collected_at"])

    def test_unavailable_reasons_present_when_deps_missing(self):
        """If pynvml is missing, /api/hardware tags the reason explicitly."""
        # We can't easily uninstall pynvml mid-test, but if pynvml is installed
        # and there's no GPU, we still get unavailable_reason_pynvml or
        # unavailable_reason_cuda; if pynvml itself is missing, we get the
        # ImportError-flavored reason.
        from tgraphx.dashboard.app import _collect_hardware
        info = _collect_hardware(force=True)
        # At least one of these must explain *something* — the snapshot has
        # to be honest about every "false" availability flag.
        if not info["cuda_available"]:
            assert "unavailable_reason_cuda" in info, info
        if not info["pynvml_available"]:
            # Either pynvml missing or runtime error
            assert ("unavailable_reason_pynvml" in info), info


# =========================================================================== #
# Static asset content — accessibility, palette, print, no CDN, exports        #
# =========================================================================== #

class TestStaticAssetContent:
    """Verify that the served HTML/CSS/JS contain the polish features expected
    by the prompt without spinning up a real browser."""

    def test_html_has_aria_labels(self, base_url):
        _, html = get(f"{base_url}/")
        # Every icon-only button must have an aria-label.
        assert html.count("aria-label") >= 10
        for needed in ("Skip to main content",
                       "Pause auto-refresh",
                       "Refresh now",
                       "Toggle dark/light theme",
                       "Toggle color-blind safe palette",
                       "Open navigation menu",
                       "Close navigation menu"):
            assert needed in html, f"missing aria-label text: {needed}"

    def test_html_has_skip_link(self, base_url):
        _, html = get(f"{base_url}/")
        assert 'class="skip-link"' in html
        assert 'href="#content"' in html

    def test_html_has_stale_banner_slot(self, base_url):
        _, html = get(f"{base_url}/")
        assert 'id="stale-banner"' in html
        assert 'role="alert"' in html

    def test_html_has_referrer_policy(self, base_url):
        _, html = get(f"{base_url}/")
        # Belt-and-braces: the dashboard never wants to leak its URL via Referer
        assert 'name="referrer"' in html

    def test_css_has_focus_visible(self, base_url):
        _, css = get(f"{base_url}/static/dashboard.css")
        assert ":focus-visible" in css

    def test_css_has_print_stylesheet(self, base_url):
        _, css = get(f"{base_url}/static/dashboard.css")
        assert "@media print" in css

    def test_css_has_reduced_motion(self, base_url):
        _, css = get(f"{base_url}/static/dashboard.css")
        assert "prefers-reduced-motion" in css

    def test_css_has_colorblind_palette(self, base_url):
        _, css = get(f"{base_url}/static/dashboard.css")
        assert 'data-palette="cb"' in css
        # Okabe-Ito anchor colors must be present
        assert "#0072B2" in css   # blue
        assert "#E69F00" in css   # orange
        assert "#009E73" in css   # green

    def test_css_has_skip_link_style(self, base_url):
        _, css = get(f"{base_url}/static/dashboard.css")
        assert ".skip-link" in css

    def test_css_has_toolbar(self, base_url):
        _, css = get(f"{base_url}/static/dashboard.css")
        assert ".toolbar" in css
        assert ".tb-btn" in css
        assert ".copy-btn" in css

    def test_js_has_export_helpers(self, base_url):
        _, js = get(f"{base_url}/static/dashboard.js")
        assert "metricsCsv" in js
        assert "chartCsv"   in js
        assert "chartSvg"   in js
        assert "printPage"  in js

    def test_js_has_palette_module(self, base_url):
        _, js = get(f"{base_url}/static/dashboard.js")
        assert "const Palette" in js
        # The JS sets `document.documentElement.dataset.palette = 'cb'`,
        # which corresponds to the [data-palette="cb"] CSS hook.
        assert "dataset.palette" in js or "data-palette" in js
        assert "tgx-palette" in js  # localStorage key for persistence

    def test_js_has_pause_module(self, base_url):
        _, js = get(f"{base_url}/static/dashboard.js")
        assert "const Controls" in js
        assert "togglePause" in js

    def test_js_has_range_selector(self, base_url):
        _, js = get(f"{base_url}/static/dashboard.js")
        assert "const Range" in js
        assert "Range.applySeries" in js or "Range.apply" in js

    def test_js_has_html_escape(self, base_url):
        _, js = get(f"{base_url}/static/dashboard.js")
        assert "function esc(" in js

    def test_js_has_copy_to_clipboard(self, base_url):
        _, js = get(f"{base_url}/static/dashboard.js")
        assert "function copyText" in js

    def test_no_external_cdn_anywhere(self, base_url):
        """No external CDN/font/script references in any served asset."""
        _, html = get(f"{base_url}/")
        _, css  = get(f"{base_url}/static/dashboard.css")
        _, js   = get(f"{base_url}/static/dashboard.js")
        forbidden = ("cdn.jsdelivr", "unpkg.com", "googleapis.com",
                     "cloudflare", "//cdn.", "fonts.gstatic", "google-analytics",
                     "googletagmanager")
        for asset_name, content in (("html", html), ("css", css), ("js", js)):
            for token in forbidden:
                assert token not in content, f"{asset_name} contains forbidden ref: {token}"


# =========================================================================== #
# CLI argument parsing (Prompt 6 new flags)                                    #
# =========================================================================== #

class TestCliFlags:
    """Verify that new CLI flags exist in --help output and don't crash."""

    def test_cli_help_runs(self):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "tgraphx.dashboard", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        for needed in ("--logdir", "--host", "--port", "--token",
                       "--max-metric-rows", "--refresh-interval", "--open-browser"):
            assert needed in result.stdout, f"--help missing: {needed}"

    def test_cli_missing_logdir_fails_clearly(self):
        import subprocess, sys, tempfile, os
        result = subprocess.run(
            [sys.executable, "-m", "tgraphx.dashboard", "--logdir",
             "/path/that/definitely/does/not/exist/abc123"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0
        assert "logdir does not exist" in (result.stderr + result.stdout)


# =========================================================================== #
# LAN IP helper (Prompt 6)                                                     #
# =========================================================================== #

class TestLanIpHelper:
    def test_returns_string_or_none(self):
        from tgraphx.dashboard.app import _detect_lan_ip
        ip = _detect_lan_ip()
        # Either we found a non-loopback IP (string) or detection failed (None);
        # both are valid outcomes — the function must never raise.
        assert ip is None or isinstance(ip, str)
        if isinstance(ip, str):
            assert not ip.startswith("127.")


# =========================================================================== #
# Incremental metrics API — /api/metrics?since_row=<int>                       #
# =========================================================================== #

from tgraphx.dashboard.app import _parse_metrics_incremental

class TestIncrementalMetricsParsing:
    """Unit tests for _parse_metrics_incremental (no HTTP needed)."""

    CSV = "epoch,train_loss\n0,0.9\n1,0.8\n2,0.7\n"

    def test_full_load_since_minus1(self):
        r = _parse_metrics_incremental(self.CSV, -1, 5000)
        assert r["total_row_count"] == 3
        assert len(r["rows"]) == 3
        assert r["latest_row_index"] == 2
        assert not r["reset_required"]

    def test_incremental_after_first_row(self):
        r = _parse_metrics_incremental(self.CSV, 0, 5000)
        assert len(r["rows"]) == 2
        assert r["rows"][0][0] == pytest.approx(1.0)  # epoch 1
        assert r["latest_row_index"] == 2
        assert not r["reset_required"]

    def test_incremental_after_last_row(self):
        r = _parse_metrics_incremental(self.CSV, 2, 5000)
        assert len(r["rows"]) == 0
        assert not r["reset_required"]

    def test_reset_required_when_file_shrinks(self):
        """since_row past the end signals truncation/rotation."""
        r = _parse_metrics_incremental(self.CSV, 100, 5000)
        assert r["reset_required"]

    def test_empty_csv(self):
        r = _parse_metrics_incremental("", -1, 5000)
        assert r["total_row_count"] == 0
        assert r["headers"] == []
        assert r["rows"] == []
        assert not r["reset_required"]

    def test_header_only_csv(self):
        r = _parse_metrics_incremental("epoch,loss\n", 0, 5000)
        assert r["total_row_count"] == 0

    def test_max_rows_cap_on_slice(self):
        """New-row slice is also capped to max_rows."""
        r = _parse_metrics_incremental(self.CSV, -1, 2)  # cap to 2
        assert len(r["rows"]) == 2
        assert r["truncated"]


class TestIncrementalMetricsEndpoint:
    """Integration tests via the live HTTP server."""

    @pytest.fixture(scope="class")
    def inc_server(self, tmp_path_factory):
        d = tmp_path_factory.mktemp("inc")
        with open(d / "metrics.csv", "w", newline="") as f:
            import csv as _csv
            w = _csv.writer(f)
            w.writerow(["epoch", "train_loss"])
            for i in range(5):
                w.writerow([i, 1.0 - i * 0.1])
        srv = DashboardServer(str(d), "127.0.0.1", 0, verbose=False)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        time.sleep(0.05)
        yield srv
        srv.shutdown()

    def _get(self, srv, endpoint):
        h, p = srv.server_address
        _, body = get(f"http://127.0.0.1:{p}/api/{endpoint}")
        return json.loads(body)

    def test_full_metrics_still_works(self, inc_server):
        data = self._get(inc_server, "metrics")
        assert len(data["rows"]) == 5

    def test_since_row_0_returns_remaining(self, inc_server):
        data = self._get(inc_server, "metrics?since_row=0")
        assert "latest_row_index" in data
        assert len(data["rows"]) == 4  # rows after index 0

    def test_since_row_beyond_end_no_rows(self, inc_server):
        data = self._get(inc_server, "metrics?since_row=4")
        assert len(data["rows"]) == 0
        assert not data["reset_required"]

    def test_since_row_far_beyond_reset_required(self, inc_server):
        data = self._get(inc_server, "metrics?since_row=999")
        assert data["reset_required"]

    def test_invalid_since_row_returns_400(self, inc_server):
        h, p = inc_server.server_address
        code, _ = get(f"http://127.0.0.1:{p}/api/metrics?since_row=abc")
        assert code == 400

    def test_missing_metrics_csv_incremental(self):
        """Missing metrics.csv with since_row returns safe empty response."""
        with tempfile.TemporaryDirectory() as d:
            srv = DashboardServer(d, "127.0.0.1", 0, verbose=False)
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            time.sleep(0.05)
            try:
                _, p = srv.server_address
                _, body = get(f"http://127.0.0.1:{p}/api/metrics?since_row=0")
                data = json.loads(body)
                assert data["rows"] == []
                assert not data.get("reset_required", True)
            finally:
                srv.shutdown()


# =========================================================================== #
# Multi-run API — /api/runs and /api/metrics?run=<name>                        #
# =========================================================================== #

from tgraphx.dashboard.app import _list_runs, _safe_run_path

class TestRunHelpers:
    def test_single_run_when_root_has_csv(self, tmp_path):
        (tmp_path / "metrics.csv").write_text("epoch\n1\n")
        r = _list_runs(str(tmp_path))
        assert r["mode"] == "single"
        assert r["runs"] == []

    def test_multi_run_detects_child_csvs(self, tmp_path):
        for name in ("run_a", "run_b"):
            (tmp_path / name).mkdir()
            (tmp_path / name / "metrics.csv").write_text("epoch\n1\n")
        r = _list_runs(str(tmp_path))
        assert r["mode"] == "multi"
        assert set(r["runs"]) == {"run_a", "run_b"}
        assert not r["capped"]

    def test_safe_run_path_rejects_traversal(self, tmp_path):
        (tmp_path / "run_a").mkdir()
        assert _safe_run_path(str(tmp_path), "../etc/passwd") is None
        assert _safe_run_path(str(tmp_path), "") is None
        assert _safe_run_path(str(tmp_path), "/etc/passwd") is None
        assert _safe_run_path(str(tmp_path), "run_a/../../etc") is None

    def test_safe_run_path_valid(self, tmp_path):
        (tmp_path / "run_a").mkdir()
        p = _safe_run_path(str(tmp_path), "run_a")
        assert p is not None
        assert "run_a" in p


class TestMultiRunEndpoints:
    @pytest.fixture(scope="class")
    def multi_server(self, tmp_path_factory):
        d = tmp_path_factory.mktemp("multi")
        for name, loss in [("run_a", 0.9), ("run_b", 0.5)]:
            (d / name).mkdir()
            with open(d / name / "metrics.csv", "w") as f:
                f.write(f"epoch,train_loss\n0,{loss}\n1,{loss*0.9}\n")
        srv = DashboardServer(str(d), "127.0.0.1", 0, verbose=False)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        time.sleep(0.05)
        yield srv
        srv.shutdown()

    def _get(self, srv, ep):
        _, p = srv.server_address
        code, body = get(f"http://127.0.0.1:{p}/api/{ep}")
        return code, json.loads(body)

    def test_runs_endpoint_lists_runs(self, multi_server):
        code, data = self._get(multi_server, "runs")
        assert code == 200
        assert data["mode"] == "multi"
        assert set(data["runs"]) == {"run_a", "run_b"}

    def test_metrics_with_valid_run(self, multi_server):
        code, data = self._get(multi_server, "metrics?run=run_a")
        assert code == 200
        assert len(data["rows"]) == 2

    def test_metrics_with_invalid_run_rejected(self, multi_server):
        code, _ = self._get(multi_server, "metrics?run=../etc/passwd")
        assert code == 400

    def test_metrics_with_unknown_run_rejected(self, multi_server):
        code, _ = self._get(multi_server, "metrics?run=nonexistent_xyz")
        assert code == 400


# =========================================================================== #
# Graph stats — /api/graph_stats + write_graph_stats helper                    #
# =========================================================================== #

class TestGraphStats:
    def _make_server(self, d):
        srv = DashboardServer(str(d), "127.0.0.1", 0, verbose=False)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        time.sleep(0.05)
        return srv

    def test_graph_stats_absent_returns_unavailable(self, tmp_path):
        srv = self._make_server(tmp_path)
        try:
            _, p = srv.server_address
            _, body = get(f"http://127.0.0.1:{p}/api/graph_stats")
            data = json.loads(body)
            assert data.get("available") is False
        finally:
            srv.shutdown()

    def test_graph_stats_present_returned(self, tmp_path):
        stats = {"num_nodes": 9, "num_edges": 24, "density": 0.333}
        (tmp_path / "graph_stats.json").write_text(json.dumps(stats))
        srv = self._make_server(tmp_path)
        try:
            _, p = srv.server_address
            _, body = get(f"http://127.0.0.1:{p}/api/graph_stats")
            data = json.loads(body)
            assert data["available"] is True
            assert data["num_nodes"] == 9
            assert data["num_edges"] == 24
        finally:
            srv.shutdown()

    def test_write_graph_stats_helper(self, tmp_path):
        from tgraphx.tracking import write_graph_stats
        p = str(tmp_path / "graph_stats.json")
        write_graph_stats({"num_nodes": 10, "directed": True}, p)
        data = json.loads(open(p).read())
        assert data["num_nodes"] == 10
        assert data["directed"] is True

    def test_write_graph_stats_from_graph(self, tmp_path):
        from tgraphx import Graph, build_grid_graph
        from tgraphx.tracking import write_graph_stats
        import torch
        nf = torch.randn(4, 8)
        ei = build_grid_graph(2, 2, directed=False, self_loops=True)
        g = Graph(nf, ei)
        p = str(tmp_path / "graph_stats.json")
        write_graph_stats(g, p)
        data = json.loads(open(p).read())
        assert data["num_nodes"] == 4
        assert data["num_edges"] == ei.shape[1]

    def test_malicious_string_in_graph_stats_not_raw_html(self, tmp_path):
        """Dangerous strings in graph_stats.json are served as JSON data,
        not interpreted by the server as HTML."""
        evil = {"builder": "<script>alert(1)</script>", "num_nodes": 1}
        (tmp_path / "graph_stats.json").write_text(json.dumps(evil))
        srv = self._make_server(tmp_path)
        try:
            _, p = srv.server_address
            _, body = get(f"http://127.0.0.1:{p}/api/graph_stats")
            # Body is application/json — the server sends raw JSON.
            # The JS frontend uses esc() before any innerHTML.
            data = json.loads(body)
            assert data["builder"] == "<script>alert(1)</script>"
            # The JSON body itself is not HTML so no XSS here.
        finally:
            srv.shutdown()


# =========================================================================== #
# Offline HTML export                                                           #
# =========================================================================== #

import re as _re
from tgraphx.dashboard.app import export_dashboard_html

class TestOfflineExport:
    """Offline HTML export tests — all use tmp_path directly as logdir."""

    @pytest.fixture()
    def logdir(self, tmp_path):
        (tmp_path / "metrics.csv").write_text("epoch,train_loss\n0,0.9\n1,0.7\n")
        (tmp_path / "run_metadata.json").write_text(
            json.dumps({"run_name": "test", "status": "completed"})
        )
        return tmp_path

    @pytest.fixture()
    def snap_html(self, logdir, tmp_path_factory):
        out = tmp_path_factory.mktemp("exports") / "snap.html"
        export_dashboard_html(str(logdir), str(out))
        return out.read_text()

    def test_creates_html_file(self, logdir, tmp_path_factory):
        out = tmp_path_factory.mktemp("ex2") / "snap.html"
        export_dashboard_html(str(logdir), str(out))
        assert out.exists()

    def test_inlines_css(self, snap_html):
        assert "<style>" in snap_html

    def test_inlines_js(self, snap_html):
        assert "function esc(" in snap_html

    def test_contains_snapshot_data(self, snap_html):
        assert "__TGXSNAP" in snap_html

    def test_no_external_cdn(self, snap_html):
        for bad in ("cdn.jsdelivr", "unpkg.com", "googleapis.com", "cloudflare"):
            assert bad not in snap_html

    def test_no_functional_eval(self, snap_html):
        """eval() must not appear as a callable outside comments."""
        code_only = _re.sub(r'/\*.*?\*/', '', snap_html, flags=_re.DOTALL)
        code_only = _re.sub(r'//[^\n]*', '', code_only)
        assert not _re.search(r'\beval\s*\(', code_only), "eval() found outside comments"

    def test_no_new_function(self, snap_html):
        assert "new Function" not in snap_html

    def test_no_token_in_export(self, snap_html):
        """Token must never appear in exported file."""
        assert "SECRET_TOKEN_123" not in snap_html

    def test_invalid_logdir_raises(self, tmp_path):
        with pytest.raises((ValueError, FileNotFoundError)):
            export_dashboard_html("/nonexistent/path/xyz", str(tmp_path / "out.html"))

    def test_invalid_out_dir_raises(self, tmp_path):
        with pytest.raises((ValueError, FileNotFoundError, OSError)):
            export_dashboard_html(str(tmp_path), "/nonexistent/dir/out.html")


class TestCliExportFlag:
    def test_help_mentions_export_html(self):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "tgraphx.dashboard", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert "--export-html" in result.stdout

    def test_export_via_cli(self, tmp_path):
        import subprocess, sys
        d = tmp_path / "run"
        d.mkdir()
        (d / "metrics.csv").write_text("epoch\n1\n2\n")
        out = tmp_path / "export.html"
        result = subprocess.run(
            [sys.executable, "-m", "tgraphx.dashboard",
             "--logdir", str(d), "--export-html", str(out)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()
        html = out.read_text()
        assert "__TGXSNAP" in html


# =========================================================================== #
# Hardware power/thermal fields                                                 #
# =========================================================================== #

class TestHardwarePowerThermal:
    def test_hardware_works_without_pynvml(self):
        from tgraphx.dashboard.app import _collect_hardware
        info = _collect_hardware(force=True)
        # The collector never raises regardless of pynvml state.
        assert "python" in info
        assert "cuda_available" in info

    def test_thermal_status_computed_when_temp_available(self):
        """Simulate GPU temp and verify thermal_status label."""
        from tgraphx.dashboard.app import _collect_hardware, _HARDWARE_CACHE
        info = _collect_hardware(force=True)
        if info.get("gpu_temp_c") is not None:
            assert info["gpu_thermal_status"] in ("normal", "warm", "near-throttle")
        else:
            assert info.get("gpu_thermal_status", "unknown") == "unknown" or \
                   "unavailable_reason_gpu_temp" in info

    def test_power_unavailable_reason_set_when_pynvml_absent(self):
        from tgraphx.dashboard.app import _collect_hardware
        info = _collect_hardware(force=True)
        if not info["pynvml_available"]:
            assert "unavailable_reason_pynvml" in info

    def test_js_has_thermal_chip_rendering(self, base_url):
        _, js = get(f"{base_url}/static/dashboard.js")
        assert "thermal-chip" in js

    def test_css_has_thermal_chip_styles(self, base_url):
        _, css = get(f"{base_url}/static/dashboard.css")
        assert ".thermal-chip" in css
        assert "thermal-near-throttle" in css


# =========================================================================== #
# JavaScript safety checks                                                      #
# =========================================================================== #

class TestJsSafety:
    def test_no_eval_in_js(self, base_url):
        _, js = get(f"{base_url}/static/dashboard.js")
        # No functional eval() call (comments are acceptable explanations).
        code_only = _re.sub(r'/\*.*?\*/', '', js, flags=_re.DOTALL)
        code_only = _re.sub(r'//[^\n]*', '', code_only)
        assert not _re.search(r'\beval\s*\(', code_only), "eval() found in JS code"

    def test_no_new_function_in_js(self, base_url):
        _, js = get(f"{base_url}/static/dashboard.js")
        assert "new Function(" not in js

    def test_js_has_tooltip_module(self, base_url):
        _, js = get(f"{base_url}/static/dashboard.js")
        assert "const Tooltip" in js
        assert "chart-tooltip" in js

    def test_js_has_run_selector(self, base_url):
        _, js = get(f"{base_url}/static/dashboard.js")
        assert "RunSelector" in js

    def test_js_incremental_since_row(self, base_url):
        _, js = get(f"{base_url}/static/dashboard.js")
        assert "since_row" in js
        assert "latestRowIndex" in js

    def test_js_snapshot_mode(self, base_url):
        _, js = get(f"{base_url}/static/dashboard.js")
        assert "__TGXSNAP" in js
        assert "snapshotMode" in js


# =========================================================================== #
# write_graph_stats import from top-level                                       #
# =========================================================================== #

class TestWriteGraphStatsTopLevel:
    def test_importable_from_tgraphx(self):
        from tgraphx import write_graph_stats
        assert callable(write_graph_stats)

    def test_importable_from_tracking(self):
        from tgraphx.tracking import write_graph_stats
        assert callable(write_graph_stats)
