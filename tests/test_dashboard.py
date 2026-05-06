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
