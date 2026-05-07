"""Packaging regression tests.

These tests verify that non-Python assets bundled with TGraphX are accessible
after installation so that bugs like the missing dashboard static files are
caught before a release is published.

The tests use ``importlib.resources`` (Python 3.9+) to look up files inside
the installed package tree, which works correctly for both editable installs
(``pip install -e .``) and regular wheel installs.  They do *not* invoke
``python -m build`` because that would make the test suite too slow for CI.

A separate manual verification command is included in the module docstring:

    rm -rf dist/ build/
    python -m build
    python -m pytest tests/test_packaging.py -q
    python - <<'PY'
    import zipfile, glob
    wheel = glob.glob("dist/*.whl")[0]
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        print("dashboard.css:", any("dashboard.css" in n for n in names))
        print("dashboard.js :", any("dashboard.js"  in n for n in names))
        assert any("dashboard.css" in n for n in names)
        assert any("dashboard.js"  in n for n in names)
    PY
"""

import importlib.resources
import os


def _static_path() -> str:
    """Return the absolute path to ``tgraphx/dashboard/static/``."""
    pkg = importlib.resources.files("tgraphx.dashboard")
    # ``files()`` returns a ``Traversable``; resolve to a real path so we can
    # use ``os.path`` freely even under editable installs.
    return str(pkg.joinpath("static"))


def test_dashboard_css_present():
    """dashboard.css must be distributed alongside the Python source."""
    static = _static_path()
    css = os.path.join(static, "dashboard.css")
    assert os.path.isfile(css), (
        f"dashboard.css not found at {css!r}. "
        "This means the file was not packaged. "
        "Add '[tool.setuptools.package-data]' to pyproject.toml."
    )
    assert os.path.getsize(css) > 0, "dashboard.css exists but is empty"


def test_dashboard_js_present():
    """dashboard.js must be distributed alongside the Python source."""
    static = _static_path()
    js = os.path.join(static, "dashboard.js")
    assert os.path.isfile(js), (
        f"dashboard.js not found at {js!r}. "
        "This means the file was not packaged. "
        "Add '[tool.setuptools.package-data]' to pyproject.toml."
    )
    assert os.path.getsize(js) > 0, "dashboard.js exists but is empty"


def test_dashboard_static_content_readable():
    """Both static files must be non-empty and decodable as UTF-8."""
    static = _static_path()
    for name in ("dashboard.css", "dashboard.js"):
        path = os.path.join(static, name)
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert len(content) > 100, (
            f"{name} has suspiciously little content ({len(content)} chars)"
        )
