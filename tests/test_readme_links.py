"""Validate local links in README.md and docs/index.md don't point to missing files.

We only check relative links (those that resolve to files on disk).
External http/https links are not checked in CI.
"""
from __future__ import annotations

import re
from pathlib import Path


def _extract_links(text: str, base: Path):
    """Yield (link_text, resolved_path) for all markdown [text](path) in text."""
    for m in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', text):
        href = m.group(2)
        # Skip anchors, external URLs, mailto links, and empty links.
        if href.startswith("http") or href.startswith("#") or href.startswith("mailto:") or not href:
            continue
        # Strip in-page anchors.
        href = href.split("#")[0]
        if not href:
            continue
        yield m.group(1), base / href


class TestReadmeLinks:

    def test_readme_docs_links_exist(self):
        readme = Path("README.md")
        base = Path(".")
        missing = []
        for text, path in _extract_links(readme.read_text(), base):
            if not path.exists():
                missing.append(str(path))
        assert not missing, f"README.md has {len(missing)} broken local links: {missing}"

    def test_docs_index_links_exist(self):
        index = Path("docs/index.md")
        base = Path("docs")
        missing = []
        text = index.read_text()
        for m in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', text):
            href = m.group(2)
            if href.startswith("http") or href.startswith("#") or not href:
                continue
            href = href.split("#")[0]
            if not href:
                continue
            path = base / href
            if not path.exists():
                missing.append(str(path))
        assert not missing, f"docs/index.md has {len(missing)} broken local links: {missing}"


class TestReadmeForbiddenClaims:

    def test_no_positive_pyg_dgl_replacement_claim(self):
        """README must not positively claim PyG/DGL drop-in replacement.
        Ecosystem-interop context is acceptable."""
        text = Path("README.md").read_text().lower()
        # Direct positive replacement claims are forbidden.
        assert "drop-in replacement for pyg" not in text
        assert "drop-in replacement for dgl" not in text

    def test_no_sota_claim(self):
        text = Path("README.md").read_text().lower()
        assert "state of the art" not in text
        assert "state-of-the-art" not in text

    def test_no_unsupported_billion_edge_claim(self):
        text = Path("README.md").read_text().lower()
        assert "billion-edge production" not in text

    def test_no_scary_symbols(self):
        text = Path("README.md").read_text()
        scary = ["⚠️", "❌", "⛔", "⏳", "🚫"]
        found = [s for s in scary if s in text]
        assert not found, f"README contains scary symbols: {found}"

    def test_stability_labels_present(self):
        text = Path("README.md").read_text()
        assert "Beta" in text
        assert "Experimental" in text

    def test_limitations_doc_linked(self):
        text = Path("README.md").read_text()
        assert "docs/limitations.md" in text

    def test_roadmap_doc_linked(self):
        text = Path("README.md").read_text()
        assert "docs/roadmap.md" in text
