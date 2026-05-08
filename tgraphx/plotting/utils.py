"""Plotting utilities: palettes, backends, save helpers.

Stability: Beta (v0.3.2+).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

__all__ = ["save_figure", "OKABE_ITO", "get_color_cycle"]

# Okabe-Ito colorblind-friendly palette (8 colors).
OKABE_ITO = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]


def get_color_cycle(n: int) -> list:
    """Return n colors cycling through the Okabe-Ito palette."""
    return [OKABE_ITO[i % len(OKABE_ITO)] for i in range(n)]


def _ensure_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Matplotlib is required for TGraphX plotting utilities.  "
            "Install with: pip install matplotlib"
        ) from exc


def save_figure(
    fig,
    path: str,
    formats: Sequence[str] = ("png",),
    dpi: int = 150,
    tight: bool = True,
) -> list:
    """Save a Matplotlib figure in one or more formats.

    Args:
        fig: Matplotlib ``Figure`` object.
        path: Base path without extension.  E.g. ``"/tmp/plot"``.
        formats: Iterable of format strings: ``"png"``, ``"svg"``,
            ``"pdf"``.
        dpi: DPI for raster formats.
        tight: Apply ``tight_layout`` before saving.

    Returns:
        List of written file paths.
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    if tight:
        try:
            fig.tight_layout()
        except Exception:  # pragma: no cover
            pass

    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        p = base.with_suffix(f".{fmt}")
        fig.savefig(str(p), dpi=dpi, bbox_inches="tight")
        written.append(str(p))
    return written


def _use_agg() -> None:
    """Switch to Agg backend if no display is available (headless-safe)."""
    import os
    import matplotlib
    if not os.environ.get("DISPLAY") and matplotlib.get_backend() == "TkAgg":
        try:
            matplotlib.use("Agg")
        except Exception:  # pragma: no cover
            pass
