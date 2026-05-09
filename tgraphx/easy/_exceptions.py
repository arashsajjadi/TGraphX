"""User-facing exception hierarchy for TGraphX easy mode."""
from __future__ import annotations


class TGraphXError(ValueError):
    """Base user-facing error from TGraphX easy mode."""


class TGraphXConfigError(TGraphXError):
    """Invalid configuration for an easy-mode workflow."""


class TGraphXLabelError(TGraphXError):
    """Graph labels are missing or have an unsupported type."""


class TGraphXShapeError(TGraphXError):
    """Tensor shape contract violated."""


class TGraphXUnknownNameError(TGraphXError):
    """Unknown algorithm / model / sampler name."""
