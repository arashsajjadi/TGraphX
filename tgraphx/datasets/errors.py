"""Dataset-related exception types.

Keeping these in a dedicated module lets users catch a specific failure mode
without having to ``import tgraphx`` first.
"""
from __future__ import annotations


class DatasetError(Exception):
    """Base class for all TGraphX dataset errors."""


class DatasetNotFoundError(DatasetError):
    """Raised when a dataset name is not in the registry."""


class DatasetFilesNotFoundError(DatasetError):
    """Raised when required raw/processed files are missing.

    Carries hints about whether the user can fix it with ``download=True``
    or by providing a different ``root`` path.
    """


class DatasetIntegrityError(DatasetError):
    """Raised when a downloaded artefact fails checksum verification."""


class DatasetExtractionError(DatasetError):
    """Raised when archive extraction fails or detects a path-traversal attempt."""


class OptionalDependencyError(ImportError):
    """Optional integration is unavailable.

    Subclass of :class:`ImportError` so existing ``except ImportError`` blocks
    still work, while letting downstream code catch this specific case.
    """

    def __init__(self, package: str, install_hint: str | None = None) -> None:
        msg = (
            f"This feature requires the optional dependency '{package}', which is "
            f"not installed."
        )
        if install_hint:
            msg += f"\n\nInstall hint: {install_hint}"
        super().__init__(msg)
        self.package = package
        self.install_hint = install_hint
