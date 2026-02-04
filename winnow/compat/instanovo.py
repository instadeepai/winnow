"""Compatibility layer for InstaNovo imports across versions.

This module provides a stable import interface for InstaNovo dependencies,
handling API changes between different InstaNovo versions.
"""

import warnings

# Try to import from new InstaNovo location first (>= 1.2.0
try:
    from instanovo.inference.interfaces import ScoredSequence
except ImportError:
    # Fall back to old InstaNovo location (< 1.2.0)
    try:
        from instanovo.constants import ScoredSequence

        warnings.warn(
            "You are using an older version of InstaNovo with deprecated import paths. "
            "Please upgrade to InstaNovo >= 1.2.0 for the latest features and API. "
            "Support for older InstaNovo versions will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
    except ImportError as e:
        raise ImportError(
            "Failed to import ScoredSequence from InstaNovo. "
            "Please ensure InstaNovo is installed: pip install instanovo>=1.2.0"
        ) from e

__all__ = ["ScoredSequence"]
