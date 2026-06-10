"""Shared helpers for interpreting Hydra CLI override paths."""

from __future__ import annotations

from typing import List, Optional, Set


def hydra_override_keys(overrides: Optional[List[str]]) -> Set[str]:
    """Normalised Hydra override paths (without leading ``+`` / ``~``)."""
    keys: Set[str] = set()
    if not overrides:
        return keys
    for item in overrides:
        if "=" not in item:
            continue
        path = item.split("=", 1)[0].lstrip("~+")
        keys.add(path)
    return keys
