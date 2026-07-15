"""Peptide sequence utility functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl


def _is_missing_cell(value: object) -> bool:
    """Return True when a peptide cell is absent (None, NaN, pd.NA)."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return False


def is_usable_peptide_label(value: object) -> bool:
    """Return True when a raw peptide cell contains a label or sequence string."""
    if _is_missing_cell(value):
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    if isinstance(value, pl.Series):
        return len(value) > 0
    if isinstance(value, np.ndarray):
        return value.size > 0
    if hasattr(value, "tolist"):
        tokens = value.tolist()
        return isinstance(tokens, (list, tuple)) and len(tokens) > 0
    return True


def as_token_list(value: object) -> list[str] | None:
    """Coerce a metadata cell to a non-empty AA token list, or ``None``.

    Accepts container types that already hold token strings. Does not parse raw
    peptide strings; use
    :func:`winnow.datasets.data_loaders.utils.normalize_peptide_cell` for that.
    """
    if _is_missing_cell(value):
        return None
    if isinstance(value, pl.Series):
        tokens = value.to_list()
        return list(tokens) if tokens else None
    if isinstance(value, (list, tuple)):
        tokens = list(value)
        return tokens if tokens else None
    if isinstance(value, np.ndarray):
        tokens = value.tolist()
        return tokens if isinstance(tokens, list) and tokens else None
    return None


def is_valid_peptide_tokens(value: object) -> bool:
    """Return True when a cell holds a non-empty token list (not a raw string)."""
    return as_token_list(value) is not None


def _is_standalone_modification(token: str) -> bool:
    """Check if a token is a standalone modification (not attached to an amino acid).

    Standalone modifications start with a non-alphabetic character, such as:
    - '[UNIMOD:1]' (UNIMOD notation)
    - '(+42.01)' (mass notation in parentheses)
    - '+42.01' (raw mass notation)

    Args:
        token: A single token from a tokenized peptide sequence.

    Returns:
        True if the token is a standalone modification, False otherwise.
    """
    return bool(token) and not token[0].isalpha()


def _normalize_token_list(
    tokens: list[str] | tuple[str, ...] | None,
) -> list[str] | None:
    """Coerce token containers from pandas/numpy into a plain Python list."""
    if tokens is None:
        return None
    if hasattr(tokens, "tolist"):
        return tokens.tolist()
    if isinstance(tokens, tuple):
        return list(tokens)
    return tokens


def tokens_to_proforma(tokens: list[str] | None) -> str:
    """Convert a list of tokens to a ProForma compliant string.

    Adds a hyphen after N-terminal modifications and before C-terminal modifications.
    Terminal modifications are detected by checking if the first/last token starts
    with a non-alphabetic character (e.g., '[UNIMOD:1]', '(+42.01)').

    Args:
        tokens: List of amino acid/modification tokens, or None.

    Returns:
        ProForma-compliant peptide string. Returns empty string if tokens is None
        or empty.

    Examples:
        >>> tokens_to_proforma(["P", "E", "P", "T", "I", "D", "E"])
        'PEPTIDE'
        >>> tokens_to_proforma(["[UNIMOD:1]", "P", "E", "P", "T", "I", "D", "E"])
        '[UNIMOD:1]-PEPTIDE'
        >>> tokens_to_proforma(["P", "E", "P", "T", "I", "D", "E", "[UNIMOD:2]"])
        'PEPTIDE-[UNIMOD:2]'
        >>> tokens_to_proforma(["[UNIMOD:1]", "P", "E", "P", "T", "I", "D", "E", "[UNIMOD:2]"])
        '[UNIMOD:1]-PEPTIDE-[UNIMOD:2]'
        >>> tokens_to_proforma(["M[UNIMOD:35]", "P", "E", "P", "T", "I", "D", "E"])
        'M[UNIMOD:35]PEPTIDE'
    """
    tokens = _normalize_token_list(tokens)
    if tokens is None or len(tokens) == 0:
        return ""

    # Work with a mutable copy
    tokens = list(tokens)

    prefix = ""
    suffix = ""

    # Check for N-terminal modification (first token is standalone mod)
    if len(tokens) > 1 and _is_standalone_modification(tokens[0]):
        prefix = tokens[0] + "-"
        tokens = tokens[1:]

    # Check for C-terminal modification (last token is standalone mod)
    if len(tokens) > 1 and _is_standalone_modification(tokens[-1]):
        suffix = "-" + tokens[-1]
        tokens = tokens[:-1]

    return prefix + "".join(tokens) + suffix
