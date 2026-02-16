"""Peptide sequence utility functions."""

from __future__ import annotations


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
    if not tokens:
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
