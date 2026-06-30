"""Calibration features for PSM confidence calibration.

This module re-exports all feature classes and utility functions from the
`winnow.calibration.features` subpackage for backwards compatibility.

For new code, prefer importing directly from the subpackage:

    from winnow.calibration.features import FragmentMatchFeatures

Or import specific modules:

    from winnow.calibration.features.fragment_match import FragmentMatchFeatures
"""

import warnings

# Re-export everything from the features subpackage for backwards compatibility
from winnow.calibration.features import (
    # Base classes
    CalibrationFeatures,
    FeatureDependency,
    # Feature classes
    FragmentMatchFeatures,
    ChimericFeatures,
    MassErrorPPMFeature,
    MassErrorDaFeature,
    BeamFeatures,
    RetentionTimeFeature,
    SequenceFeatures,
    TokenScoreFeatures,
    PeptideLanguageModelBackend,
    PeptideLanguageModelFeature,
    PeptideLanguageModelResult,
    normalize_peptide_for_plm,
    # Helper functions
    require_beam_predictions,
    validate_model_input_params,
    resolve_model_inputs,
    # Peak matching
    IonMatchResult,
    IonIdentificationResult,
    find_matching_ions as _find_matching_ions,
    compute_ion_identifications,
    # Ion coverage features
    compute_longest_ion_series,
    compute_complementary_ion_count,
    compute_max_ion_gap,
)

# Backwards-compatible aliases for private functions (with underscore prefix)
_require_beam_predictions = require_beam_predictions
_validate_model_input_params = validate_model_input_params
_resolve_model_inputs = resolve_model_inputs


def find_matching_ions(*args, **kwargs) -> IonMatchResult:
    """Deprecated re-export. Prefer ``winnow.calibration.features.find_matching_ions``."""
    warnings.warn(
        "Import find_matching_ions from winnow.calibration.features; "
        "the calibration_features re-export is deprecated. "
        "Access results via IonMatchResult fields (e.g. result.match_rate).",
        DeprecationWarning,
        stacklevel=2,
    )
    return _find_matching_ions(*args, **kwargs)


__all__ = [
    # Base classes
    "CalibrationFeatures",
    "FeatureDependency",
    # Feature classes
    "FragmentMatchFeatures",
    "ChimericFeatures",
    "MassErrorPPMFeature",
    "MassErrorDaFeature",
    "BeamFeatures",
    "RetentionTimeFeature",
    "SequenceFeatures",
    "TokenScoreFeatures",
    "PeptideLanguageModelBackend",
    "PeptideLanguageModelFeature",
    "PeptideLanguageModelResult",
    "normalize_peptide_for_plm",
    # Helper functions
    "require_beam_predictions",
    "validate_model_input_params",
    "resolve_model_inputs",
    # Peak matching
    "IonMatchResult",
    "IonIdentificationResult",
    "find_matching_ions",
    "compute_ion_identifications",
    # Ion coverage features
    "compute_longest_ion_series",
    "compute_complementary_ion_count",
    "compute_max_ion_gap",
    # Backwards-compatible aliases
    "_require_beam_predictions",
    "_validate_model_input_params",
    "_resolve_model_inputs",
]
