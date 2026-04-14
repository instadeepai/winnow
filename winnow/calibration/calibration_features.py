"""Calibration features for PSM confidence calibration.

This module re-exports all feature classes and utility functions from the
`winnow.calibration.features` subpackage for backwards compatibility.

For new code, prefer importing directly from the subpackage:

    from winnow.calibration.features import FragmentMatchFeatures
    from winnow.calibration.features import compute_spectrum_match_quality

Or import specific modules:

    from winnow.calibration.features.fragment_match import FragmentMatchFeatures
"""

# Re-export everything from the features subpackage for backwards compatibility
from winnow.calibration.features import (
    # Base classes
    CalibrationFeatures,
    FeatureDependency,
    # Feature classes
    FragmentMatchFeatures,
    ChimericFeatures,
    MassErrorFeature,
    BeamFeatures,
    RetentionTimeFeature,
    SequenceFeatures,
    TokenScoreFeatures,
    # Helper functions
    require_beam_predictions,
    validate_model_input_params,
    resolve_model_inputs,
    # Peak matching
    find_matching_ions,
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

__all__ = [
    # Base classes
    "CalibrationFeatures",
    "FeatureDependency",
    # Feature classes
    "FragmentMatchFeatures",
    "ChimericFeatures",
    "MassErrorFeature",
    "BeamFeatures",
    "RetentionTimeFeature",
    "SequenceFeatures",
    "TokenScoreFeatures",
    # Helper functions
    "require_beam_predictions",
    "validate_model_input_params",
    "resolve_model_inputs",
    # Peak matching
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
