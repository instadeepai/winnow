"""Calibration features subpackage.

This module provides feature classes and utility functions for PSM calibration.
"""

# Base classes
from winnow.calibration.features.base import (
    CalibrationFeatures,
    FeatureDependency,
)

# Feature classes
from winnow.calibration.features.fragment_match import FragmentMatchFeatures
from winnow.calibration.features.chimeric import ChimericFeatures
from winnow.calibration.features.mass_error import MassErrorFeature
from winnow.calibration.features.beam import BeamFeatures
from winnow.calibration.features.retention_time import RetentionTimeFeature
from winnow.calibration.features.sequence import SequenceFeatures
from winnow.calibration.features.token_score import TokenScoreFeatures

# Utility functions
from winnow.calibration.features.utils import (
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
]
