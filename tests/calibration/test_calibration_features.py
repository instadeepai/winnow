"""Tests for backwards-compatible exports in calibration_features."""

import warnings

from winnow.calibration.calibration_features import find_matching_ions
from winnow.calibration.features import IonMatchResult


class TestFindMatchingIonsBackwardsCompat:
    def test_legacy_import_emits_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = find_matching_ions(
                [100.0],
                [100.0],
                [1000.0],
                ["b1+1"],
            )

        assert len(caught) == 1
        assert issubclass(caught[0].category, DeprecationWarning)
        assert "winnow.calibration.features" in str(caught[0].message)
        assert isinstance(result, IonMatchResult)
        assert result.match_rate == 1.0
        assert result.match_intensity == 1.0
        assert result.matched_ion_annotations == ["b1+1"]
        assert result.matched_ion_mz == [100.0]
        assert result.matched_ion_intensities == [1000.0]
