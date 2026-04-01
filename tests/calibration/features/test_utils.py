"""Unit tests for winnow calibration feature utility functions."""

import pytest
import pandas as pd
from typing import Optional
from unittest.mock import patch

from winnow.calibration.features.utils import (
    find_matching_ions,
    validate_model_input_params,
    resolve_model_inputs,
    compute_longest_ion_series,
    compute_complementary_ion_count,
    compute_max_ion_gap,
    compute_b_y_intensity_ratio,
    compute_spectral_angle,
    compute_xcorr,
)
from winnow.calibration.features.fragment_match import FragmentMatchFeatures
from winnow.calibration.features.chimeric import ChimericFeatures
from winnow.datasets.calibration_dataset import CalibrationDataset


class TestIonMatchFunctions:
    """Test utility functions used by calibration features for ion matching."""

    def test_find_matching_ions_exact_match(self):
        """Test find_matching_ions with exact m/z matches."""
        source_mz = [100.0, 200.0, 300.0]
        target_mz = [100.0, 200.0, 400.0]
        target_intensities = [1000.0, 2000.0, 4000.0]
        tolerance = 0.01

        match_fraction, average_intensity, *_ = find_matching_ions(
            source_mz,
            target_mz,
            target_intensities,
            source_annotations=["b1+1", "b2+1", "b3+1"],
            mz_tolerance=tolerance,
        )

        # Function returns fraction of matched ions (2/3) and normalised intensity
        assert match_fraction == pytest.approx(
            2 / 3, rel=1e-10, abs=1e-10
        )  # 2 matches out of 3 source ions
        total_intensity = sum(target_intensities)  # 7000.0
        match_intensity = 1000.0 + 2000.0  # 3000.0
        expected_intensity = match_intensity / total_intensity  # 3000/7000
        assert average_intensity == pytest.approx(
            expected_intensity, rel=1e-10, abs=1e-10
        )

    def test_find_matching_ions_with_tolerance(self):
        """Test find_matching_ions with tolerance-based matching."""
        source_mz = [100.0, 200.0]
        target_mz = [100.005, 200.01]  # Within tolerance
        target_intensities = [1000.0, 2000.0]
        tolerance = 0.02

        match_fraction, average_intensity, *_ = find_matching_ions(
            source_mz,
            target_mz,
            target_intensities,
            source_annotations=["b1+1", "b2+1"],
            mz_tolerance=tolerance,
        )

        # All source ions match, so fraction = 1.0
        assert match_fraction == 1.0  # 2/2 matches
        # All target intensity is matched, so normalised intensity = 1.0
        assert average_intensity == 1.0  # 3000/3000

    def test_find_matching_ions_outside_tolerance(self):
        """Test find_matching_ions with m/z values outside tolerance."""
        source_mz = [100.0, 200.0]
        target_mz = [100.05, 200.1]  # Outside tolerance
        target_intensities = [1000.0, 2000.0]
        tolerance = 0.01

        match_fraction, average_intensity, *_ = find_matching_ions(
            source_mz,
            target_mz,
            target_intensities,
            source_annotations=["b1+1", "b2+1"],
            mz_tolerance=tolerance,
        )

        assert match_fraction == 0
        assert average_intensity == 0.0

    def test_find_matching_ions_no_matches(self):
        """Test find_matching_ions with no matches."""
        source_mz = [100.0]
        target_mz = [200.0]  # No match within tolerance
        target_intensities = [1000.0]

        match_fraction, average_intensity, *_ = find_matching_ions(
            source_mz,
            target_mz,
            target_intensities,
            source_annotations=["b1+1"],
            mz_tolerance=0.01,
        )
        assert match_fraction == 0.0  # 0 matches / 1 source ion
        assert average_intensity == 0.0  # 0 match intensity / 1000 total intensity

    def test_find_matching_ions_prevents_double_matching(self):
        """Test that each observed peak can only be matched once."""
        # Two theoretical ions very close together, only one observed peak
        source_mz = [100.0, 100.005]
        target_mz = [100.002]  # Within tolerance of both source ions
        target_intensities = [1000.0]
        tolerance = 0.02

        match_fraction, average_intensity, matched_annotations, *_ = find_matching_ions(
            source_mz,
            target_mz,
            target_intensities,
            source_annotations=["b1+1", "b2+1"],
            mz_tolerance=tolerance,
        )

        # Only one source ion should match (the first one gets the peak)
        assert match_fraction == pytest.approx(0.5)  # 1 match / 2 source ions
        assert len(matched_annotations) == 1
        assert matched_annotations[0] == "b1+1"

    def test_find_matching_ions_fallback_to_second_best(self):
        """Test fallback to next nearest peak when closest is already matched."""
        # First source ion takes the middle peak, second should fall back to the other
        source_mz = [100.0, 100.01]
        target_mz = [100.002, 100.015]  # Both within tolerance of second source
        target_intensities = [1000.0, 2000.0]
        tolerance = 0.02

        match_fraction, average_intensity, matched_annotations, *_ = find_matching_ions(
            source_mz,
            target_mz,
            target_intensities,
            source_annotations=["b1+1", "b2+1"],
            mz_tolerance=tolerance,
        )

        # Both source ions should match (to different observed peaks)
        assert match_fraction == 1.0  # 2 matches / 2 source ions
        assert len(matched_annotations) == 2

    def test_find_matching_ions_isotope_masking(self):
        """Test that isotope peaks are masked and not available for subsequent M0 matches."""
        # First source ion at 100.0 has isotope at ~101.003 (for +1 charge)
        # Second source ion at 101.0 should NOT match the isotope peak
        source_mz = [100.0, 101.0]
        # Observed peaks: M0 at 100.0, M+1 isotope at 101.003, and another at 150.0
        target_mz = [100.0, 101.003, 150.0]
        target_intensities = [1000.0, 500.0, 2000.0]
        tolerance = 0.02

        match_fraction, average_intensity, matched_annotations, *_ = find_matching_ions(
            source_mz,
            target_mz,
            target_intensities,
            source_annotations=["b1+1", "b2+1"],  # +1 charge, isotope spacing ~1.003
            mz_tolerance=tolerance,
        )

        # First ion matches (M0 at 100.0, isotope at 101.003)
        # Second ion at 101.0 should NOT match 101.003 (already claimed as isotope)
        assert match_fraction == pytest.approx(0.5)  # Only 1 M0 match / 2 source ions
        assert len(matched_annotations) == 1
        assert matched_annotations[0] == "b1+1"
        # Intensity should include M0 (1000) + isotope (500) = 1500 / 3500 total
        assert average_intensity == pytest.approx(1500.0 / 3500.0)

    def test_find_matching_ions_returns_aligned_m0_intensities(self):
        """Test that aligned_m0_intensities has same length as source_mz."""
        source_mz = [100.0, 200.0, 300.0]
        target_mz = [100.0, 200.0, 400.0]  # 300.0 has no match
        target_intensities = [1000.0, 2000.0, 4000.0]
        tolerance = 0.01

        (
            _match_fraction,
            _average_intensity,
            _matched_annotations,
            _matched_mz,
            _matched_intensities_isotopic,
            aligned_m0_intensities,
        ) = find_matching_ions(
            source_mz,
            target_mz,
            target_intensities,
            source_annotations=["b1+1", "b2+1", "b3+1"],
            mz_tolerance=tolerance,
        )

        # aligned_m0_intensities should have same length as source_mz
        assert len(aligned_m0_intensities) == len(source_mz)
        # First two should have M0 intensities, third should be 0.0 (no match)
        assert aligned_m0_intensities[0] == 1000.0  # M0 intensity for 100.0
        assert aligned_m0_intensities[1] == 2000.0  # M0 intensity for 200.0
        assert aligned_m0_intensities[2] == 0.0  # No match for 300.0

    def test_find_matching_ions_aligned_intensities_are_m0_only(self):
        """Test that aligned_m0_intensities contains only M0, not isotope intensities."""
        source_mz = [100.0]
        # Observed: M0 at 100.0 and M+1 isotope at ~101.003
        target_mz = [100.0, 101.003]
        target_intensities = [1000.0, 500.0]
        tolerance = 0.02

        (
            _match_fraction,
            _average_intensity,
            _matched_annotations,
            _matched_mz,
            matched_intensities_isotopic,
            aligned_m0_intensities,
        ) = find_matching_ions(
            source_mz,
            target_mz,
            target_intensities,
            source_annotations=["b1+1"],  # +1 charge
            mz_tolerance=tolerance,
        )

        # aligned_m0_intensities should only have M0 intensity
        assert aligned_m0_intensities[0] == 1000.0
        # matched_intensities_isotopic should include M0 + isotope
        assert matched_intensities_isotopic[0] == 1500.0  # 1000 + 500


class TestModelInputHelpers:
    """Tests for validate_model_input_params and resolve_model_inputs utility functions."""

    def test_validate_no_conflict_passes(self):
        """No error when constants and columns have disjoint keys."""
        validate_model_input_params(
            {"collision_energies": 25},
            {"fragmentation_types": "col_frag"},
        )

    def test_validate_one_none_passes(self):
        """No error when one of the dicts is None."""
        validate_model_input_params({"collision_energies": 25}, None)
        validate_model_input_params(None, {"collision_energies": "ce_col"})

    def test_validate_both_none_passes(self):
        """No error when both dicts are None."""
        validate_model_input_params(None, None)

    def test_validate_conflict_raises(self):
        """ValueError raised when the same key appears in both dicts."""
        with pytest.raises(ValueError, match="collision_energies"):
            validate_model_input_params(
                {"collision_energies": 25},
                {"collision_energies": "ce_col"},
            )

    def test_validate_conflict_lists_all_conflicting_keys(self):
        """Error message includes all conflicting keys."""
        with pytest.raises(ValueError, match="fragmentation_types") as exc_info:
            validate_model_input_params(
                {"collision_energies": 25, "fragmentation_types": "HCD"},
                {"collision_energies": "ce_col", "fragmentation_types": "frag_col"},
            )
        assert "collision_energies" in str(exc_info.value)
        assert "fragmentation_types" in str(exc_info.value)

    def test_conflict_detected_at_construction_time_fragment_match_feature(self):
        """FragmentMatchFeatures raises ValueError at construction when keys conflict."""
        with pytest.raises(ValueError, match="collision_energies"):
            FragmentMatchFeatures(
                mz_tolerance=0.02,
                model_input_constants={"collision_energies": 25},
                model_input_columns={"collision_energies": "ce_col"},
            )

    def test_conflict_detected_at_construction_time_chimeric_features(self):
        """ChimericFeatures raises ValueError at construction when keys conflict."""
        with pytest.raises(ValueError, match="collision_energies"):
            ChimericFeatures(
                mz_tolerance=0.02,
                model_input_constants={"collision_energies": 25},
                model_input_columns={"collision_energies": "ce_col"},
            )

    def test_resolve_constant_is_tiled(self):
        """A constant value is tiled across all rows."""
        inputs = self._make_inputs(3)
        metadata = self._make_metadata(3)
        result = resolve_model_inputs(
            inputs=inputs,
            metadata=metadata,
            required_model_inputs=[
                "peptide_sequences",
                "precursor_charges",
                "collision_energies",
            ],
            auto_populated={"peptide_sequences", "precursor_charges"},
            constants={"collision_energies": 30},
            columns=None,
            model_name="TestModel",
        )
        assert list(result["collision_energies"]) == [30, 30, 30]

    def test_resolve_column_is_used_per_row(self):
        """Values are pulled per-row from the named metadata column."""
        inputs = self._make_inputs(3)
        metadata = self._make_metadata(3, extra={"nce": [20, 25, 30]})
        result = resolve_model_inputs(
            inputs=inputs,
            metadata=metadata,
            required_model_inputs=[
                "peptide_sequences",
                "precursor_charges",
                "collision_energies",
            ],
            auto_populated={"peptide_sequences", "precursor_charges"},
            constants=None,
            columns={"collision_energies": "nce"},
            model_name="TestModel",
        )
        assert list(result["collision_energies"]) == [20, 25, 30]

    def test_resolve_auto_populated_are_skipped(self):
        """Auto-populated inputs are not touched even if listed in required_model_inputs."""
        original_sequences = ["AG", "GA", "SP"]
        inputs = pd.DataFrame(
            {
                "peptide_sequences": original_sequences,
                "precursor_charges": [2, 2, 3],
            }
        )
        metadata = self._make_metadata(3)
        result = resolve_model_inputs(
            inputs=inputs,
            metadata=metadata,
            required_model_inputs=["peptide_sequences", "precursor_charges"],
            auto_populated={"peptide_sequences", "precursor_charges"},
            constants={"peptide_sequences": "SHOULD_NOT_OVERWRITE"},
            columns=None,
            model_name="TestModel",
        )
        assert list(result["peptide_sequences"]) == original_sequences

    def test_resolve_missing_input_raises(self):
        """ValueError is raised when a required input is not covered."""
        inputs = self._make_inputs(2)
        metadata = self._make_metadata(2)
        with pytest.raises(ValueError, match="collision_energies"):
            resolve_model_inputs(
                inputs=inputs,
                metadata=metadata,
                required_model_inputs=[
                    "peptide_sequences",
                    "precursor_charges",
                    "collision_energies",
                ],
                auto_populated={"peptide_sequences", "precursor_charges"},
                constants=None,
                columns=None,
                model_name="MyModel",
            )

    def test_resolve_missing_input_error_names_the_model(self):
        """The missing-input error message includes the model name."""
        inputs = self._make_inputs(2)
        metadata = self._make_metadata(2)
        with pytest.raises(ValueError, match="MySpecificModel"):
            resolve_model_inputs(
                inputs=inputs,
                metadata=metadata,
                required_model_inputs=["peptide_sequences", "collision_energies"],
                auto_populated={"peptide_sequences"},
                constants=None,
                columns=None,
                model_name="MySpecificModel",
            )

    def test_resolve_no_extra_inputs_needed(self):
        """No error when all required inputs are auto-populated."""
        inputs = self._make_inputs(2)
        metadata = self._make_metadata(2)
        result = resolve_model_inputs(
            inputs=inputs,
            metadata=metadata,
            required_model_inputs=["peptide_sequences", "precursor_charges"],
            auto_populated={"peptide_sequences", "precursor_charges"},
            constants=None,
            columns=None,
            model_name="iRTModel",
        )
        assert list(result.columns) == ["peptide_sequences", "precursor_charges"]

    def test_fragment_match_feature_passes_constant_to_model(self):
        """FragmentMatchFeatures correctly passes model_input_constants to the Koina model call."""
        feature = FragmentMatchFeatures(
            mz_tolerance=0.02,
            unsupported_residues=["U", "O", "X"],
            model_input_constants={"collision_energies": 30},
        )
        metadata = pd.DataFrame(
            {
                "confidence": [0.9],
                "prediction": [["A", "G"]],
                "precursor_charge": [2],
                "spectrum_id": [0],
                "mz_array": [[100.0, 200.0]],
                "intensity_array": [[1000.0, 2000.0]],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        mock_predictions = pd.DataFrame(
            {
                "peptide_sequences": ["AG"],
                "precursor_charges": [2],
                "collision_energies": [30],
                "intensities": [0.5],
                "mz": [100.0],
                "annotation": ["b1+1"],
            },
            index=[0],
        )

        with patch(
            "winnow.calibration.features.fragment_match.koinapy.Koina"
        ) as mock_koina:
            mock_model = mock_koina.return_value
            mock_model.model_inputs = [
                "peptide_sequences",
                "precursor_charges",
                "collision_energies",
            ]
            mock_model.predict.return_value = mock_predictions
            feature.compute(dataset)

        call_args_df = mock_model.predict.call_args[0][0]
        assert list(call_args_df["collision_energies"]) == [30]

    def test_fragment_match_feature_passes_column_to_model(self):
        """FragmentMatchFeatures uses per-row metadata column values when model_input_columns is set."""
        feature = FragmentMatchFeatures(
            mz_tolerance=0.02,
            unsupported_residues=["U", "O", "X"],
            model_input_columns={"collision_energies": "nce"},
        )
        metadata = pd.DataFrame(
            {
                "confidence": [0.9, 0.8],
                "prediction": [["A", "G"], ["G", "A"]],
                "precursor_charge": [2, 2],
                "spectrum_id": [0, 1],
                "nce": [28, 35],
                "mz_array": [[100.0, 200.0], [110.0, 210.0]],
                "intensity_array": [[1000.0, 2000.0], [1100.0, 2100.0]],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        mock_predictions = pd.DataFrame(
            {
                "peptide_sequences": ["AG", "GA"],
                "precursor_charges": [2, 2],
                "collision_energies": [28, 35],
                "intensities": [0.5, 0.6],
                "mz": [100.0, 110.0],
                "annotation": ["b1+1", "b2+1"],
            },
            index=[0, 1],
        )

        with patch(
            "winnow.calibration.features.fragment_match.koinapy.Koina"
        ) as mock_koina:
            mock_model = mock_koina.return_value
            mock_model.model_inputs = [
                "peptide_sequences",
                "precursor_charges",
                "collision_energies",
            ]
            mock_model.predict.return_value = mock_predictions
            feature.compute(dataset)

        call_args_df = mock_model.predict.call_args[0][0]
        assert list(call_args_df["collision_energies"]) == [28, 35]

    def _make_inputs(self, n: int) -> pd.DataFrame:
        """Return a minimal inputs DataFrame with auto-populated columns."""
        return pd.DataFrame(
            {
                "peptide_sequences": ["AG"] * n,
                "precursor_charges": [2] * n,
            }
        )

    def _make_metadata(self, n: int, extra: Optional[dict] = None) -> pd.DataFrame:
        """Return a minimal metadata DataFrame, optionally with extra columns."""
        data = {"spectrum_id": list(range(n))}
        if extra:
            data.update(extra)
        return pd.DataFrame(data)


class TestSpectrumMatchQualityFunctions:
    """Test spectrum match quality utility functions."""

    # --- Ion Coverage Features ---

    def test_compute_longest_ion_series_consecutive(self):
        """Test longest ion series with consecutive ions."""
        annotations = ["b1+1", "b2+1", "b3+1", "y1+1", "b5+1", "b6+1"]
        assert compute_longest_ion_series(annotations, "b") == 3  # b1, b2, b3
        assert compute_longest_ion_series(annotations, "y") == 1  # only y1

    def test_compute_longest_ion_series_non_consecutive(self):
        """Test longest ion series with gaps."""
        annotations = ["b1+1", "b3+1", "b5+1", "b7+1"]
        assert compute_longest_ion_series(annotations, "b") == 1  # no consecutive

    def test_compute_longest_ion_series_empty(self):
        """Test longest ion series with no matching ions."""
        annotations = ["y1+1", "y2+1", "y3+1"]
        assert compute_longest_ion_series(annotations, "b") == 0

    def test_compute_longest_ion_series_with_duplicates(self):
        """Test that duplicate indices don't inflate the count."""
        annotations = ["b1+1", "b1+1", "b2+1", "b2+1", "b3+1"]
        assert compute_longest_ion_series(annotations, "b") == 3

    def test_compute_complementary_ion_count_basic(self):
        """Test complementary ion count with matching pairs."""
        annotations = ["b1+1", "y3+1", "b2+1"]
        assert compute_complementary_ion_count(annotations, peptide_length=4) == 1

    def test_compute_complementary_ion_count_multiple(self):
        """Test complementary ion count with multiple pairs."""
        annotations = ["b1+1", "y4+1", "b2+1", "y3+1"]
        assert compute_complementary_ion_count(annotations, peptide_length=5) == 2

    def test_compute_complementary_ion_count_no_pairs(self):
        """Test complementary ion count with no complementary pairs."""
        annotations = ["b1+1", "b2+1", "y1+1", "y2+1"]
        assert compute_complementary_ion_count(annotations, peptide_length=5) == 0

    def test_compute_max_ion_gap_basic(self):
        """Test max ion gap calculation."""
        matched_mz = [100.0, 150.0, 300.0]
        assert compute_max_ion_gap(matched_mz) == 150.0

    def test_compute_max_ion_gap_unsorted(self):
        """Test that max ion gap handles unsorted input."""
        matched_mz = [300.0, 100.0, 150.0]
        assert compute_max_ion_gap(matched_mz) == 150.0

    def test_compute_max_ion_gap_insufficient_ions(self):
        """Test max ion gap with fewer than 2 ions."""
        assert compute_max_ion_gap([100.0]) == 0.0
        assert compute_max_ion_gap([]) == 0.0

    # --- B/Y Intensity Ratio ---

    def test_compute_b_y_intensity_ratio_both_ion_types(self):
        """Test b/y intensity ratio with both b and y ions."""
        annotations = ["b1+1", "b2+1", "y1+1", "y2+1"]
        intensities = [1000.0, 2000.0, 500.0, 500.0]  # b=3000, y=1000
        ratio = compute_b_y_intensity_ratio(annotations, intensities)
        assert ratio == pytest.approx(3.0, rel=1e-6)

    def test_compute_b_y_intensity_ratio_only_b_ions(self):
        """Test b/y ratio when only b ions are matched (y=0, uses epsilon)."""
        annotations = ["b1+1", "b2+1", "b3+1"]
        intensities = [1000.0, 2000.0, 3000.0]  # b=6000, y=0
        ratio = compute_b_y_intensity_ratio(annotations, intensities, epsilon=1e-8)
        # Should be very large: 6000 / 1e-8 = 6e11
        assert ratio == pytest.approx(6000.0 / 1e-8, rel=1e-6)

    def test_compute_b_y_intensity_ratio_only_y_ions(self):
        """Test b/y ratio when only y ions are matched."""
        annotations = ["y1+1", "y2+1"]
        intensities = [1000.0, 2000.0]  # b=0, y=3000
        ratio = compute_b_y_intensity_ratio(annotations, intensities)
        # Should be ~0: 0 / (3000 + epsilon)
        assert ratio == pytest.approx(0.0, abs=1e-10)

    def test_compute_b_y_intensity_ratio_no_ions(self):
        """Test b/y ratio when no ions are matched."""
        ratio = compute_b_y_intensity_ratio([], [])
        assert ratio == 0.0

    # --- Spectral Angle ---

    def test_compute_spectral_angle_perfect_match(self):
        """Test spectral angle with identical intensity vectors."""
        theoretical = [1.0, 2.0, 3.0]
        observed = [1.0, 2.0, 3.0]
        angle = compute_spectral_angle(theoretical, observed)
        # Perfect match should give 1.0
        assert angle == pytest.approx(1.0, abs=1e-10)

    def test_compute_spectral_angle_scaled_match(self):
        """Test spectral angle with proportionally scaled intensities."""
        theoretical = [1.0, 2.0, 3.0]
        observed = [2.0, 4.0, 6.0]  # Scaled by 2x
        angle = compute_spectral_angle(theoretical, observed)
        # Proportional vectors should give 1.0 (same direction)
        assert angle == pytest.approx(1.0, abs=1e-10)

    def test_compute_spectral_angle_orthogonal(self):
        """Test spectral angle with orthogonal vectors."""
        theoretical = [1.0, 0.0]
        observed = [0.0, 1.0]
        angle = compute_spectral_angle(theoretical, observed)
        # Orthogonal vectors should give 0.0
        assert angle == pytest.approx(0.0, abs=1e-10)

    def test_compute_spectral_angle_partial_match(self):
        """Test spectral angle with partial match."""
        theoretical = [1.0, 1.0, 1.0]
        observed = [1.0, 0.0, 0.0]  # Only first ion matched
        angle = compute_spectral_angle(theoretical, observed)
        # Should be between 0 and 1
        assert 0.0 < angle < 1.0

    def test_compute_spectral_angle_no_observed_matches(self):
        """Test spectral angle when no ions are observed (all zeros)."""
        theoretical = [1.0, 2.0, 3.0]
        observed = [0.0, 0.0, 0.0]
        angle = compute_spectral_angle(theoretical, observed)
        # No matches should give 0.0
        assert angle == 0.0

    def test_compute_spectral_angle_missing_data(self):
        """Test spectral angle with NaN theoretical intensities."""
        import math

        angle = compute_spectral_angle(math.nan, [1.0, 2.0])
        # Missing data should return 0.0
        assert angle == 0.0

    def test_compute_spectral_angle_empty_lists(self):
        """Test spectral angle with empty lists."""
        angle = compute_spectral_angle([], [])
        assert angle == 0.0

    def test_compute_spectral_angle_length_mismatch_raises(self):
        """Test that mismatched lengths raise ValueError."""
        theoretical = [1.0, 2.0, 3.0]
        observed = [1.0, 2.0]  # Different length
        with pytest.raises(ValueError, match="must align"):
            compute_spectral_angle(theoretical, observed)


class TestXcorr:
    """Test the fast xcorr score function."""

    def test_matching_peaks_positive_score(self):
        """Peaks exactly at theoretical positions should produce a positive xcorr."""
        theoretical_mz = [200.0, 400.0, 600.0]
        observed_mz = [200.0, 400.0, 600.0]
        observed_intensities = [1000.0, 2000.0, 1500.0]

        score = compute_xcorr(observed_mz, observed_intensities, theoretical_mz)
        assert score > 0.0

    def test_no_matching_peaks_low_score(self):
        """Peaks far from theoretical positions should yield a low or negative xcorr."""
        theoretical_mz = [200.0, 400.0, 600.0]
        observed_mz = [250.0, 450.0, 650.0]
        observed_intensities = [1000.0, 2000.0, 1500.0]

        matching_score = compute_xcorr(
            [200.0, 400.0, 600.0], observed_intensities, [200.0, 400.0, 600.0]
        )
        non_matching_score = compute_xcorr(
            observed_mz, observed_intensities, theoretical_mz
        )
        assert non_matching_score < matching_score

    def test_nan_theoretical_returns_zero(self):
        """NaN theoretical m/z (missing prediction) returns 0.0."""
        import math

        score = compute_xcorr([100.0], [1000.0], math.nan)
        assert score == 0.0

    def test_empty_theoretical_returns_zero(self):
        """Empty theoretical m/z list returns 0.0."""
        score = compute_xcorr([100.0], [1000.0], [])
        assert score == 0.0

    def test_empty_observed_returns_zero(self):
        """Empty observed spectrum returns 0.0."""
        score = compute_xcorr([], [], [100.0, 200.0])
        assert score == 0.0

    def test_uniform_spectrum_much_lower_than_matching(self):
        """A uniform observed spectrum should score much lower than a matching one.

        When every bin has the same intensity, the background subtraction
        largely cancels the signal. Some residual remains due to edge effects
        in the background window near spectrum boundaries, but the score
        should be much lower than when peaks align with theoretical positions.
        """
        observed_mz = [float(i) for i in range(50, 500)]
        observed_intensities = [100.0] * len(observed_mz)
        theoretical_mz = [100.0, 200.0, 300.0]

        uniform_score = compute_xcorr(observed_mz, observed_intensities, theoretical_mz)

        sparse_mz = [100.0, 200.0, 300.0]
        sparse_intensities = [1000.0, 1000.0, 1000.0]
        matching_score = compute_xcorr(sparse_mz, sparse_intensities, theoretical_mz)

        assert uniform_score < matching_score * 0.25

    def test_scale_invariance(self):
        """Xcorr should be the same regardless of observed intensity scale.

        Window normalization rescales each region to a fixed value, so
        multiplying all intensities by a constant should not change the score.
        """
        theoretical_mz = [200.0, 400.0, 600.0]
        observed_mz = [200.0, 300.0, 400.0, 500.0, 600.0]
        intensities_1x = [1000.0, 500.0, 2000.0, 300.0, 1500.0]
        intensities_100x = [i * 100.0 for i in intensities_1x]

        score_1x = compute_xcorr(observed_mz, intensities_1x, theoretical_mz)
        score_100x = compute_xcorr(observed_mz, intensities_100x, theoretical_mz)
        assert score_1x == pytest.approx(score_100x, rel=1e-10)

    def test_more_matches_higher_score(self):
        """More matching peaks should produce a higher xcorr than fewer matches."""
        observed_mz = [100.0, 200.0, 300.0, 400.0, 500.0]
        observed_intensities = [1000.0] * 5

        score_2_matches = compute_xcorr(
            observed_mz, observed_intensities, [100.0, 200.0]
        )
        score_4_matches = compute_xcorr(
            observed_mz, observed_intensities, [100.0, 200.0, 300.0, 400.0]
        )
        assert score_4_matches > score_2_matches

    def test_single_peak_positive(self):
        """A single observed peak matching a single theoretical ion should score positive."""
        score = compute_xcorr([300.0], [5000.0], [300.0])
        assert score > 0.0
