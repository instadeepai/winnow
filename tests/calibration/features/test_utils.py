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
    _validate_mz_tolerance,
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

        match_fraction, average_intensity, _, _ = find_matching_ions(
            source_mz,
            target_mz,
            target_intensities,
            source_annotations=["b1+1", "b2+1", "b3+1"],
            mz_tolerance_da=0.01,
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

        match_fraction, average_intensity, _, _ = find_matching_ions(
            source_mz,
            target_mz,
            target_intensities,
            source_annotations=["b1+1", "b2+1"],
            mz_tolerance_da=0.02,
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

        match_fraction, average_intensity, _, _ = find_matching_ions(
            source_mz,
            target_mz,
            target_intensities,
            source_annotations=["b1+1", "b2+1"],
            mz_tolerance_da=0.01,
        )

        assert match_fraction == 0
        assert average_intensity == 0.0

    def test_find_matching_ions_no_matches(self):
        """Test find_matching_ions with no matches."""
        source_mz = [100.0]
        target_mz = [200.0]  # No match within tolerance
        target_intensities = [1000.0]

        match_fraction, average_intensity, _, _ = find_matching_ions(
            source_mz,
            target_mz,
            target_intensities,
            source_annotations=["b1+1"],
            mz_tolerance_da=0.01,
        )
        assert match_fraction == 0.0  # 0 matches / 1 source ion
        assert average_intensity == 0.0  # 0 match intensity / 1000 total intensity

    def test_find_matching_ions_ppm_tolerance(self):
        """Test find_matching_ions with ppm-based tolerance."""
        source_mz = [500.0, 1000.0]
        # At 20 ppm: 500 * 20/1e6 = 0.01 Da, 1000 * 20/1e6 = 0.02 Da
        target_mz = [500.009, 1000.019]  # Within 20 ppm
        target_intensities = [1000.0, 2000.0]

        match_fraction, _, _, _ = find_matching_ions(
            source_mz,
            target_mz,
            target_intensities,
            source_annotations=["b1+1", "b2+1"],
            mz_tolerance_ppm=20,
        )
        assert match_fraction == 1.0

    def test_find_matching_ions_ppm_scales_with_mz(self):
        """Test that ppm tolerance scales: same Da offset matches at high m/z but not at low m/z."""
        source_mz_low = [100.0]
        source_mz_high = [1000.0]
        offset = 0.015  # 150 ppm at m/z 100, 15 ppm at m/z 1000

        match_low, _, _, _ = find_matching_ions(
            source_mz_low,
            [100.0 + offset],
            [1000.0],
            source_annotations=["b1+1"],
            mz_tolerance_ppm=20,
        )
        match_high, _, _, _ = find_matching_ions(
            source_mz_high,
            [1000.0 + offset],
            [1000.0],
            source_annotations=["b1+1"],
            mz_tolerance_ppm=20,
        )
        assert match_low == 0.0  # 150 ppm > 20 ppm, no match
        assert match_high == 1.0  # 15 ppm < 20 ppm, match

    def test_find_matching_ions_both_tolerances_raises(self):
        """Setting both mz_tolerance_ppm and mz_tolerance_da raises ValueError."""
        with pytest.raises(ValueError, match="not both"):
            find_matching_ions(
                [100.0],
                [100.0],
                [1000.0],
                source_annotations=["b1+1"],
                mz_tolerance_ppm=20,
                mz_tolerance_da=0.02,
            )

    def test_find_matching_ions_neither_tolerance_raises(self):
        """Setting neither mz_tolerance_ppm nor mz_tolerance_da raises ValueError."""
        with pytest.raises(ValueError, match="Exactly one"):
            find_matching_ions(
                [100.0],
                [100.0],
                [1000.0],
                source_annotations=["b1+1"],
            )


class TestValidateMzTolerance:
    """Test the _validate_mz_tolerance helper."""

    def test_ppm_only_passes(self):
        _validate_mz_tolerance(mz_tolerance_ppm=20, mz_tolerance_da=None)

    def test_da_only_passes(self):
        _validate_mz_tolerance(mz_tolerance_ppm=None, mz_tolerance_da=0.02)

    def test_both_set_raises(self):
        with pytest.raises(ValueError, match="not both"):
            _validate_mz_tolerance(mz_tolerance_ppm=20, mz_tolerance_da=0.02)

    def test_neither_set_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            _validate_mz_tolerance(mz_tolerance_ppm=None, mz_tolerance_da=None)


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
                mz_tolerance_ppm=20,
                model_input_constants={"collision_energies": 25},
                model_input_columns={"collision_energies": "ce_col"},
            )

    def test_conflict_detected_at_construction_time_chimeric_features(self):
        """ChimericFeatures raises ValueError at construction when keys conflict."""
        with pytest.raises(ValueError, match="collision_energies"):
            ChimericFeatures(
                mz_tolerance_ppm=20,
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
            mz_tolerance_ppm=20,
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
            mz_tolerance_ppm=20,
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
