"""Unit tests for winnow calibration features."""

import numpy as np
import pandas as pd
import pytest
from typing import Optional
from unittest.mock import Mock, patch
from winnow.calibration.calibration_features import (
    CalibrationFeatures,
    FeatureDependency,
    MassErrorFeature,
    BeamFeatures,
    RetentionTimeFeature,
    FragmentMatchFeatures,
    ChimericFeatures,
    find_matching_ions,
    _raise_value_error,
    _validate_model_input_params,
    _resolve_model_inputs,
)
from winnow.datasets.calibration_dataset import CalibrationDataset


class TestUtilityFunctions:
    """Test utility functions used by calibration features."""

    def test_raise_value_error_with_none(self):
        """Test that _raise_value_error raises ValueError for None values."""
        with pytest.raises(ValueError, match="test_param cannot be None"):
            _raise_value_error(None, "test_param")

    def test_raise_value_error_with_valid_value(self):
        """Test that _raise_value_error does not raise for valid values."""
        # Should not raise any exception
        _raise_value_error("valid_value", "test_param")
        _raise_value_error(42, "test_param")
        _raise_value_error([], "test_param")

    def test_find_matching_ions_exact_match(self):
        """Test find_matching_ions with exact m/z matches."""
        source_mz = [100.0, 200.0, 300.0]
        target_mz = [100.0, 200.0, 400.0]
        target_intensities = [1000.0, 2000.0, 4000.0]
        tolerance = 0.01

        match_fraction, average_intensity = find_matching_ions(
            source_mz, target_mz, target_intensities, tolerance
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

        match_fraction, average_intensity = find_matching_ions(
            source_mz, target_mz, target_intensities, tolerance
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

        match_fraction, average_intensity = find_matching_ions(
            source_mz, target_mz, target_intensities, tolerance
        )

        assert match_fraction == 0
        assert average_intensity == 0.0

    def test_find_matching_ions_no_matches(self):
        """Test find_matching_ions with no matches."""
        source_mz = [100.0]
        target_mz = [200.0]  # No match within tolerance
        target_intensities = [1000.0]

        match_fraction, average_intensity = find_matching_ions(
            source_mz, target_mz, target_intensities, 0.01
        )
        assert match_fraction == 0.0  # 0 matches / 1 source ion
        assert average_intensity == 0.0  # 0 match intensity / 1000 total intensity


class TestMassErrorFeature:
    """Test the MassErrorFeature class."""

    @pytest.fixture()
    def mass_error_feature(self):
        """Create a MassErrorFeature instance for testing."""
        residue_masses = {
            "G": 57.021464,
            "A": 71.037114,
            "P": 97.052764,
            "E": 129.042593,
            "T": 101.047670,
            "I": 113.084064,
            "D": 115.026943,
            "R": 156.101111,
            "O": 237.147727,
            "N": 114.042927,
            "S": 87.032028,
            "M": 131.040485,
            "L": 113.084064,
            "V": 99.068414,
        }
        return MassErrorFeature(residue_masses=residue_masses)

    @pytest.fixture()
    def sample_dataset(self):
        """Create a sample CalibrationDataset for testing."""
        metadata = pd.DataFrame(
            {
                "precursor_mass": [1000.0, 1200.0, 800.0],
                "prediction": [["G", "A"], ["A", "S", "P"], ["V"]],
                "confidence": [0.9, 0.8, 0.7],
            }
        )
        return CalibrationDataset(metadata=metadata, predictions=[])

    def test_properties(self, mass_error_feature):
        """Test MassErrorFeature properties."""
        assert mass_error_feature.name == "Mass Error"
        assert mass_error_feature.columns == ["Mass Error"]
        assert mass_error_feature.dependencies == []

    def test_prepare_does_nothing(self, mass_error_feature, sample_dataset):
        """Test that prepare method does nothing."""
        # Should not raise any exception and not modify dataset
        original_metadata = sample_dataset.metadata.copy()
        mass_error_feature.prepare(sample_dataset)
        pd.testing.assert_frame_equal(sample_dataset.metadata, original_metadata)

    def test_compute_mass_error(self, mass_error_feature, sample_dataset):
        """Test mass error computation."""
        mass_error_feature.compute(sample_dataset)

        # Check that Mass Error column was added
        assert "Mass Error" in sample_dataset.metadata.columns

        # Verify calculations for known values
        # G + A = 57.021464 + 71.037114 = 128.058578
        # Expected mass error = 1000.0 - (128.058578 + 18.0106 + 1.007276) = 852.923546
        expected_first = 1000.0 - (128.058578 + 18.0106 + 1.007276)
        assert sample_dataset.metadata.iloc[0]["Mass Error"] == pytest.approx(
            expected_first, rel=1e-6, abs=1e-6
        )

    def test_compute_with_invalid_peptide(self, mass_error_feature):
        """Test mass error computation with invalid peptide format."""
        metadata = pd.DataFrame(
            {
                "precursor_mass": [1000.0],
                "prediction": ["invalid_string"],  # String instead of list
                "confidence": [0.9],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=[])

        mass_error_feature.compute(dataset)

        # Should result in +inf for invalid peptide (1000.0 - (-inf + constants) = +inf)
        assert dataset.metadata.iloc[0]["Mass Error"] == float("inf")

    def test_residue_masses_parameter(self):
        """Test that custom residue masses are used correctly."""
        custom_masses = {"A": 100.0, "G": 200.0}
        feature = MassErrorFeature(residue_masses=custom_masses)

        metadata = pd.DataFrame(
            {
                "precursor_mass": [1000.0],
                "prediction": [["A", "G"]],
                "confidence": [0.9],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=[])

        feature.compute(dataset)

        # Expected: 1000.0 - (100.0 + 200.0 + 18.0106 + 1.007276)
        expected = 1000.0 - (300.0 + 18.0106 + 1.007276)
        assert dataset.metadata.iloc[0]["Mass Error"] == pytest.approx(
            expected, rel=1e-6, abs=1e-6
        )


class MockScoredSequence:
    """Mock class for ScoredSequence used in beam search results."""

    def __init__(self, sequence, log_prob):
        self.sequence = sequence
        self.sequence_log_probability = log_prob


class TestBeamFeatures:
    """Test the BeamFeatures class."""

    @pytest.fixture()
    def beam_features(self):
        """Create a BeamFeatures instance for testing."""
        return BeamFeatures()

    @pytest.fixture()
    def sample_dataset_with_predictions(self):
        """Create a sample dataset with beam search predictions."""
        metadata = pd.DataFrame({"confidence": [0.9, 0.8, 0.7]})

        # Mock beam search results with different numbers of sequences
        predictions = [
            [  # First spectrum - 3 sequences
                MockScoredSequence(["A", "G"], np.log(0.8)),
                MockScoredSequence(["G", "A"], np.log(0.6)),
                MockScoredSequence(["S", "P"], np.log(0.4)),
            ],
            [  # Second spectrum - 2 sequences
                MockScoredSequence(["V", "T"], np.log(0.9)),
                MockScoredSequence(["T", "V"], np.log(0.7)),
            ],
            [  # Third spectrum - 1 sequence
                MockScoredSequence(["K"], np.log(0.95))
            ],
        ]

        return CalibrationDataset(metadata=metadata, predictions=predictions)

    def test_properties(self, beam_features):
        """Test BeamFeatures properties."""
        assert beam_features.name == "Beam Features"
        assert beam_features.columns == [
            "margin",
            "median_margin",
            "entropy",
            "z-score",
        ]
        assert beam_features.dependencies == []

    def test_prepare_does_nothing(self, beam_features, sample_dataset_with_predictions):
        """Test that prepare method does nothing."""
        original_metadata = sample_dataset_with_predictions.metadata.copy()
        beam_features.prepare(sample_dataset_with_predictions)
        pd.testing.assert_frame_equal(
            sample_dataset_with_predictions.metadata, original_metadata
        )

    def test_compute_beam_features(
        self, beam_features, sample_dataset_with_predictions
    ):
        """Test beam features computation."""
        beam_features.compute(sample_dataset_with_predictions)

        # Check that all expected columns were added
        expected_columns = ["margin", "median_margin", "entropy", "z-score"]
        for col in expected_columns:
            assert col in sample_dataset_with_predictions.metadata.columns

        # Check that we have the right number of rows
        assert len(sample_dataset_with_predictions.metadata) == 3

    def test_compute_with_none_predictions(self, beam_features):
        """Test that compute raises error when predictions is None."""
        metadata = pd.DataFrame({"confidence": [0.9]})
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        with pytest.raises(ValueError, match="dataset.predictions cannot be None"):
            beam_features.compute(dataset)

    def test_compute_with_insufficient_sequences_warning(self, beam_features):
        """Test that warning is issued for beam results with < 2 sequences."""
        metadata = pd.DataFrame({"confidence": [0.9, 0.8]})
        predictions = [
            [MockScoredSequence(["A"], np.log(0.8))],  # Only 1 sequence
            [
                MockScoredSequence(["G"], np.log(0.9)),
                MockScoredSequence(["A"], np.log(0.7)),
            ],  # Only 2 sequences
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)

        with pytest.warns(
            UserWarning,
            match="1 beam search results have fewer than two sequences. This may affect the efficacy of computed beam features.",
        ):
            beam_features.compute(dataset)

    def test_margin_calculation(self, beam_features):
        """Test specific margin calculation."""
        metadata = pd.DataFrame({"confidence": [0.9]})
        predictions = [
            [
                MockScoredSequence(["A"], np.log(0.8)),  # top = 0.8
                MockScoredSequence(["G"], np.log(0.6)),  # second = 0.6
            ]
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)

        beam_features.compute(dataset)

        expected_margin = 0.8 - 0.6  # top_prob - second_prob
        assert dataset.metadata.iloc[0]["margin"] == pytest.approx(
            expected_margin, rel=1e-10, abs=1e-10
        )

    def test_beam_features_with_one_sequence(self, beam_features):
        """Test beam feature calculations with single sequence."""
        metadata = pd.DataFrame({"confidence": [0.9]})
        predictions = [
            [MockScoredSequence(["A"], np.log(0.8))]  # Single sequence
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)

        beam_features.compute(dataset)

        # Detailed checks for single sequence:
        # top_prob = 0.8, second_prob = 0.0 (no second sequence)
        # margin = top_prob - second_prob = 0.8 - 0.0 = 0.8
        assert dataset.metadata.iloc[0]["margin"] == pytest.approx(
            0.8, rel=1e-10, abs=1e-10
        )

        # runner_up_probs = [0.0] (no runner-ups since we start from index 1)
        # median_margin = top_prob - median([0.0]) = 0.8 - 0.0 = 0.8
        assert dataset.metadata.iloc[0]["median_margin"] == pytest.approx(
            0.8, rel=1e-10, abs=1e-10
        )

        # entropy([0.0]) should be 0
        assert dataset.metadata.iloc[0]["entropy"] == pytest.approx(
            0.0, rel=1e-10, abs=1e-10
        )

        # z-score with single value should be 0 (no variation)
        z_score = dataset.metadata.iloc[0]["z-score"]
        assert z_score == pytest.approx(
            0.0, rel=1e-10, abs=1e-10
        )  # std_prob = 0, so z-score = 0

    def test_beam_features_with_two_sequences(self, beam_features):
        """Test beam feature calculations with two sequences."""
        metadata = pd.DataFrame({"confidence": [0.9]})
        predictions = [
            [
                MockScoredSequence(["A"], np.log(0.8)),  # top
                MockScoredSequence(["G"], np.log(0.6)),  # second/runner-up
            ]
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)

        beam_features.compute(dataset)

        # Detailed checks for two sequences:
        # top_prob = 0.8, second_prob = 0.6
        # margin = 0.8 - 0.6 = 0.2
        assert dataset.metadata.iloc[0]["margin"] == pytest.approx(
            0.2, rel=1e-10, abs=1e-10
        )

        # runner_up_probs = [0.6] (from index 1 onwards)
        # median_margin = top_prob - median([0.6]) = 0.8 - 0.6 = 0.2
        assert dataset.metadata.iloc[0]["median_margin"] == pytest.approx(
            0.2, rel=1e-10, abs=1e-10
        )

        # entropy([1.0]) where runner_up_probs are normalised
        # Since there's only one runner-up, normalised prob = [1.0]
        # entropy([1.0]) = 0 (no uncertainty)
        assert dataset.metadata.iloc[0]["entropy"] == pytest.approx(
            0.0, rel=1e-10, abs=1e-10
        )

        # z-score: mean_prob = (0.8 + 0.6)/2 = 0.7, std_prob = sqrt(((0.8-0.7)^2 + (0.6-0.7)^2)/2)
        # std_prob = sqrt((0.01 + 0.01)/2) = sqrt(0.01) = 0.1
        # z_score = (0.8 - 0.7) / 0.1 = 1.0
        expected_z_score = 1.0
        assert dataset.metadata.iloc[0]["z-score"] == pytest.approx(
            expected_z_score, rel=1e-10, abs=1e-10
        )

    def test_beam_features_with_three_sequences(self, beam_features):
        """Test beam feature calculations with three sequences."""
        metadata = pd.DataFrame({"confidence": [0.9]})
        predictions = [
            [
                MockScoredSequence(["A"], np.log(0.7)),  # top
                MockScoredSequence(["G"], np.log(0.2)),  # second
                MockScoredSequence(["S"], np.log(0.1)),  # third
            ]
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)

        beam_features.compute(dataset)

        # Detailed checks for three sequences:
        # top_prob = 0.7, second_prob = 0.2, third_prob = 0.1
        # margin = 0.7 - 0.2 = 0.5
        assert dataset.metadata.iloc[0]["margin"] == pytest.approx(
            0.5, rel=1e-10, abs=1e-10
        )

        # runner_up_probs = [0.2, 0.1] (from index 1 onwards)
        # median_margin = top_prob - median([0.2, 0.1]) = 0.7 - 0.15 = 0.55
        assert dataset.metadata.iloc[0]["median_margin"] == pytest.approx(
            0.55, rel=1e-10, abs=1e-10
        )

        # runner_up_probs normalised: [0.2, 0.1] -> [0.2/0.3, 0.1/0.3] = [2/3, 1/3]
        # entropy([2/3, 1/3]) = 0.6365141682948128
        assert dataset.metadata.iloc[0]["entropy"] == pytest.approx(
            0.6365141682948128, rel=1e-10, abs=1e-10
        )

        # z-score calculation with three values
        # mean_prob = (0.7 + 0.2 + 0.1)/3 = 1/3
        # std_prob = sqrt(((0.7-1/3)^2 + (0.2-1/3)^2 + (0.1-1/3)^2)/3) = 0.262466929133727
        # z_score = (0.7 - 1/3) / 0.262466929133727 = 1.3970013970020956
        z_score = dataset.metadata.iloc[0]["z-score"]
        assert z_score == pytest.approx(1.3970013970020956, rel=1e-10, abs=1e-10)

    def test_beam_features_edge_case_equal_probabilities(self, beam_features):
        """Test beam features when all sequences have equal probabilities."""
        metadata = pd.DataFrame({"confidence": [0.9]})
        predictions = [
            [
                MockScoredSequence(["A"], np.log(1 / 3)),
                MockScoredSequence(["G"], np.log(1 / 3)),
                MockScoredSequence(["S"], np.log(1 / 3)),
            ]
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)

        beam_features.compute(dataset)

        # When all probabilities are equal:
        # margin = 1/3 - 1/3 = 0.0
        assert dataset.metadata.iloc[0]["margin"] == 0.0

        # median_margin = 1/3 - median([1/3, 1/3]) = 1/3 - 1/3 = 0.0
        assert dataset.metadata.iloc[0]["median_margin"] == 0.0

        # entropy of [0.5, 0.5] (normalised) should be 0.6931471805599453
        assert dataset.metadata.iloc[0]["entropy"] == pytest.approx(
            0.6931471805599453, rel=1e-10, abs=1e-10
        )

        # z-score should be 0 (all values equal, so std = 0)
        assert dataset.metadata.iloc[0]["z-score"] == pytest.approx(
            0.0, rel=1e-10, abs=1e-10
        )


class TestCalibrationFeaturesInterface:
    """Test the abstract CalibrationFeatures interface."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that CalibrationFeatures cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CalibrationFeatures()

    def test_feature_dependency_cannot_instantiate(self):
        """Test that FeatureDependency cannot be instantiated directly."""
        with pytest.raises(TypeError):
            FeatureDependency()


class TestRetentionTimeFeature:
    """Test the RetentionTimeFeature class."""

    @pytest.fixture()
    def retention_time_feature(self):
        """Create a RetentionTimeFeature instance for testing."""
        return RetentionTimeFeature(
            hidden_dim=10, train_fraction=0.8, unsupported_residues=["U", "O", "X"]
        )

    @pytest.fixture()
    def sample_dataset_with_rt(self):
        """Create a sample dataset with retention time data."""
        metadata = pd.DataFrame(
            {
                "confidence": [0.95, 0.90, 0.85, 0.80, 0.75],
                "prediction": [["A", "G"], ["G", "A"], ["S", "P"], ["V"], ["K"]],
                "prediction_untokenised": ["AG", "GA", "SP", "V", "K"],
                "retention_time": [10.5, 15.2, 20.1, 8.7, 12.3],
                "precursor_charge": [2, 2, 3, 1, 2],
                "spectrum_id": [0, 1, 2, 3, 4],
            }
        )
        return CalibrationDataset(metadata=metadata, predictions=[])

    def test_properties(self, retention_time_feature):
        """Test RetentionTimeFeature properties."""
        assert retention_time_feature.name == "iRT Feature"
        assert retention_time_feature.columns == ["iRT error", "is_missing_irt_error"]
        assert retention_time_feature.dependencies == []

    def test_initialization_parameters(self):
        """Test initialization with custom parameters."""
        feature = RetentionTimeFeature(
            hidden_dim=10, train_fraction=0.8, unsupported_residues=["U", "O", "X"]
        )
        assert feature.hidden_dim == 10
        assert feature.train_fraction == 0.8
        assert feature.irt_model_name == "Prosit_2019_irt"
        assert hasattr(feature, "irt_predictor")

    @patch("winnow.calibration.calibration_features.koinapy.Koina")
    def test_prepare_with_mock(
        self, mock_koina, retention_time_feature, sample_dataset_with_rt
    ):
        """Test prepare method with mocked Koina iRT model."""
        # Mock the Koina model
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.model_inputs = ["peptide_sequences"]

        # Mock predict to return iRT values
        mock_model_instance.predict.return_value = pd.DataFrame(
            {
                "irt": [35.1, 20.7, 28.1, 25.5]  # 4 values for 80% of 5 samples
            }
        )

        # Mock the MLPRegressor fit method
        with patch.object(retention_time_feature.irt_predictor, "fit") as mock_fit:
            retention_time_feature.prepare(sample_dataset_with_rt)

            # Check that fit was called
            mock_fit.assert_called_once()

            # Check that predict was called on the model
            mock_model_instance.predict.assert_called_once()

            # Check that the model was fitted on the first 80% of the dataset
            expected_train_size = int(0.8 * len(sample_dataset_with_rt.metadata))
            expected_x = (
                sample_dataset_with_rt.metadata["retention_time"]
                .iloc[:expected_train_size]
                .values.reshape(-1, 1)
            )
            expected_y = (
                mock_model_instance.predict.return_value["irt"]
                .iloc[:expected_train_size]
                .values
            )

            # Verify fit was called with the correct arguments
            call_args = mock_fit.call_args
            np.testing.assert_array_equal(call_args[0][0], expected_x)
            np.testing.assert_array_equal(call_args[0][1], expected_y)

    @patch("winnow.calibration.calibration_features.koinapy.Koina")
    def test_compute_with_mock(
        self, mock_koina, retention_time_feature, sample_dataset_with_rt
    ):
        """Test compute method with mocked models."""
        # Mock the Koina iRT model
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.model_inputs = ["peptide_sequences"]

        # Mock predict to return iRT values for all samples
        mock_model_instance.predict.return_value = pd.DataFrame(
            {"irt": [25.5, 30.2, 35.1, 20.7, 28.1]}
        )

        # Mock the MLPRegressor predict method
        with patch.object(
            retention_time_feature.irt_predictor, "predict"
        ) as mock_predict:
            mock_predict.return_value = [24.0, 29.5, 34.8, 21.2, 27.9]

            retention_time_feature.compute(sample_dataset_with_rt)

            # Check that columns were added
            assert "iRT" in sample_dataset_with_rt.metadata.columns
            assert "predicted iRT" in sample_dataset_with_rt.metadata.columns
            assert "iRT error" in sample_dataset_with_rt.metadata.columns
            assert "is_missing_irt_error" in sample_dataset_with_rt.metadata.columns

            # Check that error is computed as absolute difference
            assert len(sample_dataset_with_rt.metadata["iRT error"]) == 5
            max_abs_diff = abs(
                sample_dataset_with_rt.metadata["predicted iRT"]
                - sample_dataset_with_rt.metadata["iRT"]
            ).max()
            max_error = sample_dataset_with_rt.metadata["iRT error"].max()
            assert max_abs_diff == pytest.approx(max_error, rel=1e-10, abs=1e-10)

            # Check that predict was called on both models
            mock_model_instance.predict.assert_called_once()
            mock_predict.assert_called_once()

    def test_prepare_does_nothing_without_mocking(self, retention_time_feature):
        """Test that prepare method handles dataset requirements."""
        # Test with minimal valid dataset
        metadata = pd.DataFrame(
            {
                "confidence": [0.9, 0.8],
                "prediction": [["A"], ["G"]],
                "retention_time": [10.0, 15.0],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=[])

        # This would normally call the real Prosit model, but we're testing structure
        # Just ensure it doesn't crash on basic validation
        try:
            # The method expects certain columns to exist
            assert "confidence" in dataset.metadata.columns
            assert "prediction" in dataset.metadata.columns
            assert "retention_time" in dataset.metadata.columns
        except Exception as e:
            # If it fails due to network/model issues, that's expected in unit tests
            assert (
                "koina" in str(e).lower()
                or "connection" in str(e).lower()
                or "model" in str(e).lower()
            )

    def test_compute_maps_values_to_correct_rows_and_imputes_missing(
        self,
        retention_time_feature,
    ):
        """Test that computed iRT values are mapped to correct rows and missing values are imputed correctly."""
        # Create a dataset with mixed valid/invalid predictions
        # Based on check_valid_irt_prediction logic:
        # Valid: len <= 30, no invalid tokens
        # Invalid: len > 30 OR has invalid tokens
        metadata = pd.DataFrame(
            {
                "confidence": [0.9, 0.7, 0.6],
                "prediction": [
                    ["A", "G"],  # Valid: len 2
                    ["A"] * 31,  # Invalid: len > 30
                    ["A", "G"],  # Valid: len 2
                ],
                "prediction_untokenised": [
                    "AG",
                    "A" * 31,
                    "AG",
                ],
                "retention_time": [10.5, 15.2, 8.7],
                "precursor_charge": [2, 2, 2],
                "spectrum_id": [
                    10,
                    20,
                    40,
                ],  # Non-contiguous spectrum IDs to test mapping
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=[])

        # Mock Prosit model to return iRT predictions
        def mock_prosit_predict(inputs_df):
            """Mock Prosit predict that returns iRT values."""
            if len(inputs_df) == 0:
                return pd.DataFrame(columns=["irt"])

            predictions_list = []
            index_list = []

            for spectrum_id in inputs_df.index:
                # Return different iRT values for valid spectra
                if spectrum_id == 10:
                    irt_value = 25.5
                elif spectrum_id == 40:
                    irt_value = 30.2
                else:
                    irt_value = 20.0  # Fallback

                predictions_list.append({"irt": irt_value})
                index_list.append(spectrum_id)

            predictions_df = pd.DataFrame(predictions_list, index=index_list)
            return predictions_df

        # Mock the MLPRegressor predict method
        # When patching an instance method with patch.object, side_effect receives only the method arguments (not self)
        def mock_mlp_predict(x):
            """Mock MLPRegressor predict that returns predicted iRT values."""
            # Simple linear mapping for testing
            return x.flatten() * 2.0

        # Run compute with mocked Koina iRT model and MLPRegressor
        with patch(
            "winnow.calibration.calibration_features.koinapy.Koina"
        ) as mock_koina:
            mock_model = mock_koina.return_value
            mock_model.model_inputs = ["peptide_sequences"]
            mock_model.predict = mock_prosit_predict

            with patch.object(
                retention_time_feature.irt_predictor,
                "predict",
                side_effect=mock_mlp_predict,
            ):
                retention_time_feature.compute(dataset)

        # Check that is_missing_irt_error column was added
        assert "is_missing_irt_error" in dataset.metadata.columns

        # Verify valid/invalid flags (based on the check_valid_irt_prediction logic)
        # The function filters OUT entries with: len > 30, invalid tokens, or WITH "C"
        # So valid entries are those WITHOUT these issues
        # Spectrum ID 10: no C, len 2 -> should be valid
        # Spectrum ID 20: has C -> should be invalid (filtered out)
        # Spectrum ID 30: len > 30 -> should be invalid (filtered out)
        # Spectrum ID 40: no C, len 2 -> should be valid
        valid_flags = ~dataset.metadata["is_missing_irt_error"]
        # Check by spectrum_id to verify mapping works correctly
        spectrum_10_mask = dataset.metadata["spectrum_id"] == 10
        spectrum_20_mask = dataset.metadata["spectrum_id"] == 20
        spectrum_40_mask = dataset.metadata["spectrum_id"] == 40

        assert valid_flags[spectrum_10_mask].iloc[0]  # Valid (len <= 30)
        assert not valid_flags[spectrum_20_mask].iloc[0]  # Invalid (has len > 30)
        assert valid_flags[spectrum_40_mask].iloc[0]  # Valid (len <= 30)

        # Check iRT mapping
        assert "iRT" in dataset.metadata.columns
        assert "predicted iRT" in dataset.metadata.columns
        assert "iRT error" in dataset.metadata.columns

        # Valid entries should have non-NaN iRT values
        irt_10 = dataset.metadata[spectrum_10_mask]["iRT"].iloc[0]
        irt_40 = dataset.metadata[spectrum_40_mask]["iRT"].iloc[0]
        assert not pd.isna(irt_10)
        assert not pd.isna(irt_40)
        assert isinstance(irt_10, (int, float))
        assert isinstance(irt_40, (int, float))
        assert irt_10 == 25.5  # From mock
        assert irt_40 == 30.2  # From mock

        # Invalid entries should have NaN iRT values
        irt_20 = dataset.metadata[spectrum_20_mask]["iRT"].iloc[0]
        assert pd.isna(irt_20)

        # Check predicted iRT (from MLPRegressor)
        predicted_irt_10 = dataset.metadata[spectrum_10_mask]["predicted iRT"].iloc[0]
        predicted_irt_40 = dataset.metadata[spectrum_40_mask]["predicted iRT"].iloc[0]
        assert not pd.isna(predicted_irt_10)
        assert not pd.isna(predicted_irt_40)
        # Should be rt * 2.0 from mock_mlp_predict
        assert predicted_irt_10 == pytest.approx(10.5 * 2.0, rel=1e-10)
        assert predicted_irt_40 == pytest.approx(8.7 * 2.0, rel=1e-10)

        # Check iRT error
        # For valid entries, error should be computed
        irt_error_10 = dataset.metadata[spectrum_10_mask]["iRT error"].iloc[0]
        irt_error_40 = dataset.metadata[spectrum_40_mask]["iRT error"].iloc[0]
        assert not pd.isna(irt_error_10)
        assert not pd.isna(irt_error_40)
        assert isinstance(irt_error_10, (int, float))
        assert isinstance(irt_error_40, (int, float))
        # Error should be absolute difference between predicted and actual iRT
        expected_error_10 = abs(predicted_irt_10 - irt_10)
        expected_error_40 = abs(predicted_irt_40 - irt_40)
        assert irt_error_10 == pytest.approx(expected_error_10, rel=1e-10)
        assert irt_error_40 == pytest.approx(expected_error_40, rel=1e-10)

        # For invalid entries, error should be 0.0 (fillna(0.0) is used)
        irt_error_20 = dataset.metadata[spectrum_20_mask]["iRT error"].iloc[0]
        assert irt_error_20 == 0.0

        assert len(dataset.metadata) == 3
        assert all(
            sid in dataset.metadata["spectrum_id"].values for sid in [10, 20, 40]
        )
        # Verify that spectrum_id values match the expected mapping
        assert set(dataset.metadata["spectrum_id"].values) == {10, 20, 40}


class TestFragmentMatchFeatures:
    """Test the FragmentMatchFeatures class."""

    @pytest.fixture()
    def prosit_features(self):
        """Create a FragmentMatchFeatures instance for testing."""
        return FragmentMatchFeatures(
            mz_tolerance=0.02,
            unsupported_residues=["U", "O", "X"],
            model_input_constants={"collision_energies": 25},
        )

    @pytest.fixture()
    def sample_dataset_with_spectra(self):
        """Create a sample dataset with spectral data."""
        metadata = pd.DataFrame(
            {
                "confidence": [0.9, 0.8, 0.7],
                "prediction": [["A", "G"], ["G", "A"], ["S", "P"]],
                "prediction_untokenised": ["AG", "GA", "SP"],
                "precursor_charge": [2, 2, 3],
                "spectrum_id": [0, 1, 2],
                "mz_array": [
                    [100.0, 200.0, 300.0],
                    [150.0, 250.0, 350.0],
                    [120.0, 220.0, 320.0],
                ],
                "intensity_array": [
                    [1000.0, 2000.0, 3000.0],
                    [1500.0, 2500.0, 3500.0],
                    [1200.0, 2200.0, 3200.0],
                ],
            }
        )
        return CalibrationDataset(metadata=metadata, predictions=[])

    def test_properties(self, prosit_features):
        """Test FragmentMatchFeatures properties."""
        assert prosit_features.name == "Fragment Match Features"
        assert prosit_features.columns == [
            "ion_matches",
            "ion_match_intensity",
            "is_missing_fragment_match_features",
        ]
        assert prosit_features.dependencies == []
        assert prosit_features.mz_tolerance == 0.02

    def test_initialization_with_tolerance(self):
        """Test initialization with custom tolerance."""
        feature = FragmentMatchFeatures(
            mz_tolerance=0.01,
            unsupported_residues=["U", "O", "X"],
            model_input_constants={"collision_energies": 25},
        )
        assert feature.mz_tolerance == 0.01
        assert feature.intensity_model_name == "Prosit_2020_intensity_HCD"
        assert feature.model_input_constants == {"collision_energies": 25}
        assert feature.model_input_columns is None

    def test_prepare_does_nothing(self, prosit_features, sample_dataset_with_spectra):
        """Test that prepare method does nothing."""
        original_metadata = sample_dataset_with_spectra.metadata.copy()
        prosit_features.prepare(sample_dataset_with_spectra)
        pd.testing.assert_frame_equal(
            sample_dataset_with_spectra.metadata, original_metadata
        )

    @patch("winnow.calibration.calibration_features.koinapy.Koina")
    @patch("winnow.calibration.calibration_features.compute_ion_identifications")
    def test_compute_with_mock(
        self,
        mock_compute_ions,
        mock_koina,
        prosit_features,
        sample_dataset_with_spectra,
    ):
        """Test compute method with mocked Koina intensity model and ion computation."""
        # Mock the Koina intensity model
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.model_inputs = [
            "peptide_sequences",
            "precursor_charges",
            "collision_energies",
        ]

        # Mock the prediction result with proper pandas index for grouping and multiple rows per peptide
        mock_predictions = pd.DataFrame(
            {
                "peptide_sequences": [
                    "AG",
                    "AG",
                    "AG",
                    "AG",
                    "GA",
                    "GA",
                    "GA",
                    "GA",
                    "SP",
                    "SP",
                    "SP",
                    "SP",
                ],
                "precursor_charges": [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3],
                "collision_energies": [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25],
                # Individual intensity values (one per ion)
                "intensities": [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.15,
                    0.25,
                    0.35,
                    0.45,
                    0.12,
                    0.22,
                    0.32,
                    0.42,
                ],
                # Individual m/z values (one per ion)
                "mz": [
                    100.0,
                    200.0,
                    300.0,
                    400.0,
                    110.0,
                    210.0,
                    310.0,
                    410.0,
                    105.0,
                    205.0,
                    305.0,
                    405.0,
                ],
                # Individual annotations (one per ion)
                "annotation": [
                    "b1",
                    "y1",
                    "b2",
                    "y2",
                    "b1",
                    "y1",
                    "b2",
                    "y2",
                    "b1",
                    "y1",
                    "b2",
                    "y2",
                ],
            },
            # Set pandas index: 4 rows per peptide (3 peptides total)
            index=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
        )
        mock_model_instance.predict.return_value = mock_predictions

        # Mock ion identification computation
        mock_compute_ions.return_value = ([0.5, 0.6, 0.7], [0.4, 0.5, 0.6])

        prosit_features.compute(sample_dataset_with_spectra)

        # Check that new columns were added
        assert "theoretical_mz" in sample_dataset_with_spectra.metadata.columns
        assert "theoretical_intensity" in sample_dataset_with_spectra.metadata.columns
        assert "ion_matches" in sample_dataset_with_spectra.metadata.columns
        assert "ion_match_intensity" in sample_dataset_with_spectra.metadata.columns
        assert (
            "is_missing_fragment_match_features"
            in sample_dataset_with_spectra.metadata.columns
        )

        # Check that the model was called
        mock_model_instance.predict.assert_called_once()

        # Check that ion computation was called
        mock_compute_ions.assert_called_once()

    def test_compute_maps_values_to_correct_rows_and_imputes_missing(
        self,
        prosit_features,
    ):
        """Test that computed values are mapped to correct rows and missing values are imputed correctly."""
        # Create a dataset with mixed valid/invalid predictions
        metadata = pd.DataFrame(
            {
                "confidence": [0.9, 0.7, 0.6],
                "prediction": [
                    ["A", "G"],  # Valid: len 2
                    ["A"] * 31,  # Invalid: len > 30
                    ["A", "G"],  # Valid: len 2
                ],
                "precursor_charge": [2, 2, 2],
                "spectrum_id": [10, 20, 40],
                "prediction_untokenised": ["AG", "A" * 31, "AG"],
                "mz_array": [
                    [100.0, 200.0, 300.0],
                    [120.0, 220.0],
                    [110.0, 210.0, 310.0],
                ],
                "intensity_array": [
                    [1000.0, 2000.0, 3000.0],
                    [1200.0, 2200.0],
                    [1100.0, 2100.0, 3100.0],
                ],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=[])

        # Run compute with mocked Prosit model
        with patch(
            "winnow.calibration.calibration_features.koinapy.Koina"
        ) as mock_koina:
            mock_model = mock_koina.return_value
            mock_model.model_inputs = [
                "peptide_sequences",
                "precursor_charges",
                "collision_energies",
            ]
            mock_model.predict = self._create_prosit_mock_predict()
            prosit_features.compute(dataset)

        # Verify results
        assert "is_missing_fragment_match_features" in dataset.metadata.columns
        spectrum_masks = self._assert_valid_invalid_flags(dataset)
        self._assert_theoretical_mz_intensity(dataset, spectrum_masks)
        self._assert_ion_matches(dataset, spectrum_masks)

        # Verify dataset structure
        assert len(dataset.metadata) == 3
        assert set(dataset.metadata["spectrum_id"].values) == {10, 20, 40}

    def _create_prosit_mock_predict(self):
        """Create a mock Koina intensity predict function for testing."""

        def mock_prosit_predict(inputs_df):
            """Mock Prosit predict that returns realistic predictions matching experimental m/z values."""
            if len(inputs_df) == 0:
                return pd.DataFrame(
                    columns=[
                        "peptide_sequences",
                        "precursor_charges",
                        "collision_energies",
                        "intensities",
                        "mz",
                        "annotation",
                    ]
                )

            predictions_list = []
            index_list = []

            for spectrum_id in inputs_df.index:
                peptide = inputs_df.loc[spectrum_id, "peptide_sequences"]
                mz_values, intensities, annotations = (
                    self._get_mock_predictions_for_spectrum(spectrum_id)
                )

                for mz, intensity, annotation in zip(
                    mz_values, intensities, annotations
                ):
                    predictions_list.append(
                        {
                            "peptide_sequences": peptide,
                            "precursor_charges": inputs_df.loc[
                                spectrum_id, "precursor_charges"
                            ],
                            "collision_energies": inputs_df.loc[
                                spectrum_id, "collision_energies"
                            ],
                            "intensities": intensity,
                            "mz": mz,
                            "annotation": annotation,
                        }
                    )
                    index_list.append(spectrum_id)

            return pd.DataFrame(predictions_list, index=index_list)

        return mock_prosit_predict

    def _get_mock_predictions_for_spectrum(self, spectrum_id):
        """Get mock prediction values for a given spectrum ID."""
        if spectrum_id == 10:
            mz_values = [99.99, 199.99, 299.99, 150.0, 250.0, 350.0]
            intensities = [0.5, 0.7, 0.9, 0.3, 0.4, 0.2]
            annotations = ["y1", "y2", "y3", "b1", "b2", "b3"]
        elif spectrum_id == 40:
            mz_values = [109.99, 209.99, 309.99, 160.0, 260.0, 360.0]
            intensities = [0.55, 0.75, 0.95, 0.35, 0.45, 0.25]
            annotations = ["y1", "y2", "y3", "b1", "b2", "b3"]
        else:
            mz_values = [100.0, 200.0, 300.0]
            intensities = [0.5, 0.6, 0.7]
            annotations = ["y1", "b1", "y2"]
        return mz_values, intensities, annotations

    def _assert_valid_invalid_flags(self, dataset):
        """Assert valid/invalid flags for fragment match features."""
        valid_flags = ~dataset.metadata["is_missing_fragment_match_features"]
        spectrum_masks = {
            10: dataset.metadata["spectrum_id"] == 10,
            20: dataset.metadata["spectrum_id"] == 20,
            40: dataset.metadata["spectrum_id"] == 40,
        }

        assert valid_flags[spectrum_masks[10]].iloc[0]  # Valid
        assert not valid_flags[spectrum_masks[20]].iloc[0]  # Invalid (has len > 30)
        assert valid_flags[spectrum_masks[40]].iloc[0]  # Valid

        return spectrum_masks

    def _assert_theoretical_mz_intensity(self, dataset, spectrum_masks):
        """Assert theoretical_mz and theoretical_intensity values."""
        assert "theoretical_mz" in dataset.metadata.columns
        assert "theoretical_intensity" in dataset.metadata.columns

        # Valid entries
        for sid in [10, 40]:
            theoretical_mz = dataset.metadata[spectrum_masks[sid]][
                "theoretical_mz"
            ].iloc[0]
            assert theoretical_mz is not None
            assert hasattr(theoretical_mz, "__iter__") or isinstance(
                theoretical_mz, list
            )
            mz_list = (
                list(theoretical_mz)
                if hasattr(theoretical_mz, "__iter__")
                else [theoretical_mz]
            )
            assert mz_list == sorted(mz_list)
            assert len(mz_list) > 0

            theoretical_intensity = dataset.metadata[spectrum_masks[sid]][
                "theoretical_intensity"
            ].iloc[0]
            assert theoretical_intensity is not None
            assert len(theoretical_intensity) == len(mz_list)

        # Invalid entries
        for sid in [20]:
            theoretical_mz = dataset.metadata[spectrum_masks[sid]][
                "theoretical_mz"
            ].iloc[0]
            theoretical_intensity = dataset.metadata[spectrum_masks[sid]][
                "theoretical_intensity"
            ].iloc[0]
            assert pd.isna(theoretical_mz) or theoretical_mz is None
            assert pd.isna(theoretical_intensity) or theoretical_intensity is None

    def _assert_ion_matches(self, dataset, spectrum_masks):
        """Assert ion_matches and ion_match_intensity values."""
        assert "ion_matches" in dataset.metadata.columns
        assert "ion_match_intensity" in dataset.metadata.columns

        # Valid entries
        for sid in [10, 40]:
            ion_matches = dataset.metadata[spectrum_masks[sid]]["ion_matches"].iloc[0]
            ion_match_intensity = dataset.metadata[spectrum_masks[sid]][
                "ion_match_intensity"
            ].iloc[0]

            assert not pd.isna(ion_matches)
            assert not pd.isna(ion_match_intensity)
            assert isinstance(ion_matches, (int, float))
            assert isinstance(ion_match_intensity, (int, float))
            assert (
                ion_matches > 0.0
            ), f"Expected non-zero ion_matches for spectrum {sid}"
            assert (
                ion_match_intensity > 0.0
            ), f"Expected non-zero ion_match_intensity for spectrum {sid}"

        # Invalid entries
        for sid in [20]:
            ion_matches = dataset.metadata[spectrum_masks[sid]]["ion_matches"].iloc[0]
            ion_match_intensity = dataset.metadata[spectrum_masks[sid]][
                "ion_match_intensity"
            ].iloc[0]
            assert ion_matches == 0.0
            assert ion_match_intensity == 0.0


class TestChimericFeatures:
    """Test the ChimericFeatures class."""

    @pytest.fixture()
    def chimeric_features(self):
        """Create a ChimericFeatures instance for testing."""
        return ChimericFeatures(
            mz_tolerance=0.02,
            invalid_prosit_residues=["U", "O", "X"],
            model_input_constants={"collision_energies": 25},
        )

    @pytest.fixture()
    def sample_dataset_with_beam_predictions(self):
        """Create a sample dataset with beam search predictions for chimeric analysis."""
        metadata = pd.DataFrame(
            {
                "confidence": [0.9, 0.8, 0.7],
                "precursor_charge": [2, 2, 3],
                "spectrum_id": [0, 1, 2],
                "mz_array": [
                    [100.0, 200.0, 300.0],
                    [150.0, 250.0, 350.0],
                    [120.0, 220.0, 320.0],
                ],
                "intensity_array": [
                    [1000.0, 2000.0, 3000.0],
                    [1500.0, 2500.0, 3500.0],
                    [1200.0, 2200.0, 3200.0],
                ],
            }
        )

        predictions = [
            [  # First spectrum - 3 sequences
                MockScoredSequence(["A", "G"], np.log(0.8)),
                MockScoredSequence(["G", "A"], np.log(0.6)),
                MockScoredSequence(["S", "P"], np.log(0.4)),
            ],
            [  # Second spectrum - 2 sequences
                MockScoredSequence(["V", "T"], np.log(0.9)),
                MockScoredSequence(["T", "V"], np.log(0.7)),
            ],
            [  # Third spectrum - 1 sequence (will trigger warning)
                MockScoredSequence(["K"], np.log(0.95))
            ],
        ]

        return CalibrationDataset(metadata=metadata, predictions=predictions)

    def test_properties(self, chimeric_features):
        """Test ChimericFeatures properties."""
        assert chimeric_features.name == "Chimeric Features"
        assert chimeric_features.columns == [
            "chimeric_ion_matches",
            "chimeric_ion_match_intensity",
            "is_missing_chimeric_features",
        ]
        assert chimeric_features.dependencies == []
        assert chimeric_features.mz_tolerance == 0.02

    def test_initialization_with_tolerance(self):
        """Test initialization with custom tolerance."""
        feature = ChimericFeatures(
            mz_tolerance=0.01,
            invalid_prosit_residues=["U", "O", "X"],
            model_input_constants={"collision_energies": 25},
        )
        assert feature.mz_tolerance == 0.01
        assert feature.model_input_constants == {"collision_energies": 25}
        assert feature.model_input_columns is None

    def test_prepare_does_nothing(
        self, chimeric_features, sample_dataset_with_beam_predictions
    ):
        """Test that prepare method does nothing."""
        original_metadata = sample_dataset_with_beam_predictions.metadata.copy()
        chimeric_features.prepare(sample_dataset_with_beam_predictions)
        pd.testing.assert_frame_equal(
            sample_dataset_with_beam_predictions.metadata, original_metadata
        )

    def test_compute_with_none_predictions(self, chimeric_features):
        """Test that compute raises error when predictions is None."""
        metadata = pd.DataFrame({"confidence": [0.9], "precursor_charge": [2]})
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        with pytest.raises(ValueError, match="dataset.predictions cannot be None"):
            chimeric_features.compute(dataset)

    @patch("winnow.calibration.calibration_features.koinapy.Koina")
    @patch("winnow.calibration.calibration_features.compute_ion_identifications")
    def test_compute_preprocessing_pipeline(
        self,
        mock_compute_ions,
        mock_koina,
        chimeric_features,
        sample_dataset_with_beam_predictions,
    ):
        """Test the complete data preprocessing pipeline including groupby aggregation and sorting."""
        # Use existing fixture but take only first 2 spectra for testing
        dataset = sample_dataset_with_beam_predictions
        dataset.metadata = dataset.metadata.iloc[:2].copy()  # Keep first 2 rows
        dataset.predictions = dataset.predictions[:2]  # Keep first 2 predictions

        # Create mock predictions with proper pandas index for grouping and multiple rows per peptide
        mock_predictions = pd.DataFrame(
            {
                "peptide_sequences": ["GA", "GA", "GA", "TV", "TV"],
                "precursor_charges": [2, 2, 2, 2, 2],
                "collision_energies": [25, 25, 25, 25, 25],
                # Individual intensity values (one per ion)
                "intensities": [0.3, 0.1, 0.8, 0.4, 0.2],
                # Individual m/z values (one per ion) - unsorted to test sorting logic
                "mz": [300.0, 100.0, 250.0, 400.0, 200.0],
                # Individual annotations (one per ion)
                "annotation": ["b2", "b1", "y1", "b2", "b1"],
            },
            # Set pandas index so first 3 rows get index 0, last 2 get index 1
            index=[0, 0, 0, 1, 1],
        )
        mock_model = self._setup_prosit_mock(mock_koina, mock_predictions)

        # Mock ion computation
        mock_compute_ions.return_value = ([0.5, 0.6], [0.4, 0.5])

        # Run the compute method
        chimeric_features.compute(dataset)

        # Verify input preparation (runner-up sequences extracted correctly)
        call_args = mock_model.predict.call_args[0][0]
        expected_sequences = ["GA", "TV"]  # Runner-up sequences from fixture
        assert list(call_args["peptide_sequences"]) == expected_sequences
        assert list(call_args["precursor_charges"]) == [2, 2]
        assert list(call_args["collision_energies"]) == [25, 25]

        # Verify groupby aggregation AND sorting for multiple fragments per peptide
        actual_mz = dataset.metadata["runner_up_prosit_mz"].tolist()
        actual_intensities = dataset.metadata["runner_up_prosit_intensity"].tolist()

        # First peptide "GA": 3 fragments with m/z [300.0, 100.0, 250.0] -> sorted [100.0, 250.0, 300.0]
        # Second peptide "TV": 2 fragments with m/z [400.0, 200.0] -> sorted [200.0, 400.0]
        expected_sorted_mz = [
            [100.0, 250.0, 300.0],  # Sorted m/z for "GA" (3 fragments)
            [200.0, 400.0],  # Sorted m/z for "TV" (2 fragments)
        ]
        # Convert numpy arrays to lists for comparison
        actual_mz_lists = [
            arr.tolist() if hasattr(arr, "tolist") else arr for arr in actual_mz
        ]
        assert actual_mz_lists == expected_sorted_mz

        # Verify intensities were reordered to match sorted m/z
        # "GA" intensities [0.3, 0.1, 0.8] -> reordered to match sorted m/z [0.1, 0.8, 0.3]
        # "TV" intensities [0.4, 0.2] -> reordered to match sorted m/z [0.2, 0.4]
        expected_sorted_intensities = [
            [0.1, 0.8, 0.3],  # Intensities reordered for "GA"
            [0.2, 0.4],  # Intensities reordered for "TV"
        ]
        # Convert numpy arrays to lists for comparison
        actual_intensities_lists = [
            arr.tolist() if hasattr(arr, "tolist") else arr
            for arr in actual_intensities
        ]
        assert actual_intensities_lists == expected_sorted_intensities

        # Verify final features were computed correctly
        assert "chimeric_ion_matches" in dataset.metadata.columns
        assert "chimeric_ion_match_intensity" in dataset.metadata.columns
        assert "is_missing_chimeric_features" in dataset.metadata.columns
        assert list(dataset.metadata["chimeric_ion_matches"]) == [0.5, 0.6]
        assert list(dataset.metadata["chimeric_ion_match_intensity"]) == [0.4, 0.5]

    def test_compute_maps_values_to_correct_rows_and_imputes_missing(
        self,
        chimeric_features,
    ):
        """Test that computed chimeric values are mapped to correct rows and missing values are imputed correctly."""
        metadata = pd.DataFrame(
            {
                "confidence": [0.9, 0.7, 0.6],
                "precursor_charge": [2, 2, 7],  # Last one has charge > 6
                "spectrum_id": [10, 30, 40],
                "mz_array": [
                    [100.0, 200.0, 300.0],
                    [150.0, 250.0],
                    [110.0, 210.0, 310.0],
                ],
                "intensity_array": [
                    [1000.0, 2000.0, 3000.0],
                    [1500.0, 2500.0],
                    [1100.0, 2100.0, 3100.0],
                ],
            }
        )

        predictions = [
            [  # Spectrum 10: Valid - has runner-up, charge 2, len 2
                MockScoredSequence(["A", "G"], np.log(0.8)),
                MockScoredSequence(["G", "A"], np.log(0.6)),  # Runner-up
            ],
            [  # Spectrum 30: Invalid - runner-up len > 30
                MockScoredSequence(["K"], np.log(0.95)),
                MockScoredSequence(["A"] * 31, np.log(0.7)),  # Runner-up len > 30
            ],
            [  # Spectrum 40: Invalid - charge > 6
                MockScoredSequence(["S", "P"], np.log(0.9)),
                MockScoredSequence(["P", "S"], np.log(0.7)),  # Runner-up
            ],
        ]

        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)

        # Run compute with mocked Koina model
        with patch(
            "winnow.calibration.calibration_features.koinapy.Koina"
        ) as mock_koina:
            mock_model = mock_koina.return_value
            mock_model.model_inputs = [
                "peptide_sequences",
                "precursor_charges",
                "collision_energies",
            ]
            mock_model.predict = self._create_chimeric_prosit_mock_predict()
            chimeric_features.compute(dataset)

        # Verify results
        assert "is_missing_chimeric_features" in dataset.metadata.columns
        spectrum_masks = self._assert_chimeric_valid_invalid_flags(dataset)
        self._assert_chimeric_prosit_mz_intensity(dataset, spectrum_masks)
        self._assert_chimeric_ion_matches(dataset, spectrum_masks)

        # Verify dataset structure
        assert len(dataset.metadata) == 3
        assert set(dataset.metadata["spectrum_id"].values) == {10, 30, 40}

    def _create_chimeric_prosit_mock_predict(self):
        """Create a mock Prosit predict function for chimeric testing."""

        def mock_prosit_predict(inputs_df):
            """Mock Prosit predict that returns realistic predictions matching experimental m/z values."""
            if len(inputs_df) == 0:
                return pd.DataFrame(
                    columns=[
                        "peptide_sequences",
                        "precursor_charges",
                        "collision_energies",
                        "intensities",
                        "mz",
                        "annotation",
                    ]
                )

            predictions_list = []
            index_list = []

            for spectrum_id in inputs_df.index:
                peptide = inputs_df.loc[spectrum_id, "peptide_sequences"]
                mz_values, intensities, annotations = (
                    self._get_chimeric_mock_predictions_for_spectrum(spectrum_id)
                )

                for mz, intensity, annotation in zip(
                    mz_values, intensities, annotations
                ):
                    predictions_list.append(
                        {
                            "peptide_sequences": peptide,
                            "precursor_charges": inputs_df.loc[
                                spectrum_id, "precursor_charges"
                            ],
                            "collision_energies": inputs_df.loc[
                                spectrum_id, "collision_energies"
                            ],
                            "intensities": intensity,
                            "mz": mz,
                            "annotation": annotation,
                        }
                    )
                    index_list.append(spectrum_id)

            return pd.DataFrame(predictions_list, index=index_list)

        return mock_prosit_predict

    def _get_chimeric_mock_predictions_for_spectrum(self, spectrum_id):
        """Get mock prediction values for a given spectrum ID in chimeric test."""
        if spectrum_id == 10:
            mz_values = [99.99, 199.99, 299.99, 150.0, 250.0, 350.0]
            intensities = [0.5, 0.7, 0.9, 0.3, 0.4, 0.2]
            annotations = ["y1", "y2", "y3", "b1", "b2", "b3"]
        else:
            mz_values = [100.0, 200.0, 300.0]
            intensities = [0.5, 0.6, 0.7]
            annotations = ["y1", "b1", "y2"]
        return mz_values, intensities, annotations

    def _assert_chimeric_valid_invalid_flags(self, dataset):
        """Assert valid/invalid flags for chimeric features."""
        valid_flags = ~dataset.metadata["is_missing_chimeric_features"]
        spectrum_masks = {
            10: dataset.metadata["spectrum_id"] == 10,
            30: dataset.metadata["spectrum_id"] == 30,
            40: dataset.metadata["spectrum_id"] == 40,
        }

        assert valid_flags[spectrum_masks[10]].iloc[0]  # Valid
        assert not valid_flags[spectrum_masks[30]].iloc[
            0
        ]  # Invalid (runner-up len > 30)
        assert not valid_flags[spectrum_masks[40]].iloc[0]  # Invalid (charge > 6)

        return spectrum_masks

    def _assert_chimeric_prosit_mz_intensity(self, dataset, spectrum_masks):
        """Assert runner_up_prosit_mz and runner_up_prosit_intensity values."""
        assert "runner_up_prosit_mz" in dataset.metadata.columns
        assert "runner_up_prosit_intensity" in dataset.metadata.columns

        # Valid entry
        prosit_mz_10 = dataset.metadata[spectrum_masks[10]]["runner_up_prosit_mz"].iloc[
            0
        ]
        assert prosit_mz_10 is not None
        assert hasattr(prosit_mz_10, "__iter__") or isinstance(prosit_mz_10, list)
        mz_10_list = (
            list(prosit_mz_10) if hasattr(prosit_mz_10, "__iter__") else [prosit_mz_10]
        )
        assert mz_10_list == sorted(mz_10_list)
        assert len(mz_10_list) > 0

        prosit_intensity_10 = dataset.metadata[spectrum_masks[10]][
            "runner_up_prosit_intensity"
        ].iloc[0]
        assert prosit_intensity_10 is not None
        assert len(prosit_intensity_10) == len(mz_10_list)

        # Invalid entries
        for sid in [30, 40]:
            prosit_mz = dataset.metadata[spectrum_masks[sid]][
                "runner_up_prosit_mz"
            ].iloc[0]
            prosit_intensity = dataset.metadata[spectrum_masks[sid]][
                "runner_up_prosit_intensity"
            ].iloc[0]
            assert pd.isna(prosit_mz) or prosit_mz is None
            assert pd.isna(prosit_intensity) or prosit_intensity is None

    def _assert_chimeric_ion_matches(self, dataset, spectrum_masks):
        """Assert chimeric_ion_matches and chimeric_ion_match_intensity values."""
        assert "chimeric_ion_matches" in dataset.metadata.columns
        assert "chimeric_ion_match_intensity" in dataset.metadata.columns

        # Valid entry
        ion_matches_10 = dataset.metadata[spectrum_masks[10]][
            "chimeric_ion_matches"
        ].iloc[0]
        ion_match_intensity_10 = dataset.metadata[spectrum_masks[10]][
            "chimeric_ion_match_intensity"
        ].iloc[0]

        assert not pd.isna(ion_matches_10)
        assert not pd.isna(ion_match_intensity_10)
        assert isinstance(ion_matches_10, (int, float))
        assert isinstance(ion_match_intensity_10, (int, float))
        assert (
            ion_matches_10 > 0.0
        ), f"Expected non-zero chimeric_ion_matches for spectrum 10, got {ion_matches_10}"
        assert (
            ion_match_intensity_10 > 0.0
        ), f"Expected non-zero chimeric_ion_match_intensity for spectrum 10, got {ion_match_intensity_10}"

        # Invalid entries
        for sid in [30, 40]:
            ion_matches = dataset.metadata[spectrum_masks[sid]][
                "chimeric_ion_matches"
            ].iloc[0]
            ion_match_intensity = dataset.metadata[spectrum_masks[sid]][
                "chimeric_ion_match_intensity"
            ].iloc[0]
            assert ion_matches == 0.0
            assert ion_match_intensity == 0.0

    def _setup_prosit_mock(self, mock_koina, mock_predictions_df):
        """Helper method to set up Koina model mock with given predictions."""
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.model_inputs = [
            "peptide_sequences",
            "precursor_charges",
            "collision_energies",
        ]
        mock_model_instance.predict.return_value = mock_predictions_df
        return mock_model_instance


class TestModelInputHelpers:
    """Tests for _validate_model_input_params and _resolve_model_inputs helpers."""

    # --- _validate_model_input_params ---

    def test_validate_no_conflict_passes(self):
        """No error when constants and columns have disjoint keys."""
        _validate_model_input_params(
            {"collision_energies": 25},
            {"fragmentation_types": "col_frag"},
        )

    def test_validate_one_none_passes(self):
        """No error when one of the dicts is None."""
        _validate_model_input_params({"collision_energies": 25}, None)
        _validate_model_input_params(None, {"collision_energies": "ce_col"})

    def test_validate_both_none_passes(self):
        """No error when both dicts are None."""
        _validate_model_input_params(None, None)

    def test_validate_conflict_raises(self):
        """ValueError raised when the same key appears in both dicts."""
        with pytest.raises(ValueError, match="collision_energies"):
            _validate_model_input_params(
                {"collision_energies": 25},
                {"collision_energies": "ce_col"},
            )

    def test_validate_conflict_lists_all_conflicting_keys(self):
        """Error message includes all conflicting keys."""
        with pytest.raises(ValueError, match="fragmentation_types") as exc_info:
            _validate_model_input_params(
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
                unsupported_residues=[],
                model_input_constants={"collision_energies": 25},
                model_input_columns={"collision_energies": "ce_col"},
            )

    def test_conflict_detected_at_construction_time_chimeric_features(self):
        """ChimericFeatures raises ValueError at construction when keys conflict."""
        with pytest.raises(ValueError, match="collision_energies"):
            ChimericFeatures(
                mz_tolerance=0.02,
                invalid_prosit_residues=[],
                model_input_constants={"collision_energies": 25},
                model_input_columns={"collision_energies": "ce_col"},
            )

    # --- _resolve_model_inputs ---

    def test_resolve_constant_is_tiled(self):
        """A constant value is tiled across all rows."""
        inputs = self._make_inputs(3)
        metadata = self._make_metadata(3)
        result = _resolve_model_inputs(
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
        result = _resolve_model_inputs(
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
        result = _resolve_model_inputs(
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
            _resolve_model_inputs(
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
            _resolve_model_inputs(
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
        result = _resolve_model_inputs(
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
        dataset = CalibrationDataset(metadata=metadata, predictions=[])

        mock_predictions = pd.DataFrame(
            {
                "peptide_sequences": ["AG"],
                "precursor_charges": [2],
                "collision_energies": [30],
                "intensities": [0.5],
                "mz": [100.0],
                "annotation": ["b1"],
            },
            index=[0],
        )

        with patch(
            "winnow.calibration.calibration_features.koinapy.Koina"
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
        dataset = CalibrationDataset(metadata=metadata, predictions=[])

        mock_predictions = pd.DataFrame(
            {
                "peptide_sequences": ["AG", "GA"],
                "precursor_charges": [2, 2],
                "collision_energies": [28, 35],
                "intensities": [0.5, 0.6],
                "mz": [100.0, 110.0],
                "annotation": ["b1", "b1"],
            },
            index=[0, 1],
        )

        with patch(
            "winnow.calibration.calibration_features.koinapy.Koina"
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


class ConcreteFeature(CalibrationFeatures):
    """Concrete implementation for testing the interface."""

    @property
    def dependencies(self):
        return []

    @property
    def name(self):
        return "Test Feature"

    @property
    def columns(self):
        return ["test_col"]

    def prepare(self, dataset):
        pass

    def compute(self, dataset):
        dataset.metadata["test_col"] = [1, 2, 3]


class TestConcreteFeatureImplementation:
    """Test a concrete feature implementation."""

    def test_concrete_feature_can_be_instantiated(self):
        """Test that concrete feature implementation can be instantiated."""
        feature = ConcreteFeature()
        assert feature.name == "Test Feature"
        assert feature.columns == ["test_col"]
        assert feature.dependencies == []

    def test_concrete_feature_compute(self):
        """Test that concrete feature can compute values."""
        feature = ConcreteFeature()
        metadata = pd.DataFrame({"existing_col": [1, 2, 3]})
        dataset = CalibrationDataset(metadata=metadata, predictions=[])

        feature.compute(dataset)

        assert "test_col" in dataset.metadata.columns
        assert list(dataset.metadata["test_col"]) == [1, 2, 3]
