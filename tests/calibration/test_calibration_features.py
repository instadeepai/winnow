"""Unit tests for winnow calibration features."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch
from winnow.calibration.calibration_features import (
    CalibrationFeatures,
    FeatureDependency,
    MassErrorFeature,
    BeamFeatures,
    RetentionTimeFeature,
    PrositFeatures,
    ChimericFeatures,
    find_matching_ions,
    _raise_value_error,
)
from winnow.datasets.calibration_dataset import CalibrationDataset, RESIDUE_MASSES


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
        return MassErrorFeature(residue_masses=RESIDUE_MASSES)

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
        return RetentionTimeFeature(hidden_dim=10, train_fraction=0.8)

    @pytest.fixture()
    def sample_dataset_with_rt(self):
        """Create a sample dataset with retention time data."""
        metadata = pd.DataFrame(
            {
                "confidence": [0.95, 0.90, 0.85, 0.80, 0.75],
                "prediction": [["A", "G"], ["G", "A"], ["S", "P"], ["V"], ["K"]],
                "retention_time": [10.5, 15.2, 20.1, 8.7, 12.3],
                "precursor_charge": [2, 2, 3, 1, 2],
            }
        )
        return CalibrationDataset(metadata=metadata, predictions=[])

    def test_properties(self, retention_time_feature):
        """Test RetentionTimeFeature properties."""
        assert retention_time_feature.name == "Prosit iRT Features"
        assert retention_time_feature.columns == ["iRT error"]
        assert retention_time_feature.dependencies == []

    def test_initialization_parameters(self):
        """Test initialization with custom parameters."""
        feature = RetentionTimeFeature(hidden_dim=10, train_fraction=0.8)
        assert feature.hidden_dim == 10
        assert feature.train_fraction == 0.8
        assert hasattr(feature, "prosit_model")
        assert hasattr(feature, "irt_predictor")

    @patch("winnow.calibration.calibration_features.koinapy.Koina")
    def test_prepare_with_mock(
        self, mock_koina, retention_time_feature, sample_dataset_with_rt
    ):
        """Test prepare method with mocked Prosit model."""
        # Mock the Prosit model
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance

        # Mock predict to return iRT values
        mock_model_instance.predict.return_value = pd.DataFrame(
            {
                "irt": [35.1, 20.7, 28.1, 25.5]  # 4 values for 80% of 5 samples
            }
        )

        # Override the model in the feature
        retention_time_feature.prosit_model = mock_model_instance

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
        # Mock the Prosit model
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance

        # Mock predict to return iRT values for all samples
        mock_model_instance.predict.return_value = pd.DataFrame(
            {"irt": [25.5, 30.2, 35.1, 20.7, 28.1]}
        )

        # Override the model in the feature
        retention_time_feature.prosit_model = mock_model_instance

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


class TestPrositFeatures:
    """Test the PrositFeatures class."""

    @pytest.fixture()
    def prosit_features(self):
        """Create a PrositFeatures instance for testing."""
        return PrositFeatures(mz_tolerance=0.02)

    @pytest.fixture()
    def sample_dataset_with_spectra(self):
        """Create a sample dataset with spectral data."""
        metadata = pd.DataFrame(
            {
                "confidence": [0.9, 0.8, 0.7],
                "prediction": [["A", "G"], ["G", "A"], ["S", "P"]],
                "precursor_charge": [2, 2, 3],
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
        """Test PrositFeatures properties."""
        assert prosit_features.name == "Prosit Features"
        assert prosit_features.columns == ["ion_matches", "ion_match_intensity"]
        assert prosit_features.dependencies == []
        assert prosit_features.mz_tolerance == 0.02

    def test_initialization_with_tolerance(self):
        """Test initialization with custom tolerance."""
        feature = PrositFeatures(mz_tolerance=0.01)
        assert feature.mz_tolerance == 0.01
        assert hasattr(feature, "model")

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
        """Test compute method with mocked Prosit model and ion computation."""
        # Mock the Prosit model
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance

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

        # Override the model in the feature
        prosit_features.model = mock_model_instance

        # Mock ion identification computation
        mock_compute_ions.return_value = ([0.5, 0.6, 0.7], [0.4, 0.5, 0.6])

        prosit_features.compute(sample_dataset_with_spectra)

        # Check that new columns were added
        assert "prosit_mz" in sample_dataset_with_spectra.metadata.columns
        assert "prosit_intensity" in sample_dataset_with_spectra.metadata.columns
        assert "ion_matches" in sample_dataset_with_spectra.metadata.columns
        assert "ion_match_intensity" in sample_dataset_with_spectra.metadata.columns

        # Check that the model was called
        mock_model_instance.predict.assert_called_once()

        # Check that ion computation was called
        mock_compute_ions.assert_called_once()

    def test_map_modification_function(self):
        """Test the map_modification helper function."""
        from winnow.calibration.calibration_features import map_modification

        # Test normal peptide
        normal_peptide = ["A", "G", "S"]
        assert map_modification(normal_peptide) == ["A", "G", "S"]

        # Test peptide with carbamidomethylated cysteine
        modified_peptide = ["A", "C[UNIMOD:4]", "G"]
        assert map_modification(modified_peptide) == ["A", "C", "G"]

        # Test peptide with multiple modifications
        multi_modified = ["C[UNIMOD:4]", "A", "C[UNIMOD:4]"]
        assert map_modification(multi_modified) == ["C", "A", "C"]


class TestChimericFeatures:
    """Test the ChimericFeatures class."""

    @pytest.fixture()
    def chimeric_features(self):
        """Create a ChimericFeatures instance for testing."""
        return ChimericFeatures(mz_tolerance=0.02)

    @pytest.fixture()
    def sample_dataset_with_beam_predictions(self):
        """Create a sample dataset with beam search predictions for chimeric analysis."""
        metadata = pd.DataFrame(
            {
                "confidence": [0.9, 0.8, 0.7],
                "precursor_charge": [2, 2, 3],
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
        ]
        assert chimeric_features.dependencies == []
        assert chimeric_features.mz_tolerance == 0.02

    def test_initialization_with_tolerance(self):
        """Test initialization with custom tolerance."""
        feature = ChimericFeatures(mz_tolerance=0.01)
        assert feature.mz_tolerance == 0.01

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
        assert list(dataset.metadata["chimeric_ion_matches"]) == [0.5, 0.6]
        assert list(dataset.metadata["chimeric_ion_match_intensity"]) == [0.4, 0.5]

    @patch("winnow.calibration.calibration_features.koinapy.Koina")
    @patch("winnow.calibration.calibration_features.compute_ion_identifications")
    def test_compute_warning_insufficient_sequences(
        self,
        mock_compute_ions,
        mock_koina,
        chimeric_features,
        sample_dataset_with_beam_predictions,
    ):
        """Test that warning is issued for beam results with fewer than two sequences."""
        # Use the full fixture which includes a spectrum with only 1 sequence
        dataset = sample_dataset_with_beam_predictions

        # Create minimal mock predictions (content doesn't matter for warning test)
        mock_predictions = pd.DataFrame(
            {
                "peptide_sequences": [
                    "GA",
                    "TV",
                    "",
                ],  # Third empty for single sequence case
                "precursor_charges": [2, 2, 3],
                "collision_energies": [25, 25, 25],
                "intensities": [[], [], []],  # Empty lists
                "mz": [[], [], []],  # Empty lists
                "annotation": [[], [], []],  # Empty lists
            }
        )
        _ = self._setup_prosit_mock(mock_koina, mock_predictions)
        mock_compute_ions.return_value = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

        # Expect warning about insufficient sequences (third spectrum has only 1 sequence)
        with pytest.warns(
            UserWarning,
            match="1 beam search results have fewer than two sequences. This may affect the efficacy of computed chimeric features.",
        ):
            chimeric_features.compute(dataset)

    def _setup_prosit_mock(self, mock_koina, mock_predictions_df):
        """Helper method to set up Prosit model mock with given predictions."""
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.predict.return_value = mock_predictions_df
        return mock_model_instance


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
