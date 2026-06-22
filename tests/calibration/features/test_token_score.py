"""Unit tests for winnow calibration feature TokenScoreFeatures."""

import pytest
import pandas as pd
import numpy as np
from winnow.calibration.features.token_score import TokenScoreFeatures
from winnow.datasets.calibration_dataset import CalibrationDataset
from tests.calibration.features.conftest import MockScoredSequence


class TestTokenScoreFeatures:
    """Test the TokenScoreFeatures class."""

    @pytest.fixture()
    def token_score_features(self):
        """Create a TokenScoreFeatures instance for testing."""
        return TokenScoreFeatures()

    @pytest.fixture()
    def sample_dataset_with_beam_predictions(self):
        """Create a CalibrationDataset with beam predictions containing token log probs."""
        metadata = pd.DataFrame(
            {
                "prediction": [["A", "C", "D"], ["G", "H"], ["L", "M", "N", "P"]],
                "confidence": [0.9, 0.8, 0.7],
            }
        )
        # Token log probabilities that convert to probabilities:
        # Row 0: exp([-0.105, -0.223, -0.357]) ≈ [0.9, 0.8, 0.7] -> min=0.7
        # Row 1: exp([-0.051, -0.163]) ≈ [0.95, 0.85] -> min=0.85
        # Row 2: exp([-0.511, -0.357, -0.223, -0.105]) ≈ [0.6, 0.7, 0.8, 0.9] -> min=0.6
        predictions = [
            [
                MockScoredSequence(
                    sequence=["A", "C", "D"],
                    log_prob=-0.5,
                    token_log_probs=np.log([0.9, 0.8, 0.7]),
                )
            ],
            [
                MockScoredSequence(
                    sequence=["G", "H"],
                    log_prob=-0.4,
                    token_log_probs=np.log([0.95, 0.85]),
                )
            ],
            [
                MockScoredSequence(
                    sequence=["L", "M", "N", "P"],
                    log_prob=-0.6,
                    token_log_probs=np.log([0.6, 0.7, 0.8, 0.9]),
                )
            ],
        ]
        return CalibrationDataset(metadata=metadata, predictions=predictions)

    @pytest.fixture()
    def sample_dataset_no_beam_predictions(self):
        """Create a CalibrationDataset without beam predictions."""
        metadata = pd.DataFrame(
            {
                "prediction": [["A", "C", "D"]],
                "confidence": [0.9],
            }
        )
        return CalibrationDataset(metadata=metadata, predictions=None)

    @pytest.fixture()
    def sample_dataset_missing_token_log_probs(self):
        """Create a CalibrationDataset with beam predictions but no token log probs."""
        metadata = pd.DataFrame(
            {
                "prediction": [["A", "C"]],
                "confidence": [0.9],
            }
        )
        predictions = [
            [
                MockScoredSequence(
                    sequence=["A", "C"],
                    log_prob=-0.5,
                    token_log_probs=None,
                )
            ]
        ]
        return CalibrationDataset(metadata=metadata, predictions=predictions)

    def test_properties(self, token_score_features):
        """Test TokenScoreFeatures properties."""
        assert token_score_features.name == "Token Score Features"
        assert token_score_features.columns == [
            "min_token_probability",
            "std_token_probability",
        ]
        assert token_score_features.dependencies == []

    def test_prepare_does_nothing(
        self, token_score_features, sample_dataset_with_beam_predictions
    ):
        """Test that prepare is a no-op."""
        original_metadata = sample_dataset_with_beam_predictions.metadata.copy()
        token_score_features.prepare(sample_dataset_with_beam_predictions)
        pd.testing.assert_frame_equal(
            sample_dataset_with_beam_predictions.metadata, original_metadata
        )

    def test_compute_raises_without_beam_predictions(
        self, token_score_features, sample_dataset_no_beam_predictions
    ):
        """Test that compute raises ValueError when beam predictions are not available."""
        with pytest.raises(ValueError, match="requires beam predictions"):
            token_score_features.compute(sample_dataset_no_beam_predictions)

    def test_compute_raises_without_token_log_probs(
        self, token_score_features, sample_dataset_missing_token_log_probs
    ):
        """Test that compute raises ValueError when token_log_probabilities is None."""
        with pytest.raises(
            ValueError, match="Token log probabilities are not available"
        ):
            token_score_features.compute(sample_dataset_missing_token_log_probs)

    def test_compute_min_token_probability(
        self, token_score_features, sample_dataset_with_beam_predictions
    ):
        """Test that min_token_probability is computed correctly."""
        token_score_features.compute(sample_dataset_with_beam_predictions)

        assert (
            "min_token_probability"
            in sample_dataset_with_beam_predictions.metadata.columns
        )
        # Expected: min of exp(log_probs) = min of probabilities
        expected_mins = [0.7, 0.85, 0.6]
        actual_mins = list(
            sample_dataset_with_beam_predictions.metadata["min_token_probability"]
        )
        for expected, actual in zip(expected_mins, actual_mins):
            assert actual == pytest.approx(expected, abs=1e-6)

    def test_compute_std_token_probability(
        self, token_score_features, sample_dataset_with_beam_predictions
    ):
        """Test that std_token_probability is computed correctly."""
        token_score_features.compute(sample_dataset_with_beam_predictions)

        assert (
            "std_token_probability"
            in sample_dataset_with_beam_predictions.metadata.columns
        )
        # Calculate expected std values from probabilities
        expected_stds = [
            np.std([0.9, 0.8, 0.7]),
            np.std([0.95, 0.85]),
            np.std([0.6, 0.7, 0.8, 0.9]),
        ]
        actual_stds = list(
            sample_dataset_with_beam_predictions.metadata["std_token_probability"]
        )
        for expected, actual in zip(expected_stds, actual_stds):
            assert actual == pytest.approx(expected, abs=1e-6)

    def test_compute_single_token_sequence(self, token_score_features):
        """Test that single-token sequences have std=0."""
        metadata = pd.DataFrame(
            {
                "prediction": [["A"]],
                "confidence": [0.9],
            }
        )
        predictions = [
            [
                MockScoredSequence(
                    sequence=["A"],
                    log_prob=-0.1,
                    token_log_probs=np.array([-0.1]),
                )
            ]
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)
        token_score_features.compute(dataset)

        assert dataset.metadata["std_token_probability"].iloc[0] == 0.0
        assert dataset.metadata["min_token_probability"].iloc[0] == pytest.approx(
            np.exp(-0.1), abs=1e-6
        )
