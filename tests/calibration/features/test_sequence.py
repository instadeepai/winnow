"""Unit tests for winnow calibration feature SequenceFeatures."""

import pytest
import pandas as pd
from winnow.calibration.features.sequence import SequenceFeatures
from winnow.datasets.calibration_dataset import CalibrationDataset


class TestSequenceFeatures:
    """Test the SequenceFeatures class."""

    @pytest.fixture()
    def sequence_features(self):
        """Create a SequenceFeatures instance for testing."""
        return SequenceFeatures()

    @pytest.fixture()
    def sample_dataset(self):
        """Create a sample CalibrationDataset for testing."""
        metadata = pd.DataFrame(
            {
                "prediction": [
                    ["A", "C", "D", "K"],
                    ["G", "H", "I", "R"],
                    ["L", "M", "N"],
                    ["P", "Q"],
                ],
                "precursor_charge": [2, 3, 2, 1],
                "confidence": [0.9, 0.8, 0.7, 0.6],
            }
        )
        return CalibrationDataset(metadata=metadata, predictions=None)

    def test_properties(self, sequence_features):
        """Test SequenceFeatures properties."""
        assert sequence_features.name == "Sequence Features"
        assert sequence_features.columns == [
            "sequence_length",
            "precursor_charge",
        ]
        assert sequence_features.dependencies == []

    def test_prepare_does_nothing(self, sequence_features, sample_dataset):
        """Test that prepare method does nothing."""
        original_metadata = sample_dataset.metadata.copy()
        sequence_features.prepare(sample_dataset)
        pd.testing.assert_frame_equal(sample_dataset.metadata, original_metadata)

    def test_compute_sequence_length(self, sequence_features, sample_dataset):
        """Test that sequence_length is computed correctly for each row."""
        sequence_features.prepare(sample_dataset)
        sequence_features.compute(sample_dataset)

        assert "sequence_length" in sample_dataset.metadata.columns
        expected_lengths = [4, 4, 3, 2]
        assert list(sample_dataset.metadata["sequence_length"]) == expected_lengths

    def test_compute_precursor_charge_unchanged(
        self, sequence_features, sample_dataset
    ):
        """Test that precursor_charge column is preserved (not modified)."""
        original_charges = list(sample_dataset.metadata["precursor_charge"])
        sequence_features.prepare(sample_dataset)
        sequence_features.compute(sample_dataset)

        # precursor_charge should remain the same (it's just included in columns)
        assert list(sample_dataset.metadata["precursor_charge"]) == original_charges
