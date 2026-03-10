"""Unit tests for winnow CalibrationDataset."""

import numpy as np
import pandas as pd
import pytest
from winnow.datasets.calibration_dataset import CalibrationDataset


class MockScoredSequence:
    """Mock class for ScoredSequence used in beam search results."""

    def __init__(self, sequence, log_prob=0.0):
        self.sequence = sequence
        self.sequence_log_probability = log_prob


class TestCalibrationDataset:
    """Test the CalibrationDataset class."""

    @pytest.fixture()
    def sample_metadata(self):
        """Create sample metadata DataFrame."""
        return pd.DataFrame(
            {
                "confidence": [0.9, 0.8, 0.7, 0.6, 0.0],
                "prediction": [["A", "G"], ["G", "A"], ["S", "P"], ["V"], None],
                "sequence": [["A", "G"], ["G", "A"], ["A", "P"], ["V"], ["P"]],
                "precursor_mz": [100.508, 100.508, 125.511, 75.505, 90.507],
                "precursor_charge": [2, 2, 2, 2, 2],
                "correct": [True, True, False, True, False],
            }
        )

    @pytest.fixture()
    def sample_predictions(self):
        """Create sample beam search predictions."""
        return [
            [
                MockScoredSequence(["A", "G"], np.log(0.8)),
                MockScoredSequence(["G", "A"], np.log(0.6)),
            ],
            [
                MockScoredSequence(["G", "A"], np.log(0.7)),
                MockScoredSequence(["A", "G"], np.log(0.5)),
            ],
            [MockScoredSequence(["S", "P"], np.log(0.6))],  # One sequence in prediction
            [MockScoredSequence(["V"], np.log(0.9))],  # One sequence in prediction
            None,  # No predictions for this entry
        ]

    @pytest.fixture()
    def calibration_dataset(self, sample_metadata, sample_predictions):
        """Create a CalibrationDataset instance for testing."""
        return CalibrationDataset(
            metadata=sample_metadata, predictions=sample_predictions
        )

    def test_initialization(self, sample_metadata, sample_predictions):
        """Test CalibrationDataset initialization."""
        dataset = CalibrationDataset(
            metadata=sample_metadata, predictions=sample_predictions
        )

        assert isinstance(dataset.metadata, pd.DataFrame)
        assert isinstance(dataset.predictions, list)
        assert len(dataset.metadata) == 5
        assert len(dataset.predictions) == 5

    def test_confidence_column_property(self, calibration_dataset):
        """Test confidence_column property."""
        assert calibration_dataset.confidence_column == "confidence"

    def test_length(self, calibration_dataset):
        """Test __len__ method."""
        assert len(calibration_dataset) == 5

    def test_getitem(self, calibration_dataset):
        """Test __getitem__ method."""
        metadata_row, prediction = calibration_dataset[0]

        assert isinstance(metadata_row, pd.Series)
        assert metadata_row["confidence"] == 0.9
        assert len(prediction) == 2  # Two sequences in first prediction

    def test_getitem_with_none_prediction(self, calibration_dataset):
        """Test __getitem__ with None prediction."""
        metadata_row, prediction = calibration_dataset[
            4
        ]  # Last item has None prediction

        assert isinstance(metadata_row, pd.Series)
        assert prediction is None

    def test_getitem_out_of_bounds(self, calibration_dataset):
        """Test __getitem__ with invalid index."""
        with pytest.raises(IndexError):
            _ = calibration_dataset[10]

    def test_save_basic(self, calibration_dataset, tmp_path):
        """Test basic save functionality."""
        save_dir = tmp_path / "test_dataset"

        calibration_dataset.save(save_dir)

        # Check that directory was created
        assert save_dir.exists()
        assert save_dir.is_dir()

        # Check that metadata.csv was created
        metadata_file = save_dir / "metadata.csv"
        assert metadata_file.exists()

        # Check that predictions.pkl was created (since predictions exist)
        predictions_file = save_dir / "predictions.pkl"
        assert predictions_file.exists()

    def test_save_without_predictions(self, sample_metadata, tmp_path):
        """Test save functionality without predictions."""
        dataset = CalibrationDataset(metadata=sample_metadata, predictions=None)
        save_dir = tmp_path / "test_dataset_no_pred"

        dataset.save(save_dir)

        # Check that metadata.csv was created but not predictions.pkl
        metadata_file = save_dir / "metadata.csv"
        predictions_file = save_dir / "predictions.pkl"

        assert metadata_file.exists()
        assert not predictions_file.exists()

    def test_save_handles_sequence_strings(self, tmp_path):
        """Test that save converts sequence lists to strings."""
        metadata = pd.DataFrame(
            {
                "confidence": [0.9],
                "prediction": [["A", "G"]],
                "sequence": [["A", "G"]],
                "correct": [True],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=[None])
        save_dir = tmp_path / "test_string_conversion"

        dataset.save(save_dir)

        # Read back the saved CSV and check string conversion
        saved_metadata = pd.read_csv(save_dir / "metadata.csv")
        assert saved_metadata.iloc[0]["prediction"] == "AG"
        assert saved_metadata.iloc[0]["sequence"] == "AG"

    def test_filter_entries_basic(self, calibration_dataset):
        """Test basic filtering functionality."""
        # Filter out low confidence entries (keep high confidence)
        filtered = calibration_dataset.filter_entries(
            metadata_predicate=lambda row: row["confidence"] <= 0.7
        )

        assert (
            len(filtered) == 2
        )  # Should keep entries with confidence > 0.7 (0.9, 0.8)
        assert all(conf > 0.7 for conf in filtered.metadata["confidence"])

    def test_filter_entries_predictions(self, calibration_dataset):
        """Test filtering based on predictions."""
        # Filter out entries with predictions having less than 2 sequences and confidence <= 0.7
        filtered = calibration_dataset.filter_entries(
            predictions_predicate=lambda beam: (
                beam is None
                or len(beam) < 2
                or beam[0].sequence_log_probability <= np.log(0.7)
            )
        )

        assert len(filtered) == 1  # Should keep only first entry

    def test_filter_entries_combined(self, calibration_dataset):
        """Test filtering with both metadata and predictions predicates."""
        filtered = calibration_dataset.filter_entries(
            metadata_predicate=lambda row: row["confidence"] <= 0.7,
            predictions_predicate=lambda beam: beam is None,
        )

        # The method filters out entries that match EITHER predicate
        # Original confidences: [0.9, 0.8, 0.7, 0.6, 0.5]
        # Original predictions: [2seq, 2seq, 1seq, 1seq, None]
        # Filter out: indices 2,3,4 (conf <= 0.7) + index 4 (None prediction) = {2, 3, 4}
        # Keeps: indices 0, 1 (confidence 0.9, 0.8)
        assert len(filtered) == 2

    def test_filter_entries_empty_result(self, calibration_dataset):
        """Test filtering that results in empty dataset."""
        filtered = calibration_dataset.filter_entries(
            metadata_predicate=lambda row: row["confidence"]
            >= 0.0  # Filter out everything
        )

        assert len(filtered) == 0
        assert isinstance(filtered.metadata, pd.DataFrame)
        assert len(filtered.predictions) == 0

    def test_filter_entries_all_pass(self, calibration_dataset):
        """Test filtering where all entries pass (no filtering)."""
        filtered = (
            calibration_dataset.filter_entries()
        )  # Use default predicates (always False)

        assert len(filtered) == len(calibration_dataset)

    def test_filter_preserves_index_reset(self, calibration_dataset):
        """Test that filtering resets the DataFrame index."""
        # Filter out specific confidence values to get non-contiguous remaining indices
        filtered = calibration_dataset.filter_entries(
            metadata_predicate=lambda row: row["confidence"]
            in [0.8, 0.6]  # Remove indices 1, 3
        )

        # Check that the index was reset to be contiguous
        # Should keep indices 0, 2, 4 (confidence 0.9, 0.7, 0.5) which become 0, 1, 2
        assert list(filtered.metadata.index) == [0, 1, 2]

    def test_filter_entries_predictions_len_predicate_handles_none(
        self, calibration_dataset
    ):
        """Ensure len(beam) predicate handles None beams explicitly."""
        # Entries with None beam or <2 items are filtered out, leaving only those with >=2
        filtered = calibration_dataset.filter_entries(
            predictions_predicate=lambda beam: beam is None or len(beam) < 2
        )

        assert len(filtered) == 2

    def test_filter_entries_predictions_raises_error_on_none_beam(
        self, calibration_dataset
    ):
        """Test that accessing beam[0] on None beam raises helpful error."""
        with pytest.raises(ValueError, match="beam is None"):
            calibration_dataset.filter_entries(
                predictions_predicate=lambda beam: beam[0].sequence_log_probability
                <= np.log(0.7)
            )

    def test_filter_entries_predictions_raises_error_on_none_len(
        self, calibration_dataset
    ):
        """Test that calling len() on None beam raises helpful error."""
        with pytest.raises(ValueError, match="beam is None"):
            calibration_dataset.filter_entries(
                predictions_predicate=lambda beam: len(beam) < 2
            )

    def test_filter_entries_predictions_raises_error_on_empty_beam(
        self, calibration_dataset
    ):
        """Test that accessing beam[0] on empty beam raises helpful error."""
        # Create a dataset with an empty beam (empty list instead of None)
        empty_beam_dataset = CalibrationDataset(
            metadata=pd.DataFrame({"confidence": [0.5]}), predictions=[[]]
        )
        with pytest.raises(ValueError, match="beam is empty"):
            empty_beam_dataset.filter_entries(
                predictions_predicate=lambda beam: beam[0].sequence_log_probability
                <= np.log(0.7)
            )

    def test_filter_entries_predictions_raises_error_on_too_short_beam(
        self, calibration_dataset
    ):
        """Test that accessing beam[1] when beam has only 1 element raises helpful error."""
        # Create a dataset with a beam that has only 1 element
        short_beam_dataset = CalibrationDataset(
            metadata=pd.DataFrame({"confidence": [0.5]}),
            predictions=[[MockScoredSequence(["A"], np.log(0.8))]],
        )
        with pytest.raises(ValueError, match="too short"):
            short_beam_dataset.filter_entries(
                predictions_predicate=lambda beam: beam[1].sequence_log_probability
                <= np.log(0.7)
            )

    def test_filter_entries_handles_empty_dataset(self):
        """Test that filtering an empty dataset returns an empty dataset."""
        empty_dataset = CalibrationDataset(metadata=pd.DataFrame(), predictions=[])
        filtered = empty_dataset.filter_entries(lambda row: True)
        assert len(filtered) == 0

    def test_to_csv(self, calibration_dataset, tmp_path):
        """Test saving metadata to CSV."""
        csv_path = tmp_path / "test_metadata.csv"

        calibration_dataset.to_csv(csv_path)

        assert csv_path.exists()

        # Read back and verify content
        saved_df = pd.read_csv(csv_path)
        assert len(saved_df) == len(calibration_dataset.metadata)

    def test_to_parquet(self, calibration_dataset, tmp_path):
        """Test saving metadata to parquet."""
        parquet_path = tmp_path / "test_metadata.parquet"

        calibration_dataset.to_parquet(str(parquet_path))

        assert parquet_path.exists()

        # Read back and verify content
        saved_df = pd.read_parquet(parquet_path)
        assert len(saved_df) == len(calibration_dataset.metadata)

    def test_length_consistency(self, calibration_dataset):
        """Test that length is consistent between metadata and predictions."""
        # This should not raise an assertion error
        assert len(calibration_dataset) == len(calibration_dataset.metadata)
        assert len(calibration_dataset) == len(calibration_dataset.predictions)

    def test_length_inconsistency_raises_error(self, sample_metadata):
        """Test that length inconsistency raises assertion error."""
        # Create mismatched lengths
        short_predictions = [None, None]  # Only 2 predictions for 5 metadata rows

        with pytest.raises(
            AssertionError, match="Length of metadata and predictions must match"
        ):
            CalibrationDataset(metadata=sample_metadata, predictions=short_predictions)

    def test_empty_dataset(self):
        """Test operations on empty dataset."""
        empty_metadata = pd.DataFrame()
        empty_dataset = CalibrationDataset(metadata=empty_metadata, predictions=[])

        assert len(empty_dataset) == 0

        # Test filtering empty dataset
        filtered = empty_dataset.filter_entries(lambda row: True)
        assert len(filtered) == 0

    def test_dataset_with_complex_metadata(self):
        """Test dataset with various column types."""
        complex_metadata = pd.DataFrame(
            {
                "confidence": [0.9, 0.8],
                "prediction": [["A", "G"], ["G", "A"]],
                "numeric_col": [1.5, 2.7],
                "string_col": ["text1", "text2"],
                "bool_col": [True, False],
                "list_col": [[1, 2], [3, 4]],
            }
        )

        dataset = CalibrationDataset(
            metadata=complex_metadata, predictions=[None, None]
        )

        assert len(dataset) == 2

        # Test that all data types are preserved
        assert dataset.metadata["numeric_col"].dtype in [np.float64, float]
        assert dataset.metadata["string_col"].dtype == object
        assert dataset.metadata["bool_col"].dtype == bool

    def test_getitem_negative_index(self, calibration_dataset):
        """Test __getitem__ with negative index."""
        # Get last item
        metadata_row, prediction = calibration_dataset[-1]

        assert metadata_row["confidence"] == 0.0
        assert prediction is None

    def test_save_creates_parent_directories(self, calibration_dataset, tmp_path):
        """Test that save creates parent directories if they don't exist."""
        nested_dir = tmp_path / "level1" / "level2" / "dataset"

        calibration_dataset.save(nested_dir)

        assert nested_dir.exists()
        assert (nested_dir / "metadata.csv").exists()

    def test_metadata_modification_after_creation(self, calibration_dataset):
        """Test that modifying metadata after creation works correctly."""
        # Add a new column
        calibration_dataset.metadata["new_column"] = range(len(calibration_dataset))

        assert "new_column" in calibration_dataset.metadata.columns
        assert list(calibration_dataset.metadata["new_column"]) == [0, 1, 2, 3, 4]

    def test_predictions_modification_after_creation(self, calibration_dataset):
        """Test that modifying predictions after creation works correctly."""
        # Modify one prediction
        calibration_dataset.predictions[0] = [MockScoredSequence(["X"], 0.5)]

        _, modified_prediction = calibration_dataset[0]
        assert len(modified_prediction) == 1
        assert modified_prediction[0].sequence == ["X"]

    # ------------------------------------------------------------------
    # predictions=None (no beam predictions)
    # ------------------------------------------------------------------

    def test_predictions_none_initialization(self, sample_metadata):
        """Test CalibrationDataset initialization with predictions=None."""
        dataset = CalibrationDataset(metadata=sample_metadata, predictions=None)

        assert dataset.predictions is None
        assert len(dataset.metadata) == 5

    def test_predictions_none_default(self):
        """Test that predictions defaults to None."""
        metadata = pd.DataFrame({"confidence": [0.9]})
        dataset = CalibrationDataset(metadata=metadata)

        assert dataset.predictions is None

    def test_predictions_none_length(self, sample_metadata):
        """Test __len__ with predictions=None."""
        dataset = CalibrationDataset(metadata=sample_metadata, predictions=None)

        # Length should be based on metadata
        assert len(dataset) == 5

    def test_predictions_none_getitem(self, sample_metadata):
        """Test __getitem__ with predictions=None."""
        dataset = CalibrationDataset(metadata=sample_metadata, predictions=None)

        metadata_row, prediction = dataset[0]

        assert isinstance(metadata_row, pd.Series)
        assert prediction is None

    def test_predictions_none_filter_entries_metadata_only(self, sample_metadata):
        """Test filtering with predictions=None using only metadata predicate."""
        dataset = CalibrationDataset(metadata=sample_metadata, predictions=None)

        filtered = dataset.filter_entries(
            metadata_predicate=lambda row: row["confidence"] <= 0.7
        )

        assert filtered.predictions is None
        assert len(filtered) == 2

    def test_predictions_none_filter_entries_raises_for_predictions_predicate(
        self, sample_metadata
    ):
        """Test that using predictions_predicate with predictions=None raises error."""
        dataset = CalibrationDataset(metadata=sample_metadata, predictions=None)

        with pytest.raises(
            ValueError,
            match="Cannot use predictions_predicate when predictions is None",
        ):
            dataset.filter_entries(
                predictions_predicate=lambda beam: beam is None or len(beam) < 2
            )

    def test_predictions_none_save(self, sample_metadata, tmp_path):
        """Test save functionality with predictions=None."""
        dataset = CalibrationDataset(metadata=sample_metadata, predictions=None)
        save_dir = tmp_path / "test_dataset_none_pred"

        dataset.save(save_dir)

        # Check that metadata.csv was created but not predictions.pkl
        metadata_file = save_dir / "metadata.csv"
        predictions_file = save_dir / "predictions.pkl"

        assert metadata_file.exists()
        assert not predictions_file.exists()
