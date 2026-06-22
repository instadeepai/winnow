"""Tests for WinnowDatasetLoader."""

import pickle
from pathlib import Path

import pandas as pd
import pytest

from winnow.datasets.data_loaders import WinnowDatasetLoader


class TestWinnowDatasetLoader:
    """Tests for WinnowDatasetLoader."""

    @pytest.fixture()
    def loader(self, full_residue_masses):
        return WinnowDatasetLoader(
            residue_masses=full_residue_masses,
            residue_remapping={},
        )

    @pytest.fixture()
    def metadata_dir(self, tmp_path):
        """Minimal metadata.csv directory (no sequence column, no pkl)."""
        df = pd.DataFrame(
            {
                "prediction": ["AG", "MG"],
                "confidence": [0.9, 0.8],
                "mz_array": ["[100.0, 200.0, 300.0]", "[150.0, 250.0, 350.0]"],
                "intensity_array": [
                    "[1000.0, 2000.0, 3000.0]",
                    "[1500.0, 2500.0, 3500.0]",
                ],
            }
        )
        df.to_csv(tmp_path / "metadata.csv", index=False)
        return tmp_path

    @pytest.fixture()
    def metadata_dir_with_sequence(self, tmp_path):
        """metadata.csv directory that includes a ground-truth sequence column."""
        df = pd.DataFrame(
            {
                "sequence": ["AG", "MG"],
                "prediction": ["AG", "MG"],
                "confidence": [0.9, 0.8],
                "mz_array": ["[100.0, 200.0, 300.0]", "[150.0, 250.0, 350.0]"],
                "intensity_array": [
                    "[1000.0, 2000.0, 3000.0]",
                    "[1500.0, 2500.0, 3500.0]",
                ],
            }
        )
        df.to_csv(tmp_path / "metadata.csv", index=False)
        return tmp_path

    @pytest.fixture()
    def metadata_dir_numpy_arrays(self, tmp_path):
        """metadata.csv with mz_array / intensity_array in numpy print format (no commas)."""
        df = pd.DataFrame(
            {
                "prediction": ["AG"],
                "confidence": [0.9],
                "mz_array": ["[100.   200.   300. ]"],
                "intensity_array": ["[1000.   2000.   3000. ]"],
            }
        )
        df.to_csv(tmp_path / "metadata.csv", index=False)
        return tmp_path

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_raises_when_predictions_path_provided(self, loader, tmp_path):
        """WinnowDatasetLoader does not accept predictions_path."""
        with pytest.raises(ValueError, match="predictions_path is not used"):
            loader.load(data_path=tmp_path, predictions_path=Path("something.pkl"))

    def test_raises_when_metadata_csv_missing(self, loader, tmp_path):
        """FileNotFoundError when the directory has no metadata.csv."""
        with pytest.raises(FileNotFoundError, match="metadata.csv"):
            loader.load(data_path=tmp_path)

    # ------------------------------------------------------------------
    # Successful load – no predictions.pkl
    # ------------------------------------------------------------------

    def test_prediction_column_is_tokenised(self, loader, metadata_dir):
        """prediction column should be converted from ProForma string to token list."""
        dataset = loader.load(data_path=metadata_dir)
        assert isinstance(dataset.metadata["prediction"].iloc[0], list)
        assert dataset.metadata["prediction"].iloc[0] == ["A", "G"]

    def test_sequence_column_is_tokenised_when_present(
        self, loader, metadata_dir_with_sequence
    ):
        """sequence column should be tokenised when it exists in the CSV."""
        dataset = loader.load(data_path=metadata_dir_with_sequence)
        assert isinstance(dataset.metadata["sequence"].iloc[0], list)
        assert dataset.metadata["sequence"].iloc[0] == ["A", "G"]

    def test_mz_array_parsed_comma_format(self, loader, metadata_dir):
        """mz_array written with commas should be parsed to a Python list."""
        dataset = loader.load(data_path=metadata_dir)
        mz = dataset.metadata["mz_array"].iloc[0]
        assert isinstance(mz, list)
        assert mz == pytest.approx([100.0, 200.0, 300.0])

    def test_intensity_array_parsed_comma_format(self, loader, metadata_dir):
        """intensity_array written with commas should be parsed to a Python list."""
        dataset = loader.load(data_path=metadata_dir)
        intensity = dataset.metadata["intensity_array"].iloc[0]
        assert isinstance(intensity, list)
        assert intensity == pytest.approx([1000.0, 2000.0, 3000.0])

    def test_mz_array_parsed_numpy_format(self, loader, metadata_dir_numpy_arrays):
        """mz_array in numpy print format (spaces, no commas) should be parsed."""
        dataset = loader.load(data_path=metadata_dir_numpy_arrays)
        mz = dataset.metadata["mz_array"].iloc[0]
        assert isinstance(mz, list)
        assert mz == pytest.approx([100.0, 200.0, 300.0])

    # ------------------------------------------------------------------
    # Successful load – with predictions.pkl
    # ------------------------------------------------------------------

    def test_loads_predictions_pkl_when_present(self, loader, metadata_dir):
        """predictions.pkl should be loaded and returned as the predictions attribute."""
        # Build a minimal predictions list and pickle it
        fake_predictions = [[None], [None]]
        with (metadata_dir / "predictions.pkl").open("wb") as f:
            pickle.dump(fake_predictions, f)

        dataset = loader.load(data_path=metadata_dir)
        assert dataset.predictions == fake_predictions

    def test_predictions_length_matches_metadata(self, loader, metadata_dir):
        """len(predictions) must equal len(metadata) when pkl is present."""
        rows = 2  # matches metadata_dir fixture
        fake_predictions = [None] * rows
        with (metadata_dir / "predictions.pkl").open("wb") as f:
            pickle.dump(fake_predictions, f)

        dataset = loader.load(data_path=metadata_dir)
        assert len(dataset.predictions) == len(dataset.metadata)
