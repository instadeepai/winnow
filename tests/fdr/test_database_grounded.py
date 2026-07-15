"""Unit tests for winnow DatabaseGroundedFDRControl."""

import pytest
import pandas as pd
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.fdr.database_grounded import DatabaseGroundedFDRControl


def _as_dataset(df: pd.DataFrame) -> CalibrationDataset:
    return CalibrationDataset(metadata=df)


class TestDatabaseGroundedFDRControl:
    """Test the DatabaseGroundedFDRControl class."""

    @pytest.fixture()
    def db_fdr_control(self):
        """Create a DatabaseGroundedFDRControl instance for testing."""
        return DatabaseGroundedFDRControl(confidence_feature="confidence", drop=0)

    @pytest.fixture()
    def sample_dataset_df(self):
        """Finalised sample dataset DataFrame for testing."""
        return pd.DataFrame(
            {
                "sequence": [list("PEPTIDE"), list("PROTEIN"), list("SAMPLE")],
                "prediction": [list("PEPTIDE"), list("PROTIEN"), list("SAMPL")],
                "confidence": [0.9, 0.8, 0.7],
                "correct": [True, False, False],
                "valid_sequence": [True, True, True],
                "precursor_mz": [500.504, 400.672, 400.504],
                "precursor_charge": [2, 3, 2],
            }
        )

    def test_initialization(self, db_fdr_control):
        """Test DatabaseGroundedFDRControl initialization."""
        assert db_fdr_control.confidence_feature == "confidence"
        assert db_fdr_control._fdr_values is None
        assert db_fdr_control._confidence_scores is None

    def test_fit_basic(self, db_fdr_control, sample_dataset_df):
        """Test basic fitting functionality."""
        db_fdr_control.fit(_as_dataset(sample_dataset_df))

        assert hasattr(db_fdr_control, "preds")
        assert hasattr(db_fdr_control, "_fdr_values")
        assert hasattr(db_fdr_control, "_confidence_scores")
        assert db_fdr_control._fdr_values is not None
        assert db_fdr_control._confidence_scores is not None

    def test_fit_with_parameters(self, db_fdr_control):
        """Test fit with custom parameters."""
        sample_df = pd.DataFrame(
            {
                "sequence": [list("TEST")],
                "prediction": [list("TEST")],
                "confidence": [0.9],
                "correct": [True],
                "valid_sequence": [True],
            }
        )

        db_fdr_control.fit(_as_dataset(sample_df))

        assert hasattr(db_fdr_control, "preds")
        assert len(db_fdr_control.preds) == 1
        assert db_fdr_control.preds.iloc[0]["confidence"] == 0.9

    def test_fit_with_empty_data(self, db_fdr_control):
        """Test that fit method handles empty data."""
        empty_data = _as_dataset(
            pd.DataFrame(
                {
                    "prediction": pd.Series([], dtype=object),
                    "sequence": pd.Series([], dtype=object),
                    "correct": pd.Series([], dtype=bool),
                    "valid_sequence": pd.Series([], dtype=bool),
                    "confidence": pd.Series([], dtype=float),
                }
            )
        )
        with pytest.raises(AssertionError, match="Fit method requires non-empty data"):
            db_fdr_control.fit(empty_data)

    def test_fit_requires_finalised_metadata(self, db_fdr_control):
        """Fit refuses metadata missing loader-derived correctness labels."""
        dataset = _as_dataset(
            pd.DataFrame(
                {
                    "sequence": [list("PEPTIDE")],
                    "prediction": [list("PEPTIDE")],
                    "confidence": [0.9],
                }
            )
        )
        with pytest.raises(ValueError, match="finalised labelled metadata"):
            db_fdr_control.fit(dataset)

    def test_fit_excludes_invalid_sequence_rows_from_fdr_curve(self):
        """Rows with valid_sequence=False must not enter the precision curve."""
        ctrl = DatabaseGroundedFDRControl(confidence_feature="confidence", drop=0)
        dataset = _as_dataset(
            pd.DataFrame(
                {
                    "sequence": [list("PEPTIDE"), None],
                    "prediction": [list("PEPTIDE"), list("WRONG")],
                    "confidence": [0.9, 0.99],
                    "correct": [True, False],
                    "valid_sequence": [True, False],
                }
            )
        )
        ctrl.fit(dataset)
        assert ctrl._fdr_values.tolist() == [0.0]
        assert ctrl._confidence_scores.tolist() == [0.9]

    def test_fit_raises_when_no_valid_ground_truth_sequences(self, db_fdr_control):
        dataset = _as_dataset(
            pd.DataFrame(
                {
                    "sequence": [None, None],
                    "prediction": [list("AG"), list("AG")],
                    "confidence": [0.9, 0.8],
                    "correct": [False, False],
                    "valid_sequence": [False, False],
                }
            )
        )
        with pytest.raises(ValueError, match="valid_sequence=True"):
            db_fdr_control.fit(dataset)

    def test_get_confidence_cutoff_requires_fitting(self, db_fdr_control):
        """Test that get_confidence_cutoff requires fitting first."""
        with pytest.raises(
            AttributeError, match=r"FDR method not fitted, please call `fit\(\)` first"
        ):
            db_fdr_control.get_confidence_cutoff(0.05)

    def test_compute_fdr_requires_fitting(self, db_fdr_control):
        """Test that compute_fdr requires fitting first."""
        with pytest.raises(
            AttributeError, match=r"FDR method not fitted, please call `fit\(\)` first"
        ):
            db_fdr_control.compute_fdr(0.8)
