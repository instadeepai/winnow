"""Unit tests for winnow DatabaseGroundedFDRControl."""

import pytest
import numpy as np
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
        sample_df = _as_dataset(
            pd.DataFrame(
                {
                    "sequence": [list("TEST")],
                    "prediction": [list("TEST")],
                    "confidence": [0.9],
                    "correct": [True],
                    "valid_sequence": [True],
                }
            )
        )

        db_fdr_control.fit(sample_df)

        assert hasattr(db_fdr_control, "preds")
        assert len(db_fdr_control.preds) == 1
        assert db_fdr_control.preds.iloc[0]["confidence"] == 0.9

    def test_fit_uses_custom_correct_column(self, db_fdr_control):
        """Test that fit() proceeds with a custom correctness column, possibly without a sequence column (i.e. proteome hit proxy)."""
        sample_df = _as_dataset(
            pd.DataFrame(
                {
                    "prediction": [list("PEPTIDE"), list("PROTIEN")],
                    "confidence": [0.9, 0.8],
                    "proteome_hit": [True, False],
                }
            )
        )

        db_fdr_control.fit(sample_df, correct_column="proteome_hit")

        assert list(db_fdr_control.preds.columns) == ["proteome_hit", "confidence"]
        assert db_fdr_control.preds["proteome_hit"].tolist() == [True, False]

    def test_fit_rejects_null_proxy_correctness_rows(self, db_fdr_control):
        """Null proxy correctness verdicts are rejected."""
        sample_df = _as_dataset(
            pd.DataFrame(
                {
                    "prediction": [list("PEPTIDE"), list("PROTIEN"), list("SAMPLE")],
                    "confidence": [0.9, 0.95, 0.8],
                    "proteome_hit": [True, None, False],
                }
            )
        )

        with pytest.raises(ValueError, match="contains missing values"):
            db_fdr_control.fit(sample_df, correct_column="proteome_hit")

    def test_fit_proxy_ignores_valid_sequence_when_sequence_present(
        self, db_fdr_control
    ):
        """With a proxy column, sequence validity is ignored even if sequence exists."""
        sample_df = _as_dataset(
            pd.DataFrame(
                {
                    "sequence": [list("PEPTIDE"), None, list("SAMPLE")],
                    "prediction": [list("PEPTIDE"), list("WRONG"), list("SAMPL")],
                    "confidence": [0.9, 0.95, 0.8],
                    "correct": [True, False, False],
                    "valid_sequence": [True, False, True],
                    "proteome_hit": [True, False, True],
                }
            )
        )

        db_fdr_control.fit(sample_df, correct_column="proteome_hit")

        # valid_sequence=False row (conf 0.95) is kept under the proxy path.
        assert len(db_fdr_control.preds) == 3
        assert db_fdr_control.preds["proteome_hit"].tolist() == [False, True, True]
        assert db_fdr_control.preds["confidence"].tolist() == [0.95, 0.9, 0.8]
        np.testing.assert_allclose(db_fdr_control._fdr_values, [1.0, 0.5, 1.0 / 3.0])
        assert db_fdr_control._confidence_scores.tolist() == [0.95, 0.9, 0.8]

    def test_fit_rejects_non_bool_proxy_labels(self, db_fdr_control):
        """String proxy labels must not be silently truthy-coerced."""
        sample_df = _as_dataset(
            pd.DataFrame(
                {
                    "prediction": [list("PEPTIDE"), list("PROTIEN")],
                    "confidence": [0.9, 0.8],
                    "proteome_hit": ["true", "false"],
                }
            )
        )
        with pytest.raises(ValueError, match="must be a boolean or numeric series"):
            db_fdr_control.fit(sample_df, correct_column="proteome_hit")

    def test_fit_requires_custom_correct_column(self, db_fdr_control):
        """Missing proxy column uses a proxy-specific error message."""
        sample_df = _as_dataset(
            pd.DataFrame(
                {
                    "prediction": [list("PEPTIDE")],
                    "confidence": [0.9],
                }
            )
        )
        with pytest.raises(ValueError, match="requires metadata columns.*proteome_hit"):
            db_fdr_control.fit(sample_df, correct_column="proteome_hit")

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
        with pytest.raises(ValueError, match="Fit method requires non-empty data"):
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

    def test_fit_excludes_invalid_sequence_without_sequence_column(self):
        """valid_sequence is honoured even when no sequence column is present."""
        ctrl = DatabaseGroundedFDRControl(confidence_feature="confidence", drop=0)
        # Hand-built metadata: correct + valid_sequence without sequence.
        dataset = CalibrationDataset(
            metadata=pd.DataFrame(
                {
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
        assert len(ctrl.preds) == 1

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
