"""Unit tests for winnow DatabaseGroundedFDRControl."""

import pytest
import pandas as pd
from winnow.fdr.database_grounded import DatabaseGroundedFDRControl


class TestDatabaseGroundedFDRControl:
    """Test the DatabaseGroundedFDRControl class."""

    @pytest.fixture()
    def db_fdr_control(self):
        """Create a DatabaseGroundedFDRControl instance for testing."""
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
        }
        return DatabaseGroundedFDRControl(
            confidence_feature="confidence", residue_masses=residue_masses
        )

    @pytest.fixture()
    def sample_dataset_df(self):
        """Create a sample dataset DataFrame for testing."""
        return pd.DataFrame(
            {
                "sequence": ["PEPTIDE", "PROTEIN", "SAMPLE"],
                "prediction": ["PEPTIDE", "PROTIEN", "SAMPL"],  # One mismatch
                "confidence": [0.9, 0.8, 0.7],
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
        # Convert sequences to list format as expected by the implementation
        sample_dataset_df = sample_dataset_df.copy()
        sample_dataset_df["prediction"] = sample_dataset_df["prediction"].apply(list)

        # Should not raise an exception
        db_fdr_control.fit(sample_dataset_df)

        # Check that fit created the required attributes
        assert hasattr(db_fdr_control, "preds")
        assert hasattr(db_fdr_control, "_fdr_values")
        assert hasattr(db_fdr_control, "_confidence_scores")
        assert db_fdr_control._fdr_values is not None
        assert db_fdr_control._confidence_scores is not None

    def test_fit_with_parameters(self, db_fdr_control):
        """Test fit with custom parameters."""
        sample_df = pd.DataFrame(
            {"sequence": ["TEST"], "prediction": [list("TEST")], "confidence": [0.9]}
        )

        db_fdr_control.fit(sample_df)

        # Check that fit created the required attributes
        assert hasattr(db_fdr_control, "preds")
        assert len(db_fdr_control.preds) == 1
        assert db_fdr_control.preds.iloc[0]["confidence"] == 0.9

    def test_fit_with_empty_data(self, db_fdr_control):
        """Test that fit method handles empty data."""
        empty_data = pd.DataFrame()
        with pytest.raises(AssertionError, match="Fit method requires non-empty data"):
            db_fdr_control.fit(empty_data)

    def test_get_confidence_cutoff_requires_fitting(self, db_fdr_control):
        """Test that get_confidence_cutoff requires fitting first."""
        # Should raise AttributeError if not fitted
        with pytest.raises(
            AttributeError, match=r"FDR method not fitted, please call `fit\(\)` first"
        ):
            db_fdr_control.get_confidence_cutoff(0.05)

    def test_compute_fdr_requires_fitting(self, db_fdr_control):
        """Test that compute_fdr requires fitting first."""
        # Should raise AttributeError if not fitted
        with pytest.raises(
            AttributeError, match=r"FDR method not fitted, please call `fit\(\)` first"
        ):
            db_fdr_control.compute_fdr(0.8)
