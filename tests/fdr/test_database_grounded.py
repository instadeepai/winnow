"""Unit tests for winnow DatabaseGroundedFDRControl."""

import pytest
from unittest.mock import patch, Mock
import pandas as pd
from winnow.fdr.database_grounded import DatabaseGroundedFDRControl


class TestDatabaseGroundedFDRControl:
    """Test the DatabaseGroundedFDRControl class."""

    @pytest.fixture()
    def db_fdr_control(self):
        """Create a DatabaseGroundedFDRControl instance for testing."""
        return DatabaseGroundedFDRControl(confidence_feature="confidence")

    @pytest.fixture()
    def sample_dataset_df(self):
        """Create a sample dataset DataFrame for testing."""
        return pd.DataFrame(
            {
                "sequence": ["PEPTIDE", "PROTEIN", "SAMPLE"],
                "prediction": ["PEPTIDE", "PROTIEN", "SAMPL"],  # One mismatch
                "confidence": [0.9, 0.8, 0.7],
                "precursor_mass": [1000.0, 1200.0, 800.0],
            }
        )

    def test_initialization(self, db_fdr_control):
        """Test DatabaseGroundedFDRControl initialization."""
        assert db_fdr_control.confidence_feature == "confidence"
        assert db_fdr_control._fdr_values is None
        assert db_fdr_control._confidence_scores is None

    @patch("winnow.fdr.database_grounded.Metrics")
    def test_fit_basic(self, mock_metrics, db_fdr_control, sample_dataset_df):
        """Test basic fitting functionality."""
        # Mock the Metrics class and its methods
        mock_metrics_instance = Mock()
        mock_metrics.return_value = mock_metrics_instance
        mock_metrics_instance._split_peptide = lambda x: list(x)

        residue_masses = {
            "P": 100.0,
            "E": 110.0,
            "T": 120.0,
            "I": 130.0,
            "D": 140.0,
            "R": 150.0,
            "O": 160.0,
            "N": 170.0,
            "S": 180.0,
            "A": 190.0,
            "M": 200.0,
            "L": 210.0,
        }

        # Should not raise an exception
        db_fdr_control.fit(sample_dataset_df, residue_masses)

        # Check that metrics was called
        mock_metrics.assert_called_once()

    def test_fit_with_parameters(self, db_fdr_control):
        """Test fit with custom parameters."""
        sample_df = pd.DataFrame(
            {"sequence": ["TEST"], "prediction": ["TEST"], "confidence": [0.9]}
        )
        residue_masses = {"T": 100.0, "E": 110.0, "S": 120.0}

        with patch("winnow.fdr.database_grounded.Metrics") as mock_metrics:
            mock_metrics_instance = Mock()
            mock_metrics.return_value = mock_metrics_instance
            mock_metrics_instance._split_peptide = lambda x: list(x)

            db_fdr_control.fit(
                sample_df, residue_masses, isotope_error_range=(0, 2), drop=5
            )

            # Check that Metrics was initialized with correct parameters
            mock_metrics.assert_called_once()

    def test_fit_with_empty_data(self, db_fdr_control):
        """Test that fit method handles empty data."""
        empty_data = pd.DataFrame()
        with pytest.raises(AssertionError, match="Fit method requires non-empty data"):
            db_fdr_control.fit(empty_data, residue_masses={"A": 71.03})

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
