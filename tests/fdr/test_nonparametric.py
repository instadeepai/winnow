"""Unit tests for winnow NonParametricFDRControl."""

import pytest
import numpy as np
import pandas as pd
from winnow.fdr.nonparametric import NonParametricFDRControl


class TestNonParametricFDRControl:
    """Test the NonParametricFDRControl class."""

    @pytest.fixture()
    def nonparametric_fdr_control(self):
        """Create a NonParametricFDRControl instance for testing."""
        return NonParametricFDRControl()

    @pytest.fixture()
    def sample_confidence_data(self):
        """Create sample confidence data for testing."""
        return pd.DataFrame(
            {"confidence": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]}
        )

    @pytest.fixture()
    def fitted_fdr_control(self, nonparametric_fdr_control, sample_confidence_data):
        """Create a fitted NonParametricFDRControl instance for testing."""
        nonparametric_fdr_control.fit(sample_confidence_data)
        return nonparametric_fdr_control

    def test_initialization(self, nonparametric_fdr_control):
        """Test NonParametricFDRControl initialization."""
        assert nonparametric_fdr_control._confidence_scores is None
        assert nonparametric_fdr_control._sorted_indices is None
        assert nonparametric_fdr_control._is_correct is None
        assert nonparametric_fdr_control._null_scores is None

    def test_fit_basic(self, nonparametric_fdr_control, sample_confidence_data):
        """Test basic fitting functionality."""
        nonparametric_fdr_control.fit(sample_confidence_data)

        # Check that the model has been fitted
        assert nonparametric_fdr_control._confidence_scores is not None
        assert nonparametric_fdr_control._sorted_indices is not None
        assert nonparametric_fdr_control._fdr_values is not None

        # Check that scores are sorted in descending order
        assert np.all(
            nonparametric_fdr_control._confidence_scores[:-1]
            >= nonparametric_fdr_control._confidence_scores[1:]
        )

    def test_fit_with_single_value(self, nonparametric_fdr_control):
        """Test fitting with a single confidence value."""
        single_data = pd.DataFrame({"confidence": [0.8]})
        nonparametric_fdr_control.fit(single_data)

        assert len(nonparametric_fdr_control._confidence_scores) == 1
        assert nonparametric_fdr_control._confidence_scores[0] == 0.8

    def test_fit_with_duplicate_values(self, nonparametric_fdr_control):
        """Test fitting with duplicate confidence values."""
        duplicate_data = pd.DataFrame({"confidence": [0.9, 0.8, 0.8, 0.7, 0.7, 0.6]})
        nonparametric_fdr_control.fit(duplicate_data)

        assert len(nonparametric_fdr_control._confidence_scores) == 6

    def test_fit_with_empty_data(self, nonparametric_fdr_control):
        """Test that fit method handles empty data."""
        empty_data = pd.DataFrame({"confidence": []})
        with pytest.raises(AssertionError, match="Fit method requires non-empty data"):
            nonparametric_fdr_control.fit(empty_data)

    def test_get_confidence_cutoff_requires_fitting(self, nonparametric_fdr_control):
        """Test that get_confidence_cutoff requires fitting first."""
        with pytest.raises(
            AttributeError, match=r"FDR method not fitted, please call `fit\(\)` first"
        ):
            nonparametric_fdr_control.get_confidence_cutoff(0.05)

    def test_compute_fdr_requires_fitting(self, nonparametric_fdr_control):
        """Test that compute_fdr requires fitting first."""
        with pytest.raises(
            AttributeError, match=r"FDR method not fitted, please call `fit\(\)` first"
        ):
            nonparametric_fdr_control.compute_fdr(0.8)
