"""Unit tests for winnow FDR control base class."""

import numpy as np
import pandas as pd
import pytest
from winnow.fdr.base import FDRControl
from winnow.datasets.psm_dataset import PSMDataset


class ConcreteFDRControl(FDRControl):
    """Concrete implementation of FDRControl for testing."""

    def __init__(self):
        self.confidence_thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
        self.fdr_values = [0.01, 0.02, 0.05, 0.1, 0.2]

    def fit(self, dataset):
        pass

    def get_confidence_cutoff(self, threshold: float) -> float:
        # Simple linear interpolation for testing
        if threshold <= 0.01:
            return 0.9
        elif threshold <= 0.02:
            return 0.8
        elif threshold <= 0.05:
            return 0.7
        elif threshold <= 0.1:
            return 0.6
        else:
            return 0.5

    def compute_fdr(self, score: float) -> float:
        # Simple step function for testing
        if score >= 0.9:
            return 0.01
        elif score >= 0.8:
            return 0.02
        elif score >= 0.7:
            return 0.05
        elif score >= 0.6:
            return 0.1
        else:
            return 0.2


class TestFDRControlBase:
    """Test the base FDRControl class."""

    @pytest.fixture()
    def fdr_control(self):
        """Create a concrete FDRControl instance for testing."""
        return ConcreteFDRControl()

    @pytest.fixture()
    def sample_psm_dataset(self):
        """Create a sample PSMDataset for testing."""
        spectra = [np.array([[100.0, 1000.0]]) for _ in range(5)]
        peptides = [np.array([i]) for i in range(5)]
        confidence_scores = [0.95, 0.85, 0.75, 0.65, 0.45]

        return PSMDataset.from_dataset(spectra, peptides, confidence_scores)

    @pytest.fixture()
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {"confidence": [0.95, 0.85, 0.75, 0.65, 0.45], "other_col": [1, 2, 3, 4, 5]}
        )

    def test_cannot_instantiate_abstract_class(self):
        """Test that FDRControl cannot be instantiated directly."""
        with pytest.raises(TypeError):
            FDRControl()

    def test_filter_entries(self, fdr_control, sample_psm_dataset):
        """Test filtering PSMs based on confidence threshold."""
        # Filter at 5% FDR (should keep confidence >= 0.7)
        filtered_dataset = fdr_control.filter_entries(sample_psm_dataset, 0.05)

        assert len(filtered_dataset) == 3  # Only first 3 PSMs have confidence >= 0.7
        for psm in filtered_dataset:
            assert psm.confidence >= 0.7

    def test_add_psm_fdr(self, fdr_control, sample_dataframe):
        """Test adding PSM-specific FDR values to DataFrame."""
        result_df = fdr_control.add_psm_fdr(sample_dataframe, "confidence")

        # Check that psm_fdr column was added
        assert "psm_fdr" in result_df.columns
        assert len(result_df) == len(sample_dataframe)

        # Check that original DataFrame is not modified
        assert "psm_fdr" not in sample_dataframe.columns

        # Check expected FDR values based on our concrete implementation
        expected_fdrs = [0.01, 0.02, 0.05, 0.1, 0.2]
        assert list(result_df["psm_fdr"]) == expected_fdrs

    def test_add_psm_qvalue(self, fdr_control, sample_dataframe):
        """Test adding PSM-specific q-values to DataFrame."""
        result_df = fdr_control.add_psm_qvalue(sample_dataframe, "confidence")

        # Check that psm_qvalue column was added
        assert "psm_qvalue" in result_df.columns
        assert len(result_df) == len(sample_dataframe)

        # Check expected q-values based on our concrete implementation
        expected_qvalues = [0.01, 0.02, 0.05, 0.1, 0.2]
        assert list(result_df["psm_qvalue"]) == expected_qvalues

    def test_get_confidence_curve(self, fdr_control):
        """Test getting confidence curve for FDR thresholds."""
        fdr_thresholds, confidence_scores = fdr_control.get_confidence_curve(
            resolution=0.01, min_confidence=0.01, max_confidence=0.11
        )

        assert len(fdr_thresholds) == len(confidence_scores)
        assert len(fdr_thresholds) == 10  # (0.11 - 0.01) / 0.01

        # Check that confidence decreases as FDR threshold increases
        for i in range(1, len(confidence_scores)):
            assert confidence_scores[i] <= confidence_scores[i - 1]
