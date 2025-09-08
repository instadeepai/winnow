"""Unit tests for winnow PSMDataset and PeptideSpectrumMatch."""

import numpy as np
import pytest
from winnow.datasets.psm_dataset import PeptideSpectrumMatch, PSMDataset


class TestPeptideSpectrumMatch:
    """Test the PeptideSpectrumMatch class."""

    def test_creation(self):
        """Test basic creation of PeptideSpectrumMatch."""
        spectrum = np.array([[100.0, 1000.0], [200.0, 2000.0]])
        peptide = np.array([1, 2, 3, 4])
        confidence = 0.95

        psm = PeptideSpectrumMatch(
            spectrum=spectrum, peptide=peptide, confidence=confidence
        )

        assert np.array_equal(psm.spectrum, spectrum)
        assert np.array_equal(psm.peptide, peptide)
        assert psm.confidence == confidence

    def test_confidence_comparison_greater(self):
        """Test that PSM with higher confidence compares correctly."""
        spectrum1 = np.array([[100.0, 1000.0]])
        peptide1 = np.array([1, 2])

        psm_high = PeptideSpectrumMatch(spectrum1, peptide1, 0.9)
        psm_low = PeptideSpectrumMatch(spectrum1, peptide1, 0.7)

        assert psm_high >= psm_low
        assert not (psm_low >= psm_high)

    def test_confidence_comparison_equal(self):
        """Test that PSMs with equal confidence compare correctly."""
        spectrum1 = np.array([[100.0, 1000.0]])
        peptide1 = np.array([1, 2])

        psm1 = PeptideSpectrumMatch(spectrum1, peptide1, 0.8)
        psm2 = PeptideSpectrumMatch(spectrum1, peptide1, 0.8)

        assert psm1 >= psm2
        assert psm2 >= psm1


class TestPSMDataset:
    """Test the PSMDataset class."""

    @pytest.fixture()
    def sample_psms(self):
        """Create sample PSMs for testing."""
        spectrum1 = np.array([[100.0, 1000.0], [200.0, 2000.0]])
        spectrum2 = np.array([[150.0, 1500.0], [250.0, 2500.0]])
        peptide1 = np.array([1, 2, 3])
        peptide2 = np.array([4, 5, 6])

        psm1 = PeptideSpectrumMatch(spectrum1, peptide1, 0.9)
        psm2 = PeptideSpectrumMatch(spectrum2, peptide2, 0.8)

        return [psm1, psm2]

    def test_creation_from_list(self, sample_psms):
        """Test PSMDataset creation from list of PSMs."""
        dataset = PSMDataset(peptide_spectrum_matches=sample_psms)

        assert len(dataset) == 2
        assert dataset.peptide_spectrum_matches == sample_psms

    def test_from_dataset_classmethod(self):
        """Test PSMDataset creation using from_dataset class method."""
        spectra = [np.array([[100.0, 1000.0]]), np.array([[200.0, 2000.0]])]
        peptides = [np.array([1, 2]), np.array([3, 4])]
        confidence_scores = [0.9, 0.7]

        dataset = PSMDataset.from_dataset(spectra, peptides, confidence_scores)

        assert len(dataset) == 2
        assert dataset[0].confidence == 0.9
        assert dataset[1].confidence == 0.7
        assert np.array_equal(dataset[0].peptide, np.array([1, 2]))
        assert np.array_equal(dataset[1].peptide, np.array([3, 4]))

    def test_indexing(self, sample_psms):
        """Test PSMDataset indexing."""
        dataset = PSMDataset(peptide_spectrum_matches=sample_psms)

        # Test valid indexing
        assert dataset[0] == sample_psms[0]
        assert dataset[1] == sample_psms[1]

        # Test negative indexing
        assert dataset[-1] == sample_psms[-1]

    def test_indexing_out_of_bounds(self, sample_psms):
        """Test PSMDataset indexing with invalid indices."""
        dataset = PSMDataset(peptide_spectrum_matches=sample_psms)

        with pytest.raises(IndexError):
            _ = dataset[2]

        with pytest.raises(IndexError):
            _ = dataset[-3]

    def test_length(self, sample_psms):
        """Test PSMDataset length calculation."""
        dataset = PSMDataset(peptide_spectrum_matches=sample_psms)
        assert len(dataset) == 2

        # Test empty dataset
        empty_dataset = PSMDataset(peptide_spectrum_matches=[])
        assert len(empty_dataset) == 0

    def test_iteration(self, sample_psms):
        """Test PSMDataset iteration."""
        dataset = PSMDataset(peptide_spectrum_matches=sample_psms)

        collected_psms = list(dataset)
        assert collected_psms == sample_psms

        # Test iteration with enumerate
        for i, psm in enumerate(dataset):
            assert psm == sample_psms[i]

    def test_empty_dataset(self):
        """Test operations on empty PSMDataset."""
        empty_dataset = PSMDataset(peptide_spectrum_matches=[])

        assert len(empty_dataset) == 0
        assert list(empty_dataset) == []

    def test_from_dataset_empty_sequences(self):
        """Test from_dataset with empty sequences."""
        dataset = PSMDataset.from_dataset([], [], [])
        assert len(dataset) == 0

    def test_from_dataset_mismatched_lengths(self):
        """Test from_dataset with sequences of different lengths."""
        spectra = [np.array([[100.0, 1000.0]])]
        peptides = [np.array([1, 2]), np.array([3, 4])]  # Different length
        confidence_scores = [0.9]

        with pytest.raises(ValueError, match="All sequences must have the same length"):
            PSMDataset.from_dataset(spectra, peptides, confidence_scores)
