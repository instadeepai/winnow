"""Unit tests for winnow calibration feature MassErrorFeature."""

import pytest
import pandas as pd

from winnow.calibration.features.mass_error import MassErrorFeature
from winnow.calibration.features.constants import CARBON_ISOTOPE_MASS_SHIFT
from winnow.datasets.calibration_dataset import CalibrationDataset


class TestMassErrorFeature:
    """Test the MassErrorFeature class."""

    H2O = 18.0106
    PROTON = 1.007276

    @pytest.fixture()
    def residue_masses(self):
        return {
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
            "V": 99.068414,
        }

    @pytest.fixture()
    def mass_error_feature(self, residue_masses):
        """Create a MassErrorFeature instance for testing."""
        return MassErrorFeature(residue_masses=residue_masses)

    def test_properties(self, mass_error_feature):
        """Test MassErrorFeature properties."""
        assert mass_error_feature.name == "Mass Error"
        assert mass_error_feature.columns == ["mass_error_ppm"]
        assert mass_error_feature.dependencies == []

    def test_default_isotope_error_range(self, mass_error_feature):
        """Test default isotope error range is (0, 1)."""
        assert mass_error_feature.isotope_error_range == (0, 1)

    def test_prepare_does_nothing(self, mass_error_feature):
        """Test that prepare method does nothing."""
        metadata = pd.DataFrame(
            {
                "precursor_mz": [500.0],
                "precursor_charge": [2],
                "prediction": [["G", "A"]],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)
        original = dataset.metadata.copy()
        mass_error_feature.prepare(dataset)
        pd.testing.assert_frame_equal(dataset.metadata, original)

    def test_compute_monoisotopic_match(self, residue_masses):
        """When the measured m/z exactly equals the theoretical m/z, ppm error should be ~0."""
        peptide = ["G", "A"]
        charge = 2
        theo_mz = self._theoretical_mz(residue_masses, peptide, charge)

        feature = MassErrorFeature(
            residue_masses=residue_masses, isotope_error_range=(0, 0)
        )
        metadata = pd.DataFrame(
            {
                "precursor_mz": [theo_mz],
                "precursor_charge": [charge],
                "prediction": [peptide],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)
        feature.compute(dataset)

        assert dataset.metadata.iloc[0]["mass_error_ppm"] == pytest.approx(
            0.0, abs=1e-6
        )

    def test_compute_ppm_formula(self, residue_masses):
        """Verify the ppm formula matches the expected calculation."""
        peptide = ["G", "A"]
        charge = 2
        measured_mz = 500.503638
        theo_mz = self._theoretical_mz(residue_masses, peptide, charge)

        feature = MassErrorFeature(
            residue_masses=residue_masses, isotope_error_range=(0, 0)
        )
        metadata = pd.DataFrame(
            {
                "precursor_mz": [measured_mz],
                "precursor_charge": [charge],
                "prediction": [peptide],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)
        feature.compute(dataset)

        expected_ppm = (theo_mz - measured_mz) / measured_mz * 1e6
        assert dataset.metadata.iloc[0]["mass_error_ppm"] == pytest.approx(
            expected_ppm, rel=1e-6
        )

    def test_isotope_correction_selects_best(self, residue_masses):
        """When measured m/z matches M+1 isotope, isotope=1 should give a smaller error."""
        peptide = ["G", "A", "P"]
        charge = 2
        theo_mz = self._theoretical_mz(residue_masses, peptide, charge)
        # Simulate instrument selecting M+1 peak
        measured_mz = theo_mz + CARBON_ISOTOPE_MASS_SHIFT / charge

        feature = MassErrorFeature(
            residue_masses=residue_masses, isotope_error_range=(0, 1)
        )
        metadata = pd.DataFrame(
            {
                "precursor_mz": [measured_mz],
                "precursor_charge": [charge],
                "prediction": [peptide],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)
        feature.compute(dataset)

        # With isotope=1 correction, the error should be ~0
        assert dataset.metadata.iloc[0]["mass_error_ppm"] == pytest.approx(0.0, abs=1.0)

    def test_isotope_correction_keeps_sign(self, residue_masses):
        """Verify the selected ppm error preserves its sign."""
        peptide = ["G", "A"]
        charge = 2
        theo_mz = self._theoretical_mz(residue_masses, peptide, charge)
        # Measured is slightly above M+1 → corrected error should be small and positive
        offset = 0.001
        measured_mz = theo_mz + CARBON_ISOTOPE_MASS_SHIFT / charge + offset

        feature = MassErrorFeature(
            residue_masses=residue_masses, isotope_error_range=(0, 1)
        )
        metadata = pd.DataFrame(
            {
                "precursor_mz": [measured_mz],
                "precursor_charge": [charge],
                "prediction": [peptide],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)
        feature.compute(dataset)

        # isotope=1 gives: theo_mz - (measured - 1.00335/2) / measured * 1e6
        # = theo_mz - (theo_mz + offset) / measured * 1e6 → small negative
        result = dataset.metadata.iloc[0]["mass_error_ppm"]
        assert result < 0

    def test_compute_with_invalid_peptide(self, residue_masses):
        """When prediction is not a list, mass_error_ppm should be a large sentinel value."""
        feature = MassErrorFeature(residue_masses=residue_masses)
        metadata = pd.DataFrame(
            {
                "precursor_mz": [500.503638],
                "precursor_charge": [2],
                "prediction": ["invalid_string"],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        with pytest.warns(UserWarning, match="not valid peptide sequences"):
            feature.compute(dataset)

        assert dataset.metadata.iloc[0]["mass_error_ppm"] == float("inf")

    def test_custom_residue_masses(self):
        """Test that custom residue masses are used correctly."""
        custom_masses = {"A": 100.0, "G": 200.0}
        feature = MassErrorFeature(
            residue_masses=custom_masses, isotope_error_range=(0, 0)
        )

        peptide = ["A", "G"]
        charge = 2
        theo_mz = self._theoretical_mz(custom_masses, peptide, charge)
        measured_mz = 500.503638

        metadata = pd.DataFrame(
            {
                "precursor_mz": [measured_mz],
                "precursor_charge": [charge],
                "prediction": [peptide],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)
        feature.compute(dataset)

        expected_ppm = (theo_mz - measured_mz) / measured_mz * 1e6
        assert dataset.metadata.iloc[0]["mass_error_ppm"] == pytest.approx(
            expected_ppm, rel=1e-6
        )

    def _theoretical_mz(self, residue_masses, peptide, charge):
        """Compute theoretical m/z for a peptide at a given charge."""
        neutral = sum(residue_masses[r] for r in peptide) + self.H2O
        return (neutral + charge * self.PROTON) / charge
