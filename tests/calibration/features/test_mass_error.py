"""Unit tests for winnow calibration features MassErrorPPMFeature and MassErrorDaFeature."""

import pytest
import pandas as pd

from winnow.calibration.features.mass_error import (
    MassErrorDaFeature,
    MassErrorPPMFeature,
)
from winnow.calibration.features.constants import (
    CARBON_ISOTOPE_MASS_SHIFT,
    H2O_MASS,
    PROTON_MASS,
)
from winnow.datasets.calibration_dataset import CalibrationDataset


@pytest.fixture()
def residue_masses():
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


def theoretical_mz(residue_masses, peptide, charge):
    """Compute theoretical m/z for a peptide at a given charge."""
    neutral = sum(residue_masses[r] for r in peptide) + H2O_MASS
    return (neutral + charge * PROTON_MASS) / charge


class TestMassErrorPPMFeature:
    """Test the MassErrorPPMFeature class."""

    @pytest.fixture()
    def mass_error_feature(self, residue_masses):
        """Create a MassErrorPPMFeature instance for testing."""
        return MassErrorPPMFeature(residue_masses=residue_masses)

    def test_properties(self, mass_error_feature):
        """Test MassErrorPPMFeature properties."""
        assert mass_error_feature.name == "Mass Error (ppm)"
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
        theo_mz = theoretical_mz(residue_masses, peptide, charge)

        feature = MassErrorPPMFeature(
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
        theo_mz = theoretical_mz(residue_masses, peptide, charge)

        feature = MassErrorPPMFeature(
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
        theo_mz = theoretical_mz(residue_masses, peptide, charge)
        measured_mz = theo_mz + CARBON_ISOTOPE_MASS_SHIFT / charge

        feature = MassErrorPPMFeature(
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

        assert dataset.metadata.iloc[0]["mass_error_ppm"] == pytest.approx(0.0, abs=1.0)

    def test_isotope_correction_keeps_sign(self, residue_masses):
        """Verify the selected ppm error preserves its sign."""
        peptide = ["G", "A"]
        charge = 2
        theo_mz = theoretical_mz(residue_masses, peptide, charge)
        offset = 0.001
        measured_mz = theo_mz + CARBON_ISOTOPE_MASS_SHIFT / charge + offset

        feature = MassErrorPPMFeature(
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

        result = dataset.metadata.iloc[0]["mass_error_ppm"]
        assert result < 0

    def test_compute_with_invalid_peptide(self, residue_masses):
        """When prediction is not a list, compute should raise."""
        feature = MassErrorPPMFeature(residue_masses=residue_masses)
        metadata = pd.DataFrame(
            {
                "precursor_mz": [500.503638],
                "precursor_charge": [2],
                "prediction": ["invalid_string"],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        with pytest.raises(ValueError, match="not valid peptide sequences"):
            feature.compute(dataset)

    def test_custom_residue_masses(self):
        """Test that custom residue masses are used correctly."""
        custom_masses = {"A": 100.0, "G": 200.0}
        feature = MassErrorPPMFeature(
            residue_masses=custom_masses, isotope_error_range=(0, 0)
        )

        peptide = ["A", "G"]
        charge = 2
        theo_mz = theoretical_mz(custom_masses, peptide, charge)
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


class TestMassErrorDaFeature:
    """Test the MassErrorDaFeature class."""

    @pytest.fixture()
    def mass_error_feature(self, residue_masses):
        """Create a MassErrorDaFeature instance for testing."""
        return MassErrorDaFeature(residue_masses=residue_masses)

    def test_properties(self, mass_error_feature):
        """Test MassErrorDaFeature properties."""
        assert mass_error_feature.name == "Mass Error (Da)"
        assert mass_error_feature.columns == ["mass_error_da"]
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
        """When the measured m/z exactly equals the theoretical m/z, Da error should be ~0."""
        peptide = ["G", "A"]
        charge = 2
        theo_mz = theoretical_mz(residue_masses, peptide, charge)

        feature = MassErrorDaFeature(
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

        assert dataset.metadata.iloc[0]["mass_error_da"] == pytest.approx(0.0, abs=1e-6)

    def test_compute_da_formula(self, residue_masses):
        """Verify the Daltons formula matches the expected calculation."""
        peptide = ["G", "A"]
        charge = 2
        measured_mz = 500.503638
        theo_mz = theoretical_mz(residue_masses, peptide, charge)

        feature = MassErrorDaFeature(
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

        expected_da = (theo_mz - measured_mz) * charge
        assert dataset.metadata.iloc[0]["mass_error_da"] == pytest.approx(
            expected_da, rel=1e-6
        )

    def test_isotope_correction_selects_best(self, residue_masses):
        """When measured m/z matches M+1 isotope, isotope=1 should give a smaller error."""
        peptide = ["G", "A", "P"]
        charge = 2
        theo_mz = theoretical_mz(residue_masses, peptide, charge)
        measured_mz = theo_mz + CARBON_ISOTOPE_MASS_SHIFT / charge

        feature = MassErrorDaFeature(
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

        assert dataset.metadata.iloc[0]["mass_error_da"] == pytest.approx(0.0, abs=1e-3)

    def test_isotope_correction_keeps_sign(self, residue_masses):
        """Verify the selected Da error preserves its sign."""
        peptide = ["G", "A"]
        charge = 2
        theo_mz = theoretical_mz(residue_masses, peptide, charge)
        offset = 0.001
        measured_mz = theo_mz + CARBON_ISOTOPE_MASS_SHIFT / charge + offset

        feature = MassErrorDaFeature(
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

        result = dataset.metadata.iloc[0]["mass_error_da"]
        assert result < 0

    def test_compute_with_invalid_peptide(self, residue_masses):
        """When prediction is not a list, compute should raise."""
        feature = MassErrorDaFeature(residue_masses=residue_masses)
        metadata = pd.DataFrame(
            {
                "precursor_mz": [500.503638],
                "precursor_charge": [2],
                "prediction": ["invalid_string"],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        with pytest.raises(ValueError, match="not valid peptide sequences"):
            feature.compute(dataset)

    def test_custom_residue_masses(self):
        """Test that custom residue masses are used correctly."""
        custom_masses = {"A": 100.0, "G": 200.0}
        feature = MassErrorDaFeature(
            residue_masses=custom_masses, isotope_error_range=(0, 0)
        )

        peptide = ["A", "G"]
        charge = 2
        theo_mz = theoretical_mz(custom_masses, peptide, charge)
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

        expected_da = (theo_mz - measured_mz) * charge
        assert dataset.metadata.iloc[0]["mass_error_da"] == pytest.approx(
            expected_da, rel=1e-6
        )

    def test_ppm_and_da_features_are_consistent(self, residue_masses):
        """PPM and Da features should agree on isotope selection and scale."""
        peptide = ["G", "A", "P"]
        charge = 3
        theo_mz = theoretical_mz(residue_masses, peptide, charge)
        measured_mz = theo_mz + 0.002

        metadata = pd.DataFrame(
            {
                "precursor_mz": [measured_mz],
                "precursor_charge": [charge],
                "prediction": [peptide],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        ppm_feature = MassErrorPPMFeature(residue_masses=residue_masses)
        da_feature = MassErrorDaFeature(residue_masses=residue_masses)
        ppm_feature.compute(dataset)
        da_feature.compute(dataset)

        ppm = dataset.metadata.iloc[0]["mass_error_ppm"]
        da = dataset.metadata.iloc[0]["mass_error_da"]
        assert da == pytest.approx(ppm * measured_mz / 1e6 * charge, rel=1e-6)
