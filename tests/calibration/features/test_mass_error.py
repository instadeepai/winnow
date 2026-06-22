"""Unit tests for winnow calibration feature MassErrorFeature."""

import pytest
import pandas as pd

from winnow.calibration.features.mass_error import MassErrorFeature
from winnow.datasets.calibration_dataset import CalibrationDataset


class TestMassErrorFeature:
    """Test the MassErrorFeature class."""

    @pytest.fixture()
    def mass_error_feature(self):
        """Create a MassErrorFeature instance for testing."""
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
            "V": 99.068414,
        }
        return MassErrorFeature(residue_masses=residue_masses)

    @pytest.fixture()
    def sample_dataset(self):
        """Create a sample CalibrationDataset for testing.

        precursor_mz and precursor_charge are chosen so that the derived MH+ masses are
        1000.0, 1200.0, and 800.0 Da respectively, enabling straightforward verification:
            MH+ = precursor_mz * charge - (charge - 1) * proton_mass
        """
        metadata = pd.DataFrame(
            {
                # MH+ = 500.503638 * 2 - 1 * 1.007276 = 1000.0
                # MH+ = 400.671517 * 3 - 2 * 1.007276 = 1200.0
                # MH+ = 400.503638 * 2 - 1 * 1.007276 = 800.0
                "precursor_mz": [500.503638, 400.671517, 400.503638],
                "precursor_charge": [2, 3, 2],
                "prediction": [["G", "A"], ["A", "S", "P"], ["V"]],
                "confidence": [0.9, 0.8, 0.7],
            }
        )
        return CalibrationDataset(metadata=metadata, predictions=None)

    def test_properties(self, mass_error_feature):
        """Test MassErrorFeature properties."""
        assert mass_error_feature.name == "Mass Error"
        assert mass_error_feature.columns == ["mass_error"]
        assert mass_error_feature.dependencies == []

    def test_prepare_does_nothing(self, mass_error_feature, sample_dataset):
        """Test that prepare method does nothing."""
        # Should not raise any exception and not modify dataset
        original_metadata = sample_dataset.metadata.copy()
        mass_error_feature.prepare(sample_dataset)
        pd.testing.assert_frame_equal(sample_dataset.metadata, original_metadata)

    def test_compute_mass_error(self, mass_error_feature, sample_dataset):
        """Test mass error computation."""
        mass_error_feature.compute(sample_dataset)

        # Check that mass_error column was added and precursor_mass (MH+) was derived
        assert "mass_error" in sample_dataset.metadata.columns
        assert "precursor_mass" in sample_dataset.metadata.columns

        # Verify first row: precursor_mz=500.503638, charge=2
        #   MH+ observed = 500.503638 * 2 - (2-1) * 1.007276 = 1001.007276 - 1.007276 = 1000.0
        #   G + A dehydrated = 57.021464 + 71.037114 = 128.058578
        #   theoretical MH+ = 128.058578 + 18.0106 + 1.007276 = 147.076454
        #   mass_error = 1000.0 - 147.076454 = 852.923546
        proton_mass = 1.007276
        mz, charge = 500.503638, 2
        expected_precursor_mass = mz * charge - (charge - 1) * proton_mass
        expected_theoretical = 128.058578 + 18.0106 + proton_mass
        expected_first = expected_precursor_mass - expected_theoretical
        assert sample_dataset.metadata.iloc[0]["precursor_mass"] == pytest.approx(
            expected_precursor_mass, rel=1e-6, abs=1e-6
        )
        assert sample_dataset.metadata.iloc[0]["mass_error"] == pytest.approx(
            expected_first, rel=1e-6, abs=1e-6
        )

    def test_compute_with_invalid_peptide(self, mass_error_feature):
        """Test mass error computation with invalid peptide format.

        When the prediction is not a list, the dehydrated theoretical mass is set to
        -inf, making the theoretical MH+ also -inf, and the resulting mass error +inf.
        """
        metadata = pd.DataFrame(
            {
                "precursor_mz": [500.503638],
                "precursor_charge": [2],
                "prediction": ["invalid_string"],  # String instead of list
                "confidence": [0.9],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        mass_error_feature.compute(dataset)

        # MH+ observed is finite; theoretical MH+ is -inf → mass_error = +inf
        assert dataset.metadata.iloc[0]["mass_error"] == float("inf")

    def test_residue_masses_parameter(self):
        """Test that custom residue masses are used correctly.

        precursor_mz=500.503638 with charge=2 gives MH+=1000.0, so the mass error
        is simply 1000.0 minus the theoretical MH+ built from the custom masses.
        """
        custom_masses = {"A": 100.0, "G": 200.0}
        feature = MassErrorFeature(residue_masses=custom_masses)

        proton_mass = 1.007276
        metadata = pd.DataFrame(
            {
                "precursor_mz": [500.503638],
                "precursor_charge": [2],
                "prediction": [["A", "G"]],
                "confidence": [0.9],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        feature.compute(dataset)

        # MH+ observed = 500.503638 * 2 - 1 * 1.007276 = 1000.0
        # theoretical MH+ = 100.0 + 200.0 + 18.0106 + 1.007276 = 319.017876
        # mass_error = 1000.0 - 319.017876
        expected = 1000.0 - (300.0 + 18.0106 + proton_mass)
        assert dataset.metadata.iloc[0]["mass_error"] == pytest.approx(
            expected, rel=1e-6, abs=1e-6
        )
