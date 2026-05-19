"""Tests for MassErrorDaFeature."""

import pandas as pd

from winnow.calibration.features.mass_error import MassErrorDaFeature
from winnow.datasets.calibration_dataset import CalibrationDataset


def test_mass_error_da_column_added() -> None:
    residue_masses = {"A": 71.037114, "G": 57.021464}
    meta = pd.DataFrame(
        {
            "precursor_mz": [100.0],
            "precursor_charge": [1],
            "prediction": [["A", "G"]],
        }
    )
    meta["confidence"] = [0.5]
    dataset = CalibrationDataset(metadata=meta)

    feature = MassErrorDaFeature(
        residue_masses=residue_masses, isotope_error_range=(0, 0)
    )
    feature.compute(dataset)

    assert "mass_error_da" in dataset.metadata.columns
    assert dataset.metadata["mass_error_da"].notna().all()
