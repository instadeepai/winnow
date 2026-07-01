from typing import Dict, List, Tuple

import numpy as np

from winnow.calibration.features.base import CalibrationFeatures, FeatureDependency
from winnow.calibration.features.constants import (
    CARBON_ISOTOPE_MASS_SHIFT,
    H2O_MASS,
    PROTON_MASS,
)
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.datasets.data_loaders.utils import is_valid_peptide_tokens


def _validate_dataset(dataset: CalibrationDataset) -> None:
    if "precursor_mz" not in dataset.metadata.columns:
        raise ValueError(
            "precursor_mz column not found in dataset. This is required for mass error computation."
        )
    if "precursor_charge" not in dataset.metadata.columns:
        raise ValueError(
            "precursor_charge column not found in dataset. This is required for mass error computation."
        )


def _compute_signed_mass_errors(
    dataset: CalibrationDataset,
    residue_masses: Dict[str, float],
    isotope_error_range: Tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return signed mass errors as (ppm, daltons) with isotope correction."""
    measured_mz = dataset.metadata["precursor_mz"]
    charge = dataset.metadata["precursor_charge"]

    invalid_mask = ~dataset.metadata["prediction"].apply(is_valid_peptide_tokens)
    if invalid_mask.any():
        n_invalid = invalid_mask.sum()
        raise ValueError(
            f"{n_invalid} prediction(s) are not valid peptide sequences "
            f"(expected non-empty list of residue tokens)."
        )

    neutral_mass = dataset.metadata["prediction"].apply(
        lambda peptide: sum(residue_masses[residue] for residue in peptide) + H2O_MASS
    )
    theoretical_mz = (neutral_mass + charge * PROTON_MASS) / charge

    isotope_offsets = range(isotope_error_range[0], isotope_error_range[1] + 1)
    ppm_per_isotope = np.column_stack(
        [
            (
                theoretical_mz
                - (measured_mz - isotope * CARBON_ISOTOPE_MASS_SHIFT / charge)
            )
            / measured_mz
            * 1e6
            for isotope in isotope_offsets
        ]
    )
    da_per_isotope = np.column_stack(
        [
            (
                theoretical_mz
                - (measured_mz - isotope * CARBON_ISOTOPE_MASS_SHIFT / charge)
            )
            * charge
            for isotope in isotope_offsets
        ]
    )
    best_idx = np.argmin(np.abs(ppm_per_isotope), axis=1)
    row_idx = np.arange(len(best_idx))
    ppm = ppm_per_isotope[row_idx, best_idx]
    da = da_per_isotope[row_idx, best_idx]
    return ppm, da


class MassErrorPPMFeature(CalibrationFeatures):
    """Calculates the signed precursor mass error in ppm, correcting for possible isotope peak selection."""

    INVALID_PPM: float = float("inf")

    def __init__(
        self,
        residue_masses: Dict[str, float],
        isotope_error_range: Tuple[int, int] = (0, 1),
    ) -> None:
        super().__init__()
        self.residue_masses = residue_masses
        self.isotope_error_range = isotope_error_range

    @property
    def dependencies(self) -> List[FeatureDependency]:
        """Returns a list of dependencies required before computing the feature.

        Since this feature does not depend on other features, it returns an empty list.

        Returns:
            List[FeatureDependency]: An empty list.
        """
        return []

    @property
    def columns(self) -> List[str]:
        """Defines the column name for this feature.

        Returns:
            List[str]: A list containing the feature name.
        """
        return ["mass_error_ppm"]

    @property
    def name(self) -> str:
        """Returns the name of the feature.

        This method provides the natural language identifier used as the key for the feature.

        Returns:
            str: The feature identifier.
        """
        return "Mass Error (ppm)"

    def prepare(self, dataset: CalibrationDataset) -> None:
        """Prepares the dataset before feature computation.

        This method is intended to perform any preprocessing required before computing the feature.

        Args:
            dataset (CalibrationDataset): The dataset to prepare.
        """

    def compute(
        self,
        dataset: CalibrationDataset,
    ) -> None:
        """Computes the signed precursor mass error in ppm, correcting for isotope peak selection.

        For each isotope offset in ``isotope_error_range``, computes:

            ppm = (mz_theoretical - (mz_measured - isotope * 1.00335 / z)) / mz_measured * 1e6

        The isotope offset producing the smallest absolute error is selected,
        and its signed ppm value is stored.

        Args:
            dataset (CalibrationDataset): The dataset containing ``precursor_mz``,
                ``precursor_charge``, and ``prediction`` columns.
        """
        _validate_dataset(dataset)

        ppm, _ = _compute_signed_mass_errors(
            dataset,
            self.residue_masses,
            self.isotope_error_range,
        )
        dataset.metadata["mass_error_ppm"] = ppm


class MassErrorDaFeature(CalibrationFeatures):
    """Signed precursor mass error in Daltons (neutral-mass scale), with isotope correction."""

    def __init__(
        self,
        residue_masses: Dict[str, float],
        isotope_error_range: Tuple[int, int] = (0, 1),
    ) -> None:
        super().__init__()
        self.residue_masses = residue_masses
        self.isotope_error_range = isotope_error_range

    @property
    def dependencies(self) -> List[FeatureDependency]:
        """Returns a list of dependencies required before computing the feature.

        Since this feature does not depend on other features, it returns an empty list.

        Returns:
            List[FeatureDependency]: An empty list.
        """
        return []

    @property
    def columns(self) -> List[str]:
        """Defines the column name for this feature."""
        return ["mass_error_da"]

    @property
    def name(self) -> str:
        """Returns the name of the feature."""
        return "Mass Error (Da)"

    def prepare(self, dataset: CalibrationDataset) -> None:
        """Prepares the dataset before feature computation."""
        return

    def compute(self, dataset: CalibrationDataset) -> None:
        """Computes the signed precursor mass error in Daltons (neutral-mass scale), with isotope correction.

        For each isotope offset in ``isotope_error_range``, computes:

            da = (mz_theoretical - (mz_measured - isotope * 1.00335 / z)) * z

        The isotope offset producing the smallest absolute error is selected,
        and its signed Daltons value is stored.

        Args:
            dataset (CalibrationDataset): The dataset containing ``precursor_mz``,
                ``precursor_charge``, and ``prediction`` columns.
        """
        _validate_dataset(dataset)

        _, da = _compute_signed_mass_errors(
            dataset,
            self.residue_masses,
            self.isotope_error_range,
        )
        dataset.metadata["mass_error_da"] = da
