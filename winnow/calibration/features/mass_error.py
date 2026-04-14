from typing import Dict, List, Tuple
import warnings

import numpy as np

from winnow.calibration.features.base import CalibrationFeatures, FeatureDependency
from winnow.calibration.features.constants import CARBON_ISOTOPE_MASS_SHIFT
from winnow.datasets.calibration_dataset import CalibrationDataset


class MassErrorFeature(CalibrationFeatures):
    """Calculates the signed precursor mass error in ppm, correcting for possible isotope peak selection."""

    INVALID_PPM: float = float("inf")
    h2o_mass: float = 18.0106
    proton_mass: float = 1.007276

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
        return "Mass Error"

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
        if "precursor_mz" not in dataset.metadata.columns:
            raise ValueError(
                "precursor_mz column not found in dataset. This is required for mass error computation."
            )
        if "precursor_charge" not in dataset.metadata.columns:
            raise ValueError(
                "precursor_charge column not found in dataset. This is required for mass error computation."
            )

        measured_mz = dataset.metadata["precursor_mz"]
        charge = dataset.metadata["precursor_charge"]

        # Compute theoretical m/z from peptide sequence
        invalid_mask = ~dataset.metadata["prediction"].apply(
            lambda p: isinstance(p, list)
        )
        if invalid_mask.any():
            n_invalid = invalid_mask.sum()
            warnings.warn(
                f"{n_invalid} prediction(s) are not valid peptide sequences "
                f"(expected list of residue tokens). These will receive a large "
                f"mass error value ({self.INVALID_PPM} ppm).",
                stacklevel=2,
            )

        neutral_mass = dataset.metadata["prediction"].apply(
            lambda peptide: (
                sum(self.residue_masses[residue] for residue in peptide) + self.h2o_mass
                if isinstance(peptide, list)
                else 0.0
            )
        )
        theoretical_mz = (neutral_mass + charge * self.proton_mass) / charge

        # Compute ppm error for each isotope offset, keep the one closest to zero
        isotope_offsets = range(
            self.isotope_error_range[0], self.isotope_error_range[1] + 1
        )
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
        best_idx = np.argmin(np.abs(ppm_per_isotope), axis=1)
        result = ppm_per_isotope[np.arange(len(best_idx)), best_idx]
        result[invalid_mask.values] = self.INVALID_PPM
        dataset.metadata["mass_error_ppm"] = result
