from typing import List, Dict

from winnow.calibration.features.base import CalibrationFeatures, FeatureDependency
from winnow.datasets.calibration_dataset import CalibrationDataset


class MassErrorFeature(CalibrationFeatures):
    """Calculates the difference between the observed precursor mass and the theoretical mass."""

    h2o_mass: float = 18.0106
    proton_mass: float = 1.007276

    def __init__(self, residue_masses: Dict[str, float]) -> None:
        super().__init__()
        self.residue_masses = residue_masses

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
        return ["mass_error"]

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
        """Computes the mass error for each peptide.

        The mass error is calculated as the difference between the observed precursor mass and the theoretical peptide mass,
        accounting for the mass of water (H2O) and a proton (H+), which are added during ionisation.

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

        # Compute MH+ precursor mass from precursor m/z and charge
        dataset.metadata["precursor_mass"] = dataset.metadata[
            "precursor_mz"
        ] * dataset.metadata["precursor_charge"] - (
            (dataset.metadata["precursor_charge"] - 1) * self.proton_mass
        )

        # Compute dehydrated theoretical mass from peptide sequence
        dehydrated_theoretical_mass = dataset.metadata["prediction"].apply(
            lambda peptide: sum(self.residue_masses[residue] for residue in peptide)
            if isinstance(peptide, list)
            else float("-inf")
        )
        # Compute theoretical MH+ mass: residues + H2O (peptide backbone) + H+ (ionisation)
        theoretical_mass = (
            dehydrated_theoretical_mass + self.h2o_mass + self.proton_mass
        )

        # Compute mass error from precursor mass and theoretical mass
        dataset.metadata[self.columns[0]] = (
            dataset.metadata["precursor_mass"] - theoretical_mass
        )
