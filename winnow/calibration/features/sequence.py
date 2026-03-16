from typing import List

from winnow.calibration.features.base import CalibrationFeatures, FeatureDependency
from winnow.datasets.calibration_dataset import CalibrationDataset


class SequenceFeatures(CalibrationFeatures):
    """Computes basic sequence features intended to help the calibrator distinguish between tryptic and non-tryptic peptides."""

    def __init__(self) -> None:
        """Initialize SequenceFeatures."""
        pass

    @property
    def dependencies(self) -> List[FeatureDependency]:
        """Returns a list of dependencies required before computing the feature."""
        return []

    @property
    def name(self) -> str:
        """Returns the name of the feature."""
        return "Sequence Features"

    @property
    def columns(self) -> List[str]:
        """Returns the columns of the feature."""
        return ["sequence_length", "precursor_charge", "is_c_term_tryptic"]

    def prepare(self, dataset: CalibrationDataset) -> None:
        """Prepares the dataset for the feature computation."""
        pass

    def compute(self, dataset: CalibrationDataset) -> None:
        """Computes the feature for the dataset.

        Computes:
            - sequence_length: Number of residues in the predicted peptide
            - is_c_term_tryptic: Whether the C-terminal residue is K or R (tryptic cleavage)

        Note: precursor_charge is listed in columns() but is expected to already
        exist in the metadata from the data loader.
        """
        dataset.metadata["sequence_length"] = dataset.metadata["prediction"].apply(len)
        dataset.metadata["is_c_term_tryptic"] = dataset.metadata["prediction"].apply(
            lambda seq: seq[-1] in ["K", "R"] if seq else False
        )
