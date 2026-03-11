from typing import List

from winnow.calibration.features.base import CalibrationFeatures, FeatureDependency
from winnow.datasets.calibration_dataset import CalibrationDataset


class SequenceFeatures(CalibrationFeatures):
    """Computes basic sequence features that capture peptide properties affecting fragmentation and identification confidence."""

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
        return ["sequence_length", "precursor_charge"]

    def prepare(self, dataset: CalibrationDataset) -> None:
        """Prepares the dataset for the feature computation."""
        pass

    def compute(self, dataset: CalibrationDataset) -> None:
        """Computes the feature for the dataset.

        Computes:
            - sequence_length: Number of residues in the predicted peptide

        Note: precursor_charge is listed in columns() but is expected to already
        exist in the metadata from the data loader.
        """
        dataset.metadata["sequence_length"] = dataset.metadata["prediction"].apply(len)
