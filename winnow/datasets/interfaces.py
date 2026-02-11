"""Defines interfaces for dataset loading functionality.

This module provides abstract interfaces that define the contract for dataset loaders.
"""

from typing import Protocol, Optional, Tuple
from pathlib import Path
from winnow.datasets.calibration_dataset import CalibrationDataset


class DatasetLoader(Protocol):
    """Protocol defining the interface for dataset loaders.

    Any class implementing this protocol must provide a load method that returns a CalibrationDataset.
    """

    def __init__(
        self,
        residue_masses: dict[str, float],
        residue_remapping: dict[str, str] | None = None,
        isotope_error_range: Tuple[int, int] = (0, 1),
    ) -> None:
        """Initialise the DatasetLoader.

        Args:
            residue_masses: The mapping of residues to their masses (ProForma notation).
            residue_remapping: Optional mapping of input notations to ProForma notation. Defaults to None.
            isotope_error_range: The range of isotope errors to consider when matching peptides. Defaults to (0, 1).
        """
        ...

    def load(
        self, *, data_path: Path, predictions_path: Optional[Path] = None, **kwargs
    ) -> CalibrationDataset:
        """Load a dataset from the specified source(s).

        Args:
            data_path: Primary data source path (spectrum data, MGF file, or directory)
            predictions_path: Optional predictions source path (not needed for WinnowDatasetLoader)
            **kwargs: Additional loader-specific arguments

        Returns:
            CalibrationDataset: The loaded dataset
        """
        ...
