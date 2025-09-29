"""Defines interfaces for dataset loading functionality.

This module provides abstract interfaces that define the contract for dataset loaders.
"""

from typing import Protocol, Optional
from pathlib import Path
from winnow.datasets.calibration_dataset import CalibrationDataset


class DatasetLoader(Protocol):
    """Protocol defining the interface for dataset loaders.

    Any class implementing this protocol must provide a load method that returns a CalibrationDataset.
    """

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
