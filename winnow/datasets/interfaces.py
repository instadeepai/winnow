"""Defines interfaces for dataset loading functionality.

This module provides abstract interfaces that define the contract for dataset loaders.
"""

from typing import Protocol
from pathlib import Path
from winnow.datasets.calibration_dataset import CalibrationDataset


class DatasetLoader(Protocol):
    """Protocol defining the interface for dataset loaders.

    Any class implementing this protocol must provide a load method that returns a CalibrationDataset.
    The specific arguments to load() are determined by the implementing class.
    """

    def load(self, *args: Path, **kwargs) -> CalibrationDataset:
        """Load a dataset from the specified source(s).

        Args:
            *args: Path arguments specific to the implementing loader
            **kwargs: Keyword arguments specific to the implementing loader

        Returns:
            CalibrationDataset: The loaded dataset
        """
        ...
