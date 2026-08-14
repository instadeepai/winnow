"""PointNovo dataset loader (not yet implemented)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.datasets.interfaces import DatasetLoader


class PointNovoDatasetLoader(DatasetLoader):
    """Loader for PointNovo format predictions.

    Note: This loader is not yet implemented.
    """

    def __init__(
        self,
        residue_masses: dict[str, float],
        residue_remapping: dict[str, str] | None = None,
        isotope_error_range: tuple[int, int] = (0, 1),
    ) -> None:
        """Initialise the loader with the common dataset-loader options.

        The loader does not use these options until PointNovo support is implemented,
        but defining the initializer makes this a concrete implementation of the
        ``DatasetLoader`` protocol on Python 3.10.
        """
        del residue_masses, residue_remapping, isotope_error_range

    def load(
        self, *, data_path: Path, predictions_path: Path | None = None, **kwargs: Any
    ) -> CalibrationDataset:
        """Load a calibration dataset from PointNovo predictions."""
        raise NotImplementedError("PointNovoDatasetLoader is not yet implemented")
