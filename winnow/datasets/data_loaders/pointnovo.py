"""PointNovo dataset loader (not yet implemented)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.datasets.interfaces import DatasetLoader


class PointNovoDatasetLoader(DatasetLoader):
    """Loader for PointNovo format predictions.

    Note: This loader is not yet implemented.
    """

    def load(
        self, *, data_path: Path, predictions_path: Optional[Path] = None, **kwargs: Any
    ) -> CalibrationDataset:
        """Load a calibration dataset from PointNovo predictions."""
        raise NotImplementedError("PointNovoDatasetLoader is not yet implemented")
