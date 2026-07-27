"""Supervised PyTorch Dataset for calibrator training on pre-computed features.

``FeatureDataset`` is a training-only adapter: labelled feature rows for
:meth:`~winnow.calibration.calibrator.ProbabilityCalibrator.fit_from_features`.
For feature computation and inference, use
:class:`~winnow.datasets.calibration_dataset.CalibrationDataset`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FeatureDataset(Dataset):
    """Supervised training Dataset of pre-computed feature rows and labels.

    Each sample is a ``(features_tensor, label_tensor)`` pair for supervised training.
    Labels are required; this type is not used for inference
    (use :class:`~winnow.datasets.calibration_dataset.CalibrationDataset`
    and :meth:`~winnow.calibration.calibrator.ProbabilityCalibrator.predict`).

    Construct from in-memory arrays or load labelled Parquet via
    :meth:`from_parquet`.

    Args:
        features: 2-D array of shape ``(n_samples, n_features)``.
        labels: 1-D array of shape ``(n_samples,)``.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        if len(features) != len(labels):
            raise ValueError(
                f"features ({len(features)}) and labels ({len(labels)}) "
                f"must have the same length"
            )
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        feature_columns: Sequence[str] | None = None,
    ) -> FeatureDataset:
        """Load a labelled feature matrix from Parquet for training.

        If ``path`` is a directory, all ``*.parquet`` files inside it are
        read and concatenated.  The ``correct`` column is required and used
        as the label.

        When ``feature_columns`` is provided, only those columns are used
        as features (in the given order).  Otherwise all numeric columns
        except ``correct`` are used. That fallback is only appropriate for
        lean matrices that already contain exclusively the training schema;
        for two-phase calibrator training always prefer passing
        ``feature_columns=["confidence", *calibrator.columns]``.

        Args:
            path: A ``.parquet`` file or a directory containing
                ``*.parquet`` files.
            feature_columns: Ordered list of column names to use as
                features.  If ``None``, all numeric columns (excluding
                ``correct``) are used.

        Returns:
            A new ``FeatureDataset`` instance.

        Raises:
            FileNotFoundError: If no Parquet files are found at ``path``.
            ValueError: If the ``correct`` label column is missing.
        """
        path = Path(path)
        if path.is_dir():
            parquet_files = sorted(path.glob("*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No .parquet files found in directory {path}")
            df = pl.concat([pl.read_parquet(f) for f in parquet_files])
        else:
            df = pl.read_parquet(path)

        if "correct" not in df.columns:
            raise ValueError(
                f"Parquet at {path} must contain a 'correct' column "
                f"for labels. Found columns: {df.columns}"
            )

        labels = df["correct"].to_numpy().astype(np.float32)

        if feature_columns is not None:
            missing = [c for c in feature_columns if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Feature columns missing from Parquet: {missing}. "
                    f"Available: {df.columns}"
                )
            features = df.select(feature_columns).to_numpy().astype(np.float32)
        else:
            import polars.selectors

            features = (
                df.drop("correct")
                .select(polars.selectors.numeric())
                .to_numpy()
                .astype(np.float32)
            )
            logger.warning(
                "No feature_columns specified; using all %d numeric columns "
                "from %s. For two-phase calibrator training this can silently "
                "include metadata columns and disagree with "
                "calibrator.columns; prefer passing "
                "feature_columns=['confidence', *calibrator.columns] "
                "(or an equivalent ordered schema) instead.",
                features.shape[1],
                path,
            )

        return cls(features=features, labels=labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
