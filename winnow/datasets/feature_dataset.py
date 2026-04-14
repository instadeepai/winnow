"""PyTorch Dataset wrapper for pre-computed calibration features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """Wraps numpy feature/label arrays as a PyTorch Dataset.

    Each sample is a ``(features_tensor, label_tensor)`` pair.  The
    dataset can be constructed from in-memory arrays or loaded from
    Parquet files via :meth:`from_parquet`.

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
    def from_parquet(cls, path: str | Path) -> FeatureDataset:
        """Load features from a single Parquet file or a directory of Parquets.

        If ``path`` is a directory, all ``*.parquet`` files inside it are
        read and concatenated.  The ``correct`` column is used as the
        label; all remaining numeric columns become the feature matrix.

        Args:
            path: A ``.parquet`` file or a directory containing
                ``*.parquet`` files.

        Returns:
            A new ``FeatureDataset`` instance.

        Raises:
            FileNotFoundError: If no Parquet files are found at ``path``.
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
        features = df.drop("correct").to_numpy().astype(np.float32)

        return cls(features=features, labels=labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
