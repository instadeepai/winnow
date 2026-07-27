"""Supervised PyTorch Dataset for calibrator training on pre-computed features.

``FeatureDataset`` is a training-only adapter: labelled feature rows for
:meth:`~winnow.calibration.calibrator.ProbabilityCalibrator.fit_from_features`.
For feature computation and inference, use
:class:`~winnow.datasets.calibration_dataset.CalibrationDataset`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import polars as pl
import polars.selectors
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from winnow.calibration.calibrator import ProbabilityCalibrator

CONFIDENCE_COLUMN = "confidence"
LABEL_COLUMN = "correct"


def _non_float_castable_columns(df: pl.DataFrame, columns: Sequence[str]) -> list[str]:
    """Return columns that cannot be cast to float32 (not numeric or boolean)."""
    castable = set(
        df.select(columns)
        .select(polars.selectors.numeric() | polars.selectors.boolean())
        .columns
    )
    return [name for name in columns if name not in castable]


def _normalise_feature_column_names(columns: Sequence[str]) -> list[str]:
    """Return non-confidence feature names; strip ``confidence`` if present."""
    normalised: list[str] = []
    seen: set[str] = set()
    for name in columns:
        if name == CONFIDENCE_COLUMN:
            continue
        if name in seen:
            raise ValueError(
                f"Duplicate feature column name {name!r} in FeatureDataset columns."
            )
        seen.add(name)
        normalised.append(name)
    return normalised


class FeatureDataset(Dataset):
    """Supervised training Dataset of pre-computed feature rows and labels.

    Each sample is a ``(features_tensor, label_tensor)`` pair for supervised training.
    Labels are required; this type is not used for inference
    (use :class:`~winnow.datasets.calibration_dataset.CalibrationDataset`
    and :meth:`~winnow.calibration.calibrator.ProbabilityCalibrator.predict`).

    The feature matrix layout is always ``[confidence, *columns]``: confidence
    occupies index 0 and is required, but is **not** listed in :attr:`columns`.
    ``columns`` holds only the non-confidence feature names in matrix order,
    matching :attr:`~winnow.calibration.calibrator.ProbabilityCalibrator.columns`.

    Construct from in-memory arrays, load a wide labelled Parquet via
    :meth:`from_parquet`, then align to a calibrator with :meth:`select_for`.

    Args:
        features: 2-D array of shape ``(n_samples, n_features)``.
        labels: 1-D array of shape ``(n_samples,)``.
        columns: Non-confidence feature column names in matrix order (indices
            1..). Do not include ``confidence``; if it is passed it is ignored.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        columns: Sequence[str],
    ) -> None:
        if len(features) != len(labels):
            raise ValueError(
                f"features ({len(features)}) and labels ({len(labels)}) "
                f"must have the same length"
            )
        if features.ndim != 2:
            raise ValueError(
                f"features must be 2-D (n_samples, n_features), got shape "
                f"{getattr(features, 'shape', None)}."
            )

        column_names = _normalise_feature_column_names(columns)
        expected_width = 1 + len(column_names)
        if features.shape[1] != expected_width:
            raise ValueError(
                f"Feature matrix width ({features.shape[1]}) must equal "
                f"1 + len(columns) ({expected_width}): confidence plus "
                f"non-confidence columns {column_names}."
            )

        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self._columns = column_names

    @property
    def columns(self) -> list[str]:
        """Non-confidence feature column names in matrix order.

        Confidence is always at index 0 of :attr:`features` and is not included.
        """
        return list(self._columns)

    @classmethod
    def from_parquet(cls, path: str | Path) -> FeatureDataset:
        """Load a labelled wide feature matrix from Parquet for training.

        If ``path`` is a directory, all ``*.parquet`` files inside it are
        read and concatenated. Requires ``correct`` (label) and ``confidence``
        (always placed at matrix index 0). All other numeric or boolean
        columns become the wide feature set in stable file order.

        To train, call :meth:`select_for` with a calibrator so the matrix
        matches ``calibrator.columns`` (extras raise at
        :meth:`~winnow.calibration.calibrator.ProbabilityCalibrator.fit_from_features`
        if left unaligned).

        Args:
            path: A ``.parquet`` file or a directory containing
                ``*.parquet`` files.

        Returns:
            A new ``FeatureDataset`` instance.

        Raises:
            FileNotFoundError: If no Parquet files are found at ``path``.
            ValueError: If ``correct`` or ``confidence`` is missing, or if
                those / feature columns are not numeric or boolean.
        """
        path = Path(path)
        if path.is_dir():
            parquet_files = sorted(path.glob("*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No .parquet files found in directory {path}")
            df = pl.concat([pl.read_parquet(f) for f in parquet_files])
        else:
            df = pl.read_parquet(path)

        if LABEL_COLUMN not in df.columns:
            raise ValueError(
                f"Parquet at {path} must contain a '{LABEL_COLUMN}' column "
                f"for labels. Found columns: {df.columns}"
            )
        if CONFIDENCE_COLUMN not in df.columns:
            raise ValueError(
                f"Parquet at {path} must contain a '{CONFIDENCE_COLUMN}' column. "
                f"Found columns: {df.columns}"
            )

        required = [LABEL_COLUMN, CONFIDENCE_COLUMN]
        invalid_required = _non_float_castable_columns(df, required)
        if invalid_required:
            detail = ", ".join(
                f"{name} ({df.schema[name]})" for name in invalid_required
            )
            raise ValueError(
                f"Column(s) '{LABEL_COLUMN}' and '{CONFIDENCE_COLUMN}' must be "
                f"numeric or boolean to cast to float32; got non-castable "
                f"column(s): {detail}."
            )

        labels = df[LABEL_COLUMN].to_numpy().astype(np.float32)

        other_castable = [
            name
            for name in df.select(
                polars.selectors.numeric() | polars.selectors.boolean()
            ).columns
            if name not in {LABEL_COLUMN, CONFIDENCE_COLUMN}
        ]
        invalid_others = _non_float_castable_columns(df, other_castable)
        if invalid_others:
            detail = ", ".join(f"{name} ({df.schema[name]})" for name in invalid_others)
            raise ValueError(
                f"Feature column(s) must be numeric or boolean to cast "
                f"to float32; got non-castable column(s): {detail}."
            )

        ordered = [CONFIDENCE_COLUMN, *other_castable]
        features = df.select(ordered).to_numpy().astype(np.float32)
        return cls(features=features, labels=labels, columns=other_castable)

    def select_for(self, calibrator: "ProbabilityCalibrator") -> FeatureDataset:
        """Return a new dataset aligned to ``calibrator.columns``.

        Builds matrix layout ``[confidence, *calibrator.columns]``. Does not
        mutate this instance, so a wide matrix can be reused for multiple
        calibrator schemas.

        Args:
            calibrator: A :class:`~winnow.calibration.calibrator.ProbabilityCalibrator` object.

        Returns:
            A new ``FeatureDataset`` whose :attr:`columns` equal
            ``list(calibrator.columns)``.

        Raises:
            ValueError: If any name in ``calibrator.columns`` is missing from
                this dataset's :attr:`columns`.
        """
        expected = list(calibrator.columns)
        available = set(self.columns)
        missing = [name for name in expected if name not in available]
        if missing:
            raise ValueError(
                f"Cannot select_for(calibrator): missing feature column(s) "
                f"{missing}. Available FeatureDataset.columns: {self.columns}. "
                f"Expected calibrator.columns: {expected}."
            )

        name_to_index = {name: i + 1 for i, name in enumerate(self.columns)}
        indices = [0] + [name_to_index[name] for name in expected]
        selected = self.features[:, indices].cpu().numpy()
        labels = self.labels.cpu().numpy()
        return FeatureDataset(features=selected, labels=labels, columns=expected)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
