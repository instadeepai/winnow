"""Loader for previously saved CalibrationDataset instances."""

from __future__ import annotations

import ast
import pickle
import re
from pathlib import Path
from typing import Any, Optional, Tuple

import pandas as pd
from instanovo.utils.metrics import Metrics
from instanovo.utils.residues import ResidueSet

from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.datasets.interfaces import DatasetLoader


class WinnowDatasetLoader(DatasetLoader):
    """Loader for previously saved CalibrationDataset instances."""

    def __init__(
        self,
        residue_masses: dict[str, float],
        residue_remapping: dict[str, str],
        isotope_error_range: Tuple[int, int] = (0, 1),
    ) -> None:
        """Initialise the WinnowDatasetLoader."""
        self.metrics = Metrics(
            residue_set=ResidueSet(
                residue_masses=residue_masses, residue_remapping=residue_remapping
            ),
            isotope_error_range=isotope_error_range,
        )

    def load(
        self, *, data_path: Path, predictions_path: Optional[Path] = None, **kwargs: Any
    ) -> CalibrationDataset:
        """Load a previously saved CalibrationDataset."""
        if predictions_path is not None:
            raise ValueError("predictions_path is not used for WinnowDatasetLoader")

        metadata_csv_path = data_path / Path("metadata.csv")
        if not metadata_csv_path.exists():
            raise FileNotFoundError(
                f"Winnow dataset loader expects a CSV file containing metadata at {metadata_csv_path}. "
                f"The specified directory {data_path} should contain a 'metadata.csv' file "
                f"with PSM metadata from a previously saved Winnow dataset."
            )

        try:
            with metadata_csv_path.open(mode="r") as metadata_file:
                metadata = pd.read_csv(metadata_file)
        except Exception as e:
            raise ValueError(
                f"Failed to read metadata.csv from Winnow dataset directory {data_path}. "
                f"The file should be a valid CSV containing PSM metadata. Error: {e}"
            ) from e

        if "sequence" in metadata.columns:
            metadata["sequence"] = metadata["sequence"].apply(
                self.metrics._split_peptide
            )
        metadata["prediction"] = metadata["prediction"].apply(
            self.metrics._split_peptide
        )
        metadata["mz_array"] = metadata["mz_array"].apply(
            lambda s: (
                ast.literal_eval(s)
                if "," in s
                else ast.literal_eval(
                    re.sub(r"(\n?)(\s+)", ", ", re.sub(r"\[\s+", "[", s))
                )
            )
        )
        metadata["intensity_array"] = metadata["intensity_array"].apply(
            lambda s: (
                ast.literal_eval(s)
                if "," in s
                else ast.literal_eval(
                    re.sub(r"(\n?)(\s+)", ", ", re.sub(r"\[\s+", "[", s))
                )
            )
        )

        predictions_pkl_path = data_path / Path("predictions.pkl")
        if predictions_pkl_path.exists():
            try:
                with predictions_pkl_path.open(mode="rb") as predictions_file:
                    predictions = pickle.load(predictions_file)
            except Exception as e:
                raise ValueError(
                    f"Failed to load predictions.pkl from Winnow dataset directory {data_path}. "
                    f"The file should be a pickled beam predictions object from a previously saved Winnow dataset. "
                    f"Error: {e}"
                ) from e
        else:
            predictions = None
        return CalibrationDataset(metadata=metadata, predictions=predictions)
