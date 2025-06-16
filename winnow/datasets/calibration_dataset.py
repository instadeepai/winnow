"""Defines data structures to handle datasets used for model calibration in peptide sequencing tasks.

It integrates metadata, predictions, and spectra data to support various operations such as filtering, merging, and evaluating peptide sequence predictions against spectra.

Classes:
    CalibrationDataset: The main class for storing and processing calibration datasets.
    DatasetLoader: Protocol defining the interface for dataset loaders (see interfaces.py).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
import pickle

import numpy as np
import pandas as pd

from instanovo.utils.residues import ResidueSet
from instanovo.utils.metrics import Metrics
from instanovo.inference.beam_search import ScoredSequence


RESIDUE_MASSES: dict[str, float] = {
    "G": 57.021464,
    "A": 71.037114,
    "S": 87.032028,
    "P": 97.052764,
    "V": 99.068414,
    "T": 101.047670,
    "C": 103.009185,
    "L": 113.084064,
    "I": 113.084064,
    "N": 114.042927,
    "D": 115.026943,
    "Q": 128.058578,
    "K": 128.094963,
    "E": 129.042593,
    "M": 131.040485,
    "H": 137.058912,
    "F": 147.068414,
    "R": 156.101111,
    "Y": 163.063329,
    "W": 186.079313,
    # Modifications
    "M[UNIMOD:35]": 147.035400,  # Oxidation
    "N[UNIMOD:7]": 115.026943,  # Deamidation
    "Q[UNIMOD:7]": 129.042594,  # Deamidation
    "C[UNIMOD:4]": 160.030649,  # Carboxyamidomethylation
    "S[UNIMOD:21]": 166.998028,  # Phosphorylation
    "T[UNIMOD:21]": 181.01367,  # Phosphorylation
    "Y[UNIMOD:21]": 243.029329,  # Phosphorylation
    "[UNIMOD:385]": -17.026549,  # Ammonia Loss
    "[UNIMOD:5]": 43.005814,  # Carbamylation
    "[UNIMOD:1]": 42.010565,  # Acetylation
    # External data modifications
    "C[UNIMOD:312]": 222.013284,  # Cysteinylation
    "E[UNIMOD:27]": 111.032028,  # Glu -> pyro-Glu
    "Q[UNIMOD:28]": 111.032029,  # Gln -> pyro-Gln
    "(+25.98)": 25.980265,  # Carbamylation & NH3 loss
}

residue_set = ResidueSet(residue_masses=RESIDUE_MASSES)
metrics = Metrics(residue_set=residue_set, isotope_error_range=[0, 1])


@dataclass
class CalibrationDataset:
    """A class to store and process calibration datasets for peptide sequencing.

    It holds metadata and prediction results and provides various utility methods for filtering,
    saving, and evaluating data. For loading datasets from various file formats, see the
    concrete implementations of the DatasetLoader interface in data_loaders.py.

    Attributes:
        metadata (pd.DataFrame): DataFrame containing metadata and predictions.
        predictions (List[Optional[List[ScoredSequence]]]): List of beam search results.
    """

    metadata: pd.DataFrame
    predictions: List[Optional[List[ScoredSequence]]]

    def save(self, data_dir: Path) -> None:
        """Save a `CalibrationDataset` to a directory.

        Args:
            data_dir (Path): Directory to save the dataset. This will contain `metadata.csv` and
                            optionally, `predictions.pkl` for serialized beam search results.
        """
        data_dir.mkdir(parents=True, exist_ok=True)
        with (data_dir / "metadata.csv").open(mode="w") as metadata_file:
            output_metadata = self.metadata.copy(deep=True)
            if "sequence" in output_metadata.columns:
                output_metadata["sequence"] = output_metadata["sequence"].apply(
                    lambda peptide_list: "".join(peptide_list)
                )
            output_metadata["prediction"] = output_metadata["prediction"].apply(
                lambda peptide_list: "".join(peptide_list)
            )
            output_metadata.to_csv(metadata_file, index=False)

        if self.predictions:
            with (data_dir / "predictions.pkl").open(mode="wb") as predictions_file:
                pickle.dump(self.predictions, predictions_file)

    @property
    def confidence_column(self) -> str:
        """Returns the column name that stores confidence scores in the dataset."""
        return "confidence"

    def filter_entries(
        self,
        metadata_predicate: Callable[[Any], bool] = lambda row: False,
        predictions_predicate: Callable[[Any], bool] = lambda beam: False,
    ) -> "CalibrationDataset":
        """Filters the dataset based on the specified conditions for both metadata and predictions.

        The filtering is done by using two predicates: one for the metadata (applied to each row) and one for the predictions (applied to each beam).

        Args:
            metadata_predicate (Callable[[Any], bool], optional): A function that takes a row from the metadata DataFrame and returns a boolean indicating whether the row should be kept. Defaults to a predicate that always returns False, keeping all rows.
            predictions_predicate (Callable[[Any], bool], optional): A function that takes a beam (prediction) and returns a boolean indicating whether the prediction should be kept. Defaults to a predicate that always returns False, keeping all predictions.

        Returns:
            CalibrationDataset: A new instance of `CalibrationDataset` containing only the entries that satisfy the conditions specified by the predicates.
        """
        filter_idxs = []

        # -- Get filter indices for metadata condition
        (metadata_filter_idxs,) = np.where(
            self.metadata.apply(metadata_predicate, axis=1).values
        )
        filter_idxs.extend(metadata_filter_idxs.tolist())

        # -- Get filter indices for predictions condition
        predictions_filter_idxs = [
            idx
            for idx, beam in enumerate(self.predictions)
            if predictions_predicate(beam)
        ]
        filter_idxs.extend(predictions_filter_idxs)

        filter_idxs_set = set(filter_idxs)

        # -- Gather predictions
        predictions = [
            prediction
            for idx, prediction in enumerate(self.predictions)
            if idx not in filter_idxs_set
        ]

        # -- Gather metadata
        selection_idxs = [
            idx for idx in range(len(self.metadata)) if idx not in filter_idxs_set
        ]
        metadata = self.metadata.iloc[selection_idxs].copy(deep=True)
        metadata = metadata.reset_index(drop=True)

        return CalibrationDataset(predictions=predictions, metadata=metadata)

    def to_csv(self, path: Path) -> None:
        """Saves the dataset metadata to a CSV file.

        Args:
            path (str): Path to the output CSV file.
        """
        self.metadata.to_csv(path)

    def to_parquet(self, path: str) -> None:
        """Saves the dataset metadata to a parquet file.

        Args:
            path (str): Path to the output parquet file.
        """
        self.metadata.to_parquet(path)

    def __len__(self) -> int:
        """Returns the number of entries in the dataset."""
        assert self.metadata.shape[0] == len(self.predictions)
        return len(self.predictions)

    def __getitem__(self, index) -> Tuple[pd.Series, List[ScoredSequence]]:
        """Retrieves a metadata row and its corresponding prediction.

        Args:
            index (int): Index of the desired entry.

        Returns:
            Tuple[pd.Series, List[ScoredSequence]]: The metadata row and its associated predictions.
        """
        return self.metadata.iloc[index], self.predictions[index]
