"""Defines data structures to handle datasets used for model calibration in peptide sequencing tasks.

It integrates metadata, predictions, and spectra data to support various operations such as filtering, merging, and evaluating peptide sequence predictions against spectra.

Classes:
    CalibrationDataset: The main class for storing and processing calibration datasets.
    DatasetLoader: Protocol defining the interface for dataset loaders (see interfaces.py).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union
import pickle

import numpy as np
import pandas as pd

from winnow.compat.instanovo import ScoredSequence
from winnow.utils.peptide import tokens_to_proforma


@dataclass
class CalibrationDataset:
    """A class to store and process calibration datasets for peptide sequencing.

    It holds metadata and prediction results and provides various utility methods for filtering,
    saving, and evaluating data. For loading datasets from various file formats, see the
    concrete implementations of the DatasetLoader interface in data_loaders.py.

    Attributes:
        metadata (pd.DataFrame): DataFrame containing metadata and predictions.
        predictions (Optional[List[Optional[List[ScoredSequence]]]]): List of beam search results.
            Can be None if beam predictions were not loaded (e.g., single-prediction mode).
    """

    metadata: pd.DataFrame
    predictions: Optional[List[Optional[List[ScoredSequence]]]] = None

    def __post_init__(self):
        """Validate that metadata and predictions have matching lengths."""
        # Allow predictions to be None (no beam predictions available)
        # But if predictions are provided, they must match metadata length
        if self.predictions is not None and len(self.metadata) != len(self.predictions):
            raise AssertionError("Length of metadata and predictions must match")

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
                    tokens_to_proforma
                )
            output_metadata["prediction"] = output_metadata["prediction"].apply(
                tokens_to_proforma
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

        The filtering is done by using two predicates: one for the metadata (applied to each row)
        and one for the predictions (applied to each beam).

        Args:
            metadata_predicate (Callable[[Any], bool], optional): A function that takes a row
                from the metadata DataFrame and returns True if the row should be filtered OUT.
                Defaults to a predicate that always returns False, keeping all rows.
            predictions_predicate (Callable[[Any], bool], optional): A function that takes a
                beam (prediction) and returns True if the prediction should be filtered OUT.
                Defaults to a predicate that always returns False, keeping all predictions.
                Cannot be used (with non-default value) if predictions is None.

        Returns:
            CalibrationDataset: A new instance of CalibrationDataset containing only the entries
                for which the conditions specified by the predicates are False.

        Raises:
            ValueError: If predictions_predicate is non-default but predictions is None.
        """
        filter_idxs = []

        # If the dataset is empty, return it unchanged.
        if len(self.metadata) == 0:
            return self

        # -- Get filter indices for metadata condition
        (metadata_filter_idxs,) = np.where(
            self.metadata.apply(metadata_predicate, axis=1).values
        )
        filter_idxs.extend(metadata_filter_idxs.tolist())

        # -- Get filter indices for predictions condition
        predictions_filter_idxs = []
        # Check if user provided a non-default predictions_predicate
        default_predicate = lambda beam: False  # noqa: E731
        is_default_predicate = (
            predictions_predicate.__code__.co_code == default_predicate.__code__.co_code
        )

        if self.predictions is None:
            if not is_default_predicate:
                raise ValueError(
                    "Cannot use predictions_predicate when predictions is None. "
                    "This dataset was loaded without beam predictions."
                )
            # No predictions to filter, skip
        else:
            for idx, beam in enumerate(self.predictions):
                try:
                    # Pass None or empty beams as-is to the predicate, allowing it to handle them
                    if predictions_predicate(beam):
                        predictions_filter_idxs.append(idx)
                except (IndexError, AttributeError, TypeError) as e:
                    # Provide helpful error message if predicate crashes on None/empty/short beams
                    raise self._create_predicate_error_message(e, beam, idx) from e
            filter_idxs.extend(predictions_filter_idxs)

        filter_idxs_set = set(filter_idxs)

        # -- Gather predictions
        if self.predictions is not None:
            predictions: Optional[List[Optional[List[ScoredSequence]]]] = [
                prediction
                for idx, prediction in enumerate(self.predictions)
                if idx not in filter_idxs_set
            ]
        else:
            predictions = None

        # -- Gather metadata
        selection_idxs = [
            idx for idx in range(len(self.metadata)) if idx not in filter_idxs_set
        ]
        metadata = self.metadata.iloc[selection_idxs].copy(deep=True)
        metadata = metadata.reset_index(drop=True)

        return CalibrationDataset(predictions=predictions, metadata=metadata)

    def to_csv(self, path: Union[Path, str]) -> None:
        """Saves the dataset metadata to a CSV file.

        Args:
            path (str): Path to the output CSV file.
        """
        if isinstance(path, str):
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata.to_csv(path)

    def to_parquet(self, path: Union[Path, str]) -> None:
        """Saves the dataset metadata to a parquet file.

        Args:
            path (str): Path to the output parquet file.
        """
        if isinstance(path, str):
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata.to_parquet(path)

    def _create_predicate_error_message(
        self, error: Exception, beam: Optional[List[ScoredSequence]], idx: int
    ) -> ValueError:
        """Create a helpful error message when predicate evaluation fails.

        Args:
            error: The exception that was raised.
            beam: The beam that caused the error (may be None or empty).
            idx: The index of the entry that caused the error.

        Returns:
            ValueError with a helpful error message.
        """
        if beam is None:
            beam_status = "None"
            suggestion = (
                "Please handle None beams explicitly in your predicate, "
                "e.g., 'lambda beam: beam is None or (beam is not None and ...)'"
            )
        elif isinstance(error, IndexError):
            # IndexError means beam is too short for the accessed index
            # Could be empty (len == 0) or just too short (len < accessed_index)
            try:
                beam_length = len(beam)
                if beam_length == 0:
                    beam_status = "empty"
                    suggestion = (
                        "Please handle empty beams explicitly in your predicate, "
                        "e.g., 'lambda beam: len(beam) == 0 or (len(beam) > 0 and ...)'"
                    )
                else:
                    beam_status = f"too short (length {beam_length})"
                    suggestion = (
                        f"The beam has length {beam_length} but the predicate tried to access "
                        f"an index beyond this. Please check the beam length before accessing "
                        f"specific indices, e.g., 'lambda beam: len(beam) > 0 and beam[0]....' "
                        f"or 'lambda beam: len(beam) > 1 and beam[1]....'"
                    )
            except TypeError:
                # If len() fails, beam is likely None (shouldn't happen due to first check)
                beam_status = "None or invalid"
                suggestion = (
                    "Please handle None beams explicitly in your predicate, "
                    "e.g., 'lambda beam: beam is None or (beam is not None and ...)'"
                )
        else:
            # AttributeError or TypeError (other than from len())
            try:
                beam_length = len(beam)
                if beam_length == 0:
                    beam_status = "empty"
                else:
                    beam_status = f"non-empty (length {beam_length})"
            except TypeError:
                beam_status = "None or invalid"
            suggestion = "Please check the beam structure before accessing attributes or indices."

        return ValueError(
            f"Error evaluating predictions_predicate on entry {idx}: {error}. "
            f"The beam is {beam_status}. {suggestion}"
        )

    def __len__(self) -> int:
        """Returns the number of entries in the dataset."""
        return len(self.metadata)

    def __getitem__(
        self, index: int
    ) -> Tuple[pd.Series, Optional[List[ScoredSequence]]]:
        """Retrieves a metadata row and its corresponding prediction.

        Args:
            index (int): Index of the desired entry.

        Returns:
            Tuple[pd.Series, Optional[List[ScoredSequence]]]: The metadata row and its
                associated predictions. Predictions will be None if beam predictions
                were not loaded for this dataset.
        """
        predictions = self.predictions[index] if self.predictions is not None else None
        return self.metadata.iloc[index], predictions
