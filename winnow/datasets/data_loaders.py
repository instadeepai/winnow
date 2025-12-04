"""Concrete implementations of dataset loaders for different file formats.

This module provides concrete implementations of the DatasetLoader interface for various
file formats and data sources used in peptide sequencing tasks.
"""

import ast
import pickle
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
from pyteomics import mztab
from instanovo.utils.residues import ResidueSet
from instanovo.utils.metrics import Metrics

from winnow.datasets.interfaces import DatasetLoader
from winnow.datasets.calibration_dataset import (
    CalibrationDataset,
    ScoredSequence,
)


class InstaNovoDatasetLoader(DatasetLoader):
    """Loader for InstaNovo predictions in CSV format."""

    def __init__(
        self,
        residue_masses: dict[str, float],
        residue_remapping: dict[str, str],
        isotope_error_range: Tuple[int, int] = (0, 1),
    ) -> None:
        """Initialise the InstaNovoDatasetLoader.

        Args:
            residue_masses: The mapping of residue masses to UNIMOD tokens.
            residue_remapping: The mapping of residue notations to UNIMOD tokens.
            isotope_error_range: The range of isotope errors to consider when matching peptides.
        """
        self.metrics = Metrics(
            residue_set=ResidueSet(
                residue_masses=residue_masses, residue_remapping=residue_remapping
            ),
            isotope_error_range=isotope_error_range,
        )

    @staticmethod
    def _load_beam_preds(
        predictions_path: Path | str,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Loads a dataset from a CSV file and optionally filters it.

        Args:
            predictions_path (Path | str): The path to the CSV file containing the predictions.

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame]: A tuple containing the predictions and beams dataframes.
        """
        predictions_path = Path(predictions_path)
        if predictions_path.suffix != ".csv":
            raise ValueError(
                f"Unsupported file format for InstaNovo beam predictions: {predictions_path.suffix}. Supported format is .csv."
            )
        df = pl.read_csv(predictions_path)
        # Use polars column selectors to split dataframe
        beam_df = df.select(
            cs.contains(
                "instanovo_predictions_beam", "instanovo_log_probabilities_beam"
            )
        )
        preds_df = df.select(
            ~cs.contains(
                ["instanovo_predictions_beam", "instanovo_log_probabilities_beam"]
            )
        )
        return preds_df, beam_df

    @staticmethod
    def _load_spectrum_data(spectrum_path: Path | str) -> Tuple[pl.DataFrame, bool]:
        """Loads spectrum data from either a Parquet or IPC file.

        Args:
            spectrum_path (Path | str): The path to the spectrum data file.

        Returns:
            Tuple[pl.DataFrame, bool]: A tuple containing the spectrum data and a boolean indicating whether the dataset has ground truth labels.
        """
        spectrum_path = Path(spectrum_path)

        if spectrum_path.suffix == ".parquet":
            df = pl.read_parquet(spectrum_path)
        elif spectrum_path.suffix == ".ipc":
            df = pl.read_ipc(spectrum_path)
        else:
            raise ValueError(
                f"Unsupported file format for spectrum data: {spectrum_path.suffix}. Supported formats are .parquet and .ipc."
            )

        if "sequence" in df.columns:
            has_labels = True
        else:
            has_labels = False

        return df, has_labels

    @staticmethod
    def _merge_spectrum_data(
        beam_dataset: pd.DataFrame, spectrum_dataset: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge the input and output data from the de novo sequencing model.

        Args:
            beam_dataset (pd.DataFrame): The dataframe containing the beam predictions.
            spectrum_dataset (pd.DataFrame): The dataframe containing the spectrum data.

        Returns:
            pd.DataFrame: The merged dataframe.
        """
        merged_df = pd.merge(
            beam_dataset,
            spectrum_dataset,
            on=["spectrum_id"],
            suffixes=("_from_beams", ""),
        )
        merged_df = merged_df.drop(
            columns=[
                col + "_from_beams"
                for col in beam_dataset.columns
                if col in spectrum_dataset.columns and col != "spectrum_id"
            ],
            axis=1,
        )

        if len(merged_df) != len(beam_dataset):
            raise ValueError(
                f"Merge conflict: Expected {len(beam_dataset)} rows, but got {len(merged_df)}."
            )

        return merged_df

    def load(
        self, *, data_path: Path, predictions_path: Optional[Path] = None, **kwargs: Any
    ) -> CalibrationDataset:
        """Load a CalibrationDataset from InstaNovo CSV predictions.

        Args:
            data_path: Path to the spectrum data file
            predictions_path: Path to the IPC or parquet beam predictions file
            **kwargs: Not used

        Returns:
            CalibrationDataset: An instance of the CalibrationDataset class containing metadata and predictions.

        Raises:
            ValueError: If predictions_path is None
        """
        if predictions_path is None:
            raise ValueError("predictions_path is required for InstaNovoDatasetLoader")

        beam_predictions_path = predictions_path
        inputs, has_labels = self._load_spectrum_data(data_path)
        inputs = self._process_spectrum_data(inputs, has_labels)

        predictions, beams = self._load_beam_preds(beam_predictions_path)
        beams = self._process_beams(beams)
        predictions = self._process_predictions(predictions.to_pandas(), has_labels)

        predictions = self._merge_spectrum_data(predictions, inputs)
        predictions = self._evaluate_predictions(predictions, has_labels)

        return CalibrationDataset(metadata=predictions, predictions=beams)

    def _process_beams(
        self, beam_df: pl.DataFrame
    ) -> List[Optional[List[ScoredSequence]]]:
        """Processes beam predictions into scored sequences.

        Args:
            beam_df (pl.DataFrame): The dataframe containing the beam predictions.

        Returns:
            List[Optional[List[ScoredSequence]]]: A list of scored sequences for each row in the dataframe.
        """

        def convert_row_to_scored_sequences(
            row: dict,
        ) -> Optional[List[ScoredSequence]]:
            scored_sequences = []
            num_beams = len(row) // 2

            for beam in range(num_beams):
                seq_col, log_prob_col, token_log_prob_col = (
                    f"instanovo_predictions_beam_{beam}",
                    f"instanovo_log_probabilities_beam_{beam}",
                    f"token_log_probabilities_beam_{beam}",
                )
                sequence, log_prob, token_log_prob = (
                    row.get(seq_col),
                    row.get(log_prob_col, float("-inf")),
                    row.get(token_log_prob_col),
                )

                if sequence and log_prob > float("-inf"):
                    scored_sequences.append(
                        ScoredSequence(
                            sequence=self.metrics._split_peptide(sequence),
                            mass_error=None,
                            sequence_log_probability=log_prob,
                            token_log_probabilities=token_log_prob,
                        )
                    )

            return scored_sequences or None

        # Apply L -> I transformation to multiple columns using polars with_columns
        beam_df = beam_df.with_columns(
            [
                pl.col(col).str.replace_all("L", "I")
                for col in beam_df.columns
                if "instanovo_predictions_beam" in col
            ]
        )

        # Converts each row of the polars dataframe to a list of scored sequences representing the beam predictions for that row/spectrum.
        # All the beams are then stored in a list representing the entire dataset.
        return [
            convert_row_to_scored_sequences(row)
            for row in beam_df.iter_rows(named=True)
        ]

    def _process_spectrum_data(
        self, df: pl.DataFrame, has_labels: bool
    ) -> pd.DataFrame:
        """Processes the input data from the de novo sequencing model.

        Args:
            df (pl.DataFrame): The dataframe containing the spectrum data.
            has_labels (bool): Whether the dataset has ground truth labels.

        Returns:
            pd.DataFrame: The processed dataframe.
        """
        # Convert to pandas for downstream compatibility
        df = df.to_pandas()
        if has_labels:
            df["sequence"] = (
                df["sequence"]
                .apply(
                    lambda peptide: peptide.replace("L", "I")
                    if isinstance(peptide, str)
                    else peptide
                )
                .apply(self.metrics._split_peptide)
            )
        return df

    def _process_predictions(
        self, dataset: pd.DataFrame, has_labels: bool
    ) -> pd.DataFrame:
        """Processes the predictions obtained from saved beams.

        Args:
            dataset (pd.DataFrame): The dataframe containing the predictions.
            has_labels (bool): Whether the dataset has ground truth labels.

        Returns:
            pd.DataFrame: The processed dataframe.
        """
        rename_dict = {
            "predictions": "prediction_untokenised",
            "predictions_tokenised": "prediction",
            "log_probs": "confidence",
        }
        if has_labels:
            rename_dict["sequence"] = "sequence_untokenised"
        dataset.rename(rename_dict, axis=1, inplace=True)

        dataset["prediction"] = dataset["prediction"].apply(
            lambda peptide: peptide.split(", ") if isinstance(peptide, str) else peptide
        )

        dataset.loc[dataset["confidence"] == -1.0, "confidence"] = float("-inf")
        dataset["confidence"] = dataset["confidence"].apply(np.exp)

        if has_labels:
            dataset["sequence_untokenised"] = dataset["sequence_untokenised"].apply(
                lambda peptide: peptide.replace("L", "I")
                if isinstance(peptide, str)
                else peptide
            )
            dataset["sequence"] = dataset["sequence_untokenised"].apply(
                self.metrics._split_peptide
            )
        dataset["prediction"] = dataset["prediction"].apply(
            lambda peptide: [
                "I" if amino_acid == "L" else amino_acid for amino_acid in peptide
            ]
            if isinstance(peptide, list)
            else peptide
        )
        dataset["prediction_untokenised"] = dataset["prediction_untokenised"].apply(
            lambda peptide: peptide.replace("L", "I")
            if isinstance(peptide, str)
            else peptide
        )

        return dataset

    def _evaluate_predictions(
        self, dataset: pd.DataFrame, has_labels: bool
    ) -> pd.DataFrame:
        """Evaluates predictions in a dataset by checking validity and accuracy.

        Args:
            dataset (pd.DataFrame): The dataframe containing the predictions.
            has_labels (bool): Whether the dataset has ground truth labels.

        Returns:
            pd.DataFrame: The processed dataframe.
        """
        if has_labels:
            dataset["valid_peptide"] = dataset["sequence"].apply(
                lambda peptide: isinstance(peptide, list)
            )
        dataset["valid_prediction"] = dataset["prediction"].apply(
            lambda peptide: isinstance(peptide, list)
        )
        if has_labels:
            dataset["num_matches"] = dataset.apply(
                lambda row: self.metrics._novor_match(
                    row["sequence"], row["prediction"]
                )
                if isinstance(row["sequence"], list)
                and isinstance(row["prediction"], list)
                else 0,
                axis=1,
            )
            dataset["correct"] = dataset.apply(
                lambda row: (
                    row["num_matches"] == len(row["sequence"]) == len(row["prediction"])
                    if isinstance(row["sequence"], list)
                    and isinstance(row["prediction"], list)
                    else False
                ),
                axis=1,
            )
        return dataset


class MZTabDatasetLoader(DatasetLoader):
    """Loader for MZTab predictions from both traditional search engines and Casanovo outputs.

    This loader expects MZTab files with specific column names and formats:

    Required MZTab Columns:
        - spectra_ref: Spectrum identifier with format containing "index=N" (e.g., "ms_run[1]:index=123")
        - sequence: Peptide sequence string (may contain modifications in either UNIMOD or Casanovo format)
        - search_engine_score[1]: Confidence score for the prediction

    Optional MZTab Columns:
        - opt_ms_run[1]_aa_scores: Comma-separated amino acid level scores (Casanovo)
          If missing (traditional search engines), token_log_probabilities will be set to None

    Expected Spectrum Data Format:
        - Parquet or IPC file with spectrum metadata
        - Row indices should match the extracted indices from MZTab spectra_ref
        - Optional 'sequence' column for ground truth labels

    Note: The loader handles both single prediction per spectrum and multiple predictions
    per spectrum, creating beam predictions with List[ScoredSequence] structure. Works with
    both traditional database search engines and Casanovo outputs, returning a single beam prediction if only one prediction is present.
    """

    def __init__(
        self,
        residue_masses: dict[str, float],
        residue_remapping: dict[str, str],
        isotope_error_range: Tuple[int, int] = (0, 1),
    ) -> None:
        """Initialise the MZTabDatasetLoader.

        Args:
            residue_masses: The mapping of residue masses to UNIMOD tokens.
            residue_remapping: The mapping of residue notations to UNIMOD tokens.
            isotope_error_range: The range of isotope errors to consider when matching peptides.
        """
        self.metrics = Metrics(
            residue_set=ResidueSet(
                residue_masses=residue_masses, residue_remapping=residue_remapping
            ),
            isotope_error_range=isotope_error_range,
        )

    @staticmethod
    def _load_spectrum_data(spectrum_path: Path | str) -> Tuple[pl.DataFrame, bool]:
        """Load spectrum data from either a Parquet or IPC file.

        Args:
            spectrum_path: Path to spectrum data file

        Returns:
            DataFrame containing spectrum data
        """
        spectrum_path = Path(spectrum_path)
        has_labels = False

        if spectrum_path.suffix == ".parquet":
            df = pl.read_parquet(spectrum_path)
        elif spectrum_path.suffix == ".ipc":
            df = pl.read_ipc(spectrum_path)
        else:
            raise ValueError(
                f"Unsupported file format for spectrum data: {spectrum_path.suffix}. Supported formats are .parquet and .ipc."
            )

        if "sequence" in df.columns:
            has_labels = True

        return df, has_labels

    @staticmethod
    def _load_dataset(predictions_path: Path | str) -> pl.DataFrame:
        """Load predictions from mzTab file.

        Args:
            predictions_path: Path to mzTab file containing predictions

        Returns:
            DataFrame containing predictions
        """
        predictions_path = Path(predictions_path)
        if predictions_path.suffix != ".mztab":
            raise ValueError(
                f"Unsupported file format for MZTab predictions: {predictions_path.suffix}. Supported format is .mztab."
            )
        predictions = mztab.MzTab(str(predictions_path)).spectrum_match_table
        return pl.DataFrame(predictions)

    def load(
        self, *, data_path: Path, predictions_path: Optional[Path] = None, **kwargs: Any
    ) -> CalibrationDataset:
        """Load a calibration dataset from MZTab predictions.

        Args:
            data_path: Path to the spectrum data file
            predictions_path: Path to the MZTab predictions file
            **kwargs: Not used

        Returns:
            CalibrationDataset: An instance of the CalibrationDataset class containing metadata and predictions.

        Raises:
            ValueError: If predictions_path is None
        """
        if predictions_path is None:
            raise ValueError("predictions_path is required for MZTabDatasetLoader")
        # Load and process spectrum data
        spectrum_data, has_labels = self._load_spectrum_data(data_path)
        spectrum_data = self._process_spectrum_data(spectrum_data, has_labels)

        # Load and process predictions
        predictions = self._load_dataset(predictions_path)
        predictions = self._process_predictions(predictions)
        predictions = self._tokenize(
            predictions, "prediction_untokenised", "prediction"
        )

        # Get top predictions for metadata
        top_predictions = self._get_top_predictions(predictions)

        # Merge data and compute statistics
        metadata = self._merge_data(spectrum_data, top_predictions)
        metadata = self._evaluate_predictions(metadata, has_labels)

        # Convert to pandas and ensure prediction column is a list
        metadata_pd = metadata.to_pandas()
        metadata_pd["prediction"] = metadata_pd["prediction"].apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )

        # Create beam predictions in the same order as the final metadata
        # Extract the indices from the merged metadata to ensure alignment
        ordered_indices = metadata.get_column("index").to_list()
        beam_predictions = self._create_beam_predictions(predictions, ordered_indices)

        return CalibrationDataset(metadata=metadata_pd, predictions=beam_predictions)

    def _process_predictions(self, predictions: pl.DataFrame) -> pl.DataFrame:
        """Process raw predictions into a standardized format.

        Args:
            predictions: Raw predictions from mzTab file

        Returns:
            Processed predictions with standardized columns
        """
        # Check if amino acid scores are available (Casanovo) or missing (traditional search engines)
        has_aa_scores = "opt_ms_run[1]_aa_scores" in predictions.columns

        # Build list of columns to process
        columns_to_add = [
            # Extract spectrum index from spectra_ref (e.g., "ms_run[1]:index=123" -> 123)
            pl.col("spectra_ref")
            .str.extract(r"index=(\d+)")
            .cast(pl.Int64)
            .alias("index"),
            # Replace L with I for proteomics normalisation
            pl.col("sequence").str.replace("L", "I").alias("prediction_untokenised"),
            pl.col("search_engine_score[1]").alias("confidence"),
        ]

        # Add amino acid scores if available, otherwise create None column
        if has_aa_scores:
            columns_to_add.append(
                # Parse a string of comma-separated scores into list of floats
                pl.col("opt_ms_run[1]_aa_scores")
                .str.split(",")
                .cast(pl.List(pl.Float64))
                .alias("token_scores")
            )
            columns_to_drop = [
                "search_engine_score[1]",
                "opt_ms_run[1]_aa_scores",
                "sequence",
            ]
        else:
            columns_to_add.append(
                # Create None column for traditional search engines that don't provide token scores
                pl.lit(None, dtype=pl.List(pl.Float64)).alias("token_scores")
            )
            columns_to_drop = ["search_engine_score[1]", "sequence"]

        predictions = predictions.with_columns(columns_to_add).drop(columns_to_drop)

        # Sort predictions by index and confidence to ensure correct ordering
        predictions = predictions.sort(
            ["index", "confidence"], descending=[False, True]
        )

        return predictions

    def _tokenize(
        self,
        predictions: pl.DataFrame,
        untokenised_column: str,
        tokenised_column: str,
    ) -> pl.DataFrame:
        """Tokenize peptide strings into lists of amino acids and map modifications.

        Args:
            predictions: Processed predictions or sequence labels
            untokenised_column: Name of the column containing the untokenised sequence
            tokenised_column: Name of the column to name the tokenised sequence

        Returns:
            Predictions with tokenized sequences
        """
        return predictions.with_columns(
            # Map modifications to UNIMOD format (e.g., "M+15.995" -> "M[UNIMOD:35]")
            pl.col(untokenised_column)
            .map_elements(self._map_modifications, return_dtype=pl.Utf8)
            .alias(tokenised_column)
        ).with_columns(
            # Split sequence string into list of amino acid tokens
            pl.col(tokenised_column)
            .map_elements(self.metrics._split_peptide, return_dtype=pl.List(pl.Utf8))
            .alias(tokenised_column)
        )

    def _create_beam_predictions(
        self, predictions: pl.DataFrame, valid_spectra_indices: List[int]
    ) -> List[Optional[List[ScoredSequence]]]:
        """Create beam predictions from MZTab predictions.

        Args:
            predictions: DataFrame containing predictions
            valid_spectra_indices: List of indices corresponding to valid spectra in the merged metadata.

        Returns:
            List of beam predictions
        """
        # Create ScoredSequence objects for each spectrum's predictions
        beam_predictions = []
        for spectrum_index in valid_spectra_indices:
            # Get all predictions for this spectrum, sorted by confidence
            spectrum_preds = predictions.filter(pl.col("index") == spectrum_index)
            spectrum_preds = spectrum_preds.sort("confidence", descending=True)

            # Convert to ScoredSequence objects
            scored_sequences = []
            for row in spectrum_preds.iter_rows(named=True):
                scored_sequences.append(
                    ScoredSequence(
                        sequence=row["prediction"],
                        mass_error=None,
                        sequence_log_probability=row["confidence"],
                        token_log_probabilities=row[
                            "token_scores"
                        ],  # None for traditional search engines
                    )
                )
            beam_predictions.append(scored_sequences if scored_sequences else None)
        return beam_predictions

    def _map_modifications(self, sequence: str) -> str:
        """Map modifications to UNIMOD."""
        for mod, unimod in self.metrics.residue_remapping.items():
            sequence = sequence.replace(mod, unimod)
        return sequence

    def _get_top_predictions(self, predictions: pl.DataFrame) -> pl.DataFrame:
        """Get highest scoring prediction for each spectrum.

        Args:
            predictions: Tokenized predictions (pre-sorted by confidence within each spectrum)

        Returns:
            DataFrame containing only the highest scoring prediction per spectrum
        """
        # Since predictions are already sorted by confidence within each spectrum,
        # we can simply take the first row for each spectrum using a window function
        return predictions.filter(pl.int_range(0, pl.len()).over("index") == 0)

    def _process_spectrum_data(
        self, spectrum_data: pl.DataFrame, has_labels: bool
    ) -> pl.DataFrame:
        """Process spectrum data into standardized format.

        Args:
            spectrum_data: Raw spectrum data

        Returns:
            Processed spectrum data with standardised columns
        """
        if has_labels:
            spectrum_data = spectrum_data.with_columns(
                # Replace L with I
                pl.col("sequence").str.replace("L", "I").alias("sequence_untokenised")
            )
            spectrum_data = self._tokenize(
                spectrum_data, "sequence_untokenised", "sequence"
            )

        return spectrum_data

    def _merge_data(
        self, spectrum_data: pl.DataFrame, predictions: pl.DataFrame
    ) -> pl.DataFrame:
        """Merge spectrum data with top prediction for each spectrum.

        Args:
            spectrum_data: Processed spectrum data
            predictions: Top predictions for each spectrum

        Returns:
            Merged DataFrame containing both spectrum data and predictions
        """
        return (
            spectrum_data.with_row_index()  # Add row numbers as "index" column
            .join(predictions, left_on="index", right_on="index", how="inner")
            .sort("index")  # Ensure final order matches original spectrum order
            # Keep index column for beam predictions ordering
        )

    def _evaluate_predictions(
        self, metadata: pl.DataFrame, has_labels: bool
    ) -> pl.DataFrame:
        """Compute match statistics between predictions and ground truth.

        Args:
            metadata: Merged data containing both predictions and ground truth
            has_labels: Whether the dataset has ground truth labels

        Returns:
            DataFrame with added match statistics
        """
        # Check validity of tokenised sequences
        metadata = metadata.with_columns(
            [
                pl.col("prediction")
                .map_elements(
                    lambda x: isinstance(x, pl.Series), return_dtype=pl.Boolean
                )
                .alias("valid_prediction"),
            ]
        )

        if has_labels:
            metadata = metadata.with_columns(
                [
                    pl.col("sequence")
                    .map_elements(
                        lambda x: isinstance(x, pl.Series), return_dtype=pl.Boolean
                    )
                    .alias("valid_peptide"),
                ]
            )

        # Compute match statistics using struct to pass multiple columns to function
        return metadata.with_columns(
            [
                # Count matching amino acids between prediction and ground truth
                pl.struct(["sequence", "prediction"])
                .map_elements(
                    lambda row: self.metrics._novor_match(
                        row["sequence"], row["prediction"]
                    )
                    if isinstance(row["sequence"], list)
                    and isinstance(row["prediction"], list)
                    else 0,
                    return_dtype=pl.Int64,
                )
                .alias("num_matches"),
            ]
        ).with_columns(
            [
                # Check if prediction is completely correct (all amino acids match)
                pl.struct(["sequence", "prediction", "num_matches"])
                .map_elements(
                    lambda row: (
                        row["num_matches"]
                        == len(row["sequence"])
                        == len(row["prediction"])
                        if isinstance(row["sequence"], list)
                        and isinstance(row["prediction"], list)
                        else False
                    ),
                    return_dtype=pl.Boolean,
                )
                .alias("correct")
            ]
        )


class PointNovoDatasetLoader(DatasetLoader):
    """Loader for PointNovo format predictions.

    Note: This loader is not yet implemented.
    """

    def load(
        self, *, data_path: Path, predictions_path: Optional[Path] = None, **kwargs: Any
    ) -> CalibrationDataset:
        """Load a calibration dataset from PointNovo predictions.

        Args:
            data_path: Path to the spectrum data file
            predictions_path: Path to the predictions file
            **kwargs: Not used

        Returns:
            CalibrationDataset: A dataset containing merged spectra and PointNovo predictions.

        Raises:
            NotImplementedError: This loader is not yet implemented.
            ValueError: If predictions_path is None
        """
        raise NotImplementedError("PointNovoDatasetLoader is not yet implemented")


class WinnowDatasetLoader(DatasetLoader):
    """Loader for previously saved CalibrationDataset instances."""

    def __init__(
        self,
        residue_masses: dict[str, float],
        residue_remapping: dict[str, str],
        isotope_error_range: Tuple[int, int] = (0, 1),
    ) -> None:
        """Initialise the WinnowDatasetLoader.

        Args:
            residue_masses: The mapping of residue masses to UNIMOD tokens.
            residue_remapping: The mapping of residue notations to UNIMOD tokens.
            isotope_error_range: The range of isotope errors to consider when matching peptides.
        """
        self.metrics = Metrics(
            residue_set=ResidueSet(
                residue_masses=residue_masses, residue_remapping=residue_remapping
            ),
            isotope_error_range=isotope_error_range,
        )

    def load(
        self, *, data_path: Path, predictions_path: Optional[Path] = None, **kwargs: Any
    ) -> CalibrationDataset:
        """Load a previously saved CalibrationDataset.

        Args:
            data_path: Path to the directory containing the dataset
            predictions_path: Not used (for compatibility with DatasetLoader interface)
            **kwargs: Not used

        Returns:
            CalibrationDataset: The loaded dataset.
        """
        if predictions_path is not None:
            raise ValueError("predictions_path is not used for WinnowDatasetLoader")

        metadata_csv_path = data_path / "metadata.csv"
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
            lambda s: ast.literal_eval(s)
            if "," in s
            else ast.literal_eval(re.sub(r"(\n?)(\s+)", ", ", re.sub(r"\[\s+", "[", s)))
        )
        metadata["intensity_array"] = metadata["intensity_array"].apply(
            lambda s: ast.literal_eval(s)
            if "," in s
            else ast.literal_eval(re.sub(r"(\n?)(\s+)", ", ", re.sub(r"\[\s+", "[", s)))
        )

        predictions_pkl_path = data_path / "predictions.pkl"
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
