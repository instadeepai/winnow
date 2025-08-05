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

from winnow.datasets.interfaces import DatasetLoader
from winnow.datasets.calibration_dataset import (
    CalibrationDataset,
    ScoredSequence,
)
from winnow.constants import metrics, INVALID_PROSIT_TOKENS


class InstaNovoDatasetLoader(DatasetLoader):
    """Loader for InstaNovo predictions in CSV format."""

    @staticmethod
    def _load_beam_preds(
        predictions_path: Path,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Loads a dataset from a CSV file and optionally filters it.

        Args:
            predictions_path (Path): The path to the CSV file containing the predictions.

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame]: A tuple containing the predictions and beams dataframes.
        """

        def _filter_dataset_for_prosit(df: pl.DataFrame) -> pl.DataFrame:
            """Applies filters to remove unsupported modifications."""
            print("Applying dataset filters...")

            # Filter out invalid tokens (~ negates condition in polars)
            for token in INVALID_PROSIT_TOKENS:
                df = df.filter(~df["preds"].str.contains(token))
                df = df.filter(~df["preds_beam_1"].str.contains(token))

            # Filter out unmodified cysteine using polars string operations
            df = df.filter(~df["preds_tokenised"].str.contains("C,"))

            # Filter out unmodified cysteine using regex for negative lookahead
            # NOTE: This is a workaround for the fact that polars does not support negative lookahead in its string operations
            pattern = re.compile(r"C(?!\[)")
            indexes_to_drop = [
                idx
                for idx, row in enumerate(df.iter_rows(named=True))
                if pattern.search(row["preds_beam_1"])
            ]
            df = df.filter(~pl.Series(range(len(df))).is_in(indexes_to_drop))

            return df

        df = pl.read_csv(predictions_path)
        df = _filter_dataset_for_prosit(df)
        # Use polars column selectors to split dataframe
        beam_df = df.select(cs.contains("_beam_"))
        preds_df = df.select(~cs.contains(["_beam_", "_log_probs_"]))
        return preds_df, beam_df

    @staticmethod
    def _process_beams(beam_df: pl.DataFrame) -> List[Optional[List[ScoredSequence]]]:
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
                    f"preds_beam_{beam}",
                    f"log_probs_beam_{beam}",
                    f"token_log_probs_{beam}",
                )
                sequence, log_prob, token_log_prob = (
                    row.get(seq_col),
                    row.get(log_prob_col, float("-inf")),
                    row.get(token_log_prob_col),
                )

                if sequence and log_prob > float("-inf"):
                    scored_sequences.append(
                        ScoredSequence(
                            sequence=metrics._split_peptide(sequence),
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
                if "preds_beam" in col
            ]
        )

        # Converts each row of the polars dataframe to a list of scored sequences representing the beam predictions for that row/spectrum.
        # All the beams are then stored in a list representing the entire dataset.
        return [
            convert_row_to_scored_sequences(row)
            for row in beam_df.iter_rows(named=True)
        ]

    @staticmethod
    def _process_predictions(dataset: pd.DataFrame, has_labels: bool) -> pd.DataFrame:
        """Processes the predictions obtained from saved beams.

        Args:
            dataset (pd.DataFrame): The dataframe containing the predictions.
            has_labels (bool): Whether the dataset has ground truth labels.

        Returns:
            pd.DataFrame: The processed dataframe.
        """
        rename_dict = {
            "preds": "prediction_untokenised",
            "preds_tokenised": "prediction",
            "log_probs": "confidence",
        }
        if has_labels:
            rename_dict["sequence"] = "sequence_untokenised"
        dataset.rename(rename_dict, axis=1, inplace=True)

        dataset["prediction"] = dataset["prediction"].apply(
            lambda peptide: peptide.split(", ")
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
                metrics._split_peptide
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
            raise ValueError(f"Unsupported file format: {spectrum_path.suffix}")

        if "sequence" in df.columns:
            has_labels = True
        else:
            has_labels = False

        return df, has_labels

    @staticmethod
    def _process_spectrum_data(df: pl.DataFrame, has_labels: bool) -> pd.DataFrame:
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
                .apply(metrics._split_peptide)
            )
        return df

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

    @staticmethod
    def _evaluate_predictions(dataset: pd.DataFrame, has_labels: bool) -> pd.DataFrame:
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
                lambda row: metrics._novor_match(row["sequence"], row["prediction"])
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

    def load(self, *args: Path, **kwargs: Any) -> CalibrationDataset:
        """Load a CalibrationDataset from InstaNovo CSV predictions.

        Args:
            *args: Should contain spectrum_path and beam_predictions_path in that order
            **kwargs: Not used

        Returns:
            CalibrationDataset: An instance of the CalibrationDataset class containing metadata and predictions.

        Raises:
            ValueError: If incorrect number of positional arguments provided
        """
        if len(args) != 2:
            raise ValueError(
                "Expected exactly 2 positional arguments: spectrum_path and beam_predictions_path"
            )
        spectrum_path, beam_predictions_path = args

        inputs, has_labels = self._load_spectrum_data(spectrum_path)
        inputs = self._process_spectrum_data(inputs, has_labels)

        predictions, beams = self._load_beam_preds(beam_predictions_path)
        beams = self._process_beams(beams)
        predictions = self._process_predictions(predictions.to_pandas(), has_labels)

        predictions = self._merge_spectrum_data(predictions, inputs)
        predictions = self._evaluate_predictions(predictions, has_labels)

        return CalibrationDataset(metadata=predictions, predictions=beams)


class CasanovoDatasetLoader(DatasetLoader):
    """Loader for Casanovo predictions.

    Note: This loader is not yet implemented.
    """

    def load(self, *args: Path, **kwargs: Any) -> CalibrationDataset:
        """Load a calibration dataset from Casanovo predictions.

        Args:
            *args: Should contain labelled_path, mgf_path, and predictions_path in that order
            **kwargs: Not used

        Returns:
            CalibrationDataset: A dataset containing merged labelled data, spectra, and predictions.

        Raises:
            NotImplementedError: This loader is not yet implemented.
            ValueError: If incorrect number of positional arguments provided
        """
        if len(args) != 3:
            raise ValueError(
                "Expected exactly 3 positional arguments: labelled_path, mgf_path, and predictions_path"
            )
        raise NotImplementedError("CasanovoDatasetLoader is not yet implemented")

        # -- Load labelled data
        # labelled = pl.read_ipc(labelled_path).to_pandas()
        # labelled.rename({"spectrum_index": "scan"}, axis=1, inplace=True)

        # -- Load MGF
        # raw_mgf = list(mgf.read(open(mgf_path)))
        # for spectrum in raw_mgf:
        #     spectrum["scan"] = int(
        #         spectrum["params"]["title"].replace('"', "").split("scan=")[-1]
        #     )

        # -- Load predictions
        # predictions = mztab.MzTab(predictions_path).spectrum_match_table
        # predictions["index"] = predictions["spectra_ref"].apply(
        #     lambda index: int(index.split("=")[-1])
        # )
        # predictions = predictions.set_index("index", drop=False)
        # predictions["scan"] = predictions["index"].apply(
        #     lambda index: raw_mgf[index]["scan"]
        # )

        # Process peptide sequences
        # predictions["sequence"] = predictions["sequence"].apply(
        #     lambda sequence: sequence.replace("M+15.995", "M(ox)")
        #     .replace("C+57.021", "C")
        #     .replace("N+0.984", "N(+.98)")
        #     .replace("Q+0.984", "Q(+.98)")
        # )

        # Merge labelled data with predictions
        # predictions = pd.merge(labelled, predictions, on="scan")
        # columns = [
        #     "scan",
        #     "mz_array",
        #     "intensity_array",
        #     "charge",
        #     "retention_time",
        #     "precursor_mass",
        #     "Sequence",
        #     "modified_sequence",
        #     "sequence",
        #     "search_engine_score[1]",
        # ]
        # predictions = predictions[columns]
        # predictions.rename(
        #     {
        #         "retention_time": "retention_time",
        #         "Sequence": "peptide",
        #         "sequence": "prediction",
        #         "search_engine_score[1]": "confidence",
        #         "charge": "precursor_charge",
        #     },
        #     axis=1,
        #     inplace=True,
        # )

        # Filter invalid sequences
        # predictions = predictions[
        #     predictions["prediction"].apply(
        #         lambda peptide: not (peptide.startswith("+") or peptide.startswith("-"))
        #     )
        # ]

        # Normalise sequences
        # predictions["peptide"] = predictions["peptide"].apply(
        #     lambda peptide: peptide.replace("L", "I")
        #     if isinstance(peptide, str)
        #     else peptide
        # )
        # predictions["prediction"] = predictions["prediction"].apply(
        #     lambda peptide: peptide.replace("L", "I")
        #     if isinstance(peptide, str)
        #     else peptide
        # )

        # Tokenise sequences
        # predictions["peptide"] = predictions["peptide"].apply(
        #     lambda peptide: re.split(r"(?<=.)(?=[A-Z])", peptide)
        # )
        # predictions["prediction"] = predictions["prediction"].apply(
        #     lambda peptide: re.split(r"(?<=.)(?=[A-Z])", peptide)
        # )

        # Compute match statistics
        # predictions["num_matches"] = predictions.apply(
        #     lambda row: (
        #         metrics._novor_match(row["peptide"], row["prediction"])
        #         if not (
        #             isinstance(row["peptide"], float)
        #             or isinstance(row["prediction"], float)
        #         )
        #         else 0
        #     ),
        #     axis=1,
        # )
        # predictions["correct"] = predictions.apply(
        #     lambda row: (
        #         row["num_matches"] == len(row["peptide"]) == len(row["prediction"])
        #         if not (
        #             isinstance(row["peptide"], float)
        #             or isinstance(row["prediction"], float)
        #         )
        #         else False
        #     ),
        #     axis=1,
        # )
        # return CalibrationDataset(metadata=predictions, predictions=[None] * len(predictions))


class PointNovoDatasetLoader(DatasetLoader):
    """Loader for PointNovo format predictions.

    Note: This loader is not yet implemented.
    """

    def load(self, *args: Path, **kwargs: Any) -> CalibrationDataset:
        """Load a calibration dataset from PointNovo predictions.

        Args:
            *args: Should contain mgf_path and predictions_path in that order
            **kwargs: Not used

        Returns:
            CalibrationDataset: A dataset containing merged spectra and PointNovo predictions.

        Raises:
            NotImplementedError: This loader is not yet implemented.
            ValueError: If incorrect number of positional arguments provided
        """
        if len(args) != 2:
            raise ValueError(
                "Expected exactly 2 positional arguments: mgf_path and predictions_path"
            )
        raise NotImplementedError("PointNovoDatasetLoader is not yet implemented")

        # -- Load MGF file
        # data_dict = defaultdict(list)
        # for spectrum in mgf.read(open(mgf_path)):
        #     data_dict["scan"].append(spectrum["params"]["scans"])
        #     data_dict["peptide"].append(spectrum["params"]["seq"])
        #     data_dict["precursor_charge"].append(float(spectrum["params"]["charge"][0]))
        #     data_dict["precursor_mass"].append(float(spectrum["params"]["pepmass"][0]))
        #     data_dict["retention_time"].append(float(spectrum["params"]["rtinseconds"]))
        #     data_dict["mz_array"].append(spectrum["m/z array"])
        #     data_dict["intensity_array"].append(spectrum["intensity array"])
        # spectra = pd.DataFrame(data=data_dict)

        # Load predictions
        # predictions = pd.read_csv(predictions_path, sep="\t")
        # predictions = predictions[
        #     ["feature_id", "predicted_sequence", "predicted_score"]
        # ]
        # predictions.rename(
        #     {
        #         "feature_id": "scan",
        #         "predicted_sequence": "prediction",
        #         "predicted_score": "confidence",
        #     },
        #     axis=1,
        #     inplace=True,
        # )

        # Merge spectra with predictions
        # dataset = pd.merge(predictions, spectra, how="left", on="scan")

        # Normalise sequences
        # dataset["peptide"] = dataset["peptide"].apply(
        #     lambda peptide: peptide.replace("L", "I")
        #     .replace("M(+15.99)", "M(ox)")
        #     .replace("C(+57.02)", "C")
        #     if isinstance(peptide, str)
        #     else peptide
        # )

        # Tokenise sequences
        # dataset["peptide"] = dataset["peptide"].apply(
        #     lambda peptide: re.split(r"(?<=.)(?=[A-Z])", peptide)
        # )
        # dataset["prediction"] = dataset["prediction"].apply(
        #     lambda peptide: peptide.replace("L", "I")
        #     .replace("C(Carbamidomethylation)", "C")
        #     .replace("N(Deamidation)", "N(+.98)")
        #     .replace("Q(Deamidation)", "Q(+.98)")
        #     if isinstance(peptide, str)
        #     else peptide
        # )
        # dataset["prediction"] = dataset["prediction"].map(
        #     lambda sequence: sequence.split(",") if isinstance(sequence, str) else []
        # )

        # Compute match statistics
        # dataset["num_matches"] = dataset.apply(
        #     lambda row: (
        #         metrics._novor_match(row["peptide"], row["prediction"])
        #         if not (
        #             isinstance(row["peptide"], float)
        #             or isinstance(row["prediction"], float)
        #         )
        #         else 0
        #     ),
        #     axis=1,
        # )
        # dataset["correct"] = dataset.apply(
        #     lambda row: (
        #         row["num_matches"] == len(row["peptide"]) == len(row["prediction"])
        #         if not (
        #             isinstance(row["peptide"], float)
        #             or isinstance(row["prediction"], float)
        #         )
        #         else False
        #     ),
        #     axis=1,
        # )
        # return CalibrationDataset(metadata=dataset, predictions=[None] * len(dataset))


class SavedDatasetLoader(DatasetLoader):
    """Loader for previously saved CalibrationDataset instances."""

    def load(self, *args: Path, **kwargs: Any) -> CalibrationDataset:
        """Load a previously saved CalibrationDataset.

        Args:
            *args: Should contain exactly one argument: data_dir
            **kwargs: Not used

        Returns:
            CalibrationDataset: The loaded dataset.

        Raises:
            ValueError: If incorrect number of positional arguments provided
        """
        if len(args) != 1:
            raise ValueError("Expected exactly 1 positional argument: data_dir")
        data_dir = args[0]

        with (data_dir / "metadata.csv").open(mode="r") as metadata_file:
            metadata = pd.read_csv(metadata_file)
            if "sequence" in metadata.columns:
                metadata["sequence"] = metadata["sequence"].apply(
                    metrics._split_peptide
                )
            metadata["prediction"] = metadata["prediction"].apply(
                metrics._split_peptide
            )
            metadata["mz_array"] = metadata["mz_array"].apply(
                lambda s: ast.literal_eval(s)
                if "," in s
                else ast.literal_eval(
                    re.sub(r"(\n?)(\s+)", ", ", re.sub(r"\[\s+", "[", s))
                )
            )
            metadata["intensity_array"] = metadata["intensity_array"].apply(
                lambda s: ast.literal_eval(s)
                if "," in s
                else ast.literal_eval(
                    re.sub(r"(\n?)(\s+)", ", ", re.sub(r"\[\s+", "[", s))
                )
            )

        predictions_path = data_dir / "predictions.pkl"
        if predictions_path.exists():
            with predictions_path.open(mode="rb") as predictions_file:
                predictions = pickle.load(predictions_file)
        else:
            predictions = None
        return CalibrationDataset(metadata=metadata, predictions=predictions)
