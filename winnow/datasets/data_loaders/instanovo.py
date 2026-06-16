"""InstaNovo CSV dataset loader."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
from instanovo.utils.metrics import Metrics
from instanovo.utils.residues import ResidueSet

if TYPE_CHECKING:
    from matchms import Spectrum

from winnow.compat.instanovo import ScoredSequence
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.datasets.interfaces import DatasetLoader


class InstaNovoDatasetLoader(DatasetLoader):
    """Loader for InstaNovo predictions in CSV format."""

    _DEFAULT_COLUMN_MAPPING: dict[str, str] = {
        "predictions": "predictions",
        "predictions_tokenised": "predictions_tokenised",
        "log_probability": "log_probs",
    }

    def __init__(
        self,
        residue_masses: dict[str, float],
        residue_remapping: Optional[dict[str, str]] = None,
        isotope_error_range: Tuple[int, int] = (0, 1),
        beam_columns: Optional[dict[str, str]] = None,
        add_index_cols: bool = False,
        column_mapping: Optional[dict[str, str]] = None,
    ) -> None:
        """Initialise the InstaNovoDatasetLoader.

        Args:
            residue_masses: The mapping of residues to their masses (ProForma notation).
            residue_remapping: The mapping of input notations to ProForma notation.
            isotope_error_range: The range of isotope errors to consider when matching peptides.
            beam_columns: The names of the beam columns to substring match in the predictions file.
            add_index_cols: If True, add ``experiment_name`` and ``spectrum_id`` to parquet/ipc
                inputs. MGF inputs always get these columns regardless of this flag.
            column_mapping: Mapping from logical column names (``predictions``,
                ``predictions_tokenised``, ``log_probability``) to the actual CSV column
                names produced by the InstaNovo version you are loading.  Defaults are
                ``{"predictions": "predictions", "predictions_tokenised":
                "predictions_tokenised", "log_probability": "log_probs"}``.
        """
        self.metrics = Metrics(
            residue_set=ResidueSet(
                residue_masses=residue_masses, residue_remapping=residue_remapping
            ),
            isotope_error_range=isotope_error_range,
        )
        self.beam_columns = beam_columns
        self.add_index_cols = add_index_cols
        self.column_mapping = {
            **self._DEFAULT_COLUMN_MAPPING,
            **(column_mapping or {}),
        }

    @staticmethod
    def _df_from_matchms(spectra: list[Spectrum]) -> pl.DataFrame:
        """Convert a list of Matchms spectra to a polars DataFrame.

        Includes only metadata columns that matchms exposes for at least one spectrum.
        ``scan_number`` is always a 0-based enumerate index.

        Args:
            spectra: List of Matchms spectrum objects.

        Returns:
            The polars DataFrame.
        """
        metadata_map = {
            "precursor_mz": "precursor_mz",
            "charge": "precursor_charge",
            "retention_time": "retention_time",
        }
        sequence_keys = ("seq", "peptide_sequence")

        all_metadata_keys: set[str] = set()
        for spectrum in spectra:
            all_metadata_keys.update(spectrum.metadata.keys())

        active_columns = {
            mgf_key: col_name
            for mgf_key, col_name in metadata_map.items()
            if mgf_key in all_metadata_keys
        }

        sequence_key = next((k for k in sequence_keys if k in all_metadata_keys), None)

        data: dict[str, list[Any]] = {"scan_number": []}
        for col_name in active_columns.values():
            data[col_name] = []
        if sequence_key:
            data["sequence"] = []
        data["mz_array"] = []
        data["intensity_array"] = []

        for i, spectrum in enumerate(spectra):
            data["scan_number"].append(i)
            for mgf_key, col_name in active_columns.items():
                data[col_name].append(spectrum.metadata.get(mgf_key))
            if sequence_key:
                data["sequence"].append(spectrum.metadata.get(sequence_key))
            data["mz_array"].append(spectrum.peaks.mz)
            data["intensity_array"].append(spectrum.peaks.intensities)

        return pl.DataFrame(data)

    @staticmethod
    def _add_index_cols(df: pl.DataFrame, fp: Path | str) -> pl.DataFrame:
        """Add ``experiment_name`` and ``spectrum_id`` to align with InstaNovo CSV output.

        If ``scan_number`` is present, ``spectrum_id`` is ``experiment_name:scan_number``.
        Otherwise uses a row index, matching InstaNovo's data_handler fallback.
        """
        exp_name = Path(fp).stem
        df = df.with_columns(pl.lit(exp_name).alias("experiment_name").cast(pl.Utf8))
        if "scan_number" in df.columns:
            df = df.with_columns(
                (
                    pl.col("experiment_name")
                    + ":"
                    + pl.col("scan_number").cast(pl.Utf8)
                ).alias("spectrum_id")
            )
        else:
            df = df.with_row_index("idx")
            df = df.with_columns(
                (pl.col("experiment_name") + ":" + pl.col("idx").cast(pl.Utf8)).alias(
                    "spectrum_id"
                )
            )
            df = df.drop("idx")
        return df

    @staticmethod
    def _merge_spectrum_data(
        preds_dataset: pd.DataFrame, spectrum_dataset: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge the input and output data from the de novo sequencing model.

        Args:
            preds_dataset (pd.DataFrame): The dataframe containing the beam predictions.
            spectrum_dataset (pd.DataFrame): The dataframe containing the spectrum data.

        Returns:
            pd.DataFrame: The merged dataframe.
        """
        # Merge the predictions and input datasets on the spectrum_id column
        # There should be no duplicate columns between the two datasets except for spectrum_id
        merged_dataset = pd.merge(
            preds_dataset,
            spectrum_dataset,
            on=["spectrum_id"],
            suffixes=("_from_preds", "_from_inputs"),
            how="inner",
        )

        # If the number of rows in the merged dataset is not equal to the number of rows in the predictions dataset, it is likely that spectrum_id is not unique in the input dataset.
        # It is possible for the inputs dataset to have more rows than the predictions dataset if the DNS model filtered out some spectra before prediction.
        if len(merged_dataset) != len(preds_dataset):
            raise ValueError(
                f"Merge conflict: Expected {len(preds_dataset)} rows, but got {len(merged_dataset)}."
            )

        return merged_dataset

    @staticmethod
    def _load_predictions_without_beams(predictions_path: Path | str) -> pl.DataFrame:
        """Load predictions CSV without parsing beam columns.

        Used when beam_columns is None (beam predictions disabled).

        Args:
            predictions_path: Path to the CSV file containing predictions.

        Returns:
            pl.DataFrame: The predictions dataframe.

        Raises:
            ValueError: If the file format is not CSV.
        """
        predictions_path = Path(predictions_path)
        if predictions_path.suffix != ".csv":
            raise ValueError(
                f"Unsupported file format for InstaNovo predictions: {predictions_path.suffix}. "
                "Supported format is .csv."
            )
        return pl.read_csv(predictions_path)

    def load(
        self, *, data_path: Path, predictions_path: Optional[Path] = None, **kwargs: Any
    ) -> CalibrationDataset:
        """Load a CalibrationDataset from InstaNovo CSV predictions.

        Args:
            data_path: Path to the spectrum data file
            predictions_path: Path to the predictions CSV file
            **kwargs: Not used

        Returns:
            CalibrationDataset: An instance of the CalibrationDataset class containing
                metadata and optionally beam predictions (if beam_columns is configured).

        Raises:
            ValueError: If predictions_path is None
        """
        if predictions_path is None:
            raise ValueError("predictions_path is required for InstaNovoDatasetLoader")

        inputs, has_labels = self._load_spectrum_data(data_path)
        inputs = self._process_spectrum_data(inputs, has_labels)

        # Load beam predictions only if beam_columns is configured
        if self.beam_columns:
            predictions_df, beams_df = self._load_beam_preds(predictions_path)
            beams = self._process_beams(beams_df)
        else:
            predictions_df = self._load_predictions_without_beams(predictions_path)
            beams = None

        predictions = self._process_predictions(
            predictions_df.to_pandas(), inputs.columns
        )
        predictions = self._merge_spectrum_data(predictions, inputs)
        predictions = self._evaluate_predictions(predictions, has_labels)

        return CalibrationDataset(metadata=predictions, predictions=beams)

    def _load_spectrum_data(
        self, spectrum_path: Path | str
    ) -> Tuple[pl.DataFrame, bool]:
        """Loads spectrum data from either a Parquet, IPC or MGF file.

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
        elif spectrum_path.suffix == ".mgf":
            from matchms.importing import load_from_mgf

            spectra = list(load_from_mgf(str(spectrum_path)))
            df = self._df_from_matchms(spectra)
        else:
            raise ValueError(
                f"Unsupported file format for spectrum data: {spectrum_path.suffix}. Supported formats are .parquet, .ipc and .mgf."
            )

        if spectrum_path.suffix == ".mgf" or self.add_index_cols:
            df = self._add_index_cols(df, spectrum_path)

        if "sequence" in df.columns:
            has_labels = True
        else:
            has_labels = False

        return df, has_labels

    def _validate_beam_columns(self, columns: List[str]) -> None:
        """Validate that each beam column prefix matches at least one column in the dataframe.

        Each prefix in self.beam_columns must match one or more columns in the form
        <prefix><beam_index>, where <beam_index> is a non-negative integer
        (e.g., predictions_beam_0, predictions_beam_1).

        Args:
            columns: List of column names from the predictions dataframe.

        Raises:
            ValueError: If any beam column prefix does not match at least one column.
                The error message lists all missing prefixes and the available columns.
        """
        assert self.beam_columns is not None  # to pass type checking

        missing_prefixes = []
        for prefix in self.beam_columns.values():
            # Pattern: exact prefix followed by one or more digits at end of column name
            pattern = re.compile(rf"^{re.escape(prefix)}\d+$")
            if not any(pattern.match(col) for col in columns):
                missing_prefixes.append(prefix)
        if missing_prefixes:
            raise ValueError(
                f"Cannot find columns matching the following beam column prefixes in predictions file: {missing_prefixes}. \n"
                f"Expected column names of the form '<prefix><beam_index>' (e.g., '{missing_prefixes[0]}0'). \n"
                f"Available columns: {columns}\n"
            )

    def _load_beam_preds(
        self,
        predictions_path: Path | str,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Load beam predictions from a CSV file and split into predictions and beams dataframes.

        Args:
            predictions_path: Path to the CSV file containing the predictions.

        Returns:
            A tuple of (predictions_df, beams_df) where beams_df contains only the
            beam-indexed columns and predictions_df contains the remaining columns.

        Raises:
            ValueError: If the file format is not CSV or if beam column validation fails.
        """
        predictions_path = Path(predictions_path)
        if predictions_path.suffix != ".csv":
            raise ValueError(
                f"Unsupported file format for InstaNovo beam predictions: {predictions_path.suffix}. Supported format is .csv."
            )
        df = pl.read_csv(predictions_path)

        self._validate_beam_columns(df.columns)
        assert self.beam_columns is not None  # to pass type checking

        # Use polars column selectors to split dataframe
        beam_df = df.select(cs.contains(*self.beam_columns.values()))
        preds_df = df.select(~cs.contains([*self.beam_columns.values()]))
        return preds_df, beam_df

    def _process_beams(
        self,
        beam_df: pl.DataFrame,
    ) -> List[Optional[List[ScoredSequence]]]:
        """Processes beam predictions into scored sequences.

        Args:
            beam_df (pl.DataFrame): The dataframe containing the beam predictions.

        Returns:
            List[Optional[List[ScoredSequence]]]: A list of scored sequences for each row in the dataframe.
        """
        assert self.beam_columns is not None, (
            "beam_columns must be set"
        )  # to pass type checking

        def convert_row_to_scored_sequences(
            row: dict,
        ) -> Optional[List[ScoredSequence]]:
            assert self.beam_columns is not None, (
                "beam_columns must be set"
            )  # to pass type checking

            scored_sequences = []
            num_beams = len(row) // len(self.beam_columns)

            for beam in range(num_beams):
                seq_col, log_prob_col, token_log_prob_col = (
                    f"{self.beam_columns['sequence']}{beam}",
                    f"{self.beam_columns['log_probability']}{beam}",
                    f"{self.beam_columns['token_log_probabilities']}{beam}",
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
                            token_log_probabilities=ast.literal_eval(token_log_prob)
                            if isinstance(token_log_prob, str)
                            else token_log_prob,
                        )
                    )

            return scored_sequences or None

        # Apply L -> I transformation to multiple columns using polars with_columns
        beam_df = beam_df.with_columns(
            [
                pl.col(col).str.replace_all("L", "I")
                for col in beam_df.columns
                if self.beam_columns["sequence"] in col
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
                    lambda peptide: (
                        peptide.replace("L", "I")
                        if isinstance(peptide, str)
                        else peptide
                    )
                )
                .apply(self.metrics._split_peptide)
            )
        return df

    def _process_predictions(
        self, preds_dataset: pd.DataFrame, input_dataset_columns: List[str]
    ) -> pd.DataFrame:
        """Processes the predictions obtained from saved beams.

        Args:
            preds_dataset (pd.DataFrame): The dataframe containing the predictions.
            input_dataset_columns (List[str]): The columns of the input dataset.

        Returns:
            pd.DataFrame: The processed dataframe.
        """
        # Drop duplicate columns from the input dataset except spectrum_id
        preds_dataset = preds_dataset.drop(
            columns=[
                col
                for col in preds_dataset.columns
                if col in input_dataset_columns and col != "spectrum_id"
            ]
        )

        rename_dict = {
            self.column_mapping["predictions"]: "prediction_untokenised",
            self.column_mapping["predictions_tokenised"]: "prediction",
            self.column_mapping["log_probability"]: "confidence",
        }
        missing_cols = [
            col for col in rename_dict.keys() if col not in preds_dataset.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Required columns {missing_cols} not found in predictions dataset. "
                f"If you are using an older InstaNovo version, set column_mapping in "
                f"the data_loader config to match the CSV headers."
            )
        preds_dataset.rename(rename_dict, axis=1, inplace=True)

        preds_dataset["confidence"] = preds_dataset["confidence"].apply(np.exp)

        preds_dataset["prediction"] = preds_dataset["prediction"].apply(
            lambda peptide: peptide.split(", ") if isinstance(peptide, str) else peptide
        )
        preds_dataset["prediction"] = preds_dataset["prediction"].apply(
            lambda peptide: (
                ["I" if amino_acid == "L" else amino_acid for amino_acid in peptide]
                if isinstance(peptide, list)
                else peptide
            )
        )

        preds_dataset["prediction_untokenised"] = preds_dataset[
            "prediction_untokenised"
        ].apply(
            lambda peptide: (
                peptide.replace("L", "I") if isinstance(peptide, str) else peptide
            )
        )

        return preds_dataset

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
            dataset["valid_sequence"] = dataset["sequence"].apply(
                lambda peptide: isinstance(peptide, list)
            )
        dataset["valid_prediction"] = dataset["prediction"].apply(
            lambda peptide: isinstance(peptide, list)
        )
        if has_labels:
            dataset["num_matches"] = dataset.apply(
                lambda row: (
                    self.metrics._novor_match(row["sequence"], row["prediction"])
                    if isinstance(row["sequence"], list)
                    and isinstance(row["prediction"], list)
                    else 0
                ),
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
