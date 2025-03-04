"""Defines data structures to handle datasets used for model calibration in peptide sequencing tasks.

It integrates metadata, predictions, and spectra data to support various operations such as filtering, merging, and evaluating peptide sequence predictions against spectra.

Classes:
    CalibrationDataset: The main class for storing and processing calibration datasets.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
from pyteomics import mztab, mgf

from instanovo.utils.metrics import Metrics
from instanovo.inference.beam_search import ScoredSequence


RESIDUE_MASSES: dict[str, float] = {
    "G": 57.021464,
    "A": 71.037114,
    "S": 87.032028,
    "P": 97.052764,
    "V": 99.068414,
    "T": 101.047670,
    #   "C(+57.02)": 160.030649, # 103.009185 + 57.021464
    "C": 160.030649,  # C+57.021 V1
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
    #   "M(+15.99)": 147.035400, # Met oxidation:   131.040485 + 15.994915
    "M(ox)": 147.035400,  # Met oxidation:   131.040485 + 15.994915 V1
    "N(+.98)": 115.026943,  # Asn deamidation: 114.042927 +  0.984016
    "Q(+.98)": 129.042594,  # Gln deamidation: 128.058578 +  0.984016
}


metrics = Metrics(residues=RESIDUE_MASSES, isotope_error_range=[0, 1])


@dataclass
class CalibrationDataset:
    """A class to store and process calibration datasets for peptide sequencing.

    It holds metadata and prediction results and provides various utility methods for loading, processing, filtering, and saving data.
    """

    metadata: pd.DataFrame
    predictions: List[Optional[List[ScoredSequence]]]

    @property
    def confidence_column(self) -> str:
        """Returns the column name that stores confidence scores in the dataset."""
        return "confidence"

    @staticmethod
    def _load_dataset(predictions_path: Path) -> Tuple[pl.DataFrame, pl.DataFrame]:
        def _process_dataset(
            df: pl.DataFrame,
        ) -> pl.DataFrame:  # TODO: remove this method or make its alterations optional
            """Filter out rows with modifications."""
            print(
                "Filtering out predictions containing modifications and invalid precursor charges."
            )
            df = df.filter(~pl.col("preds").str.contains("(", literal=True))
            df = df.filter(~pl.col("targets").str.contains(r"[a-z]"))
            df = df.filter(pl.col("precursor_charge") <= 10)
            return df

        df = pl.read_csv(predictions_path)
        df = _process_dataset(df)
        beam_df = df.select(cs.contains("_beam_"))
        preds_df = df.select(
            ~cs.contains(
                ["_beam_", "_log_probs_"]
            )  # TODO: check what cols are necessary
        )
        return preds_df, beam_df

    @staticmethod
    def _process_beams(beam_df):
        def convert_row_to_scored_sequences(
            row: dict,
        ) -> Optional[List[ScoredSequence]]:
            scored_sequences = []

            for i in range(int(len(row) / 2)):
                seq_col = f"preds_beam_{i}"
                log_prob_col = f"log_probs_beam_{i}"

                sequence = row.get(seq_col)
                log_prob = row.get(log_prob_col, float("-inf"))

                if sequence is not None and log_prob > float("-inf"):
                    scored_sequences.append(
                        ScoredSequence(
                            sequence=sequence, mass_error=None, log_probability=log_prob
                        )
                    )

            return (
                scored_sequences if scored_sequences else None
            )  # Ensure empty beams become None

        beam_df = beam_df.with_columns(
            [
                pl.col(col).str.replace_all("L", "I")
                for col in beam_df.columns
                if "preds_beam" in col
            ]
        )

        # Convert the entire DataFrame
        scored_sequences_list: List[Optional[List[ScoredSequence]]] = [
            convert_row_to_scored_sequences(row)
            for row in beam_df.iter_rows(named=True)
        ]

        return scored_sequences_list

    @staticmethod
    def _process_dataset(dataset: pd.DataFrame, has_labels: bool):
        rename_dict = {"preds": "prediction", "log_probs": "confidence"}
        if has_labels:
            rename_dict["targets"] = "peptide"
        dataset.rename(rename_dict, axis=1, inplace=True)
        dataset.loc[dataset["confidence"] == -1.0, "confidence"] = float("-inf")
        dataset["confidence"] = dataset["confidence"].apply(np.exp)
        if has_labels:
            dataset["peptide"] = dataset["peptide"].apply(
                lambda x: x.replace("L", "I") if isinstance(x, str) else x
            )
        dataset["prediction"] = dataset["prediction"].apply(
            lambda x: x.replace("L", "I") if isinstance(x, str) else x
        )
        dataset["preds_tokenised"] = dataset["preds_tokenised"].apply(
            lambda x: x.replace("L", "I") if isinstance(x, str) else x
        )
        return dataset

    @staticmethod
    def _load_spectrum_data(spectrum_path: Path):
        return pl.read_ipc(spectrum_path)

    @staticmethod
    def _process_spectrum_data(df: pl.DataFrame, has_labels: bool):
        if has_labels:
            df = df.select(  # TODO: check the inclusion/exclusion of columns
                [
                    "Retention time",
                    "Mass",
                    "Modified sequence",
                    "mz_theo",
                    "local_index",
                    "spectrum_index",
                    "global_index",
                    "Mass values",
                    "Intensity",
                ]
            )
        else:
            df = df[  # TODO: check the inclusion/exclusion of columns
                [
                    "Retention time",
                    "Mass",
                    "Modified sequence",
                    "Mass values",
                    "Intensity",
                ]
            ]
        df = df.rename({"Mass values": "mz_array", "Intensity": "intensity_array"})
        return df.to_pandas()

    @staticmethod
    def _merge_spectrum_data(
        dataset: pd.DataFrame, spectrum_dataset: pd.DataFrame, has_labels: bool
    ):
        if has_labels:
            return pd.merge(
                dataset,
                spectrum_dataset,
                on=["spectrum_index", "global_index"],
                how="left",
            )
        else:
            spectrum_dataset = (
                spectrum_dataset[  # TODO: check the inclusion/exclusion of columns
                    ["Retention time", "Mass", "Modified sequence"]
                ]
            )
            return pd.concat(
                [dataset, spectrum_dataset.iloc[0 : len(dataset)]], axis=1
            )  # TODO: patch this better. For now, unlabelled inputs and outputs must contain the same information in the same order.

    @staticmethod
    def _evaluate_predictions(dataset: pd.DataFrame, has_labels: bool):
        if has_labels:
            dataset["peptide"] = dataset["peptide"].apply(metrics._split_peptide)
            dataset["valid_peptide"] = dataset["peptide"].apply(
                lambda x: isinstance(x, list)
            )
        dataset["prediction"] = dataset["prediction"].apply(metrics._split_peptide)
        dataset["valid_prediction"] = dataset["prediction"].apply(
            lambda x: isinstance(x, list)
        )
        if has_labels:
            dataset["num_matches"] = dataset.apply(
                lambda row: metrics._novor_match(row["peptide"], row["prediction"])
                if isinstance(row["peptide"], list)
                and isinstance(row["prediction"], list)
                else 0,
                axis=1,
            )
            dataset["correct"] = dataset.apply(
                lambda row: (
                    row["num_matches"] == len(row["peptide"]) == len(row["prediction"])
                    if isinstance(row["peptide"], list)
                    and isinstance(row["prediction"], list)
                    else False
                ),
                axis=1,
            )
        return dataset

    @classmethod
    def from_predictions_csv(
        cls, spectrum_path: Path, predictions_path: Path
    ) -> "CalibrationDataset":
        """Loads a CalibrationDataset from a CSV file, handling both labelled and unlabelled inputs."""
        # TODO:
        # 1 Read in two input files: IN output and input.
        # 2 Create beam features in expected format to create dataset.predictions
        # 3 Merge the two input files to create dataset.metadata
        # 4 Evaluate merged metadata
        dataset, predictions = cls._load_dataset(predictions_path)
        predictions = cls._process_beams(predictions)
        has_labels = "targets" in dataset.columns
        dataset = cls._process_dataset(dataset.to_pandas(), has_labels)
        spectrum_dataset = cls._load_spectrum_data(spectrum_path)
        spectrum_dataset = cls._process_spectrum_data(spectrum_dataset, has_labels)
        dataset = cls._merge_spectrum_data(dataset, spectrum_dataset, has_labels)
        dataset = cls._evaluate_predictions(dataset, has_labels)
        return cls(metadata=dataset, predictions=predictions)

    @classmethod
    def from_predictions_mztab(
        cls, labelled_path: Path, mgf_path: Path, predictions_path: Path
    ) -> "CalibrationDataset":
        """Load a calibration dataset from an MzTab predictions file.

        Args:
            labelled_path (Path): Path to the labelled dataset in IPC format.
            mgf_path (Path): Path to the MGF file containing spectral data.
            predictions_path (Path): Path to the MzTab predictions file.

        Returns:
            CalibrationDataset: A dataset containing merged labelled data, spectra, and predictions.
        """
        # -- Load labelled data
        labelled = pl.read_ipc(labelled_path).to_pandas()
        labelled.rename({"spectrum_index": "scan"}, axis=1, inplace=True)

        # -- Load MGF
        raw_mgf = list(mgf.read(open(mgf_path)))
        for spectrum in raw_mgf:
            spectrum["scan"] = int(
                spectrum["params"]["title"].replace('"', "").split("scan=")[-1]
            )

        # -- Load predictions
        predictions = mztab.MzTab(predictions_path).spectrum_match_table
        predictions["index"] = predictions["spectra_ref"].apply(
            lambda index: int(index.split("=")[-1])
        )
        predictions = predictions.set_index("index", drop=False)
        predictions["scan"] = predictions["index"].apply(
            lambda index: raw_mgf[index]["scan"]
        )

        # Process peptide sequences
        predictions["sequence"] = predictions["sequence"].apply(
            lambda sequence: sequence.replace("M+15.995", "M(ox)")
            .replace("C+57.021", "C")
            .replace("N+0.984", "N(+.98)")
            .replace("Q+0.984", "Q(+.98)")
        )

        # Merge labelled data with predictions
        predictions = pd.merge(labelled, predictions, on="scan")
        columns = [
            "scan",
            "mz_array",
            "intensity_array",
            "charge",
            "Retention time",
            "Mass",
            "Sequence",
            "modified_sequence",
            "sequence",
            "search_engine_score[1]",
        ]
        predictions = predictions[columns]
        predictions.rename(
            {
                "Retention time": "retention_time",
                "Sequence": "peptide",
                "sequence": "prediction",
                "search_engine_score[1]": "confidence",
                "charge": "precursor_charge",
            },
            axis=1,
            inplace=True,
        )

        # Filter invalid sequences
        predictions = predictions[
            predictions["prediction"].apply(
                lambda peptide: not (peptide.startswith("+") or peptide.startswith("-"))
            )
        ]

        # Normalise sequences
        predictions["peptide"] = predictions["peptide"].apply(
            lambda peptide: peptide.replace("L", "I")
            if isinstance(peptide, str)
            else peptide
        )
        predictions["prediction"] = predictions["prediction"].apply(
            lambda peptide: peptide.replace("L", "I")
            if isinstance(peptide, str)
            else peptide
        )

        # Tokenise sequences
        predictions["peptide"] = predictions["peptide"].apply(
            lambda peptide: re.split(r"(?<=.)(?=[A-Z])", peptide)
        )
        predictions["prediction"] = predictions["prediction"].apply(
            lambda peptide: re.split(r"(?<=.)(?=[A-Z])", peptide)
        )

        # Compute match statistics
        predictions["num_matches"] = predictions.apply(
            lambda row: (
                metrics._novor_match(row["peptide"], row["prediction"])
                if not (
                    isinstance(row["peptide"], float)
                    or isinstance(row["prediction"], float)
                )
                else 0
            ),
            axis=1,
        )
        predictions["correct"] = predictions.apply(
            lambda row: (
                row["num_matches"] == len(row["peptide"]) == len(row["prediction"])
                if not (
                    isinstance(row["peptide"], float)
                    or isinstance(row["prediction"], float)
                )
                else False
            ),
            axis=1,
        )
        return cls(metadata=predictions, predictions=[None] * len(predictions))

    @classmethod
    def from_pointnovo_predictions(
        cls, mgf_path: Path, predictions_path: Path
    ) -> "CalibrationDataset":
        """Load a calibration dataset from PointNovo predictions.

        Args:
            mgf_path (Path): Path to the MGF file containing spectral data.
            predictions_path (Path): Path to the PointNovo predictions file.

        Returns:
            CalibrationDataset: A dataset containing merged spectra and PointNovo predictions.
        """
        # -- Load MGF file
        data_dict = defaultdict(list)
        for spectrum in mgf.read(open(mgf_path)):
            data_dict["scan"].append(spectrum["params"]["scans"])
            data_dict["peptide"].append(spectrum["params"]["seq"])
            data_dict["precursor_charge"].append(float(spectrum["params"]["charge"][0]))
            data_dict["Mass"].append(float(spectrum["params"]["pepmass"][0]))
            data_dict["retention_time"].append(float(spectrum["params"]["rtinseconds"]))
            data_dict["mz_array"].append(spectrum["m/z array"])
            data_dict["intensity_array"].append(spectrum["intensity array"])
        spectra = pd.DataFrame(data=data_dict)

        # Load predictions
        predictions = pd.read_csv(predictions_path, sep="\t")

        predictions = predictions[
            ["feature_id", "predicted_sequence", "predicted_score"]
        ]
        predictions.rename(
            {
                "feature_id": "scan",
                "predicted_sequence": "prediction",
                "predicted_score": "confidence",
            },
            axis=1,
            inplace=True,
        )

        # Merge spectra with predictions
        dataset = pd.merge(predictions, spectra, how="left", on="scan")

        # Normalise sequences
        dataset["peptide"] = dataset["peptide"].apply(
            lambda peptide: peptide.replace("L", "I")
            .replace("M(+15.99)", "M(ox)")
            .replace("C(+57.02)", "C")
            if isinstance(peptide, str)
            else peptide
        )

        # Tokenise sequences
        dataset["peptide"] = dataset["peptide"].apply(
            lambda peptide: re.split(r"(?<=.)(?=[A-Z])", peptide)
        )
        dataset["prediction"] = dataset["prediction"].apply(
            lambda peptide: peptide.replace("L", "I")
            .replace("C(Carbamidomethylation)", "C")
            .replace("N(Deamidation)", "N(+.98)")
            .replace("Q(Deamidation)", "Q(+.98)")
            if isinstance(peptide, str)
            else peptide
        )
        dataset["prediction"] = dataset["prediction"].map(
            lambda sequence: sequence.split(",") if isinstance(sequence, str) else []
        )

        # Compute match statistics
        dataset["num_matches"] = dataset.apply(
            lambda row: (
                metrics._novor_match(row["peptide"], row["prediction"])
                if not (
                    isinstance(row["peptide"], float)
                    or isinstance(row["prediction"], float)
                )
                else 0
            ),
            axis=1,
        )
        dataset["correct"] = dataset.apply(
            lambda row: (
                row["num_matches"] == len(row["peptide"]) == len(row["prediction"])
                if not (
                    isinstance(row["peptide"], float)
                    or isinstance(row["prediction"], float)
                )
                else False
            ),
            axis=1,
        )
        return cls(metadata=dataset, predictions=[None] * len(dataset))

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

    def to_csv(self, path: str) -> None:
        """Saves the dataset metadata to a CSV file.

        Args:
            path (str): Path to the output CSV file.
        """
        self.metadata.to_csv(path)

    def __getitem__(self, index) -> Tuple[pd.Series, List[ScoredSequence]]:
        """Retrieves a metadata row and its corresponding prediction.

        Args:
            index (int): Index of the desired entry.

        Returns:
            Tuple[pd.Series, List[ScoredSequence]]: The metadata row and its associated predictions.
        """
        return self.metadata.iloc[index], self.predictions[index]

    def __len__(self) -> int:
        """Returns the number of entries in the dataset."""
        assert self.metadata.shape[0] == len(self.predictions)
        return len(self.predictions)
