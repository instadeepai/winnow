"""Defines data structures to handle datasets used for model calibration in peptide sequencing tasks.

It integrates metadata, predictions, and spectra data to support various operations such as filtering, merging, and evaluating peptide sequence predictions against spectra.

Classes:
    CalibrationDataset: The main class for storing and processing calibration datasets.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
import ast
from typing import Any, Callable, List, Optional, Tuple
import pickle

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
from pyteomics import mztab, mgf

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
}

RESIDUE_REMAPPING: dict[str, str] = {
    "M(ox)": "M[UNIMOD:35]",  # Oxidation
    "M(+15.99)": "M[UNIMOD:35]",
    "S(p)": "S[UNIMOD:21]",  # Phosphorylation
    "T(p)": "T[UNIMOD:21]",
    "Y(p)": "Y[UNIMOD:21]",
    "S(+79.97)": "S[UNIMOD:21]",
    "T(+79.97)": "T[UNIMOD:21]",
    "Y(+79.97)": "Y[UNIMOD:21]",
    "Q(+0.98)": "Q[UNIMOD:7]",  # Deamidation
    "N(+0.98)": "N[UNIMOD:7]",
    "Q(+.98)": "Q[UNIMOD:7]",
    "N(+.98)": "N[UNIMOD:7]",
    "C(+57.02)": "C[UNIMOD:4]",  # Carboxyamidomethylation
    "(+42.01)": "[UNIMOD:1]",  # Acetylation
    "(+43.01)": "[UNIMOD:5]",  # Carbamylation
    "(-17.03)": "[UNIMOD:385]",  # Loss of ammonia
}

INVALID_PROSIT_TOKENS: list = [
    "\\+25.98",
    "UNIMOD:7",
    "UNIMOD:21",
    "UNIMOD:1",
    "UNIMOD:5",
    "UNIMOD:385",
    # Each C is also treated as Cysteine with carbamidomethylation in Prosit.
]


residue_set = ResidueSet(
    residue_masses=RESIDUE_MASSES, residue_remapping=RESIDUE_REMAPPING
)
metrics = Metrics(residue_set=residue_set, isotope_error_range=[0, 1])


@dataclass
class CalibrationDataset:
    """A class to store and process calibration datasets for peptide sequencing.

    It holds metadata and prediction results and provides various utility methods for loading, processing, filtering, and saving data.
    """

    metadata: pd.DataFrame
    predictions: List[Optional[List[ScoredSequence]]]

    def __post_init__(self):
        """Validate that metadata and predictions have matching lengths."""
        # Allow empty predictions list (no predictions available)
        # But if predictions are provided, they must match metadata length
        if self.predictions and len(self.metadata) != len(self.predictions):
            raise AssertionError("Length of metadata and predictions must match")

    def save(self, data_dir: Path) -> None:
        """Save a `CalibrationDataset` to a directory.

        Args:
            data_dir (Path): Directory to save the dataset. This will contain `metadata.csv` and
                            optionally, `predictions.pkl` for serialized beam search results.
        """
        data_dir.mkdir(parents=True)
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

    @staticmethod
    def _load_dataset(
        predictions_path: Path,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, bool]:
        """Loads a dataset from a CSV file and optionally filters it.

        Args:
            predictions_path (Path): Path to the dataset CSV file.
            apply_filters (bool): Whether to apply filtering to remove invalid rows.

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame, bool]:
                - preds_df: DataFrame with predictions excluding beam-related columns.
                - beam_df: DataFrame with only beam-related columns.
                - has_labels: Boolean indicating if the dataset contains sequence labels.
        """

        def _filter_dataset_for_prosit(df: pl.DataFrame) -> pl.DataFrame:
            """Applies filters to remove unsupported modifications from the first and second beam results."""
            print("Applying dataset filters...")

            # Remove legacy tokens and invalid modifications for Prosit models
            for token in INVALID_PROSIT_TOKENS:
                df = df.filter(~df["preds"].str.contains(token))
                df = df.filter(~df["preds_beam_1"].str.contains(token))

            # Filter out unmodified Cysteine in second beam result for Prosit models.
            # We re-annotate the remaining modified Cysteine as "C" when passing to Prosit during iRT and intensity prediction.
            df = df.filter(~df["preds_tokenised"].str.contains("C,"))

            # Drop rows where 'preds_beam_1' contains 'C' not followed by '[' (i.e, unmodified Cysteine)
            pattern = re.compile(r"C(?!\[)")  # Polars does not yet support lookahead.
            indexes_to_drop = [
                idx
                for idx, row in enumerate(df.iter_rows(named=True))
                if pattern.search(row["preds_beam_1"])
            ]
            df = df.filter(~pl.Series(range(len(df))).is_in(indexes_to_drop))

            return df

        # Read dataset
        df = pl.read_csv(predictions_path)
        has_labels = "sequence" in df.columns

        # Apply filtering
        df = _filter_dataset_for_prosit(df)

        # Split dataset into beam and prediction DataFrames
        beam_df = df.select(cs.contains("_beam_"))
        preds_df = df.select(~cs.contains(["_beam_", "_log_probs_"]))

        return preds_df, beam_df, has_labels

    @staticmethod
    def _process_beams(beam_df: pl.DataFrame) -> List[Optional[List[ScoredSequence]]]:
        """Processes beam predictions by converting them into scored sequences.

        Args:
            beam_df (pl.DataFrame): DataFrame containing beam search predictions.

        Returns:
            List[Optional[List[ScoredSequence]]]: A list of scored sequences for each row.
        """

        def convert_row_to_scored_sequences(
            row: dict,
        ) -> Optional[List[ScoredSequence]]:
            """Converts a row into a list of ScoredSequence objects."""
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

            return scored_sequences or None  # Ensure empty beams become None

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
    def _process_dataset(dataset: pd.DataFrame, has_labels: bool) -> pd.DataFrame:
        """Processes the predictions obtained from saved beams.

        Args:
            dataset (pd.DataFrame): DataFrame containing predictions.

        Returns:
            pd.Dataframe: The processed DataFrame containing predictions.
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
    def _load_spectrum_data(spectrum_path: Path | str) -> pl.DataFrame:
        """Loads spectrum data from either a Parquet or IPC file.

        Args:
            spectrum_path (Path | str): Path to the spectrum data file.

        Returns:
            pl.DataFrame: The loaded spectrum data.
        """
        spectrum_path = Path(spectrum_path)  # Ensure it's a Path object

        if spectrum_path.suffix == ".parquet":
            return pl.read_parquet(spectrum_path)
        elif spectrum_path.suffix == ".ipc":
            return pl.read_ipc(spectrum_path)
        else:
            raise ValueError(f"Unsupported file format: {spectrum_path.suffix}")

    @staticmethod
    def _process_spectrum_data(df: pl.DataFrame, has_labels: bool) -> pd.DataFrame:
        """Processes the input data from the de novo sequencing model.

        Args:
            dataset (pl.DataFrame): DataFrame containing the model input data.
            has_labels (bool): Whether the dataset includes ground truth labels.

        Returns:
            pd.Dataframe: The processed DataFrame containing containing the model input data.
        """
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
        """Merge the input and output data from the de novo sequencing model, using inner join on `spectrum_id`.

        Args:
            beam_dataset (pd.DataFrame): DataFrame containing the predictions.
            spectrum_dataset (pd.DataFrame): DataFrame containing the input data.

        Returns:
            pd.DataFrame: The merged dataset containing both predictions and input metadata.
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
            dataset (pd.DataFrame): The dataset containing sequences and predictions.
            has_labels (bool): Whether the dataset includes ground truth labels.

        Returns:
            pd.DataFrame: The dataset with additional evaluation metrics.
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

    @classmethod
    def from_predictions_csv(
        cls, spectrum_path: Path, beam_predictions_path: Path
    ) -> "CalibrationDataset":
        """Loads a CalibrationDataset from a CSV file, handling both labelled and unlabelled inputs.

        Args:
            spectrum_path (Path): Path to the spectrum data file.
            beam_predictions_path (Path): Path to the CSV file containing beam search predictions.

        Returns:
            CalibrationDataset: An instance of the CalibrationDataset class containing metadata and predictions.
        """
        predictions, beams, has_labels = cls._load_dataset(beam_predictions_path)
        beams = cls._process_beams(beams)
        predictions = cls._process_dataset(predictions.to_pandas(), has_labels)
        inputs = cls._load_spectrum_data(spectrum_path)
        inputs = cls._process_spectrum_data(inputs, has_labels)
        predictions = cls._merge_spectrum_data(predictions, inputs)
        predictions = cls._evaluate_predictions(predictions, has_labels)
        return cls(metadata=predictions, predictions=beams)

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
        raise NotImplementedError
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
            "retention_time",
            "precursor_mass",
            "Sequence",
            "modified_sequence",
            "sequence",
            "search_engine_score[1]",
        ]
        predictions = predictions[columns]
        predictions.rename(
            {
                "retention_time": "retention_time",
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
        raise NotImplementedError
        # -- Load MGF file
        data_dict = defaultdict(list)
        for spectrum in mgf.read(open(mgf_path)):
            data_dict["scan"].append(spectrum["params"]["scans"])
            data_dict["peptide"].append(spectrum["params"]["seq"])
            data_dict["precursor_charge"].append(float(spectrum["params"]["charge"][0]))
            data_dict["precursor_mass"].append(float(spectrum["params"]["pepmass"][0]))
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

    @classmethod
    def load(cls, data_dir: Path) -> "CalibrationDataset":
        """Load `CalibrationDataset` saved using the `save` method from a directory.

        Args:
            data_dir (Path): Path to a directory containing `metadata.csv` and
                            optionally, `predictions.pkl` for serialized beam search results.

        Returns:
            CalibrationDataset: The loaded dataset.
        """
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
            with (data_dir / "predictions.pkl").open(mode="rb") as predictions_file:
                predictions = pickle.load(predictions_file)
        else:
            predictions = None
        return cls(metadata=metadata, predictions=predictions)

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
            CalibrationDataset: A new instance of `CalibrationDataset` containing only the entries for which the conditions specified by the predicates are False.
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
