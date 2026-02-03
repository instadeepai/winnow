from abc import ABCMeta, abstractmethod
import bisect
from math import exp, isnan
from typing import Dict, List, Tuple, Iterator, Optional
import warnings

import pandas as pd
import numpy as np
from numpy import median
from scipy.stats import entropy
from sklearn.neural_network import MLPRegressor

import koinapy

from winnow.datasets.calibration_dataset import CalibrationDataset


def _raise_value_error(value, name: str):
    """Raise a ValueError if the given value is None."""
    if value is None:
        raise ValueError(f"{name} cannot be None")


class FeatureDependency(metaclass=ABCMeta):
    """Base class for common dependencies of several features."""

    @property
    @abstractmethod
    def name(self) -> str:
        """A natural language identifier for the feature.

        This is used as the key for the feature in the `ProbabilityCalibrator`
        class and the name of the feature in the feature dataframe.

        Returns:
            str: The feature identifier
        """

    @abstractmethod
    def compute(
        self,
        dataset: CalibrationDataset,
    ) -> Dict[str, pd.Series]:
        """Compute the feature dependencies based on the given dataset.

        Args:
            dataset (CalibrationDataset): The dataset containing metadata and predictions.

        Returns:
            Dict[str, pd.Series]: A dictionary mapping feature dependency names to computed values.
        """
        pass


class CalibrationFeatures(metaclass=ABCMeta):
    """The abstract interface for features for the calibration classifier."""

    irt_predictor: Optional[MLPRegressor] = None

    @property
    @abstractmethod
    def dependencies(self) -> List[FeatureDependency]:
        """A list of dependencies that need to be computed before a feature.

        Returns:
            List[FeatureDependency]:
                The list of dependencies.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """A natural language identifier for the feature.

        This is used as the key for the feature in the `ProbabilityCalibrator` class and the name of the feature in the feature dataframe.

        Returns:
            str: The feature identifier.
        """

    @property
    @abstractmethod
    def columns(self) -> List[str]:
        """A natural language identifier for the feature.

        This is used as the key for the feature in the `ProbabilityCalibrator`
        class and the name of the feature in the feature dataframe.

        Returns:
            str: The feature identifier
        """

    @abstractmethod
    def prepare(self, dataset: CalibrationDataset) -> None:
        """Prepare the dataset for feature computation.

        This method is intended to perform any preprocessing required before computing the feature.

        Args:
            dataset (CalibrationDataset): The dataset to prepare for feature computation.
        """
        pass

    @abstractmethod
    def compute(self, dataset: CalibrationDataset) -> None:
        """Compute the feature for the dataset.

        This method calculates the feature values based on the dataset and its dependencies.

        Args:
            dataset (CalibrationDataset): The dataset on which to compute the feature.
        """
        pass


def find_matching_ions(
    source_mz: List[float],
    target_mz: List[float],
    target_intensities: List[float],
    mz_tolerance: float,
) -> Tuple[float, float]:
    """Finds the matching ions between source and target spectra based on mass-to-charge ratio (m/z).

    Identifies ions from the source spectrum that match ions in the target spectrum within a specified mass tolerance.

    Args:
        source_mz (List[float]): List of m/z values from the source spectrum.
        target_mz (List[float]): List of m/z values from the target spectrum.
        target_intensities (List[float]): List of intensities corresponding to the target m/z values.
        mz_tolerance (float): Tolerance for matching m/z values between source and target spectra.

    Returns:
        Tuple[float, float]: The fraction of matched ions and the average intensity of matched ions.
    """
    if isinstance(source_mz, float) and isnan(source_mz):
        return 0.0, 0.0
    num_matches, match_intensity = 0, 0.0
    for ion_mz in source_mz:
        nearest = bisect.bisect_left(target_mz, ion_mz)
        if nearest < len(target_mz):
            if target_mz[nearest] - ion_mz < mz_tolerance:
                num_matches += 1
                match_intensity += target_intensities[nearest]
                continue
        if nearest > 0:
            if ion_mz - target_mz[nearest - 1] < mz_tolerance:
                num_matches += 1
                match_intensity += target_intensities[nearest - 1]
    return num_matches / len(source_mz), match_intensity / sum(target_intensities)


def compute_ion_identifications(
    dataset: pd.DataFrame, source_column: str, mz_tolerance: float
) -> Iterator[Tuple[List[float], List[float]]]:
    """Computes the ion match rate and match intensity for each spectrum in the dataset.

    Finds how well the theoretical ions (from the `source_column`) match the experimental ions in the dataset.

    Args:
        dataset (pd.DataFrame): DataFrame containing the mass spectrum data.
        source_column (str): Column name containing the theoretical m/z values.
        mz_tolerance (float): Mass tolerance used to match ions.

    Returns:
        Tuple: A tuple containing two lists:
            - A list of the ion match rates.
            - A list of the average match intensities.
    """
    matches = [
        find_matching_ions(
            source_mz=row[source_column],
            target_mz=row["mz_array"],
            target_intensities=row["intensity_array"],
            mz_tolerance=mz_tolerance,
        )
        for _, row in dataset.iterrows()
    ]
    return zip(*matches)


class PrositFeatures(CalibrationFeatures):
    """A class for extracting features related to Prosit: a machine learning-based intensity prediction tool for peptide fragmentation."""

    def __init__(
        self,
        mz_tolerance: float,
        invalid_prosit_tokens: List[str],
        learn_from_missing: bool = True,
        prosit_intensity_model_name: str = "Prosit_2020_intensity_HCD",
    ) -> None:
        """Initialize PrositFeatures.

        Args:
            mz_tolerance (float): The mass-to-charge tolerance for ion matching.
            invalid_prosit_tokens (List[str]): The tokens to consider as invalid for Prosit intensity prediction.
            learn_from_missing (bool): Whether to learn from missing data by including a missingness indicator column.
                If False, an error will be raised when invalid spectra are encountered.
                Defaults to True.
            prosit_intensity_model_name (str): The name of the Prosit intensity model to use.
                Defaults to "Prosit_2020_intensity_HCD".
        """
        self.mz_tolerance = mz_tolerance
        self.invalid_prosit_tokens = invalid_prosit_tokens
        self.learn_from_missing = learn_from_missing
        self.prosit_intensity_model_name = prosit_intensity_model_name

    @property
    def dependencies(self) -> List[FeatureDependency]:
        """Returns the list of feature dependencies for the PrositFeatures.

        Since no dependencies are required, this method returns an empty list.

        Returns:
            List[FeatureDependency]: An empty list, as there are no dependencies for Prosit features.
        """
        return []

    @property
    def name(self) -> str:
        """Returns the name of the feature.

        This method provides the natural language identifier used as the key for the feature.

        Returns:
            str: The feature identifier, "Prosit Features".
        """
        return "Prosit Features"

    @property
    def columns(self) -> List[str]:
        """Returns the columns associated with the Prosit features.

        The columns include ion matches and the corresponding ion match intensities.

        Returns:
            List[str]: A list of column names: ["ion_matches", "ion_match_intensity"] and optionally
                "is_missing_prosit_features" if learn_from_missing is True.
        """
        columns = ["ion_matches", "ion_match_intensity"]
        if self.learn_from_missing:
            columns.append("is_missing_prosit_features")
        return columns

    def check_valid_prosit_prediction(self, dataset: CalibrationDataset) -> pd.Series:
        """Check which predictions are valid for Prosit intensity prediction.

        Args:
            dataset (CalibrationDataset): The dataset to check.

        Returns:
            pd.Series: A series of booleans indicating whether the prediction is valid for Prosit intensity prediction.
        """
        # Filter out invalid spectra for Prosit intensity prediction
        filtered_dataset = (
            dataset.filter_entries(
                metadata_predicate=lambda row: row["precursor_charge"] > 6
            )
            .filter_entries(metadata_predicate=lambda row: len(row["prediction"]) > 30)
            .filter_entries(
                metadata_predicate=lambda row: (
                    any(
                        token in row["prediction_untokenised"]
                        for token in self.invalid_prosit_tokens
                    )
                )
            )
            .filter_entries(
                metadata_predicate=lambda row: (
                    any(token == "C" for token in row["prediction"])
                )
            )
        )

        # Obtain valid indices
        valid_spectrum_ids = filtered_dataset.metadata["spectrum_id"]

        # Create boolean series indicating whether the prediction is valid for Prosit intensity prediction
        is_valid_prosit_prediction = pd.Series(
            dataset.metadata["spectrum_id"].isin(valid_spectrum_ids),
        )

        return is_valid_prosit_prediction

    def prepare(self, dataset: CalibrationDataset) -> None:
        """Prepares the dataset for feature computation.

        This method is intended to perform any preprocessing required before computing the feature.

        Args:
            dataset (CalibrationDataset): The dataset to prepare.
        """
        return

    def compute(self, dataset: CalibrationDataset) -> None:
        """Computes the Prosit features for the given dataset.

        Uses the Prosit model to predict intensities based on peptide sequences, precursor charges and collision energies. Processes and sorts predictions, aligns ion match intensities and mass-to-charge ratios (m/z), and stores the results in the dataset metadata.

        Args:
            dataset (CalibrationDataset): The dataset containing metadata required for predictions.

        Raises:
            ValueError: If learn_from_missing is False and invalid spectra are found in the dataset.
        """
        # Check which predictions are valid for Prosit intensity prediction
        is_valid_prosit_prediction = self.check_valid_prosit_prediction(dataset)
        dataset.metadata["is_missing_prosit_features"] = ~is_valid_prosit_prediction

        # If not learning from missing data, raise error when invalid spectra are found
        if not self.learn_from_missing:
            n_invalid = (~is_valid_prosit_prediction).sum()
            if n_invalid > 0:
                raise ValueError(
                    f"Found {n_invalid} spectra with missing Prosit features. "
                    f"When learn_from_missing=False, all spectra must be valid for Prosit prediction. "
                    f"Please filter your dataset to remove:\n"
                    f"  - Peptides longer than 30 amino acids\n"
                    f"  - Precursor charges greater than 6\n"
                    f"  - Peptides with unsupported modifications (e.g., {', '.join(self.invalid_prosit_tokens[:3])}...)\n"
                    f"Or set learn_from_missing=True to handle missing data automatically."
                )

        original_indices = dataset.metadata.index

        # Filter out invalid spectra for Prosit intensity prediction
        valid_prosit_input = dataset.filter_entries(
            metadata_predicate=lambda row: row["is_missing_prosit_features"]
        )

        # Prepare input data
        inputs = pd.DataFrame()
        inputs["peptide_sequences"] = np.array(
            ["".join(peptide) for peptide in valid_prosit_input.metadata["prediction"]]
        )
        inputs["precursor_charges"] = np.array(
            valid_prosit_input.metadata["precursor_charge"]
        )
        inputs["collision_energies"] = np.array(len(valid_prosit_input.metadata) * [25])
        inputs.index = valid_prosit_input.metadata["spectrum_id"]

        model = koinapy.Koina(self.prosit_intensity_model_name)
        predictions: pd.DataFrame = model.predict(inputs)

        # Group predictions by spectrum_id to get one row per peptide
        # We make a temporary column spectrum_id_col to enable grouping by spectrum_id,
        # and we name this spectrum_id_col to avoid naming conflicts with the index
        predictions["spectrum_id_col"] = predictions.index
        grouped_predictions = predictions.groupby(by="spectrum_id_col").agg(
            {
                "peptide_sequences": "first",
                "precursor_charges": "first",
                "collision_energies": "first",
                "intensities": list,
                "mz": list,
                "annotation": list,
            }
        )
        # Sort intensities by m/z to match experimental data
        grouped_predictions["intensities"] = grouped_predictions.apply(
            lambda row: np.array(row["intensities"])[np.argsort(row["mz"])].tolist(),
            axis=1,
        )
        # Sort annotations by m/z to match experimental data
        grouped_predictions["annotation"] = grouped_predictions.apply(
            lambda row: np.array(row["annotation"])[np.argsort(row["mz"])].tolist(),
            axis=1,
        )
        # Sort m/z values to match experimental data
        grouped_predictions["mz"] = grouped_predictions["mz"].apply(np.sort)

        # Match computed metadata to valid spectra and impute missing values for invalid spectra
        # i.e., if is_missing_prosit_features is True, then prosit_mz and prosit_intensity are NaN
        dataset.metadata.index = dataset.metadata["spectrum_id"]
        dataset.metadata["prosit_mz"] = grouped_predictions["mz"].reindex(
            dataset.metadata["spectrum_id"],
            fill_value=np.nan,
        )
        dataset.metadata["prosit_intensity"] = grouped_predictions[
            "intensities"
        ].reindex(dataset.metadata["spectrum_id"], fill_value=np.nan)

        # Revert to original indices
        dataset.metadata.index = original_indices

        # Compute ion matches and match intensity
        # Zeros are returned for rows with missing Prosit-predicted spectra
        ion_matches, match_intensity = compute_ion_identifications(
            dataset=dataset.metadata,
            source_column="prosit_mz",
            mz_tolerance=self.mz_tolerance,
        )

        dataset.metadata["ion_matches"] = ion_matches
        dataset.metadata["ion_match_intensity"] = match_intensity


class ChimericFeatures(CalibrationFeatures):
    """Computes chimeric features for calibration.

    This class predicts ion intensities for runner-up peptide sequences using the Prosit model
    and computes chimeric ion matches based on mass-to-charge ratios (m/z). The computed features
    are stored in the dataset metadata.
    """

    def __init__(
        self,
        mz_tolerance: float,
        invalid_prosit_tokens: List[str],
        learn_from_missing: bool = True,
        prosit_intensity_model_name: str = "Prosit_2020_intensity_HCD",
    ) -> None:
        """Initialize ChimericFeatures.

        Args:
            mz_tolerance (float): The mass-to-charge tolerance for ion matching.
            invalid_prosit_tokens (List[str]): The tokens to consider as invalid for Prosit intensity prediction.
            learn_from_missing (bool): Whether to learn from missing data by including a missingness indicator column.
                If False, an error will be raised when invalid spectra are encountered.
                Defaults to True.
            prosit_intensity_model_name (str): The name of the Prosit intensity model to use.
                Defaults to "Prosit_2020_intensity_HCD".
        """
        self.mz_tolerance = mz_tolerance
        self.learn_from_missing = learn_from_missing
        self.invalid_prosit_tokens = invalid_prosit_tokens
        self.prosit_intensity_model_name = prosit_intensity_model_name

    @property
    def dependencies(self) -> List[FeatureDependency]:
        """Returns a list of dependencies required before computing the feature.

        Since this feature does not depend on other features, it returns an empty list.

        Returns:
            List[FeatureDependency]: An empty list.
        """
        return []

    @property
    def name(self) -> str:
        """Returns the name of the feature.

        This method provides the natural language identifier used as the key for the feature.

        Returns:
            str: The feature name.
        """
        return "Chimeric Features"

    @property
    def columns(self) -> List[str]:
        """Returns the column names for the computed features.

        Returns:
            List[str]: A list of column names: ["chimeric_ion_matches", "chimeric_ion_match_intensity"]
                and optionally "is_missing_chimeric_features" if learn_from_missing is True.
        """
        columns = [
            "chimeric_ion_matches",
            "chimeric_ion_match_intensity",
        ]
        if self.learn_from_missing:
            columns.append("is_missing_chimeric_features")
        return columns

    def check_valid_chimeric_prosit_prediction(
        self, dataset: CalibrationDataset
    ) -> pd.Series:
        """Check which predictions are valid for chimeric Prosit intensity prediction.

        Args:
            dataset (CalibrationDataset): The dataset to check.

        Returns:
            pd.Series: A series of booleans indicating whether the prediction is valid for chimeric Prosit intensity prediction.
        """
        # Filter out invalid spectra for chimeric Prosit intensity prediction
        filtered_dataset = (
            dataset.filter_entries(predictions_predicate=lambda beam: len(beam) < 2)
            .filter_entries(metadata_predicate=lambda row: row["precursor_charge"] > 6)
            .filter_entries(
                predictions_predicate=lambda beam: len(beam) > 1
                and len(beam[1].sequence) > 30
            )
            .filter_entries(
                predictions_predicate=lambda beam: (
                    len(beam) > 1
                    and any(
                        token in "".join(beam[1].sequence)
                        for token in self.invalid_prosit_tokens
                    )
                )
            )
            .filter_entries(
                predictions_predicate=lambda beam: (
                    len(beam) > 1 and any(token == "C" for token in beam[1].sequence)
                )
            )
        )

        # Obtain valid indices
        valid_spectrum_ids = filtered_dataset.metadata["spectrum_id"]

        # Create boolean series indicating whether the prediction is valid for chimeric Prosit intensity prediction
        is_valid_chimeric_prosit_prediction = pd.Series(
            dataset.metadata["spectrum_id"].isin(valid_spectrum_ids),
        )

        return is_valid_chimeric_prosit_prediction

    def prepare(self, dataset: CalibrationDataset) -> None:
        """Prepares the dataset before feature computation.

        This method is intended to perform any preprocessing required before computing the feature.

        Args:
            dataset (CalibrationDataset): The dataset to prepare.
        """
        return

    def compute(self, dataset: CalibrationDataset) -> None:
        """Computes chimeric features for the given dataset.

        Uses the Prosit model to predict intensities for runner-up peptide sequences. The method processes predictions by sorting and grouping them, aligns ion match intensities and mass-to-charge ratios (m/z), and stores the results in the dataset metadata.

        Args:
            dataset (CalibrationDataset): The dataset containing metadata for predictions.

        Raises:
            ValueError: If learn_from_missing is False and invalid spectra are found in the dataset.
        """
        # Ensure dataset.predictions is not None
        _raise_value_error(dataset.predictions, "dataset.predictions")

        # Check which predictions are valid for Prosit intensity prediction
        is_valid_chimeric_prosit_prediction = (
            self.check_valid_chimeric_prosit_prediction(dataset)
        )
        dataset.metadata[
            "is_missing_chimeric_features"
        ] = ~is_valid_chimeric_prosit_prediction

        # If not learning from missing data, raise error when invalid spectra are found
        if not self.learn_from_missing:
            n_invalid = (~is_valid_chimeric_prosit_prediction).sum()
            if n_invalid > 0:
                raise ValueError(
                    f"Found {n_invalid} spectra with missing chimeric features. "
                    f"When learn_from_missing=False, all spectra must have valid runner-up sequences for Prosit prediction. "
                    f"Please filter your dataset to remove:\n"
                    f"  - Spectra without runner-up sequences (beam search required)\n"
                    f"  - Runner-up peptides longer than 30 amino acids\n"
                    f"  - Runner-up peptides with precursor charges greater than 6\n"
                    f"  - Runner-up peptides with unsupported modifications (e.g., {', '.join(self.invalid_prosit_tokens[:3])}...)\n"
                    f"Or set learn_from_missing=True to handle missing data automatically."
                )

        original_indices = dataset.metadata.index

        # Filter out invalid spectra for Prosit intensity prediction
        valid_chimeric_prosit_input = dataset.filter_entries(
            metadata_predicate=lambda row: row["is_missing_chimeric_features"]
        )

        # Prepare input data
        inputs = pd.DataFrame()
        inputs["peptide_sequences"] = np.array(
            [
                "".join(items[1].sequence)  # type: ignore
                for items in valid_chimeric_prosit_input.predictions
            ]
        )
        inputs["precursor_charges"] = np.array(
            valid_chimeric_prosit_input.metadata["precursor_charge"]
        )
        inputs["collision_energies"] = np.array(
            len(valid_chimeric_prosit_input.metadata) * [25]
        )
        inputs.index = valid_chimeric_prosit_input.metadata["spectrum_id"]

        model = koinapy.Koina(self.prosit_intensity_model_name)
        predictions: pd.DataFrame = model.predict(inputs)

        # Group predictions by spectrum_id to get one row per peptide
        # We make a temporary column spectrum_id_col to enable grouping by spectrum_id,
        # and we name this spectrum_id_col to avoid naming conflicts with the index
        predictions["spectrum_id_col"] = predictions.index
        grouped_predictions = predictions.groupby(by="spectrum_id_col").agg(
            {
                "peptide_sequences": "first",
                "precursor_charges": "first",
                "collision_energies": "first",
                "intensities": list,
                "mz": list,
                "annotation": list,
            }
        )
        # Sort intensities by m/z to match experimental data
        grouped_predictions["intensities"] = grouped_predictions.apply(
            lambda row: np.array(row["intensities"])[np.argsort(row["mz"])].tolist(),
            axis=1,
        )
        # Sort annotations by m/z to match experimental data
        grouped_predictions["annotation"] = grouped_predictions.apply(
            lambda row: np.array(row["annotation"])[np.argsort(row["mz"])].tolist(),
            axis=1,
        )
        # Sort m/z values to match experimental data
        grouped_predictions["mz"] = grouped_predictions["mz"].apply(np.sort)

        # Match computed metadata to valid spectra and impute missing values for invalid spectra
        # Reindex to match dataset.metadata.index and fill missing values with NaN
        dataset.metadata.index = dataset.metadata["spectrum_id"]
        dataset.metadata["runner_up_prosit_mz"] = grouped_predictions["mz"].reindex(
            dataset.metadata["spectrum_id"], fill_value=np.nan
        )
        dataset.metadata["runner_up_prosit_intensity"] = grouped_predictions[
            "intensities"
        ].reindex(dataset.metadata["spectrum_id"], fill_value=np.nan)

        # Revert to original indices
        dataset.metadata.index = original_indices

        # Compute ion matches and match intensity
        # Zeros are returned for rows with missing Prosit-predicted spectra
        ion_matches, match_intensity = compute_ion_identifications(
            dataset=dataset.metadata,
            source_column="runner_up_prosit_mz",
            mz_tolerance=self.mz_tolerance,
        )

        dataset.metadata["chimeric_ion_matches"] = ion_matches
        dataset.metadata["chimeric_ion_match_intensity"] = match_intensity


class MassErrorFeature(CalibrationFeatures):
    """Calculates the difference between the observed precursor mass and the theoretical mass."""

    h2o_mass: float = 18.0106
    proton_mass: float = 1.007276

    def __init__(self, residue_masses: Dict[str, float]) -> None:
        super().__init__()
        self.residue_masses = residue_masses

    @property
    def dependencies(self) -> List[FeatureDependency]:
        """Returns a list of dependencies required before computing the feature.

        Since this feature does not depend on other features, it returns an empty list.

        Returns:
            List[FeatureDependency]: An empty list.
        """
        return []

    @property
    def columns(self) -> List[str]:
        """Defines the column name for this feature.

        Returns:
            List[str]: A list containing the feature name.
        """
        return [self.name]

    @property
    def name(self) -> str:
        """Returns the name of the feature.

        This method provides the natural language identifier used as the key for the feature.

        Returns:
            str: The feature identifier.
        """
        return "Mass Error"

    def prepare(self, dataset: CalibrationDataset) -> None:
        """Prepares the dataset before feature computation.

        This method is intended to perform any preprocessing required before computing the feature.

        Args:
            dataset (CalibrationDataset): The dataset to prepare.
        """
        return

    def compute(
        self,
        dataset: CalibrationDataset,
    ) -> None:
        """Computes the mass error for each peptide.

        The mass error is calculated as the difference between the observed precursor mass and the theoretical peptide mass, accounting for the mass of water (H2O) and a proton (H+), which are added during ionisation.

        Args:
            dataset (CalibrationDataset): The dataset containing observed masses and peptide sequences.
        """
        theoretical_mass = dataset.metadata["prediction"].apply(
            lambda peptide: sum(self.residue_masses[residue] for residue in peptide)
            if isinstance(peptide, list)
            else float("-inf")
        )
        dataset.metadata[self.name] = dataset.metadata["precursor_mass"] - (
            theoretical_mass + self.h2o_mass + self.proton_mass
        )


class BeamFeatures(CalibrationFeatures):
    """Calculates the margin, median margin and entropy of beam runners-up."""

    @property
    def dependencies(self) -> List[FeatureDependency]:
        """Returns a list of dependencies required before computing the feature.

        Since this feature does not depend on other features, it returns an empty list.

        Returns:
            List[FeatureDependency]: An empty list.
        """
        return []

    @property
    def name(self) -> str:
        """Returns the name of the feature.

        This method provides the natural language identifier used as the key for the feature.

        Returns:
            str: The feature identifier.
        """
        return "Beam Features"

    @property
    def columns(self) -> List[str]:
        """Defines the column names for the computed features.

        Returns:
            List[str]: A list of column names: ["margin", "median_margin", "entropy"].
        """
        return ["margin", "median_margin", "entropy", "z-score"]

    def prepare(self, dataset: CalibrationDataset) -> None:
        """Prepares the dataset before feature computation.

        This method is intended to perform any preprocessing required before computing the feature.

        Args:
            dataset (CalibrationDataset): The dataset to prepare.
        """
        return

    def compute(self, dataset: CalibrationDataset) -> None:
        """Computes margin, median margin and entropy for beam search runners-up.

        - Margin: Difference between the highest probability sequence and the second-best sequence.
        - Median Margin: Difference between the highest probability sequence and the median probability of the runner-ups.
        - Entropy: Shannon entropy of the normalised probabilities of the runner-up sequences.
        - Z-score: Distance between the top beam score and the population mean over all beam results for that spectra in units of the standard deviation.

        These metrics help assess the confidence of the top prediction relative to lower-ranked candidates.

        Args:
            dataset (CalibrationDataset): The dataset containing beam search predictions.
        """
        # Ensure dataset.predictions is not None
        _raise_value_error(dataset.predictions, "dataset.predictions")

        count = sum(len(prediction) < 2 for prediction in dataset.predictions)  # type: ignore
        if count > 0:
            warnings.warn(
                f"{count} beam search results have fewer than two sequences. "
                "This may affect the efficacy of computed beam features."
            )

        top_probs = [
            exp(prediction[0].sequence_log_probability) if len(prediction) >= 1 else 0.0  # type: ignore
            for prediction in dataset.predictions
        ]
        second_probs = [
            exp(prediction[1].sequence_log_probability) if len(prediction) >= 2 else 0.0  # type: ignore
            for prediction in dataset.predictions
        ]
        second_margin = [
            top_prob - second_prob
            for top_prob, second_prob in zip(top_probs, second_probs)
        ]
        runner_up_probs = [
            [exp(item.sequence_log_probability) for item in prediction[1:]]  # type: ignore
            if len(prediction) >= 2  # type: ignore
            else [0.0]
            for prediction in dataset.predictions
        ]
        normalised_runner_up_probs = [
            [probability / sum(probabilities) for probability in probabilities]
            if sum(probabilities) != 0
            else 0.0
            for probabilities in runner_up_probs
        ]
        runner_up_entropy = [
            entropy(probs) if probs != 0 else 0.0
            for probs in normalised_runner_up_probs
        ]
        runner_up_median = [median(probs) for probs in runner_up_probs]
        median_margin = [
            top_prob - median_prob
            for top_prob, median_prob in zip(top_probs, runner_up_median)
        ]

        # Function to compute mean, std, and z-score over a row's beam results
        def row_beam_z_score(row):
            probabilities = [exp(beam.sequence_log_probability) for beam in row]
            mean_prob = np.mean(probabilities)
            std_prob = np.std(probabilities)
            if std_prob == 0:  # Avoid division by zero
                return 0  # Assign zero if all values are the same
            return (probabilities[0] - mean_prob) / std_prob

        z_score = [row_beam_z_score(prediction) for prediction in dataset.predictions]

        # dataset.metadata['confidence'] = top_probs
        dataset.metadata["margin"] = second_margin
        dataset.metadata["median_margin"] = median_margin
        dataset.metadata["entropy"] = runner_up_entropy
        dataset.metadata["z-score"] = z_score


class RetentionTimeFeature(CalibrationFeatures):
    """Computes Prosit iRT features and calibrates an iRT predictor.

    This feature uses the Prosit model to predict indexed retention times (iRT) for peptides and trains a regression model to calibrate predictions based on observed retention times.
    """

    irt_predictor: MLPRegressor

    def __init__(
        self,
        hidden_dim: int,
        train_fraction: float,
        invalid_prosit_tokens: List[str],
        learn_from_missing: bool = True,
        seed: int = 42,
        learning_rate_init: float = 0.001,
        alpha: float = 0.0001,
        max_iter: int = 200,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        prosit_irt_model_name: str = "Prosit_2019_irt",
    ) -> None:
        """Initialize RetentionTimeFeature.

        Args:
            hidden_dim (int): Hidden dimension size for the MLP regressor.
            train_fraction (float): Fraction of data to use for training the iRT calibrator.
            invalid_prosit_tokens (List[str]): The tokens to consider as invalid for Prosit iRT prediction.
            learn_from_missing (bool): Whether to learn from missing data by including a missingness indicator column.
                If False, an error will be raised when invalid spectra are encountered.
                Defaults to True.
            seed (int): Random seed for the regressor. Defaults to 42.
            learning_rate_init (float): The initial learning rate. Defaults to 0.001.
            alpha (float): L2 regularisation parameter. Defaults to 0.0001.
            max_iter (int): Maximum number of training iterations. Defaults to 200.
            early_stopping (bool): Whether to use early stopping to terminate training. Defaults to False.
            validation_fraction (float): Proportion of training data to use for early stopping validation. Defaults to 0.1.
            prosit_irt_model_name (str): The name of the Prosit iRT model to use.
                Defaults to "Prosit_2019_irt".
        """
        self.train_fraction = train_fraction
        self.hidden_dim = hidden_dim
        self.learn_from_missing = learn_from_missing
        self.invalid_prosit_tokens = invalid_prosit_tokens
        self.prosit_irt_model_name = prosit_irt_model_name
        self.irt_predictor = MLPRegressor(
            hidden_layer_sizes=[hidden_dim],
            random_state=seed,
            learning_rate_init=learning_rate_init,
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
        )

    @property
    def dependencies(self) -> List[FeatureDependency]:
        """Returns a list of dependencies required before computing the feature.

        Since this feature does not depend on other features, it returns an empty list.

        Returns:
            List[FeatureDependency]: An empty list.
        """
        return []

    @property
    def name(self) -> str:
        """Returns the name of the feature.

        This method provides the natural language identifier used as the key for the feature.

        Returns:
            str: The feature identifier.
        """
        return "Prosit iRT Features"

    @property
    def columns(self) -> List[str]:
        """Defines the column names for the computed features.

        Returns:
            List[str]: A list containing "iRT error" and optionally "is_missing_irt_error"
                if learn_from_missing is True.
        """
        columns = ["iRT error"]
        if self.learn_from_missing:
            columns.append("is_missing_irt_error")
        return columns

    def check_valid_irt_prediction(self, dataset: CalibrationDataset) -> pd.Series:
        """Check which predictions are valid for iRT prediction.

        Args:
            dataset (CalibrationDataset): The dataset to check.

        Returns:
            pd.Series: A series of booleans indicating whether the prediction is valid for iRT prediction.
        """
        # Filter out invalid spectra for Prosit iRT prediction
        filtered_dataset = (
            dataset.filter_entries(
                metadata_predicate=lambda row: len(row["prediction"]) > 30
            )
            .filter_entries(
                metadata_predicate=lambda row: (
                    any(
                        token in row["prediction_untokenised"]
                        for token in self.invalid_prosit_tokens
                    )
                )
            )
            .filter_entries(
                metadata_predicate=lambda row: (
                    any(token == "C" for token in row["prediction"])
                )
            )
        )

        # Obtain valid indices
        valid_spectrum_ids = filtered_dataset.metadata["spectrum_id"]

        # Create boolean series indicating whether the prediction is valid for Prosit iRT prediction
        is_valid_irt_prediction = pd.Series(
            dataset.metadata["spectrum_id"].isin(valid_spectrum_ids),
        )

        return is_valid_irt_prediction

    def prepare(self, dataset: CalibrationDataset) -> None:
        """Prepares the dataset by training an iRT calibration model.

        This method:
        1. Selects high-confidence peptide sequences.
        2. Uses the Prosit model to predict iRT values for them.
        3. Trains an MLPRegressor to map observed retention times to Prosit-predicted iRT values.

        Args:
            dataset (CalibrationDataset): The dataset containing peptide sequences and retention times.
        """
        # Create a copy of the dataset to avoid modifying the original
        dataset_copy = CalibrationDataset(
            metadata=dataset.metadata.copy(deep=True),
            predictions=dataset.predictions.copy() if dataset.predictions else [],
        )

        # Check which predictions are valid for Prosit iRT prediction
        is_valid_irt_prediction = self.check_valid_irt_prediction(dataset_copy)
        dataset_copy.metadata["is_missing_irt_error"] = ~is_valid_irt_prediction
        valid_irt_input = dataset_copy.filter_entries(
            metadata_predicate=lambda row: row["is_missing_irt_error"]
        )

        # Prepare training data
        # Select the most confident valid peptide identifications to create training labels
        train_data = valid_irt_input.metadata
        train_data = train_data.sort_values(by="confidence", ascending=False)
        train_data = train_data.iloc[: int(self.train_fraction * len(train_data))]

        inputs = pd.DataFrame()
        inputs["peptide_sequences"] = np.array(
            ["".join(peptide) for peptide in train_data["prediction"]]
        )
        inputs = inputs.set_index(train_data.index)

        prosit_model = koinapy.Koina(self.prosit_irt_model_name)
        predictions = prosit_model.predict(inputs)
        train_data["iRT"] = predictions["irt"]

        # -- Fit model
        x, y = train_data["retention_time"].values, train_data["iRT"].values
        self.irt_predictor.fit(x.reshape(-1, 1), y)

    def compute(self, dataset: CalibrationDataset) -> None:
        """Computes the iRT error by comparing observed retention times to predicted iRT values.

        This method:
        1. Uses the Prosit model to predict iRT values for all peptides.
        2. Uses the trained MLPRegressor to predict iRT values based on observed retention times.
        3. Computes the absolute error between predicted and actual iRT values.

        Args:
            dataset (CalibrationDataset): The dataset containing peptide sequences and retention times.

        Raises:
            ValueError: If learn_from_missing is False and invalid spectra are found in the dataset.
        """
        # Check which predictions are valid for Prosit iRT prediction
        is_valid_irt_prediction = self.check_valid_irt_prediction(dataset)
        dataset.metadata["is_missing_irt_error"] = ~is_valid_irt_prediction

        # If not learning from missing data, raise error when invalid spectra are found
        if not self.learn_from_missing:
            n_invalid = (~is_valid_irt_prediction).sum()
            if n_invalid > 0:
                raise ValueError(
                    f"Found {n_invalid} spectra with missing retention time features. "
                    f"When learn_from_missing=False, all spectra must be valid for iRT prediction. "
                    f"Please filter your dataset to remove:\n"
                    f"  - Spectra without retention time data\n"
                    f"  - Peptides longer than 30 amino acids\n"
                    f"  - Precursor charges greater than 6\n"
                    f"  - Peptides with unsupported modifications (e.g., {', '.join(self.invalid_prosit_tokens[:3])}...)\n"
                    f"Or set learn_from_missing=True to handle missing data automatically."
                )

        original_indices = dataset.metadata.index

        # Filter out invalid spectra for Prosit iRT prediction
        valid_irt_input = dataset.filter_entries(
            metadata_predicate=lambda row: row["is_missing_irt_error"]
        )

        # Prepare input data
        inputs = pd.DataFrame()
        inputs["peptide_sequences"] = np.array(
            ["".join(peptide) for peptide in valid_irt_input.metadata["prediction"]]
        )
        inputs.index = valid_irt_input.metadata["spectrum_id"]

        prosit_model = koinapy.Koina(self.prosit_irt_model_name)
        predictions = prosit_model.predict(inputs)
        predictions["spectrum_id"] = predictions.index

        # Match computed metadata to valid spectra and impute missing values for invalid spectra
        # Reindex to match dataset.metadata.index and fill missing values with NaN
        dataset.metadata.index = dataset.metadata["spectrum_id"]
        dataset.metadata["iRT"] = predictions["irt"].reindex(
            dataset.metadata["spectrum_id"], fill_value=np.nan
        )

        # Predict iRT using the trained MLPRegressor
        # Note that we will always obtain a predicted iRT value for each spectrum here,
        # even if the spectrum is invalid for Prosit, because we predict using observed retention time.
        dataset.metadata["predicted iRT"] = self.irt_predictor.predict(
            dataset.metadata["retention_time"].values.reshape(-1, 1)
        )

        # Revert to original indices
        dataset.metadata.index = original_indices

        # Compute iRT error
        # Set zeros for rows where "iRT" is missing
        dataset.metadata["iRT error"] = np.abs(
            dataset.metadata["predicted iRT"] - dataset.metadata["iRT"]
        ).fillna(0.0)
