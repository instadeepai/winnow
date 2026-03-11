from typing import List
import warnings
import numpy as np
from math import exp
from scipy.stats import entropy
from numpy import median

from winnow.calibration.features.base import CalibrationFeatures, FeatureDependency
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.calibration.features.utils import require_beam_predictions


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
            List[str]: A list of column names: ["margin", "median_margin", "entropy", "z-score"].
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
        # Ensure dataset.predictions is not None (beams required for margin/entropy calculations)
        require_beam_predictions(dataset, "BeamFeatures")
        assert dataset.predictions is not None

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
