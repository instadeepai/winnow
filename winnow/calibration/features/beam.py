from typing import Dict, List
import warnings
import numpy as np
from math import exp
from scipy.stats import entropy
from numpy import median

from winnow.calibration.features.base import CalibrationFeatures, FeatureDependency
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.calibration.features.utils import require_beam_predictions


def _normalise_li(tokens: List[str]) -> List[str]:
    """Replace every 'I' token with 'L' so leucine/isoleucine are treated identically."""
    return ["L" if t == "I" else t for t in tokens]


def _normalised_levenshtein(sequence_a: List[str], sequence_b: List[str]) -> float:
    """Normalised token-level Levenshtein distance between two sequences.

    Tokens are mapped to consecutive integers per pair comparison so the dynamic
    program compares small ints instead of strings.

    Returns the edit distance divided by the length of the longer sequence,
    giving a value in [0, 1]. Returns 0.0 when both sequences are empty.
    """
    residue_to_id: Dict[str, int] = {}

    def _to_ints(sequence: List[str]) -> List[int]:
        return [
            residue_to_id.setdefault(residue, len(residue_to_id))
            for residue in sequence
        ]

    a_ints = _to_ints(sequence_a)
    b_ints = _to_ints(sequence_b)
    return _normalised_levenshtein_ints(a_ints, b_ints)


def _normalised_levenshtein_ints(a: List[int], b: List[int]) -> float:
    """Levenshtein on integer token ids; result normalised by max(len(a), len(b))."""
    n, m = len(a), len(b)
    max_len = max(n, m)
    if max_len == 0:
        return 0.0

    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            res = prev[j] + 1
            if curr[j - 1] + 1 < res:
                res = curr[j - 1] + 1
            if prev[j - 1] + cost < res:
                res = prev[j - 1] + cost
            curr[j] = res
        prev, curr = curr, prev
    return prev[m] / max_len


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
            List[str]: A list of column names: ["margin", "median_margin", "entropy", "z-score", "edit_distance"].
        """
        return ["margin", "median_margin", "entropy", "z-score", "edit_distance"]

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
        - Edit distance: Normalised Levenshtein distance between the top-1 and top-2 sequences, treating I/L as the same residue.

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

        edit_distances: List[float] = []
        for prediction in dataset.predictions:
            if len(prediction) < 2:  # type: ignore
                edit_distances.append(1.0)
            else:
                edit_distances.append(
                    _normalised_levenshtein(
                        _normalise_li(prediction[0].sequence),  # type: ignore
                        _normalise_li(prediction[1].sequence),  # type: ignore
                    )
                )

        dataset.metadata["margin"] = second_margin
        dataset.metadata["median_margin"] = median_margin
        dataset.metadata["entropy"] = runner_up_entropy
        dataset.metadata["z-score"] = z_score
        dataset.metadata["edit_distance"] = edit_distances
