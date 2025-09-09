import warnings

import pandas as pd
import numpy as np
from numpy.typing import NDArray

from winnow.fdr.base import FDRControl


class NonParametricFDRControl(FDRControl):
    """A non-parametric FDR control method that estimates FDR directly from confidence scores.

    This implementation uses a non-parametric approach to estimate false discovery rates
    by computing the cumulative error probabilities across sorted confidence scores.
    It does not make any assumptions about the underlying distribution of scores.
    """

    def __init__(self) -> None:
        """Initialize the non-parametric FDR control method."""
        super().__init__()
        self._confidence_scores: NDArray[np.float64] | None = None
        self._sorted_indices: NDArray[np.int64] | None = None
        self._is_correct: NDArray[np.bool_] | None = None
        self._null_scores: NDArray[np.float64] | None = None

    def fit(self, dataset: pd.DataFrame) -> None:
        """Fit the FDR control method to a dataset of confidence scores.

        Args:
            dataset (pd.DataFrame):
                An array of confidence scores from the dataset.
        """
        assert len(dataset) > 0, "Fit method requires non-empty data"
        dataset = dataset.to_numpy()

        # Store sorted confidence scores and their indices
        self._sorted_indices = np.argsort(-dataset)  # Sort in descending order
        self._confidence_scores = dataset[self._sorted_indices]

        # Compute error probabilities (1 - confidence)
        error_probabilities = 1 - self._confidence_scores

        # Compute cumulative error probabilities
        cum_error_probabilities = np.cumsum(error_probabilities)

        # Compute counts for each position
        counts = np.arange(1, len(error_probabilities) + 1)

        # Compute FDR as ratio of cumulative errors to counts
        self._fdr_values = cum_error_probabilities / counts

    def get_confidence_cutoff(self, threshold: float) -> float:
        """Compute the confidence score cutoff for a given FDR threshold.

        Args:
            threshold (float):
                The target FDR threshold, where 0 < threshold < 1.

        Returns:
            float:
                The confidence score cutoff corresponding to the specified FDR level.
                Returns 1.0 if FDR is always above threshold (no valid cutoff).
                Returns 0.0 if FDR is always below threshold (all scores valid).
        """
        if self._confidence_scores is None:
            raise AttributeError("FDR method not fitted, please call `fit()` first")

        # NOTE: FDR is computed as a cumulative average of errors over descending confidence scores.
        # This guarantees that FDR is a monotonically decreasing function of confidence score,
        # with strict increases for each newly included lower-confidence prediction (with a correspondingly higher error)
        # and the same FDR value outputted for an identical confidence score.

        # Check edge cases first
        if np.all(self._fdr_values > threshold):
            warnings.warn(
                f"FDR is always above threshold {threshold}. No valid cutoff exists. "
                "Returning 1.0 since no scores meet the FDR requirement.",
                UserWarning,
            )
            return 1.0
        elif np.all(self._fdr_values <= threshold):
            warnings.warn(
                f"FDR is always below threshold {threshold}. All scores meet the FDR requirement. "
                "Returning 0.0 since all scores are valid.",
                UserWarning,
            )
            return 0.0

        idx = np.searchsorted(self._fdr_values, threshold, side="right") - 1

        # Return the last score where FDR is below threshold
        return float(self._confidence_scores[idx])

    def compute_fdr(self, score: float) -> float:
        """Compute FDR estimate at a given confidence cutoff.

        P(incorrect | S >= s)

        Args:
            score (float): The confidence cutoff.

        Returns:
            float: The FDR estimate at the given score
        """
        if self._confidence_scores is None:
            raise AttributeError("FDR method not fitted, please call `fit()` first")

        # Find the least conservative index where scores drop below the cutoff
        idx = np.searchsorted(-self._confidence_scores, -score, side="right")

        # If the score is below the range of fitted confidence scores, return a conservative estimate of 1.0
        if idx == len(self._confidence_scores) and score < self._confidence_scores[-1]:
            warnings.warn(
                f"Score {score} is below the range of fitted confidence scores (min: {self._confidence_scores[-1]:.4f}). "
                f"Cannot compute FDR from fitted data. Returning conservative estimate of 1.0.",
                UserWarning,
            )
            return 1.0
        # If the score is above the range of fitted confidence scores, return a conservative estimate of the FDR at the lowest score
        elif idx == 0:
            warnings.warn(
                f"Score {score} is above the range of fitted confidence scores (max: {self._confidence_scores[0]:.4f}). "
                f"Cannot compute FDR from fitted data. Returning conservative estimate of {self._fdr_values[0]:.4f}.",
                UserWarning,
            )

        # Compute FDR for the scores above cutoff
        return float(self._fdr_values[idx])

    def compute_posterior_probability(self, score: float) -> float:
        """Compute posterior error probability (PEP) for a given confidence score.

        We assume that the confidence scores are calibrated, so that the posterior
        probability of an incorrect match is simply 1 - the confidence score:

        P(incorrect | S = s) = 1 - s

        Args:
            score (float): The confidence score.

        Returns:
            float: The PEP estimate
        """
        return 1 - score

    def add_psm_pep(
        self, dataset_metadata: pd.DataFrame, confidence_col: str
    ) -> pd.DataFrame:
        """Add PSM-specific posterior error probabilities as a new column to the dataset."""
        dataset_metadata = dataset_metadata.copy()
        dataset_metadata["psm_pep"] = dataset_metadata[confidence_col].apply(
            self.compute_posterior_probability
        )
        return dataset_metadata
