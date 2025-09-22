import pandas as pd
import numpy as np
from numpy.typing import NDArray

from winnow.fdr.base import FDRControl


class NonParametricFDRControl(FDRControl):
    """A non-parametric false discovery rate (FDR) control method that estimates FDR directly from confidence scores.

    This implementation uses a non-parametric approach to estimate FDR by computing the cumulative error probabilities across sorted confidence scores.
    It does not make any assumptions about the underlying distribution of scores.
    """

    def __init__(self) -> None:
        """Initialise the non-parametric FDR control method."""
        super().__init__()
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
