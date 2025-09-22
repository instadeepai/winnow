"""A module with the base interface for FDR control classes."""

from abc import ABCMeta
from abc import abstractmethod
from typing import Iterable, Tuple, TypeVar
import warnings

import numpy as np
import pandas as pd

from jaxtyping import Float
from numpy.typing import NDArray

from winnow.datasets.psm_dataset import PSMDataset

T = TypeVar("T", bound=Iterable)


class FDRControl(metaclass=ABCMeta):
    """The interface for FDR control classes."""

    def __init__(self) -> None:
        self._fdr_values: NDArray[np.float64] | None = None
        self._confidence_scores: NDArray[np.float64] | None = None

    @abstractmethod
    def fit(self, dataset: T) -> None:
        """Fit parameters of FDR control method to a dataset.

        This method should use the `self._fdr_values` and `self._confidence_scores` attributes to store the FDR values and confidence scores.

        Args:
            dataset (T):
                The dataset to ground FDR estimates to.
        """
        pass

    def filter_entries(self, dataset: PSMDataset, threshold: float) -> PSMDataset:
        """Filter PSMs to return results at a given FDR threshold.

        Args:
            dataset (Dataset):
                The dataset of PSMs with confidence scores to filter.

            threshold (float):
                The FDR control threshold to limit to. Must satisfy
                0 < `threshold` < 1.

        Returns:
            Dataset:
                The set of PSMs with confidence above the cutoff for
                the target FDR threshold
        """
        confidence_cutoff = self.get_confidence_cutoff(threshold=threshold)
        return PSMDataset(
            peptide_spectrum_matches=[
                psm for psm in dataset if psm.confidence > confidence_cutoff
            ]
        )

    def get_confidence_cutoff(self, threshold: float) -> float:
        """Compute the confidence score cutoff for a given FDR threshold.

        Args:
            threshold (float):
                The target FDR threshold, where 0 < threshold < 1.

        Returns:
            float:
                The confidence score cutoff corresponding to the specified FDR level.
        """
        if self._confidence_scores is None or self._fdr_values is None:
            raise AttributeError("FDR method not fitted, please call `fit()` first")

        # NOTE: The false discovery rate (FDR) is computed as the cumulative average across predictions
        # ranked in descending order of confidence. This guarantees that FDR is a monotonically decreasing
        # function of confidence score: it exhibits strict increases when lower-confidence predictions with
        # higher error rates are incorporated, while remaining constant for predictions that share the same
        # confidence score.

        # Find the least conservative index where FDR is at or below threshold
        idx = np.searchsorted(self._fdr_values, threshold, side="right") - 1

        # If all observed PSM-specific FDR thresholds are above the threshold, this means that no scores meet the FDR requirement.
        # In this case, we do not return a confidence score cutoff.
        if idx == -1:
            warnings.warn(
                f"FDR threshold {threshold} is below the range of fitted FDR thresholds (min: {self._fdr_values[0]:.4f}). "
                f"Cannot compute an accurate FDR estimate from fitted data. Returning NaN.",
                UserWarning,
            )
            return np.nan

        # If the threshold is above the range of fitted FDR thresholds, this means that all scores
        # meet the FDR requirement.
        # In this case, return a conservative estimate of the lowest fitted confidence score.
        # We do not automatically return a confidence score of 0.0 because we cannot guarantee that FDR
        # would not increase above the threshold for scores below the minimum fitted confidence score.
        elif idx == len(self._fdr_values) - 1 and threshold > self._fdr_values[-1]:
            warnings.warn(
                f"FDR threshold {threshold} is above the range of fitted FDR thresholds (max: {self._fdr_values[-1]:.4f}). "
                f"Cannot compute an accurate FDR estimate from fitted data. Returning conservative estimate of {self._confidence_scores[idx]:.4f}.",
                UserWarning,
            )

        return self._confidence_scores[idx]

    def compute_fdr(self, score: float) -> float:
        """Computes the false discovery rate (FDR) for all PSMs with a confidence score >= `score`.

        This method calculates the FDR as the probability of a PSM being incorrect given
        that its confidence score is greater than or equal to the given cutoff `score`.
        It is formally defined as: `P(incorrect | S >= s)`.
        It is important to note that FDR, like q-values, depends upon the entire dataset,
        whereas posterior error probability (PEP) depends only on the single given PSM's confidence score.

        Args:
            score (float): The confidence score cutoff, where 0 < score < 1.

        Returns:
            float: The estimated FDR for all PSMs at or above the given score.
        """
        if self._confidence_scores is None or self._fdr_values is None:
            raise AttributeError("FDR method not fitted, please call `fit()` first")

        # Find the least conservative index where confidence scores are at or above the cutoff
        idx = np.searchsorted(-self._confidence_scores, -score, side="left")

        # If the score is below the range of fitted confidence scores, return a conservative estimate of 1.0
        if idx == len(self._confidence_scores) and score < self._confidence_scores[-1]:
            warnings.warn(
                f"Score {score} is below the range of fitted confidence scores (min: {self._confidence_scores[-1]:.4f}). "
                f"Cannot compute FDR from fitted data. Returning conservative estimate of 1.0.",
                UserWarning,
            )
            return 1.0
        # If the score is above the range of fitted confidence scores, return a conservative estimate of the FDR at the lowest score
        elif idx == 0 and score > self._confidence_scores[0]:
            warnings.warn(
                f"Score {score} is above the range of fitted confidence scores (max: {self._confidence_scores[0]:.4f}). "
                f"Cannot compute FDR from fitted data. Returning conservative estimate of {self._fdr_values[0]:.4f}.",
                UserWarning,
            )

        return self._fdr_values[idx]

    def add_psm_fdr(
        self, dataset_metadata: pd.DataFrame, confidence_col: str
    ) -> pd.DataFrame:
        """Add PSM-specific FDR values as a new column to the dataset."""
        dataset_metadata = dataset_metadata.copy()
        dataset_metadata["psm_fdr"] = dataset_metadata[confidence_col].apply(
            self.compute_fdr
        )
        return dataset_metadata

    def add_psm_q_value(
        self, dataset_metadata: pd.DataFrame, confidence_col: str
    ) -> pd.DataFrame:
        """Add PSM-specific q-values as a new column to the dataset.

        Q-values are calculated using the following algorithm:
        1. Sort by confidence in descending order.
        2. Calculate FDR for each PSM.
        3. Traverse identifications from lowest score to highest, storing the lowest
           estimated FDR (FDRmin) that has been observed so far (i.e., for each
           identification, retrieve the assigned FDR value, if this value is larger
           then FDRmin, retain FDRmin. Else q-value = FDR value and FDRmin = FDR value).
        """
        # Sort by confidence scores in descending order
        sorted_data = dataset_metadata.sort_values(
            confidence_col, ascending=False
        ).copy()

        # Calculate FDR for each PSM
        sorted_data["fdr"] = sorted_data[confidence_col].apply(self.compute_fdr)

        # Calculate q-values: traverse from lowest score to highest (reverse order)
        q_values: list[float] = []
        fdr_min = float("inf")

        # We sort FDR values in descending order and append to end of list,
        #   then reverse to get original order.
        # This better leverages the underlying list data structure
        for current_fdr in reversed(sorted_data["fdr"].tolist()):
            if current_fdr > fdr_min:  # Retain FDRmin
                q_values.append(fdr_min)
            else:  # q-value = FDR value and FDRmin = FDR value
                q_values.append(current_fdr)
                fdr_min = current_fdr

        q_values.reverse()  # Reverse again to align order with confidence scores

        # Add q-values to the sorted dataframe
        sorted_data["psm_q_value"] = q_values

        # Restore original order by merging back with original dataframe
        result = dataset_metadata.merge(
            sorted_data[["psm_q_value"]], left_index=True, right_index=True, how="left"
        )

        return result

    def get_confidence_curve(
        self, resolution: float, min_confidence: float, max_confidence: float
    ) -> Tuple[Float[np.ndarray, "threshold"], Float[np.ndarray, "threshold"]]:  # noqa: F821
        """Return the curve with the confidence cutoff for corresponding FDR thresholds.

        Args:
            resolution (float):
                The uniform space between FDR thresholds for the curve.


        Returns:
            List[Tuple[float, float]]:

        """
        fdr_thresholds = np.arange(min_confidence, max_confidence, resolution)
        confidence_scores = np.array(
            [
                self.get_confidence_cutoff(threshold=threshold)
                for threshold in fdr_thresholds
            ]
        )
        return fdr_thresholds, confidence_scores
