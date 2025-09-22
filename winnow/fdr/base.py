"""A module with the base interface for FDR control classes."""

from abc import ABCMeta
from abc import abstractmethod
from typing import Iterable, Tuple, TypeVar

import numpy as np
import pandas as pd

from jaxtyping import Float

from winnow.datasets.psm_dataset import PSMDataset

T = TypeVar("T", bound=Iterable)


class FDRControl(metaclass=ABCMeta):
    """The interface for FDR control classes."""

    @abstractmethod
    def fit(self, dataset: T) -> None:
        """Fit parameters of FDR control method to a dataset.

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

    @abstractmethod
    def get_confidence_cutoff(self, threshold: float) -> float:
        """Return the confidence cutoff corresponding to a given FDR threshold.

        Args:
            threshold (float):
                The target FDR threshold. Must satisfy 0 < `threshold` < 1.

        Returns:
            float:
                The confidence cutoff corresponding to the target FDR threshold.
        """
        pass

    @abstractmethod
    def compute_fdr(self, score: float) -> float:
        """Compute FDR for a given confidence score."""
        pass

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
