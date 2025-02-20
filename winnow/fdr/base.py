"""A module with the base interface for FDR control classes."""

from abc import ABCMeta
from abc import abstractmethod
from typing import Iterable, Tuple, TypeVar

import numpy as np

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
