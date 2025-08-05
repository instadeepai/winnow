import bisect
from typing import Tuple
import pandas as pd
import warnings
import numpy as np
from instanovo.utils.metrics import Metrics
from winnow.fdr.base import FDRControl
from winnow.datasets.calibration_dataset import residue_set


class DatabaseGroundedFDRControl(FDRControl):
    """Performs False Discovery Rate (FDR) control by grounding predictions against a reference database.

    This method estimates FDR thresholds by comparing model-predicted peptides to ground-truth peptides from a database.
    """

    def __init__(self, confidence_feature: str) -> None:
        self.fdr_thresholds: list[float] = []
        self.confidence_scores: list[float] = []
        self.confidence_feature = confidence_feature

    def fit(  # type: ignore
        self,
        dataset: pd.DataFrame,
        residue_masses: dict[str, float],
        isotope_error_range: Tuple[int, int] = (0, 1),
        drop: int = 10,
        correct_column: str = "correct",
    ) -> None:
        """Computes the precision-recall curve by comparing model predictions to database-grounded peptide sequences.

        Args:
            dataset (pd.DataFrame):
                A DataFrame containing the following columns:
                - 'peptide': Ground-truth peptide sequences.
                - 'prediction': Model-predicted peptide sequences.
                - 'confidence': Confidence scores associated with predictions.

            residue_masses (dict[str, float]): A dictionary mapping amino acid residues to their respective masses.

            isotope_error_range (Tuple[int, int], optional): Range of isotope errors to consider when matching peptides. Defaults to (0, 1).

            drop (int): Number of top-scoring predictions to exclude when computing FDR thresholds. Defaults to 10.
        """
        metrics = Metrics(
            residue_set=residue_set, isotope_error_range=isotope_error_range
        )

        if correct_column == "correct":
            dataset["sequence"] = dataset["sequence"].apply(metrics._split_peptide)
            # dataset["prediction"] = dataset["prediction"].apply(metrics._split_peptide)
            dataset["num_matches"] = dataset.apply(
                lambda row: (
                    metrics._novor_match(row["sequence"], row["prediction"])
                    if isinstance(row["prediction"], list)
                    else 0
                ),
                axis=1,
            )
            dataset[correct_column] = dataset.apply(
                lambda row: row["num_matches"]
                == len(row["sequence"])
                == len(row["prediction"]),
                axis=1,
            )
        # If correct column is "proteome_hit", we expect the column to be a boolean that is precomputed
        self.preds = dataset[[correct_column, self.confidence_feature]]

        dataset = dataset.sort_values(
            by=self.confidence_feature, axis=0, ascending=False
        )
        precision = np.cumsum(dataset[correct_column]) / np.arange(1, len(dataset) + 1)
        confidence = np.array(dataset[self.confidence_feature])

        self.fdr_thresholds = list(1.0 - precision[drop:])
        self.confidence_scores = list(confidence[drop:])

        self.reversed_fdr_thresholds = list(reversed(self.fdr_thresholds))
        self.reversed_confidence_scores = list(reversed(self.confidence_scores))

    def get_confidence_cutoff(self, threshold: float) -> float:
        """Compute the confidence score cutoff for a given FDR threshold.

        This function determines the confidence score above which PSMs should be retained
        to maintain the desired FDR level.

        Args:
            threshold (float):
                The target FDR threshold, where 0 < threshold < 1.

        Returns:
            float:
                The confidence score cutoff corresponding to the specified FDR level.
        """
        idx = bisect.bisect_right(self.fdr_thresholds, threshold) - 1

        if idx < 0:
            return np.nan

        return self.confidence_scores[idx].item()  # type: ignore

    def compute_fdr(self, score: float) -> float:
        """Compute FDR estimate at a given confidence cutoff.

        Args:
            score (float): The confidence cutoff.

        Returns:
            float: The FDR estimate
        """
        # Find the index where this score would be inserted in the sorted confidence scores
        idx = bisect.bisect_right(self.reversed_confidence_scores, score)

        if (
            idx >= len(self.reversed_confidence_scores)
            and self.reversed_fdr_thresholds[-1] == 0
        ):
            return 0

        elif idx >= len(self.reversed_fdr_thresholds):
            warnings.warn(
                f"Score {score} is too high for FDR control. Decreasing the drop parameter during fitting may improve results."
            )
            return np.nan

        # Return the FDR threshold at this index
        return self.reversed_fdr_thresholds[idx]
