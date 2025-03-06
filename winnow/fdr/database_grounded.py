import bisect
from typing import Tuple
import pandas as pd

import numpy as np

from instanovo.utils.metrics import Metrics
from winnow.fdr.base import FDRControl


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
            residues=residue_masses, isotope_error_range=isotope_error_range
        )

        dataset["peptide"] = dataset["peptide"].apply(metrics._split_peptide)
        dataset["prediction"] = dataset["prediction"].apply(metrics._split_peptide)

        dataset["num_matches"] = dataset.apply(
            lambda row: (
                metrics._novor_match(row["peptide"], row["prediction"])
                if isinstance(row["prediction"], list)
                else 0
            ),
            axis=1,
        )
        dataset["correct"] = dataset.apply(
            lambda row: row["num_matches"]
            == len(row["peptide"])
            == len(row["prediction"]),
            axis=1,
        )
        self.preds = dataset[["correct", self.confidence_feature]]

        dataset = dataset.sort_values(
            by=self.confidence_feature, axis=0, ascending=False
        )
        precision = np.cumsum(dataset["correct"]) / np.arange(1, len(dataset) + 1)
        confidence = np.array(dataset[self.confidence_feature])

        self.fdr_thresholds = list(1.0 - precision[drop:])
        self.confidence_scores = list(confidence[drop:])

    def get_confidence_cutoff(self, threshold: float) -> float:
        """Compute confidence cutoff for a given FDR threshold."""
        return self.confidence_scores[
            bisect.bisect_left(self.fdr_thresholds, threshold)
        ]

    def compute_fdr(self, score: float) -> float:
        """Compute false discovery rate for a given confidence score."""
        # FDR = [no. false positives >= score s] / [no. total matches >= score s]
        preds_ge_score = self.preds[self.preds[self.confidence_feature] >= score]
        return (len(preds_ge_score["correct"]) - sum(preds_ge_score["correct"])) / len(
            preds_ge_score["correct"]
        )
