import bisect
from typing import Tuple
import pandas as pd

import numpy as np

from instanovo.utils.metrics import Metrics
from winnow.fdr.base import FDRControl


class DatabaseGroundedFDRControl(FDRControl):
    """Perform FDR control by grounding to database predictions
    """

    def __init__(self) -> "DatabaseGroundedFDRControl":
        self.fdr_thresholds = []
        self.confidence_scores = []

    def fit(
        self, dataset: pd.DataFrame, residue_masses: dict[str, float],
        isotope_error_range: Tuple[int, int]=(0, 1), drop: int = 10
    ) -> None:
        """Compute the precision recall curve of model predictions against database predictions.

        Args:
            dataset (pandas.DataFrame['peptide', 'prediction', 'confidence']):
                A data frame with the spectra, gold peptides and model predictions.
        """
        metrics = Metrics(
            residues=residue_masses,
            isotope_error_range=isotope_error_range
        )

        dataset['peptide'] = dataset['peptide'].apply(metrics._split_peptide)
        dataset['prediction'] = dataset['prediction'].apply(metrics._split_peptide)


        dataset['num_matches'] = dataset.apply(
            lambda row: (
                metrics._novor_match(row['peptide'], row['prediction'])
                if isinstance(row['prediction'], list) else 0
            ),
            axis=1
        )
        dataset['correct'] = dataset.apply(
            lambda row: row['num_matches'] == len(row['peptide']) == len(row['prediction']),
            axis=1
        )
        self.preds = dataset[['correct', 'confidence']]
        
        dataset = dataset.sort_values(by='confidence', axis=0, ascending=False)
        precision = np.cumsum(dataset['correct']) / np.arange(1, len(dataset) + 1)
        confidence = np.array(dataset['confidence'])

        self.fdr_thresholds = list(1.0 - precision[drop:])
        self.confidence_scores = list(confidence[drop:])

    def get_confidence_cutoff(self, threshold: float) -> float:
        return self.confidence_scores[bisect.bisect_left(self.fdr_thresholds, threshold)]

    def compute_fdr(self, score: float) -> float:
        # FDR = [no. false positives >= score s] / [no. total matches >= score s]
        preds_ge_score = self.preds[self.preds['confidence'] >= score]
        return (len(preds_ge_score['correct']) - sum(preds_ge_score['correct'])) / len(preds_ge_score['correct'])
