from typing import Tuple
import pandas as pd
import numpy as np
from instanovo.utils.metrics import Metrics
from instanovo.utils.residues import ResidueSet

from winnow.fdr.base import FDRControl


class DatabaseGroundedFDRControl(FDRControl):
    """Performs false discovery rate (FDR) control by grounding predictions against a reference database.

    This method estimates FDR thresholds by comparing model-predicted peptides to ground-truth peptides from a database.
    """

    def __init__(
        self,
        confidence_feature: str,
        residue_masses: dict[str, float],
        isotope_error_range: Tuple[int, int] = (0, 1),
        drop: int = 10,
    ) -> None:
        super().__init__()
        self.confidence_feature = confidence_feature
        self.residue_masses = residue_masses
        self.isotope_error_range = isotope_error_range
        self.drop = drop

        self.metrics = Metrics(
            residue_set=ResidueSet(residue_masses=residue_masses),
            isotope_error_range=isotope_error_range,
        )

    def fit(  # type: ignore
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Computes the precision-recall curve by comparing model predictions to database-grounded peptide sequences.

        Args:
            dataset (pd.DataFrame):
                A DataFrame containing the following columns:
                - 'peptide': Ground-truth peptide sequences.
                - 'prediction': Model-predicted peptide sequences.
                - 'confidence': Confidence scores associated with predictions.
        """
        assert len(dataset) > 0, "Fit method requires non-empty data"

        dataset["sequence"] = dataset["sequence"].apply(self.metrics._split_peptide)

        dataset["num_matches"] = dataset.apply(
            lambda row: (
                self.metrics._novor_match(row["sequence"], row["prediction"])
                if isinstance(row["prediction"], list)
                else 0
            ),
            axis=1,
        )
        dataset["correct"] = dataset.apply(
            lambda row: row["num_matches"]
            == len(row["sequence"])
            == len(row["prediction"]),
            axis=1,
        )
        self.preds = dataset[["correct", self.confidence_feature]]

        dataset = dataset.sort_values(
            by=self.confidence_feature, axis=0, ascending=False
        )
        precision = np.cumsum(dataset["correct"]) / np.arange(1, len(dataset) + 1)
        confidence = np.array(dataset[self.confidence_feature])

        self._fdr_values = np.array(1 - precision[self.drop :])
        self._confidence_scores = confidence[self.drop :]
