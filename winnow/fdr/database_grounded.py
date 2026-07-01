from typing import Tuple

import numpy as np
import pandas as pd
from instanovo.utils.metrics import Metrics
from instanovo.utils.residues import ResidueSet

from winnow.datasets.data_loaders.utils import (
    finalize_peptide_metadata,
    require_labelled_rows,
)
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

        Per-row correctness is derived from ``sequence`` vs ``prediction``. Rows with
        ``valid_sequence=False`` receive ``correct=False`` but are excluded from the
        FDR curve.

        Args:
            dataset: DataFrame with ``sequence``, ``prediction``, and the confidence
                column named by ``confidence_feature``.
        """
        assert len(dataset) > 0, "Fit method requires non-empty data"

        finalize_peptide_metadata(
            dataset,
            self.metrics,
            has_labels=True,
        )
        self.preds = dataset[["correct", self.confidence_feature]]

        mask = require_labelled_rows(dataset, context="Database-grounded FDR fit")
        labelled = dataset.loc[mask].sort_values(
            by=self.confidence_feature, ascending=False
        )

        precision = np.cumsum(labelled["correct"]) / np.arange(1, len(labelled) + 1)
        confidence = np.array(labelled[self.confidence_feature])

        self._fdr_values = np.array(1 - precision[self.drop :])
        self._confidence_scores = confidence[self.drop :]
