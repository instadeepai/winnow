import numpy as np

from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.datasets.data_loaders.utils import require_labelled_rows
from winnow.fdr.base import FDRControl


class DatabaseGroundedFDRControl(FDRControl[CalibrationDataset]):
    """Performs false discovery rate (FDR) control by grounding predictions against a reference database.

    This method estimates FDR thresholds by comparing model-predicted peptides to ground-truth peptides from a database.
    It expects finalised labelled metadata from a DatasetLoader, including ``correct`` and ``valid_sequence``.
    """

    def __init__(
        self,
        confidence_feature: str,
        drop: int = 10,
    ) -> None:
        super().__init__()
        self.confidence_feature = confidence_feature
        self.drop = drop

    def fit(self, dataset: CalibrationDataset) -> None:
        """Computes the precision-recall curve from finalised correctness labels.

        Rows with ``valid_sequence=False`` are excluded from the FDR curve. Per-row
        correctness must already be present in ``correct``.

        Args:
            dataset: Finalised calibration dataset with ``correct``, ``valid_sequence``,
                and the confidence column named by ``confidence_feature``.

        Raises:
            ValueError: If required finalised label columns are missing.
        """
        assert len(dataset) > 0, "Fit method requires non-empty data"

        metadata = dataset.metadata
        missing = [
            column
            for column in ("correct", "valid_sequence", self.confidence_feature)
            if column not in metadata.columns
        ]
        if missing:
            raise ValueError(
                "This operation requires finalised labelled metadata with "
                f"{', '.join(repr(c) for c in missing)}. "
                "Load labelled data through a DatasetLoader before fitting/running "
                "diagnostics."
            )

        self.preds = metadata[["correct", self.confidence_feature]]

        mask = require_labelled_rows(metadata, context="Database-grounded FDR fit")
        labelled = metadata.loc[mask].sort_values(
            by=self.confidence_feature, ascending=False
        )

        precision = np.cumsum(labelled["correct"]) / np.arange(1, len(labelled) + 1)
        confidence = np.array(labelled[self.confidence_feature])

        self._fdr_values = np.array(1 - precision[self.drop :])
        self._confidence_scores = confidence[self.drop :]
