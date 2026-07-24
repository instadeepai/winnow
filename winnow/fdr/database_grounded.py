import numpy as np

from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.datasets.data_loaders.utils import (
    SEQUENCE_DERIVED_CORRECT_COLUMN,
    coerce_bool_labels,
)
from winnow.fdr.base import FDRControl


class DatabaseGroundedFDRControl(FDRControl[CalibrationDataset]):
    """Performs false discovery rate (FDR) control by grounding predictions against a reference database.

    This method estimates FDR thresholds from per-row correctness labels ranked by
    confidence. Correctness may come from the sequence-derived ``correct`` column
    (with ``valid_sequence``) or from a custom proxy column (e.g. ``proteome_hit``)
    that needs neither ground-truth sequence nor ``valid_sequence``.
    """

    def __init__(
        self,
        confidence_feature: str,
        drop: int = 10,
    ) -> None:
        super().__init__()
        self.confidence_feature = confidence_feature
        self.drop = drop

    def fit(
        self,
        dataset: CalibrationDataset,
        correct_column: str = SEQUENCE_DERIVED_CORRECT_COLUMN,
    ) -> None:
        """Computes the precision-recall curve from finalised correctness labels.

        Row eligibility depends on the correctness source:

        - Default sequence-derived correctness (``correct_column == "correct"``): only
          rows with ``valid_sequence=True`` are used.
        - Custom proxy correctness (any other ``correct_column``, e.g.
          ``"proteome_hit"``): sequence validity is ignored; a coexisting
          ``sequence`` / ``valid_sequence`` column has no effect.

        Eligible rows must have a non-null boolean or numeric value in
        ``correct_column``; missing values and other dtypes are rejected.

        Args:
            dataset: Calibration dataset with the confidence column named by
                ``confidence_feature`` and per-row correctness in ``correct_column``.
                When ``correct_column`` is ``"correct"``, ``valid_sequence`` must also
                be present.
            correct_column: Name of the column containing per-row correctness labels.
                Defaults to ``"correct"``.

        Raises:
            ValueError: If required columns are missing, labels are missing or not
                boolean/numeric, or no labelled rows remain after eligibility
                filtering.
        """
        if len(dataset) == 0:
            raise ValueError("Fit method requires non-empty data")

        metadata = dataset.metadata
        use_sequence_validity = correct_column == SEQUENCE_DERIVED_CORRECT_COLUMN

        required = [correct_column, self.confidence_feature]
        if use_sequence_validity:
            required.append("valid_sequence")
        missing = [column for column in required if column not in metadata.columns]
        if missing:
            missing_repr = ", ".join(repr(c) for c in missing)
            if use_sequence_validity:
                raise ValueError(
                    "This operation requires finalised labelled metadata with "
                    f"{missing_repr}. "
                    "Load labelled data through a DatasetLoader before fitting."
                )
            raise ValueError(
                "Database-grounded FDR fit requires metadata columns "
                f"{missing_repr} (correctness column "
                f"{correct_column!r} and confidence feature "
                f"{self.confidence_feature!r})."
            )

        if use_sequence_validity:
            mask = metadata["valid_sequence"].fillna(False).to_numpy(dtype=bool)
            if not mask.any():
                raise ValueError(
                    "Database-grounded FDR fit requires at least one row with "
                    "valid_sequence=True."
                )
        else:
            # Custom proxy correctness: all rows are eligible.
            mask = np.ones(len(metadata), dtype=bool)

        labelled = metadata.loc[mask].sort_values(
            by=self.confidence_feature, ascending=False
        )
        if len(labelled) == 0:
            raise ValueError("No labelled rows available for FDR fit")

        correct = coerce_bool_labels(labelled[correct_column], correct_column).to_numpy(
            dtype=bool
        )
        self.preds = labelled[[correct_column, self.confidence_feature]]

        precision = np.cumsum(correct) / np.arange(1, len(labelled) + 1)
        confidence = np.array(labelled[self.confidence_feature])

        self._fdr_values = np.array(1 - precision[self.drop :])
        self._confidence_scores = confidence[self.drop :]
