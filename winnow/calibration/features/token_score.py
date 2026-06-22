from typing import List

import numpy as np

from winnow.calibration.features.base import CalibrationFeatures, FeatureDependency
from winnow.calibration.features.utils import require_beam_predictions
from winnow.datasets.calibration_dataset import CalibrationDataset


class TokenScoreFeatures(CalibrationFeatures):
    """Computes token score features for the top prediction.

    These features capture token-level confidence information that may help
    identify low-confidence positions within a prediction. All computations
    use probabilities (converted from stored log-probabilities via exp()).
    """

    def __init__(self) -> None:
        """Initialize TokenScoreFeatures."""
        pass

    @property
    def dependencies(self) -> List[FeatureDependency]:
        """Returns a list of dependencies required before computing the feature."""
        return []

    @property
    def name(self) -> str:
        """Returns the name of the feature."""
        return "Token Score Features"

    @property
    def columns(self) -> List[str]:
        """Returns the columns of the feature."""
        return ["min_token_probability", "std_token_probability"]

    def prepare(self, dataset: CalibrationDataset) -> None:
        """Prepares the dataset for the feature computation.

        Args:
            dataset: The calibration dataset to prepare.
        """
        pass

    def compute(self, dataset: CalibrationDataset) -> None:
        """Computes the feature for the dataset.

        Computes per-row:
            - min_token_probability: Minimum token probability for the top prediction
            - std_token_probability: Standard deviation of token probabilities

        Args:
            dataset: The calibration dataset containing predictions with token log probabilities.

        Raises:
            ValueError: If beam predictions are not available or token_log_probabilities is None.
        """
        require_beam_predictions(dataset, "TokenScoreFeatures")
        assert dataset.predictions is not None

        # Check that token log probabilities are available for the top prediction
        if any(
            prediction[0].token_log_probabilities is None  # type: ignore
            for prediction in dataset.predictions
        ):
            raise ValueError(
                "Token log probabilities are not available for the top prediction. "
                "This is required for token score features computation."
            )

        # Convert log probabilities to probabilities and compute features
        min_probs = []
        std_probs = []

        for prediction in dataset.predictions:
            token_log_probs = prediction[0].token_log_probabilities  # type: ignore
            token_probs = np.exp(token_log_probs)

            if len(token_probs) > 0:
                min_probs.append(float(np.min(token_probs)))
            else:
                min_probs.append(0.0)

            if len(token_probs) > 1:
                std_probs.append(float(np.std(token_probs)))
            else:
                std_probs.append(0.0)

        dataset.metadata["min_token_probability"] = min_probs
        dataset.metadata["std_token_probability"] = std_probs
