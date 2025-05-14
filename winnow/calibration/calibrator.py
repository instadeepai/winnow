"""Contains classes and functions for probability recalibration."""

from typing import Dict, List, Tuple, Union, TypeVar


import numpy as np
from sklearn.linear_model import LogisticRegression
from jaxtyping import Float, Array

from winnow.calibration.calibration_features import (
    CalibrationFeatures,
    FeatureDependency,
)
from winnow.datasets.calibration_dataset import CalibrationDataset

# Define dimension names as type variables
N = TypeVar("N", bound=int)  # number of samples
M = TypeVar("M", bound=int)  # number of features


class ProbabilityCalibrator:
    """A class for recalibrating probabilities for a de novo peptide sequencing method.

    This class provides functionality to recalibrate predicted probabilities by fitting a logistic regression model using various features computed from a calibration dataset.
    """

    def __init__(self, seed: int = 42) -> None:
        self.feature_dict: Dict[str, CalibrationFeatures] = {}
        self.dependencies: Dict[str, FeatureDependency] = {}
        self.dependency_reference_counter: Dict[str, int] = {}
        self.classifier = LogisticRegression(random_state=seed)

    @property
    def columns(self) -> List[str]:
        """Returns the list of column names corresponding to the features added to the calibrator.

        Returns:
            List[str]: A list of column names representing the features used for calibration.
        """
        return [
            column
            for feature in self.feature_dict.values()
            for column in feature.columns
        ]

    @property
    def features(self) -> List[str]:
        """Get the list of features added to the calibrator.

        Returns:
            List[str]: The list of feature names
        """
        return list(self.feature_dict.keys())

    def add_feature(self, feature: CalibrationFeatures) -> None:
        """Add a feature for the classifier used for calibration.

        This method ensures that the feature is unique and its dependencies are tracked.

        Args:
            feature (CalibrationFeatures): The feature to be added to the calibrator.
        """
        if feature.name not in self.feature_dict:
            self.feature_dict[feature.name] = feature
            for dependency in feature.dependencies:
                if dependency.name in self.dependencies:
                    self.dependency_reference_counter[dependency.name] += 1
                else:
                    self.dependencies[dependency.name] = dependency
                    self.dependency_reference_counter[dependency.name] = 1
        else:
            raise KeyError(f"Feature {feature.name} in feature set.")

    def add_features(self, features: List[CalibrationFeatures]) -> None:
        """Add features for the classifier used for calibration.

        Args:
            features (List[CalibrationFeatures]): A list of features to be added to the calibrator.
        """
        for feature in features:
            self.add_feature(feature)

    def remove_feature(self, name: str) -> None:
        """Remove a feature for the classifier used for calibration.

        This method also removes any dependencies that are no longer required.

        Args:
            name (str): The name of the feature to be removed.
        """
        feature = self.feature_dict.pop(name)
        for dependency in feature.dependencies:
            self.dependency_reference_counter[dependency.name] -= 1
            if self.dependency_reference_counter[dependency.name] == 0:
                self.dependency_reference_counter.pop(dependency.name)
                self.dependencies.pop(dependency.name)

    def fit(self, dataset: CalibrationDataset) -> None:
        """Fit the logistic regression model using the given calibration dataset.

        This method computes the features from the dataset, applies log-odds transformation to the confidence scores,
        and trains a logistic regression model for recalibrating probabilities.

        Args:
            dataset (CalibrationDataset): The dataset used for training the classifier.
        """
        result = self.compute_features(dataset=dataset, labelled=True)
        if len(result) != 3:
            raise ValueError("Expected 3 values from compute_features in fit mode")
        confidences, other_features, labels = result

        # Apply log-odds transformation to confidence scores
        log_s = np.log(confidences)
        log_1_minus_s = np.log(1 - confidences)

        # Combine transformed confidence scores with other features
        z = np.column_stack([log_s, log_1_minus_s, other_features])

        self.classifier.fit(z, labels)

    def compute_features(
        self, dataset: CalibrationDataset, labelled: bool
    ) -> Union[
        Tuple[Float[Array, "N"], Float[Array, "N M"]],
        Tuple[Float[Array, "N"], Float[Array, "N M"], Float[Array, "N"]],
    ]:
        """Compute the features for the dataset, including any dependencies and feature calculations.

        This method handles both labelled and unlabelled datasets. It computes the necessary features and returns them for model training or prediction.
        The confidence scores are returned separately from other features to allow for log-odds transformation.

        Args:
            dataset (CalibrationDataset): The dataset from which features are computed.
            labelled (bool): Whether the dataset contains labels for supervised learning.

        Returns:
            Union[
                Tuple[Float[Array, "N"], Float[Array, "N M"]],
                Tuple[Float[Array, "N"], Float[Array, "N M"], Float[Array, "N"]]
            ]:
                - If `labelled` is True: A tuple containing (confidence scores, other features, labels)
                - If `labelled` is False: A tuple containing (confidence scores, other features)
        """
        for dependency in self.dependencies.values():
            dependency.compute(dataset=dataset)

        for feature in self.feature_dict.values():
            if labelled:
                feature.prepare(dataset=dataset)
            feature.compute(dataset=dataset)

        # Get confidence scores separately
        confidences = dataset.metadata[dataset.confidence_column].values

        # Get other features
        if self.columns:
            other_features = dataset.metadata[self.columns].values
        else:
            other_features = np.zeros((len(dataset.metadata), 0))

        if labelled:
            labels = dataset.metadata["correct"].values
            return confidences, other_features, labels
        else:
            return confidences, other_features

    def predict(self, dataset: CalibrationDataset) -> None:
        """Predict the calibrated probabilities for a given dataset.

        This method computes the features, applies log-odds transformation to the confidence scores,
        and uses the trained classifier to predict the calibrated probabilities for the dataset.
        The calibrated probabilities are stored in the dataset under the "calibrated_confidence" column.

        Args:
            dataset (CalibrationDataset): The dataset for which predictions are made.
        """
        result = self.compute_features(dataset=dataset, labelled=False)
        if len(result) != 2:
            raise ValueError("Expected 2 values from compute_features in predict mode")
        confidences, other_features = result

        # Apply log-odds transformation to confidence scores
        log_s = np.log(confidences)
        log_1_minus_s = np.log(1 - confidences)

        # Combine transformed confidence scores with other features
        z = np.column_stack([log_s, log_1_minus_s, other_features])

        correct_probs = self.classifier.predict_proba(z)
        dataset.metadata["calibrated_confidence"] = correct_probs[:, 1].tolist()
