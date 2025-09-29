"""Contains classes and functions for probability recalibration."""

from typing import Dict, List, Tuple, Union
from pathlib import Path
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from jaxtyping import Float

from winnow.calibration.calibration_features import (
    CalibrationFeatures,
    FeatureDependency,
    PrositFeatures,
    MassErrorFeature,
    RetentionTimeFeature,
    ChimericFeatures,
    BeamFeatures,
)
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.constants import RESIDUE_MASSES


class ProbabilityCalibrator:
    """A class for recalibrating probabilities for a de novo peptide sequencing method.

    This class provides functionality to recalibrate predicted probabilities by fitting a logistic regression model using various features computed from a calibration dataset.
    """

    def __init__(self, seed: int = 42) -> None:
        self.feature_dict: Dict[str, CalibrationFeatures] = {}
        self.dependencies: Dict[str, FeatureDependency] = {}
        self.dependency_reference_counter: Dict[str, int] = {}
        self.classifier = MLPClassifier(
            random_state=seed,
            hidden_layer_sizes=(50, 50),
            learning_rate_init=0.001,
            alpha=0.0001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
        )
        self.scaler = StandardScaler()

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

    @classmethod
    def save(cls, calibrator: "ProbabilityCalibrator", path: Path) -> None:
        """Save the calibrator to a file.

        Args:
            calibrator (ProbabilityCalibrator): The calibrator to save.
            path (Path): The path to save the calibrator to.
        """
        path.mkdir(parents=True)
        calibrator_classifier_path = path / "calibrator.pkl"
        irt_predictor_path = path / "irt_predictor.pkl"
        scaler_path = path / "scaler.pkl"

        with calibrator_classifier_path.open(mode="wb") as f:
            pickle.dump(calibrator.classifier, f)

        if "Prosit iRT Features" in calibrator.feature_dict:
            with irt_predictor_path.open(mode="wb") as f:
                pickle.dump(
                    calibrator.feature_dict["Prosit iRT Features"].irt_predictor, f
                )

        with scaler_path.open(mode="wb") as f:
            pickle.dump(calibrator.scaler, f)

    @classmethod
    def load(cls, path: Path) -> "ProbabilityCalibrator":
        """Load the calibrator from a file.

        Args:
            path (Path): The path to load the calibrator from.

        Returns:
            ProbabilityCalibrator: A new instance of the calibrator loaded from the file.
        """
        calibrator = cls()

        # Initialise the features that were used when saving
        calibrator.add_feature(MassErrorFeature(residue_masses=RESIDUE_MASSES))
        calibrator.add_feature(
            PrositFeatures(mz_tolerance=0.02)
        )  # Default value, should match training
        calibrator.add_feature(
            RetentionTimeFeature(hidden_dim=10, train_fraction=0.1)
        )  # Default values
        calibrator.add_feature(ChimericFeatures(mz_tolerance=0.02))  # Default value
        calibrator.add_feature(BeamFeatures())

        # Now load the saved data
        calibrator.load_classifier(path / "calibrator.pkl")
        calibrator.load_irt_predictor(path / "irt_predictor.pkl")
        calibrator.load_scaler(path / "scaler.pkl")
        return calibrator

    def load_classifier(self, path: Path) -> None:
        """Load the classifier from a file.

        Args:
            path (Path): The path to load the classifier from.
        """
        with path.open(mode="rb") as f:
            self.classifier = pickle.load(f)

    def load_irt_predictor(self, path: Path) -> None:
        """Load the iRT predictor from a file.

        Args:
            path (Path): The path to load the iRT predictor from.
        """
        with path.open(mode="rb") as f:
            self.feature_dict["Prosit iRT Features"].irt_predictor = pickle.load(f)  # type: ignore

    def load_scaler(self, path: Path) -> None:
        """Load the scaler from a file.

        Args:
            path (Path): The path to load the scaler from.
        """
        with path.open(mode="rb") as f:
            self.scaler = pickle.load(f)

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

        This method computes the features from the dataset, prepares the labels, and trains a logistic regression model for recalibrating probabilities.

        Args:
            dataset (CalibrationDataset): The dataset used for training the classifier.
        """
        features, labels = self.compute_features(dataset=dataset, labelled=True)
        # Fit and transform features with scaler
        features_scaled = self.scaler.fit_transform(features)
        self.classifier.fit(features_scaled, labels)

    def compute_features(
        self, dataset: CalibrationDataset, labelled: bool
    ) -> Union[
        Float[np.ndarray, "batch feature"],
        Tuple[Float[np.ndarray, "batch feature"], Float[np.ndarray, "batch"]],  # noqa: F821
    ]:
        """Compute the features for the dataset, including any dependencies and feature calculations.

        This method handles both labelled and unlabelled datasets. It computes the necessary features and returns them for model training or prediction.

        Args:
            dataset (CalibrationDataset): The dataset from which features are computed.
            labelled (bool): Whether the dataset contains labels for supervised learning.

        Returns:
            Union[
                Float[np.ndarray, "batch feature"],
                Tuple[Float[np.ndarray, "batch feature"], Float[np.ndarray, "batch"]]
            ]:
                - If `labelled` is True: A tuple containing the computed feature matrix and the corresponding labels.
                - If `labelled` is False: Only the computed feature matrix.
        """
        for dependency in self.dependencies.values():
            dependency.compute(dataset=dataset)

        for feature in self.feature_dict.values():
            if labelled:
                feature.prepare(dataset=dataset)
            feature.compute(dataset=dataset)

        feature_columns = [dataset.confidence_column]
        feature_columns.extend(self.columns)
        features = dataset.metadata[feature_columns]

        if labelled:
            labels = dataset.metadata["correct"]
            return features.values, labels.values
        else:
            return features.values

    def predict(self, dataset: CalibrationDataset) -> None:
        """Predict the calibrated probabilities for a given dataset.

        This method computes the features and uses the trained classifier to predict the calibrated probabilities for the dataset. The calibrated probabilities are stored in the dataset under the "calibrated_confidence" column.

        Args:
            dataset (CalibrationDataset): The dataset for which predictions are made.
        """
        features = self.compute_features(dataset=dataset, labelled=False)
        # Transform features with scaler
        features_scaled = self.scaler.transform(features)
        correct_probs = self.classifier.predict_proba(features_scaled)
        dataset.metadata["calibrated_confidence"] = correct_probs[:, 1].tolist()
