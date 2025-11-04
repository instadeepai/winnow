"""Contains classes and functions for probability recalibration."""

from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from jaxtyping import Float
from huggingface_hub import snapshot_download
from winnow.calibration.calibration_features import (
    CalibrationFeatures,
    FeatureDependency,
)
from winnow.datasets.calibration_dataset import CalibrationDataset


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
    def save(cls, calibrator: "ProbabilityCalibrator", dir_path: Path) -> None:
        """Save the calibrator to a file.

        Args:
            calibrator (ProbabilityCalibrator): The calibrator to save.
            dir_path (Path): The path to the directory where the calibrator checkpoint will be saved.
        """
        dir_path.mkdir(parents=True)
        pickle.dump(calibrator, open(dir_path / "calibrator.pkl", "wb"))

    @classmethod
    def load(
        cls,
        pretrained_model_name_or_path: Union[
            Path, str
        ] = "InstaDeepAI/winnow-general-model",
        cache_dir: Optional[Path] = None,
    ) -> "ProbabilityCalibrator":
        """Load a pretrained calibrator from a local path or HuggingFace repository. If the path is a local directory path, it will be used directly. If it is a HuggingFace repository identifier, it will be downloaded from HuggingFace.

        Args:
            pretrained_model_name_or_path (Union[Path, str]): The local directory path (e.g., Path("./my-model-directory")) or the HuggingFace repository identifier (e.g., "InstaDeepAI/winnow-general-model").
            cache_dir (Optional[Path]): Directory to cache the HuggingFace model.
        """
        dir_path = Path(pretrained_model_name_or_path)

        # If the path exists locally, use it directly.
        if dir_path.exists():
            # Resolve relative paths to absolute, canonical paths
            dir_path = dir_path.resolve()
        # Otherwise download it from HuggingFace.
        else:
            # If no cache directory is provided, use the default cache directory.
            if cache_dir is None:
                cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            # Download from HuggingFace
            dir_path = Path(
                snapshot_download(
                    repo_id=pretrained_model_name_or_path,
                    repo_type="model",
                    cache_dir=cache_dir,
                )
            )

        # Load the calibrator object
        loaded_obj = pickle.load(open(dir_path / "calibrator.pkl", "rb"))

        # Check if this is a legacy checkpoint (MLPClassifier instead of ProbabilityCalibrator)
        if isinstance(loaded_obj, MLPClassifier):
            error_msg = (
                "Legacy checkpoint format detected. The checkpoint directory contains "
                "an old format where calibrator.pkl contains only the MLPClassifier "
                "instead of the full ProbabilityCalibrator object.\n"
                "Legacy checkpoints cannot be automatically migrated because they lack "
                "the feature and dependency information required by the current version. "
                "We cannot correctly infer the trained feature set with old versions.\n"
                "To resolve this, retrain the calibrator using the current version with your training dataset. "
                "The new format will save the complete ProbabilityCalibrator object including all features and dependencies."
            )
            raise ValueError(error_msg)

        elif not isinstance(loaded_obj, ProbabilityCalibrator):
            raise ValueError(
                f"Loaded object is of type {type(loaded_obj).__name__}, expected ProbabilityCalibrator."
            )
        else:
            return loaded_obj

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
