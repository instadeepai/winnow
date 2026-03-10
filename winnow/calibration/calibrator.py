"""Contains classes and functions for probability recalibration."""

from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
from dataclasses import dataclass
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from numpy.typing import NDArray
from huggingface_hub import snapshot_download
from omegaconf import DictConfig

from winnow.calibration.calibration_features import (
    CalibrationFeatures,
    FeatureDependency,
)
from winnow.datasets.calibration_dataset import CalibrationDataset


@dataclass
class TrainingHistory:
    """Container for training history metrics from calibrator fitting.

    Attributes:
        loss_curve: List of training loss values at each iteration.
        validation_scores: List of validation scores at each iteration (only if early_stopping=True).
        final_training_loss: The final training loss value.
        final_validation_score: The final validation score (only if early_stopping=True).
        n_iter: Number of iterations the solver ran.
    """

    loss_curve: List[float]
    validation_scores: Optional[List[float]]
    final_training_loss: float
    final_validation_score: Optional[float]
    n_iter: int

    def save(self, path: Union[Path, str]) -> None:
        """Save the training history to a JSON file.

        Args:
            path: Path to save the JSON file.
        """
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "loss_curve": self.loss_curve,
            "validation_scores": self.validation_scores,
            "final_training_loss": self.final_training_loss,
            "final_validation_score": self.final_validation_score,
            "n_iter": self.n_iter,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[Path, str]) -> "TrainingHistory":
        """Load training history from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            TrainingHistory: The loaded training history.
        """
        import json

        with open(path) as f:
            data = json.load(f)

        return cls(
            loss_curve=data["loss_curve"],
            validation_scores=data.get("validation_scores"),
            final_training_loss=data["final_training_loss"],
            final_validation_score=data.get("final_validation_score"),
            n_iter=data["n_iter"],
        )

    def plot(
        self,
        output_path: Optional[Union[Path, str]] = None,
        show: bool = False,
    ) -> None:
        """Plot the training and validation loss curves.

        Creates a visualization of the training progress showing the loss curve
        and validation scores (if available) over training iterations.

        Args:
            output_path (Optional[Union[Path, str]]): Path to save the plot image.
                If None, the plot is not saved. Defaults to None.
            show (bool): Whether to display the plot interactively. Defaults to False.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        iterations = range(1, len(self.loss_curve) + 1)
        ax.plot(
            iterations,
            self.loss_curve,
            label="Training Loss",
            color="#2563eb",
            linewidth=2,
        )

        if self.validation_scores is not None:
            # Validation scores are accuracy-like (higher is better), so we plot them on a secondary axis
            ax2 = ax.twinx()
            ax2.plot(
                iterations,
                self.validation_scores,
                label="Validation Score",
                color="#dc2626",
                linewidth=2,
                linestyle="--",
            )
            ax2.set_ylabel("Validation Score", color="#dc2626", fontsize=12)
            ax2.tick_params(axis="y", labelcolor="#dc2626")
            ax2.legend(loc="upper right")

        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Training Loss", color="#2563eb", fontsize=12)
        ax.tick_params(axis="y", labelcolor="#2563eb")
        ax.set_title("Calibrator Training Progress", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # Add text annotation with final metrics
        final_text = f"Final Training Loss: {self.final_training_loss:.6f}"
        if self.final_validation_score is not None:
            final_text += f"\nFinal Validation Score: {self.final_validation_score:.6f}"
        final_text += f"\nIterations: {self.n_iter}"

        ax.text(
            0.02,
            0.02,
            final_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        plt.tight_layout()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)


class ProbabilityCalibrator:
    """A class for recalibrating probabilities for a de novo peptide sequencing method.

    This class provides functionality to recalibrate predicted probabilities by fitting an MLP classifier using various features computed from a calibration dataset.
    """

    def __init__(
        self,
        seed: int = 42,
        features: Optional[
            Union[List[CalibrationFeatures], Dict[str, CalibrationFeatures], DictConfig]
        ] = None,
        hidden_layer_sizes: Tuple[int, ...] = (50, 50),
        learning_rate_init: float = 0.001,
        alpha: float = 0.0001,
        max_iter: int = 1000,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
    ) -> None:
        """Initialise the probability calibrator.

        Args:
            seed (int): Random seed for the classifier. Defaults to 42.
            features (Optional[Union[List[CalibrationFeatures], Dict[str, CalibrationFeatures], DictConfig]]):
                Features to add to the calibrator. Can be a list or dict of CalibrationFeatures objects.
                If None, no features are added. Defaults to None.
            hidden_layer_sizes (Tuple[int, ...]): The number of neurons in each hidden layer. Defaults to (50, 50).
            learning_rate_init (float): The initial learning rate. Defaults to 0.001.
            alpha (float): L2 regularisation parameter. Defaults to 0.0001.
            max_iter (int): Maximum number of training iterations. Defaults to 1000.
            early_stopping (bool): Whether to use early stopping to terminate training. Defaults to True.
            validation_fraction (float): Proportion of training data to use for early stopping validation. Defaults to 0.1.
        """
        self.feature_dict: Dict[str, CalibrationFeatures] = {}
        self.dependencies: Dict[str, FeatureDependency] = {}
        self.dependency_reference_counter: Dict[str, int] = {}
        self.classifier = MLPClassifier(
            random_state=seed,
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=learning_rate_init,
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
        )
        self.scaler = StandardScaler()

        # Add features if provided
        if features is not None:
            if isinstance(features, (dict, DictConfig)):
                self.add_features(list(features.values()))
            else:
                self.add_features(list(features))

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
    def save(
        cls, calibrator: "ProbabilityCalibrator", dir_path: Union[Path, str]
    ) -> None:
        """Save the calibrator to a file.

        Args:
            calibrator (ProbabilityCalibrator): The calibrator to save.
            dir_path (Path): The path to the directory where the calibrator checkpoint will be saved.
        """
        if isinstance(dir_path, str):
            dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        pickle.dump(calibrator, open(dir_path / "calibrator.pkl", "wb"))

    @classmethod
    def load(
        cls,
        pretrained_model_name_or_path: Union[
            Path, str
        ] = "InstaDeepAI/winnow-general-model",
        cache_dir: Optional[Path] = None,
    ) -> "ProbabilityCalibrator":
        """Load a pretrained calibrator from a local path or Hugging Face repository. If the path is a local directory path, it will be used directly. If it is a Hugging Face repository identifier, it will be downloaded from Hugging Face.

        Args:
            pretrained_model_name_or_path (Union[Path, str]): The local directory path (e.g., "./my-model-directory") or the Hugging Face repository identifier (e.g., "InstaDeepAI/winnow-general-model").
            cache_dir (Optional[Path]): Directory to cache the Hugging Face model.
        """
        dir_path = Path(pretrained_model_name_or_path)

        # If the path exists locally, use it directly.
        if dir_path.exists():
            # Resolve relative paths to absolute, canonical paths
            dir_path = dir_path.resolve()
        # Otherwise download it from Hugging Face.
        else:
            # If no cache directory is provided, use the default cache directory.
            if cache_dir is None:
                cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            # Download from Hugging Face
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

    def fit(self, dataset: CalibrationDataset) -> TrainingHistory:
        """Fit the MLP classifier using the given calibration dataset.

        This method computes the features from the dataset, prepares the labels, and trains an MLP classifier for recalibrating probabilities.

        Args:
            dataset (CalibrationDataset): The dataset used for training the classifier.

        Returns:
            TrainingHistory: A dataclass containing training metrics including loss curves
                and validation scores (if early_stopping is enabled).
        """
        features, labels = self.compute_features(dataset=dataset, labelled=True)
        # Fit and transform features with scaler
        features_scaled = self.scaler.fit_transform(features)
        self.classifier.fit(features_scaled, labels)

        # Extract training history from the fitted classifier
        loss_curve = list(self.classifier.loss_curve_)
        final_training_loss = loss_curve[-1] if loss_curve else float("nan")

        # Validation scores are only available if early_stopping was enabled
        validation_scores: Optional[List[float]] = None
        final_validation_score: Optional[float] = None
        if (
            hasattr(self.classifier, "validation_scores_")
            and self.classifier.validation_scores_
        ):
            validation_scores = list(self.classifier.validation_scores_)
            final_validation_score = (
                validation_scores[-1] if validation_scores else None
            )

        return TrainingHistory(
            loss_curve=loss_curve,
            validation_scores=validation_scores,
            final_training_loss=final_training_loss,
            final_validation_score=final_validation_score,
            n_iter=self.classifier.n_iter_,
        )

    def compute_features(
        self, dataset: CalibrationDataset, labelled: bool
    ) -> Union[
        NDArray[np.float64],
        Tuple[NDArray[np.float64], NDArray[np.float64]],
    ]:
        """Compute the features for the dataset, including any dependencies and feature calculations.

        This method handles both labelled and unlabelled datasets. It computes the necessary features and returns them for model training or prediction.

        Args:
            dataset (CalibrationDataset): The dataset from which features are computed.
            labelled (bool): Whether the dataset contains labels for supervised learning.

        Returns:
            Union[
                NDArray[np.float64],
                Tuple[NDArray[np.float64], NDArray[np.float64]]
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
