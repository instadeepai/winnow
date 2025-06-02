"""Contains classes and functions for probability recalibration."""

from typing import Dict, List, Tuple, Union, Optional
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
from winnow.datasets.calibration_dataset import CalibrationDataset, RESIDUE_MASSES


class ProbabilityCalibrator:
    """A class for recalibrating probabilities for a de novo peptide sequencing method.

    This class provides functionality to recalibrate predicted probabilities by fitting a logistic regression model using various features computed from a calibration dataset.
    It also supports label shift correction using the Saerens-style EM algorithm.
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
        )
        self.scaler = StandardScaler()
        self.prior_train: Optional[float] = None
        self.prior_test: Optional[float] = None
        self.prior_test_source: Optional[str] = None  # 'true_labels' or 'em_estimate'

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
        prior_path = path / "prior.pkl"

        with calibrator_classifier_path.open(mode="wb") as f:
            pickle.dump(calibrator.classifier, f)

        with irt_predictor_path.open(mode="wb") as f:
            pickle.dump(calibrator.feature_dict["Prosit iRT Features"].irt_predictor, f)  # type: ignore

        with scaler_path.open(mode="wb") as f:
            pickle.dump(calibrator.scaler, f)

        with prior_path.open(mode="wb") as f:
            pickle.dump({"prior_train": calibrator.prior_train}, f)

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
        calibrator.load_prior(path / "prior.pkl")
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

    def load_prior(self, path: Path) -> None:
        """Load the prior from a file.

        Args:
            path (Path): The path to load the prior from.
        """
        with path.open(mode="rb") as f:
            prior_data = pickle.load(f)
            self.prior_train = prior_data["prior_train"]

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

        This method computes the features from the dataset, prepares the labels, and trains a binary classifier for recalibrating probabilities.
        It also estimates and stores the training prior.

        Args:
            dataset (CalibrationDataset): The dataset used for training the classifier.
        """
        features, labels = self.compute_features(dataset=dataset, labelled=True)
        # Fit and transform features with scaler
        features_scaled = self.scaler.fit_transform(features)
        self.classifier.fit(features_scaled, labels)

        # Estimate and store training prior
        self.prior_train = float(np.mean(labels))

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

    def predict(
        self,
        dataset: CalibrationDataset,
        correct_label_shift: bool = True,
        max_iters: int = 20,
        tol: float = 1e-4,
        use_true_labels_for_prior: bool = False,
    ) -> None:
        """Predict the calibrated probabilities for a given dataset.

        This method computes the features and uses the trained classifier to predict the calibrated probabilities for the dataset.
        If correct_label_shift is True, it also applies prior shift correction using either:
        - True labels (if available and use_true_labels_for_prior is True)
        - EM algorithm (if true labels are not available or use_true_labels_for_prior is False)
        The calibrated probabilities are stored in the dataset under the "calibrated_confidence" column.

        Args:
            dataset (CalibrationDataset): The dataset for which predictions are made.
            correct_label_shift (bool, optional): Whether to apply label shift correction. Defaults to True.
            max_iters (int, optional): Maximum number of EM iterations. Defaults to 20.
            tol (float, optional): Convergence tolerance for EM. Defaults to 1e-4.
            use_true_labels_for_prior (bool, optional): Whether to use true labels to calculate test prior if available. Defaults to False.
        """
        features = self.compute_features(dataset=dataset, labelled=False)
        # Transform features with scaler
        features_scaled = self.scaler.transform(features)
        p_raw = self.classifier.predict_proba(features_scaled)[:, 1]

        if correct_label_shift and self.prior_train is not None:
            # Try to use true labels for prior if requested and available
            if use_true_labels_for_prior and "correct" in dataset.metadata.columns:
                self.prior_test = float(np.mean(dataset.metadata["correct"]))
                self.prior_test_source = "true_labels"
                p_adapted = self._update_posterior(
                    p_raw, self.prior_train, self.prior_test
                )
            else:
                # Fall back to EM estimation
                p_adapted, self.prior_test = self._em_prior_shift_loop(
                    p_raw=p_raw,
                    prior_train=self.prior_train,
                    max_iters=max_iters,
                    tol=tol,
                )
                self.prior_test_source = "em_estimate"
            dataset.metadata["calibrated_confidence"] = p_adapted.tolist()
        else:
            dataset.metadata["calibrated_confidence"] = p_raw.tolist()

    def _update_posterior(
        self, p_raw: np.ndarray, prior_train: float, prior_test: float
    ) -> np.ndarray:
        """Update posteriors to account for prior shift using the Saerens formula.

        Args:
            p_raw (np.ndarray): Raw posterior probabilities under training prior
            prior_train (float): Training set prior probability
            prior_test (float): Test set prior probability

        Returns:
            np.ndarray: Adapted posterior probabilities
        """
        alpha = prior_test / prior_train
        beta = (1 - prior_test) / (1 - prior_train)
        numerator = alpha * p_raw
        denominator = alpha * p_raw + beta * (1 - p_raw)
        return numerator / denominator

    def _update_prior(self, p_adapted: np.ndarray) -> float:
        """Update the test prior based on mean of adapted posteriors.

        Args:
            p_adapted (np.ndarray): Adapted posterior probabilities

        Returns:
            float: New estimate of test prior
        """
        return float(np.mean(p_adapted))

    def _em_prior_shift_loop(
        self,
        p_raw: np.ndarray,
        prior_train: float,
        max_iters: int = 20,
        tol: float = 1e-4,
    ) -> Tuple[np.ndarray, float]:
        """Run EM until prior_test converges or max iterations reached.

        Args:
            p_raw (np.ndarray): Raw posterior probabilities
            prior_train (float): Training set prior probability
            max_iters (int, optional): Maximum number of EM iterations. Defaults to 20.
            tol (float, optional): Convergence tolerance. Defaults to 1e-4.

        Returns:
            Tuple[np.ndarray, float]: Final adapted posteriors and estimated test prior
        """
        prior_test = prior_train

        for _ in range(max_iters):
            # E-step: get adapted posteriors
            p_adapted = self._update_posterior(p_raw, prior_train, prior_test)

            # M-step: compute new prior_test
            new_prior_test = self._update_prior(p_adapted)

            # Check convergence
            if abs(new_prior_test - prior_test) < tol:
                prior_test = new_prior_test
                break

            prior_test = new_prior_test

        return p_adapted, prior_test
