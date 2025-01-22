"""Contains classes and functions for probability recalibration"""

from typing import Dict, List, Tuple, Union


import numpy as np
from sklearn.linear_model import LogisticRegression
from jaxtyping import Float

from winnow.calibration.calibration_features import CalibrationFeatures, FeatureDependency
from winnow.datasets.calibration_dataset import CalibrationDataset


class ProbabilityCalibrator:
    """This class calculates recalibrated probabilities for a de novo sequencing method
    """
    def __init__(self, seed: int = 42) -> None:
        self.feature_dict: Dict[str, CalibrationFeatures] = {}
        self.dependencies: Dict[str, FeatureDependency]= {}
        self.dependency_reference_counter: Dict[str, int] = {}
        self.classifier = LogisticRegression(random_state=seed)

    @property
    def columns(self) -> List[str]:
        return [column for feature in self.feature_dict.values()
                for column in feature.columns]

    @property
    def features(self) -> List[str]:
        """Get the list of features added to the calibrator.

        Returns:
            List[str]: The list of feature names
        """
        return list(self.feature_dict.keys())

    def add_feature(self, feature: CalibrationFeatures) -> None:
        """Add a feature for the classifier used for calibration.
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
            raise KeyError(f'Feature {feature.name} in feature set.')

    def add_features(self, features: List[CalibrationFeatures]) -> None:
        for feature in features:
            self.add_feature(feature)

    def remove_feature(self, name: str) -> None:
        """Remove a feature for the classifier used for calibration.
        """
        feature = self.feature_dict.pop(name)
        for dependency in feature.dependencies:
            self.dependency_reference_counter[dependency.name] -= 1
            if self.dependency_reference_counter[dependency.name] == 0:
                self.dependency_reference_counter.pop(dependency.name)
                self.dependencies.pop(dependency.name)

    def fit(self, dataset: CalibrationDataset) -> None:
        features, labels = self.compute_features(
            dataset=dataset, labelled=True
        )
        self.classifier.fit(features, labels)

    # TODO: make feature calculation out-of-place
    def compute_features(
        self, dataset: CalibrationDataset, labelled: bool
    ) -> Union[
        Float[np.ndarray, "batch feature"],
        Tuple[
            Float[np.ndarray, "batch feature"],
            Float[np.ndarray, "batch"]
        ]
    ]:
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
            labels = dataset.metadata['correct']
            return features.values, labels.values
        else:
            return features.values

    def predict(self, dataset: CalibrationDataset):
        features = self.compute_features(dataset=dataset, labelled=False)
        correct_probs = self.classifier.predict_proba(features)
        dataset.metadata['calibrated_confidence'] = correct_probs[:, 1].tolist()
    
    @staticmethod
    def to_csv(dataset: CalibrationDataset, path: str) -> None:
        dataset.metadata.to_csv(path)
