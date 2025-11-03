"""Unit tests for winnow ProbabilityCalibrator."""

import pickle
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from winnow.calibration.calibrator import ProbabilityCalibrator
from winnow.calibration.calibration_features import (
    CalibrationFeatures,
    FeatureDependency,
)
from winnow.datasets.calibration_dataset import CalibrationDataset


class MockFeatureDependency(FeatureDependency):
    """Mock feature dependency for testing."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def compute(self, dataset: CalibrationDataset):
        # Add mock data to dataset - ensure it matches dataset length
        dataset.metadata[f"{self.name}_data"] = list(range(len(dataset.metadata)))


class MockCalibrationFeature(CalibrationFeatures):
    """Mock calibration feature for testing."""

    def __init__(self, name: str, columns=None, dependencies=None):
        self._name = name
        self._columns = columns or [name]
        self._dependencies = dependencies or []

    @property
    def dependencies(self):
        return self._dependencies

    @property
    def name(self):
        return self._name

    @property
    def columns(self):
        return self._columns

    def prepare(self, dataset):
        pass

    def compute(self, dataset):
        # Add mock feature data
        for col in self.columns:
            dataset.metadata[col] = np.random.random(len(dataset.metadata))


class TestProbabilityCalibrator:
    """Test the ProbabilityCalibrator class."""

    @pytest.fixture()
    def calibrator(self):
        """Create a ProbabilityCalibrator instance for testing."""
        return ProbabilityCalibrator(seed=42)

    @pytest.fixture()
    def sample_dataset(self):
        """Create a sample CalibrationDataset for testing."""
        metadata = pd.DataFrame(
            {"confidence": [0.9, 0.8, 0.7, 0.6, 0.5], "other_col": [1, 2, 3, 4, 5]}
        )
        return CalibrationDataset(metadata=metadata, predictions=[None] * 5)

    @pytest.fixture()
    def labelled_dataset(self):
        """Create a dataset with labels for supervised learning."""
        # Create a larger dataset to support early stopping with validation_fraction=0.1
        # Need at least 40 samples to have 4+ validation samples for stratified split
        n_samples = 50
        np.random.seed(42)  # For reproducible test data

        metadata = pd.DataFrame(
            {
                "confidence": np.random.uniform(0.1, 0.99, n_samples),
                "correct": np.random.choice([0, 1], n_samples),
                "feature1": np.random.uniform(1.0, 10.0, n_samples),
                "feature2": np.random.uniform(0.1, 1.0, n_samples),
            }
        )
        return CalibrationDataset(metadata=metadata, predictions=[None] * n_samples)

    def test_initialization(self, calibrator):
        """Test ProbabilityCalibrator initialisation."""
        assert isinstance(calibrator.feature_dict, dict)
        assert isinstance(calibrator.dependencies, dict)
        assert isinstance(calibrator.dependency_reference_counter, dict)
        assert len(calibrator.feature_dict) == 0
        assert len(calibrator.dependencies) == 0

    def test_initialization_with_custom_seed(self):
        """Test calibrator initialisation with custom seed."""
        calibrator = ProbabilityCalibrator(seed=123)
        # The seed should be passed to the MLPClassifier
        assert calibrator.classifier.random_state == 123

    def test_columns_property_empty(self, calibrator):
        """Test columns property when no features are added."""
        assert calibrator.columns == []

    def test_features_property_empty(self, calibrator):
        """Test features property when no features are added."""
        assert calibrator.features == []

    def test_add_feature_basic(self, calibrator):
        """Test adding a basic feature without dependencies."""
        feature = MockCalibrationFeature("test_feature", ["col1", "col2"])
        calibrator.add_feature(feature)

        assert "test_feature" in calibrator.feature_dict
        assert calibrator.feature_dict["test_feature"] == feature
        assert calibrator.columns == ["col1", "col2"]
        assert calibrator.features == ["test_feature"]

    def test_add_feature_with_dependencies(self, calibrator):
        """Test adding a feature with dependencies."""
        dependency = MockFeatureDependency("test_dep")
        feature = MockCalibrationFeature("test_feature", ["col1"], [dependency])

        calibrator.add_feature(feature)

        assert "test_feature" in calibrator.feature_dict
        assert "test_dep" in calibrator.dependencies
        assert calibrator.dependency_reference_counter["test_dep"] == 1

    def test_add_multiple_features_shared_dependency(self, calibrator):
        """Test adding multiple features that share a dependency."""
        dependency = MockFeatureDependency("shared_dep")
        feature1 = MockCalibrationFeature("feature1", ["col1"], [dependency])
        feature2 = MockCalibrationFeature("feature2", ["col2"], [dependency])

        calibrator.add_feature(feature1)
        calibrator.add_feature(feature2)

        assert calibrator.dependency_reference_counter["shared_dep"] == 2
        assert len(calibrator.dependencies) == 1  # Only one dependency instance

    def test_add_duplicate_feature_raises_error(self, calibrator):
        """Test that adding a duplicate feature raises KeyError."""
        feature = MockCalibrationFeature("duplicate_feature")
        calibrator.add_feature(feature)

        with pytest.raises(KeyError, match="Feature duplicate_feature in feature set"):
            calibrator.add_feature(feature)

    def test_remove_feature_basic(self, calibrator):
        """Test removing a feature without dependencies."""
        feature = MockCalibrationFeature("removable_feature")
        calibrator.add_feature(feature)

        assert "removable_feature" in calibrator.feature_dict

        calibrator.remove_feature("removable_feature")

        assert "removable_feature" not in calibrator.feature_dict
        assert calibrator.columns == []

    def test_remove_feature_with_dependencies(self, calibrator):
        """Test removing a feature with dependencies."""
        dependency = MockFeatureDependency("removable_dep")
        feature = MockCalibrationFeature("removable_feature", dependencies=[dependency])

        calibrator.add_feature(feature)
        calibrator.remove_feature("removable_feature")

        assert "removable_feature" not in calibrator.feature_dict
        assert "removable_dep" not in calibrator.dependencies
        assert "removable_dep" not in calibrator.dependency_reference_counter

    def test_remove_feature_shared_dependency(self, calibrator):
        """Test removing one feature when dependency is shared."""
        dependency = MockFeatureDependency("shared_dep")
        feature1 = MockCalibrationFeature("feature1", dependencies=[dependency])
        feature2 = MockCalibrationFeature("feature2", dependencies=[dependency])

        calibrator.add_feature(feature1)
        calibrator.add_feature(feature2)

        # Remove one feature - dependency should remain
        calibrator.remove_feature("feature1")

        assert "feature1" not in calibrator.feature_dict
        assert "feature2" in calibrator.feature_dict
        assert "shared_dep" in calibrator.dependencies  # Still needed by feature2
        assert calibrator.dependency_reference_counter["shared_dep"] == 1

    def test_remove_nonexistent_feature_raises_error(self, calibrator):
        """Test that removing a nonexistent feature raises KeyError."""
        with pytest.raises(KeyError):
            calibrator.remove_feature("nonexistent")

    def test_compute_features_unlabelled(self, calibrator, sample_dataset):
        """Test computing features for unlabelled dataset."""
        feature = MockCalibrationFeature("test_feature", ["test_col"])
        calibrator.add_feature(feature)

        features = calibrator.compute_features(sample_dataset, labelled=False)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(sample_dataset.metadata)
        assert features.shape[1] == 2  # confidence column + one feature column

    def test_compute_features_labelled(self, calibrator, labelled_dataset):
        """Test computing features for labelled dataset."""
        feature = MockCalibrationFeature("test_feature", ["test_col"])
        calibrator.add_feature(feature)

        features, labels = calibrator.compute_features(labelled_dataset, labelled=True)

        assert isinstance(features, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert features.shape[0] == len(labelled_dataset.metadata)
        assert labels.shape[0] == len(labelled_dataset.metadata)

    def test_compute_features_with_dependencies(self, calibrator, sample_dataset):
        """Test computing features with dependencies."""
        dependency = MockFeatureDependency("test_dep")
        feature = MockCalibrationFeature("test_feature", dependencies=[dependency])
        calibrator.add_feature(feature)

        calibrator.compute_features(sample_dataset, labelled=False)

        # Check that dependency was computed (adds column to dataset)
        assert f"{dependency.name}_data" in sample_dataset.metadata.columns

    def test_fit(self, calibrator, labelled_dataset):
        """Test fitting the calibrator."""
        feature = MockCalibrationFeature("test_feature", ["test_col"])
        calibrator.add_feature(feature)

        # Should not raise any exception
        calibrator.fit(labelled_dataset)

        # Check that scaler and classifier were fitted (basic smoke test)
        assert hasattr(calibrator.scaler, "mean_")  # Scaler fitted
        assert hasattr(calibrator.classifier, "classes_")  # Classifier fitted

    def test_predict_after_fit(self, calibrator, labelled_dataset, sample_dataset):
        """Test prediction after fitting."""
        feature = MockCalibrationFeature("test_feature", ["test_col"])
        calibrator.add_feature(feature)

        # Fit first
        calibrator.fit(labelled_dataset)

        # Then predict
        calibrator.predict(sample_dataset)

        # Check that calibrated confidence was added to dataset
        assert "calibrated_confidence" in sample_dataset.metadata.columns

    def test_predict_without_fit_should_fail(self, calibrator, sample_dataset):
        """Test that prediction fails if calibrator hasn't been fitted."""
        feature = MockCalibrationFeature("test_feature", ["test_col"])
        calibrator.add_feature(feature)

        # Should raise an exception since classifier isn't fitted
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            calibrator.predict(sample_dataset)

    def test_save_creates_directory(self, calibrator, tmp_path):
        """Test saving calibrator creates directory."""
        save_path = tmp_path / "test_calibrator"

        ProbabilityCalibrator.save(calibrator, save_path)

        assert save_path.exists()
        assert save_path.is_dir()
        assert (save_path / "calibrator.pkl").exists()

    def test_save_and_load_preserves_state(
        self, calibrator, labelled_dataset, tmp_path
    ):
        """Test that save and load preserves all calibrator state including features."""
        # Add features with specific parameters
        feature1 = MockCalibrationFeature("test_feature1", ["col1", "col2"])
        feature2 = MockCalibrationFeature("test_feature2", ["col3"])
        calibrator.add_feature(feature1)
        calibrator.add_feature(feature2)

        # Fit the calibrator
        calibrator.fit(labelled_dataset)

        # Save calibrator
        save_path = tmp_path / "test_calibrator"
        ProbabilityCalibrator.save(calibrator, save_path)

        # Load calibrator
        loaded_calibrator = ProbabilityCalibrator.load(save_path)

        # Verify all state is preserved
        assert loaded_calibrator.features == calibrator.features
        assert loaded_calibrator.columns == calibrator.columns
        assert len(loaded_calibrator.feature_dict) == len(calibrator.feature_dict)
        assert len(loaded_calibrator.dependencies) == len(calibrator.dependencies)

        # Verify classifier and scaler are fitted
        assert hasattr(loaded_calibrator.classifier, "classes_")
        assert hasattr(loaded_calibrator.scaler, "mean_")

        # Verify we can make predictions with loaded calibrator
        test_dataset = labelled_dataset
        loaded_calibrator.predict(test_dataset)
        assert "calibrated_confidence" in test_dataset.metadata.columns

    def test_load_nonexistent_path_raises_error(self):
        """Test that loading from nonexistent path raises error."""
        with pytest.raises(FileNotFoundError):
            ProbabilityCalibrator.load(Path("/nonexistent/path"))

    def test_classifier_parameters(self):
        """Test that classifier is initialised with correct parameters."""
        calibrator = ProbabilityCalibrator(seed=42)

        assert calibrator.classifier.random_state == 42
        assert calibrator.classifier.hidden_layer_sizes == (50, 50)
        assert calibrator.classifier.learning_rate_init == 0.001
        assert calibrator.classifier.alpha == 0.0001
        assert calibrator.classifier.max_iter == 1000

    def test_scaler_initialization(self, calibrator):
        """Test that scaler is properly initialised."""
        from sklearn.preprocessing import StandardScaler

        assert isinstance(calibrator.scaler, StandardScaler)

    def test_empty_dataset_handling(self, calibrator):
        """Test handling of empty datasets."""
        empty_metadata = pd.DataFrame({"confidence": []})  # Include confidence column
        empty_dataset = CalibrationDataset(metadata=empty_metadata, predictions=[])

        feature = MockCalibrationFeature("test_feature")
        calibrator.add_feature(feature)

        # Should handle empty dataset gracefully
        features = calibrator.compute_features(empty_dataset, labelled=False)
        assert features.shape[0] == 0

    def test_load_legacy_checkpoint_mlpclassifier_raises_error(self, tmp_path):
        """Test that loading a legacy checkpoint (MLPClassifier) raises a clear error."""
        # Create a legacy checkpoint: calibrator.pkl containing only MLPClassifier
        legacy_path = tmp_path / "legacy_checkpoint"
        legacy_path.mkdir(parents=True)

        from sklearn.neural_network import MLPClassifier

        # Save MLPClassifier directly (legacy format)
        legacy_classifier = MLPClassifier(random_state=42)
        pickle.dump(legacy_classifier, open(legacy_path / "calibrator.pkl", "wb"))

        # Attempt to load - should raise ValueError with helpful message
        with pytest.raises(ValueError, match="Legacy checkpoint format detected"):
            ProbabilityCalibrator.load(legacy_path)

        # Verify the error message contains helpful information
        try:
            ProbabilityCalibrator.load(legacy_path)
        except ValueError as e:
            error_msg = str(e)
            assert "Legacy checkpoint format detected" in error_msg
            assert "MLPClassifier" in error_msg
            assert "cannot correctly infer the trained feature set" in error_msg
            assert "retrain" in error_msg.lower() or "retraining" in error_msg.lower()
