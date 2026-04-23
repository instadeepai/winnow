"""Unit tests for winnow ProbabilityCalibrator."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from winnow.calibration.calibrator import (
    CalibratorNetwork,
    ProbabilityCalibrator,
    TrainingHistory,
)
from winnow.calibration.calibration_features import (
    CalibrationFeatures,
    FeatureDependency,
)
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.datasets.feature_dataset import FeatureDataset


class MockFeatureDependency(FeatureDependency):
    """Mock feature dependency for testing."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def compute(self, dataset: CalibrationDataset):
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
        for col in self.columns:
            dataset.metadata[col] = np.random.random(len(dataset.metadata))


class MockKoinaFeature(MockCalibrationFeature):
    """Feature with Koina-style model inputs for override tests."""

    def __init__(self):
        super().__init__("mock_koina", ["kcol"])
        self.model_input_constants = {"collision_energies": 20}
        self.model_input_columns = {"fragmentation_types": "frag_col"}


class TestCalibratorNetwork:
    """Test the CalibratorNetwork nn.Module."""

    def test_forward_shape(self):
        """Test that output has correct shape."""
        net = CalibratorNetwork(input_dim=5, hidden_dims=(16, 8))
        x = torch.randn(10, 5)
        out = net(x)
        assert out.shape == (10,)

    def test_single_sample(self):
        """Test with a single sample."""
        net = CalibratorNetwork(input_dim=3, hidden_dims=(4,))
        x = torch.randn(1, 3)
        out = net(x)
        assert out.shape == (1,)

    def test_custom_dropout(self):
        """Test that dropout parameter is accepted."""
        net = CalibratorNetwork(input_dim=3, hidden_dims=(8,), dropout=0.5)
        x = torch.randn(5, 3)
        net.eval()
        out = net(x)
        assert out.shape == (5,)


class TestTrainingHistory:
    """Test TrainingHistory dataclass."""

    def test_save_load_roundtrip(self, tmp_path):
        """Test that save/load produces identical history."""
        history = TrainingHistory(
            train_losses=[0.5, 0.4, 0.3],
            val_losses=[0.6, 0.5, 0.4],
            val_accuracies=[0.7, 0.8, 0.85],
            best_epoch=2,
            epochs_trained=3,
        )
        path = tmp_path / "history.json"
        history.save(path)
        loaded = TrainingHistory.load(path)

        assert loaded.train_losses == history.train_losses
        assert loaded.val_losses == history.val_losses
        assert loaded.val_accuracies == history.val_accuracies
        assert loaded.best_epoch == history.best_epoch
        assert loaded.epochs_trained == history.epochs_trained

    def test_save_without_validation(self, tmp_path):
        """Test save/load when no validation data was used."""
        history = TrainingHistory(
            train_losses=[0.5, 0.4],
            best_epoch=1,
            epochs_trained=2,
        )
        path = tmp_path / "history.json"
        history.save(path)
        loaded = TrainingHistory.load(path)

        assert loaded.val_losses is None
        assert loaded.val_accuracies is None

    def test_json_format(self, tmp_path):
        """Test that saved file is valid JSON with expected keys."""
        history = TrainingHistory(train_losses=[0.5], epochs_trained=1)
        path = tmp_path / "history.json"
        history.save(path)

        with open(path) as f:
            data = json.load(f)

        assert "train_losses" in data
        assert "epochs_trained" in data

    def test_plot_saves_file(self, tmp_path):
        """Test that plot saves an image file."""
        history = TrainingHistory(
            train_losses=[0.5, 0.4, 0.3],
            val_losses=[0.6, 0.5, 0.4],
            val_accuracies=[0.7, 0.8, 0.85],
            best_epoch=2,
            epochs_trained=3,
        )
        plot_path = tmp_path / "plot.png"
        history.plot(output_path=plot_path)
        assert plot_path.exists()


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
    def feature_dataset(self):
        """Create a FeatureDataset for training tests."""
        n = 100
        np.random.seed(42)
        features = np.random.randn(n, 3).astype(np.float32)
        labels = np.random.choice([0.0, 1.0], n).astype(np.float32)
        return FeatureDataset(features=features, labels=labels)

    @pytest.fixture()
    def labelled_dataset(self):
        """Create a labelled CalibrationDataset for compute_features tests."""
        n_samples = 50
        np.random.seed(42)
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
        assert len(calibrator.feature_dict) == 0
        assert calibrator.network is None

    def test_initialization_with_params(self):
        """Test calibrator initialisation with custom parameters."""
        calibrator = ProbabilityCalibrator(
            hidden_dims=(64, 32),
            dropout=0.2,
            learning_rate=0.01,
            seed=123,
        )
        assert calibrator.hidden_dims == (64, 32)
        assert calibrator.dropout == 0.2
        assert calibrator.learning_rate == 0.01
        assert calibrator.seed == 123
        assert calibrator.val_early_stopping_max_psms == 10000
        assert calibrator.val_subsample_seed is None

    def test_apply_koina_model_input_overrides(self):
        """Inference-time Koina constant/column overrides merge into features."""
        calibrator = ProbabilityCalibrator()
        calibrator.add_feature(MockKoinaFeature())
        calibrator.apply_koina_model_input_overrides(
            model_input_constants={"collision_energies": 30},
        )
        feat = calibrator.feature_dict["mock_koina"]
        assert feat.model_input_constants == {
            "collision_energies": 30,
        }
        assert feat.model_input_columns == {"fragmentation_types": "frag_col"}
        calibrator.apply_koina_model_input_overrides(
            model_input_columns={"collision_energies": "nce_col"},
        )
        assert feat.model_input_constants is None
        assert feat.model_input_columns == {
            "fragmentation_types": "frag_col",
            "collision_energies": "nce_col",
        }

    def test_columns_property_empty(self, calibrator):
        """Test columns property when no features are added."""
        assert calibrator.columns == []

    def test_feature_names_empty(self, calibrator):
        """Test feature_names property when no features are added."""
        assert calibrator.feature_names == []

    def test_add_feature_basic(self, calibrator):
        """Test adding a basic feature without dependencies."""
        feature = MockCalibrationFeature("test_feature", ["col1", "col2"])
        calibrator.add_feature(feature)

        assert "test_feature" in calibrator.feature_dict
        assert calibrator.columns == ["col1", "col2"]
        assert calibrator.feature_names == ["test_feature"]

    def test_add_feature_with_dependencies(self, calibrator):
        """Test adding a feature with dependencies."""
        dependency = MockFeatureDependency("test_dep")
        feature = MockCalibrationFeature("test_feature", ["col1"], [dependency])
        calibrator.add_feature(feature)

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
        assert len(calibrator.dependencies) == 1

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
        calibrator.remove_feature("removable_feature")

        assert "removable_feature" not in calibrator.feature_dict
        assert calibrator.columns == []

    def test_remove_feature_with_dependencies(self, calibrator):
        """Test removing a feature with dependencies."""
        dependency = MockFeatureDependency("removable_dep")
        feature = MockCalibrationFeature(
            "removable_feature",
            dependencies=[dependency],
        )
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
        calibrator.remove_feature("feature1")

        assert "feature1" not in calibrator.feature_dict
        assert "feature2" in calibrator.feature_dict
        assert "shared_dep" in calibrator.dependencies
        assert calibrator.dependency_reference_counter["shared_dep"] == 1

    def test_remove_nonexistent_feature_raises_error(self, calibrator):
        """Test that removing a nonexistent feature raises KeyError."""
        with pytest.raises(KeyError):
            calibrator.remove_feature("nonexistent")

    def test_compute_features_mutates_metadata(self, calibrator, sample_dataset):
        """Test that compute_features adds feature columns to metadata."""
        feature = MockCalibrationFeature("test_feature", ["test_col"])
        calibrator.add_feature(feature)

        calibrator.compute_features(sample_dataset)

        assert "test_col" in sample_dataset.metadata.columns

    def test_extract_feature_matrix_unlabelled(self, calibrator, sample_dataset):
        """Test extracting unlabelled feature matrix after compute_features."""
        feature = MockCalibrationFeature("test_feature", ["test_col"])
        calibrator.add_feature(feature)

        calibrator.compute_features(sample_dataset)
        features = calibrator._extract_feature_matrix(sample_dataset, labelled=False)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(sample_dataset.metadata)
        assert features.shape[1] == 2  # confidence + one feature column

    def test_extract_feature_matrix_labelled(self, calibrator, labelled_dataset):
        """Test extracting labelled feature matrix after compute_features."""
        feature = MockCalibrationFeature("test_feature", ["test_col"])
        calibrator.add_feature(feature)

        calibrator.compute_features(labelled_dataset)
        features, labels = calibrator._extract_feature_matrix(
            labelled_dataset, labelled=True
        )

        assert features.shape[0] == len(labelled_dataset.metadata)
        assert labels.shape[0] == len(labelled_dataset.metadata)

    def test_compute_features_with_dependencies(self, calibrator, sample_dataset):
        """Test computing features with dependencies."""
        dependency = MockFeatureDependency("test_dep")
        feature = MockCalibrationFeature("test_feature", dependencies=[dependency])
        calibrator.add_feature(feature)

        calibrator.compute_features(sample_dataset)

        assert f"{dependency.name}_data" in sample_dataset.metadata.columns

    def test_fit_from_features_returns_history(self, feature_dataset):
        """Test that fit_from_features returns a TrainingHistory."""
        calibrator = ProbabilityCalibrator(
            max_epochs=3,
            hidden_dims=(8,),
            seed=42,
        )
        history = calibrator.fit_from_features(feature_dataset)

        assert isinstance(history, TrainingHistory)
        assert len(history.train_losses) == 3
        assert history.epochs_trained == 3
        assert calibrator.network is not None

    def test_fit_from_features_with_validation(self, feature_dataset):
        """Test fit_from_features with an explicit validation dataset."""
        np.random.seed(123)
        val_features = np.random.randn(20, 3).astype(np.float32)
        val_labels = np.random.choice([0.0, 1.0], 20).astype(np.float32)
        val_dataset = FeatureDataset(features=val_features, labels=val_labels)

        calibrator = ProbabilityCalibrator(
            max_epochs=5,
            hidden_dims=(8,),
            n_iter_no_change=3,
            seed=42,
        )
        history = calibrator.fit_from_features(feature_dataset, val_dataset)

        assert history.val_losses is not None
        assert history.val_accuracies is not None
        assert len(history.val_losses) <= 5
        assert history.final_val_loss is None
        assert history.final_val_accuracy is None

    def test_fit_from_features_val_subsample_records_full_metrics(
        self, feature_dataset
    ):
        """Large validation sets are subsampled for early stopping; full metrics logged."""
        np.random.seed(123)
        n_val = 50
        val_features = np.random.randn(n_val, 3).astype(np.float32)
        val_labels = np.random.choice([0.0, 1.0], n_val).astype(np.float32)
        val_dataset = FeatureDataset(features=val_features, labels=val_labels)

        calibrator = ProbabilityCalibrator(
            max_epochs=2,
            hidden_dims=(8,),
            n_iter_no_change=10,
            seed=42,
            val_early_stopping_max_psms=10,
            val_subsample_seed=123,
        )
        history = calibrator.fit_from_features(feature_dataset, val_dataset)

        assert history.final_val_loss is not None
        assert history.final_val_accuracy is not None
        assert len(history.val_losses) == 2

    def test_fit_from_features_sets_normalization(self, feature_dataset):
        """Test that fit_from_features computes feature normalization stats."""
        calibrator = ProbabilityCalibrator(max_epochs=1, hidden_dims=(4,))
        calibrator.fit_from_features(feature_dataset)

        assert calibrator.feature_mean is not None
        assert calibrator.feature_std is not None
        assert calibrator.feature_mean.shape == (3,)

    def test_end_to_end_fit_predict(self):
        """Test the full pipeline: fit -> compute_features (inference) -> predict."""
        n_train = 80
        np.random.seed(42)
        train_metadata = pd.DataFrame(
            {
                "confidence": np.random.uniform(0.1, 0.99, n_train),
                "correct": np.random.choice([0, 1], n_train),
            }
        )
        train_raw = CalibrationDataset(
            metadata=train_metadata,
            predictions=[None] * n_train,
        )

        calibrator = ProbabilityCalibrator(
            max_epochs=3,
            hidden_dims=(8,),
            seed=42,
        )
        feature = MockCalibrationFeature("mock_feat", ["mock_col"])
        calibrator.add_feature(feature)

        calibrator.fit(train_raw)

        n_pred = 10
        pred_metadata = pd.DataFrame(
            {
                "confidence": np.random.uniform(0.1, 0.99, n_pred),
            }
        )
        pred_raw = CalibrationDataset(
            metadata=pred_metadata,
            predictions=[None] * n_pred,
        )
        calibrator.compute_features(pred_raw)
        calibrator.predict(pred_raw)

        assert "calibrated_confidence" in pred_raw.metadata.columns
        probs = pred_raw.metadata["calibrated_confidence"]
        assert len(probs) == n_pred
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_predict_without_fit_raises(self, calibrator, sample_dataset):
        """Test that prediction fails if calibrator hasn't been fitted."""
        feature = MockCalibrationFeature("test_feature", ["test_col"])
        calibrator.add_feature(feature)

        with pytest.raises(RuntimeError, match="not been fitted or loaded"):
            calibrator.predict(sample_dataset)

    def test_save_load_roundtrip(self, tmp_path, feature_dataset):
        """Test that save/load produces a working calibrator with correct config."""
        calibrator = ProbabilityCalibrator(
            max_epochs=2,
            hidden_dims=(8, 4),
            seed=42,
        )
        feature = MockCalibrationFeature("test_feature", ["test_col"])
        calibrator.add_feature(feature)
        calibrator.fit_from_features(feature_dataset)

        ProbabilityCalibrator.save(calibrator, tmp_path / "model")

        loaded = ProbabilityCalibrator.load(tmp_path / "model")

        assert loaded.hidden_dims == calibrator.hidden_dims
        assert loaded.network is not None
        assert loaded.feature_mean is not None

        with open(tmp_path / "model" / "config.json") as f:
            config = json.load(f)
        assert "features" in config
        assert "test_feature" in config["features"]
        assert "_target_" in config["features"]["test_feature"]

    def test_save_load_weights_and_normalization_match(self, tmp_path, feature_dataset):
        """Test that weights and normalization stats survive save/load exactly."""
        calibrator = ProbabilityCalibrator(
            max_epochs=2,
            hidden_dims=(8,),
            seed=42,
        )
        calibrator.fit_from_features(feature_dataset)

        ProbabilityCalibrator.save(calibrator, tmp_path / "model")
        loaded = ProbabilityCalibrator.load(tmp_path / "model")

        calibrator.network.cpu().eval()
        loaded.network.cpu().eval()

        torch.testing.assert_close(
            calibrator.feature_mean.cpu(),
            loaded.feature_mean.cpu(),
        )
        torch.testing.assert_close(
            calibrator.feature_std.cpu(),
            loaded.feature_std.cpu(),
        )

        x = torch.randn(5, 3)
        x_norm = (x - calibrator.feature_mean.cpu()) / calibrator.feature_std.cpu()
        with torch.no_grad():
            original_out = calibrator.network(x_norm)
            loaded_out = loaded.network(x_norm)

        torch.testing.assert_close(original_out, loaded_out)

    def test_save_load_then_predict(self, tmp_path):
        """Test that a loaded calibrator can compute features and predict on new data."""
        n = 80
        np.random.seed(42)
        train_metadata = pd.DataFrame(
            {
                "confidence": np.random.uniform(0.1, 0.99, n),
                "correct": np.random.choice([0, 1], n),
            }
        )
        train_raw = CalibrationDataset(
            metadata=train_metadata,
            predictions=[None] * n,
        )

        calibrator = ProbabilityCalibrator(
            max_epochs=3,
            hidden_dims=(8,),
            seed=42,
        )
        feature = MockCalibrationFeature("feat", ["feat_col"])
        calibrator.add_feature(feature)

        calibrator.fit(train_raw)

        ProbabilityCalibrator.save(calibrator, tmp_path / "model")
        loaded = ProbabilityCalibrator.load(tmp_path / "model")

        pred_metadata = pd.DataFrame({"confidence": [0.9, 0.5, 0.1]})
        pred_ds = CalibrationDataset(
            metadata=pred_metadata,
            predictions=[None] * 3,
        )
        loaded.compute_features(pred_ds)
        loaded.predict(pred_ds)

        assert "calibrated_confidence" in pred_ds.metadata.columns
        probs = pred_ds.metadata["calibrated_confidence"]
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_early_stopping_triggers(self):
        """Test that training stops early when n_iter_no_change is exhausted."""
        np.random.seed(42)

        # Linearly separable training data: label = 1 when x > 0.
        x_train = np.linspace(-2, 2, 200).reshape(-1, 1).astype(np.float32)
        y_train = (x_train[:, 0] > 0).astype(np.float32)
        train_ds = FeatureDataset(features=x_train, labels=y_train)

        # Validation with *inverted* labels: model overfits train, val loss rises.
        x_val = np.linspace(-2, 2, 50).reshape(-1, 1).astype(np.float32)
        y_val = (x_val[:, 0] <= 0).astype(np.float32)
        val_ds = FeatureDataset(features=x_val, labels=y_val)

        calibrator = ProbabilityCalibrator(
            max_epochs=50,
            hidden_dims=(16,),
            learning_rate=0.01,
            n_iter_no_change=3,
            tol=1e-4,
            seed=42,
        )
        history = calibrator.fit_from_features(train_ds, val_ds)

        assert history.epochs_trained < 50

    def test_add_features_plural(self, calibrator):
        """Test adding multiple features at once via add_features."""
        feat1 = MockCalibrationFeature("feat_a", ["col_a"])
        feat2 = MockCalibrationFeature("feat_b", ["col_b"])
        calibrator.add_features([feat1, feat2])

        assert "feat_a" in calibrator.feature_dict
        assert "feat_b" in calibrator.feature_dict
        assert calibrator.columns == ["col_a", "col_b"]

    def test_save_unfitted_raises(self, tmp_path, calibrator):
        """Test that saving an unfitted calibrator raises RuntimeError."""
        with pytest.raises(RuntimeError, match="unfitted"):
            ProbabilityCalibrator.save(calibrator, tmp_path / "model")

    def test_load_missing_path_raises(self):
        """Test that loading from a nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ProbabilityCalibrator.load("/nonexistent/path/to/model")

    def test_empty_dataset_handling(self, calibrator):
        """Test handling of empty datasets."""
        empty_metadata = pd.DataFrame({"confidence": []})
        empty_dataset = CalibrationDataset(metadata=empty_metadata, predictions=[])

        feature = MockCalibrationFeature("test_feature")
        calibrator.add_feature(feature)

        calibrator.compute_features(empty_dataset)
        features = calibrator._extract_feature_matrix(empty_dataset, labelled=False)
        assert features.shape[0] == 0

    def test_get_config_on_mock_feature(self):
        """Test that get_config returns expected keys."""
        feature = MockCalibrationFeature("test", ["col1", "col2"])
        config = feature.get_config()

        assert "_target_" in config
        assert "MockCalibrationFeature" in config["_target_"]
        assert config["name"] == "test"

    def test_get_config_converts_omegaconf_to_plain(self):
        """Test that get_config resolves DictConfig/ListConfig to plain types."""
        from omegaconf import ListConfig

        feature = MockCalibrationFeature(
            "test",
            columns=ListConfig(["col1", "col2"]),
            dependencies=ListConfig([]),
        )
        config = feature.get_config()

        assert isinstance(config["columns"], list)
        assert not type(config["columns"]).__module__.startswith("omegaconf")
        json.dumps(config)
