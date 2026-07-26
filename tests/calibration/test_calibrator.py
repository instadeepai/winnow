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
from winnow.calibration.features.fragment_match import FragmentMatchFeatures
from winnow.utils.koina_intensity_config import KOINA_RUNTIME_CONFIG_KEYS


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


class DeterministicMockFeature(MockCalibrationFeature):
    """Mock feature with fixed column values for I/O freeze tests."""

    def compute(self, dataset):
        n = len(dataset.metadata)
        for i, col in enumerate(self.columns):
            dataset.metadata[col] = np.full(n, 0.1 * (i + 1), dtype=np.float64)


def _hand_wired_freeze_calibrator(*, batch_size: int = 1) -> ProbabilityCalibrator:
    """Build a fitted-looking calibrator on CPU without calling fit().

    Uses non-identity ``feature_mean`` / ``feature_std`` so predict freezes
    exercise normalisation. Default ``batch_size=1`` forces multi-chunk
    predict once batching lands (n=3 rows in freeze metadata).
    """
    calibrator = ProbabilityCalibrator(
        hidden_dims=(4,),
        dropout=0.0,
        batch_size=batch_size,
        seed=0,
    )
    calibrator.add_feature(DeterministicMockFeature("mock", ["f1", "f2"]))
    network = CalibratorNetwork(input_dim=3, hidden_dims=(4,), dropout=0.0).to("cpu")
    with torch.no_grad():
        for i, param in enumerate(network.parameters()):
            param.fill_(0.1 * (i + 1))
    calibrator.network = network
    calibrator.feature_mean = torch.tensor([0.5, 1.5, 0.2], dtype=torch.float32)
    calibrator.feature_std = torch.tensor([0.4, 1.0, 0.1], dtype=torch.float32)
    return calibrator


def _freeze_feature_metadata(*, labelled: bool = False) -> pd.DataFrame:
    """Literal metadata used by extract / predict I/O freeze tests."""
    data = {
        "confidence": [0.9, 0.5, 0.1],
        "f1": [1.0, 2.0, 3.0],
        "f2": [0.1, 0.2, 0.3],
    }
    if labelled:
        data["correct"] = [1.0, 0.0, 1.0]
    return pd.DataFrame(data)


# Golden outputs from the current full-tensor CPU predict path with the
# hand-wired calibrator (non-identity mean/std) and freeze metadata above.
_FREEZE_PREDICT_PROBS = [
    0.641067385673523,
    0.6681877970695496,
    0.6942363977432251,
]
# Raw network forward on unnormalised float32 features (not the predict path).
_FREEZE_NETWORK_LOGITS = [
    0.8799999952316284,
    0.9640001058578491,
    1.0480000972747803,
]


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

    @staticmethod
    def _feature_dataset_with_width(
        n_features: int,
        n: int = 100,
        columns: list[str] | None = None,
    ) -> FeatureDataset:
        """Build a labelled FeatureDataset with a given feature width.

        Matrix layout is ``[confidence, *columns]``. When ``columns`` is
        omitted, synthetic names ``c0`` .. are used for non-confidence slots.
        """
        np.random.seed(42)
        features = np.random.randn(n, n_features).astype(np.float32)
        labels = np.random.choice([0.0, 1.0], n).astype(np.float32)
        if columns is None:
            columns = [f"c{i}" for i in range(n_features - 1)]
        return FeatureDataset(features=features, labels=labels, columns=columns)

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
        """Confidence-only FeatureDataset (width 1) for schema-matched training."""
        n = 100
        np.random.seed(42)
        features = np.random.randn(n, 1).astype(np.float32)
        labels = np.random.choice([0.0, 1.0], n).astype(np.float32)
        return FeatureDataset(features=features, labels=labels, columns=[])

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
        assert calibrator.val_early_stopping_max_psms is None
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

    def test_constants_win_over_default_columns_in_composed_cfg(self):
        """Constants from CLI must not be undone by default column entries in cfg."""
        calibrator = ProbabilityCalibrator()
        calibrator.add_feature(
            FragmentMatchFeatures(
                mz_tolerance=0.02,
                mz_tolerance_unit="da",
                model_input_columns={
                    "collision_energies": "collision_energy",
                    "fragmentation_types": "frag_type",
                },
            )
        )
        calibrator.apply_koina_model_input_overrides(
            model_input_constants={
                "collision_energies": 27,
                "fragmentation_types": "HCD",
            },
            model_input_columns={
                "collision_energies": "collision_energy",
                "fragmentation_types": "frag_type",
            },
        )
        feat = calibrator.feature_dict["Fragment Match Features"]
        assert feat.model_input_constants == {
            "collision_energies": 27,
            "fragmentation_types": "HCD",
        }
        assert feat.model_input_columns is None

    def test_columns_property_empty(self, calibrator):
        """Test columns property when no features are added."""
        assert calibrator.columns == []

    def test_load_rejects_fitted_columns_missing_from_registry(
        self, tmp_path, labelled_dataset
    ):
        """Loading fails when fitted columns are no longer in the registry."""
        calibrator = ProbabilityCalibrator(seed=42)
        feature = MockCalibrationFeature("test_feature", ["test_col"])
        calibrator.add_feature(feature)
        calibrator.fit(labelled_dataset)

        calibrator.feature_dict["test_feature"]._columns = ["other_col"]
        ProbabilityCalibrator.save(calibrator, tmp_path)

        with pytest.raises(
            ValueError,
            match=r"fitted feature column\(s\) \['test_col'\].*not produced",
        ):
            ProbabilityCalibrator.load(tmp_path)

    def test_predict_rejects_fitted_columns_missing_from_registry(
        self, calibrator, labelled_dataset, sample_dataset
    ):
        """Prediction raises when fitted columns leave the live registry."""
        feature = MockCalibrationFeature("test_feature", ["test_col"])
        calibrator.add_feature(feature)
        calibrator.fit(labelled_dataset)

        calibrator.feature_dict["test_feature"]._columns = ["other_col"]

        with pytest.raises(
            ValueError,
            match=r"fitted feature column\(s\) \['test_col'\].*not produced",
        ):
            calibrator.predict(sample_dataset)

    def test_training_feature_columns_subsets_columns_property(self, calibrator):
        """training_feature_columns narrows columns without changing Feature.columns."""
        feature = MockCalibrationFeature("beam", ["margin", "entropy", "z-score"])
        calibrator.add_feature(feature)
        assert feature.columns == ["margin", "entropy", "z-score"]

        calibrator.set_training_feature_columns(["margin", "z-score"])
        assert calibrator.columns == ["margin", "z-score"]
        assert feature.columns == ["margin", "entropy", "z-score"]
        assert calibrator.training_feature_columns == ["margin", "z-score"]

    def test_training_feature_columns_unknown_name_raises(self, calibrator):
        """Unknown training column names are rejected against the registry."""
        calibrator.add_feature(MockCalibrationFeature("beam", ["margin"]))
        with pytest.raises(ValueError, match="unknown column"):
            calibrator.set_training_feature_columns(["margin", "nope"])

    def test_training_feature_columns_allows_sequential_add_feature(self, calibrator):
        """Deferred training columns are not checked on each add_feature."""
        calibrator.set_training_feature_columns(["margin", "irt_error"])
        calibrator.add_feature(MockCalibrationFeature("beam", ["margin", "entropy"]))
        calibrator.add_feature(MockCalibrationFeature("rt", ["irt_error"]))
        assert calibrator.columns == ["margin", "irt_error"]
        # Still valid once the registry is complete.
        calibrator._validate_training_feature_columns_against_registry()

    def test_training_feature_columns_deferred_unknown_raises_at_fit(self):
        """Unknown deferred training columns are caught at fit_from_features."""
        n = 40
        train = FeatureDataset(
            features=np.random.randn(n, 2).astype(np.float32),
            labels=np.random.choice([0.0, 1.0], n).astype(np.float32),
            columns=["margin"],
        )
        calibrator = ProbabilityCalibrator(max_epochs=1, hidden_dims=(4,), seed=0)
        calibrator.set_training_feature_columns(["margin", "nope"])
        calibrator.add_feature(MockCalibrationFeature("beam", ["margin"]))
        with pytest.raises(ValueError, match="unknown column"):
            calibrator.fit_from_features(train)

    def test_training_feature_columns_rejects_after_fit(self, feature_dataset):
        """Cannot change training_feature_columns after fit."""
        calibrator = ProbabilityCalibrator(max_epochs=1, hidden_dims=(4,), seed=0)
        calibrator.fit_from_features(feature_dataset)
        with pytest.raises(ValueError, match="after the calibrator has been fitted"):
            calibrator.set_training_feature_columns([])

    def test_fit_from_features_with_training_feature_subset(self, tmp_path):
        """Subset training freezes fitted columns and survives save/load."""
        n = 80
        features = np.random.randn(n, 4).astype(np.float32)
        labels = np.random.choice([0.0, 1.0], n).astype(np.float32)
        wide = FeatureDataset(
            features=features,
            labels=labels,
            columns=["margin", "entropy", "z-score"],
        )
        calibrator = ProbabilityCalibrator(max_epochs=2, hidden_dims=(8,), seed=0)
        calibrator.add_feature(
            MockCalibrationFeature("beam", ["margin", "entropy", "z-score"])
        )
        calibrator.set_training_feature_columns(["z-score", "margin"])
        train = wide.select_for(calibrator)
        assert train.columns == ["z-score", "margin"]

        calibrator.fit_from_features(train)
        assert calibrator.columns == ["z-score", "margin"]
        assert calibrator._fitted_feature_columns == ["z-score", "margin"]

        ProbabilityCalibrator.save(calibrator, tmp_path / "subset_model")
        with open(tmp_path / "subset_model" / "config.json") as f:
            config = json.load(f)
        assert config["feature_columns"] == ["z-score", "margin"]
        assert config["training_feature_columns"] == ["z-score", "margin"]
        assert "beam" in config["features"]

        loaded = ProbabilityCalibrator.load(tmp_path / "subset_model")
        assert loaded.columns == ["z-score", "margin"]
        assert loaded._registry_feature_columns() == ["margin", "entropy", "z-score"]

    def test_load_allows_fitted_subset_of_wider_registry(self, tmp_path):
        """Fitted columns may be a proper subset of registered feature columns."""
        n = 60
        features = np.random.randn(n, 2).astype(np.float32)
        labels = np.random.choice([0.0, 1.0], n).astype(np.float32)
        train = FeatureDataset(features=features, labels=labels, columns=["margin"])
        calibrator = ProbabilityCalibrator(max_epochs=1, hidden_dims=(4,), seed=0)
        calibrator.add_feature(MockCalibrationFeature("beam", ["margin", "entropy"]))
        calibrator.set_training_feature_columns(["margin"])
        calibrator.fit_from_features(train)
        ProbabilityCalibrator.save(calibrator, tmp_path / "subset")

        loaded = ProbabilityCalibrator.load(tmp_path / "subset")
        assert loaded.columns == ["margin"]
        assert "entropy" in loaded._registry_feature_columns()

    def test_load_preserves_empty_fitted_training_schema(self, tmp_path):
        """An empty fitted subset remains frozen after checkpoint loading."""
        n = 40
        confidence_only = FeatureDataset(
            features=np.random.rand(n, 1).astype(np.float32),
            labels=np.random.choice([0.0, 1.0], n).astype(np.float32),
            columns=[],
        )
        calibrator = ProbabilityCalibrator(max_epochs=1, hidden_dims=(4,), seed=0)
        calibrator.add_feature(MockCalibrationFeature("beam", ["margin", "entropy"]))
        calibrator.set_training_feature_columns([])
        calibrator.fit_from_features(confidence_only)
        ProbabilityCalibrator.save(calibrator, tmp_path / "confidence_only")

        loaded = ProbabilityCalibrator.load(tmp_path / "confidence_only")

        assert loaded._fitted_feature_columns == []
        with pytest.raises(ValueError, match="after the calibrator has been fitted"):
            loaded.set_training_feature_columns(["margin"])

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

    def test_extract_feature_matrix_unlabelled_values(self):
        """Freeze unlabelled extract values, column order, and dtype."""
        calibrator = _hand_wired_freeze_calibrator()
        dataset = CalibrationDataset(
            metadata=_freeze_feature_metadata(labelled=False),
            predictions=[None] * 3,
        )
        features = calibrator._extract_feature_matrix(dataset, labelled=False)

        expected = np.array(
            [[0.9, 1.0, 0.1], [0.5, 2.0, 0.2], [0.1, 3.0, 0.3]],
            dtype=np.float64,
        )
        assert features.dtype == np.float64
        np.testing.assert_allclose(features, expected, rtol=0, atol=1e-6)

    def test_extract_feature_matrix_labelled_values(self):
        """Freeze labelled extract feature/label values and dtypes."""
        calibrator = _hand_wired_freeze_calibrator()
        dataset = CalibrationDataset(
            metadata=_freeze_feature_metadata(labelled=True),
            predictions=[None] * 3,
        )
        features, labels = calibrator._extract_feature_matrix(dataset, labelled=True)

        expected_features = np.array(
            [[0.9, 1.0, 0.1], [0.5, 2.0, 0.2], [0.1, 3.0, 0.3]],
            dtype=np.float64,
        )
        expected_labels = np.array([1.0, 0.0, 1.0], dtype=np.float64)
        assert features.dtype == np.float64
        assert labels.dtype == np.float64
        np.testing.assert_allclose(features, expected_features, rtol=0, atol=1e-6)
        np.testing.assert_allclose(labels, expected_labels, rtol=0, atol=1e-6)

    def test_network_forward_golden_logits(self):
        """Freeze CalibratorNetwork logits for fixed weights and float32 input."""
        calibrator = _hand_wired_freeze_calibrator()
        assert calibrator.network is not None
        x = torch.tensor(
            [[0.9, 1.0, 0.1], [0.5, 2.0, 0.2], [0.1, 3.0, 0.3]],
            dtype=torch.float32,
        )
        calibrator.network.eval()
        with torch.no_grad():
            logits = calibrator.network(x)

        assert logits.dtype == torch.float32
        np.testing.assert_allclose(
            logits.cpu().numpy(),
            np.asarray(_FREEZE_NETWORK_LOGITS, dtype=np.float32),
            rtol=0,
            atol=1e-6,
        )

    def test_predict_golden_calibrated_confidence(self):
        """Freeze predict calibrated_confidence values and Python float list type."""
        calibrator = _hand_wired_freeze_calibrator(batch_size=1)
        dataset = CalibrationDataset(
            metadata=_freeze_feature_metadata(labelled=False),
            predictions=[None] * 3,
        )
        calibrator.predict(dataset)

        probs = dataset.metadata["calibrated_confidence"].tolist()
        assert len(probs) == 3
        assert all(isinstance(p, float) for p in probs)
        np.testing.assert_allclose(probs, _FREEZE_PREDICT_PROBS, rtol=0, atol=1e-6)

    def test_predict_batch_size_parity(self):
        """batch_size=1 and a large batch_size must match the golden probs."""
        for batch_size in (1, 1024):
            calibrator = _hand_wired_freeze_calibrator(batch_size=batch_size)
            dataset = CalibrationDataset(
                metadata=_freeze_feature_metadata(labelled=False),
                predictions=[None] * 3,
            )
            calibrator.predict(dataset)
            probs = dataset.metadata["calibrated_confidence"].tolist()
            np.testing.assert_allclose(probs, _FREEZE_PREDICT_PROBS, rtol=0, atol=1e-6)

    def test_predict_feature_column_order(self):
        """Metadata column permutation must not change predict output order."""
        calibrator = _hand_wired_freeze_calibrator(batch_size=1)
        permuted = _freeze_feature_metadata(labelled=False)[["f2", "confidence", "f1"]]
        dataset = CalibrationDataset(metadata=permuted, predictions=[None] * 3)
        calibrator.predict(dataset)

        probs = dataset.metadata["calibrated_confidence"].tolist()
        np.testing.assert_allclose(probs, _FREEZE_PREDICT_PROBS, rtol=0, atol=1e-6)

    def test_to_feature_dataset_matches_extract_column_order(self):
        """to_feature_dataset uses confidence then registered feature columns."""
        calibrator = _hand_wired_freeze_calibrator()
        dataset = CalibrationDataset(
            metadata=_freeze_feature_metadata(labelled=True),
            predictions=[None] * 3,
        )
        feature_dataset = calibrator.to_feature_dataset(dataset)
        features, labels = calibrator._extract_feature_matrix(dataset, labelled=True)

        assert feature_dataset.columns == list(calibrator.columns)
        assert feature_dataset.features.dtype == torch.float32
        assert feature_dataset.labels.dtype == torch.float32
        np.testing.assert_allclose(
            feature_dataset.features.numpy(),
            np.asarray(features, dtype=np.float32),
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            feature_dataset.labels.numpy(),
            np.asarray(labels, dtype=np.float32),
            rtol=0,
            atol=1e-6,
        )

    def test_compute_features_with_dependencies(self, calibrator, sample_dataset):
        """Test computing features with dependencies."""
        dependency = MockFeatureDependency("test_dep")
        feature = MockCalibrationFeature("test_feature", dependencies=[dependency])
        calibrator.add_feature(feature)

        calibrator.compute_features(sample_dataset)

        assert f"{dependency.name}_data" in sample_dataset.metadata.columns

    def test_max_epochs_must_be_positive(self):
        """max_epochs=0 is rejected at construction time."""
        with pytest.raises(ValueError, match="max_epochs must be at least 1"):
            ProbabilityCalibrator(max_epochs=0)

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
        val_features = np.random.randn(20, 1).astype(np.float32)
        val_labels = np.random.choice([0.0, 1.0], 20).astype(np.float32)
        val_dataset = FeatureDataset(
            features=val_features, labels=val_labels, columns=[]
        )

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
        val_features = np.random.randn(n_val, 1).astype(np.float32)
        val_labels = np.random.choice([0.0, 1.0], n_val).astype(np.float32)
        val_dataset = FeatureDataset(
            features=val_features, labels=val_labels, columns=[]
        )

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
        assert calibrator.feature_mean.shape == (1,)
        assert calibrator._fitted_feature_columns == []

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

    def test_save_strips_koina_runtime_keys(self, tmp_path):
        """Saved config.json must not persist runtime-only Koina settings."""
        calibrator = ProbabilityCalibrator(max_epochs=2, hidden_dims=(8,), seed=42)
        feature = FragmentMatchFeatures(
            mz_tolerance=0.02,
            mz_tolerance_unit="da",
            model_input_constants={"collision_energies": 27},
            model_input_columns={"fragmentation_types": "frag_type"},
        )
        calibrator.add_feature(feature)
        train_ds = self._feature_dataset_with_width(
            1 + len(feature.columns), columns=list(feature.columns)
        )
        calibrator.fit_from_features(train_ds)
        ProbabilityCalibrator.save(calibrator, tmp_path / "koina_model")

        with open(tmp_path / "koina_model" / "config.json") as f:
            config = json.load(f)
        feature_config = config["features"]["Fragment Match Features"]
        for key in KOINA_RUNTIME_CONFIG_KEYS:
            assert key not in feature_config
        assert feature_config["intensity_model_name"] == "Prosit_2020_intensity_HCD"

    def test_load_strips_legacy_koina_runtime_keys(self, tmp_path):
        """Loading old checkpoints ignores baked-in Koina input presets."""
        calibrator = ProbabilityCalibrator(max_epochs=2, hidden_dims=(8,), seed=42)
        feature = FragmentMatchFeatures(
            mz_tolerance=0.02,
            mz_tolerance_unit="da",
            model_input_columns={
                "collision_energies": "collision_energy",
                "fragmentation_types": "frag_type",
            },
        )
        calibrator.add_feature(feature)
        train_ds = self._feature_dataset_with_width(
            1 + len(feature.columns), columns=list(feature.columns)
        )
        calibrator.fit_from_features(train_ds)
        ProbabilityCalibrator.save(calibrator, tmp_path / "legacy_model")

        with open(tmp_path / "legacy_model" / "config.json") as f:
            config = json.load(f)
        feature_config = config["features"]["Fragment Match Features"]
        feature_config["model_input_columns"] = {
            "collision_energies": "collision_energy",
            "fragmentation_types": "frag_type",
        }
        with open(tmp_path / "legacy_model" / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        loaded = ProbabilityCalibrator.load(tmp_path / "legacy_model")
        feat = loaded.feature_dict["Fragment Match Features"]
        assert feat.model_input_constants is None
        assert feat.model_input_columns is None

    def test_save_load_roundtrip(self, tmp_path):
        """Test that save/load produces a working calibrator with correct config."""
        calibrator = ProbabilityCalibrator(
            max_epochs=2,
            hidden_dims=(8, 4),
            seed=42,
        )
        feature = MockCalibrationFeature("test_feature", ["test_col"])
        calibrator.add_feature(feature)
        train_ds = self._feature_dataset_with_width(2, columns=["test_col"])
        calibrator.fit_from_features(train_ds)

        ProbabilityCalibrator.save(calibrator, tmp_path / "model")

        loaded = ProbabilityCalibrator.load(tmp_path / "model")

        assert loaded.hidden_dims == calibrator.hidden_dims
        assert loaded.network is not None
        assert loaded.feature_mean is not None
        assert loaded.columns == ["test_col"]
        assert loaded._fitted_feature_columns == ["test_col"]

        with open(tmp_path / "model" / "config.json") as f:
            config = json.load(f)
        assert "features" in config
        assert "test_feature" in config["features"]
        assert "_target_" in config["features"]["test_feature"]
        assert config["feature_columns"] == ["test_col"]
        assert config["input_dim"] == 2

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

        x = torch.randn(5, 1)
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
        train_ds = FeatureDataset(features=x_train, labels=y_train, columns=[])

        # Validation with *inverted* labels: model overfits train, val loss rises.
        x_val = np.linspace(-2, 2, 50).reshape(-1, 1).astype(np.float32)
        y_val = (x_val[:, 0] <= 0).astype(np.float32)
        val_ds = FeatureDataset(features=x_val, labels=y_val, columns=[])

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

    def test_load_legacy_pickle_checkpoint_raises(self, tmp_path):
        """Pre-PyTorch calibrator.pkl directories raise a clear error."""
        legacy_dir = tmp_path / "legacy_checkpoint"
        legacy_dir.mkdir()
        (legacy_dir / "calibrator.pkl").write_bytes(b"fake pickle")

        with pytest.raises(
            ValueError,
            match="Legacy pickle checkpoint format is no longer supported",
        ):
            ProbabilityCalibrator.load(legacy_dir)

    def test_empty_dataset_handling(self, calibrator):
        """Test handling of empty datasets."""
        empty_metadata = pd.DataFrame({"confidence": []})
        empty_dataset = CalibrationDataset(metadata=empty_metadata, predictions=[])

        feature = MockCalibrationFeature("test_feature")
        calibrator.add_feature(feature)

        calibrator.compute_features(empty_dataset)
        features = calibrator._extract_feature_matrix(empty_dataset, labelled=False)
        assert features.shape[0] == 0

    def test_predict_empty_dataset_raises(self):
        """Predict on zero rows should fail with a clear error, not a torch dtype error."""
        n_train = 20
        train_metadata = pd.DataFrame(
            {
                "confidence": np.linspace(0.9, 0.5, n_train),
                "correct": np.ones(n_train, dtype=int),
            }
        )
        train_raw = CalibrationDataset(
            metadata=train_metadata, predictions=[None] * n_train
        )
        calibrator = ProbabilityCalibrator(max_epochs=1, hidden_dims=(4,), seed=42)
        calibrator.add_feature(MockCalibrationFeature("mock_feat", ["mock_col"]))
        calibrator.fit(train_raw)

        empty_dataset = CalibrationDataset(
            metadata=train_metadata.iloc[:0].copy(), predictions=[]
        )
        calibrator.compute_features(empty_dataset)

        with pytest.raises(ValueError, match="empty dataset"):
            calibrator.predict(empty_dataset)

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

    def test_fit_from_features_rejects_input_dim_mismatch(self):
        """Extra FeatureDataset columns must match calibrator.columns exactly."""
        n = 40
        features = np.random.randn(n, 5).astype(np.float32)
        labels = np.random.choice([0.0, 1.0], n).astype(np.float32)
        train_ds = FeatureDataset(
            features=features,
            labels=labels,
            columns=["real_col", "extra_a", "extra_b", "extra_c"],
        )

        calibrator = ProbabilityCalibrator(
            max_epochs=1, hidden_dims=(4,), seed=0, batch_size=16
        )
        calibrator.add_feature(MockCalibrationFeature("real_feat", ["real_col"]))

        with pytest.raises(
            ValueError,
            match=r"Extra:.*select_for\(calibrator\)",
        ):
            calibrator.fit_from_features(train_ds)

        assert calibrator.columns == ["real_col"]
        assert calibrator._fitted_feature_columns is None
        assert calibrator.network is None

    def test_fit_from_features_rejects_validation_column_mismatch(self):
        """Validation FeatureDataset.columns must match training columns."""
        train_ds = self._feature_dataset_with_width(2, columns=["real_col"])
        val_ds = self._feature_dataset_with_width(
            3, n=20, columns=["real_col", "extra"]
        )
        calibrator = ProbabilityCalibrator(max_epochs=1, hidden_dims=(4,), seed=0)
        calibrator.add_feature(MockCalibrationFeature("real_feat", ["real_col"]))

        with pytest.raises(
            ValueError,
            match=r"Training and validation FeatureDataset.columns must be identical",
        ):
            calibrator.fit_from_features(train_ds, val_ds)
