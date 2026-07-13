"""Tests for helpers in winnow.scripts.main."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from winnow.calibration.calibration_features import CalibrationFeatures
from winnow.calibration.calibrator import ProbabilityCalibrator, TrainingHistory
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.datasets.data_loaders import WinnowDatasetLoader
from winnow.datasets.feature_dataset import FeatureDataset
from winnow.scripts.main import (
    _compute_features_batched_metadata,
    _discover_experiment_files,
    _fit_from_calibration_datasets,
)


class RecordingCalibrator:
    """Minimal calibrator that records feature computation calls."""

    def __init__(self):
        self.compute_features_calls = 0

    def compute_features(self, dataset):
        """Record the call and add a marker feature column."""
        self.compute_features_calls += 1
        dataset.metadata["marker_feature"] = 1.0


class FakeLoader:
    """Loader that ignores file contents and returns metadata keyed by path name."""

    def load(self, data_path, predictions_path=None):
        """Return a tiny CalibrationDataset whose confidence encodes the file stem."""
        stem = Path(data_path).stem
        metadata = pd.DataFrame(
            {
                "confidence": [float(len(stem))],
                "prediction": [["A"]],
            }
        )
        return CalibrationDataset(metadata=metadata, predictions=[None])


class CapturingCalibrator:
    """Delegates to_feature_dataset to a real calibrator and captures FeatureDatasets."""

    def __init__(self, source: ProbabilityCalibrator):
        self._source = source
        self.captured: tuple[FeatureDataset | None, FeatureDataset | None] | None = None

    def to_feature_dataset(self, dataset):
        return self._source.to_feature_dataset(dataset)

    def fit_from_features(self, train_fd, val_fd):
        self.captured = (train_fd, val_fd)
        return TrainingHistory(train_losses=[], epochs_trained=0)


class _BridgeFeature(CalibrationFeatures):
    """Single-column feature used only for CLI fit-bridge freezes."""

    @property
    def name(self):
        return "feat"

    @property
    def columns(self):
        return ["f1"]

    @property
    def dependencies(self):
        return []

    def prepare(self, dataset):
        pass

    def compute(self, dataset):
        pass


def _labelled_calibration_dataset() -> CalibrationDataset:
    """CalibrationDataset with feature columns already present on metadata."""
    metadata = pd.DataFrame(
        {
            "confidence": [0.9, 0.5],
            "f1": [1.0, 2.0],
            "correct": [1.0, 0.0],
        }
    )
    return CalibrationDataset(metadata=metadata, predictions=[None, None])


# Absolute FeatureDataset contents for the labelled bridge metadata above.
_BRIDGE_FEATURES_F32 = np.array([[0.9, 1.0], [0.5, 2.0]], dtype=np.float32)
_BRIDGE_LABELS_F32 = np.array([1.0, 0.0], dtype=np.float32)


def _calibrator_with_bridge_feature() -> ProbabilityCalibrator:
    calibrator = ProbabilityCalibrator(seed=0)
    calibrator.add_feature(_BridgeFeature())
    return calibrator


def test_compute_features_loads_saved_winnow_dataset_directory(tmp_path):
    """Saved Winnow dataset directories should be passed directly to the loader."""
    metadata = pd.DataFrame(
        {
            "prediction": ["AG", "MG"],
            "confidence": [0.9, 0.8],
            "mz_array": ["[100.0, 200.0]", "[150.0, 250.0]"],
            "intensity_array": ["[1000.0, 2000.0]", "[1500.0, 2500.0]"],
        }
    )
    metadata.to_csv(tmp_path / "metadata.csv", index=False)

    loader = WinnowDatasetLoader(
        residue_masses={
            "A": 71.037114,
            "G": 57.021464,
            "M": 131.040485,
        },
        residue_remapping={},
    )
    calibrator = RecordingCalibrator()

    all_metadata = _compute_features_batched_metadata(
        spectrum_path=tmp_path,
        predictions_path=None,
        data_loader=loader,
        calibrator=calibrator,
        labelled=False,
    )

    assert calibrator.compute_features_calls == 1
    assert len(all_metadata) == 1
    assert all_metadata[0]["prediction"].tolist() == [["A", "G"], ["M", "G"]]
    assert all_metadata[0]["marker_feature"].tolist() == [1.0, 1.0]


def test_discover_experiment_files_sorted_extensions(tmp_path):
    """Discovery returns sorted supported spectrum files only."""
    (tmp_path / "b.mgf").write_text("")
    (tmp_path / "a.parquet").write_text("")
    (tmp_path / "c.ipc").write_text("")
    (tmp_path / "ignore.txt").write_text("")
    (tmp_path / "subdir").mkdir()

    files = _discover_experiment_files(tmp_path)
    assert [f.name for f in files] == ["a.parquet", "b.mgf", "c.ipc"]

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No spectrum files found"):
        _discover_experiment_files(empty)


def test_compute_features_directory_one_call_per_file(tmp_path):
    """Directory mode runs compute_features once per discovered spectrum file."""
    (tmp_path / "exp_b.parquet").write_text("")
    (tmp_path / "exp_a.parquet").write_text("")

    calibrator = RecordingCalibrator()
    all_metadata = _compute_features_batched_metadata(
        spectrum_path=tmp_path,
        predictions_path=None,
        data_loader=FakeLoader(),
        calibrator=calibrator,
        labelled=False,
    )

    assert calibrator.compute_features_calls == 2
    assert len(all_metadata) == 2
    assert [frame["confidence"].tolist() for frame in all_metadata] == [
        [float(len("exp_a"))],
        [float(len("exp_b"))],
    ]
    assert all(frame["marker_feature"].tolist() == [1.0] for frame in all_metadata)


def test_compute_features_directory_labelled_requires_sequence(tmp_path):
    """labelled=True rejects experiment files whose metadata lack sequence."""
    (tmp_path / "exp.parquet").write_text("")

    with pytest.raises(ValueError, match="sequence"):
        _compute_features_batched_metadata(
            spectrum_path=tmp_path,
            predictions_path=None,
            data_loader=FakeLoader(),
            calibrator=RecordingCalibrator(),
            labelled=True,
        )


def test_fit_from_calibration_datasets_captures_feature_datasets():
    """CLI fit bridge wraps extract output into float32 FeatureDatasets."""
    source = _calibrator_with_bridge_feature()
    train_data = _labelled_calibration_dataset()
    val_data = _labelled_calibration_dataset()
    capturing = CapturingCalibrator(source)

    history = _fit_from_calibration_datasets(capturing, train_data, val_data)

    assert isinstance(history, TrainingHistory)
    assert capturing.captured is not None
    train_fd, val_fd = capturing.captured
    assert train_fd is not None
    assert val_fd is not None
    assert train_fd.features.dtype == torch.float32
    assert train_fd.labels.dtype == torch.float32
    assert val_fd.features.dtype == torch.float32
    assert val_fd.labels.dtype == torch.float32

    np.testing.assert_allclose(
        train_fd.features.numpy(), _BRIDGE_FEATURES_F32, rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        train_fd.labels.numpy(), _BRIDGE_LABELS_F32, rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        val_fd.features.numpy(), _BRIDGE_FEATURES_F32, rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        val_fd.labels.numpy(), _BRIDGE_LABELS_F32, rtol=0, atol=1e-6
    )

    expected_features, expected_labels = source._extract_feature_matrix(
        train_data, labelled=True
    )
    assert expected_features.dtype == np.float64
    assert expected_labels.dtype == np.float64
    np.testing.assert_allclose(
        train_fd.features.numpy(),
        np.asarray(expected_features, dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        train_fd.labels.numpy(),
        np.asarray(expected_labels, dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        val_fd.features.numpy(),
        train_fd.features.numpy(),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        val_fd.labels.numpy(),
        train_fd.labels.numpy(),
        rtol=0,
        atol=1e-6,
    )


def test_fit_from_calibration_datasets_without_validation():
    """CLI fit bridge passes val_fd=None when validation data is absent."""
    source = _calibrator_with_bridge_feature()
    capturing = CapturingCalibrator(source)

    _fit_from_calibration_datasets(capturing, _labelled_calibration_dataset(), None)

    assert capturing.captured is not None
    train_fd, val_fd = capturing.captured
    assert train_fd is not None
    assert val_fd is None
    assert train_fd.features.dtype == torch.float32
    assert train_fd.labels.dtype == torch.float32
    np.testing.assert_allclose(
        train_fd.features.numpy(), _BRIDGE_FEATURES_F32, rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        train_fd.labels.numpy(), _BRIDGE_LABELS_F32, rtol=0, atol=1e-6
    )
