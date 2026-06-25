"""Tests for compute-features helper paths."""

import pandas as pd

from winnow.datasets.data_loaders import WinnowDatasetLoader
from winnow.scripts.main import _compute_features_batched_metadata


class RecordingCalibrator:
    """Minimal calibrator that records feature computation calls."""

    def __init__(self):
        self.compute_features_calls = 0

    def compute_features(self, dataset):
        """Record the call and add a marker feature column."""
        self.compute_features_calls += 1
        dataset.metadata["marker_feature"] = 1.0


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
