"""Unit tests for winnow.datasets.feature_dataset."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from winnow.datasets.feature_dataset import FeatureDataset


class TestFeatureDataset:
    """Test the FeatureDataset class."""

    def test_from_arrays(self):
        """Test construction from numpy arrays."""
        features = np.random.randn(10, 3).astype(np.float32)
        labels = np.random.choice([0.0, 1.0], 10).astype(np.float32)
        ds = FeatureDataset(features=features, labels=labels)

        assert len(ds) == 10
        x, y = ds[0]
        assert x.shape == (3,)
        assert y.shape == ()

    def test_tensors_are_float32(self):
        """Test that features and labels are stored as float32 tensors."""
        ds = FeatureDataset(
            features=np.ones((5, 2), dtype=np.float64),
            labels=np.ones(5, dtype=np.int32),
        )
        assert ds.features.dtype == torch.float32
        assert ds.labels.dtype == torch.float32

    def test_mismatched_lengths_raises(self):
        """Test that mismatched features/labels raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            FeatureDataset(
                features=np.zeros((10, 3)),
                labels=np.zeros(5),
            )

    def test_dataloader_integration(self):
        """Test that FeatureDataset works correctly with PyTorch DataLoader."""
        n = 25
        n_features = 4
        features = np.random.randn(n, n_features).astype(np.float32)
        labels = np.random.choice([0.0, 1.0], n).astype(np.float32)
        ds = FeatureDataset(features=features, labels=labels)

        loader = DataLoader(ds, batch_size=8, shuffle=True)
        total_samples = 0
        for batch_x, batch_y in loader:
            assert batch_x.shape[1] == n_features
            assert batch_x.dtype == torch.float32
            assert batch_y.dtype == torch.float32
            total_samples += len(batch_x)

        assert total_samples == n

    def test_from_parquet_single_file(self, tmp_path):
        """Test loading from a single Parquet file with correct values."""
        import polars as pl

        df = pl.DataFrame(
            {
                "feature_a": [1.0, 2.0, 3.0],
                "feature_b": [4.0, 5.0, 6.0],
                "correct": [1.0, 0.0, 1.0],
            }
        )
        path = tmp_path / "data.parquet"
        df.write_parquet(path)

        ds = FeatureDataset.from_parquet(path)
        assert len(ds) == 3
        x, y = ds[0]
        assert x.shape == (2,)
        torch.testing.assert_close(x, torch.tensor([1.0, 4.0]))
        assert y.item() == 1.0

    def test_from_parquet_directory_preserves_values(self, tmp_path):
        """Test loading from a directory preserves values across files."""
        import polars as pl

        for i in range(3):
            df = pl.DataFrame(
                {
                    "feat": [float(i * 10 + j) for j in range(5)],
                    "correct": [1.0, 0.0, 1.0, 0.0, 1.0],
                }
            )
            df.write_parquet(tmp_path / f"part_{i}.parquet")

        ds = FeatureDataset.from_parquet(tmp_path)
        assert len(ds) == 15

        all_feats = ds.features[:, 0].tolist()
        assert 0.0 in all_feats
        assert 10.0 in all_feats
        assert 20.0 in all_feats

    def test_from_parquet_missing_correct_raises(self, tmp_path):
        """Test that missing 'correct' column raises ValueError."""
        import polars as pl

        df = pl.DataFrame({"feature_a": [1.0, 2.0]})
        path = tmp_path / "data.parquet"
        df.write_parquet(path)

        with pytest.raises(ValueError, match="correct"):
            FeatureDataset.from_parquet(path)

    def test_from_parquet_empty_dir_raises(self, tmp_path):
        """Test that empty directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No .parquet"):
            FeatureDataset.from_parquet(tmp_path)

    def test_from_parquet_feature_columns(self, tmp_path):
        """Test explicit feature column selection."""
        import polars as pl

        df = pl.DataFrame(
            {
                "confidence": [0.9, 0.8],
                "feature_a": [1.0, 2.0],
                "spectrum_id": [0, 1],
                "correct": [1.0, 0.0],
            }
        )
        path = tmp_path / "data.parquet"
        df.write_parquet(path)

        ds = FeatureDataset.from_parquet(
            path, feature_columns=["confidence", "feature_a"]
        )
        assert ds.features.shape == (2, 2)
        torch.testing.assert_close(
            ds.features[0], torch.tensor([0.9, 1.0], dtype=torch.float32)
        )

    def test_from_parquet_missing_feature_column_raises(self, tmp_path):
        """Test that missing feature_columns raises ValueError."""
        import polars as pl

        df = pl.DataFrame({"feature_a": [1.0], "correct": [1.0]})
        path = tmp_path / "data.parquet"
        df.write_parquet(path)

        with pytest.raises(ValueError, match="missing from Parquet"):
            FeatureDataset.from_parquet(path, feature_columns=["confidence"])

    def test_from_parquet_boolean_feature_columns(self, tmp_path):
        """Test that boolean feature and label columns cast to float32."""
        import polars as pl

        df = pl.DataFrame(
            {
                "confidence": [0.9, 0.1],
                "is_missing": [True, False],
                "correct": [True, False],
            }
        )
        path = tmp_path / "data.parquet"
        df.write_parquet(path)

        ds = FeatureDataset.from_parquet(
            path, feature_columns=["confidence", "is_missing"]
        )
        assert ds.features.shape == (2, 2)
        torch.testing.assert_close(
            ds.features[0], torch.tensor([0.9, 1.0], dtype=torch.float32)
        )
        torch.testing.assert_close(
            ds.labels, torch.tensor([1.0, 0.0], dtype=torch.float32)
        )

    def test_from_parquet_non_numeric_feature_column_raises(self, tmp_path):
        """Test that string feature columns raise a schema ValueError."""
        import polars as pl

        df = pl.DataFrame(
            {
                "confidence": [0.9, 0.8],
                "peptide": ["PEPTIDE", "OTHER"],
                "correct": [1.0, 0.0],
            }
        )
        path = tmp_path / "data.parquet"
        df.write_parquet(path)

        with pytest.raises(ValueError, match="must be numeric or boolean.*peptide"):
            FeatureDataset.from_parquet(path, feature_columns=["confidence", "peptide"])

    def test_from_parquet_non_numeric_label_raises(self, tmp_path):
        """Test that a non-castable correct column raises ValueError."""
        import polars as pl

        df = pl.DataFrame(
            {
                "feature_a": [1.0, 2.0],
                "correct": ["yes", "no"],
            }
        )
        path = tmp_path / "data.parquet"
        df.write_parquet(path)

        with pytest.raises(ValueError, match="Label column 'correct'"):
            FeatureDataset.from_parquet(path, feature_columns=["feature_a"])
