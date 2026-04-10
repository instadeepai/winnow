"""Tests for the predict output format options."""

import os
import tempfile

import pandas as pd
import pytest

from winnow.scripts.main import separate_metadata_and_predictions


@pytest.fixture()
def sample_metadata():
    """Create a sample metadata DataFrame with all expected columns."""
    return pd.DataFrame(
        {
            "spectrum_id": ["s1:1", "s1:2", "s2:1"],
            "prediction": ["PEPTIDEK", "ALANINE", "TESTSEQ"],
            "calibrated_confidence": [0.95, 0.80, 0.90],
            "psm_fdr": [0.01, 0.05, 0.02],
            "psm_q_value": [0.01, 0.05, 0.02],
            "psm_pep": [0.005, 0.03, 0.01],
            "precursor_mz": [500.0, 600.0, 700.0],
            "precursor_charge": [2, 3, 2],
            "retention_time": [120.5, 200.3, 150.1],
            "experiment_name": ["sample1", "sample1", "sample2"],
        }
    )


class TestSeparateMetadataAndPredictions:
    """Test the legacy split output function."""

    def test_split_with_nonparametric_fdr(self, sample_metadata):
        # Patch the isinstance check by making our fake class match
        from winnow.fdr.nonparametric import NonParametricFDRControl

        fdr_control = NonParametricFDRControl.__new__(NonParametricFDRControl)

        meta, preds = separate_metadata_and_predictions(
            sample_metadata, fdr_control, "calibrated_confidence"
        )

        # preds should have spectrum_id + prediction + confidence + fdr + q_value + pep
        assert "spectrum_id" in preds.columns
        assert "prediction" in preds.columns
        assert "calibrated_confidence" in preds.columns
        assert "psm_q_value" in preds.columns
        assert "psm_pep" in preds.columns

        # metadata should NOT have prediction columns
        assert "prediction" not in meta.columns
        assert "calibrated_confidence" not in meta.columns

        # metadata should still have spectrum metadata
        assert "spectrum_id" in meta.columns
        assert "precursor_mz" in meta.columns
        assert "retention_time" in meta.columns

    def test_split_preserves_all_rows(self, sample_metadata):
        from winnow.fdr.nonparametric import NonParametricFDRControl

        fdr_control = NonParametricFDRControl.__new__(NonParametricFDRControl)

        meta, preds = separate_metadata_and_predictions(
            sample_metadata, fdr_control, "calibrated_confidence"
        )

        assert len(meta) == 3
        assert len(preds) == 3


class TestMergedOutput:
    """Test the new merged output format."""

    def test_merged_output_writes_single_tsv(self, sample_metadata):
        with tempfile.TemporaryDirectory() as tmpdir:
            merged_path = os.path.join(tmpdir, "calibrated_psms.tsv")
            sample_metadata.to_csv(merged_path, sep="\t", index=False)

            # Verify file exists and contains all columns
            result = pd.read_csv(merged_path, sep="\t")
            assert len(result) == 3
            assert "spectrum_id" in result.columns
            assert "prediction" in result.columns
            assert "calibrated_confidence" in result.columns
            assert "psm_q_value" in result.columns
            assert "precursor_mz" in result.columns
            assert "retention_time" in result.columns
            assert "experiment_name" in result.columns


class TestDirectoryParquetLoading:
    """Test that InstaNovoDatasetLoader can load parquet directories."""

    def test_load_parquet_directory(self, tmp_path):
        """Test loading multiple parquet files from a directory."""
        import polars as pl

        # Create two small parquet files
        df1 = pl.DataFrame(
            {
                "mz_array": [[100.0, 200.0]],
                "intensity_array": [[1000.0, 2000.0]],
                "precursor_mz": [500.0],
                "precursor_charge": [2],
                "retention_time": [120.0],
            }
        )
        df2 = pl.DataFrame(
            {
                "mz_array": [[150.0, 250.0]],
                "intensity_array": [[1500.0, 2500.0]],
                "precursor_mz": [600.0],
                "precursor_charge": [3],
                "retention_time": [200.0],
            }
        )

        df1.write_parquet(tmp_path / "shard_0001.parquet")
        df2.write_parquet(tmp_path / "shard_0002.parquet")

        from winnow.datasets.data_loaders import InstaNovoDatasetLoader

        loader = InstaNovoDatasetLoader(
            residue_masses={},
            residue_remapping={},
            add_index_cols=False,
        )

        result, has_labels = loader._load_spectrum_data(tmp_path)
        assert len(result) == 2
        assert result["precursor_mz"].to_list() == [500.0, 600.0]

    def test_empty_directory_raises(self, tmp_path):
        """Test that an empty directory raises ValueError."""
        from winnow.datasets.data_loaders import InstaNovoDatasetLoader

        loader = InstaNovoDatasetLoader(
            residue_masses={},
            residue_remapping={},
            add_index_cols=False,
        )

        with pytest.raises(ValueError, match="No .parquet files found"):
            loader._load_spectrum_data(tmp_path)
