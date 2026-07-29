"""Tests for PrimeNovoDatasetLoader: join on {experiment_name}:{title|label}."""

from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from omegaconf import OmegaConf

from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.datasets.data_loaders import PrimeNovoDatasetLoader

_PRIMENOVO_REMAPPING = {
    "M[+15.995]": "M[UNIMOD:35]",
    "C[+57.021]": "C[UNIMOD:4]",
    "N[+0.984]": "N[UNIMOD:7]",
    "Q[+0.984]": "Q[UNIMOD:7]",
    "[+42.011]": "[UNIMOD:1]",
    "[+43.006]": "[UNIMOD:5]",
    "[-17.027]": "[UNIMOD:385]",
}

_MID_SEQUENCE_N_TERMINAL_MODS = [
    "[+43.006-17.027]",
    "[+42.011]",
    "[+43.006]",
    "[-17.027]",
]


class TestPrimeNovoDatasetLoader:
    """Tests for PrimeNovoDatasetLoader: spectrum_id from TITLE/label, then join."""

    @pytest.fixture()
    def loader(self, full_residue_masses):
        return PrimeNovoDatasetLoader(
            residue_masses=full_residue_masses,
            residue_remapping=_PRIMENOVO_REMAPPING,
            mid_sequence_n_terminal_mods=_MID_SEQUENCE_N_TERMINAL_MODS,
        )

    @pytest.fixture()
    def mgf_path(self, tmp_path):
        path = tmp_path / "spectra.mgf"
        path.write_text(
            "BEGIN IONS\n"
            "TITLE=run_X_SCANS_1\n"
            "PEPMASS=358.5\n"
            "CHARGE=3+\n"
            "RTINSECONDS=757.5\n"
            "SEQ=GVSREEIQR\n"
            "100.0 1.0\n"
            "200.0 2.0\n"
            "END IONS\n"
            "BEGIN IONS\n"
            "TITLE=run_X_SCANS_2\n"
            "PEPMASS=500.0\n"
            "CHARGE=2+\n"
            "RTINSECONDS=820.0\n"
            "SEQ=YEDMNNYDPTK\n"
            "150.0 1.5\n"
            "END IONS\n",
            encoding="utf-8",
        )
        return path

    @pytest.fixture()
    def tsv_path(self, tmp_path):
        path = tmp_path / "denovo.tsv"
        df = pd.DataFrame(
            {
                "label": ["run_X_SCANS_1", "run_X_SCANS_2"],
                "prediction": ["GVSREELQR", "YEDM[+15.995]NNYDPTK"],
                "charge": [3, 3],
                "score": [0.9780, 0.0002],
            }
        )
        df.to_csv(path, sep="\t", index=False)
        return path

    def test_initialization(self, loader):
        assert loader.metrics is not None
        assert loader.metrics.residue_set is not None

    def test_load_predictions_reads_tsv(self, loader, tsv_path):
        df = loader._load_predictions(tsv_path)
        assert list(df.columns) == ["label", "prediction", "charge", "score"]
        assert len(df) == 2

    def test_load_predictions_rejects_unsupported_suffix(self, loader, tmp_path):
        path = tmp_path / "preds.csv"
        path.write_text("label\tprediction\tscore\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader._load_predictions(path)

    def test_load_predictions_raises_on_missing_required_column(self, loader, tmp_path):
        path = tmp_path / "missing.tsv"
        pd.DataFrame({"label": ["a"], "prediction": ["AG"]}).to_csv(
            path, sep="\t", index=False
        )
        with pytest.raises(ValueError, match="missing required column"):
            loader._load_predictions(path)

    def test_load_predictions_rejects_score_outside_unit_interval(
        self, loader, tmp_path
    ):
        path = tmp_path / "bad_score.tsv"
        pd.DataFrame(
            {
                "label": ["a"],
                "prediction": ["AG"],
                "score": [1.5],
            }
        ).to_csv(path, sep="\t", index=False)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            loader._load_predictions(path)

    def test_load_predictions_rejects_missing_score(self, loader, tmp_path):
        path = tmp_path / "missing_score.tsv"
        pd.DataFrame(
            {
                "label": ["a"],
                "prediction": ["AG"],
                "score": [None],
            }
        ).to_csv(path, sep="\t", index=False)
        with pytest.raises(ValueError, match="missing score"):
            loader._load_predictions(path)

    def test_load_spectrum_data_sets_spectrum_id_from_title(self, loader, mgf_path):
        df, has_labels = loader._load_spectrum_data(mgf_path)
        assert has_labels is True
        assert df["spectrum_id"].to_list() == [
            "spectra:run_X_SCANS_1",
            "spectra:run_X_SCANS_2",
        ]
        assert df["experiment_name"].to_list() == ["spectra", "spectra"]
        assert df["title"].to_list() == ["run_X_SCANS_1", "run_X_SCANS_2"]

    def test_load_spectrum_data_rejects_unsupported_suffix(self, loader, tmp_path):
        path = tmp_path / "spec.csv"
        path.write_text("col\n1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader._load_spectrum_data(path)

    def test_load_spectrum_data_raises_on_duplicate_titles(self, loader, tmp_path):
        path = tmp_path / "dup.mgf"
        path.write_text(
            "BEGIN IONS\nTITLE=same\nPEPMASS=100\nCHARGE=2+\n100.0 1.0\nEND IONS\n"
            "BEGIN IONS\nTITLE=same\nPEPMASS=200\nCHARGE=2+\n200.0 1.0\nEND IONS\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="TITLE values must be unique"):
            loader._load_spectrum_data(path)

    def test_load_spectrum_data_raises_on_missing_title(self, loader, tmp_path):
        path = tmp_path / "missing_title.mgf"
        path.write_text(
            "BEGIN IONS\nPEPMASS=100\nCHARGE=2+\n100.0 1.0\nEND IONS\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="non-empty TITLE"):
            loader._load_spectrum_data(path)

    def test_add_prediction_spectrum_ids_uses_label(self, loader):
        preds = pl.DataFrame(
            {"label": ["t1", "t2"], "prediction": ["AG", "MG"], "score": [0.9, 0.5]}
        )
        result = loader._add_prediction_spectrum_ids(preds, "spectra")
        assert result["spectrum_id"].to_list() == ["spectra:t1", "spectra:t2"]
        assert result["label"].to_list() == ["t1", "t2"]

    def test_process_predictions_renames_score_and_sets_untokenised(self, loader):
        preds = pl.DataFrame(
            {
                "label": ["t1"],
                "prediction": ["YEDM[+15.995]NNYDPTK"],
                "score": [0.42],
            }
        )
        result = loader._process_predictions(preds, [], "spectra")
        assert result["confidence"].to_list() == pytest.approx([0.42])
        assert result["prediction_untokenised"].to_list() == ["YEDM[+15.995]NNYDPTK"]
        assert result["spectrum_id"].to_list() == ["spectra:t1"]
        assert "score" not in result.columns

    def test_process_predictions_preserves_l_in_prediction_untokenised(self, loader):
        preds = pl.DataFrame(
            {
                "label": ["t"],
                "prediction": ["GVSREELQR"],
                "score": [0.5],
            }
        )
        result = loader._process_predictions(preds, [], "spectra")
        assert result["prediction_untokenised"].to_list() == ["GVSREELQR"]

    def test_process_predictions_filters_mid_sequence_nterm_mods(self, loader):
        preds = pl.DataFrame(
            {
                "label": ["a", "b"],
                "prediction": ["AG[+42.011]K", "AG"],
                "score": [0.1, 0.9],
            }
        )
        result = loader._process_predictions(preds, [], "spectra")
        assert result["prediction"].to_list() == ["AG"]
        assert len(result) == 1

    def test_process_predictions_respects_configured_nterm_mods(
        self, full_residue_masses
    ):
        loader = PrimeNovoDatasetLoader(
            residue_masses=full_residue_masses,
            residue_remapping=_PRIMENOVO_REMAPPING,
            mid_sequence_n_terminal_mods=["[+99.000]"],
        )
        preds = pl.DataFrame(
            {
                "label": ["a", "b"],
                "prediction": ["AG[+42.011]K", "A[+99.000]G"],
                "score": [0.1, 0.9],
            }
        )
        result = loader._process_predictions(preds, [], "spectra")
        # Default acetylation mid-seq is no longer filtered; custom mod is.
        assert result["prediction"].to_list() == ["AG[+42.011]K"]
        assert len(result) == 1

    def test_process_predictions_drops_clashing_prediction_columns(self, loader):
        preds = pl.DataFrame(
            {
                "label": ["t1"],
                "prediction": ["AG"],
                "score": [0.9],
                "experiment_name": ["wrong"],
            }
        )
        result = loader._process_predictions(
            preds, ["spectrum_id", "title", "experiment_name"], "spectra"
        )
        assert "experiment_name" not in result.columns
        assert result["spectrum_id"].to_list() == ["spectra:t1"]

    def test_merge_data_joins_on_spectrum_id(self, loader):
        preds = loader._process_predictions(
            pl.DataFrame(
                {
                    "label": ["t1", "t2"],
                    "prediction": ["AG", "MG"],
                    "score": [0.9, 0.5],
                }
            ),
            ["spectrum_id", "experiment_name", "title", "mz_array"],
            "spectra",
        )
        spectra = pl.DataFrame(
            {
                "spectrum_id": ["spectra:t1", "spectra:t2"],
                "experiment_name": ["spectra", "spectra"],
                "title": ["t1", "t2"],
                "mz_array": [[1.0], [2.0]],
            }
        )
        merged = loader._merge_data(spectra, preds)
        assert len(merged) == 2
        assert merged["spectrum_id"].to_list() == ["spectra:t1", "spectra:t2"]
        assert merged["label"].to_list() == ["t1", "t2"]
        assert "mz_array" in merged.columns

    def test_merge_data_raises_when_no_matches(self, loader):
        preds = loader._process_predictions(
            pl.DataFrame(
                {
                    "label": ["t1", "t2"],
                    "prediction": ["AG", "MG"],
                    "score": [0.9, 0.5],
                }
            ),
            ["spectrum_id", "experiment_name", "title", "mz_array"],
            "spectra",
        )
        spectra = pl.DataFrame(
            {
                "spectrum_id": ["spectra:other"],
                "experiment_name": ["spectra"],
                "title": ["other"],
                "mz_array": [[1.0]],
            }
        )
        with pytest.raises(ValueError, match="spectrum_id values not present"):
            loader._merge_data(spectra, preds)

    def test_merge_data_raises_on_orphan_prediction(self, loader):
        preds = loader._process_predictions(
            pl.DataFrame(
                {
                    "label": ["t1", "missing"],
                    "prediction": ["AG", "MG"],
                    "score": [0.9, 0.5],
                }
            ),
            ["spectrum_id", "experiment_name", "title", "mz_array"],
            "spectra",
        )
        spectra = pl.DataFrame(
            {
                "spectrum_id": ["spectra:t1"],
                "experiment_name": ["spectra"],
                "title": ["t1"],
                "mz_array": [[1.0]],
            }
        )
        with pytest.raises(ValueError, match="spectrum_id values not present"):
            loader._merge_data(spectra, preds)

    def test_merge_data_raises_on_duplicate_prediction_labels(self, loader):
        preds = loader._process_predictions(
            pl.DataFrame(
                {
                    "label": ["t1", "t1"],
                    "prediction": ["AG", "MG"],
                    "score": [0.9, 0.5],
                }
            ),
            ["spectrum_id", "experiment_name", "title", "mz_array"],
            "spectra",
        )
        spectra = pl.DataFrame(
            {
                "spectrum_id": ["spectra:t1"],
                "experiment_name": ["spectra"],
                "title": ["t1"],
                "mz_array": [[1.0]],
            }
        )
        with pytest.raises(ValueError, match="duplicate TSV label"):
            loader._merge_data(spectra, preds)

    def test_merge_data_prefers_spectrum_experiment_name(self, loader):
        preds = loader._process_predictions(
            pl.DataFrame(
                {
                    "label": ["t1"],
                    "prediction": ["AG"],
                    "score": [0.9],
                    "experiment_name": ["wrong"],
                }
            ),
            ["spectrum_id", "title", "experiment_name"],
            "spectra",
        )
        spectra = pl.DataFrame(
            {
                "spectrum_id": ["spectra:t1"],
                "title": ["t1"],
                "experiment_name": ["spectra"],
            }
        )
        merged = loader._merge_data(spectra, preds)
        assert merged["experiment_name"].to_list() == ["spectra"]

    def test_load_raises_when_predictions_path_is_none(self, loader, tmp_path):
        with pytest.raises(ValueError, match="predictions_path is required"):
            loader.load(data_path=tmp_path)

    def test_load_returns_calibration_dataset_without_beams(
        self, loader, mgf_path, tsv_path
    ):
        dataset = loader.load(data_path=mgf_path, predictions_path=tsv_path)
        assert isinstance(dataset, CalibrationDataset)
        assert dataset.predictions is None

    def test_load_aligns_on_spectrum_id(self, loader, mgf_path, tsv_path):
        dataset = loader.load(data_path=mgf_path, predictions_path=tsv_path)
        meta = dataset.metadata
        assert len(meta) == 2
        assert meta["spectrum_id"].tolist() == [
            "spectra:run_X_SCANS_1",
            "spectra:run_X_SCANS_2",
        ]
        assert meta["label"].tolist() == ["run_X_SCANS_1", "run_X_SCANS_2"]
        assert all(isinstance(p, list) for p in meta["prediction"])
        assert "score" not in meta.columns
        assert meta["confidence"].between(0.0, 1.0).all()

    def test_load_tokenises_and_remaps_via_finalize(self, loader, mgf_path, tsv_path):
        dataset = loader.load(data_path=mgf_path, predictions_path=tsv_path)
        meta = dataset.metadata
        row = meta.set_index("spectrum_id").loc["spectra:run_X_SCANS_2"]
        assert "M[UNIMOD:35]" in row["prediction"]
        assert "M[+15.995]" in row["prediction_untokenised"]

    def test_load_applies_token_level_l_to_i(self, loader, mgf_path, tsv_path):
        dataset = loader.load(data_path=mgf_path, predictions_path=tsv_path)
        meta = dataset.metadata
        row = meta.set_index("spectrum_id").loc["spectra:run_X_SCANS_1"]
        assert "L" not in row["prediction"]
        assert "L" in row["prediction_untokenised"]

    def test_load_inner_join_keeps_only_matching_spectrum_ids(
        self, loader, mgf_path, tmp_path
    ):
        path = tmp_path / "short.tsv"
        pd.DataFrame(
            {
                "label": ["run_X_SCANS_1"],
                "prediction": ["GVSREELQR"],
                "charge": [3],
                "score": [0.978],
            }
        ).to_csv(path, sep="\t", index=False)
        dataset = loader.load(data_path=mgf_path, predictions_path=path)
        assert len(dataset.metadata) == 1
        assert dataset.metadata["spectrum_id"].tolist() == ["spectra:run_X_SCANS_1"]

    def test_load_raises_when_label_has_no_matching_title(
        self, loader, mgf_path, tmp_path
    ):
        path = tmp_path / "bad.tsv"
        pd.DataFrame(
            {
                "label": ["not_in_mgf"],
                "prediction": ["GVSREELQR"],
                "charge": [3],
                "score": [0.978],
            }
        ).to_csv(path, sep="\t", index=False)
        with pytest.raises(ValueError, match="spectrum_id values not present"):
            loader.load(data_path=mgf_path, predictions_path=path)

    def test_load_evaluates_against_ground_truth(self, loader, mgf_path, tsv_path):
        dataset = loader.load(data_path=mgf_path, predictions_path=tsv_path)
        meta = dataset.metadata
        assert "valid_sequence" in meta.columns
        assert "valid_prediction" in meta.columns
        assert "num_matches" in meta.columns
        assert "correct" in meta.columns
        row = meta.set_index("spectrum_id").loc["spectra:run_X_SCANS_1"]
        assert bool(row["correct"]) is True
        assert row["num_matches"] == len(row["sequence"])


def test_primenovo_config_target() -> None:
    """Bundled Hydra YAML points at PrimeNovoDatasetLoader."""
    config_dir = Path(__file__).parents[3] / "winnow" / "configs"
    cfg = OmegaConf.load(config_dir / "data_loader" / "primenovo.yaml")
    assert cfg._target_ == "winnow.datasets.data_loaders.PrimeNovoDatasetLoader"
