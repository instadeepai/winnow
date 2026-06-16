"""Tests for InstaNovoDatasetLoader."""

import ast

import numpy as np
import pandas as pd
import polars as pl
import pytest

from winnow.datasets.data_loaders import InstaNovoDatasetLoader
from tests.datasets.data_loaders.conftest import (
    _FULL_RESIDUE_MASSES,
    _STANDARD_REMAPPING,
)


class TestInstaNovoDatasetLoader:
    """Tests for InstaNovoDatasetLoader: residue remapping and internal methods."""

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @pytest.fixture()
    def residue_masses(self):
        """Minimal residue masses used by the notation-remapping tests."""
        return {
            "G": 57.021464,
            "A": 71.037114,
            "S": 87.032028,
            "P": 97.052764,
            "M": 131.040485,
            "C": 103.009185,
            "N": 114.042927,
            "Q": 128.058578,
            "T": 101.047670,
            "Y": 163.063329,
            # Modifications
            "M[UNIMOD:35]": 147.035400,
            "C[UNIMOD:4]": 160.030649,
            "N[UNIMOD:7]": 115.026943,
            "Q[UNIMOD:7]": 129.042594,
            "S[UNIMOD:21]": 166.998028,
            "T[UNIMOD:21]": 181.01367,
            "Y[UNIMOD:21]": 243.029329,
            # Terminal modifications
            "[UNIMOD:1]": 42.010565,
            "[UNIMOD:5]": 43.005814,
            "[UNIMOD:385]": -17.026549,
        }

    @pytest.fixture()
    def residue_remapping(self):
        """Residue remapping for InstaNovo legacy notations to UNIMOD tokens."""
        return {
            "M(ox)": "M[UNIMOD:35]",  # Oxidation
            "M(+15.99)": "M[UNIMOD:35]",  # Oxidation
            "S(p)": "S[UNIMOD:21]",  # Phosphorylation
            "T(p)": "T[UNIMOD:21]",  # Phosphorylation
            "Y(p)": "Y[UNIMOD:21]",  # Phosphorylation
            "S(+79.97)": "S[UNIMOD:21]",  # Phosphorylation
            "T(+79.97)": "T[UNIMOD:21]",  # Phosphorylation
            "Y(+79.97)": "Y[UNIMOD:21]",  # Phosphorylation
            "Q(+0.98)": "Q[UNIMOD:7]",  # Deamidation
            "N(+0.98)": "N[UNIMOD:7]",  # Deamidation
            "Q(+.98)": "Q[UNIMOD:7]",  # Deamidation
            "N(+.98)": "N[UNIMOD:7]",  # Deamidation
            "C(+57.02)": "C[UNIMOD:4]",  # Carbamidomethylation
            # N-terminal modifications
            "(+42.01)": "[UNIMOD:1]",  # Acetylation
            "(+43.01)": "[UNIMOD:5]",  # Carbamylation
            "(-17.03)": "[UNIMOD:385]",  # Ammonia loss
        }

    @pytest.fixture()
    def instanovo_loader(self, residue_masses, residue_remapping):
        """InstaNovoDatasetLoader with InstaNovo-style remapping for notation tests."""
        return InstaNovoDatasetLoader(
            residue_masses=residue_masses,
            residue_remapping=residue_remapping,
        )

    @pytest.fixture()
    def loader(self):
        """InstaNovoDatasetLoader with full residues for internal-method tests."""
        return InstaNovoDatasetLoader(
            residue_masses=_FULL_RESIDUE_MASSES,
            residue_remapping=_STANDARD_REMAPPING,
            beam_columns={
                "sequence": "predictions_beam_",
                "log_probability": "predictions_log_probability_beam_",
                "token_log_probabilities": "predictions_token_log_probabilities_",
            },
        )

    # ------------------------------------------------------------------
    # Residue remapping configuration
    # ------------------------------------------------------------------

    def test_residue_remapping_residue_modifications(self, instanovo_loader):
        """Test that InstaNovo residue modification notations are properly mapped to UNIMOD."""
        residue_set = instanovo_loader.metrics.residue_set

        # Test oxidation (named and mass notation)
        assert residue_set.residue_remapping["M(ox)"] == "M[UNIMOD:35]"
        assert residue_set.residue_remapping["M(+15.99)"] == "M[UNIMOD:35]"

        # Test phosphorylation (abbreviated and mass notation)
        assert residue_set.residue_remapping["S(p)"] == "S[UNIMOD:21]"
        assert residue_set.residue_remapping["S(+79.97)"] == "S[UNIMOD:21]"

        # Test deamidation (with and without leading zero)
        assert residue_set.residue_remapping["N(+0.98)"] == "N[UNIMOD:7]"
        assert residue_set.residue_remapping["N(+.98)"] == "N[UNIMOD:7]"

    def test_residue_remapping_terminal_modifications(self, instanovo_loader):
        """Test that InstaNovo terminal modification notations are properly mapped to UNIMOD."""
        residue_set = instanovo_loader.metrics.residue_set
        assert residue_set.residue_remapping["(+42.01)"] == "[UNIMOD:1]"
        assert residue_set.residue_remapping["(-17.03)"] == "[UNIMOD:385]"

    # ------------------------------------------------------------------
    # _validate_beam_columns
    # ------------------------------------------------------------------

    def test_validate_beam_columns_raises_for_missing_prefix(self, loader):
        """Validation fails when a beam column prefix doesn't match any columns."""
        columns = ["spectrum_id", "predictions", "wrong_beam_0", "wrong_log_prob_0"]
        with pytest.raises(ValueError, match="Cannot find columns matching"):
            loader._validate_beam_columns(columns)

    def test_validate_beam_columns_error_lists_missing_prefixes(self, loader):
        """Error message should list all missing prefixes."""
        columns = [
            "spectrum_id",
            "predictions_beam_0",
        ]  # missing log_probability and token_log_probabilities
        with pytest.raises(ValueError, match="Cannot find columns") as exc_info:
            loader._validate_beam_columns(columns)
        error_msg = str(exc_info.value)
        assert "predictions_log_probability_beam_" in error_msg
        assert "predictions_token_log_probabilities_" in error_msg

    def test_validate_beam_columns_error_shows_available_columns(self, loader):
        """Error message should show available columns for debugging."""
        columns = ["spectrum_id", "my_custom_beam_0"]
        with pytest.raises(ValueError, match="Cannot find columns") as exc_info:
            loader._validate_beam_columns(columns)
        error_msg = str(exc_info.value)
        assert "spectrum_id" in error_msg
        assert "my_custom_beam_0" in error_msg

    def test_validate_beam_columns_success_with_valid_columns(self, loader):
        """Validation passes when all prefixes match at least one column."""
        columns = [
            "spectrum_id",
            "predictions_beam_0",
            "predictions_beam_1",
            "predictions_log_probability_beam_0",
            "predictions_log_probability_beam_1",
            "predictions_token_log_probabilities_0",
            "predictions_token_log_probabilities_1",
        ]
        # Should not raise
        loader._validate_beam_columns(columns)

    def test_validate_beam_columns_requires_exact_prefix_match(self, loader):
        """Column must start with prefix and end with digits (not just contain prefix)."""
        columns = [
            "some_predictions_beam_0",  # prefix not at start
            "predictions_beam_suffix",  # no trailing digits
            "predictions_log_probability_beam_0",
            "predictions_token_log_probabilities_0",
        ]
        with pytest.raises(ValueError, match="predictions_beam_"):
            loader._validate_beam_columns(columns)

    # ------------------------------------------------------------------
    # Custom beam columns
    # ------------------------------------------------------------------

    def test_custom_beam_columns_are_used_in_validation(
        self, residue_masses, residue_remapping, tmp_path
    ):
        """Loader should use custom beam_columns for validation."""
        custom_beam_columns = {
            "sequence": "my_seq_",
            "log_probability": "my_logprob_",
            "token_log_probabilities": "my_tokens_",
        }
        loader = InstaNovoDatasetLoader(
            residue_masses=residue_masses,
            residue_remapping=residue_remapping,
            beam_columns=custom_beam_columns,
        )
        # These columns match our custom prefixes
        columns = ["my_seq_0", "my_logprob_0", "my_tokens_0"]
        # Should not raise
        loader._validate_beam_columns(columns)

    def test_custom_beam_columns_reject_default_columns(
        self, residue_masses, residue_remapping
    ):
        """Loader with custom beam_columns should reject default column names."""
        custom_beam_columns = {
            "sequence": "my_seq_",
            "log_probability": "my_logprob_",
            "token_log_probabilities": "my_tokens_",
        }
        loader = InstaNovoDatasetLoader(
            residue_masses=residue_masses,
            residue_remapping=residue_remapping,
            beam_columns=custom_beam_columns,
        )
        # Default columns should fail with custom beam_columns
        columns = [
            "predictions_beam_0",
            "predictions_log_probability_beam_0",
            "predictions_token_log_probabilities_0",
        ]
        with pytest.raises(ValueError, match="my_seq_"):
            loader._validate_beam_columns(columns)

    def test_custom_beam_columns_load_beam_preds(
        self, residue_masses, residue_remapping, tmp_path
    ):
        """Custom beam columns should work end-to-end in _load_beam_preds."""
        custom_beam_columns = {
            "sequence": "seq_",
            "log_probability": "logp_",
            "token_log_probabilities": "tokp_",
        }
        loader = InstaNovoDatasetLoader(
            residue_masses=residue_masses,
            residue_remapping=residue_remapping,
            beam_columns=custom_beam_columns,
        )
        df = pd.DataFrame(
            {
                "spectrum_id": [0],
                "predictions": ["PEPTIDE"],
                "seq_0": ["PEPTIDE"],
                "logp_0": [-0.5],
                "tokp_0": ["[-0.1, -0.2]"],
            }
        )
        csv_path = tmp_path / "preds.csv"
        df.to_csv(csv_path, index=False)

        _preds_df, beam_df = loader._load_beam_preds(csv_path)
        assert "seq_0" in beam_df.columns
        assert "logp_0" in beam_df.columns
        assert "tokp_0" in beam_df.columns

    # ------------------------------------------------------------------
    # beam_columns=None (disable beam loading)
    # ------------------------------------------------------------------

    def test_beam_columns_none_load_predictions_without_beams(
        self, residue_masses, residue_remapping, tmp_path
    ):
        """_load_predictions_without_beams should load CSV without parsing beams."""
        df = pd.DataFrame(
            {
                "spectrum_id": [0],
                "predictions": ["PEPTIDE"],
                "log_probs": [-0.5],
            }
        )
        csv_path = tmp_path / "preds.csv"
        df.to_csv(csv_path, index=False)

        result = InstaNovoDatasetLoader._load_predictions_without_beams(csv_path)
        assert "spectrum_id" in result.columns
        assert "predictions" in result.columns
        assert len(result) == 1

    # ------------------------------------------------------------------
    # _load_beam_preds
    # ------------------------------------------------------------------

    def test_load_beam_preds_raises_for_non_csv(self, loader, tmp_path):
        path = tmp_path / "preds.parquet"
        path.touch()
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader._load_beam_preds(path)

    def test_load_beam_preds_splits_beam_columns(self, loader, tmp_path):
        """Beam-specific columns must end up in beam_df, not preds_df."""
        df = pd.DataFrame(
            {
                "spectrum_id": [0],
                "predictions": ["PEPTIDE"],
                "predictions_tokenised": ["P, E, P, T, I, D, E"],
                "log_probs": [-0.5],
                "predictions_beam_0": ["PEPTIDE"],
                "predictions_log_probability_beam_0": [-0.5],
                "predictions_token_log_probabilities_0": ["[-0.1]"],
            }
        )
        csv_path = tmp_path / "preds.csv"
        df.to_csv(csv_path, index=False)

        _preds_df, beam_df = loader._load_beam_preds(csv_path)
        assert "predictions_beam_0" in beam_df.columns
        assert "predictions_log_probability_beam_0" in beam_df.columns

    def test_load_beam_preds_preds_df_has_no_beam_columns(self, loader, tmp_path):
        df = pd.DataFrame(
            {
                "spectrum_id": [0],
                "predictions": ["PEPTIDE"],
                "predictions_tokenised": ["P, E, P, T, I, D, E"],
                "log_probs": [-0.5],
                "predictions_beam_0": ["PEPTIDE"],
                "predictions_log_probability_beam_0": [-0.5],
                "predictions_token_log_probabilities_0": ["[-0.1]"],
            }
        )
        csv_path = tmp_path / "preds.csv"
        df.to_csv(csv_path, index=False)

        preds_df, _ = loader._load_beam_preds(csv_path)
        assert "predictions_beam_0" not in preds_df.columns
        assert "predictions_log_probability_beam_0" not in preds_df.columns

    # ------------------------------------------------------------------
    # _load_spectrum_data
    # ------------------------------------------------------------------

    def test_load_spectrum_data_raises_for_unsupported_extension(
        self, instanovo_loader, tmp_path
    ):
        path = tmp_path / "data.csv"
        path.touch()
        with pytest.raises(ValueError, match="Unsupported file format"):
            instanovo_loader._load_spectrum_data(path)

    def test_load_spectrum_data_reads_parquet(self, instanovo_loader, tmp_path):
        df = pl.DataFrame({"mz_array": [[100.0, 200.0]], "charge": [2]})
        path = tmp_path / "data.parquet"
        df.write_parquet(path)

        result_df, _ = instanovo_loader._load_spectrum_data(path)
        assert "mz_array" in result_df.columns

    def test_load_spectrum_data_reads_ipc(self, instanovo_loader, tmp_path):
        df = pl.DataFrame({"mz_array": [[100.0, 200.0]], "charge": [2]})
        path = tmp_path / "data.ipc"
        df.write_ipc(path)

        result_df, _ = instanovo_loader._load_spectrum_data(path)
        assert "mz_array" in result_df.columns

    def test_load_spectrum_data_detects_labels_when_sequence_present(
        self, instanovo_loader, tmp_path
    ):
        df = pl.DataFrame({"sequence": ["PEPTIDE"], "charge": [2]})
        path = tmp_path / "data.parquet"
        df.write_parquet(path)

        _, has_labels = instanovo_loader._load_spectrum_data(path)
        assert has_labels is True

    def test_load_spectrum_data_no_labels_when_sequence_absent(
        self, instanovo_loader, tmp_path
    ):
        df = pl.DataFrame({"charge": [2], "mz_array": [[100.0]]})
        path = tmp_path / "data.parquet"
        df.write_parquet(path)

        _, has_labels = instanovo_loader._load_spectrum_data(path)
        assert has_labels is False

    def test_df_from_matchms_includes_only_present_metadata(self, tmp_path):
        """Columns match what matchms exposes; scan_number is always enumerate index."""
        from matchms.importing import load_from_mgf

        mgf_path = tmp_path / "one.mgf"
        mgf_path.write_text(
            "BEGIN IONS\n"
            "PEPMASS=500.0\n"
            "CHARGE=2+\n"
            "RTINSECONDS=100.0\n"
            "SEQ=PEPTIDE\n"
            "100.0 1.0\n"
            "END IONS\n",
            encoding="utf-8",
        )
        spectra = list(load_from_mgf(str(mgf_path)))
        df = InstaNovoDatasetLoader._df_from_matchms(spectra)
        assert df["scan_number"].to_list() == [0]
        assert "precursor_mz" in df.columns
        assert "precursor_charge" in df.columns
        assert "retention_time" in df.columns
        assert "sequence" in df.columns
        assert "mz_array" in df.columns
        assert "intensity_array" in df.columns

    def test_df_from_matchms_peaks_only_spectrum(self, tmp_path):
        from matchms.importing import load_from_mgf

        mgf_path = tmp_path / "minimal.mgf"
        mgf_path.write_text("BEGIN IONS\n110.0 0.5\nEND IONS\n", encoding="utf-8")
        spectra = list(load_from_mgf(str(mgf_path)))
        df = InstaNovoDatasetLoader._df_from_matchms(spectra)
        assert set(df.columns) == {
            "scan_number",
            "mz_array",
            "intensity_array",
        }
        assert df["scan_number"].to_list() == [0]

    def test_add_index_cols_uses_scan_number(self, tmp_path):
        df = pl.DataFrame({"scan_number": [0, 1]})
        fp = tmp_path / "experiment.mgf"
        fp.touch()
        out = InstaNovoDatasetLoader._add_index_cols(df, fp)
        assert out["experiment_name"].to_list() == ["experiment", "experiment"]
        assert out["spectrum_id"].to_list() == ["experiment:0", "experiment:1"]

    def test_add_index_cols_row_index_when_no_scan_number(self, tmp_path):
        df = pl.DataFrame({"precursor_mz": [400.0, 500.0]})
        fp = tmp_path / "run.parquet"
        fp.touch()
        out = InstaNovoDatasetLoader._add_index_cols(df, fp)
        assert out["spectrum_id"].to_list() == ["run:0", "run:1"]

    def test_load_spectrum_data_parquet_add_index_cols_when_enabled(
        self, residue_masses, residue_remapping, tmp_path
    ):
        loader = InstaNovoDatasetLoader(
            residue_masses=residue_masses,
            residue_remapping=residue_remapping,
            add_index_cols=True,
        )
        df = pl.DataFrame(
            {
                "scan_number": [0],
                "mz_array": [[100.0]],
                "intensity_array": [[1.0]],
            }
        )
        path = tmp_path / "spec.parquet"
        df.write_parquet(path)
        result, _ = loader._load_spectrum_data(path)
        assert "experiment_name" in result.columns
        assert "spectrum_id" in result.columns
        assert result["spectrum_id"][0] == "spec:0"

    def test_load_spectrum_data_parquet_no_index_cols_by_default(
        self, instanovo_loader, tmp_path
    ):
        df = pl.DataFrame({"mz_array": [[100.0]], "intensity_array": [[1.0]]})
        path = tmp_path / "data.parquet"
        df.write_parquet(path)
        result, _ = instanovo_loader._load_spectrum_data(path)
        assert "experiment_name" not in result.columns
        assert "spectrum_id" not in result.columns

    def test_load_spectrum_data_mgf_always_adds_index_cols(
        self, residue_masses, residue_remapping, tmp_path
    ):
        from matchms.importing import load_from_mgf

        loader = InstaNovoDatasetLoader(
            residue_masses=residue_masses,
            residue_remapping=residue_remapping,
            add_index_cols=False,
        )
        mgf_path = tmp_path / "spectra.mgf"
        mgf_path.write_text(
            "BEGIN IONS\n"
            "PEPMASS=451.25\n"
            "CHARGE=2+\n"
            "RTINSECONDS=824.5\n"
            "SEQ=PEPTIDE\n"
            "100.0 1.0\n"
            "END IONS\n",
            encoding="utf-8",
        )
        result, has_labels = loader._load_spectrum_data(mgf_path)
        assert "experiment_name" in result.columns
        assert "spectrum_id" in result.columns
        assert result["spectrum_id"][0] == "spectra:0"
        assert has_labels is True
        # Smoke: round-trip through matchms
        assert len(list(load_from_mgf(str(mgf_path)))) == 1

    # ------------------------------------------------------------------
    # _merge_spectrum_data
    # ------------------------------------------------------------------

    def test_merge_spectrum_data_raises_on_row_count_mismatch(self):
        """Merge conflict: spectrum_id in preds has no match in spectrum data."""
        preds = pd.DataFrame({"spectrum_id": [1, 2, 3], "confidence": [0.9, 0.8, 0.7]})
        spectra = pd.DataFrame(
            {"spectrum_id": [1, 2], "mz": [100.0, 200.0]}
        )  # spectrum_id=3 missing
        with pytest.raises(ValueError, match="Merge conflict"):
            InstaNovoDatasetLoader._merge_spectrum_data(preds, spectra)

    def test_merge_spectrum_data_succeeds_on_full_match(self):
        preds = pd.DataFrame({"spectrum_id": [1, 2], "confidence": [0.9, 0.8]})
        spectra = pd.DataFrame({"spectrum_id": [1, 2], "mz": [100.0, 200.0]})
        result = InstaNovoDatasetLoader._merge_spectrum_data(preds, spectra)
        assert len(result) == 2
        assert "mz" in result.columns
        assert "confidence" in result.columns

    def test_merge_spectrum_data_allows_extra_rows_in_spectrum_df(self):
        """spectrum data may have more rows than predictions (pre-filtered spectra)."""
        preds = pd.DataFrame({"spectrum_id": [1], "confidence": [0.9]})
        spectra = pd.DataFrame({"spectrum_id": [1, 2, 3], "mz": [100.0, 200.0, 300.0]})
        result = InstaNovoDatasetLoader._merge_spectrum_data(preds, spectra)
        assert len(result) == 1

    # ------------------------------------------------------------------
    # _process_beams
    # ------------------------------------------------------------------

    def test_process_beams_produces_scored_sequences(self, loader):
        beam_df = pl.DataFrame(
            {
                "predictions_beam_0": ["PEPTIDE"],
                "predictions_log_probability_beam_0": [-0.5],
                "predictions_token_log_probabilities_0": [
                    "[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]"
                ],
            }
        )
        beams = loader._process_beams(beam_df)
        assert len(beams) == 1
        assert beams[0] is not None
        assert len(beams[0]) == 1
        assert hasattr(beams[0][0], "sequence")
        assert hasattr(beams[0][0], "sequence_log_probability")

    def test_process_beams_replaces_l_with_i_in_sequences(self, loader):
        """L amino acid must be replaced with I before tokenization."""
        beam_df = pl.DataFrame(
            {
                "predictions_beam_0": ["PEPTLDE"],
                "predictions_log_probability_beam_0": [-0.5],
                "predictions_token_log_probabilities_0": [
                    "[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]"
                ],
            }
        )
        beams = loader._process_beams(beam_df)
        assert beams[0] is not None
        assert "L" not in beams[0][0].sequence

    def test_process_beams_returns_none_for_empty_row(self, loader):
        """A row where log_prob is -inf should produce None (no valid beam)."""
        # Use a valid string (not null) to avoid Polars str.replace_all type errors,
        # but set log_prob to -inf so the "if sequence and log_prob > -inf" guard fails.
        beam_df = pl.DataFrame(
            {
                "predictions_beam_0": ["PEPTIDE"],
                "predictions_log_probability_beam_0": [float("-inf")],
                "predictions_token_log_probabilities_0": ["[-0.1]"],
            }
        )
        beams = loader._process_beams(beam_df)
        assert beams[0] is None

    def test_process_beams_parses_string_token_log_probabilities(self, loader):
        """token_log_probabilities stored as a string should be parsed to a list."""
        token_str = "[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]"
        beam_df = pl.DataFrame(
            {
                "predictions_beam_0": ["PEPTIDE"],
                "predictions_log_probability_beam_0": [-0.5],
                "predictions_token_log_probabilities_0": [token_str],
            }
        )
        beams = loader._process_beams(beam_df)
        assert beams[0] is not None
        assert beams[0][0].token_log_probabilities == ast.literal_eval(token_str)

    def test_process_beams_handles_multiple_spectra(self, loader):
        """Each row in beam_df should produce an independent entry."""
        beam_df = pl.DataFrame(
            {
                "predictions_beam_0": ["PEPTIDE", "ACGM"],
                "predictions_log_probability_beam_0": [-0.5, -1.2],
                "predictions_token_log_probabilities_0": [
                    "[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]",
                    "[-0.1, -0.2, -0.3, -0.4]",
                ],
            }
        )
        beams = loader._process_beams(beam_df)
        assert len(beams) == 2
        assert all(b is not None for b in beams)

    # ------------------------------------------------------------------
    # column_mapping
    # ------------------------------------------------------------------

    def test_default_column_mapping(self, loader):
        """Default column_mapping matches the legacy hardcoded values."""
        assert loader.column_mapping == {
            "predictions": "predictions",
            "predictions_tokenised": "predictions_tokenised",
            "log_probability": "log_probs",
        }

    def test_custom_column_mapping_overrides_defaults(
        self, residue_masses, residue_remapping
    ):
        """Supplying a column_mapping merges with defaults."""
        loader = InstaNovoDatasetLoader(
            residue_masses=residue_masses,
            residue_remapping=residue_remapping,
            column_mapping={
                "predictions": "old_preds",
                "predictions_tokenised": "old_preds_tok",
                "log_probability": "old_log_prob",
            },
        )
        assert loader.column_mapping["predictions"] == "old_preds"
        assert loader.column_mapping["predictions_tokenised"] == "old_preds_tok"
        assert loader.column_mapping["log_probability"] == "old_log_prob"

    def test_custom_column_mapping_partial_override(
        self, residue_masses, residue_remapping
    ):
        """A partial column_mapping only overrides specified keys."""
        loader = InstaNovoDatasetLoader(
            residue_masses=residue_masses,
            residue_remapping=residue_remapping,
            column_mapping={"log_probability": "log_probability"},
        )
        assert loader.column_mapping["predictions"] == "predictions"
        assert loader.column_mapping["predictions_tokenised"] == "predictions_tokenised"
        assert loader.column_mapping["log_probability"] == "log_probability"

    def test_process_predictions_with_custom_column_mapping(
        self, residue_masses, residue_remapping
    ):
        """_process_predictions uses column_mapping to find CSV columns."""
        loader = InstaNovoDatasetLoader(
            residue_masses=residue_masses,
            residue_remapping=residue_remapping,
            column_mapping={
                "predictions": "old_preds",
                "predictions_tokenised": "old_preds_tok",
                "log_probability": "old_log_prob",
            },
        )
        preds_df = pd.DataFrame(
            {
                "spectrum_id": [1],
                "old_preds": ["PEPTIDE"],
                "old_preds_tok": ["P, E, P, T, I, D, E"],
                "old_log_prob": [-0.5],
            }
        )
        result = loader._process_predictions(preds_df, ["spectrum_id"])
        assert "prediction" in result.columns
        assert "prediction_untokenised" in result.columns
        assert "confidence" in result.columns

    def test_process_predictions_raises_with_wrong_column_mapping(
        self, residue_masses, residue_remapping
    ):
        """Missing CSV columns should raise ValueError mentioning column_mapping."""
        loader = InstaNovoDatasetLoader(
            residue_masses=residue_masses,
            residue_remapping=residue_remapping,
            column_mapping={
                "predictions": "nonexistent_col",
                "predictions_tokenised": "predictions_tokenised",
                "log_probability": "log_probs",
            },
        )
        preds_df = pd.DataFrame(
            {
                "spectrum_id": [1],
                "predictions": ["PEPTIDE"],
                "predictions_tokenised": ["P, E, P, T, I, D, E"],
                "log_probs": [-0.5],
            }
        )
        with pytest.raises(ValueError, match="column_mapping"):
            loader._process_predictions(preds_df, ["spectrum_id"])

    # ------------------------------------------------------------------
    # _process_predictions
    # ------------------------------------------------------------------

    def test_process_predictions_renames_columns(self, loader):
        preds_df = pd.DataFrame(
            {
                "spectrum_id": [1],
                "predictions": ["PEPTIDE"],
                "predictions_tokenised": ["P, E, P, T, I, D, E"],
                "log_probs": [-0.5],
            }
        )
        result = loader._process_predictions(preds_df, ["spectrum_id"])
        assert "prediction" in result.columns
        assert "prediction_untokenised" in result.columns
        assert "confidence" in result.columns
        assert "predictions_tokenised" not in result.columns
        assert "log_probs" not in result.columns

    def test_process_predictions_applies_exp_to_confidence(self, loader):
        log_prob = -0.5
        preds_df = pd.DataFrame(
            {
                "spectrum_id": [1],
                "predictions": ["PEPTIDE"],
                "predictions_tokenised": ["P, E, P, T, I, D, E"],
                "log_probs": [log_prob],
            }
        )
        result = loader._process_predictions(preds_df, ["spectrum_id"])
        assert result["confidence"].iloc[0] == pytest.approx(np.exp(log_prob))

    def test_process_predictions_splits_comma_separated_prediction(self, loader):
        preds_df = pd.DataFrame(
            {
                "spectrum_id": [1],
                "predictions": ["PEPTIDE"],
                "predictions_tokenised": ["P, E, P, T, I, D, E"],
                "log_probs": [-0.5],
            }
        )
        result = loader._process_predictions(preds_df, ["spectrum_id"])
        assert result["prediction"].iloc[0] == ["P", "E", "P", "T", "I", "D", "E"]

    def test_process_predictions_replaces_l_with_i_in_prediction_list(self, loader):
        preds_df = pd.DataFrame(
            {
                "spectrum_id": [1],
                "predictions": ["PEPTLDE"],
                "predictions_tokenised": ["P, E, P, T, L, D, E"],
                "log_probs": [-0.5],
            }
        )
        result = loader._process_predictions(preds_df, ["spectrum_id"])
        assert "L" not in result["prediction"].iloc[0]

    def test_process_predictions_replaces_l_with_i_in_prediction_untokenised(
        self, loader
    ):
        preds_df = pd.DataFrame(
            {
                "spectrum_id": [1],
                "predictions": ["PEPTLDE"],
                "predictions_tokenised": ["P, E, P, T, I, D, E"],
                "log_probs": [-0.5],
            }
        )
        result = loader._process_predictions(preds_df, ["spectrum_id"])
        assert "L" not in result["prediction_untokenised"].iloc[0]

    def test_process_predictions_drops_duplicate_input_columns(self, loader):
        """Columns present in input spectrum data (except spectrum_id) must be dropped."""
        preds_df = pd.DataFrame(
            {
                "spectrum_id": [1],
                "predictions": ["PEPTIDE"],
                "predictions_tokenised": ["P, E, P, T, I, D, E"],
                "log_probs": [-0.5],
                "precursor_charge": [2],
                "precursor_mz": [400.504],
            }
        )
        input_cols = ["spectrum_id", "precursor_charge", "precursor_mz"]
        result = loader._process_predictions(preds_df, input_cols)
        assert "spectrum_id" in result.columns  # spectrum_id is always kept
        assert "precursor_charge" not in result.columns
        assert "precursor_mz" not in result.columns

    # ------------------------------------------------------------------
    # _evaluate_predictions
    # ------------------------------------------------------------------

    def test_evaluate_valid_prediction_true_for_list(self, loader):
        dataset = pd.DataFrame({"prediction": [["A", "G"]]})
        result = loader._evaluate_predictions(dataset, has_labels=False)
        assert result["valid_prediction"].iloc[0]

    def test_evaluate_valid_prediction_false_for_non_list(self, loader):
        dataset = pd.DataFrame({"prediction": ["bad_string"]})
        result = loader._evaluate_predictions(dataset, has_labels=False)
        assert not result["valid_prediction"].iloc[0]

    def test_evaluate_no_label_columns_when_no_labels(self, loader):
        dataset = pd.DataFrame({"prediction": [["A", "G"]]})
        result = loader._evaluate_predictions(dataset, has_labels=False)
        assert "correct" not in result.columns
        assert "valid_sequence" not in result.columns
        assert "num_matches" not in result.columns

    def test_evaluate_correct_flag_true_on_full_match(self, loader):
        seq = ["P", "E", "P"]
        dataset = pd.DataFrame({"sequence": [seq], "prediction": [seq]})
        result = loader._evaluate_predictions(dataset, has_labels=True)
        assert result["correct"].iloc[0]

    def test_evaluate_correct_flag_false_on_different_sequence(self, loader):
        dataset = pd.DataFrame(
            {"sequence": [["P", "E", "P"]], "prediction": [["A", "E", "P"]]}
        )
        result = loader._evaluate_predictions(dataset, has_labels=True)
        assert not result["correct"].iloc[0]

    def test_evaluate_correct_flag_false_on_length_mismatch(self, loader):
        dataset = pd.DataFrame(
            {"sequence": [["P", "E", "P"]], "prediction": [["P", "E"]]}
        )
        result = loader._evaluate_predictions(dataset, has_labels=True)
        assert not result["correct"].iloc[0]

    def test_evaluate_num_matches_full_match(self, loader):
        seq = ["P", "E", "P"]
        dataset = pd.DataFrame({"sequence": [seq], "prediction": [seq]})
        result = loader._evaluate_predictions(dataset, has_labels=True)
        assert result["num_matches"].iloc[0] == len(seq)

    def test_evaluate_num_matches_zero_on_invalid_prediction(self, loader):
        """Non-list prediction should give 0 matches."""
        dataset = pd.DataFrame({"sequence": [["P", "E", "P"]], "prediction": [None]})
        result = loader._evaluate_predictions(dataset, has_labels=True)
        assert result["num_matches"].iloc[0] == 0

    # ------------------------------------------------------------------
    # load() – error handling
    # ------------------------------------------------------------------

    def test_load_raises_when_predictions_path_is_none(self, loader, tmp_path):
        with pytest.raises(ValueError, match="predictions_path is required"):
            loader.load(data_path=tmp_path)
