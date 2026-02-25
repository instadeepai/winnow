"""Unit tests for winnow data loaders."""

import ast
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.datasets.data_loaders import (
    InstaNovoDatasetLoader,
    MZTabDatasetLoader,
    PointNovoDatasetLoader,
    WinnowDatasetLoader,
)

# ---------------------------------------------------------------------------
# Shared module-level residue helpers
# ---------------------------------------------------------------------------

_FULL_RESIDUE_MASSES = {
    "G": 57.021464,
    "A": 71.037114,
    "S": 87.032028,
    "P": 97.052764,
    "V": 99.068414,
    "T": 101.047670,
    "C": 103.009185,
    "I": 113.084064,
    "N": 114.042927,
    "D": 115.026943,
    "Q": 128.058578,
    "K": 128.094963,
    "E": 129.042593,
    "M": 131.040485,
    "H": 137.058912,
    "F": 147.068414,
    "R": 156.101111,
    "Y": 163.063329,
    "W": 186.079313,
    # Common modifications
    "M[UNIMOD:35]": 147.035400,
    "C[UNIMOD:4]": 160.030649,
    "N[UNIMOD:7]": 115.026943,
    "Q[UNIMOD:7]": 129.042594,
    "[UNIMOD:1]": 42.010565,
    "[UNIMOD:5]": 43.005814,
    "[UNIMOD:385]": -17.026549,
}

_STANDARD_REMAPPING = {
    "M[Oxidation]": "M[UNIMOD:35]",
    "C[Carbamidomethyl]": "C[UNIMOD:4]",
    "[Acetyl]": "[UNIMOD:1]",
    "[Carbamyl]": "[UNIMOD:5]",
    "[Ammonia-loss]": "[UNIMOD:385]",
}

# ---------------------------------------------------------------------------
# InstaNovoDatasetLoader
# ---------------------------------------------------------------------------


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
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def test_initialization(self, instanovo_loader):
        """Test InstaNovoDatasetLoader initialization."""
        assert instanovo_loader.metrics is not None
        assert instanovo_loader.metrics.residue_set is not None

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

    def test_load_spectrum_data_raises_for_unsupported_extension(self, tmp_path):
        path = tmp_path / "data.csv"
        path.touch()
        with pytest.raises(ValueError, match="Unsupported file format"):
            InstaNovoDatasetLoader._load_spectrum_data(path)

    def test_load_spectrum_data_reads_parquet(self, tmp_path):
        df = pl.DataFrame({"mz_array": [[100.0, 200.0]], "charge": [2]})
        path = tmp_path / "data.parquet"
        df.write_parquet(path)

        result_df, _ = InstaNovoDatasetLoader._load_spectrum_data(path)
        assert "mz_array" in result_df.columns

    def test_load_spectrum_data_reads_ipc(self, tmp_path):
        df = pl.DataFrame({"mz_array": [[100.0, 200.0]], "charge": [2]})
        path = tmp_path / "data.ipc"
        df.write_ipc(path)

        result_df, _ = InstaNovoDatasetLoader._load_spectrum_data(path)
        assert "mz_array" in result_df.columns

    def test_load_spectrum_data_detects_labels_when_sequence_present(self, tmp_path):
        df = pl.DataFrame({"sequence": ["PEPTIDE"], "charge": [2]})
        path = tmp_path / "data.parquet"
        df.write_parquet(path)

        _, has_labels = InstaNovoDatasetLoader._load_spectrum_data(path)
        assert has_labels is True

    def test_load_spectrum_data_no_labels_when_sequence_absent(self, tmp_path):
        df = pl.DataFrame({"charge": [2], "mz_array": [[100.0]]})
        path = tmp_path / "data.parquet"
        df.write_parquet(path)

        _, has_labels = InstaNovoDatasetLoader._load_spectrum_data(path)
        assert has_labels is False

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
        result = loader._process_predictions(preds_df, [])
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
        result = loader._process_predictions(preds_df, [])
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
        result = loader._process_predictions(preds_df, [])
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
        result = loader._process_predictions(preds_df, [])
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
        result = loader._process_predictions(preds_df, [])
        assert "L" not in result["prediction_untokenised"].iloc[0]

    def test_process_predictions_drops_duplicate_input_columns(self, loader):
        """Columns present in input spectrum data (except spectrum_id) must be dropped."""
        preds_df = pd.DataFrame(
            {
                "spectrum_id": [1],
                "predictions": ["PEPTIDE"],
                "predictions_tokenised": ["P, E, P, T, I, D, E"],
                "log_probs": [-0.5],
                "charge": [2],
                "precursor_mass": [800.0],
            }
        )
        input_cols = ["spectrum_id", "charge", "precursor_mass"]
        result = loader._process_predictions(preds_df, input_cols)
        assert "spectrum_id" in result.columns  # spectrum_id is always kept
        assert "charge" not in result.columns
        assert "precursor_mass" not in result.columns

    # ------------------------------------------------------------------
    # _evaluate_predictions
    # ------------------------------------------------------------------

    def test_evaluate_adds_valid_prediction_column(self, loader):
        dataset = pd.DataFrame({"prediction": [["A", "G"], "bad_string", None]})
        result = loader._evaluate_predictions(dataset, has_labels=False)
        assert "valid_prediction" in result.columns

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
        assert "valid_peptide" not in result.columns
        assert "num_matches" not in result.columns

    def test_evaluate_adds_valid_peptide_when_has_labels(self, loader):
        dataset = pd.DataFrame({"sequence": [["A", "G"]], "prediction": [["A", "G"]]})
        result = loader._evaluate_predictions(dataset, has_labels=True)
        assert "valid_peptide" in result.columns
        assert result["valid_peptide"].iloc[0]

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


# ---------------------------------------------------------------------------
# MZTabDatasetLoader
# ---------------------------------------------------------------------------


class TestMZTabDatasetLoader:
    """Tests for MZTabDatasetLoader: token remapping, integration, and internal methods."""

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
            # Modifications
            "M[UNIMOD:35]": 147.035400,
            "C[UNIMOD:4]": 160.030649,
            "N[UNIMOD:7]": 115.026943,
            "Q[UNIMOD:7]": 129.042594,
            # Terminal modifications
            "[UNIMOD:1]": 42.010565,
            "[UNIMOD:5]": 43.005814,
            "[UNIMOD:385]": -17.026549,
        }

    @pytest.fixture()
    def residue_remapping(self):
        """Residue remapping for Casanovo-specific notations to UNIMOD tokens.

        Note: N-terminal modifications don't need hyphens because remapping
        happens at the token level after tokenization (which strips hyphens).
        """
        return {
            "M+15.995": "M[UNIMOD:35]",  # Oxidation
            "Q+0.984": "Q[UNIMOD:7]",  # Deamidation
            "N+0.984": "N[UNIMOD:7]",  # Deamidation
            "+42.011": "[UNIMOD:1]",  # Acetylation
            "+43.006": "[UNIMOD:5]",  # Carbamylation
            "-17.027": "[UNIMOD:385]",  # Ammonia loss
            "C+57.021": "C[UNIMOD:4]",  # Carbamidomethylation
            "C[Carbamidomethyl]": "C[UNIMOD:4]",  # Carbamidomethylation
            "M[Oxidation]": "M[UNIMOD:35]",  # Oxidation
            "N[Deamidated]": "N[UNIMOD:7]",  # Deamidation
            "Q[Deamidated]": "Q[UNIMOD:7]",  # Deamidation
            # N-terminal modifications (no hyphens - handled at token level)
            "[Acetyl]": "[UNIMOD:1]",  # Acetylation
            "[Carbamyl]": "[UNIMOD:5]",  # Carbamylation
            "[Ammonia-loss]": "[UNIMOD:385]",  # Ammonia loss
        }

    @pytest.fixture()
    def mztab_loader(self, residue_masses, residue_remapping):
        """MZTabDatasetLoader with Casanovo-style remapping for notation tests."""
        return MZTabDatasetLoader(
            residue_masses=residue_masses,
            residue_remapping=residue_remapping,
        )

    @pytest.fixture()
    def loader(self):
        """MZTabDatasetLoader with full residues for internal-method tests."""
        return MZTabDatasetLoader(
            residue_masses=_FULL_RESIDUE_MASSES,
            residue_remapping=_STANDARD_REMAPPING,
        )

    @pytest.fixture()
    def invalid_prosit_residues_unimod_only(self):
        """List of invalid residues using only UNIMOD notation."""
        return [
            "N[UNIMOD:7]",  # Deamidated asparagine
            "Q[UNIMOD:7]",  # Deamidated glutamine
            "[UNIMOD:1]",  # N-terminal acetylation
            "[UNIMOD:5]",  # N-terminal carbamylation
            "[UNIMOD:385]",  # N-terminal ammonia loss
        ]

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def test_initialization(self, mztab_loader):
        """Test MZTabDatasetLoader initialization."""
        assert mztab_loader.metrics is not None
        assert mztab_loader.metrics.residue_set is not None

    # ------------------------------------------------------------------
    # _remap_tokens – notation remapping
    # ------------------------------------------------------------------

    def test_remap_tokens_residue_modifications(self, mztab_loader):
        """Test token-level remapping of residue modifications."""
        # Test mass-based notation
        tokens = ["M+15.995", "P", "E", "P", "T", "I", "D", "E"]
        expected = ["M[UNIMOD:35]", "P", "E", "P", "T", "I", "D", "E"]
        assert mztab_loader._remap_tokens(tokens) == expected

        # Test named notation
        tokens = ["M[Oxidation]", "P", "E", "P", "T", "I", "D", "E"]
        expected = ["M[UNIMOD:35]", "P", "E", "P", "T", "I", "D", "E"]
        assert mztab_loader._remap_tokens(tokens) == expected

    def test_remap_tokens_terminal_modifications(self, mztab_loader):
        """Test token-level remapping of N-terminal modifications.

        Note: Hyphens are stripped by the tokenizer, so we only need to
        match the modification token without the hyphen.
        """
        # Test mass-based notation (tokenizer captures "+42.011" as standalone token)
        tokens = ["+42.011", "P", "E", "P", "T", "I", "D", "E"]
        expected = ["[UNIMOD:1]", "P", "E", "P", "T", "I", "D", "E"]
        assert mztab_loader._remap_tokens(tokens) == expected

        # Test named notation (tokenizer captures "[Acetyl]" without the hyphen)
        tokens = ["[Acetyl]", "P", "E", "P", "T", "I", "D", "E"]
        expected = ["[UNIMOD:1]", "P", "E", "P", "T", "I", "D", "E"]
        assert mztab_loader._remap_tokens(tokens) == expected

    def test_remap_tokens_no_modifications(self, mztab_loader):
        """Test that unmodified tokens pass through unchanged."""
        tokens = ["P", "E", "P", "T", "I", "D", "E"]
        assert mztab_loader._remap_tokens(tokens) == tokens

    def test_remap_tokens_already_unimod(self, mztab_loader):
        """Test that tokens already in Proforma format are unchanged."""
        tokens = ["M[UNIMOD:35]", "P", "E", "P", "T", "I", "D", "E"]
        assert mztab_loader._remap_tokens(tokens) == tokens

    # ------------------------------------------------------------------
    # Token remapping integration – Casanovo ↔ invalid residue detection
    # ------------------------------------------------------------------

    def test_casanovo_tokens_map_to_invalid_unimod(
        self, residue_masses, invalid_prosit_residues_unimod_only
    ):
        """Test that Casanovo tokens are detected as invalid after tokenization and remapping."""
        residue_remapping = {
            "Q+0.984": "Q[UNIMOD:7]",
            "N+0.984": "N[UNIMOD:7]",
            "+42.011": "[UNIMOD:1]",
            "+43.006": "[UNIMOD:5]",
            "-17.027": "[UNIMOD:385]",
            # No hyphens needed - tokenizer strips them
            "[Acetyl]": "[UNIMOD:1]",
            "[Carbamyl]": "[UNIMOD:5]",
            "[Ammonia-loss]": "[UNIMOD:385]",
            "N[Deamidated]": "N[UNIMOD:7]",
            "Q[Deamidated]": "Q[UNIMOD:7]",
        }

        loader = MZTabDatasetLoader(
            residue_masses=residue_masses,
            residue_remapping=residue_remapping,
        )

        # Test sequences with Casanovo notation that should be remapped
        casanovo_sequences = [
            ("PEPTIDEQ+0.984", "Q[UNIMOD:7]"),  # Deamidation
            ("N+0.984PEPTIDE", "N[UNIMOD:7]"),  # Deamidation
            ("+42.011PEPTIDE", "[UNIMOD:1]"),  # Acetylation
            ("+43.006PEPTIDE", "[UNIMOD:5]"),  # Carbamylation
            ("-17.027PEPTIDE", "[UNIMOD:385]"),  # Ammonia loss
            ("[Acetyl]-PEPTIDE", "[UNIMOD:1]"),  # Acetylation (named, with hyphen)
            ("[Carbamyl]-PEPTIDE", "[UNIMOD:5]"),  # Carbamylation (named, with hyphen)
            (
                "[Ammonia-loss]-PEPTIDE",
                "[UNIMOD:385]",
            ),  # Ammonia loss (named, with hyphen)
            ("PEPTIN[Deamidated]E", "N[UNIMOD:7]"),  # Deamidation (named)
            ("Q[Deamidated]PEPTIDE", "Q[UNIMOD:7]"),  # Deamidation (named)
        ]

        for casanovo_seq, expected_token in casanovo_sequences:
            # Tokenize first (regex strips hyphens from N-terminal mods)
            tokens = loader.metrics._split_peptide(casanovo_seq)

            # Then remap tokens
            remapped_tokens = loader._remap_tokens(tokens)

            # Verify that the remapped tokens contain the expected invalid token
            assert expected_token in remapped_tokens, (
                f"Expected token {expected_token} not found in {remapped_tokens} "
                f"(from {casanovo_seq} -> tokens {tokens})"
            )

            # Verify that the token is in the invalid list
            assert (
                expected_token in invalid_prosit_residues_unimod_only
            ), f"Residue {expected_token} should be in invalid_prosit_residues list"

    def test_valid_sequences_not_affected(
        self, residue_masses, invalid_prosit_residues_unimod_only
    ):
        """Test that valid sequences without invalid modifications pass through."""
        residue_remapping = {
            "M+15.995": "M[UNIMOD:35]",  # Oxidation (valid)
            "C+57.021": "C[UNIMOD:4]",  # Carbamidomethylation (valid)
        }

        loader = MZTabDatasetLoader(
            residue_masses=residue_masses,
            residue_remapping=residue_remapping,
        )

        valid_sequences = [
            "PEPTIDE",  # No modifications
            "M+15.995PEPTIDE",  # Oxidation (not in invalid list)
            "PEPTIDEC+57.021",  # Carbamidomethylation (not in invalid list)
        ]

        for seq in valid_sequences:
            # Tokenize first, then remap (new flow)
            tokens = loader.metrics._split_peptide(seq)
            remapped_tokens = loader._remap_tokens(tokens)
            contains_invalid = any(
                token in invalid_prosit_residues_unimod_only
                for token in remapped_tokens
            )
            assert (
                not contains_invalid
            ), f"Valid sequence {remapped_tokens} should not contain invalid tokens"

    def test_only_unimod_notation_needed_in_invalid_list(
        self, residue_masses, invalid_prosit_residues_unimod_only
    ):
        """Test that we only need UNIMOD notation in invalid_prosit_residues list.

        This test demonstrates that after tokenization and remapping, we only need
        the actual tokenized forms (e.g., "Q[UNIMOD:7]", "[UNIMOD:1]") in the
        invalid_prosit_residues list, not the Casanovo-specific notations.
        """
        residue_remapping = {
            "Q+0.984": "Q[UNIMOD:7]",
            "+42.011": "[UNIMOD:1]",
            # No hyphen needed - tokenizer strips it
            "[Carbamyl]": "[UNIMOD:5]",
        }

        loader = MZTabDatasetLoader(
            residue_masses=residue_masses,
            residue_remapping=residue_remapping,
        )

        # These Casanovo sequences should be caught by UNIMOD-only invalid list
        test_cases = [
            (
                "PEPTIDEQ+0.984",
                True,
                "Q[UNIMOD:7]",
            ),  # Should be invalid after remapping
            ("+42.011PEPTIDE", True, "[UNIMOD:1]"),  # Should be invalid after remapping
            (
                "[Carbamyl]-PEPTIDE",
                True,
                "[UNIMOD:5]",
            ),  # Should be invalid after remapping (hyphen stripped by tokenizer)
            ("PEPTIDE", False, None),  # Should be valid
            ("M[UNIMOD:35]PEPTIDE", False, None),  # Valid modification
        ]

        for seq, should_be_invalid, expected_token in test_cases:
            # Tokenize first, then remap (new flow)
            tokens = loader.metrics._split_peptide(seq)
            remapped_tokens = loader._remap_tokens(tokens)
            contains_invalid = any(
                token in invalid_prosit_residues_unimod_only
                for token in remapped_tokens
            )

            if should_be_invalid:
                assert (
                    contains_invalid
                ), f"Sequence {seq} -> {remapped_tokens} should be caught as invalid"
                assert (
                    expected_token in remapped_tokens
                ), f"Expected invalid token {expected_token} in {remapped_tokens}"
            else:
                assert (
                    not contains_invalid
                ), f"Sequence {seq} -> {remapped_tokens} should be valid"

    # ------------------------------------------------------------------
    # _load_spectrum_data
    # ------------------------------------------------------------------

    def test_load_spectrum_data_raises_for_unsupported_extension(self, tmp_path):
        path = tmp_path / "data.tsv"
        path.touch()
        with pytest.raises(ValueError, match="Unsupported file format"):
            MZTabDatasetLoader._load_spectrum_data(path)

    def test_load_spectrum_data_reads_parquet(self, tmp_path):
        df = pl.DataFrame({"charge": [2], "mz_array": [[100.0]]})
        path = tmp_path / "data.parquet"
        df.write_parquet(path)

        result_df, _ = MZTabDatasetLoader._load_spectrum_data(path)
        assert "charge" in result_df.columns

    def test_load_spectrum_data_reads_ipc(self, tmp_path):
        df = pl.DataFrame({"charge": [2], "mz_array": [[100.0]]})
        path = tmp_path / "data.ipc"
        df.write_ipc(path)

        result_df, _ = MZTabDatasetLoader._load_spectrum_data(path)
        assert "charge" in result_df.columns

    def test_load_spectrum_data_detects_labels_when_sequence_present(self, tmp_path):
        df = pl.DataFrame({"sequence": ["PEPTIDE"], "charge": [2]})
        path = tmp_path / "data.parquet"
        df.write_parquet(path)

        _, has_labels = MZTabDatasetLoader._load_spectrum_data(path)
        assert has_labels is True

    def test_load_spectrum_data_no_labels_when_sequence_absent(self, tmp_path):
        df = pl.DataFrame({"charge": [2]})
        path = tmp_path / "data.parquet"
        df.write_parquet(path)

        _, has_labels = MZTabDatasetLoader._load_spectrum_data(path)
        assert has_labels is False

    # ------------------------------------------------------------------
    # _load_dataset
    # ------------------------------------------------------------------

    def test_load_dataset_raises_for_non_mztab_extension(self, tmp_path):
        path = tmp_path / "preds.csv"
        path.touch()
        with pytest.raises(ValueError, match="Unsupported file format"):
            MZTabDatasetLoader._load_dataset(path)

    # ------------------------------------------------------------------
    # _process_predictions
    # ------------------------------------------------------------------

    @pytest.fixture()
    def minimal_predictions_df(self):
        """Minimal predictions DataFrame without aa_scores."""
        return pl.DataFrame(
            {
                "spectra_ref": ["ms_run[1]:index=42", "ms_run[1]:index=7"],
                "sequence": ["PEPTIDE", "ACGM"],
                "search_engine_score[1]": [0.9, 0.7],
            }
        )

    @pytest.fixture()
    def predictions_df_with_aa_scores(self):
        """Predictions DataFrame including Casanovo-style aa_scores."""
        return pl.DataFrame(
            {
                "spectra_ref": ["ms_run[1]:index=0"],
                "sequence": ["PEP"],
                "search_engine_score[1]": [0.95],
                "opt_ms_run[1]_aa_scores": ["-0.1,-0.2,-0.3"],
            }
        )

    def test_process_predictions_extracts_index_from_spectra_ref(
        self, loader, minimal_predictions_df
    ):
        result = loader._process_predictions(minimal_predictions_df)
        assert result["index"].to_list() == [7, 42]  # sorted by index ASC

    def test_process_predictions_replaces_l_with_i(self, loader):
        df = pl.DataFrame(
            {
                "spectra_ref": ["ms_run[1]:index=0"],
                "sequence": ["PEPTLDE"],
                "search_engine_score[1]": [0.9],
            }
        )
        result = loader._process_predictions(df)
        assert "L" not in result["prediction_untokenised"][0]

    def test_process_predictions_renames_confidence_column(
        self, loader, minimal_predictions_df
    ):
        result = loader._process_predictions(minimal_predictions_df)
        assert "confidence" in result.columns
        assert "search_engine_score[1]" not in result.columns

    def test_process_predictions_without_aa_scores_creates_null_token_scores(
        self, loader, minimal_predictions_df
    ):
        """Traditional search engines lack aa_scores; token_scores should be null."""
        result = loader._process_predictions(minimal_predictions_df)
        assert "token_scores" in result.columns
        assert result["token_scores"][0] is None

    def test_process_predictions_parses_aa_scores_as_list_of_floats(
        self, loader, predictions_df_with_aa_scores
    ):
        result = loader._process_predictions(predictions_df_with_aa_scores)
        token_scores = result["token_scores"][0].to_list()
        assert token_scores == pytest.approx([-0.1, -0.2, -0.3])

    def test_process_predictions_sorted_by_index_asc_confidence_desc(self, loader):
        df = pl.DataFrame(
            {
                "spectra_ref": [
                    "ms_run[1]:index=1",
                    "ms_run[1]:index=0",
                    "ms_run[1]:index=1",
                ],
                "sequence": ["AA", "GG", "MM"],
                "search_engine_score[1]": [0.6, 0.9, 0.8],
            }
        )
        result = loader._process_predictions(df)
        indices = result["index"].to_list()
        assert indices == sorted(indices)  # index ascending
        # For index=1, confidence 0.8 should come before 0.6
        idx1_rows = result.filter(pl.col("index") == 1)
        assert idx1_rows["confidence"][0] > idx1_rows["confidence"][1]

    # ------------------------------------------------------------------
    # _tokenize
    # ------------------------------------------------------------------

    def test_tokenize_splits_simple_sequence(self, loader):
        df = pl.DataFrame({"seq": ["PEPTIDE"]})
        result = loader._tokenize(df, "seq", "tokens")
        tokens = result["tokens"][0].to_list()
        assert tokens == ["P", "E", "P", "T", "I", "D", "E"]

    def test_tokenize_strips_hyphen_from_nterm_modification(self, loader):
        """[Acetyl]-PEPTIDE: hyphen should be stripped by tokenizer, then remapped."""
        df = pl.DataFrame({"seq": ["[Acetyl]-PEPTIDE"]})
        result = loader._tokenize(df, "seq", "tokens")
        tokens = result["tokens"][0].to_list()
        assert "[UNIMOD:1]" in tokens
        assert "-" not in tokens

    def test_tokenize_remaps_modification_tokens(self, loader):
        """M[Oxidation] should be remapped to M[UNIMOD:35] after tokenization."""
        df = pl.DataFrame({"seq": ["M[Oxidation]PEPTIDE"]})
        result = loader._tokenize(df, "seq", "tokens")
        tokens = result["tokens"][0].to_list()
        assert "M[UNIMOD:35]" in tokens
        assert "M[Oxidation]" not in tokens

    def test_tokenize_unmodified_sequence_passes_through(self, loader):
        df = pl.DataFrame({"seq": ["ACGM"]})
        result = loader._tokenize(df, "seq", "tokens")
        tokens = result["tokens"][0].to_list()
        assert tokens == ["A", "C", "G", "M"]

    # ------------------------------------------------------------------
    # _get_top_predictions
    # ------------------------------------------------------------------

    def test_get_top_predictions_returns_one_row_per_index(self, loader):
        # Pre-sorted: index ASC, confidence DESC within group
        df = pl.DataFrame(
            {
                "index": [0, 0, 1, 1],
                "confidence": [0.9, 0.5, 0.7, 0.3],
                "prediction_untokenised": ["A", "B", "C", "D"],
            }
        )
        result = loader._get_top_predictions(df)
        assert len(result) == 2

    def test_get_top_predictions_keeps_highest_confidence(self, loader):
        df = pl.DataFrame(
            {
                "index": [0, 0, 1],
                "confidence": [0.9, 0.5, 0.7],
                "prediction_untokenised": ["A", "B", "C"],
            }
        )
        result = loader._get_top_predictions(df)
        row_idx0 = result.filter(pl.col("index") == 0)
        assert row_idx0["confidence"][0] == pytest.approx(0.9)

    def test_get_top_predictions_handles_single_prediction_per_index(self, loader):
        df = pl.DataFrame(
            {
                "index": [0, 1],
                "confidence": [0.9, 0.7],
                "prediction_untokenised": ["A", "B"],
            }
        )
        result = loader._get_top_predictions(df)
        assert len(result) == 2

    # ------------------------------------------------------------------
    # _create_beam_predictions
    # ------------------------------------------------------------------

    @pytest.fixture()
    def beam_predictions_df(self):
        """DataFrame in the format expected by _create_beam_predictions."""
        return pl.DataFrame(
            {
                "index": pl.Series([0, 1], dtype=pl.Int64),
                "confidence": [0.9, 0.7],
                "prediction": [["P", "E", "P"], ["A", "C"]],
                "token_scores": pl.Series(
                    [[-0.1, -0.2, -0.3], None], dtype=pl.List(pl.Float64)
                ),
            }
        )

    def test_create_beam_predictions_returns_scored_sequences(
        self, loader, beam_predictions_df
    ):
        result = loader._create_beam_predictions(beam_predictions_df, [0, 1])
        assert len(result) == 2
        assert result[0] is not None
        assert hasattr(result[0][0], "sequence")
        assert hasattr(result[0][0], "sequence_log_probability")

    def test_create_beam_predictions_returns_none_for_missing_index(
        self, loader, beam_predictions_df
    ):
        """A requested index with no predictions in the df should produce None."""
        result = loader._create_beam_predictions(beam_predictions_df, [0, 99])
        assert result[1] is None

    def test_create_beam_predictions_preserves_order_of_valid_spectra_indices(
        self, loader, beam_predictions_df
    ):
        """Output list order must follow valid_spectra_indices, not df row order."""
        result = loader._create_beam_predictions(beam_predictions_df, [1, 0])
        # First output → index 1 (sequence ["A","C"])
        # Second output → index 0 (sequence ["P","E","P"])
        assert result[0][0].sequence == ["A", "C"]
        assert result[1][0].sequence == ["P", "E", "P"]

    def test_create_beam_predictions_multiple_preds_sorted_by_confidence_desc(
        self, loader
    ):
        """When a spectrum has multiple predictions, highest confidence comes first."""
        df = pl.DataFrame(
            {
                "index": pl.Series([0, 0], dtype=pl.Int64),
                "confidence": [0.5, 0.9],  # low confidence listed first in df
                "prediction": [["A"], ["P"]],
                "token_scores": pl.Series([None, None], dtype=pl.List(pl.Float64)),
            }
        )
        result = loader._create_beam_predictions(df, [0])
        assert result[0][0].sequence_log_probability == pytest.approx(0.9)

    # ------------------------------------------------------------------
    # load() – error handling
    # ------------------------------------------------------------------

    def test_load_raises_when_predictions_path_is_none(self, loader, tmp_path):
        with pytest.raises(ValueError, match="predictions_path is required"):
            loader.load(data_path=tmp_path)


# ---------------------------------------------------------------------------
# PointNovoDatasetLoader
# ---------------------------------------------------------------------------


class TestPointNovoDatasetLoader:
    """Tests for PointNovoDatasetLoader (not yet implemented)."""

    @pytest.fixture()
    def loader(self):
        return PointNovoDatasetLoader(residue_masses=_FULL_RESIDUE_MASSES)

    def test_load_raises_not_implemented(self, loader, tmp_path):
        """load() must raise NotImplementedError unconditionally."""
        with pytest.raises(NotImplementedError):
            loader.load(data_path=tmp_path)

    def test_load_raises_not_implemented_with_predictions_path(self, loader, tmp_path):
        """NotImplementedError is raised regardless of predictions_path."""
        with pytest.raises(NotImplementedError):
            loader.load(data_path=tmp_path, predictions_path=tmp_path / "preds.csv")


# ---------------------------------------------------------------------------
# WinnowDatasetLoader
# ---------------------------------------------------------------------------


class TestWinnowDatasetLoader:
    """Tests for WinnowDatasetLoader."""

    @pytest.fixture()
    def loader(self):
        return WinnowDatasetLoader(
            residue_masses=_FULL_RESIDUE_MASSES,
            residue_remapping={},
        )

    @pytest.fixture()
    def metadata_dir(self, tmp_path):
        """Minimal metadata.csv directory (no sequence column, no pkl)."""
        df = pd.DataFrame(
            {
                "prediction": ["AG", "MG"],
                "confidence": [0.9, 0.8],
                "mz_array": ["[100.0, 200.0, 300.0]", "[150.0, 250.0, 350.0]"],
                "intensity_array": [
                    "[1000.0, 2000.0, 3000.0]",
                    "[1500.0, 2500.0, 3500.0]",
                ],
            }
        )
        df.to_csv(tmp_path / "metadata.csv", index=False)
        return tmp_path

    @pytest.fixture()
    def metadata_dir_with_sequence(self, tmp_path):
        """metadata.csv directory that includes a ground-truth sequence column."""
        df = pd.DataFrame(
            {
                "sequence": ["AG", "MG"],
                "prediction": ["AG", "MG"],
                "confidence": [0.9, 0.8],
                "mz_array": ["[100.0, 200.0, 300.0]", "[150.0, 250.0, 350.0]"],
                "intensity_array": [
                    "[1000.0, 2000.0, 3000.0]",
                    "[1500.0, 2500.0, 3500.0]",
                ],
            }
        )
        df.to_csv(tmp_path / "metadata.csv", index=False)
        return tmp_path

    @pytest.fixture()
    def metadata_dir_numpy_arrays(self, tmp_path):
        """metadata.csv with mz_array / intensity_array in numpy print format (no commas)."""
        df = pd.DataFrame(
            {
                "prediction": ["AG"],
                "confidence": [0.9],
                "mz_array": ["[100.   200.   300. ]"],
                "intensity_array": ["[1000.   2000.   3000. ]"],
            }
        )
        df.to_csv(tmp_path / "metadata.csv", index=False)
        return tmp_path

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_raises_when_predictions_path_provided(self, loader, tmp_path):
        """WinnowDatasetLoader does not accept predictions_path."""
        with pytest.raises(ValueError, match="predictions_path is not used"):
            loader.load(data_path=tmp_path, predictions_path=Path("something.pkl"))

    def test_raises_when_metadata_csv_missing(self, loader, tmp_path):
        """FileNotFoundError when the directory has no metadata.csv."""
        with pytest.raises(FileNotFoundError, match="metadata.csv"):
            loader.load(data_path=tmp_path)

    # ------------------------------------------------------------------
    # Successful load – no predictions.pkl
    # ------------------------------------------------------------------

    def test_returns_calibration_dataset(self, loader, metadata_dir):
        dataset = loader.load(data_path=metadata_dir)
        assert isinstance(dataset, CalibrationDataset)

    def test_predictions_are_none_without_pkl(self, loader, metadata_dir):
        dataset = loader.load(data_path=metadata_dir)
        assert dataset.predictions is None

    def test_prediction_column_is_tokenised(self, loader, metadata_dir):
        """prediction column should be converted from ProForma string to token list."""
        dataset = loader.load(data_path=metadata_dir)
        assert isinstance(dataset.metadata["prediction"].iloc[0], list)
        assert dataset.metadata["prediction"].iloc[0] == ["A", "G"]

    def test_sequence_column_is_tokenised_when_present(
        self, loader, metadata_dir_with_sequence
    ):
        """sequence column should be tokenised when it exists in the CSV."""
        dataset = loader.load(data_path=metadata_dir_with_sequence)
        assert isinstance(dataset.metadata["sequence"].iloc[0], list)
        assert dataset.metadata["sequence"].iloc[0] == ["A", "G"]

    def test_mz_array_parsed_comma_format(self, loader, metadata_dir):
        """mz_array written with commas should be parsed to a Python list."""
        dataset = loader.load(data_path=metadata_dir)
        mz = dataset.metadata["mz_array"].iloc[0]
        assert isinstance(mz, list)
        assert mz == pytest.approx([100.0, 200.0, 300.0])

    def test_intensity_array_parsed_comma_format(self, loader, metadata_dir):
        """intensity_array written with commas should be parsed to a Python list."""
        dataset = loader.load(data_path=metadata_dir)
        intensity = dataset.metadata["intensity_array"].iloc[0]
        assert isinstance(intensity, list)
        assert intensity == pytest.approx([1000.0, 2000.0, 3000.0])

    def test_mz_array_parsed_numpy_format(self, loader, metadata_dir_numpy_arrays):
        """mz_array in numpy print format (spaces, no commas) should be parsed."""
        dataset = loader.load(data_path=metadata_dir_numpy_arrays)
        mz = dataset.metadata["mz_array"].iloc[0]
        assert isinstance(mz, list)
        assert mz == pytest.approx([100.0, 200.0, 300.0])

    # ------------------------------------------------------------------
    # Successful load – with predictions.pkl
    # ------------------------------------------------------------------

    def test_loads_predictions_pkl_when_present(self, loader, metadata_dir):
        """predictions.pkl should be loaded and returned as the predictions attribute."""
        # Build a minimal predictions list and pickle it
        fake_predictions = [[None], [None]]
        with (metadata_dir / "predictions.pkl").open("wb") as f:
            pickle.dump(fake_predictions, f)

        dataset = loader.load(data_path=metadata_dir)
        assert dataset.predictions == fake_predictions

    def test_predictions_length_matches_metadata(self, loader, metadata_dir):
        """len(predictions) must equal len(metadata) when pkl is present."""
        rows = 2  # matches metadata_dir fixture
        fake_predictions = [None] * rows
        with (metadata_dir / "predictions.pkl").open("wb") as f:
            pickle.dump(fake_predictions, f)

        dataset = loader.load(data_path=metadata_dir)
        assert len(dataset.predictions) == len(dataset.metadata)
