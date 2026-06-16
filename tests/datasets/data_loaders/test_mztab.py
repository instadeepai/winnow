"""Tests for MZTabDatasetLoader."""

import numpy as np
import polars as pl
import pytest

from winnow.datasets.data_loaders import MZTabDatasetLoader
from tests.datasets.data_loaders.conftest import (
    _FULL_RESIDUE_MASSES,
    _STANDARD_REMAPPING,
)


class TestMZTabDatasetLoader:
    """Unit tests for MZTabDatasetLoader."""

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
    def db_loader(self):
        """Loader mapped to traditional database-search mzTab columns."""
        return MZTabDatasetLoader(
            residue_masses=_FULL_RESIDUE_MASSES,
            residue_remapping=_STANDARD_REMAPPING,
            load_beams=False,
            column_mapping={
                "predictions": "sequence",
                "confidence": "search_engine_score[1]",
                "token_scores": None,
            },
        )

    @pytest.fixture()
    def unsupported_residues_unimod_only(self):
        """List of invalid residues using only UNIMOD notation."""
        return [
            "N[UNIMOD:7]",  # Deamidated asparagine
            "Q[UNIMOD:7]",  # Deamidated glutamine
            "[UNIMOD:1]",  # N-terminal acetylation
            "[UNIMOD:5]",  # N-terminal carbamylation
            "[UNIMOD:385]",  # N-terminal ammonia loss
        ]

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
        self, residue_masses, unsupported_residues_unimod_only
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
            assert expected_token in unsupported_residues_unimod_only, (
                f"Residue {expected_token} should be in unsupported_residues list"
            )

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
                "opt_ms_run[1]_proforma": ["PEP"],
                "search_engine_score[1]": [0.95],
                "opt_ms_run[1]_aa_scores": ["0.9,0.8,0.7"],
            }
        )

    def test_process_predictions_extracts_index_from_spectra_ref(
        self, db_loader, minimal_predictions_df
    ):
        result = db_loader._process_predictions(
            minimal_predictions_df, ["index"], is_casanovo=False
        )
        assert result["index"].to_list() == [7, 42]  # sorted by index ASC

    def test_process_predictions_replaces_l_with_i(self, db_loader):
        df = pl.DataFrame(
            {
                "spectra_ref": ["ms_run[1]:index=0"],
                "sequence": ["PEPTLDE"],
                "search_engine_score[1]": [0.9],
            }
        )
        result = db_loader._process_predictions(df, ["index"], is_casanovo=False)
        assert "L" not in result["prediction_untokenised"][0]

    def test_process_predictions_without_aa_scores_creates_null_token_scores(
        self, db_loader, minimal_predictions_df
    ):
        """Traditional search engines lack aa_scores; token_scores should be null."""
        result = db_loader._process_predictions(
            minimal_predictions_df, ["index"], is_casanovo=False
        )
        assert "token_scores" in result.columns
        assert result["token_scores"][0] is None

    def test_process_predictions_parses_aa_scores_as_list_of_floats(
        self, loader, predictions_df_with_aa_scores
    ):
        result = loader._process_predictions(
            predictions_df_with_aa_scores, ["index"], is_casanovo=True
        )
        token_scores = result["token_scores"][0].to_list()
        assert token_scores == pytest.approx([0.9, 0.8, 0.7])

    def test_process_predictions_sorted_by_index_asc_confidence_desc(self, db_loader):
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
        result = db_loader._process_predictions(df, ["index"], is_casanovo=False)
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
        result = loader._get_top_predictions(df, is_casanovo=False)
        assert len(result) == 2

    def test_get_top_predictions_keeps_highest_confidence(self, loader):
        df = pl.DataFrame(
            {
                "index": [0, 0, 1],
                "confidence": [0.9, 0.5, 0.7],
                "prediction_untokenised": ["A", "B", "C"],
            }
        )
        result = loader._get_top_predictions(df, is_casanovo=False)
        row_idx0 = result.filter(pl.col("index") == 0)
        assert row_idx0["confidence"][0] == pytest.approx(0.9)

    def test_get_top_predictions_casanovo_recovers_probability(self, loader):
        df = pl.DataFrame(
            {
                "index": [0],
                "confidence": [-0.18],
                "prediction_untokenised": ["PEP"],
            }
        )
        result = loader._get_top_predictions(df, is_casanovo=True)
        assert result["confidence"][0] == pytest.approx(0.82)

    # ------------------------------------------------------------------
    # _create_casanovo_beam_predictions
    # ------------------------------------------------------------------

    @pytest.fixture()
    def beam_predictions_df(self):
        """DataFrame in the format expected by _create_casanovo_beam_predictions."""
        return pl.DataFrame(
            {
                "index": pl.Series([0, 1], dtype=pl.Int64),
                "confidence": [0.9, 0.7],
                "prediction": [["P", "E", "P"], ["A", "C"]],
                "token_scores": pl.Series(
                    [[0.9, 0.8, 0.7], None], dtype=pl.List(pl.Float64)
                ),
            }
        )

    def test_create_casanovo_beam_predictions_returns_scored_sequences(
        self, loader, beam_predictions_df
    ):
        result = loader._create_casanovo_beam_predictions(beam_predictions_df, [0, 1])
        assert len(result) == 2
        assert result[0] is not None
        assert hasattr(result[0][0], "sequence")
        assert hasattr(result[0][0], "sequence_log_probability")

    def test_create_casanovo_beam_predictions_returns_none_for_missing_index(
        self, loader, beam_predictions_df
    ):
        """A requested index with no predictions in the df should produce None."""
        result = loader._create_casanovo_beam_predictions(beam_predictions_df, [0, 99])
        assert result[1] is None

    def test_create_casanovo_beam_predictions_preserves_order_of_valid_spectra_indices(
        self, loader, beam_predictions_df
    ):
        """Output list order must follow valid_spectra_indices, not df row order."""
        result = loader._create_casanovo_beam_predictions(beam_predictions_df, [1, 0])
        assert result[0][0].sequence == ["A", "C"]
        assert result[1][0].sequence == ["P", "E", "P"]

    def test_create_casanovo_beam_predictions_highest_confidence_first(self, loader):
        """Beam order follows native-score sort (highest confidence first)."""
        df = pl.DataFrame(
            {
                "index": pl.Series([0, 0, 0], dtype=pl.Int64),
                "confidence": [0.82, 0.05, -0.18],
                "prediction": [["HIGH"], ["MID"], ["LOW"]],
                "token_scores": pl.Series(
                    [None, None, None], dtype=pl.List(pl.Float64)
                ),
            }
        )
        result = loader._create_casanovo_beam_predictions(df, [0])
        assert result[0][0].sequence == ["HIGH"]
        assert result[0][1].sequence == ["MID"]
        assert result[0][2].sequence == ["LOW"]
        assert result[0][0].sequence_log_probability == pytest.approx(
            MZTabDatasetLoader._casanovo_raw_score_to_log_probability(0.82)
        )
        assert result[0][2].sequence_log_probability == pytest.approx(
            MZTabDatasetLoader._casanovo_raw_score_to_log_probability(-0.18)
        )

    def test_create_casanovo_beam_predictions_token_log_probs(self, loader):
        df = pl.DataFrame(
            {
                "index": pl.Series([0, 0], dtype=pl.Int64),
                "confidence": [0.9, 0.5],
                "prediction": [["P"], ["A"]],
                "token_scores": pl.Series(
                    [[0.4, 0.5, 0.6], [0.1, 0.2, 0.3]], dtype=pl.List(pl.Float64)
                ),
            }
        )
        result = loader._create_casanovo_beam_predictions(df, [0])
        assert result[0][0].sequence_log_probability == pytest.approx(np.log(0.9))
        assert result[0][0].token_log_probabilities == pytest.approx(
            [np.log(0.4), np.log(0.5), np.log(0.6)]
        )

    # ------------------------------------------------------------------
    # load() – validation
    # ------------------------------------------------------------------

    def test_load_raises_when_predictions_path_is_none(self, loader, tmp_path):
        with pytest.raises(ValueError, match="predictions_path is required"):
            loader.load(data_path=tmp_path)

    # ------------------------------------------------------------------
    # Casanovo detection, score transforms, beams, merge
    # ------------------------------------------------------------------

    def test_is_casanovo_mztab_detection(self):
        casanovo_df = pl.DataFrame({"search_engine": ["[MS, MS:1003281, Casanovo, ]"]})
        db_df = pl.DataFrame({"search_engine": ["[MS, MS:1001153, Comet, ]"]})
        assert MZTabDatasetLoader._is_casanovo_mztab(casanovo_df) is True
        assert MZTabDatasetLoader._is_casanovo_mztab(db_df) is False

    def test_casanovo_score_helpers(self):
        assert MZTabDatasetLoader._casanovo_score_to_probability(
            -0.18
        ) == pytest.approx(0.82)
        assert MZTabDatasetLoader._casanovo_score_to_probability(0.82) == pytest.approx(
            0.82
        )
        assert MZTabDatasetLoader._casanovo_raw_score_to_log_probability(
            0.82
        ) == pytest.approx(np.log(0.82))

    def test_validate_casanovo_native_psm_score_rejects_out_of_range(self):
        MZTabDatasetLoader._validate_casanovo_native_psm_score(-0.18)
        MZTabDatasetLoader._validate_casanovo_native_psm_score(1.0)
        with pytest.raises(ValueError, match="\\[-1, 1\\]"):
            MZTabDatasetLoader._validate_casanovo_native_psm_score(1.5)
        with pytest.raises(ValueError, match="log-probability"):
            MZTabDatasetLoader._validate_casanovo_native_psm_score(-1.5)

    def test_validate_casanovo_token_probability_rejects_out_of_range(self):
        MZTabDatasetLoader._validate_casanovo_token_probability(0.0)
        MZTabDatasetLoader._validate_casanovo_token_probability(1.0)
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            MZTabDatasetLoader._validate_casanovo_token_probability(-0.1)
        with pytest.raises(ValueError, match="Log-probability"):
            MZTabDatasetLoader._validate_casanovo_token_probability(1.1)

    def test_process_predictions_rejects_casanovo_psm_score_out_of_range(self, loader):
        df = pl.DataFrame(
            {
                "spectra_ref": ["ms_run[1]:index=0"],
                "opt_ms_run[1]_proforma": ["PEP"],
                "search_engine_score[1]": [1.2],
            }
        )
        with pytest.raises(ValueError, match="\\[-1, 1\\]"):
            loader._process_predictions(df, [], is_casanovo=True)

    def test_process_predictions_rejects_log_probability_token_scores(self, loader):
        df = pl.DataFrame(
            {
                "spectra_ref": ["ms_run[1]:index=0"],
                "opt_ms_run[1]_proforma": ["PEP"],
                "search_engine_score[1]": [0.9],
                "opt_ms_run[1]_aa_scores": ["0.9,-0.1,0.7"],
            }
        )
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            loader._process_predictions(df, [], is_casanovo=True)

    def test_validate_load_beams_supported_raises_for_db_search(self):
        with pytest.raises(ValueError, match="load_beams=True is only supported"):
            MZTabDatasetLoader._validate_load_beams_supported(
                is_casanovo=False, load_beams=True
            )

    def test_merge_data_picks_higher_scoring_psm(self, db_loader):
        raw = pl.DataFrame(
            {
                "spectra_ref": [
                    "ms_run[1]:index=1",
                    "ms_run[1]:index=1",
                ],
                "sequence": ["LOW", "HIGH"],
                "search_engine_score[1]": [0.3, 0.9],
            }
        )
        spectrum = pl.DataFrame(
            {
                "charge": [2, 2],
                "mz_array": [[100.0], [200.0]],
            }
        )
        processed = db_loader._process_predictions(raw, [], is_casanovo=False)
        processed = db_loader._tokenize(
            processed, "prediction_untokenised", "prediction"
        )
        top = db_loader._get_top_predictions(processed, is_casanovo=False)
        merged = db_loader._merge_data(spectrum, top)

        assert len(merged) == 1
        assert merged["prediction_untokenised"][0] == "HIGH"
        assert merged["confidence"][0] == pytest.approx(0.9)

    def test_missing_mapped_column_lists_available_columns(self, loader):
        df = pl.DataFrame(
            {
                "spectra_ref": ["ms_run[1]:index=0"],
                "search_engine_score[1]": [0.9],
            }
        )
        with pytest.raises(ValueError, match="opt_ms_run\\[1\\]_proforma"):
            loader._process_predictions(df, [], is_casanovo=True)

    def test_invalid_spectra_ref_raises(self, db_loader):
        df = pl.DataFrame(
            {
                "spectra_ref": ["not-a-valid-ref"],
                "sequence": ["PEP"],
                "search_engine_score[1]": [0.9],
            }
        )
        with pytest.raises(ValueError, match="spectra_ref"):
            db_loader._process_predictions(df, [], is_casanovo=False)
