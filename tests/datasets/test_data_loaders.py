"""Unit tests for winnow data loaders.

This module tests the token remapping functionality that converts various
notation formats (Casanovo, InstaNovo legacy) to Proforma format.
"""

import pytest
from winnow.datasets.data_loaders import (
    InstaNovoDatasetLoader,
    MZTabDatasetLoader,
)


class TestMZTabDatasetLoader:
    """Test the MZTabDatasetLoader class, particularly token remapping."""

    @pytest.fixture()
    def residue_masses(self):
        """Minimal residue masses for testing."""
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
            "M[UNIMOD:35]": 147.035400,  # Oxidation
            "C[UNIMOD:4]": 160.030649,  # Carbamidomethylation
            "N[UNIMOD:7]": 115.026943,  # Deamidation
            "Q[UNIMOD:7]": 129.042594,  # Deamidation
            # Terminal modifications
            "[UNIMOD:1]": 42.010565,  # Acetylation
            "[UNIMOD:5]": 43.005814,  # Carbamylation
            "[UNIMOD:385]": -17.026549,  # NH3 loss
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
        """Create an MZTabDatasetLoader instance for testing."""
        return MZTabDatasetLoader(
            residue_masses=residue_masses,
            residue_remapping=residue_remapping,
        )

    def test_initialization(self, mztab_loader):
        """Test MZTabDatasetLoader initialization."""
        assert mztab_loader.metrics is not None
        assert mztab_loader.metrics.residue_set is not None

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


class TestInstaNovoDatasetLoader:
    """Test the InstaNovoDatasetLoader class, particularly token remapping."""

    @pytest.fixture()
    def residue_masses(self):
        """Minimal residue masses for testing."""
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
            "M[UNIMOD:35]": 147.035400,  # Oxidation
            "C[UNIMOD:4]": 160.030649,  # Carbamidomethylation
            "N[UNIMOD:7]": 115.026943,  # Deamidation
            "Q[UNIMOD:7]": 129.042594,  # Deamidation
            "S[UNIMOD:21]": 166.998028,  # Phosphorylation
            "T[UNIMOD:21]": 181.01367,  # Phosphorylation
            "Y[UNIMOD:21]": 243.029329,  # Phosphorylation
            # Terminal modifications
            "[UNIMOD:1]": 42.010565,  # Acetylation
            "[UNIMOD:5]": 43.005814,  # Carbamylation
            "[UNIMOD:385]": -17.026549,  # NH3 loss
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
        """Create an InstaNovoDatasetLoader instance for testing."""
        return InstaNovoDatasetLoader(
            residue_masses=residue_masses,
            residue_remapping=residue_remapping,
        )

    def test_initialization(self, instanovo_loader):
        """Test InstaNovoDatasetLoader initialization."""
        assert instanovo_loader.metrics is not None
        assert instanovo_loader.metrics.residue_set is not None

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


class TestTokenRemappingIntegration:
    """Integration tests to verify token remapping works with invalid token filtering.

    These tests verify that the data loaders correctly remap tokens BEFORE
    they are checked against invalid_prosit_residues lists.
    """

    @pytest.fixture()
    def residue_masses(self):
        """Minimal residue masses for testing."""
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
            "M[UNIMOD:35]": 147.035400,  # Oxidation
            "C[UNIMOD:4]": 160.030649,  # Carbamidomethylation
            "N[UNIMOD:7]": 115.026943,  # Deamidation
            "Q[UNIMOD:7]": 129.042594,  # Deamidation
            # Terminal modifications
            "[UNIMOD:1]": 42.010565,  # Acetylation
            "[UNIMOD:5]": 43.005814,  # Carbamylation
            "[UNIMOD:385]": -17.026549,  # NH3 loss
        }

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
