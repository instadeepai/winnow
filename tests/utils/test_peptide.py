"""Unit tests for peptide utility functions.

This module tests the ProForma-compliant tokenization and detokenization
of peptide sequences, including N-terminal and C-terminal modifications.
"""

import pytest

from winnow.utils.peptide import tokens_to_proforma, _is_standalone_modification


class TestIsStandaloneModification:
    """Test the _is_standalone_modification helper function."""

    def test_unimod_notation(self):
        """Test UNIMOD notation is detected as standalone modification."""
        assert _is_standalone_modification("[UNIMOD:1]") is True
        assert _is_standalone_modification("[UNIMOD:385]") is True

    def test_mass_notation_parentheses(self):
        """Test mass notation in parentheses is detected as standalone modification."""
        assert _is_standalone_modification("(+42.01)") is True
        assert _is_standalone_modification("(-17.03)") is True

    def test_raw_mass_notation(self):
        """Test raw mass notation is detected as standalone modification."""
        assert _is_standalone_modification("+42.011") is True
        assert _is_standalone_modification("-17.026") is True
        assert _is_standalone_modification("42.011") is True

    def test_amino_acid_with_modification(self):
        """Test that amino acids with attached modifications are NOT standalone."""
        assert _is_standalone_modification("M[UNIMOD:35]") is False
        assert _is_standalone_modification("C[UNIMOD:4]") is False
        assert _is_standalone_modification("M(ox)") is False
        assert _is_standalone_modification("S(p)") is False

    def test_plain_amino_acids(self):
        """Test that plain amino acids are NOT standalone modifications."""
        assert _is_standalone_modification("A") is False
        assert _is_standalone_modification("M") is False
        assert _is_standalone_modification("P") is False

    def test_empty_and_edge_cases(self):
        """Test edge cases."""
        assert _is_standalone_modification("") is False


class TestTokensToProforma:
    """Test the tokens_to_proforma function."""

    # --- Basic cases ---

    def test_simple_peptide(self):
        """Test simple peptide without modifications."""
        tokens = ["P", "E", "P", "T", "I", "D", "E"]
        assert tokens_to_proforma(tokens) == "PEPTIDE"

    def test_empty_tokens(self):
        """Test empty token list returns empty string."""
        assert tokens_to_proforma([]) == ""
        assert tokens_to_proforma(None) == ""

    def test_single_amino_acid(self):
        """Test single amino acid."""
        assert tokens_to_proforma(["A"]) == "A"

    # --- Residue modifications (attached to amino acids) ---

    def test_residue_modification_unimod(self):
        """Test residue modification in UNIMOD notation."""
        tokens = ["M[UNIMOD:35]", "P", "E", "P", "T", "I", "D", "E"]
        assert tokens_to_proforma(tokens) == "M[UNIMOD:35]PEPTIDE"

    def test_residue_modification_parentheses(self):
        """Test residue modification in parentheses notation."""
        tokens = ["M(ox)", "P", "E", "P", "T", "I", "D", "E"]
        assert tokens_to_proforma(tokens) == "M(ox)PEPTIDE"

    def test_multiple_residue_modifications(self):
        """Test multiple residue modifications."""
        tokens = ["M[UNIMOD:35]", "P", "E", "P", "T", "I", "D", "E", "C[UNIMOD:4]"]
        assert tokens_to_proforma(tokens) == "M[UNIMOD:35]PEPTIDEC[UNIMOD:4]"

    # --- N-terminal modifications ---

    def test_n_terminal_modification_unimod(self):
        """Test N-terminal modification in UNIMOD notation adds hyphen."""
        tokens = ["[UNIMOD:1]", "P", "E", "P", "T", "I", "D", "E"]
        assert tokens_to_proforma(tokens) == "[UNIMOD:1]-PEPTIDE"

    def test_n_terminal_modification_mass(self):
        """Test N-terminal modification in mass notation adds hyphen."""
        tokens = ["(+42.01)", "P", "E", "P", "T", "I", "D", "E"]
        assert tokens_to_proforma(tokens) == "(+42.01)-PEPTIDE"

    def test_n_terminal_modification_raw_mass(self):
        """Test N-terminal modification in raw mass notation adds hyphen."""
        tokens = ["+42.011", "P", "E", "P", "T", "I", "D", "E"]
        assert tokens_to_proforma(tokens) == "+42.011-PEPTIDE"

    # --- C-terminal modifications ---

    def test_c_terminal_modification_unimod(self):
        """Test C-terminal modification in UNIMOD notation adds hyphen."""
        tokens = ["P", "E", "P", "T", "I", "D", "E", "[UNIMOD:2]"]
        assert tokens_to_proforma(tokens) == "PEPTIDE-[UNIMOD:2]"

    def test_c_terminal_modification_mass(self):
        """Test C-terminal modification in mass notation adds hyphen."""
        tokens = ["P", "E", "P", "T", "I", "D", "E", "(-0.98)"]
        assert tokens_to_proforma(tokens) == "PEPTIDE-(-0.98)"

    def test_c_terminal_modification_raw_mass(self):
        """Test C-terminal modification in raw mass notation adds hyphen."""
        tokens = ["P", "E", "P", "T", "I", "D", "E", "-0.984"]
        assert tokens_to_proforma(tokens) == "PEPTIDE--0.984"

    # --- Both N-terminal and C-terminal modifications ---

    def test_both_terminal_modifications(self):
        """Test both N-terminal and C-terminal modifications."""
        tokens = ["[UNIMOD:1]", "P", "E", "P", "T", "I", "D", "E", "[UNIMOD:2]"]
        assert tokens_to_proforma(tokens) == "[UNIMOD:1]-PEPTIDE-[UNIMOD:2]"

    def test_both_terminal_modifications_with_residue_mods(self):
        """Test terminal modifications combined with residue modifications."""
        tokens = [
            "[UNIMOD:1]",
            "M[UNIMOD:35]",
            "P",
            "E",
            "P",
            "T",
            "I",
            "D",
            "E",
            "[UNIMOD:2]",
        ]
        assert tokens_to_proforma(tokens) == "[UNIMOD:1]-M[UNIMOD:35]PEPTIDE-[UNIMOD:2]"

    # --- Edge cases ---

    def test_only_n_terminal_mod_and_one_aa(self):
        """Test N-terminal modification with single amino acid."""
        tokens = ["[UNIMOD:1]", "A"]
        assert tokens_to_proforma(tokens) == "[UNIMOD:1]-A"

    def test_only_c_terminal_mod_and_one_aa(self):
        """Test C-terminal modification with single amino acid."""
        tokens = ["A", "[UNIMOD:2]"]
        assert tokens_to_proforma(tokens) == "A-[UNIMOD:2]"

    def test_single_standalone_modification(self):
        """Test single standalone modification (edge case - unusual but handled)."""
        # This is an unusual case but should not crash
        tokens = ["[UNIMOD:1]"]
        assert tokens_to_proforma(tokens) == "[UNIMOD:1]"

    def test_does_not_mutate_input(self):
        """Test that the input list is not mutated."""
        tokens = ["[UNIMOD:1]", "P", "E", "P", "T", "I", "D", "E", "[UNIMOD:2]"]
        original = tokens.copy()
        tokens_to_proforma(tokens)
        assert tokens == original


class TestTokenizationRoundTrip:
    """Test that tokenization and detokenization work together correctly.

    These tests verify the complete flow from ProForma string -> tokens -> ProForma string.
    """

    @pytest.fixture()
    def residue_set(self):
        """Create a ResidueSet for tokenization tests."""
        from instanovo.utils.residues import ResidueSet

        return ResidueSet(
            residue_masses={
                "A": 71.037114,
                "P": 97.052764,
                "E": 129.042593,
                "T": 101.047670,
                "I": 113.084064,
                "D": 115.026943,
                "M": 131.040485,
                "M[UNIMOD:35]": 147.035400,
                "[UNIMOD:1]": 42.010565,
                "[UNIMOD:2]": 14.015650,  # Example C-terminal mod
            }
        )

    def test_n_terminal_round_trip(self, residue_set):
        """Test N-terminal modification tokenization round trip."""
        proforma_input = "[UNIMOD:1]-PEPTIDE"
        tokens = residue_set.tokenize(proforma_input)
        # Tokenizer strips the hyphen
        assert tokens == ["[UNIMOD:1]", "P", "E", "P", "T", "I", "D", "E"]
        # Detokenization adds it back
        assert tokens_to_proforma(tokens) == proforma_input

    def test_c_terminal_round_trip(self, residue_set):
        """Test C-terminal modification tokenization round trip."""
        proforma_input = "PEPTIDE-[UNIMOD:2]"
        tokens = residue_set.tokenize(proforma_input)
        # Tokenizer strips the hyphen
        assert tokens == ["P", "E", "P", "T", "I", "D", "E", "[UNIMOD:2]"]
        # Detokenization adds it back
        assert tokens_to_proforma(tokens) == proforma_input

    def test_both_terminal_round_trip(self, residue_set):
        """Test both terminal modifications tokenization round trip."""
        proforma_input = "[UNIMOD:1]-PEPTIDE-[UNIMOD:2]"
        tokens = residue_set.tokenize(proforma_input)
        # Tokenizer strips both hyphens
        assert tokens == ["[UNIMOD:1]", "P", "E", "P", "T", "I", "D", "E", "[UNIMOD:2]"]
        # Detokenization adds them back
        assert tokens_to_proforma(tokens) == proforma_input
