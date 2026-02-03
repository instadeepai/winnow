"""Unit tests for winnow data loaders.

This module tests the token remapping functionality that converts various
notation formats (Casanovo, InstaNovo legacy) to UNIMOD format.
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
        """Residue remapping for Casanovo-specific notations to UNIMOD tokens."""
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
            # N-terminal modifications
            "[Acetyl]-": "[UNIMOD:1]",  # Acetylation
            "[Carbamyl]-": "[UNIMOD:5]",  # Carbamylation
            "[Ammonia-loss]-": "[UNIMOD:385]",  # Ammonia loss
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

    def test_map_modifications_casanovo_residue_modifications(self, mztab_loader):
        """Test mapping of Casanovo residue modifications (mass and named notation)."""
        # Test mass-based notation
        assert mztab_loader._map_modifications("M+15.995") == "M[UNIMOD:35]"
        assert mztab_loader._map_modifications("N+0.984") == "N[UNIMOD:7]"

        # Test named notation
        assert mztab_loader._map_modifications("M[Oxidation]") == "M[UNIMOD:35]"
        assert mztab_loader._map_modifications("N[Deamidated]") == "N[UNIMOD:7]"

    def test_map_modifications_casanovo_terminal_modifications(self, mztab_loader):
        """Test mapping of Casanovo terminal modifications (mass and named notation)."""
        # Test mass-based notation
        assert mztab_loader._map_modifications("+42.011PEPTIDE") == "[UNIMOD:1]PEPTIDE"
        assert (
            mztab_loader._map_modifications("-17.027PEPTIDE") == "[UNIMOD:385]PEPTIDE"
        )

        # Test named notation
        assert (
            mztab_loader._map_modifications("[Acetyl]-PEPTIDE") == "[UNIMOD:1]PEPTIDE"
        )
        assert (
            mztab_loader._map_modifications("[Ammonia-loss]-PEPTIDE")
            == "[UNIMOD:385]PEPTIDE"
        )

    def test_map_modifications_complex_sequence(self, mztab_loader):
        """Test mapping in a complex peptide sequence with multiple modifications."""
        # Complex sequence with multiple modification types
        input_seq = "M+15.995PEPTIDEN+0.984C+57.021"
        expected = "M[UNIMOD:35]PEPTIDEN[UNIMOD:7]C[UNIMOD:4]"
        assert mztab_loader._map_modifications(input_seq) == expected

    def test_map_modifications_terminal_plus_residue(self, mztab_loader):
        """Test terminal modification followed by residue modifications."""
        input_seq = "+42.011PEPTM+15.995IDE"
        expected = "[UNIMOD:1]PEPTM[UNIMOD:35]IDE"
        assert mztab_loader._map_modifications(input_seq) == expected

    def test_map_modifications_no_modifications(self, mztab_loader):
        """Test that unmodified sequences pass through unchanged."""
        unmodified = "PEPTIDE"
        assert mztab_loader._map_modifications(unmodified) == unmodified

    def test_map_modifications_already_unimod(self, mztab_loader):
        """Test that sequences already in UNIMOD format are unchanged."""
        unimod_seq = "M[UNIMOD:35]PEPTIDEN[UNIMOD:7]"
        assert mztab_loader._map_modifications(unimod_seq) == unimod_seq

    def test_map_modifications_mixed_formats(self, mztab_loader):
        """Test mapping when both Casanovo and UNIMOD notations are present."""
        # If a sequence somehow has both formats, it should still work
        input_seq = "M+15.995PEPTIDEN[UNIMOD:7]C+57.021"
        expected = "M[UNIMOD:35]PEPTIDEN[UNIMOD:7]C[UNIMOD:4]"
        assert mztab_loader._map_modifications(input_seq) == expected


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
    they are checked against invalid_prosit_tokens lists.
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
    def invalid_prosit_tokens_unimod_only(self):
        """List of invalid tokens using ONLY UNIMOD notation."""
        return [
            "[UNIMOD:7]",  # Deamidation
            "[UNIMOD:1]",  # Acetylation
            "[UNIMOD:5]",  # Carbamylation
            "[UNIMOD:385]",  # Ammonia loss
        ]

    def test_casanovo_tokens_map_to_invalid_unimod(
        self, residue_masses, invalid_prosit_tokens_unimod_only
    ):
        """Test that Casanovo tokens are detected as invalid after remapping to UNIMOD."""
        residue_remapping = {
            "Q+0.984": "Q[UNIMOD:7]",
            "N+0.984": "N[UNIMOD:7]",
            "+42.011": "[UNIMOD:1]",
            "+43.006": "[UNIMOD:5]",
            "-17.027": "[UNIMOD:385]",
            "[Acetyl]-": "[UNIMOD:1]",
            "[Carbamyl]-": "[UNIMOD:5]",
            "[Ammonia-loss]-": "[UNIMOD:385]",
            "N[Deamidated]": "N[UNIMOD:7]",
            "Q[Deamidated]": "Q[UNIMOD:7]",
        }

        loader = MZTabDatasetLoader(
            residue_masses=residue_masses,
            residue_remapping=residue_remapping,
        )

        # Test sequences with Casanovo notation that should be remapped
        casanovo_sequences = [
            ("PEPTIDEQ+0.984", "PEPTIDEQ[UNIMOD:7]"),  # Deamidation
            ("N+0.984PEPTIDE", "N[UNIMOD:7]PEPTIDE"),  # Deamidation
            ("+42.011PEPTIDE", "[UNIMOD:1]PEPTIDE"),  # Acetylation
            ("+43.006PEPTIDE", "[UNIMOD:5]PEPTIDE"),  # Carbamylation
            ("-17.027PEPTIDE", "[UNIMOD:385]PEPTIDE"),  # Ammonia loss
            ("[Acetyl]-PEPTIDE", "[UNIMOD:1]PEPTIDE"),  # Acetylation (named)
            ("[Carbamyl]-PEPTIDE", "[UNIMOD:5]PEPTIDE"),  # Carbamylation (named)
            (
                "[Ammonia-loss]-PEPTIDE",
                "[UNIMOD:385]PEPTIDE",
            ),  # Ammonia loss (named)
            ("PEPTIN[Deamidated]E", "PEPTIN[UNIMOD:7]E"),  # Deamidation (named)
            ("Q[Deamidated]PEPTIDE", "Q[UNIMOD:7]PEPTIDE"),  # Deamidation (named)
        ]

        for casanovo_seq, expected_unimod in casanovo_sequences:
            # Map the modifications
            remapped_seq = loader._map_modifications(casanovo_seq)

            # Verify remapping worked
            assert (
                remapped_seq == expected_unimod
            ), f"Failed to remap {casanovo_seq} to {expected_unimod}, got {remapped_seq}"

            # Verify that the remapped sequence contains an invalid UNIMOD token
            contains_invalid = any(
                token in remapped_seq for token in invalid_prosit_tokens_unimod_only
            )
            assert contains_invalid, f"Remapped sequence {remapped_seq} should contain an invalid UNIMOD token"

    def test_valid_sequences_not_affected(
        self, residue_masses, invalid_prosit_tokens_unimod_only
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
            remapped_seq = loader._map_modifications(seq)
            contains_invalid = any(
                token in remapped_seq for token in invalid_prosit_tokens_unimod_only
            )
            assert (
                not contains_invalid
            ), f"Valid sequence {remapped_seq} should not contain invalid tokens"

    def test_only_unimod_notation_needed_in_invalid_list(
        self, residue_masses, invalid_prosit_tokens_unimod_only
    ):
        """Test that we only need UNIMOD notation in invalid_prosit_tokens list.

        This test demonstrates that after remapping, we don't need to include
        Casanovo-specific notations like '+0.984', '+42.011', etc. in the
        invalid_prosit_tokens list - only the UNIMOD equivalents are needed.
        """
        residue_remapping = {
            "Q+0.984": "Q[UNIMOD:7]",
            "+42.011": "[UNIMOD:1]",
            "[Carbamyl]-": "[UNIMOD:5]",
        }

        loader = MZTabDatasetLoader(
            residue_masses=residue_masses,
            residue_remapping=residue_remapping,
        )

        # These Casanovo sequences should be caught by UNIMOD-only invalid list
        test_cases = [
            ("PEPTIDEQ+0.984", True),  # Should be invalid after remapping
            ("+42.011PEPTIDE", True),  # Should be invalid after remapping
            ("[Carbamyl]-PEPTIDE", True),  # Should be invalid after remapping
            ("PEPTIDE", False),  # Should be valid
            ("M[UNIMOD:35]PEPTIDE", False),  # Valid modification
        ]

        for seq, should_be_invalid in test_cases:
            remapped_seq = loader._map_modifications(seq)
            contains_invalid = any(
                token in remapped_seq for token in invalid_prosit_tokens_unimod_only
            )

            if should_be_invalid:
                assert (
                    contains_invalid
                ), f"Sequence {seq} -> {remapped_seq} should be caught as invalid"
            else:
                assert (
                    not contains_invalid
                ), f"Sequence {seq} -> {remapped_seq} should be valid"
