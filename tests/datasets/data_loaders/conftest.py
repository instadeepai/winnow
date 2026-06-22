"""Shared fixtures for data loader tests."""

import pytest


@pytest.fixture(scope="session")
def full_residue_masses():
    """Standard residue masses for loader tests."""
    return {
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
        "M[UNIMOD:35]": 147.035400,
        "C[UNIMOD:4]": 160.030649,
        "N[UNIMOD:7]": 115.026943,
        "Q[UNIMOD:7]": 129.042594,
        "[UNIMOD:1]": 42.010565,
        "[UNIMOD:5]": 43.005814,
        "[UNIMOD:385]": -17.026549,
    }


@pytest.fixture(scope="session")
def standard_remapping():
    """Standard Casanovo-style residue remapping for loader tests."""
    return {
        "M[Oxidation]": "M[UNIMOD:35]",
        "C[Carbamidomethyl]": "C[UNIMOD:4]",
        "[Acetyl]": "[UNIMOD:1]",
        "[Carbamyl]": "[UNIMOD:5]",
        "[Ammonia-loss]": "[UNIMOD:385]",
    }
