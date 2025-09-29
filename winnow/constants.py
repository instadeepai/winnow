from instanovo.utils.residues import ResidueSet
from instanovo.utils.metrics import Metrics

RESIDUE_MASSES: dict[str, float] = {
    "G": 57.021464,
    "A": 71.037114,
    "S": 87.032028,
    "P": 97.052764,
    "V": 99.068414,
    "T": 101.047670,
    "C": 103.009185,
    "L": 113.084064,
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
    # Modifications
    "M[UNIMOD:35]": 147.035400,  # Oxidation
    "N[UNIMOD:7]": 115.026943,  # Deamidation
    "Q[UNIMOD:7]": 129.042594,  # Deamidation
    "C[UNIMOD:4]": 160.030649,  # Carboxyamidomethylation
    "S[UNIMOD:21]": 166.998028,  # Phosphorylation
    "T[UNIMOD:21]": 181.01367,  # Phosphorylation
    "Y[UNIMOD:21]": 243.029329,  # Phosphorylation
    "[UNIMOD:385]": -17.026549,  # Ammonia Loss
    "[UNIMOD:5]": 43.005814,  # Carbamylation
    "[UNIMOD:1]": 42.010565,  # Acetylation
    "C[UNIMOD:312]": 222.013284,  # Cysteinylation
    "E[UNIMOD:27]": 111.032028,  # Glu -> pyro-Glu
    "Q[UNIMOD:28]": 111.032029,  # Gln -> pyro-Gln
    "(+25.98)": 25.980265,  # Carbamylation & NH3 loss
}

RESIDUE_REMAPPING: dict[str, str] = {
    "M(ox)": "M[UNIMOD:35]",  # Oxidation
    "M(+15.99)": "M[UNIMOD:35]",
    "S(p)": "S[UNIMOD:21]",  # Phosphorylation
    "T(p)": "T[UNIMOD:21]",
    "Y(p)": "Y[UNIMOD:21]",
    "S(+79.97)": "S[UNIMOD:21]",
    "T(+79.97)": "T[UNIMOD:21]",
    "Y(+79.97)": "Y[UNIMOD:21]",
    "Q(+0.98)": "Q[UNIMOD:7]",  # Deamidation
    "N(+0.98)": "N[UNIMOD:7]",
    "Q(+.98)": "Q[UNIMOD:7]",
    "N(+.98)": "N[UNIMOD:7]",
    "C(+57.02)": "C[UNIMOD:4]",  # Carbamidomethylation
    "(+42.01)": "[UNIMOD:1]",  # Acetylation
    "(+43.01)": "[UNIMOD:5]",  # Carbamylation
    "(-17.03)": "[UNIMOD:385]",  # Loss of ammonia
}

CASANOVO_RESIDUE_REMAPPING: dict[str, str] = {
    "M+15.995": "M[UNIMOD:35]",  # Oxidation
    "Q+0.984": "Q[UNIMOD:7]",  # Deamidation
    "N+0.984": "N[UNIMOD:7]",  # Deamidation
    "+42.011": "[UNIMOD:1]",  # Acetylation
    "+43.006": "[UNIMOD:5]",  # Carbamylation
    "-17.027": "[UNIMOD:385]",  # Loss of ammonia
    "C+57.021": "C[UNIMOD:4]",  # Carbamidomethylation
    # "+43.006-17.027": "[UNIMOD:5][UNIMOD:385]",  # Carbamylation and Loss of ammonia
    "C[Carbamidomethyl]": "C[UNIMOD:4]",  # Carbamidomethylation
    "M[Oxidation]": "M[UNIMOD:35]",  # Met oxidation:   131.040485 + 15.994915
    "N[Deamidated]": "N[UNIMOD:7]",  # Asn deamidation: 114.042927 +  0.984016
    "Q[Deamidated]": "Q[UNIMOD:7]",  # Gln deamidation: 128.058578 +  0.984016
    # N-terminal modifications.
    "[Acetyl]-": "[UNIMOD:1]",  # Acetylation
    "[Carbamyl]-": "[UNIMOD:5]",  # Carbamylation
    "[Ammonia-loss]-": "[UNIMOD:385]",  # Ammonia loss
    # "[+25.980265]-": 25.980265     # Carbamylation and ammonia loss
}

INVALID_PROSIT_TOKENS: list = [
    "\\+25.98",
    "UNIMOD:7",
    "UNIMOD:21",
    "UNIMOD:1",
    "UNIMOD:5",
    "UNIMOD:385",
    # Each C is also treated as Cysteine with carbamidomethylation in Prosit.
]


residue_set = ResidueSet(
    residue_masses=RESIDUE_MASSES, residue_remapping=RESIDUE_REMAPPING
)
metrics = Metrics(residue_set=residue_set, isotope_error_range=[0, 1])
