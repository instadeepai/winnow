"""Feature column sets for subset calibrator training and evaluation."""

from __future__ import annotations

from typing import TypedDict


class FeatureSubsetSpec(TypedDict):
    """Metadata and column list for one feature-subset experiment."""

    description: str
    from_parquet: bool
    columns: list[str]


# Full feature matrix columns (``confidence`` + features + ``correct`` label).
FULL_FEATURE_COLUMNS: list[str] = [
    "confidence",
    "mass_error_ppm",
    "ion_matches",
    "ion_match_intensity",
    "complementary_ion_count",
    "max_ion_gap",
    "spectral_angle",
    "xcorr",
    "irt_error",
    "margin",
    "median_margin",
    "entropy",
    "z-score",
    "edit_distance",
    "min_token_probability",
    "std_token_probability",
]

_NO_XCORR_SPECTRAL = {"spectral_angle", "xcorr"}
_NO_FRAGMENT_SIMILARITY = _NO_XCORR_SPECTRAL | {
    "complementary_ion_count",
    "max_ion_gap",
    "edit_distance",
}


def _columns_excluding(*, drop: set[str]) -> list[str]:
    return [c for c in FULL_FEATURE_COLUMNS if c not in drop]


FEATURE_SUBSETS: dict[str, FeatureSubsetSpec] = {
    "no_xcorr_spectral": {
        "description": "Exclude spectral_angle and xcorr only.",
        "from_parquet": True,
        "columns": _columns_excluding(drop=_NO_XCORR_SPECTRAL),
    },
    "no_fragment_similarity": {
        "description": (
            "Exclude spectral_angle, xcorr, complementary_ion_count, "
            "max_ion_gap, and edit_distance."
        ),
        "from_parquet": True,
        "columns": _columns_excluding(drop=_NO_FRAGMENT_SIMILARITY),
    },
    "mass_error_da_no_similarity": {
        "description": (
            "Exclude mass_error_ppm and fragment-similarity features; "
            "use mass_error_da (Daltons) instead. Requires full winnow train "
            "(recomputes features from raw spectra)."
        ),
        "from_parquet": False,
        "columns": list(
            _columns_excluding(drop=_NO_FRAGMENT_SIMILARITY | {"mass_error_ppm"})
        )
        + ["mass_error_da"],
    },
}
