"""Unit tests for :mod:`winnow.scripts.annotate_proteome_hits`."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from instanovo.utils.residues import ResidueSet

from winnow.scripts.annotate_proteome_hits import (
    annotate_prediction_folder,
    filter_and_annotate_preds,
    residue_set_from_residues_yaml,
    residue_token_count,
    processed_peptide_for_match,
)
from winnow.utils.config_path import get_config_dir


@pytest.fixture()
def residue_set() -> ResidueSet:
    return residue_set_from_residues_yaml(get_config_dir() / "residues.yaml")


def test_processed_peptide_for_match_strips_mods() -> None:
    cases = [
        ("PEP(+123.45)TIDE[UNIMOD:35]K", "PEPTLDEK"),
        ("[UNIMOD:1]-PEP(+123.45)TIDE[UNIMOD:35]K", "PEPTLDEK"),
        ("PEP(foo)TAGDE", "PEPTAGDE"),
        ("(+47.01)-PEPTAGDE", "PEPTAGDE"),
        ("PEPTAGDE[Carboxyl]", "PEPTAGDE"),
        ("[Acetyl]-PEPTAGDE", "PEPTAGDE"),
        ("(N-term)PEPTAGDE", "PEPTAGDE"),
    ]
    for raw, expected in cases:
        out = processed_peptide_for_match(raw)
        assert "(" not in out, raw
        assert "[" not in out, raw
        assert out == expected, raw


def test_filter_and_annotate_preds_short_removed(residue_set: ResidueSet) -> None:
    haystack = "XXXXXXMKLLPEPTLDEMKLLYYYY"
    preds = pl.DataFrame(
        {
            "spectrum_id": ["a", "b", "c"],
            "prediction": ["PEPTIDE", "PEP", "NOTINDB"],
            "calibrated_confidence": [0.9, 0.8, 0.7],
        }
    )
    out = filter_and_annotate_preds(preds, haystack, residue_set, min_residue_length=7)
    assert out.height == 2
    hits = dict(zip(out["spectrum_id"].to_list(), out["proteome_hit"].to_list()))
    assert hits["a"] is True
    assert hits["c"] is False


def test_filter_and_annotate_preds_no_hit_across_protein_boundary(
    residue_set: ResidueSet,
) -> None:
    # Two proteins joined by the separator: "PEPTLDE" spans the boundary and
    # must not be reported as a hit.
    haystack = "AAAAPEPT\x1fLDEKKKKK"
    preds = pl.DataFrame(
        {
            "spectrum_id": ["a"],
            "prediction": ["PEPTIDE"],
            "calibrated_confidence": [0.9],
        }
    )
    out = filter_and_annotate_preds(preds, haystack, residue_set, min_residue_length=7)
    assert out.height == 1
    assert out["proteome_hit"].to_list() == [False]


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        (None, 0),
        (float("nan"), 0),
        ("", 0),
        ("   ", 0),
        (["P", "E", "P", "T", "I", "D", "E"], 7),
        ("PEPTIDE", 7),
        ("PEP(+123.45)TIDE", 7),
    ],
)
def test_residue_token_count(
    residue_set: ResidueSet, prediction: object, expected: int
) -> None:
    assert residue_token_count(prediction, residue_set) == expected  # type: ignore[arg-type]


def test_annotate_prediction_folder_round_trip(
    residue_set: ResidueSet, tmp_path: Path
) -> None:
    folder = tmp_path
    preds_path = folder / "preds_and_fdr_metrics.csv"
    meta_path = folder / "metadata.csv"

    pl.DataFrame(
        {
            "spectrum_id": ["a", "b", "c"],
            "prediction": ["PEPTIDE", "PEP", "NOTINDB"],
            "calibrated_confidence": [0.9, 0.8, 0.7],
        }
    ).write_csv(preds_path)
    pl.DataFrame(
        {
            "spectrum_id": ["a", "b", "c"],
            "scan_number": [1, 2, 3],
        }
    ).write_csv(meta_path)

    fasta_path = folder / "proteome.fasta"
    fasta_path.write_text(">prot1\nXXXXXXMKLLPEPTLDEMKLLYYYY\n")

    n_in, n_short, n_kept = annotate_prediction_folder(
        folder, fasta_path, residue_set, min_residue_length=7
    )
    assert (n_in, n_short, n_kept) == (3, 1, 2)

    preds_out = pl.read_csv(preds_path)
    assert "proteome_hit" in preds_out.columns
    assert set(preds_out["spectrum_id"].to_list()) == {"a", "c"}
    hits = dict(
        zip(preds_out["spectrum_id"].to_list(), preds_out["proteome_hit"].to_list())
    )
    assert hits["a"] is True
    assert hits["c"] is False

    meta_out = pl.read_csv(meta_path)
    assert set(meta_out["spectrum_id"].to_list()) == {"a", "c"}
