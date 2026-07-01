"""Unit tests for :mod:`winnow.scripts.annotate_proteome_hits`."""

from __future__ import annotations

import polars as pl
import pytest
from instanovo.utils.metrics import Metrics

from winnow.scripts.annotate_proteome_hits import (
    filter_and_annotate_preds,
    metrics_from_residues_yaml,
    processed_peptide_for_match,
)
from winnow.utils.config_path import get_config_dir


@pytest.fixture()
def metrics() -> Metrics:
    return metrics_from_residues_yaml(get_config_dir() / "residues.yaml")


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


def test_filter_and_annotate_preds_short_removed(metrics: Metrics) -> None:
    haystack = "XXXXXXMKLLPEPTLDEMKLLYYYY"
    preds = pl.DataFrame(
        {
            "spectrum_id": ["a", "b", "c"],
            "prediction": ["PEPTIDE", "PEP", "NOTINDB"],
            "calibrated_confidence": [0.9, 0.8, 0.7],
        }
    )
    out = filter_and_annotate_preds(preds, haystack, metrics, min_residue_length=7)
    assert out.height == 2
    hits = dict(zip(out["spectrum_id"].to_list(), out["proteome_hit"].to_list()))
    assert hits["a"] is True
    assert hits["c"] is False
