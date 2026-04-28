"""Tests for ``scripts/annotate_preds_proteome_hits.py`` (loaded by path)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl
import pytest
import yaml
from instanovo.utils.metrics import Metrics
from instanovo.utils.residues import ResidueSet

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "annotate_preds_proteome_hits.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "annotate_preds_proteome_hits", _SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ph = _load_script()


@pytest.fixture()
def metrics() -> Metrics:
    residues_path = _ROOT / "winnow/configs/residues.yaml"
    with residues_path.open() as f:
        data = yaml.safe_load(f)
    return Metrics(
        residue_set=ResidueSet(residue_masses=data["residue_masses"]),
        isotope_error_range=(0, 1),
    )


def test_processed_peptide_for_match_strips_mods() -> None:
    s = "PEP(+123.45)TIDE[UNIMOD:1]-K"
    out = ph.processed_peptide_for_match(s)
    assert "(" not in out
    assert "[" not in out
    assert out == "PEPTLDEK"


def test_filter_and_annotate_preds_short_removed(metrics: Metrics) -> None:
    haystack = "XXXXXXMKLLPEPTLDEMKLLYYYY"
    preds = pl.DataFrame(
        {
            "spectrum_id": ["a", "b", "c"],
            "prediction": ["PEPTIDE", "PEP", "NOTINDB"],
            "calibrated_confidence": [0.9, 0.8, 0.7],
        }
    )
    out = ph.filter_and_annotate_preds(preds, haystack, metrics, min_residue_length=7)
    assert out.height == 2
    hits = dict(zip(out["spectrum_id"].to_list(), out["proteome_hit"].to_list()))
    assert hits["a"] is True
    assert hits["c"] is False


def test_annotate_prediction_folder_writes(tmp_path: Path, metrics: Metrics) -> None:
    fasta = tmp_path / "db.fasta"
    fasta.write_text(">p1\nMKLLPEPTIDEMKLL\n")

    sub = tmp_path / "out" / "snakevenoms_raw"
    sub.mkdir(parents=True)
    preds = pl.DataFrame(
        {
            "spectrum_id": ["s1", "s2"],
            "prediction": ["PEPTIDE", "X"],
            "calibrated_confidence": [0.5, 0.4],
            "psm_fdr": [0.1, 0.2],
            "psm_q_value": [0.1, 0.2],
        }
    )
    meta = pl.DataFrame(
        {
            "spectrum_id": ["s1", "s2"],
            "extra": [1, 2],
        }
    )
    preds.write_csv(sub / "preds_and_fdr_metrics.csv")
    meta.write_csv(sub / "metadata.csv")

    n_in, n_short, n_kept = ph.annotate_prediction_folder(
        sub, fasta, metrics, min_residue_length=7
    )
    assert n_in == 2
    assert n_short == 1
    assert n_kept == 1

    out_preds = pl.read_csv(sub / "preds_and_fdr_metrics.csv")
    out_meta = pl.read_csv(sub / "metadata.csv")
    assert out_preds.height == 1
    assert out_meta.height == 1
    assert out_preds["spectrum_id"].item() == "s1"
    assert out_preds["proteome_hit"].item() is True
