"""Unit tests for :mod:`winnow.utils.proteome`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from instanovo.utils.residues import ResidueSet

from winnow.calibration.diagnostics import resolve_diagnostics_labels
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.datasets.data_loaders import WinnowDatasetLoader
from winnow.scripts.main import _separate_metadata_and_predictions
from winnow.utils.config_path import get_config_dir
from winnow.utils.proteome import (
    annotate_calibration_dataset,
    processed_peptide_for_match,
    residue_set_from_residues_yaml,
    residue_token_count,
)


@pytest.fixture()
def residue_set() -> ResidueSet:
    return residue_set_from_residues_yaml(get_config_dir() / "residues.yaml")


@pytest.fixture()
def residue_masses(residue_set: ResidueSet) -> dict[str, float]:
    return dict(residue_set.residue_masses)


def test_processed_peptide_for_match_strips_mods() -> None:
    cases = [
        ("PEP(+123.45)TIDE[UNIMOD:35]K", "PEPTLDEK"),
        ("[UNIMOD:1]-PEP(+123.45)TIDE[UNIMOD:35]K", "PEPTLDEK"),
        ("PEP(foo)TAGDE", "PEPTAGDE"),
        ("(+47.01)-PEPTAGDE", "PEPTAGDE"),
        ("PEPTAGDE[Carboxyl]", "PEPTAGDE"),
        ("[Acetyl]-PEPTAGDE", "PEPTAGDE"),
        ("(N-term)PEPTAGDE", "PEPTAGDE"),
        ("PEP+15.99TIDE", "PEPTLDE"),
        ("PEP-15.99TIDE", "PEPTLDE"),
        ("PEP15.99TIDE", "PEPTLDE"),
        ("+42.01-PEPTAGDE", "PEPTAGDE"),
    ]
    for raw, expected in cases:
        out = processed_peptide_for_match(raw)
        assert "(" not in out, raw
        assert "[" not in out, raw
        assert out == expected, raw


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


def test_annotate_calibration_dataset_filters_and_matches(
    residue_set: ResidueSet, tmp_path: Path
) -> None:
    dataset = CalibrationDataset(
        metadata=pd.DataFrame(
            {
                "spectrum_id": ["a", "b", "c"],
                "prediction": [
                    ["P", "E", "P", "T", "I", "D", "E"],
                    ["P", "E", "P"],
                    ["A", "A", "A", "A", "A", "A", "A"],
                ],
                "prediction_untokenised": ["PEPTIDE", "PEP", "AAAAAAA"],
                "confidence": [0.9, 0.8, 0.7],
                "mz_array": [[100.0], [200.0], [300.0]],
                "intensity_array": [[1.0], [2.0], [3.0]],
                "sequence": [
                    ["P", "E", "P", "T", "I", "D", "E"],
                    ["P", "E", "P"],
                    ["A", "A", "A", "A", "A", "A", "A"],
                ],
                "correct": [True, False, False],
                "valid_sequence": [True, True, True],
                "valid_prediction": [True, True, True],
            }
        )
    )
    fasta_path = tmp_path / "proteome.fasta"
    fasta_path.write_text(">prot1\nXXXXXXMKLLPEPTLDEMKLLYYYY\n")

    annotated, stats = annotate_calibration_dataset(
        dataset, fasta_path, residue_set, min_residue_length=7
    )
    assert (stats.num_input, stats.num_removed_short, stats.num_kept) == (3, 1, 2)
    assert set(annotated.metadata["spectrum_id"]) == {"a", "c"}
    hits = dict(
        zip(
            annotated.metadata["spectrum_id"].tolist(),
            annotated.metadata["proteome_hit"].tolist(),
        )
    )
    assert hits["a"] is True
    assert hits["c"] is False
    assert "correct" in annotated.metadata.columns


def test_annotate_from_token_list_only(residue_set: ResidueSet, tmp_path: Path) -> None:
    """Match via tokens_to_proforma when prediction_untokenised is absent."""
    dataset = CalibrationDataset(
        metadata=pd.DataFrame(
            {
                "spectrum_id": ["a", "b"],
                "prediction": [
                    ["P", "E", "P", "T", "I", "D", "E"],
                    ["A", "A", "A", "A", "A", "A", "A"],
                ],
                "confidence": [0.9, 0.7],
                "mz_array": [[100.0], [300.0]],
                "intensity_array": [[1.0], [3.0]],
            }
        )
    )
    assert "prediction_untokenised" not in dataset.metadata.columns
    fasta_path = tmp_path / "proteome.fasta"
    fasta_path.write_text(">prot1\nXXXXXXMKLLPEPTLDEMKLLYYYY\n")

    annotated, stats = annotate_calibration_dataset(
        dataset, fasta_path, residue_set, min_residue_length=7
    )
    assert (stats.num_input, stats.num_removed_short, stats.num_kept) == (2, 0, 2)
    hits = dict(
        zip(
            annotated.metadata["spectrum_id"].tolist(),
            annotated.metadata["proteome_hit"].tolist(),
        )
    )
    assert hits["a"] is True
    assert hits["b"] is False


def test_annotate_no_hit_across_protein_boundary(
    residue_set: ResidueSet, tmp_path: Path
) -> None:
    dataset = CalibrationDataset(
        metadata=pd.DataFrame(
            {
                "spectrum_id": ["a"],
                "prediction": [["P", "E", "P", "T", "I", "D", "E"]],
                "prediction_untokenised": ["PEPTIDE"],
                "confidence": [0.9],
                "mz_array": [[100.0]],
                "intensity_array": [[1.0]],
            }
        )
    )
    fasta_path = tmp_path / "proteome.fasta"
    # PEPT|LDE across the join separator must not match PEPTIDE / PEPTLDE.
    fasta_path.write_text(">p1\nAAAAPEPT\n>p2\nLDEKKKKK\n")

    annotated, _stats = annotate_calibration_dataset(
        dataset, fasta_path, residue_set, min_residue_length=7
    )
    assert annotated.metadata["proteome_hit"].tolist() == [False]


def test_annotate_save_load_round_trip(
    residue_set: ResidueSet, residue_masses: dict[str, float], tmp_path: Path
) -> None:
    dataset = CalibrationDataset(
        metadata=pd.DataFrame(
            {
                "spectrum_id": ["a", "c"],
                "prediction": [
                    ["P", "E", "P", "T", "I", "D", "E"],
                    ["A", "A", "A", "A", "A", "A", "A"],
                ],
                "prediction_untokenised": ["PEPTIDE", "AAAAAAA"],
                "confidence": [0.9, 0.7],
                "mz_array": [[100.0, 200.0], [150.0, 250.0]],
                "intensity_array": [[1.0, 2.0], [1.5, 2.5]],
                "sequence": [
                    ["P", "E", "P", "T", "I", "D", "E"],
                    ["A", "A", "A", "A", "A", "A", "A"],
                ],
                "correct": [True, False],
                "valid_sequence": [True, True],
                "valid_prediction": [True, True],
            }
        )
    )
    fasta_path = tmp_path / "proteome.fasta"
    fasta_path.write_text(">prot1\nXXXXXXMKLLPEPTLDEMKLLYYYY\n")
    annotated, _stats = annotate_calibration_dataset(
        dataset, fasta_path, residue_set, min_residue_length=7
    )

    out_dir = tmp_path / "annotated"
    annotated.save(out_dir)

    loader = WinnowDatasetLoader(residue_masses=residue_masses, residue_remapping={})
    reloaded = loader.load(data_path=out_dir)
    assert "proteome_hit" in reloaded.metadata.columns
    assert reloaded.metadata["proteome_hit"].dtype == bool or set(
        reloaded.metadata["proteome_hit"].tolist()
    ) <= {True, False}
    assert "sequence" in reloaded.metadata.columns
    assert "correct" in reloaded.metadata.columns

    labels, column = resolve_diagnostics_labels(reloaded, "precomputed", "proteome_hit")
    assert column == "proteome_hit"
    assert labels.tolist() == [True, False]


def test_separate_metadata_leaves_proteome_hit() -> None:
    metadata = pd.DataFrame(
        {
            "spectrum_id": ["a"],
            "prediction": [["P", "E", "P"]],
            "calibrated_confidence": [0.9],
            "psm_fdr": [0.01],
            "psm_q_value": [0.01],
            "proteome_hit": [True],
            "feature_x": [1.0],
        }
    )
    meta_out, preds_out = _separate_metadata_and_predictions(
        metadata, "calibrated_confidence"
    )
    assert "proteome_hit" in meta_out.columns
    assert "proteome_hit" not in preds_out.columns
    assert "feature_x" in meta_out.columns
