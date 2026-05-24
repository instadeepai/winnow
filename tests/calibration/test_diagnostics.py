"""Unit tests for winnow.calibration.diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from winnow.calibration.diagnostics import (
    SEQUENCE_LABEL_COLUMN,
    compute_correct_from_sequence,
    filter_tail,
    resolve_diagnostics_labels,
    run_calibration_diagnostic,
    signed_tail_ece_empirical,
    signed_tail_ece_isotonic,
    tail_ece,
    validate_label_config,
)
from winnow.datasets.calibration_dataset import CalibrationDataset

RESIDUE_MASSES = {"A": 71.037114, "G": 57.021464}


def _perfectly_calibrated_tail(
    n: int = 500, conf_cutoff: float = 0.5, seed: int = 0
) -> tuple:
    rng = np.random.default_rng(seed)
    scores = rng.uniform(conf_cutoff, 1.0, size=n)
    labels = (rng.uniform(size=n) < scores).astype(bool)
    return scores, labels


class TestValidateLabelConfig:
    def test_sequence_with_label_column_raises(self) -> None:
        with pytest.raises(ValueError, match="label_column must not be set"):
            validate_label_config("sequence", "correct")

    def test_precomputed_without_column_raises(self) -> None:
        with pytest.raises(ValueError, match="label_column is required"):
            validate_label_config("precomputed", None)

    def test_invalid_source_raises(self) -> None:
        with pytest.raises(ValueError, match="label_source must be"):
            validate_label_config("auto", None)

    def test_sequence_without_column_ok(self) -> None:
        validate_label_config("sequence", None)

    def test_precomputed_with_column_ok(self) -> None:
        validate_label_config("precomputed", "proteome_hit")


class TestComputeCorrectFromSequence:
    def test_full_sequence_match(self) -> None:
        meta = pd.DataFrame(
            {
                "sequence": [["A", "G"], ["A"]],
                "prediction": [["A", "G"], ["G"]],
            }
        )
        correct = compute_correct_from_sequence(meta, RESIDUE_MASSES)
        assert correct.tolist() == [True, False]


class TestResolveDiagnosticsLabels:
    def test_precomputed_proteome_hit(self) -> None:
        meta = pd.DataFrame(
            {
                "spectrum_id": ["a", "b", "c"],
                "proteome_hit": [True, False, True],
                "confidence": [0.9, 0.8, 0.7],
            }
        )
        labels, col = resolve_diagnostics_labels(
            CalibrationDataset(metadata=meta),
            "precomputed",
            "proteome_hit",
            residue_masses=RESIDUE_MASSES,
        )
        assert col == "proteome_hit"
        assert labels.tolist() == [True, False, True]

    def test_sequence_derives_correct(self) -> None:
        meta = pd.DataFrame(
            {
                "spectrum_id": ["a", "b"],
                "sequence": [["A"], ["A"]],
                "prediction": [["A"], ["G"]],
                "proteome_hit": [True, True],
            }
        )
        labels, col = resolve_diagnostics_labels(
            CalibrationDataset(metadata=meta),
            "sequence",
            None,
            residue_masses=RESIDUE_MASSES,
        )
        assert col == SEQUENCE_LABEL_COLUMN
        assert labels.tolist() == [True, False]

    def test_sequence_ignores_stale_precomputed_column(self) -> None:
        meta = pd.DataFrame(
            {
                "sequence": [["A"], ["A"]],
                "prediction": [["A"], ["G"]],
                "correct": [False, True],
            }
        )
        labels, _ = resolve_diagnostics_labels(
            CalibrationDataset(metadata=meta),
            "sequence",
            None,
            residue_masses=RESIDUE_MASSES,
        )
        assert labels.tolist() == [True, False]

    def test_precomputed_missing_column_raises(self) -> None:
        meta = pd.DataFrame({"confidence": [0.5]})
        with pytest.raises(ValueError, match="not found"):
            resolve_diagnostics_labels(
                CalibrationDataset(metadata=meta),
                "precomputed",
                "proteome_hit",
                residue_masses=RESIDUE_MASSES,
            )


class TestFilterTail:
    def test_too_few_psms_raises(self) -> None:
        scores = np.array([0.9, 0.8])
        labels = np.array([True, False])
        with pytest.raises(ValueError, match="Only 1 PSMs"):
            filter_tail(scores, labels, conf_cutoff=0.85, min_tail_psms=10)


class TestTailCalibrationMetrics:
    def test_perfect_calibration_near_zero_stece(self) -> None:
        scores, labels = _perfectly_calibrated_tail(n=2000, conf_cutoff=0.6)
        stece = signed_tail_ece_isotonic(scores, labels.astype(float), conf_cutoff=0.6)
        assert abs(stece) < 0.05

    def test_overconfident_tail_negative_stece(self) -> None:
        rng = np.random.default_rng(1)
        n = 1000
        latent = rng.beta(8, 2, size=n)
        labels = (rng.uniform(size=n) < latent).astype(bool)
        scores = np.clip(latent**0.5, 0, 1)
        stece = signed_tail_ece_isotonic(scores, labels.astype(float), conf_cutoff=0.5)
        assert stece < -0.02

    def test_tece_bounds_abs_stece(self) -> None:
        scores, labels = _perfectly_calibrated_tail(n=800, conf_cutoff=0.5)
        stece = signed_tail_ece_isotonic(scores, labels.astype(float), conf_cutoff=0.5)
        tece = tail_ece(scores, labels.astype(float), conf_cutoff=0.5)
        assert tece >= abs(stece) - 1e-9

    def test_empirical_stece_on_tail(self) -> None:
        scores = np.array([0.9, 0.85, 0.8])
        labels = np.array([1.0, 0.0, 1.0])
        stece = signed_tail_ece_empirical(scores, labels, conf_cutoff=0.8)
        assert stece == pytest.approx(labels.mean() - scores.mean())


class TestRunCalibrationDiagnostic:
    def test_within_tolerance_flag(self) -> None:
        scores, labels = _perfectly_calibrated_tail(n=1500, conf_cutoff=0.5)
        result = run_calibration_diagnostic(
            scores=scores,
            labels=labels,
            conf_cutoff=0.5,
            nominal_fdr=0.05,
            tolerance=0.5,
            label_source="sequence",
            label_column="correct",
            min_tail_psms=50,
        )
        assert result.within_tolerance

    def test_fail_tolerance(self) -> None:
        scores = np.linspace(0.7, 1.0, 200)
        labels = np.zeros(200, dtype=bool)
        result = run_calibration_diagnostic(
            scores=scores,
            labels=labels,
            conf_cutoff=0.7,
            nominal_fdr=0.05,
            tolerance=0.001,
            label_source="precomputed",
            label_column="correct",
            min_tail_psms=50,
        )
        assert not result.within_tolerance
        assert result.stece < 0
