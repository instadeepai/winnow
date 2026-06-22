"""Unit tests for winnow.calibration.diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from winnow.calibration.diagnostics import (
    SEQUENCE_LABEL_COLUMN,
    DiagnosticArrays,
    TailSlice,
    compute_correct_from_sequence,
    empirical_stece,
    filter_tail,
    fit_isotonic_calibration,
    isotonic_stece,
    isotonic_tece,
    resolve_diagnostics_labels,
    run_calibration_diagnostic,
    validate_label_config,
)
from winnow.datasets.calibration_dataset import CalibrationDataset

RESIDUE_MASSES = {"A": 71.037114, "G": 57.021464}


class TestValidateLabelConfig:
    @pytest.mark.parametrize(
        ("label_source", "label_column", "error_match"),
        [
            ("sequence", "correct", "label_column must not be set"),
            ("precomputed", None, "label_column is required"),
            ("auto", None, "label_source must be"),
        ],
    )
    def test_invalid_config_raises(
        self, label_source: str, label_column: str | None, error_match: str
    ) -> None:
        with pytest.raises(ValueError, match=error_match):
            validate_label_config(label_source, label_column)

    @pytest.mark.parametrize(
        ("label_source", "label_column"),
        [
            ("sequence", None),
            ("precomputed", "proteome_hit"),
        ],
    )
    def test_valid_config_passes(
        self, label_source: str, label_column: str | None
    ) -> None:
        validate_label_config(label_source, label_column)


class TestComputeCorrectFromSequence:
    @pytest.mark.parametrize(
        ("sequence", "prediction", "expected"),
        [
            ([["A", "G"], ["A"]], [["A", "G"], ["G"]], [True, False]),
            (
                [np.array(["A", "G"], dtype=object)],
                [np.array(["A", "G"], dtype=object)],
                [True],
            ),
        ],
    )
    def test_sequence_correctness(
        self, sequence: object, prediction: object, expected: list[bool]
    ) -> None:
        meta = pd.DataFrame({"sequence": sequence, "prediction": prediction})
        correct = compute_correct_from_sequence(meta, RESIDUE_MASSES)
        assert correct.tolist() == expected


class TestResolveDiagnosticsLabels:
    def test_precomputed_reads_column(self) -> None:
        meta = pd.DataFrame({"proteome_hit": [True, False, True]})
        labels, column = resolve_diagnostics_labels(
            CalibrationDataset(metadata=meta),
            "precomputed",
            "proteome_hit",
            residue_masses=RESIDUE_MASSES,
        )
        assert column == "proteome_hit"
        assert labels.tolist() == [True, False, True]

    def test_sequence_uses_derived_correct_column(self) -> None:
        meta = pd.DataFrame(
            {
                "sequence": [["A"], ["A"]],
                "prediction": [["A"], ["G"]],
                "correct": [False, True],
            }
        )
        labels, column = resolve_diagnostics_labels(
            CalibrationDataset(metadata=meta),
            "sequence",
            None,
            residue_masses=RESIDUE_MASSES,
        )
        assert column == SEQUENCE_LABEL_COLUMN
        assert labels.tolist() == [True, False]

    def test_precomputed_missing_column_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            resolve_diagnostics_labels(
                CalibrationDataset(metadata=pd.DataFrame({"confidence": [0.5]})),
                "precomputed",
                "proteome_hit",
                residue_masses=RESIDUE_MASSES,
            )

    def test_precomputed_string_labels_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a boolean or numeric series"):
            resolve_diagnostics_labels(
                CalibrationDataset(
                    metadata=pd.DataFrame(
                        {
                            "confidence": [0.5, 0.7, 0.6],
                            "proteome_hit": ["True", "False", "True"],
                        }
                    )
                ),
                "precomputed",
                "proteome_hit",
                residue_masses=RESIDUE_MASSES,
            )


class TestDiagnosticArrays:
    def test_from_raw_coerces_inputs(self) -> None:
        data = DiagnosticArrays.from_raw([0.9, "0.8"], [1, 0])
        assert data.scores.dtype == np.float64
        assert data.labels.dtype == bool
        assert data.scores.tolist() == pytest.approx([0.9, 0.8])
        assert data.labels.tolist() == [True, False]

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            DiagnosticArrays.from_raw([0.9, 0.8], [True])


class TestFilterTail:
    def test_keeps_only_scores_at_or_above_cutoff(self) -> None:
        data = DiagnosticArrays.from_raw(
            [0.9, 0.85, 0.4, 0.3], [True, False, True, False]
        )
        tail = filter_tail(data, conf_cutoff=0.8, min_tail_psms=1)
        assert tail.scores.tolist() == pytest.approx([0.9, 0.85])
        assert tail.labels.tolist() == [True, False]

    def test_too_few_psms_raises(self) -> None:
        data = DiagnosticArrays.from_raw([0.9, 0.8], [True, False])
        with pytest.raises(ValueError, match="Only 1 PSMs"):
            filter_tail(data, conf_cutoff=0.85, min_tail_psms=10)


class TestEmpiricalStece:
    def test_known_tail_mean(self) -> None:
        tail = TailSlice(
            scores=np.array([1.0, 0.0]),
            labels=np.array([True, False]),
        )
        assert empirical_stece(tail) == pytest.approx(0.0)


class TestIsotonicMetrics:
    def test_overconfident_tail_negative_stece(self) -> None:
        tail = TailSlice(
            scores=np.linspace(0.7, 1.0, 200),
            labels=np.zeros(200, dtype=bool),
        )
        calibration_curve = fit_isotonic_calibration(tail)
        assert isotonic_stece(tail, calibration_curve) < 0

    def test_tece_bounds_abs_stece(self) -> None:
        rng = np.random.default_rng(0)
        scores = rng.uniform(0.5, 1.0, size=800)
        labels = (rng.uniform(size=800) < scores).astype(bool)
        tail = TailSlice(scores=scores, labels=labels)
        calibration_curve = fit_isotonic_calibration(tail)
        stece = isotonic_stece(tail, calibration_curve)
        tece = isotonic_tece(tail, calibration_curve)
        assert tece >= abs(stece) - 1e-9


class TestRunCalibrationDiagnostic:
    def test_within_tolerance_follows_isotonic_stece(self) -> None:
        data = DiagnosticArrays.from_raw(
            np.linspace(0.7, 1.0, 200),
            np.zeros(200, dtype=bool),
        )
        result = run_calibration_diagnostic(
            data=data,
            conf_cutoff=0.7,
            nominal_fdr=0.05,
            tolerance=0.5,
            label_source="precomputed",
            label_column="correct",
            min_tail_psms=50,
        )
        assert result.within_tolerance == (abs(result.stece) <= 0.5)
        assert result.stece < 0

    def test_stece_empirical_uses_filtered_tail(self) -> None:
        data = DiagnosticArrays.from_raw(
            [0.9, 0.85, 0.4, 0.3],
            [True, False, True, False],
        )
        result = run_calibration_diagnostic(
            data=data,
            conf_cutoff=0.8,
            nominal_fdr=0.05,
            tolerance=1.0,
            label_source="sequence",
            label_column="correct",
            min_tail_psms=2,
        )
        tail = filter_tail(data, conf_cutoff=0.8, min_tail_psms=1)
        assert result.stece_empirical == pytest.approx(empirical_stece(tail))
        assert result.n_tail == len(tail.scores)
