"""Tail calibration diagnostics for non-parametric FDR estimation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from instanovo.utils.metrics import Metrics
from instanovo.utils.residues import ResidueSet
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression

from winnow.datasets.calibration_dataset import CalibrationDataset

LabelSource = Literal["sequence", "precomputed"]
SEQUENCE_LABEL_COLUMN = "correct"


@dataclass(frozen=True)
class DiagnosticArrays:
    """Aligned confidence scores and boolean correctness labels."""

    scores: NDArray[np.float64]
    labels: NDArray[np.bool_]

    @classmethod
    def from_raw(cls, scores: object, labels: object) -> DiagnosticArrays:
        """Coerce inputs once at the module boundary."""
        score_array = np.asarray(scores, dtype=np.float64)
        label_array = np.asarray(labels, dtype=bool)
        if score_array.shape != label_array.shape:
            raise ValueError(
                f"scores and labels must have the same length; "
                f"got {score_array.shape[0]} and {label_array.shape[0]}."
            )
        return cls(scores=score_array, labels=label_array)


@dataclass(frozen=True)
class TailSlice:
    """Scores and labels restricted to the operating tail S >= conf_cutoff."""

    scores: NDArray[np.float64]
    labels: NDArray[np.bool_]


@dataclass
class CalibrationDiagnosticResult:
    """Results from a tail calibration diagnostic run."""

    conf_cutoff: float
    n_tail: int
    nominal_fdr: float
    stece: float
    tece: float
    stece_empirical: float
    label_source: str
    label_column: str
    tolerance: float
    within_tolerance: bool
    interpretation: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


def validate_label_config(
    label_source: str,
    label_column: Optional[str],
) -> None:
    """Validate mutual exclusion between label_source and label_column."""
    if label_source not in ("sequence", "precomputed"):
        raise ValueError(
            f"diagnostics.label_source must be 'sequence' or 'precomputed', got {label_source!r}."
        )
    if label_source == "sequence" and label_column is not None:
        raise ValueError(
            "diagnostics.label_column must not be set when label_source is 'sequence'. "
            "Sequence-derived labels are written to the 'correct' column automatically."
        )
    if label_source == "precomputed" and not label_column:
        raise ValueError(
            "diagnostics.label_column is required when label_source is 'precomputed'."
        )


def _normalize_peptide_tokens(value: object, metrics: Metrics) -> list[str]:
    """Normalize a sequence or prediction cell to a list of residue tokens."""
    if isinstance(value, str):
        return metrics._split_peptide(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return []


def compute_correct_from_sequence(
    metadata: pd.DataFrame,
    residue_masses: dict[str, float],
    residue_remapping: Optional[dict[str, str]] = None,
) -> pd.Series:
    """Derive full-sequence correctness from sequence and prediction columns."""
    if "sequence" not in metadata.columns:
        raise ValueError(
            "label_source='sequence' requires a 'sequence' column in the dataset."
        )
    if "prediction" not in metadata.columns:
        raise ValueError(
            "label_source='sequence' requires a 'prediction' column in the dataset."
        )

    metrics = Metrics(
        residue_set=ResidueSet(
            residue_masses=residue_masses,
            residue_remapping=residue_remapping or {},
        )
    )
    df = metadata.copy()
    df["sequence"] = df["sequence"].apply(
        lambda value: _normalize_peptide_tokens(value, metrics)
    )
    df["prediction"] = df["prediction"].apply(
        lambda value: _normalize_peptide_tokens(value, metrics)
    )

    def _row_correct(row: pd.Series) -> bool:
        sequence = row["sequence"]
        prediction = row["prediction"]
        if not sequence or not prediction:
            return False
        num_matches = metrics._novor_match(sequence, prediction)
        return num_matches == len(sequence) == len(prediction)

    return df.apply(_row_correct, axis=1)


def _coerce_bool_labels(series: pd.Series, column: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(
            f"Label column {column!r} contains missing values; "
            "all PSMs must have a label for calibration diagnostics."
        )
    # Allow only boolean or numeric labels
    if series.dtype not in [bool, int, float]:
        raise ValueError(
            f"Label column {column!r} must be a boolean or numeric series, got {series.dtype}."
        )
    return series.astype(bool)


def resolve_diagnostics_labels(
    dataset: CalibrationDataset,
    label_source: LabelSource,
    label_column: Optional[str],
    residue_masses: dict[str, float],
    residue_remapping: Optional[dict[str, str]] = None,
) -> Tuple[pd.Series, str]:
    """Resolve boolean labels and return (labels, resolved_column_name)."""
    metadata = dataset.metadata

    if label_source == "precomputed":
        assert label_column is not None
        if label_column not in metadata.columns:
            raise ValueError(
                f"Precomputed label column {label_column!r} not found in dataset metadata. "
                f"Available columns: {list(metadata.columns)}"
            )
        labels = _coerce_bool_labels(metadata[label_column], label_column)
        return labels, label_column

    labels = compute_correct_from_sequence(metadata, residue_masses, residue_remapping)
    return labels, SEQUENCE_LABEL_COLUMN


def filter_tail(
    data: DiagnosticArrays,
    conf_cutoff: float,
    min_tail_psms: int = 1,
) -> TailSlice:
    """Restrict scores and labels to the tail S >= conf_cutoff."""
    mask = data.scores >= conf_cutoff
    n_tail = int(mask.sum())
    if n_tail < min_tail_psms:
        raise ValueError(
            f"Only {n_tail} PSMs in the tail (S >= {conf_cutoff:.4f}); "
            f"need at least {min_tail_psms} for a stable isotonic calibration estimate."
        )
    return TailSlice(scores=data.scores[mask], labels=data.labels[mask])


def empirical_calibration_curve(
    data: DiagnosticArrays | TailSlice,
    n_bins: int = 40,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Equal-frequency calibration curve. Returns (bin_mean_s, bin_mean_y, bin_weight)."""
    scores = data.scores
    labels = data.labels
    order = np.argsort(scores)
    s_sorted = scores[order]
    y_sorted = labels[order]
    edges = np.linspace(0, len(s_sorted), n_bins + 1).astype(int)
    means_s: list[float] = []
    means_y: list[float] = []
    weights: list[int] = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        means_s.append(float(s_sorted[start:end].mean()))
        means_y.append(float(y_sorted[start:end].mean()))
        weights.append(end - start)
    return (
        np.asarray(means_s, dtype=np.float64),
        np.asarray(means_y, dtype=np.float64),
        np.asarray(weights, dtype=np.int64),
    )


def empirical_stece(tail: TailSlice) -> float:
    """Estimate sTECE from pointwise labels on a pre-filtered tail."""
    return float(tail.labels.mean() - tail.scores.mean())


def fit_isotonic_calibration(tail: TailSlice) -> IsotonicRegression:
    """Fit isotonic regression mapping score -> empirical correctness rate."""
    calibration_curve = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibration_curve.fit(tail.scores, tail.labels.astype(np.float64))
    return calibration_curve


def isotonic_stece(
    tail: TailSlice,
    calibration_curve: IsotonicRegression,
) -> float:
    """Compute sTECE via isotonic estimate of c(s): mean(c_hat(s) - s) on the tail."""
    predicted_rate = calibration_curve.predict(tail.scores)
    return float(np.mean(predicted_rate - tail.scores))


def isotonic_tece(
    tail: TailSlice,
    calibration_curve: IsotonicRegression,
) -> float:
    """TECE on a pre-filtered tail with c estimated isotonically."""
    predicted_rate = calibration_curve.predict(tail.scores)
    return float(np.mean(np.abs(predicted_rate - tail.scores)))


def _interpret_stece(stece: float) -> str:
    if stece < 0:
        return (
            "Tail is over-confident: reported FDR likely understates the true error rate "
            "(more PSMs accepted than the nominal target permits)."
        )
    if stece > 0:
        return (
            "Tail is under-confident: reported FDR likely overstates the true error rate "
            "(valid discoveries may be discarded)."
        )
    return "Tail calibration residual is near zero at the operating threshold."


def run_calibration_diagnostic(
    data: DiagnosticArrays,
    conf_cutoff: float,
    nominal_fdr: float,
    tolerance: float,
    label_source: str,
    label_column: str,
    min_tail_psms: int = 100,
) -> CalibrationDiagnosticResult:
    """Compute tail calibration metrics at the confidence cutoff."""
    tail = filter_tail(data, conf_cutoff, min_tail_psms=min_tail_psms)
    calibration_curve = fit_isotonic_calibration(tail)
    stece = isotonic_stece(tail, calibration_curve)
    tece = isotonic_tece(tail, calibration_curve)
    stece_empirical = empirical_stece(tail)
    within_tolerance = abs(stece) <= tolerance
    return CalibrationDiagnosticResult(
        conf_cutoff=float(conf_cutoff),
        n_tail=int(len(tail.scores)),
        nominal_fdr=float(nominal_fdr),
        stece=stece,
        tece=tece,
        stece_empirical=stece_empirical,
        label_source=label_source,
        label_column=label_column,
        tolerance=tolerance,
        within_tolerance=within_tolerance,
        interpretation=_interpret_stece(stece),
    )


def reliability_diagram(
    data: DiagnosticArrays,
    output_path: Union[Path, str],
    conf_cutoff: float,
    n_bins: int = 20,
) -> None:
    """Save a reliability diagram for the operating tail (S >= conf_cutoff)."""
    import matplotlib.pyplot as plt

    tail = filter_tail(data, conf_cutoff, min_tail_psms=1)
    bin_s, bin_y, _weights = empirical_calibration_curve(tail, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot(bin_s, bin_y, "o-", linewidth=1.5, markersize=4, label="Empirical")
    ax.set_xlabel("Mean predicted score")
    ax.set_ylabel("Empirical correctness rate")
    ax.set_title(rf"Reliability diagram ($S \geq {conf_cutoff:.3f}$)")
    ax.set_xlim(conf_cutoff, 1.0)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_diagnostic_report(
    result: CalibrationDiagnosticResult,
    output_dir: Union[Path, str],
    plot: bool,
    data: DiagnosticArrays,
    n_bins: int,
) -> None:
    """Write JSON report and optional reliability diagram."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "diagnostic_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2)
    if plot:
        reliability_diagram(
            data,
            out / "reliability_diagram.png",
            conf_cutoff=result.conf_cutoff,
            n_bins=n_bins,
        )
