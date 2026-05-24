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
    if len(df) > 0 and isinstance(df["sequence"].iloc[0], str):
        df["sequence"] = df["sequence"].apply(metrics._split_peptide)
    if len(df) > 0 and isinstance(df["prediction"].iloc[0], str):
        df["prediction"] = df["prediction"].apply(metrics._split_peptide)

    def _row_correct(row: pd.Series) -> bool:
        if not isinstance(row["sequence"], list) or not isinstance(
            row["prediction"], list
        ):
            return False
        num_matches = metrics._novor_match(row["sequence"], row["prediction"])
        return num_matches == len(row["sequence"]) == len(row["prediction"])

    return df.apply(_row_correct, axis=1)


def _coerce_bool_labels(series: pd.Series, column: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(
            f"Label column {column!r} contains missing values; "
            "all PSMs must have a label for calibration diagnostics."
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
    scores: NDArray[np.floating],
    labels: NDArray[np.bool_],
    conf_cutoff: float,
    min_tail_psms: int = 1,
) -> Tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """Restrict scores and labels to the tail S >= conf_cutoff."""
    mask = scores >= conf_cutoff
    n_tail = int(mask.sum())
    if n_tail < min_tail_psms:
        raise ValueError(
            f"Only {n_tail} PSMs in the tail (S >= {conf_cutoff:.4f}); "
            f"need at least {min_tail_psms} for a stable isotonic calibration estimate."
        )
    return scores[mask], labels[mask]


def empirical_calibration_curve(
    scores: NDArray[np.floating],
    labels: NDArray[np.floating],
    n_bins: int = 40,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Equal-frequency calibration curve. Returns (bin_mean_s, bin_mean_y, bin_weight)."""
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


def ece(
    scores: NDArray[np.floating],
    labels: NDArray[np.floating],
    n_bins: int = 40,
) -> float:
    """Expected calibration error (equal-frequency bins)."""
    bin_s, bin_y, weights = empirical_calibration_curve(scores, labels, n_bins=n_bins)
    return float(np.sum(weights * np.abs(bin_s - bin_y)) / weights.sum())


def signed_tail_ece_empirical(
    scores: NDArray[np.floating],
    labels: NDArray[np.floating],
    conf_cutoff: float,
) -> float:
    """sTECE(conf_cutoff) = E[c(S) - S | S >= conf_cutoff], estimated from pointwise labels."""
    mask = scores >= conf_cutoff
    if not mask.any():
        return float("nan")
    return float(labels[mask].mean() - scores[mask].mean())


def fit_isotonic_calibration(
    scores: NDArray[np.floating],
    labels: NDArray[np.floating],
) -> IsotonicRegression:
    """Fit isotonic regression mapping score -> empirical correctness rate."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(scores, labels)
    return iso


def signed_tail_ece_isotonic(
    scores: NDArray[np.floating],
    labels: NDArray[np.floating],
    conf_cutoff: float,
    iso: Optional[IsotonicRegression] = None,
) -> float:
    """Compute sTECE via isotonic estimate of c(s): mean(c_hat(s) - s) on the tail."""
    tail_scores, tail_labels = filter_tail(
        scores, labels.astype(bool), conf_cutoff, min_tail_psms=1
    )
    if iso is None:
        iso = fit_isotonic_calibration(tail_scores, tail_labels.astype(float))
    c_hat = iso.predict(tail_scores)
    return float(np.mean(c_hat - tail_scores))


def tail_ece(
    scores: NDArray[np.floating],
    labels: NDArray[np.floating],
    conf_cutoff: float,
    iso: Optional[IsotonicRegression] = None,
) -> float:
    """TECE(conf_cutoff) = E[|c(S) - S| | S >= conf_cutoff] with c estimated isotonically."""
    tail_scores, tail_labels = filter_tail(
        scores, labels.astype(bool), conf_cutoff, min_tail_psms=1
    )
    if iso is None:
        iso = fit_isotonic_calibration(tail_scores, tail_labels.astype(float))
    c_hat = iso.predict(tail_scores)
    return float(np.mean(np.abs(c_hat - tail_scores)))


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
    scores: NDArray[np.floating],
    labels: NDArray[np.bool_],
    conf_cutoff: float,
    nominal_fdr: float,
    tolerance: float,
    label_source: str,
    label_column: str,
    min_tail_psms: int = 100,
    n_bins: int = 20,
) -> CalibrationDiagnosticResult:
    """Compute tail calibration metrics at the confidence cutoff."""
    tail_scores, tail_labels = filter_tail(
        np.asarray(scores, dtype=float),
        np.asarray(labels, dtype=bool),
        conf_cutoff,
        min_tail_psms=min_tail_psms,
    )
    labels_f = tail_labels.astype(float)
    iso = fit_isotonic_calibration(tail_scores, labels_f)
    stece = signed_tail_ece_isotonic(
        np.asarray(scores, dtype=float),
        np.asarray(labels, dtype=bool),
        conf_cutoff,
        iso=iso,
    )
    tece = tail_ece(
        np.asarray(scores, dtype=float),
        np.asarray(labels, dtype=bool),
        conf_cutoff,
        iso=iso,
    )
    stece_empirical = signed_tail_ece_empirical(
        np.asarray(scores, dtype=float), labels_f, conf_cutoff
    )
    within = abs(stece) <= tolerance
    return CalibrationDiagnosticResult(
        conf_cutoff=float(conf_cutoff),
        n_tail=int(len(tail_scores)),
        nominal_fdr=float(nominal_fdr),
        stece=stece,
        tece=tece,
        stece_empirical=stece_empirical,
        label_source=label_source,
        label_column=label_column,
        tolerance=tolerance,
        within_tolerance=within,
        interpretation=_interpret_stece(stece),
    )


def reliability_diagram(
    scores: NDArray[np.floating],
    labels: NDArray[np.floating],
    output_path: Union[Path, str],
    conf_cutoff: float,
    n_bins: int = 20,
) -> None:
    """Save a reliability diagram for the operating tail (S >= conf_cutoff)."""
    import matplotlib.pyplot as plt

    tail_scores, tail_labels = filter_tail(
        np.asarray(scores, dtype=float),
        np.asarray(labels, dtype=bool),
        conf_cutoff,
        min_tail_psms=1,
    )
    bin_s, bin_y, _weights = empirical_calibration_curve(
        tail_scores, tail_labels.astype(float), n_bins=n_bins
    )
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
    scores: NDArray[np.floating],
    labels: NDArray[np.floating],
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
            scores,
            labels.astype(float),
            out / "reliability_diagram.png",
            conf_cutoff=result.conf_cutoff,
            n_bins=n_bins,
        )
