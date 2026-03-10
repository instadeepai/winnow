#!/usr/bin/env python3
"""Feature ablation study for winnow calibrators.

This script trains multiple calibrator configurations on a training dataset,
evaluates each on in-domain and out-of-domain test sets, and produces:
- summary.csv: scalar metrics per config × dataset
- reliability_data.csv, pr_data.csv, fdr_data.csv: curve data for plot reproducibility
- Reliability diagrams, PR curves, and FDR curves as PNG plots
"""

from __future__ import annotations

import argparse
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from winnow.calibration.calibration_features import (
    BeamFeatures,
    ChimericFeatures,
    FragmentMatchFeatures,
    MassErrorFeature,
    RetentionTimeFeature,
)
from winnow.calibration.calibrator import ProbabilityCalibrator
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.datasets.data_loaders import InstaNovoDatasetLoader, WinnowDatasetLoader
from winnow.fdr.nonparametric import NonParametricFDRControl

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default feature construction kwargs — mirrors calibrator.yaml
DEFAULT_RESIDUE_MASSES: dict[str, float] = {
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
    "M[UNIMOD:35]": 147.035400,
    "C[UNIMOD:4]": 160.030649,
    "N[UNIMOD:7]": 115.026943,
    "Q[UNIMOD:7]": 129.042594,
    "R[UNIMOD:7]": 157.085127,
    "P[UNIMOD:35]": 113.047679,
    "S[UNIMOD:21]": 166.998028,
    "T[UNIMOD:21]": 181.01367,
    "Y[UNIMOD:21]": 243.029329,
    "C[UNIMOD:312]": 222.013284,
    "E[UNIMOD:27]": 111.032028,
    "Q[UNIMOD:28]": 111.032029,
    "[UNIMOD:1]": 42.010565,
    "[UNIMOD:5]": 43.005814,
    "[UNIMOD:385]": -17.026549,
    "(+25.98)": 25.980265,
}

DEFAULT_KOINA_KWARGS: dict[str, Any] = {
    "mz_tolerance": 0.02,
    "learn_from_missing": False,
    "intensity_model_name": "Prosit_2025_intensity_22PTM",
    "max_precursor_charge": 6,
    "max_peptide_length": 30,
    "unsupported_residues": ["[UNIMOD:5]", "[UNIMOD:385]", "(+25.98)"],
    "input_columns": {
        "collision_energies": "collision_energy",
        "fragmentation_types": "frag_type",
    },
}

DEFAULT_IRT_KWARGS: dict[str, Any] = {
    "hidden_dim": 10,
    "train_fraction": 0.1,
    "learn_from_missing": False,
    "seed": 42,
    "learning_rate_init": 0.001,
    "alpha": 0.0001,
    "max_iter": 200,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "irt_model_name": "Prosit_2025_irt_22PTM",
    "max_peptide_length": 30,
    "unsupported_residues": ["[UNIMOD:5]", "[UNIMOD:385]", "(+25.98)"],
}


# ---------------------------------------------------------------------------
# Ablation Configuration Definitions
# ---------------------------------------------------------------------------


def get_ablation_configs() -> OrderedDict[str, Callable[[], list]]:
    """Return an ordered dict mapping config names to feature list factories.

    Each factory is a callable that returns a fresh list of CalibrationFeatures
    objects, ensuring no shared mutable state across configurations.
    """

    def raw_confidence() -> list:
        """No additional features — confidence score alone."""
        return []

    def beam_uncertainty() -> list:
        """Beam search uncertainty features only."""
        return [BeamFeatures()]

    def prosit_irt_intensity() -> list:
        """Prosit / iRT / intensity agreement features."""
        return [
            MassErrorFeature(residue_masses=DEFAULT_RESIDUE_MASSES),
            FragmentMatchFeatures(**DEFAULT_KOINA_KWARGS),
            RetentionTimeFeature(**DEFAULT_IRT_KWARGS),
            ChimericFeatures(
                mz_tolerance=DEFAULT_KOINA_KWARGS["mz_tolerance"],
                learn_from_missing=DEFAULT_KOINA_KWARGS["learn_from_missing"],
                prosit_intensity_model_name=DEFAULT_KOINA_KWARGS[
                    "intensity_model_name"
                ],
                max_precursor_charge=DEFAULT_KOINA_KWARGS["max_precursor_charge"],
                max_peptide_length=DEFAULT_KOINA_KWARGS["max_peptide_length"],
                unsupported_residues=DEFAULT_KOINA_KWARGS["unsupported_residues"],
                model_input_constants=DEFAULT_KOINA_KWARGS["model_input_constants"],
            ),
        ]

    def full_model() -> list:
        """All features combined."""
        return prosit_irt_intensity() + [BeamFeatures()]

    return OrderedDict(
        [
            ("raw_confidence", raw_confidence),
            ("beam_uncertainty", beam_uncertainty),
            ("prosit_irt_intensity", prosit_irt_intensity),
            ("full_model", full_model),
        ]
    )


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------


def load_dataset(
    path: Path,
    predictions_path: Path | None = None,
    loader_type: str = "instanovo",
) -> CalibrationDataset:
    """Load a CalibrationDataset from the given path.

    Args:
        path: Path to spectrum data file or Winnow dataset directory.
        predictions_path: Path to predictions file (required for instanovo loader).
        loader_type: One of "instanovo" or "winnow".

    Returns:
        CalibrationDataset ready for calibration.
    """
    if loader_type == "winnow":
        return WinnowDatasetLoader(
            residue_masses=DEFAULT_RESIDUE_MASSES,
            residue_remapping={},
        ).load(data_path=path)

    # Default: instanovo
    if predictions_path is None:
        raise ValueError("predictions_path required for instanovo loader")
    return InstaNovoDatasetLoader(
        residue_masses=DEFAULT_RESIDUE_MASSES,
        residue_remapping={},
        beam_columns={
            "predictions_beam": "predictions_beam_{i}",
            "predictions_log_probability_beam": "predictions_log_probability_beam_{i}",
            "predictions_token_log_probabilities_beam": "predictions_token_log_probabilities_beam_{i}",
        },
    ).load(data_path=path, predictions_path=predictions_path)


def filter_dataset(dataset: CalibrationDataset) -> CalibrationDataset:
    """Filter out rows with empty or invalid predictions."""
    filtered = dataset.filter_entries(
        metadata_predicate=lambda row: not isinstance(row["prediction"], list),
    ).filter_entries(metadata_predicate=lambda row: not row["prediction"])
    return filtered


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------


@dataclass
class MetricsResult:
    """Container for all computed metrics for one config × dataset."""

    config_name: str
    dataset_name: str
    # Scalars (may be None if labels unavailable)
    roc_auc: float | None = None
    brier_score: float | None = None
    pr_auc: float | None = None
    ece: float | None = None
    tail_ece: float | None = None
    psms_at_1pct_fdr: int | None = None
    psms_at_5pct_fdr: int | None = None
    # Curve data
    reliability_bin_centers: list[float] = field(default_factory=list)
    reliability_fraction_positives: list[float] = field(default_factory=list)
    reliability_bin_counts: list[int] = field(default_factory=list)
    pr_recall: list[float] = field(default_factory=list)
    pr_precision: list[float] = field(default_factory=list)
    fdr_thresholds: list[float] = field(default_factory=list)
    fdr_n_accepted: list[int] = field(default_factory=list)


def compute_ece(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Expected Calibration Error and return bin data.

    Returns:
        ece: Expected Calibration Error
        bin_centers: center of each bin
        fraction_positives: observed fraction of positives per bin
        bin_counts: number of samples per bin
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    fraction_positives = np.zeros(n_bins)
    mean_predicted = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = bin_indices == i
        bin_counts[i] = mask.sum()
        if bin_counts[i] > 0:
            fraction_positives[i] = y_true[mask].mean()
            mean_predicted[i] = y_prob[mask].mean()

    # ECE: weighted average of |fraction_positives - mean_predicted|
    total = bin_counts.sum()
    if total > 0:
        ece = np.sum(bin_counts * np.abs(fraction_positives - mean_predicted)) / total
    else:
        ece = 0.0

    return ece, bin_centers, fraction_positives, bin_counts


def compute_tail_ece(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.9
) -> float:
    """Compute ECE restricted to predictions with confidence >= threshold."""
    mask = y_prob >= threshold
    if mask.sum() == 0:
        return 0.0
    return compute_ece(y_true[mask], y_prob[mask], n_bins=10)[0]


def compute_fdr_curve(
    confidence: np.ndarray, thresholds: np.ndarray
) -> tuple[list[float], list[int]]:
    """Compute number of accepted PSMs at each FDR threshold.

    Uses NonParametricFDRControl to estimate FDR from calibrated confidence.

    Args:
        confidence: Array of calibrated confidence scores.
        thresholds: Array of FDR threshold values to evaluate.

    Returns:
        fdr_thresholds: list of threshold values
        n_accepted: list of PSM counts accepted at each threshold
    """
    fdr_control = NonParametricFDRControl()
    fdr_control.fit(pd.DataFrame({"confidence": confidence})["confidence"])

    n_accepted = []
    for thresh in thresholds:
        try:
            cutoff = fdr_control.get_confidence_cutoff(threshold=thresh)
            n_accepted.append(int((confidence >= cutoff).sum()))
        except (ValueError, IndexError):
            # No cutoff found for this threshold
            n_accepted.append(0)

    return thresholds.tolist(), n_accepted


def compute_metrics(
    dataset: CalibrationDataset, config_name: str, dataset_name: str
) -> MetricsResult:
    """Compute all metrics for a single config × dataset pair.

    Args:
        dataset: CalibrationDataset with 'calibrated_confidence' column populated.
        config_name: Name of the ablation configuration.
        dataset_name: Name of the dataset (e.g., 'indomain', 'ood').

    Returns:
        MetricsResult with all computed metrics.
    """
    meta = dataset.metadata
    result = MetricsResult(config_name=config_name, dataset_name=dataset_name)

    y_prob = meta["calibrated_confidence"].values

    # Check if labels are available
    has_labels = "correct" in meta.columns
    if has_labels:
        y_true = meta["correct"].values.astype(int)

        # Discrimination metrics
        try:
            result.roc_auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            result.roc_auc = None

        result.brier_score = brier_score_loss(y_true, y_prob)

        try:
            result.pr_auc = average_precision_score(y_true, y_prob)
        except ValueError:
            result.pr_auc = None

        # Calibration metrics
        ece, bin_centers, frac_pos, bin_counts = compute_ece(y_true, y_prob, n_bins=10)
        result.ece = ece
        result.tail_ece = compute_tail_ece(y_true, y_prob, threshold=0.9)

        result.reliability_bin_centers = bin_centers.tolist()
        result.reliability_fraction_positives = frac_pos.tolist()
        result.reliability_bin_counts = bin_counts.tolist()

        # PR curve data
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        result.pr_recall = recall.tolist()
        result.pr_precision = precision.tolist()

    # FDR curve (label-free)
    fdr_thresholds = np.linspace(0.001, 0.1, 100)
    result.fdr_thresholds, result.fdr_n_accepted = compute_fdr_curve(
        y_prob, fdr_thresholds
    )

    # PSMs at specific FDR thresholds
    fdr_control = NonParametricFDRControl()
    fdr_control.fit(pd.DataFrame({"confidence": y_prob})["confidence"])
    try:
        cutoff_1pct = fdr_control.get_confidence_cutoff(threshold=0.01)
        result.psms_at_1pct_fdr = int((y_prob >= cutoff_1pct).sum())
    except (ValueError, IndexError):
        result.psms_at_1pct_fdr = 0
    try:
        cutoff_5pct = fdr_control.get_confidence_cutoff(threshold=0.05)
        result.psms_at_5pct_fdr = int((y_prob >= cutoff_5pct).sum())
    except (ValueError, IndexError):
        result.psms_at_5pct_fdr = 0

    return result


# ---------------------------------------------------------------------------
# Output Generation
# ---------------------------------------------------------------------------


def save_summary_csv(results: list[MetricsResult], output_dir: Path) -> None:
    """Save scalar metrics to summary.csv."""
    rows = []
    for r in results:
        rows.append(
            {
                "config": r.config_name,
                "dataset": r.dataset_name,
                "roc_auc": r.roc_auc,
                "brier_score": r.brier_score,
                "pr_auc": r.pr_auc,
                "ece": r.ece,
                "tail_ece": r.tail_ece,
                "psms_at_1pct_fdr": r.psms_at_1pct_fdr,
                "psms_at_5pct_fdr": r.psms_at_5pct_fdr,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "summary.csv", index=False)
    logger.info(f"Saved summary.csv with {len(rows)} rows")


def save_reliability_data(results: list[MetricsResult], output_dir: Path) -> None:
    """Save reliability curve data to reliability_data.csv."""
    rows = []
    for r in results:
        if not r.reliability_bin_centers:
            continue
        for _, (center, frac, count) in enumerate(
            zip(
                r.reliability_bin_centers,
                r.reliability_fraction_positives,
                r.reliability_bin_counts,
            )
        ):
            rows.append(
                {
                    "config": r.config_name,
                    "dataset": r.dataset_name,
                    "bin_center": center,
                    "fraction_positives": frac,
                    "bin_count": count,
                }
            )
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "reliability_data.csv", index=False)
        logger.info(f"Saved reliability_data.csv with {len(rows)} rows")


def save_pr_data(results: list[MetricsResult], output_dir: Path) -> None:
    """Save precision-recall curve data to pr_data.csv."""
    rows = []
    for r in results:
        if not r.pr_recall:
            continue
        for recall, precision in zip(r.pr_recall, r.pr_precision):
            rows.append(
                {
                    "config": r.config_name,
                    "dataset": r.dataset_name,
                    "recall": recall,
                    "precision": precision,
                }
            )
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "pr_data.csv", index=False)
        logger.info(f"Saved pr_data.csv with {len(rows)} rows")


def save_fdr_data(results: list[MetricsResult], output_dir: Path) -> None:
    """Save FDR curve data to fdr_data.csv."""
    rows = []
    for r in results:
        if not r.fdr_thresholds:
            continue
        for thresh, n_acc in zip(r.fdr_thresholds, r.fdr_n_accepted):
            rows.append(
                {
                    "config": r.config_name,
                    "dataset": r.dataset_name,
                    "fdr_threshold": thresh,
                    "n_accepted": n_acc,
                }
            )
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "fdr_data.csv", index=False)
        logger.info(f"Saved fdr_data.csv with {len(rows)} rows")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Color scheme for configurations
CONFIG_COLORS = {
    "raw_confidence": "#1f77b4",
    "beam_uncertainty": "#ff7f0e",
    "prosit_irt_intensity": "#2ca02c",
    "full_model": "#d62728",
}

CONFIG_LABELS = {
    "raw_confidence": "Raw confidence",
    "beam_uncertainty": "Beam uncertainty",
    "prosit_irt_intensity": "Prosit/iRT/intensity",
    "full_model": "Full model",
}


def plot_reliability_diagrams(
    results: list[MetricsResult], output_dir: Path, dataset_name: str
) -> None:
    """Plot reliability diagrams for a single dataset, overlaying all configs."""
    dataset_results = [r for r in results if r.dataset_name == dataset_name]

    # Skip if no reliability data
    if not any(r.reliability_bin_centers for r in dataset_results):
        logger.warning(f"No reliability data for {dataset_name}, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    for r in dataset_results:
        if not r.reliability_bin_centers:
            continue
        color = CONFIG_COLORS.get(r.config_name, "#333333")
        label = CONFIG_LABELS.get(r.config_name, r.config_name)
        ax.plot(
            r.reliability_bin_centers,
            r.reliability_fraction_positives,
            "o-",
            color=color,
            label=f"{label} (ECE={r.ece:.3f})",
            markersize=6,
        )

    ax.set_xlabel("Mean predicted confidence", fontsize=12)
    ax.set_ylabel("Fraction of positives", fontsize=12)
    ax.set_title(f"Reliability Diagram — {dataset_name}", fontsize=14)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / f"reliability_{dataset_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_pr_curves(
    results: list[MetricsResult], output_dir: Path, dataset_name: str
) -> None:
    """Plot precision-recall curves for a single dataset, overlaying all configs."""
    dataset_results = [r for r in results if r.dataset_name == dataset_name]

    # Skip if no PR data
    if not any(r.pr_recall for r in dataset_results):
        logger.warning(f"No PR data for {dataset_name}, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for r in dataset_results:
        if not r.pr_recall:
            continue
        color = CONFIG_COLORS.get(r.config_name, "#333333")
        label = CONFIG_LABELS.get(r.config_name, r.config_name)
        ax.plot(
            r.pr_recall,
            r.pr_precision,
            color=color,
            label=f"{label} (PR-AUC={r.pr_auc:.3f})" if r.pr_auc else label,
            linewidth=2,
        )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curves — {dataset_name}", fontsize=14)
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / f"pr_curves_{dataset_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_fdr_curves(
    results: list[MetricsResult], output_dir: Path, dataset_name: str
) -> None:
    """Plot FDR curves for a single dataset, overlaying all configs."""
    dataset_results = [r for r in results if r.dataset_name == dataset_name]

    fig, ax = plt.subplots(figsize=(8, 6))

    for r in dataset_results:
        if not r.fdr_thresholds:
            continue
        color = CONFIG_COLORS.get(r.config_name, "#333333")
        label = CONFIG_LABELS.get(r.config_name, r.config_name)
        ax.plot(
            r.fdr_thresholds,
            r.fdr_n_accepted,
            color=color,
            label=label,
            linewidth=2,
        )

    # Add vertical lines at key thresholds
    ax.axvline(x=0.01, color="gray", linestyle="--", alpha=0.5, label="1% FDR")
    ax.axvline(x=0.05, color="gray", linestyle=":", alpha=0.5, label="5% FDR")

    ax.set_xlabel("FDR threshold", fontsize=12)
    ax.set_ylabel("PSMs accepted", fontsize=12)
    ax.set_title(f"FDR Curves — {dataset_name}", fontsize=14)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 0.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / f"fdr_curves_{dataset_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run feature ablation study for winnow calibrators.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using InstaNovo format
  python scripts/ablation_study.py \\
      --train-spectra data/train/spectra.ipc \\
      --train-predictions data/train/predictions.csv \\
      --indomain-spectra data/test_indomain/spectra.ipc \\
      --indomain-predictions data/test_indomain/predictions.csv \\
      --ood-spectra data/test_ood/spectra.ipc \\
      --ood-predictions data/test_ood/predictions.csv \\
      --output-dir results/ablation

  # Using Winnow format (pre-saved datasets)
  python scripts/ablation_study.py \\
      --train-spectra data/train/ \\
      --indomain-spectra data/test_indomain/ \\
      --ood-spectra data/test_ood/ \\
      --loader-type winnow \\
      --output-dir results/ablation
        """,
    )

    parser.add_argument(
        "--train-spectra",
        type=Path,
        required=True,
        help="Path to training spectrum data (IPC file or Winnow directory)",
    )
    parser.add_argument(
        "--train-predictions",
        type=Path,
        default=None,
        help="Path to training predictions CSV (required for InstaNovo loader)",
    )
    parser.add_argument(
        "--indomain-spectra",
        type=Path,
        required=True,
        help="Path to in-domain test spectrum data",
    )
    parser.add_argument(
        "--indomain-predictions",
        type=Path,
        default=None,
        help="Path to in-domain test predictions CSV",
    )
    parser.add_argument(
        "--ood-spectra",
        type=Path,
        required=True,
        help="Path to out-of-domain test spectrum data",
    )
    parser.add_argument(
        "--ood-predictions",
        type=Path,
        default=None,
        help="Path to out-of-domain test predictions CSV",
    )
    parser.add_argument(
        "--loader-type",
        type=str,
        choices=["instanovo", "winnow"],
        default="instanovo",
        help="Data loader type (default: instanovo)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save output files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for calibrator training (default: 42)",
    )

    return parser.parse_args()


def load_all_datasets(
    args: argparse.Namespace,
) -> tuple[CalibrationDataset, CalibrationDataset, CalibrationDataset]:
    """Load and filter all datasets from CLI arguments.

    Returns:
        Tuple of (train_dataset, indomain_dataset, ood_dataset).
    """
    logger.info("Loading training dataset...")
    train_dataset = filter_dataset(
        load_dataset(args.train_spectra, args.train_predictions, args.loader_type)
    )
    logger.info(f"Training dataset: {len(train_dataset.metadata)} spectra")

    logger.info("Loading in-domain test dataset...")
    indomain_dataset = filter_dataset(
        load_dataset(args.indomain_spectra, args.indomain_predictions, args.loader_type)
    )
    logger.info(f"In-domain test dataset: {len(indomain_dataset.metadata)} spectra")

    logger.info("Loading OOD test dataset...")
    ood_dataset = filter_dataset(
        load_dataset(args.ood_spectra, args.ood_predictions, args.loader_type)
    )
    logger.info(f"OOD test dataset: {len(ood_dataset.metadata)} spectra")

    return train_dataset, indomain_dataset, ood_dataset


def log_metrics_result(result: MetricsResult) -> None:
    """Log key metrics from a MetricsResult."""
    if result.roc_auc is not None:
        logger.info(f"  ROC-AUC: {result.roc_auc:.4f}")
    if result.brier_score is not None:
        logger.info(f"  Brier score: {result.brier_score:.4f}")
    if result.pr_auc is not None:
        logger.info(f"  PR-AUC: {result.pr_auc:.4f}")
    if result.ece is not None:
        logger.info(f"  ECE: {result.ece:.4f}")
    logger.info(f"  PSMs @ 1% FDR: {result.psms_at_1pct_fdr}")
    logger.info(f"  PSMs @ 5% FDR: {result.psms_at_5pct_fdr}")


def evaluate_on_test_datasets(
    calibrator: ProbabilityCalibrator,
    test_datasets: list[tuple[CalibrationDataset, str]],
    config_name: str,
) -> list[MetricsResult]:
    """Evaluate a calibrator on multiple test datasets and return results."""
    results = []
    for test_dataset, dataset_name in test_datasets:
        logger.info(f"Predicting on {dataset_name} dataset...")
        test_copy = CalibrationDataset(
            metadata=test_dataset.metadata.copy(),
            predictions=test_dataset.predictions,
        )
        calibrator.predict(test_copy)

        logger.info(f"Computing metrics for {dataset_name}...")
        result = compute_metrics(test_copy, config_name, dataset_name)
        results.append(result)
        log_metrics_result(result)

    return results


def save_all_outputs(results: list[MetricsResult], output_dir: Path) -> None:
    """Save all CSV and plot outputs."""
    logger.info(f"\n{'=' * 60}")
    logger.info("Saving outputs...")
    logger.info(f"{'=' * 60}")

    save_summary_csv(results, output_dir)
    save_reliability_data(results, output_dir)
    save_pr_data(results, output_dir)
    save_fdr_data(results, output_dir)

    for dataset_name in ["indomain", "ood"]:
        plot_reliability_diagrams(results, output_dir, dataset_name)
        plot_pr_curves(results, output_dir, dataset_name)
        plot_fdr_curves(results, output_dir, dataset_name)


def main() -> None:
    """Run the feature ablation study."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    train_dataset, indomain_dataset, ood_dataset = load_all_datasets(args)
    test_datasets = [(indomain_dataset, "indomain"), (ood_dataset, "ood")]

    ablation_configs = get_ablation_configs()
    logger.info(f"Running {len(ablation_configs)} ablation configurations")

    all_results: list[MetricsResult] = []

    for config_name, feature_factory in ablation_configs.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Configuration: {config_name}")
        logger.info(f"{'=' * 60}")

        calibrator = ProbabilityCalibrator(
            seed=args.seed,
            features=feature_factory(),
            hidden_layer_sizes=(50, 50),
            learning_rate_init=0.001,
            alpha=0.0001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
        )
        logger.info(
            f"Features: {calibrator.features if calibrator.features else '(none)'}"
        )

        logger.info("Fitting calibrator...")
        history = calibrator.fit(train_dataset)
        logger.info(f"Training completed in {history.n_iter} iterations")
        logger.info(f"Final training loss: {history.final_training_loss:.6f}")
        if history.final_validation_score is not None:
            logger.info(f"Final validation score: {history.final_validation_score:.6f}")

        results = evaluate_on_test_datasets(calibrator, test_datasets, config_name)
        all_results.extend(results)

    save_all_outputs(all_results, args.output_dir)
    logger.info(f"\nAblation study complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
