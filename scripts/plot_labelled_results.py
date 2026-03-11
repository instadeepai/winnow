#!/usr/bin/env python3
"""Visualization script for calibration quality and FDR estimation evaluation.

This script generates comprehensive plots to evaluate:
- Calibration quality: reliability diagrams, confidence distributions, ECE
- FDR estimation quality: precision-recall curves, FDR calibration, PSM counts

Accepts either:
1. A directory containing winnow output files (metadata.csv + preds_and_fdr_metrics.csv)
2. A single CSV file with all required columns

Required columns:
- `confidence`: Original (pre-calibration) confidence scores
- `calibrated_confidence`: Post-calibration confidence scores
- `correct`: Binary correctness labels (computed from sequence matching if not present)

For full FDR mode, additionally requires:
- `sequence`: Ground-truth peptide sequence
- `prediction`: Predicted peptide sequence
"""

from __future__ import annotations

import argparse
import ast
import logging
import warnings
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from winnow.fdr.database_grounded import DatabaseGroundedFDRControl
from winnow.fdr.nonparametric import NonParametricFDRControl

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set seaborn theme
sns.set_theme(style="white", palette="colorblind", context="paper", font_scale=1.5)

# Default residue masses for DatabaseGroundedFDRControl
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

# Plot styling
COLORS = {
    "fairy": "#FFCAE9",
    "magenta": "#8E5572",
    "ash": "#BBC5AA",
    "ebony": "#5A6650",
    "sky": "#7FC8F8",
    "navy": "#3C81AE",
}

# Semantic color mappings
COLOR_RAW = COLORS["navy"]
COLOR_CALIBRATED = COLORS["magenta"]
COLOR_CORRECT = COLORS["ebony"]
COLOR_INCORRECT = COLORS["magenta"]
COLOR_NONPARAMETRIC = COLORS["sky"]
COLOR_DATABASE = COLORS["ebony"]


# ---------------------------------------------------------------------------
# Data Loading and Validation
# ---------------------------------------------------------------------------


def parse_list_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Parse a column from string representation to list if needed.

    Handles both Python list literals (e.g., "['A', 'B']") and the format
    saved by pandas when writing lists to CSV.

    Args:
        df: DataFrame with the column.
        col: Column name to parse.

    Returns:
        DataFrame with parsed column.
    """
    if col not in df.columns:
        return df

    # Check if already a list
    first_val = df[col].iloc[0]
    if isinstance(first_val, list):
        return df

    # Skip if not a string
    if not isinstance(first_val, str):
        return df

    # Try to parse as list literal
    if first_val.startswith("["):
        df[col] = df[col].apply(ast.literal_eval)

    return df


def compute_correct_column(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 'correct' column from sequence and prediction matching.

    Uses the same matching logic as DatabaseGroundedFDRControl.

    Args:
        df: DataFrame with 'sequence' and 'prediction' columns.

    Returns:
        DataFrame with 'correct' column added.
    """
    from instanovo.utils.metrics import Metrics
    from instanovo.utils.residues import ResidueSet

    metrics = Metrics(
        residue_set=ResidueSet(residue_masses=DEFAULT_RESIDUE_MASSES),
        isotope_error_range=(0, 1),
    )

    def is_correct(row: pd.Series) -> bool:
        seq = row["sequence"]
        pred = row["prediction"]

        # Handle string inputs by splitting
        if isinstance(seq, str):
            seq = metrics._split_peptide(seq)
        if isinstance(pred, str):
            pred = metrics._split_peptide(pred)

        num_matches = metrics._novor_match(seq, pred)
        return num_matches == len(seq) == len(pred)

    df["correct"] = df.apply(is_correct, axis=1)
    return df


def load_winnow_output(input_dir: Path) -> pd.DataFrame:
    """Load and merge winnow output files (metadata.csv + preds_and_fdr_metrics.csv).

    Args:
        input_dir: Directory containing the output files.

    Returns:
        Merged DataFrame with all columns.
    """
    metadata_path = input_dir / "metadata.csv"
    preds_path = input_dir / "preds_and_fdr_metrics.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {input_dir}")
    if not preds_path.exists():
        raise FileNotFoundError(f"preds_and_fdr_metrics.csv not found in {input_dir}")

    logger.info(f"Loading metadata from {metadata_path}")
    metadata_df = pd.read_csv(metadata_path)

    logger.info(f"Loading predictions from {preds_path}")
    preds_df = pd.read_csv(preds_path)

    # Drop duplicate index columns (pandas adds 'Unnamed: 0' when saving with index=True)
    for df in [metadata_df, preds_df]:
        unnamed_cols = [c for c in df.columns if c.startswith("Unnamed:")]
        if unnamed_cols:
            df.drop(columns=unnamed_cols, inplace=True)

    # Merge on spectrum_id
    logger.info("Merging datasets on spectrum_id")
    df = metadata_df.merge(preds_df, on="spectrum_id", how="inner")
    logger.info(f"Merged dataset: {len(df)} rows")

    return df


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load calibrated dataset from file or directory.

    If input_path is a directory, loads and merges winnow output files.
    If input_path is a file, loads it directly as a CSV.

    Args:
        input_path: Path to CSV file or directory containing winnow outputs.

    Returns:
        DataFrame with calibrated dataset.
    """
    if input_path.is_dir():
        df = load_winnow_output(input_path)
    else:
        logger.info(f"Loading dataset from {input_path}")
        df = pd.read_csv(input_path)
        # Drop unnamed index columns
        unnamed_cols = [c for c in df.columns if c.startswith("Unnamed:")]
        if unnamed_cols:
            df.drop(columns=unnamed_cols, inplace=True)
        logger.info(f"Loaded {len(df)} rows")

    # Parse list columns that may have been saved as strings
    for col in ["sequence", "prediction"]:
        df = parse_list_column(df, col)

    return df


def validate_columns(df: pd.DataFrame, mode: Literal["calibration", "full"]) -> None:
    """Validate that required columns are present.

    Args:
        df: DataFrame to validate.
        mode: Visualization mode.

    Raises:
        ValueError: If required columns are missing.
    """
    # For calibration mode, we need correct column (will compute if missing)
    required_calibration = ["confidence", "calibrated_confidence"]
    required_full = required_calibration + ["sequence", "prediction"]

    required = required_full if mode == "full" else required_calibration
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns for mode '{mode}': {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def ensure_correct_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the 'correct' column exists, computing it if necessary.

    Args:
        df: DataFrame that should have sequence and prediction columns.

    Returns:
        DataFrame with 'correct' column.
    """
    if "correct" in df.columns:
        return df

    if "sequence" not in df.columns or "prediction" not in df.columns:
        raise ValueError(
            "Cannot compute 'correct' column: 'sequence' and 'prediction' columns required"
        )

    logger.info("Computing 'correct' column from sequence matching...")
    df = compute_correct_column(df)
    correct_count = df["correct"].sum()
    accuracy_pct = 100 * correct_count / len(df)
    logger.info(f"Correct predictions: {correct_count}/{len(df)} ({accuracy_pct:.1f}%)")

    return df


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------


def compute_ece(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Expected Calibration Error and return bin data.

    Args:
        y_true: Binary correctness labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins for calibration.

    Returns:
        ece: Expected Calibration Error
        bin_centers: Center of each bin
        fraction_positives: Observed fraction of positives per bin
        bin_counts: Number of samples per bin
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

    total = bin_counts.sum()
    if total > 0:
        ece = np.sum(bin_counts * np.abs(fraction_positives - mean_predicted)) / total
    else:
        ece = 0.0

    return ece, bin_centers, fraction_positives, bin_counts


# ---------------------------------------------------------------------------
# Calibration Plots
# ---------------------------------------------------------------------------


def plot_reliability_diagram(
    df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
) -> None:
    """Plot reliability diagrams before and after calibration.

    Args:
        df: DataFrame with confidence, calibrated_confidence, correct columns.
        output_dir: Directory to save plots.
        formats: List of output formats (e.g., ['png', 'pdf']).
    """
    y_true = df["correct"].values.astype(int)
    conf_before = df["confidence"].values
    conf_after = df["calibrated_confidence"].values

    ece_before, centers_before, frac_before, _ = compute_ece(y_true, conf_before)
    ece_after, centers_after, frac_after, _ = compute_ece(y_true, conf_after)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect", linewidth=1)

    # Raw confidence
    ax.plot(
        centers_before,
        frac_before,
        "o-",
        color=COLOR_RAW,
        label=f"Raw (ECE={ece_before:.3f})",
        markersize=6,
        linewidth=2,
    )

    # Calibrated confidence
    ax.plot(
        centers_after,
        frac_after,
        "s-",
        color=COLOR_CALIBRATED,
        label=f"Calibrated (ECE={ece_after:.3f})",
        markersize=6,
        linewidth=2,
    )

    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability Diagram")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    sns.despine()

    plt.tight_layout()
    save_figure(fig, output_dir, "reliability_diagram", formats)
    plt.close(fig)


def plot_confidence_distributions(
    df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
) -> None:
    """Plot confidence distributions for correct vs incorrect PSMs.

    Args:
        df: DataFrame with confidence, calibrated_confidence, correct columns.
        output_dir: Directory to save plots.
        formats: List of output formats.
    """
    correct_mask = df["correct"].astype(bool)
    conf_before = df["confidence"].values
    conf_after = df["calibrated_confidence"].values

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Raw confidence
    ax = axes[0]
    ax.hist(
        conf_before[correct_mask],
        bins=50,
        alpha=0.7,
        color=COLOR_CORRECT,
        label="Correct",
        density=True,
    )
    ax.hist(
        conf_before[~correct_mask],
        bins=50,
        alpha=0.7,
        color=COLOR_INCORRECT,
        label="Incorrect",
        density=True,
    )
    ax.set_xlabel("Raw Confidence")
    ax.set_ylabel("Density")
    ax.set_title("Raw Confidence")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    sns.despine(ax=ax)

    # Calibrated confidence
    ax = axes[1]
    ax.hist(
        conf_after[correct_mask],
        bins=50,
        alpha=0.7,
        color=COLOR_CORRECT,
        label="Correct",
        density=True,
    )
    ax.hist(
        conf_after[~correct_mask],
        bins=50,
        alpha=0.7,
        color=COLOR_INCORRECT,
        label="Incorrect",
        density=True,
    )
    ax.set_xlabel("Calibrated Confidence")
    ax.set_ylabel("Density")
    ax.set_title("Calibrated Confidence")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    sns.despine(ax=ax)

    plt.tight_layout()
    save_figure(fig, output_dir, "confidence_distributions", formats)
    plt.close(fig)


def plot_ece_comparison(
    df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
) -> None:
    """Plot ECE comparison bar chart.

    Args:
        df: DataFrame with confidence, calibrated_confidence, correct columns.
        output_dir: Directory to save plots.
        formats: List of output formats.
    """
    y_true = df["correct"].values.astype(int)
    ece_before, _, _, _ = compute_ece(y_true, df["confidence"].values)
    ece_after, _, _, _ = compute_ece(y_true, df["calibrated_confidence"].values)

    fig, ax = plt.subplots(figsize=(5, 4))

    bars = ax.bar(
        ["Raw", "Calibrated"],
        [ece_before, ece_after],
        color=[COLOR_RAW, COLOR_CALIBRATED],
        edgecolor="black",
        linewidth=0.5,
    )

    # Add value labels on bars
    for bar, val in zip(bars, [ece_before, ece_after]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylabel("Expected Calibration Error (ECE)")
    ax.set_title("ECE Comparison")
    ax.set_ylim(0, max(ece_before, ece_after) * 1.2)
    sns.despine()

    plt.tight_layout()
    save_figure(fig, output_dir, "ece_comparison", formats)
    plt.close(fig)


# ---------------------------------------------------------------------------
# FDR Plots
# ---------------------------------------------------------------------------


def compute_fdr_data(
    df: pd.DataFrame,
    confidence_col: str,
) -> tuple[NonParametricFDRControl, DatabaseGroundedFDRControl]:
    """Compute FDR using both non-parametric and database-grounded methods.

    Args:
        df: DataFrame with required columns.
        confidence_col: Name of confidence column to use.

    Returns:
        Tuple of (nonparametric_fdr, database_fdr) fitted controllers.
    """
    # Non-parametric FDR
    np_fdr = NonParametricFDRControl()
    np_fdr.fit(df[confidence_col])

    # Database-grounded FDR
    db_fdr = DatabaseGroundedFDRControl(
        confidence_feature=confidence_col,
        residue_masses=DEFAULT_RESIDUE_MASSES,
        drop=0,  # Don't drop any for visualization
    )
    db_fdr.fit(df[[confidence_col, "sequence", "prediction"]].copy())

    return np_fdr, db_fdr


def plot_precision_recall_curves(
    df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
) -> None:
    """Plot precision-recall curves for raw and calibrated confidence.

    Args:
        df: DataFrame with required columns.
        output_dir: Directory to save plots.
        formats: List of output formats.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    for conf_col, label, color in [
        ("confidence", "Raw", COLOR_RAW),
        ("calibrated_confidence", "Calibrated", COLOR_CALIBRATED),
    ]:
        sorted_df = df.sort_values(by=conf_col, ascending=False)
        correct = sorted_df["correct"].values.astype(int)
        precision = np.cumsum(correct) / np.arange(1, len(correct) + 1)
        recall = np.cumsum(correct) / correct.sum()
        ax.plot(recall, precision, color=color, linewidth=2, label=label)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")
    sns.despine()

    plt.tight_layout()
    save_figure(fig, output_dir, "pr_curves", formats)
    plt.close(fig)


def plot_fdr_calibration(
    df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    focus_range: bool = False,
) -> None:
    """Plot estimated FDR vs actual (database-grounded) FDR.

    Args:
        df: DataFrame with required columns.
        output_dir: Directory to save plots.
        formats: List of output formats.
        focus_range: If True, focus on 0-0.1 FDR range.
    """
    # Sort by calibrated confidence
    sorted_df = df.sort_values(by="calibrated_confidence", ascending=False)
    correct = sorted_df["correct"].values.astype(int)

    # True FDR at each rank
    true_fdr = 1 - (np.cumsum(correct) / np.arange(1, len(correct) + 1))

    # Estimated FDR: use precomputed if available, else compute
    if "psm_fdr" in sorted_df.columns:
        estimated_fdr = sorted_df["psm_fdr"].values
    else:
        estimated_fdr_individual = 1 - sorted_df["calibrated_confidence"].values
        estimated_fdr = np.cumsum(estimated_fdr_individual) / np.arange(
            1, len(estimated_fdr_individual) + 1
        )

    fig, ax = plt.subplots(figsize=(6, 6))

    if focus_range:
        mask = true_fdr <= 0.1
        max_val = 0.1
        suffix = "_focused"
        title_suffix = " (0–10%)"
    else:
        mask = np.ones(len(true_fdr), dtype=bool)
        # Use actual data extent, not always 1.0
        max_val = max(true_fdr.max(), estimated_fdr.max())
        max_val = min(max_val * 1.05, 1.0)  # Add 5% margin, cap at 1.0
        suffix = ""
        title_suffix = ""

    # Perfect calibration line
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Perfect", linewidth=1)

    ax.scatter(
        true_fdr[mask],
        estimated_fdr[mask],
        alpha=0.3,
        s=2,
        color=COLOR_CALIBRATED,
    )

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel("True FDR (Database-Grounded)")
    ax.set_ylabel("Estimated FDR (Non-Parametric)")
    ax.set_title(f"FDR Calibration{title_suffix}")
    ax.set_aspect("equal")
    sns.despine()

    plt.tight_layout()
    save_figure(fig, output_dir, f"fdr_calibration{suffix}", formats)
    plt.close(fig)


def plot_psms_at_fdr_thresholds(
    df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
) -> None:
    """Plot PSM counts at different FDR thresholds.

    Args:
        df: DataFrame with required columns.
        output_dir: Directory to save plots.
        formats: List of output formats.
    """
    thresholds = [0.01, 0.05, 0.10]
    threshold_labels = ["1%", "5%", "10%"]

    # Compute FDR controllers
    np_fdr, db_fdr = compute_fdr_data(df, "calibrated_confidence")

    np_counts = []
    db_counts = []
    conf = df["calibrated_confidence"].values

    # Suppress warnings about FDR thresholds above fitted range
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="FDR threshold .* is above the range")
        for thresh in thresholds:
            # Non-parametric
            try:
                cutoff = np_fdr.get_confidence_cutoff(threshold=thresh)
                np_counts.append(int((conf >= cutoff).sum()))
            except (ValueError, IndexError):
                np_counts.append(0)

            # Database-grounded
            try:
                cutoff = db_fdr.get_confidence_cutoff(threshold=thresh)
                db_counts.append(int((conf >= cutoff).sum()))
            except (ValueError, IndexError):
                db_counts.append(0)

    fig, ax = plt.subplots(figsize=(6, 5))

    x = np.arange(len(thresholds))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        np_counts,
        width,
        label="Non-Parametric",
        color=COLOR_NONPARAMETRIC,
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        db_counts,
        width,
        label="Database-Grounded",
        color=COLOR_DATABASE,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height):,}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_xlabel("FDR Threshold")
    ax.set_ylabel("PSMs Accepted")
    ax.set_title("PSMs at FDR Thresholds")
    ax.set_xticks(x)
    ax.set_xticklabels(threshold_labels)
    ax.legend(loc="upper left")
    sns.despine()

    plt.tight_layout()
    save_figure(fig, output_dir, "psms_at_fdr_thresholds", formats)
    plt.close(fig)


def compute_psms_at_thresholds(
    fdr_control: NonParametricFDRControl | DatabaseGroundedFDRControl,
    confidence: np.ndarray,
    thresholds: np.ndarray,
) -> list[int]:
    """Compute number of PSMs accepted at each FDR threshold.

    Args:
        fdr_control: Fitted FDR control object.
        confidence: Array of confidence scores.
        thresholds: Array of FDR thresholds.

    Returns:
        List of PSM counts at each threshold.
    """
    psms = []
    for thresh in thresholds:
        try:
            cutoff = fdr_control.get_confidence_cutoff(threshold=thresh)
            psms.append(int((confidence >= cutoff).sum()))
        except (ValueError, IndexError):
            psms.append(psms[-1] if psms else 0)
    return psms


def plot_fdr_curves(
    df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    focus_range: bool = False,
) -> None:
    """Plot FDR curves comparing non-parametric vs database-grounded.

    Args:
        df: DataFrame with required columns.
        output_dir: Directory to save plots.
        formats: List of output formats.
        focus_range: If True, focus on 0-0.1 FDR range.
    """
    np_fdr, db_fdr = compute_fdr_data(df, "calibrated_confidence")
    conf = df["calibrated_confidence"].values

    max_fdr = 0.1 if focus_range else 0.5
    fdr_thresholds = np.linspace(0.001, max_fdr, 100)
    suffix = "_focused" if focus_range else ""
    title_suffix = " (0–10% FDR)" if focus_range else ""

    # Suppress warnings about FDR thresholds above fitted range
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="FDR threshold .* is above the range")
        np_psms = compute_psms_at_thresholds(np_fdr, conf, fdr_thresholds)
        db_psms = compute_psms_at_thresholds(db_fdr, conf, fdr_thresholds)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        fdr_thresholds,
        np_psms,
        color=COLOR_NONPARAMETRIC,
        linewidth=2,
        label="Non-Parametric",
    )
    ax.plot(
        fdr_thresholds,
        db_psms,
        color=COLOR_DATABASE,
        linewidth=2,
        label="Database-Grounded",
    )

    # Add reference lines at 1% and 5% FDR
    ax.axvline(x=0.01, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0.05, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("FDR Threshold")
    ax.set_ylabel("PSMs Accepted")
    ax.set_title(f"FDR Curves{title_suffix}")
    ax.legend(loc="lower right")
    sns.despine()

    plt.tight_layout()
    save_figure(fig, output_dir, f"fdr_curves{suffix}", formats)
    plt.close(fig)


def plot_fdr_vs_confidence(
    df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    focus_range: bool = False,
) -> None:
    """Plot FDR vs confidence score.

    Args:
        df: DataFrame with required columns.
        output_dir: Directory to save plots.
        formats: List of output formats.
        focus_range: If True, focus on 0-0.1 FDR range.
    """
    # Sort by calibrated confidence descending
    sorted_df = df.sort_values(by="calibrated_confidence", ascending=False)
    conf = sorted_df["calibrated_confidence"].values
    correct = sorted_df["correct"].values.astype(int)

    # True FDR at each rank (database-grounded)
    true_fdr = 1 - (np.cumsum(correct) / np.arange(1, len(correct) + 1))

    # Estimated FDR: use precomputed if available, else compute
    if "psm_fdr" in sorted_df.columns:
        estimated_fdr = sorted_df["psm_fdr"].values
    else:
        estimated_fdr_individual = 1 - conf
        estimated_fdr = np.cumsum(estimated_fdr_individual) / np.arange(
            1, len(estimated_fdr_individual) + 1
        )

    fig, ax = plt.subplots(figsize=(6, 5))

    suffix = "_focused" if focus_range else ""
    title_suffix = " (0–10% FDR)" if focus_range else ""

    if focus_range:
        # Only show points where both FDRs are <= 0.1
        mask = (true_fdr <= 0.1) | (estimated_fdr <= 0.1)
    else:
        mask = np.ones(len(conf), dtype=bool)

    ax.plot(
        conf[mask],
        true_fdr[mask],
        color=COLOR_DATABASE,
        linewidth=2,
        label="Database-Grounded",
    )
    ax.plot(
        conf[mask],
        estimated_fdr[mask],
        color=COLOR_NONPARAMETRIC,
        linewidth=2,
        label="Non-Parametric",
    )

    # Reference lines
    ax.axhline(y=0.01, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=0.05, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    # Determine y limit based on actual data extent
    if focus_range:
        y_max = 0.1
    else:
        y_max = min(max(true_fdr[mask].max(), estimated_fdr[mask].max()) * 1.1, 1.0)

    ax.set_xlim(conf[mask].min() if len(conf[mask]) > 0 else 0, 1)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("FDR")
    ax.set_title(f"FDR vs Confidence{title_suffix}")
    ax.legend(loc="upper right")
    sns.despine()

    plt.tight_layout()
    save_figure(fig, output_dir, f"fdr_vs_confidence{suffix}", formats)
    plt.close(fig)


def compute_qvalues(
    fdr_values: np.ndarray,
) -> np.ndarray:
    """Compute q-values from FDR values.

    Q-values are the minimum FDR at which a PSM would be accepted.
    Computed by taking the cumulative minimum from the end.

    Args:
        fdr_values: Array of FDR values (sorted by decreasing confidence).

    Returns:
        Array of q-values.
    """
    # Traverse from lowest score to highest, keeping track of minimum FDR seen
    q_values = np.zeros_like(fdr_values)
    fdr_min = float("inf")

    for i in range(len(fdr_values) - 1, -1, -1):
        if fdr_values[i] < fdr_min:
            fdr_min = fdr_values[i]
        q_values[i] = fdr_min

    return q_values


def compute_psms_at_qvalue_thresholds(
    q_values: np.ndarray,
    thresholds: np.ndarray,
) -> list[int]:
    """Compute number of PSMs accepted at each q-value threshold.

    Args:
        q_values: Array of q-values (sorted by decreasing confidence).
        thresholds: Array of q-value thresholds.

    Returns:
        List of PSM counts at each threshold.
    """
    psms = []
    for thresh in thresholds:
        # Count PSMs with q-value <= threshold
        count = int((q_values <= thresh).sum())
        psms.append(count)
    return psms


def plot_qvalue_curves(
    df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    focus_range: bool = False,
) -> None:
    """Plot q-value curves comparing non-parametric vs database-grounded.

    Args:
        df: DataFrame with required columns.
        output_dir: Directory to save plots.
        formats: List of output formats.
        focus_range: If True, focus on 0-0.1 q-value range.
    """
    # Sort by calibrated confidence descending
    sorted_df = df.sort_values(by="calibrated_confidence", ascending=False)
    correct = sorted_df["correct"].values.astype(int)

    # Database-grounded FDR (true FDR) and q-values
    db_fdr = 1 - (np.cumsum(correct) / np.arange(1, len(correct) + 1))
    db_qvalues = compute_qvalues(db_fdr)

    # Non-parametric q-values: use precomputed if available, else compute
    if "psm_q_value" in sorted_df.columns:
        np_qvalues = sorted_df["psm_q_value"].values
    else:
        conf = sorted_df["calibrated_confidence"].values
        np_fdr = np.cumsum(1 - conf) / np.arange(1, len(conf) + 1)
        np_qvalues = compute_qvalues(np_fdr)

    max_qval = 0.1 if focus_range else 0.5
    qval_thresholds = np.linspace(0.001, max_qval, 100)
    suffix = "_focused" if focus_range else ""
    title_suffix = " (0–10%)" if focus_range else ""

    np_psms = compute_psms_at_qvalue_thresholds(np_qvalues, qval_thresholds)
    db_psms = compute_psms_at_qvalue_thresholds(db_qvalues, qval_thresholds)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        qval_thresholds,
        np_psms,
        color=COLOR_NONPARAMETRIC,
        linewidth=2,
        label="Non-Parametric",
    )
    ax.plot(
        qval_thresholds,
        db_psms,
        color=COLOR_DATABASE,
        linewidth=2,
        label="Database-Grounded",
    )

    # Add reference lines at 1% and 5% q-value
    ax.axvline(x=0.01, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0.05, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Q-value Threshold")
    ax.set_ylabel("PSMs Accepted")
    ax.set_title(f"Q-value Curves{title_suffix}")
    ax.legend(loc="lower right")
    sns.despine()

    plt.tight_layout()
    save_figure(fig, output_dir, f"qvalue_curves{suffix}", formats)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output Utilities
# ---------------------------------------------------------------------------


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    name: str,
    formats: list[str],
) -> None:
    """Save figure in specified formats.

    Args:
        fig: Matplotlib figure to save.
        output_dir: Directory to save to.
        name: Base filename (without extension).
        formats: List of formats (e.g., ['png', 'pdf']).
    """
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=150, bbox_inches="tight", format=fmt)
        logger.info(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize calibration quality and FDR estimation performance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using winnow output directory (contains metadata.csv + preds_and_fdr_metrics.csv)
  python scripts/plot_labelled_results.py \\
      --input results/winnow_output/ \\
      --output-dir results/plots \\
      --mode full

  # Using a single CSV file
  python scripts/plot_labelled_results.py \\
      --input results/calibrated_dataset.csv \\
      --output-dir results/plots \\
      --mode calibration

  # Output both PNG and PDF
  python scripts/plot_labelled_results.py \\
      --input results/winnow_output/ \\
      --output-dir results/plots \\
      --mode full \\
      --format png pdf
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to winnow output directory or single CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for output plots",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["calibration", "full"],
        default="calibration",
        help="Visualization mode: 'calibration' (quality only) or 'full' (+ FDR)",
    )
    parser.add_argument(
        "--format",
        type=str,
        nargs="+",
        choices=["png", "pdf"],
        default=["png"],
        help="Output format(s): png, pdf, or both",
    )

    return parser.parse_args()


def main() -> None:
    """Run the visualization script."""
    args = parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # Load and validate data
    df = load_dataset(args.input)
    validate_columns(df, args.mode)

    # Ensure correct column exists (compute from sequence matching if needed)
    df = ensure_correct_column(df)

    formats = args.format
    logger.info(f"Output formats: {formats}")

    # Generate calibration plots
    logger.info("Generating calibration quality plots...")
    plot_reliability_diagram(df, args.output_dir, formats)
    plot_confidence_distributions(df, args.output_dir, formats)
    plot_ece_comparison(df, args.output_dir, formats)

    # Generate FDR plots if in full mode
    if args.mode == "full":
        logger.info("Generating FDR estimation plots...")
        plot_precision_recall_curves(df, args.output_dir, formats)

        # Full range plots
        plot_fdr_calibration(df, args.output_dir, formats, focus_range=False)
        plot_fdr_curves(df, args.output_dir, formats, focus_range=False)
        plot_fdr_vs_confidence(df, args.output_dir, formats, focus_range=False)
        plot_qvalue_curves(df, args.output_dir, formats, focus_range=False)

        # Focused range plots (0-0.1)
        plot_fdr_calibration(df, args.output_dir, formats, focus_range=True)
        plot_fdr_curves(df, args.output_dir, formats, focus_range=True)
        plot_fdr_vs_confidence(df, args.output_dir, formats, focus_range=True)
        plot_qvalue_curves(df, args.output_dir, formats, focus_range=True)

        plot_psms_at_fdr_thresholds(df, args.output_dir, formats)

    logger.info(f"\nVisualization complete! Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
