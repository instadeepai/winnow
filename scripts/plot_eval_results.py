"""Generate publication-quality evaluation plots from ``winnow predict`` outputs.

Supports both annotated (database-grounded) and raw (proteome-hit) evaluation
modes, producing six plots per project: precision-recall, FDR run, true vs
estimated FDR (full + zoomed), probability calibration, and before/after score
histograms.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from rich.logging import RichHandler

from winnow.fdr.nonparametric import NonParametricFDRControl

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    logger.addHandler(RichHandler())

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

# ---------------------------------------------------------------------------
# Dataset display names
# ---------------------------------------------------------------------------
DATASET_DISPLAY_NAMES: dict[str, str] = {
    "gluc": "HeLa degradome",
    "helaqc": "HeLa single shot",
    "herceptin": "Herceptin",
    "immuno": "Immunopeptidomics-1",
    "sbrodae": "Scalindua brodae",
    "snakevenoms": "Snake venomics",
    "tplantibodies": "Therapeutic nanobodies",
    "woundfluids": "Wound exudates",
    "PXD014877": "C. elegans",
    "PXD023064": "Immunopeptidomics-2",
    "PXD009935": "Immunopeptidomics-3",
    "Astral": "Astral E. coli",
}

# Fixed colour indices from the colorblind palette for correct/incorrect
_PALETTE = sns.color_palette("colorblind")
_CORRECT_COLOUR = _PALETTE[0]
_INCORRECT_COLOUR = _PALETTE[3]
_MAIN_LINE_COLOUR = _PALETTE[1]
_IDEAL_LINE_COLOUR = "grey"

sns.set_theme(style="white", palette="colorblind", context="paper", font_scale=1.5)


def _display_name(key: str) -> str:
    """Look up the publication-ready display name for a dataset key."""
    return DATASET_DISPLAY_NAMES.get(key, key)


def _ground_truth_qualifier(eval_type: str) -> str:
    """Return the appropriate ground truth qualifier string."""
    if eval_type in ("annotated", "labelled"):
        return "database label"
    return "proteome hit"


def _save_fig(fig: plt.Figure, base_path: Path) -> None:
    """Save figure as both PNG and PDF."""
    fig.savefig(f"{base_path}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"{base_path}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# PR curve (non-standard cumulative definition)
# ---------------------------------------------------------------------------
def _compute_precision_recall(df: pd.DataFrame) -> pd.DataFrame:
    """Non-standard cumulative PR curve matching the codebase convention."""
    sorted_df = df.sort_values("calibrated_confidence", ascending=False)
    labels = sorted_df["correct"].values
    cum_correct = np.cumsum(labels)
    n = len(labels)
    precision = cum_correct / np.arange(1, n + 1)
    recall = cum_correct / n
    return pd.DataFrame({"precision": precision, "recall": recall})


def plot_precision_recall(
    df: pd.DataFrame,
    project: str,
    eval_type: str,
    output_dir: Path,
) -> None:
    """Plot precision-recall curve."""
    display = _display_name(project)
    qualifier = _ground_truth_qualifier(eval_type)
    pr = _compute_precision_recall(df)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pr["recall"], pr["precision"], color=_MAIN_LINE_COLOUR, lw=1.5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision\u2013recall curve \u2014 {display}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.text(
        0.02,
        0.02,
        f"Ground truth: {qualifier}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )
    fig.tight_layout()
    _save_fig(fig, output_dir / f"pr_curve_{project}")


# ---------------------------------------------------------------------------
# FDR run plot
# ---------------------------------------------------------------------------
def plot_fdr_run(
    df: pd.DataFrame,
    project: str,
    eval_type: str,
    output_dir: Path,
) -> None:
    """Plot calibrated confidence vs estimated PSM FDR."""
    display = _display_name(project)

    fdr_ctrl = NonParametricFDRControl()
    fdr_ctrl.fit(dataset=df["calibrated_confidence"])
    fdr_df = fdr_ctrl.add_psm_fdr(df.copy(), confidence_col="calibrated_confidence")
    fdr_df = fdr_df.sort_values("calibrated_confidence")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        fdr_df["calibrated_confidence"].values,
        fdr_df["psm_fdr"].values,
        color=_MAIN_LINE_COLOUR,
        lw=1.5,
    )
    ax.set_xlabel("Calibrated confidence")
    ax.set_ylabel("Estimated PSM FDR")
    ax.set_title(f"FDR run plot \u2014 {display}")
    fig.tight_layout()
    _save_fig(fig, output_dir / f"fdr_run_{project}")


# ---------------------------------------------------------------------------
# True FDR vs estimated FDR
# ---------------------------------------------------------------------------
def _compute_true_vs_estimated_fdr(df: pd.DataFrame) -> pd.DataFrame:
    """Compute true and estimated FDR arrays, sorted by confidence descending."""
    sorted_df = df.sort_values("calibrated_confidence", ascending=False).reset_index(
        drop=True
    )
    labels = sorted_df["correct"].values.astype(float)
    ranks = np.arange(1, len(labels) + 1)

    true_fdr = 1.0 - np.cumsum(labels) / ranks

    fdr_ctrl = NonParametricFDRControl()
    fdr_ctrl.fit(dataset=sorted_df["calibrated_confidence"])
    with_fdr = fdr_ctrl.add_psm_fdr(sorted_df, confidence_col="calibrated_confidence")

    return pd.DataFrame(
        {
            "estimated_fdr": with_fdr["psm_fdr"].values,
            "true_fdr": true_fdr,
        }
    )


def plot_true_vs_estimated_fdr(
    df: pd.DataFrame,
    project: str,
    eval_type: str,
    output_dir: Path,
    *,
    zoomed: bool = False,
) -> None:
    """Plot true FDR vs estimated FDR."""
    display = _display_name(project)
    qualifier = _ground_truth_qualifier(eval_type)
    fdr_data = _compute_true_vs_estimated_fdr(df)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        fdr_data["estimated_fdr"],
        fdr_data["true_fdr"],
        color=_MAIN_LINE_COLOUR,
        lw=1.5,
        label="Observed",
    )
    lim = 0.1 if zoomed else 1.0
    ax.plot([0, lim], [0, lim], ls="--", color=_IDEAL_LINE_COLOUR, lw=1, label="Ideal")
    ax.set_xlabel("Estimated FDR")
    ax.set_ylabel(f"True FDR ({qualifier})")
    suffix = " (0\u20130.1)" if zoomed else ""
    ax.set_title(f"True vs estimated FDR{suffix} \u2014 {display}")
    if zoomed:
        ax.set_xlim(0, 0.1)
        ax.set_ylim(0, 0.1)
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    fig.tight_layout()
    tag = "fdr_true_vs_est_zoom" if zoomed else "fdr_true_vs_est"
    _save_fig(fig, output_dir / f"{tag}_{project}")


# ---------------------------------------------------------------------------
# Probability calibration (reliability diagram)
# ---------------------------------------------------------------------------
def _compute_calibration_curve(
    df: pd.DataFrame,
    pred_col: str,
    label_col: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Fixed-width bin calibration curve."""
    data = df[[pred_col, label_col]].dropna().copy()
    data[pred_col] = data[pred_col].clip(0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_cats = pd.cut(data[pred_col], bins=bins, include_lowest=True)
    bin_cats.name = "bin"
    grouped = (
        data.groupby(bin_cats, observed=True)
        .agg(
            pred_mean=(pred_col, "mean"),
            empirical=(label_col, "mean"),
            count=(label_col, "size"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["count"] > 0]
    grouped["bin_center"] = grouped["bin"].apply(lambda iv: (iv.left + iv.right) / 2)
    return grouped[["pred_mean", "empirical", "count", "bin_center"]]


def plot_calibration(
    df: pd.DataFrame,
    project: str,
    eval_type: str,
    output_dir: Path,
) -> None:
    """Plot probability calibration (reliability diagram)."""
    display = _display_name(project)
    qualifier = _ground_truth_qualifier(eval_type)
    cal = _compute_calibration_curve(df, "calibrated_confidence", "correct")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        cal["pred_mean"],
        cal["empirical"],
        marker="o",
        color=_MAIN_LINE_COLOUR,
        label="Calibrator",
    )
    ax.plot([0, 1], [0, 1], ls="--", color=_IDEAL_LINE_COLOUR, lw=1, label="Ideal")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel(f"Empirical accuracy ({qualifier})")
    ax.set_title(f"Probability calibration \u2014 {display}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_fig(fig, output_dir / f"calibration_{project}")


# ---------------------------------------------------------------------------
# Before/after score histograms
# ---------------------------------------------------------------------------
def plot_score_histograms(
    df: pd.DataFrame,
    project: str,
    eval_type: str,
    output_dir: Path,
) -> None:
    """Plot before/after score histograms with correct/incorrect overlays."""
    display = _display_name(project)
    qualifier = _ground_truth_qualifier(eval_type)

    correct_mask = df["correct"].astype(bool)
    correct_label = f"Correct ({qualifier})"
    incorrect_label = f"Incorrect ({qualifier})"

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=False)

    # Before calibration
    ax = axes[0]
    bins_before = np.linspace(0, 1, 51)
    ax.hist(
        df.loc[correct_mask, "confidence"],
        bins=bins_before,
        alpha=0.6,
        color=_CORRECT_COLOUR,
        label=correct_label,
    )
    ax.hist(
        df.loc[~correct_mask, "confidence"],
        bins=bins_before,
        alpha=0.6,
        color=_INCORRECT_COLOUR,
        label=incorrect_label,
    )
    ax.set_xlabel("Raw confidence")
    ax.set_ylabel("Count")
    ax.set_title("Before calibration")
    ax.legend(fontsize=8)

    # After calibration
    ax = axes[1]
    bins_after = np.linspace(0, 1, 51)
    ax.hist(
        df.loc[correct_mask, "calibrated_confidence"],
        bins=bins_after,
        alpha=0.6,
        color=_CORRECT_COLOUR,
        label=correct_label,
    )
    ax.hist(
        df.loc[~correct_mask, "calibrated_confidence"],
        bins=bins_after,
        alpha=0.6,
        color=_INCORRECT_COLOUR,
        label=incorrect_label,
    )
    ax.set_xlabel("Calibrated confidence")
    ax.set_ylabel("Count")
    ax.set_title("After calibration")
    ax.legend(fontsize=8)

    fig.suptitle(f"Score distributions \u2014 {display}", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, output_dir / f"score_histograms_{project}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _load_project_data(
    predictions_root: Path,
    project: str,
    suffix: str,
    eval_type: str,
) -> pd.DataFrame:
    """Load and merge metadata.csv and preds_and_fdr_metrics.csv for a project."""
    folder = predictions_root / f"{project}_{suffix}"
    preds_path = folder / "preds_and_fdr_metrics.csv"
    meta_path = folder / "metadata.csv"
    if not preds_path.is_file():
        raise FileNotFoundError(f"Missing predictions file: {preds_path}")

    preds_df = pd.read_csv(preds_path)

    if meta_path.is_file():
        meta_df = pd.read_csv(meta_path)
        # Drop columns already present in preds to avoid duplicates on merge
        overlap = [
            c for c in meta_df.columns if c in preds_df.columns and c != "spectrum_id"
        ]
        if overlap:
            meta_df = meta_df.drop(columns=overlap)
        df = preds_df.merge(meta_df, on="spectrum_id", how="left")
    else:
        df = preds_df

    if eval_type in ("raw", "unlabelled"):
        if "proteome_hit" not in df.columns:
            raise ValueError(
                f"Expected 'proteome_hit' column for eval-type={eval_type} in {preds_path}"
            )
        df["correct"] = df["proteome_hit"].astype(float)

    required = ["confidence", "calibrated_confidence", "correct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {preds_path}")

    return df


def generate_all_plots(
    df: pd.DataFrame,
    project: str,
    eval_type: str,
    output_dir: Path,
) -> None:
    """Generate all 6 plots for a single project."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_precision_recall(df, project, eval_type, output_dir)
    plot_fdr_run(df, project, eval_type, output_dir)
    plot_true_vs_estimated_fdr(df, project, eval_type, output_dir, zoomed=False)
    plot_true_vs_estimated_fdr(df, project, eval_type, output_dir, zoomed=True)
    plot_calibration(df, project, eval_type, output_dir)
    plot_score_histograms(df, project, eval_type, output_dir)


_EVAL_TYPE_SUFFIX: dict[str, str] = {
    "annotated": "annotated",
    "raw": "raw",
    "labelled": "labelled",
    "unlabelled": "unlabelled",
}


@app.command()
def main(
    predictions_root: Annotated[
        Path,
        typer.Option(
            "--predictions-root",
            help="Root directory containing per-project prediction folders.",
        ),
    ],
    projects: Annotated[
        str,
        typer.Option(
            "--projects",
            help="Space- or comma-separated project keys (e.g. 'helaqc,gluc' or 'helaqc gluc').",
        ),
    ],
    eval_type: Annotated[
        str,
        typer.Option(
            "--eval-type",
            help="Evaluation type: annotated, raw, labelled, or unlabelled.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory to save plots and summary CSVs."),
    ],
) -> None:
    """Generate evaluation plots from winnow predict outputs."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="%H:%M:%S")

    if eval_type not in _EVAL_TYPE_SUFFIX:
        raise typer.BadParameter(
            f"Unknown eval-type {eval_type!r}. Expected one of: {list(_EVAL_TYPE_SUFFIX)}"
        )

    project_list = [p.strip() for p in projects.replace(",", " ").split() if p.strip()]
    if not project_list:
        raise typer.BadParameter("No projects specified.")

    suffix = _EVAL_TYPE_SUFFIX[eval_type]
    output_dir.mkdir(parents=True, exist_ok=True)

    for project in project_list:
        display = _display_name(project)
        logger.info("Processing %s (%s, eval-type=%s)...", project, display, eval_type)

        df = _load_project_data(predictions_root, project, suffix, eval_type)
        logger.info("  Loaded %d rows", len(df))

        summary_cols = ["confidence", "calibrated_confidence", "correct"]
        if "psm_fdr" in df.columns:
            summary_cols.append("psm_fdr")
        if "psm_q_value" in df.columns:
            summary_cols.append("psm_q_value")
        df[summary_cols].to_csv(output_dir / f"{project}_summary.csv", index=False)

        generate_all_plots(df, project, eval_type, output_dir)
        logger.info("  Plots saved to %s", output_dir)

    logger.info("Done. All plots saved to %s", output_dir)


if __name__ == "__main__":
    app()
