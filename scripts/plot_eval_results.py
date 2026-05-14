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
    "sbrodae": "$\\it{Scalindua\\;brodae}$",
    "snakevenoms": "Snake venomics",
    "tplantibodies": "Therapeutic nanobodies",
    "woundfluids": "Wound exudates",
    "PXD014877": "$\\it{C.\\;elegans}$",
    "PXD023064": "Immunopeptidomics-2",
    "PXD009935": "Immunopeptidomics-3",
    "Astral": "Astral $\\it{E.\\;coli}$",
}

# Paul Tol "bright" palette (colorblind-safe)
_PALETTE = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]
_CORRECT_COLOUR = _PALETTE[2]
_INCORRECT_COLOUR = _PALETTE[1]
_MAIN_LINE_COLOUR = _PALETTE[0]
_RAW_LINE_COLOUR = _PALETTE[5]
_IDEAL_LINE_COLOUR = _PALETTE[6]
_BAND_COLOUR = _PALETTE[0]

sns.set_theme(style="white", palette=_PALETTE, context="paper", font_scale=1.5)

_DIAGNOSTIC_ALPHAS = (0.01, 0.05, 0.10)
_HOEFFDING_DELTA = 0.05


def _display_name(key: str) -> str:
    """Look up the publication-ready display name for a dataset key."""
    return DATASET_DISPLAY_NAMES.get(key, key)


def _ground_truth_qualifier(eval_type: str) -> str:
    """Return the title-friendly ground truth qualifier for plot titles."""
    if eval_type in ("annotated", "labelled"):
        return "using database search"
    return "using proteome mapping"


def _save_fig(fig: plt.Figure, base_path: Path) -> None:
    """Save figure as both PNG and PDF."""
    fig.savefig(f"{base_path}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"{base_path}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# PR curve (non-standard cumulative definition)
# ---------------------------------------------------------------------------
def _compute_precision_recall(
    df: pd.DataFrame, confidence_col: str = "calibrated_confidence"
) -> pd.DataFrame:
    """Non-standard cumulative PR curve matching the codebase convention."""
    sorted_df = df.sort_values(confidence_col, ascending=False)
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
    pr_cal = _compute_precision_recall(df, "calibrated_confidence")
    pr_raw = _compute_precision_recall(df, "confidence")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        pr_raw["recall"],
        pr_raw["precision"],
        color=_RAW_LINE_COLOUR,
        lw=1.5,
        label="Raw confidence",
    )
    ax.plot(
        pr_cal["recall"],
        pr_cal["precision"],
        color=_MAIN_LINE_COLOUR,
        lw=1.5,
        label="Calibrated confidence",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{display} precision-recall {qualifier}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9, loc="lower right")
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
    """Plot calibrated confidence vs estimated and true PSM FDR."""
    display = _display_name(project)
    qualifier = _ground_truth_qualifier(eval_type)

    fdr_ctrl = NonParametricFDRControl()
    fdr_ctrl.fit(dataset=df["calibrated_confidence"])
    fdr_df = fdr_ctrl.add_psm_fdr(df.copy(), confidence_col="calibrated_confidence")
    fdr_df = fdr_df.sort_values("calibrated_confidence")

    true_fdr_ctrl = _fit_database_grounded_fdr(df)
    true_fdr_df = true_fdr_ctrl.add_psm_fdr(
        df.copy(), confidence_col="calibrated_confidence"
    )
    true_fdr_df = true_fdr_df.sort_values("calibrated_confidence")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        fdr_df["calibrated_confidence"].values,
        fdr_df["psm_fdr"].values,
        color=_MAIN_LINE_COLOUR,
        lw=1.5,
        label="Non-parametric",
    )
    ax.plot(
        true_fdr_df["calibrated_confidence"].values,
        true_fdr_df["psm_fdr"].values,
        color=_RAW_LINE_COLOUR,
        lw=1.5,
        label="Database-grounded",
    )
    ax.set_xlabel("Calibrated confidence")
    ax.set_ylabel("PSM FDR")
    ax.set_title(f"{display} FDR run {qualifier}")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    _save_fig(fig, output_dir / f"fdr_run_{project}")


# ---------------------------------------------------------------------------
# Q-value run plot
# ---------------------------------------------------------------------------
def _fit_database_grounded_fdr(
    df: pd.DataFrame,
    confidence_col: str = "calibrated_confidence",
    correct_col: str = "correct",
    drop: int = 10,
) -> NonParametricFDRControl:
    """Fit an FDR controller using ground-truth labels.

    Replicates the fitting logic of ``DatabaseGroundedFDRControl`` (computing
    FDR as 1 − precision over sorted predictions, with the first *drop* entries
    removed) without pulling in the instanovo dependency.
    """
    sorted_desc = df.sort_values(confidence_col, ascending=False)
    labels = sorted_desc[correct_col].values.astype(float)
    precision = np.cumsum(labels) / np.arange(1, len(labels) + 1)
    confidence = sorted_desc[confidence_col].values

    ctrl = NonParametricFDRControl()
    ctrl._fdr_values = (1.0 - precision)[drop:]
    ctrl._confidence_scores = confidence[drop:]
    return ctrl


def plot_q_value_run(
    df: pd.DataFrame,
    project: str,
    eval_type: str,
    output_dir: Path,
) -> None:
    """Plot calibrated confidence vs estimated and true PSM q-values."""
    if "psm_q_value" not in df.columns:
        logger.warning(
            "Skipping q-value run plot for %s: psm_q_value column missing", project
        )
        return

    display = _display_name(project)
    qualifier = _ground_truth_qualifier(eval_type)

    sorted_df = df.sort_values("calibrated_confidence")

    true_fdr_ctrl = _fit_database_grounded_fdr(df)
    qval_input = df[["calibrated_confidence"]].copy()
    true_q_df = true_fdr_ctrl.add_psm_q_value(
        qval_input, confidence_col="calibrated_confidence"
    )
    true_q_df = true_q_df.sort_values("calibrated_confidence")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        sorted_df["calibrated_confidence"].values,
        sorted_df["psm_q_value"].values,
        color=_MAIN_LINE_COLOUR,
        lw=1.5,
        label="Non-parametric",
    )
    ax.plot(
        true_q_df["calibrated_confidence"].values,
        true_q_df["psm_q_value"].values,
        color=_RAW_LINE_COLOUR,
        lw=1.5,
        label="Database-grounded",
    )
    ax.set_xlabel("Calibrated confidence")
    ax.set_ylabel("PSM q-value")
    ax.set_title(f"{display} q-value run {qualifier}")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    _save_fig(fig, output_dir / f"qvalue_run_{project}")


# ---------------------------------------------------------------------------
# FDR / q-value run plots with Hoeffding confidence bands
# ---------------------------------------------------------------------------
def _hoeffding_band_arrays(n: int) -> np.ndarray:
    """Compute pointwise Hoeffding half-widths for ranks 1..n (descending confidence)."""
    ranks = np.arange(1, n + 1)
    return np.sqrt(np.log(2.0 / _HOEFFDING_DELTA) / (2.0 * ranks))


def plot_fdr_run_with_bands(
    df: pd.DataFrame,
    project: str,
    eval_type: str,
    output_dir: Path,
) -> None:
    """FDR run plot with Hoeffding 95% confidence band on the non-parametric curve."""
    display = _display_name(project)
    qualifier = _ground_truth_qualifier(eval_type)

    fdr_ctrl = NonParametricFDRControl()
    fdr_ctrl.fit(dataset=df["calibrated_confidence"])
    fdr_df = fdr_ctrl.add_psm_fdr(df.copy(), confidence_col="calibrated_confidence")
    fdr_df = fdr_df.sort_values("calibrated_confidence")

    true_fdr_ctrl = _fit_database_grounded_fdr(df)
    true_fdr_df = true_fdr_ctrl.add_psm_fdr(
        df.copy(), confidence_col="calibrated_confidence"
    )
    true_fdr_df = true_fdr_df.sort_values("calibrated_confidence")

    fdr_vals = fdr_df["psm_fdr"].values
    conf_vals = fdr_df["calibrated_confidence"].values
    n = len(fdr_vals)
    hw = _hoeffding_band_arrays(n)[::-1]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(
        conf_vals,
        np.clip(fdr_vals - hw, 0, None),
        np.clip(fdr_vals + hw, None, 1),
        color=_BAND_COLOUR,
        alpha=0.2,
        label="95% Hoeffding bound (sampling noise)",
    )
    ax.plot(
        conf_vals,
        fdr_vals,
        color=_MAIN_LINE_COLOUR,
        lw=1.5,
        label="Non-parametric",
    )
    ax.plot(
        true_fdr_df["calibrated_confidence"].values,
        true_fdr_df["psm_fdr"].values,
        color=_RAW_LINE_COLOUR,
        lw=1.5,
        label="Database-grounded",
    )
    ax.set_xlabel("Calibrated confidence")
    ax.set_ylabel("PSM FDR")
    ax.set_title(f"{display} FDR run with sampling error bounds {qualifier}")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    _save_fig(fig, output_dir / f"fdr_run_bands_{project}")


def plot_q_value_run_with_bands(
    df: pd.DataFrame,
    project: str,
    eval_type: str,
    output_dir: Path,
) -> None:
    """Q-value run plot with Hoeffding 95% confidence band on the non-parametric curve."""
    if "psm_q_value" not in df.columns:
        logger.warning(
            "Skipping banded q-value run plot for %s: psm_q_value column missing",
            project,
        )
        return

    display = _display_name(project)
    qualifier = _ground_truth_qualifier(eval_type)

    sorted_df = df.sort_values("calibrated_confidence")

    true_fdr_ctrl = _fit_database_grounded_fdr(df)
    qval_input = df[["calibrated_confidence"]].copy()
    true_q_df = true_fdr_ctrl.add_psm_q_value(
        qval_input, confidence_col="calibrated_confidence"
    )
    true_q_df = true_q_df.sort_values("calibrated_confidence")

    qvals = sorted_df["psm_q_value"].values
    conf_vals = sorted_df["calibrated_confidence"].values
    n = len(qvals)
    hw = _hoeffding_band_arrays(n)[::-1]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(
        conf_vals,
        np.clip(qvals - hw, 0, None),
        np.clip(qvals + hw, None, 1),
        color=_BAND_COLOUR,
        alpha=0.2,
        label="95% Hoeffding bound (sampling noise)",
    )
    ax.plot(
        conf_vals,
        qvals,
        color=_MAIN_LINE_COLOUR,
        lw=1.5,
        label="Non-parametric",
    )
    ax.plot(
        true_q_df["calibrated_confidence"].values,
        true_q_df["psm_q_value"].values,
        color=_RAW_LINE_COLOUR,
        lw=1.5,
        label="Database-grounded",
    )
    ax.set_xlabel("Calibrated confidence")
    ax.set_ylabel("PSM q-value")
    ax.set_title(f"{display} q-value run with sampling error bounds {qualifier}")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    _save_fig(fig, output_dir / f"qvalue_run_bands_{project}")


# ---------------------------------------------------------------------------
# True FDR vs estimated FDR
# ---------------------------------------------------------------------------
def _compute_true_vs_estimated_fdr(df: pd.DataFrame) -> pd.DataFrame:
    """Compute true and estimated FDR arrays, sorted by confidence descending."""
    sorted_df = df.sort_values("calibrated_confidence", ascending=False).reset_index(
        drop=True
    )

    fdr_ctrl = NonParametricFDRControl()
    fdr_ctrl.fit(dataset=sorted_df["calibrated_confidence"])
    with_est_fdr = fdr_ctrl.add_psm_fdr(
        sorted_df, confidence_col="calibrated_confidence"
    )

    true_fdr_ctrl = _fit_database_grounded_fdr(sorted_df)
    with_true_fdr = true_fdr_ctrl.add_psm_fdr(
        sorted_df, confidence_col="calibrated_confidence"
    )

    return pd.DataFrame(
        {
            "estimated_fdr": with_est_fdr["psm_fdr"].values,
            "true_fdr": with_true_fdr["psm_fdr"].values,
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
    ax.plot(
        [0, lim],
        [0, lim],
        ls="--",
        color=_IDEAL_LINE_COLOUR,
        lw=1,
        label="Perfectly calibrated",
    )
    ax.set_xlabel("Non-parametric estimated FDR")
    ax.set_ylabel("Database-grounded FDR")
    zoom_suffix = " (0 to 0.1)" if zoomed else ""
    ax.set_title(f"{display} true vs estimated FDR{zoom_suffix} {qualifier}")
    if zoomed:
        ax.set_xlim(0, 0.1)
        ax.set_ylim(0, 0.1)
    ax.legend(fontsize=9, loc="upper left")
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


def _estimate_calibration_values(
    df: pd.DataFrame,
    pred_col: str,
    label_col: str,
    n_bins: int = 20,
) -> np.ndarray:
    """Estimate c(s) for each PSM via binned calibration.

    Returns an array of the same length as *df* where each entry is the
    empirical accuracy of the bin that PSM falls into.
    """
    scores = df[pred_col].values.clip(0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(scores, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    labels = df[label_col].values.astype(float)
    bin_sums = np.bincount(bin_idx, weights=labels, minlength=n_bins)
    bin_counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
    bin_counts[bin_counts == 0] = 1.0
    bin_means = bin_sums / bin_counts
    return bin_means[bin_idx]


def _hoeffding_halfwidth(k: int, delta: float = _HOEFFDING_DELTA) -> float:
    """Hoeffding 95% confidence half-width for a mean of *k* bounded [0,1] r.v.s."""
    if k <= 0:
        return float("nan")
    return float(np.sqrt(np.log(2.0 / delta) / (2.0 * k)))


def plot_calibration(
    df: pd.DataFrame,
    project: str,
    eval_type: str,
    output_dir: Path,
) -> None:
    """Plot probability calibration (reliability diagram)."""
    display = _display_name(project)
    qualifier = _ground_truth_qualifier(eval_type)
    cal_calibrated = _compute_calibration_curve(df, "calibrated_confidence", "correct")
    cal_raw = _compute_calibration_curve(df, "confidence", "correct")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        cal_raw["pred_mean"],
        cal_raw["empirical"],
        marker="D",
        color=_RAW_LINE_COLOUR,
        label="Raw confidence",
    )
    ax.plot(
        cal_calibrated["pred_mean"],
        cal_calibrated["empirical"],
        marker="o",
        color=_MAIN_LINE_COLOUR,
        label="Calibrated confidence",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        ls="--",
        color=_IDEAL_LINE_COLOUR,
        lw=1,
        label="Perfectly calibrated",
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"{display} probability calibration {qualifier}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="lower right")
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

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=False)

    # Before calibration
    ax = axes[0]
    bins_before = np.linspace(0, 1, 51)
    ax.hist(
        df.loc[correct_mask, "confidence"],
        bins=bins_before,
        alpha=0.6,
        color=_CORRECT_COLOUR,
        label="Correct",
    )
    ax.hist(
        df.loc[~correct_mask, "confidence"],
        bins=bins_before,
        alpha=0.6,
        color=_INCORRECT_COLOUR,
        label="Incorrect",
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
        label="Correct",
    )
    ax.hist(
        df.loc[~correct_mask, "calibrated_confidence"],
        bins=bins_after,
        alpha=0.6,
        color=_INCORRECT_COLOUR,
        label="Incorrect",
    )
    ax.set_xlabel("Calibrated confidence")
    ax.set_ylabel("Count")
    ax.set_title("After calibration")
    ax.legend(fontsize=8)

    fig.suptitle(f"{display} score distributions {qualifier}", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, output_dir / f"score_histograms_{project}")


# ---------------------------------------------------------------------------
# Diagnostics CSV
# ---------------------------------------------------------------------------
def _is_labelled(eval_type: str) -> bool:
    return eval_type in ("annotated", "labelled")


def _compute_diagnostics(
    df: pd.DataFrame,
    eval_type: str,
    alphas: tuple[float, ...] = _DIAGNOSTIC_ALPHAS,
) -> pd.DataFrame:
    """Compute FDR diagnostics at each target alpha.

    Label-dependent metrics (sTECE, TECE, realised FDR, etc.) are only
    populated for annotated/labelled eval types.
    """
    labelled = _is_labelled(eval_type)

    np_ctrl = NonParametricFDRControl()
    np_ctrl.fit(dataset=df["calibrated_confidence"])

    if labelled:
        db_ctrl = _fit_database_grounded_fdr(df)
        c_hat = _estimate_calibration_values(df, "calibrated_confidence", "correct")
        scores = df["calibrated_confidence"].values.clip(0.0, 1.0)

    rows: list[dict] = []
    for alpha in alphas:
        tau_hat = np_ctrl.get_confidence_cutoff(threshold=alpha)
        if np.isnan(tau_hat):
            rows.append({"alpha": alpha, "tau_hat": float("nan")})
            continue

        mask_hat = df["calibrated_confidence"].values >= tau_hat
        k = int(mask_hat.sum())
        est_fdr = float(np_ctrl.compute_fdr(tau_hat))
        eps = _hoeffding_halfwidth(k)

        row: dict = {
            "alpha": alpha,
            "tau_hat": float(tau_hat),
            "k_accepted": k,
            "estimated_fdr": est_fdr,
            "hoeffding_halfwidth": eps,
        }

        if labelled:
            residuals = c_hat[mask_hat] - scores[mask_hat]
            row["stece"] = float(np.mean(residuals))
            row["tece"] = float(np.mean(np.abs(residuals)))
            row["tece_2"] = float(np.sqrt(np.mean(residuals**2)))

            realised_fdr = float(db_ctrl.compute_fdr(tau_hat))
            row["realised_fdr"] = realised_fdr
            row["fdr_bias"] = est_fdr - realised_fdr

            tau_star = db_ctrl.get_confidence_cutoff(threshold=alpha)
            row["tau_star"] = float(tau_star)
            if not np.isnan(tau_star):
                k_star = int((df["calibrated_confidence"].values >= tau_star).sum())
                row["discovery_count_shift"] = k - k_star
            else:
                row["discovery_count_shift"] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


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
    """Generate all plots for a single project."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_precision_recall(df, project, eval_type, output_dir)
    plot_fdr_run(df, project, eval_type, output_dir)
    plot_fdr_run_with_bands(df, project, eval_type, output_dir)
    plot_q_value_run(df, project, eval_type, output_dir)
    plot_q_value_run_with_bands(df, project, eval_type, output_dir)
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

        true_fdr_ctrl = _fit_database_grounded_fdr(df)
        db_fdr = true_fdr_ctrl.add_psm_fdr(
            df[["calibrated_confidence"]].copy(), confidence_col="calibrated_confidence"
        )
        df["db_grounded_psm_fdr"] = db_fdr["psm_fdr"]
        db_qval = true_fdr_ctrl.add_psm_q_value(
            df[["calibrated_confidence"]].copy(), confidence_col="calibrated_confidence"
        )
        df["db_grounded_psm_q_value"] = db_qval["psm_q_value"]

        summary_cols = ["confidence", "calibrated_confidence", "correct"]
        if "psm_fdr" in df.columns:
            summary_cols.append("psm_fdr")
        summary_cols.append("db_grounded_psm_fdr")
        if "psm_q_value" in df.columns:
            summary_cols.append("psm_q_value")
        summary_cols.append("db_grounded_psm_q_value")
        df[summary_cols].to_csv(output_dir / f"{project}_summary.csv", index=False)

        diag = _compute_diagnostics(df, eval_type)
        diag.to_csv(output_dir / f"{project}_diagnostics.csv", index=False)
        logger.info("  Diagnostics saved (%d alpha levels)", len(diag))

        generate_all_plots(df, project, eval_type, output_dir)
        logger.info("  Plots saved to %s", output_dir)

    logger.info("Done. All plots saved to %s", output_dir)


if __name__ == "__main__":
    app()
