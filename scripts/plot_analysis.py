"""Generate analysis plots from Winnow predict outputs.

Usage:
    python scripts/plot_analysis.py \
        --predictions-dir results/instanovo_helaqc_predictions_test/ \
        --split test \
        --label-mode labelled \
        --fasta fasta/human.fasta \
        [--model-dir models/instanovo_helaqc] \
        [--output-dir results/instanovo_helaqc_predictions_test/plots/]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import yaml
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from winnow.calibration.calibrator import TrainingHistory  # noqa: E402
from winnow.fdr.database_grounded import DatabaseGroundedFDRControl  # noqa: E402
from scripts.annotate_preds_proteome_hits import (  # noqa: E402
    filter_and_annotate_preds,
    load_proteome_haystack,
)

# ── Style — Paul Tol "bright" palette (colour-blind safe) ────────────
_PALETTE = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]
_CORRECT_COLOUR = _PALETTE[0]
_INCORRECT_COLOUR = _PALETTE[1]
_MAIN_LINE_COLOUR = _PALETTE[0]
_RAW_LINE_COLOUR = _PALETTE[5]
_IDEAL_LINE_COLOUR = _PALETTE[6]

sns.set_theme(style="white", palette=_PALETTE, context="paper", font_scale=1.5)
warnings.filterwarnings("ignore", module="winnow")


def _spine_fmt(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.8)


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    base = out_dir / name
    fig.savefig(f"{base}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"{base}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved {name}")


# ── Plot functions ────────────────────────────────────────────────────


def _df_for_raw_confidence_plots(df: pl.DataFrame) -> pl.DataFrame:
    """Drop PSMs with negative raw confidence (Casanovo mass-mismatch penalty scores)."""
    if "confidence" not in df.columns:
        return df
    n_neg = int((df["confidence"] < 0).sum())
    if n_neg == 0:
        return df
    print(
        f"  excluding {n_neg} PSMs with negative raw confidence "
        "from raw-confidence plots"
    )
    return df.filter(pl.col("confidence") >= 0)


def plot_calibration_curves(
    df: pl.DataFrame,
    label_col: str,
    title: str,
    bins: int = 10,
    df_raw: pl.DataFrame | None = None,
) -> plt.Figure:
    """Plot reliability curves for calibrated and (optionally) raw confidence."""
    fig, ax = plt.subplots(figsize=(8, 6))

    frac_pos, mean_pred = calibration_curve(
        df[label_col].to_numpy(),
        df["calibrated_confidence"].to_numpy(),
        n_bins=bins,
        strategy="uniform",
    )
    ax.plot(
        mean_pred,
        frac_pos,
        marker="o",
        color=_MAIN_LINE_COLOUR,
        label="Calibrated confidence",
        linewidth=1.5,
        markersize=6,
        zorder=3,
    )

    if df_raw is not None and len(df_raw) > 0 and "confidence" in df_raw.columns:
        frac_pos, mean_pred = calibration_curve(
            df_raw[label_col].to_numpy(),
            df_raw["confidence"].to_numpy(),
            n_bins=bins,
            strategy="uniform",
        )
        ax.plot(
            mean_pred,
            frac_pos,
            marker="D",
            color=_RAW_LINE_COLOUR,
            label="Raw confidence",
            linewidth=1.5,
            markersize=6,
            zorder=3,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        color=_IDEAL_LINE_COLOUR,
        label="Perfectly calibrated",
        alpha=0.7,
        zorder=2,
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.grid(False)
    _spine_fmt(ax)
    return fig


def plot_pr_curves(
    df: pl.DataFrame,
    label_col: str,
    title: str,
    df_raw: pl.DataFrame | None = None,
) -> plt.Figure:
    """Plot precision–recall curves for calibrated and (optionally) raw confidence."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sorted_cal = df.sort("calibrated_confidence", descending=True)
    labels = sorted_cal[label_col].to_numpy()
    cum = np.cumsum(labels)
    precision = cum / np.arange(1, len(labels) + 1)
    recall = cum / len(labels)
    ax.plot(
        recall,
        precision,
        color=_MAIN_LINE_COLOUR,
        label="Calibrated confidence",
        linewidth=1.5,
    )

    if df_raw is not None and len(df_raw) > 0 and "confidence" in df_raw.columns:
        sorted_raw = df_raw.sort("confidence", descending=True)
        labels = sorted_raw[label_col].to_numpy()
        cum = np.cumsum(labels)
        precision = cum / np.arange(1, len(labels) + 1)
        recall = cum / len(labels)
        ax.plot(
            recall,
            precision,
            color=_RAW_LINE_COLOUR,
            label="Raw confidence",
            linewidth=1.5,
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    ax.grid(False)
    _spine_fmt(ax)
    return fig


def plot_confidence_histogram(
    df: pl.DataFrame,
    label_col: str,
    conf_col: str,
    col_label: str,
    title: str,
    bins: int = 50,
) -> plt.Figure:
    """Plot confidence histograms with KDE overlays for correct vs incorrect PSMs."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    pos = df.filter(pl.col(label_col))
    neg = df.filter(~pl.col(label_col))
    n_data = neg[conf_col].to_numpy()
    p_data = pos[conf_col].to_numpy()

    ax.hist(
        n_data,
        bins=bins,
        alpha=0.6,
        label="Incorrect",
        density=False,
        edgecolor="black",
        color=_INCORRECT_COLOUR,
    )
    ax.hist(
        p_data,
        bins=bins,
        alpha=0.6,
        label="Correct",
        density=False,
        edgecolor="black",
        color=_CORRECT_COLOUR,
    )

    x_min = min(n_data.min(), p_data.min())
    x_max = max(n_data.max(), p_data.max())
    x_grid = np.linspace(x_min, x_max, 300)
    bin_width = (x_max - x_min) / bins if bins > 1 else 1.0

    if len(n_data) > 1:
        y_neg = gaussian_kde(n_data)(x_grid) * len(n_data) * bin_width
        ax.plot(x_grid, y_neg, color=_INCORRECT_COLOUR, lw=1.5)
    if len(p_data) > 1:
        y_pos = gaussian_kde(p_data)(x_grid) * len(p_data) * bin_width
        ax.plot(x_grid, y_pos, color=_CORRECT_COLOUR, lw=1.5)

    ax.set_xlabel(col_label)
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper center")
    ax.grid(False)
    ax.set_title(title)
    _spine_fmt(ax)
    fig.tight_layout()
    return fig


def _fit_db_fdr(
    df: pl.DataFrame,
    correct_col: str,
    residue_masses: dict,
    confidence_feature: str = "calibrated_confidence",
    drop: int = 10,
    use_proteome_shortcut: bool = False,
) -> DatabaseGroundedFDRControl:
    """Fit a DatabaseGroundedFDRControl, using a proteome shortcut if labels lack sequences."""
    ctrl = DatabaseGroundedFDRControl(
        confidence_feature=confidence_feature,
        residue_masses=residue_masses,
        drop=drop,
    )
    if use_proteome_shortcut:
        sorted_df = df.sort(confidence_feature, descending=True)
        correct_vals = sorted_df[correct_col].to_numpy().astype(float)
        confidence_vals = sorted_df[confidence_feature].to_numpy()
        precision = np.cumsum(correct_vals) / np.arange(1, len(sorted_df) + 1)
        ctrl._fdr_values = np.array(1 - precision[drop:])
        ctrl._confidence_scores = confidence_vals[drop:]
    else:
        ctrl.fit(dataset=df.to_pandas(), correct_column=correct_col)
    return ctrl


def plot_fdr_accuracy(
    df: pl.DataFrame,
    correct_col: str,
    residue_masses: dict,
    title: str,
    metric: str = "fdr",
    use_proteome_shortcut: bool = False,
) -> plt.Figure:
    """Compare non-parametric vs database-grounded FDR or q-value vs confidence."""
    fig, ax = plt.subplots(figsize=(8, 6))
    col_name = "psm_fdr" if metric == "fdr" else "psm_q_value"
    winnow_col = col_name

    ctrl = _fit_db_fdr(
        df, correct_col, residue_masses, use_proteome_shortcut=use_proteome_shortcut
    )

    if metric == "fdr":
        db_pd = ctrl.add_psm_fdr(df.to_pandas(), "calibrated_confidence")
    else:
        df_pd = df.to_pandas()
        if "psm_q_value" in df_pd.columns:
            df_pd = df_pd.drop(columns=["psm_q_value"])
        db_pd = ctrl.add_psm_q_value(df_pd, "calibrated_confidence")
    db_df = pl.from_pandas(db_pd).select(["spectrum_id", col_name])

    merged = (
        df.select(["spectrum_id", "calibrated_confidence", winnow_col])
        .join(db_df, on="spectrum_id", how="inner", suffix="_db")
        .sort("calibrated_confidence")
    )

    conf = merged["calibrated_confidence"].to_numpy()
    ax.plot(
        conf,
        merged[winnow_col].to_numpy(),
        color=_MAIN_LINE_COLOUR,
        label="Non-parametric",
        linewidth=1.5,
    )
    ax.plot(
        conf,
        merged[f"{col_name}_db"].to_numpy(),
        color=_RAW_LINE_COLOUR,
        label="Database-grounded",
        linewidth=1.5,
    )

    ax.set_xlabel("Calibrated confidence")
    ylabel = "FDR" if metric == "fdr" else "Q-value"
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(False)
    _spine_fmt(ax)
    return fig


def plot_ranked_qvalue(
    df: pl.DataFrame,
    correct_col: str,
    residue_masses: dict,
    title: str,
) -> plt.Figure:
    """Ranked predictions vs q-value (non-parametric & database-grounded)."""
    ctrl = _fit_db_fdr(df, correct_col, residue_masses)
    test_pd = df.to_pandas()
    test_pd_no_q = test_pd.drop(columns=["psm_q_value"], errors="ignore")
    db_q = ctrl.add_psm_q_value(test_pd_no_q, "calibrated_confidence")

    sorted_np = test_pd.sort_values(
        "calibrated_confidence", ascending=False
    ).reset_index(drop=True)
    sorted_db = db_q.sort_values("calibrated_confidence", ascending=False).reset_index(
        drop=True
    )
    ranks = np.arange(1, len(sorted_np) + 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        ranks,
        sorted_np["psm_q_value"].values,
        color=_MAIN_LINE_COLOUR,
        label="Non-parametric",
        linewidth=1.5,
    )
    ax.plot(
        ranks,
        sorted_db["psm_q_value"].values,
        color=_RAW_LINE_COLOUR,
        label="Database-grounded",
        linewidth=1.5,
    )
    ax.set_xlabel("Ranked predictions")
    ax.set_ylabel("Q-value")
    ax.set_title(title)
    ax.legend(loc="upper left")
    _spine_fmt(ax)
    return fig


def plot_ranked_fdr_pep(df: pl.DataFrame, title: str) -> plt.Figure:
    """Ranked predictions vs non-parametric FDR and PEP."""
    sorted_df = df.sort("calibrated_confidence", descending=True)
    ranks = np.arange(1, len(sorted_df) + 1)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        ranks,
        sorted_df["psm_fdr"].to_numpy(),
        color=_MAIN_LINE_COLOUR,
        label="FDR",
        linewidth=1.5,
    )
    if "psm_pep" in sorted_df.columns:
        ax.plot(
            ranks,
            sorted_df["psm_pep"].to_numpy(),
            color=_PALETTE[3],
            label="PEP",
            linewidth=1.5,
        )
    ax.set_xlabel("Ranked predictions")
    ax.set_ylabel("Error rate")
    ax.set_title(title)
    ax.legend(loc="upper left")
    _spine_fmt(ax)
    return fig


def plot_ranked_fdr_raw_vs_cal(
    df: pl.DataFrame,
    correct_col: str,
    residue_masses: dict,
    title: str,
    metric: str = "fdr",
    df_raw: pl.DataFrame | None = None,
) -> plt.Figure:
    """Ranked predictions vs FDR/q-value for non-parametric and database-grounded on raw+calibrated."""
    test_pd = df.to_pandas()
    test_pd_no_q = test_pd.drop(columns=["psm_q_value", "psm_fdr"], errors="ignore")

    np_cal = test_pd.sort_values("calibrated_confidence", ascending=False).reset_index(
        drop=True
    )

    db_cal_ctrl = _fit_db_fdr(
        df, correct_col, residue_masses, confidence_feature="calibrated_confidence"
    )
    raw_df = df_raw if df_raw is not None else df
    db_raw_ctrl = _fit_db_fdr(
        raw_df, correct_col, residue_masses, confidence_feature="confidence"
    )

    col_name = "psm_fdr" if metric == "fdr" else "psm_q_value"
    add_fn = "add_psm_fdr" if metric == "fdr" else "add_psm_q_value"
    sort_col_cal = "calibrated_confidence"
    sort_col_raw = "confidence"

    db_cal = getattr(db_cal_ctrl, add_fn)(test_pd_no_q.copy(), sort_col_cal)
    db_cal = db_cal.sort_values(sort_col_cal, ascending=False).reset_index(drop=True)

    raw_pd_no_q = raw_df.to_pandas().drop(
        columns=["psm_q_value", "psm_fdr"], errors="ignore"
    )
    db_raw = getattr(db_raw_ctrl, add_fn)(raw_pd_no_q.copy(), sort_col_raw)
    db_raw = db_raw.sort_values(sort_col_raw, ascending=False).reset_index(drop=True)

    ranks_cal = np.arange(1, len(np_cal) + 1)
    ranks_raw = np.arange(1, len(db_raw) + 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ylabel = "FDR" if metric == "fdr" else "Q-value"
    np_col = "psm_fdr" if metric == "fdr" else "psm_q_value"
    ax.plot(
        ranks_cal,
        np_cal[np_col].values,
        color=_MAIN_LINE_COLOUR,
        label="Non-parametric (calibrated)",
        linewidth=1.5,
    )
    ax.plot(
        ranks_cal,
        db_cal[col_name].values,
        color=_RAW_LINE_COLOUR,
        label="Database-grounded (calibrated)",
        linewidth=1.5,
    )
    ax.plot(
        ranks_raw,
        db_raw[col_name].values,
        color=_PALETTE[3],
        label="Database-grounded (raw)",
        linewidth=1.5,
    )
    ax.set_xlabel("Ranked predictions")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left")
    _spine_fmt(ax)
    return fig


def plot_bar_psms_fdr(
    df: pl.DataFrame,
    correct_col: str,
    residue_masses: dict,
    title: str,
    df_raw: pl.DataFrame | None = None,
) -> plt.Figure:
    """Bar plot of PSMs at q-value thresholds (calibrated vs raw, database-grounded)."""
    test_pd = df.to_pandas()
    test_pd_no_q = test_pd.drop(columns=["psm_q_value", "psm_fdr"], errors="ignore")

    db_cal_ctrl = _fit_db_fdr(
        df, correct_col, residue_masses, confidence_feature="calibrated_confidence"
    )
    raw_df = df_raw if df_raw is not None else df
    db_raw_ctrl = _fit_db_fdr(
        raw_df, correct_col, residue_masses, confidence_feature="confidence"
    )

    db_cal = db_cal_ctrl.add_psm_q_value(test_pd_no_q.copy(), "calibrated_confidence")
    raw_pd_no_q = raw_df.to_pandas().drop(
        columns=["psm_q_value", "psm_fdr"], errors="ignore"
    )
    db_raw = db_raw_ctrl.add_psm_q_value(raw_pd_no_q.copy(), "confidence")

    thresholds = [0.001, 0.01, 0.05, 0.1]
    counts_cal = [int((db_cal["psm_q_value"] <= t).sum()) for t in thresholds]
    counts_raw = [int((db_raw["psm_q_value"] <= t).sum()) for t in thresholds]

    x = np.arange(len(thresholds))
    width, gap = 0.32, 0.04

    fig, ax = plt.subplots(figsize=(8, 6))
    bars_cal = ax.bar(
        x - width / 2 - gap / 2,
        counts_cal,
        width,
        label="Calibrated confidence",
        color=_MAIN_LINE_COLOUR,
        edgecolor="black",
        linewidth=1,
    )
    bars_raw = ax.bar(
        x + width / 2 + gap / 2,
        counts_raw,
        width,
        label="Raw confidence",
        color=_RAW_LINE_COLOUR,
        edgecolor="black",
        linewidth=1,
    )
    ax.set_xlabel("FDR threshold")
    ax.set_ylabel("Peptide-spectrum matches")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in thresholds])
    ax.legend(loc="upper left")

    for bar_group in [bars_cal, bars_raw]:
        for bar in bar_group:
            h = bar.get_height()
            ax.annotate(
                f"{h:,}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    _spine_fmt(ax)
    return fig


def plot_raw_vs_cal_scatter(
    df: pl.DataFrame,
    label_col: str,
    title: str,
) -> plt.Figure:
    """Raw confidence vs calibrated confidence coloured by correctness."""
    fig, ax = plt.subplots(figsize=(8, 7))
    inc = df.filter(~pl.col(label_col))
    cor = df.filter(pl.col(label_col))
    ax.scatter(
        inc["confidence"].to_numpy(),
        inc["calibrated_confidence"].to_numpy(),
        c=_INCORRECT_COLOUR,
        label="Incorrect",
        s=10,
        alpha=0.3,
        rasterized=True,
    )
    ax.scatter(
        cor["confidence"].to_numpy(),
        cor["calibrated_confidence"].to_numpy(),
        c=_CORRECT_COLOUR,
        label="Correct",
        s=10,
        alpha=0.3,
        rasterized=True,
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color=_IDEAL_LINE_COLOUR,
        linestyle="--",
        linewidth=1,
        label="Identity",
    )
    ax.set_xlabel("Raw confidence")
    ax.set_ylabel("Calibrated confidence")
    ax.set_title(title)
    ax.legend(loc="upper left")
    _spine_fmt(ax)
    return fig


def plot_pca_features(
    df: pl.DataFrame,
    label_col: str,
    title: str,
) -> tuple[plt.Figure, PCA, list[str]]:
    """PCA of calibrator features coloured by correctness."""
    feature_cols = [
        "confidence",
        "mass_error_ppm",
        "ion_matches",
        "ion_match_intensity",
        "complementary_ion_count",
        "max_ion_gap",
        "spectral_angle",
        "xcorr",
        "irt_error",
        "margin",
        "median_margin",
        "entropy",
        "z-score",
        "edit_distance",
        "min_token_probability",
        "std_token_probability",
    ]
    available = [c for c in feature_cols if c in df.columns]
    feat_df = df.select(available).to_pandas().dropna()
    labels = df.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in available]))[
        label_col
    ].to_numpy()

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feat_df.values)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(features_scaled)

    fig, ax = plt.subplots(figsize=(8, 7))
    mask_inc, mask_cor = ~labels, labels
    ax.scatter(
        coords[mask_inc, 0],
        coords[mask_inc, 1],
        c=_INCORRECT_COLOUR,
        label="Incorrect",
        s=10,
        alpha=0.3,
        rasterized=True,
    )
    ax.scatter(
        coords[mask_cor, 0],
        coords[mask_cor, 1],
        c=_CORRECT_COLOUR,
        label="Correct",
        s=10,
        alpha=0.3,
        rasterized=True,
    )
    ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    _spine_fmt(ax)
    return fig, pca, available


def plot_pca_loadings(
    pca: PCA,
    feature_names: list[str],
    title: str,
) -> plt.Figure:
    """PCA loadings for PC1 and PC2, ordered by |PC1|."""
    pretty = {
        "confidence": "Raw confidence",
        "mass_error_ppm": "Log absolute mass error (ppm)",
        "ion_matches": "Ion matches",
        "ion_match_intensity": "Ion match intensity",
        "complementary_ion_count": "Complementary ion count",
        "max_ion_gap": "Maximum ion gap",
        "spectral_angle": "Spectral angle",
        "xcorr": "Cross-correlation",
        "irt_error": "Retention time error",
        "margin": "Margin",
        "median_margin": "Median margin",
        "entropy": "Entropy",
        "z-score": "Z-score",
        "edit_distance": "Edit distance",
        "min_token_probability": "Minimum token probability",
        "std_token_probability": "Token probability std. dev.",
    }
    pc1 = pca.components_[0]
    pc2 = pca.components_[1]
    names = [pretty.get(c, c) for c in feature_names]
    order = np.argsort(np.abs(pc1))[::-1]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(y, pc1[order], color=_MAIN_LINE_COLOUR, alpha=0.6, edgecolor="black")
    ax.barh(y, pc2[order], color=_RAW_LINE_COLOUR, alpha=0.4, edgecolor="black")
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("Loading value")
    ax.set_title(title)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.legend(
        handles=[
            Patch(facecolor=_MAIN_LINE_COLOUR, alpha=0.6, label="PC 1 loading"),
            Patch(facecolor=_RAW_LINE_COLOUR, alpha=0.4, label="PC 2 loading"),
        ],
        loc="lower right",
    )
    _spine_fmt(ax)
    return fig


def plot_scatter_feature_vs_conf(
    df: pl.DataFrame,
    label_col: str,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str | None = None,
    y_label: str | None = None,
) -> plt.Figure:
    """Scatter of x_col vs y_col coloured by correctness."""
    fig, ax = plt.subplots(figsize=(8, 7))
    inc = df.filter(~pl.col(label_col))
    cor = df.filter(pl.col(label_col))
    ax.scatter(
        inc[x_col].to_numpy(),
        inc[y_col].to_numpy(),
        c=_INCORRECT_COLOUR,
        label="Incorrect",
        s=10,
        alpha=0.3,
        rasterized=True,
    )
    ax.scatter(
        cor[x_col].to_numpy(),
        cor[y_col].to_numpy(),
        c=_CORRECT_COLOUR,
        label="Correct",
        s=10,
        alpha=0.3,
        rasterized=True,
    )
    ax.set_xlabel(x_label or x_col.replace("_", " ").title())
    ax.set_ylabel(y_label or y_col.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(loc="upper left")
    _spine_fmt(ax)
    return fig


# ── Main logic ────────────────────────────────────────────────────────


def _load_residue_masses() -> dict:
    cfg_path = REPO_ROOT / "winnow" / "configs" / "residues.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)["residue_masses"]


def _load_data(predictions_dir: Path) -> pl.DataFrame:
    preds = pl.read_csv(predictions_dir / "preds_and_fdr_metrics.csv")
    meta_path = predictions_dir / "metadata.csv"
    if meta_path.exists():
        meta = pl.read_csv(meta_path)
        preds = preds.join(meta, on="spectrum_id", how="inner")
    return preds


_SPLIT_DISPLAY_NAMES = {
    "test": "test set",
    "unlabelled": "unlabelled space",
    "raw_less_train": "full search space",
}


def _split_display(split: str) -> str:
    key = split.strip().replace("-", "_")
    return _SPLIT_DISPLAY_NAMES.get(key, split)


def _dns_model_tag(dns_model: str | None) -> str:
    return f" ({dns_model})" if dns_model else ""


def _split_title(split_label: str, dns_model: str | None = None) -> str:
    return f"{split_label}{_dns_model_tag(dns_model)}"


def _eval_title(
    prefix: str,
    split_label: str,
    eval_kind: str,
    dns_model: str | None = None,
) -> str:
    return f"{prefix} {_split_title(split_label, dns_model)}\nusing {eval_kind}"


def _save_training_history_plot(model_dir: Path, out_dir: Path) -> None:
    hist_path = model_dir / "training_history.json"
    if not hist_path.exists():
        return
    print("Plotting training history")
    th = TrainingHistory.load(str(hist_path))
    th.plot(output_path=out_dir / "training_history.png", show=False)
    print("  saved training_history")


def _plot_calibration_and_pr(
    df: pl.DataFrame,
    df_raw_conf: pl.DataFrame,
    split: str,
    labelled: bool,
    out_dir: Path,
    dns_model: str | None = None,
) -> None:
    split_label = _split_display(split)
    print("Plotting calibration curves")
    if labelled:
        fig = plot_calibration_curves(
            df,
            "correct",
            _eval_title(
                "Calibration curves for", split_label, "database search", dns_model
            ),
            df_raw=df_raw_conf,
        )
        _save(fig, out_dir, f"calibration_{split}_db_search")

    fig = plot_calibration_curves(
        df,
        "proteome_hit",
        _eval_title(
            "Calibration curves for", split_label, "proteome mapping", dns_model
        ),
        df_raw=df_raw_conf,
    )
    _save(fig, out_dir, f"calibration_{split}_proteome")

    print("Plotting PR curves")
    if labelled:
        fig = plot_pr_curves(
            df,
            "correct",
            _eval_title("PR curves for", split_label, "database search", dns_model),
            df_raw=df_raw_conf,
        )
        _save(fig, out_dir, f"pr_{split}_db_search")

    fig = plot_pr_curves(
        df,
        "proteome_hit",
        _eval_title("PR curves for", split_label, "proteome mapping", dns_model),
        df_raw=df_raw_conf,
    )
    _save(fig, out_dir, f"pr_{split}_proteome")


def _plot_confidence_histograms(
    df: pl.DataFrame,
    df_raw_conf: pl.DataFrame,
    split: str,
    labelled: bool,
    out_dir: Path,
    dns_model: str | None = None,
) -> None:
    split_label = _split_display(split)
    print("Plotting confidence histograms")
    for conf_col, conf_label, tag in [
        ("confidence", "Raw confidence", "raw"),
        ("calibrated_confidence", "Calibrated confidence", "cal"),
    ]:
        hist_df = df_raw_conf if conf_col == "confidence" else df
        if labelled:
            fig = plot_confidence_histogram(
                hist_df,
                "correct",
                conf_col,
                conf_label,
                _eval_title(
                    f"{conf_label} for", split_label, "database search", dns_model
                ),
            )
            _save(fig, out_dir, f"hist_{tag}_{split}_db_search")

        fig = plot_confidence_histogram(
            hist_df,
            "proteome_hit",
            conf_col,
            conf_label,
            _eval_title(
                f"{conf_label} for", split_label, "proteome mapping", dns_model
            ),
        )
        _save(fig, out_dir, f"hist_{tag}_{split}_proteome")


def _plot_fdr_accuracy_plots(
    df: pl.DataFrame,
    split: str,
    labelled: bool,
    residue_masses: dict,
    out_dir: Path,
    dns_model: str | None = None,
) -> None:
    split_label = _split_display(split)
    print("Plotting FDR accuracy")
    use_shortcut = not labelled
    for metric, tag in [("fdr", "fdr"), ("q_value", "qvalue")]:
        metric_name = "FDR" if metric == "fdr" else "Q-value"
        if labelled:
            fig = plot_fdr_accuracy(
                df,
                "correct",
                residue_masses,
                _eval_title(
                    f"{metric_name} accuracy for",
                    split_label,
                    "database search",
                    dns_model,
                ),
                metric="fdr" if metric == "fdr" else "q_value",
                use_proteome_shortcut=False,
            )
            _save(fig, out_dir, f"{tag}_{split}_db_search")

        fig = plot_fdr_accuracy(
            df,
            "proteome_hit",
            residue_masses,
            _eval_title(
                f"{metric_name} accuracy for",
                split_label,
                "proteome mapping",
                dns_model,
            ),
            metric="fdr" if metric == "fdr" else "q_value",
            use_proteome_shortcut=use_shortcut,
        )
        _save(fig, out_dir, f"{tag}_{split}_proteome")


def _plot_labelled_diagnostics(
    df: pl.DataFrame,
    df_raw_conf: pl.DataFrame,
    split: str,
    residue_masses: dict,
    out_dir: Path,
    dns_model: str | None = None,
) -> None:
    split_label = _split_display(split)
    title_split = _split_title(split_label, dns_model)
    label_col = "correct"
    print("Plotting labelled-only diagnostics")

    fig = plot_ranked_qvalue(
        df,
        label_col,
        residue_masses,
        _eval_title(
            "Ranked predictions vs q-value for",
            split_label,
            "database search",
            dns_model,
        ),
    )
    _save(fig, out_dir, f"ranked_qvalue_{split}_db_search")

    fig = plot_ranked_fdr_pep(
        df, f"Ranked predictions vs FDR and PEP for {title_split}"
    )
    _save(fig, out_dir, f"ranked_fdr_pep_{split}_nonparametric")

    for metric, file_tag, metric_name in [
        ("fdr", "fdr", "FDR"),
        ("q_value", "qvalue", "q-value"),
    ]:
        fig = plot_ranked_fdr_raw_vs_cal(
            df,
            label_col,
            residue_masses,
            _eval_title(
                f"Ranked predictions vs {metric_name} for",
                split_label,
                "database search",
                dns_model,
            ),
            metric=metric,
            df_raw=df_raw_conf,
        )
        _save(fig, out_dir, f"ranked_{file_tag}_raw_vs_cal_{split}_db_search")

    fig = plot_bar_psms_fdr(
        df,
        label_col,
        residue_masses,
        f"PSMs at database-grounded FDR thresholds for {title_split}",
        df_raw=df_raw_conf,
    )
    _save(fig, out_dir, f"bar_psms_fdr_thresholds_{split}_db_search")

    if len(df_raw_conf) > 0:
        fig = plot_raw_vs_cal_scatter(
            df_raw_conf,
            label_col,
            f"Raw vs calibrated confidence for {title_split}",
        )
        _save(fig, out_dir, f"scatter_raw_vs_cal_confidence_{split}")

    if "margin" in df.columns and len(df_raw_conf) > 0:
        fig = plot_scatter_feature_vs_conf(
            df_raw_conf,
            label_col,
            "margin",
            "confidence",
            f"Raw confidence vs margin for {title_split}",
            x_label="Margin",
            y_label="Raw confidence",
        )
        _save(fig, out_dir, f"scatter_raw_confidence_vs_margin_{split}")

        fig = plot_scatter_feature_vs_conf(
            df,
            label_col,
            "margin",
            "calibrated_confidence",
            f"Calibrated confidence vs margin for {title_split}",
            x_label="Margin",
            y_label="Calibrated confidence",
        )
        _save(fig, out_dir, f"scatter_cal_confidence_vs_margin_{split}")

    fig, pca_model, feat_names = plot_pca_features(
        df, label_col, f"PCA of calibrator features for {title_split}"
    )
    _save(fig, out_dir, f"pca_features_{split}")

    fig = plot_pca_loadings(
        pca_model, feat_names, "PCA loadings for first two principal components"
    )
    _save(fig, out_dir, f"pca_loadings_pc1_pc2_{split}")


def main() -> None:
    """Load predictions, annotate proteome hits, and write analysis plots."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Winnow predict output dir with preds_and_fdr_metrics.csv",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Split id for filenames; titles use test set / unlabelled space / full search space",
    )
    parser.add_argument(
        "--label-mode",
        choices=["labelled", "unlabelled"],
        required=True,
        help="labelled = has 'correct' column; unlabelled = proteome mapping only",
    )
    parser.add_argument(
        "--fasta", type=Path, required=True, help="FASTA file for proteome annotation"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (defaults to predictions-dir/plots/)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Model directory for training history plot",
    )
    parser.add_argument(
        "--dns-model",
        type=str,
        default=None,
        help="Upstream DNS model name for plot titles (e.g. InstaNovo, Casanovo, $\\pi$-PrimeNovo)",
    )
    args = parser.parse_args()

    out_dir = args.output_dir or (args.predictions_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    split = args.split
    dns_model = args.dns_model
    labelled = args.label_mode == "labelled"
    residue_masses = _load_residue_masses()

    print(f"Loading predictions from {args.predictions_dir}")
    df = _load_data(args.predictions_dir)

    from instanovo.utils.metrics import Metrics
    from instanovo.utils.residues import ResidueSet

    metrics = Metrics(
        residue_set=ResidueSet(residue_masses=residue_masses),
        isotope_error_range=(0, 1),
    )

    print(f"Annotating with proteome hits from {args.fasta}")
    haystack = load_proteome_haystack(str(args.fasta))
    df = filter_and_annotate_preds(df, haystack, metrics, min_residue_length=7)
    df_raw_conf = _df_for_raw_confidence_plots(df)

    if args.model_dir is not None:
        _save_training_history_plot(args.model_dir, out_dir)

    _plot_calibration_and_pr(
        df, df_raw_conf, split, labelled, out_dir, dns_model=dns_model
    )
    _plot_confidence_histograms(
        df, df_raw_conf, split, labelled, out_dir, dns_model=dns_model
    )
    _plot_fdr_accuracy_plots(
        df, split, labelled, residue_masses, out_dir, dns_model=dns_model
    )

    if labelled:
        _plot_labelled_diagnostics(
            df, df_raw_conf, split, residue_masses, out_dir, dns_model=dns_model
        )

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
