"""Generate feature investigation plots from calibrator training feature matrices.

Produces KDE, scatter, violin, correlation, discriminative-power, pairplot,
mirror-spectrum, retention-time, token-stem, and beam-stem figures matching the
style of ``analysis/feature_investigation_new.ipynb``.

Usage:
    python scripts/plot_feature_investigation.py \
        --features-train models/instanovo_helaqc/features_train.parquet \
        [--features-val models/instanovo_helaqc/features_val.parquet] \
        [--metadata-train models/instanovo_helaqc/metadata_train.parquet] \
        [--metadata-val models/instanovo_helaqc/metadata_val.parquet] \
        [--predictions-csv held_out_projects/.../predictions.csv] \
        [--output-dir models/instanovo_helaqc/feature_investigation_plots]
"""

from __future__ import annotations

import argparse
import ast
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy import stats as sp_stats
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Style — Paul Tol "bright" palette (colour-blind safe)
# ---------------------------------------------------------------------------
_PALETTE = [
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
]
_CORRECT_COLOUR = _PALETTE[0]
_INCORRECT_COLOUR = _PALETTE[1]
_NEUTRAL_COLOUR = _PALETTE[6]

_HIGH_CONF_BEAM_COLOUR = _PALETTE[2]  # green
_LOW_CONF_BEAM_COLOUR = _PALETTE[5]  # purple
_MED_CONF_BEAM_COLOUR = _PALETTE[3]  # yellow

_OBS_COLOUR = _PALETTE[3]  # yellow (observed spectrum)
_THEO_COLOUR = _PALETTE[5]  # purple (predicted spectrum)

HUE_LABEL_CORRECT = "Correct"
HUE_LABEL_INCORRECT = "Incorrect"
HUE_ORDER = [HUE_LABEL_CORRECT, HUE_LABEL_INCORRECT]
HUE_PALETTE = {
    HUE_LABEL_CORRECT: _CORRECT_COLOUR,
    HUE_LABEL_INCORRECT: _INCORRECT_COLOUR,
}

sns.set_theme(style="white", palette=_PALETTE, context="paper", font_scale=1.5)

_SUNSET_COLORS = [
    "#364B9A",
    "#4A7BB7",
    "#6EA6CD",
    "#98CAE1",
    "#C2E4EF",
    "#EAECCC",
    "#FEDA8B",
    "#FDB366",
    "#F67E4B",
    "#DD3D2D",
    "#A50026",
]


def _diverging_cmap() -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list("tol_sunset", _SUNSET_COLORS, N=256)
    cmap.set_bad(color="#FFFFFF")
    return cmap


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
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

FRAGMENT_FEATURES = [
    "ion_matches",
    "ion_match_intensity",
    "complementary_ion_count",
    "max_ion_gap",
    "spectral_angle",
    "xcorr",
]

BEAM_FEATURES = ["margin", "median_margin", "entropy", "z-score", "edit_distance"]

TOKEN_FEATURES = ["min_token_probability", "std_token_probability"]

SKEWED_FEATURES = {"irt_error"}

_NICE_LABELS: dict[str, str] = {
    "ion_matches": "Ion match rate",
    "ion_match_intensity": "Ion match intensity",
    "complementary_ion_count": "Complementary ion count",
    "max_ion_gap": "Max ion gap",
    "spectral_angle": "Spectral angle",
    "xcorr": "Cross-correlation (XCorr)",
    "mass_error_ppm": "Precursor mass error (ppm)",
    "log_abs_mass_error_ppm": "Log-absolute mass error (ln ppm)",
    "mass_error_da": "Precursor mass error (Da)",
    "irt_error": "iRT prediction error",
    "confidence": "Model confidence",
    "margin": "Beam margin",
    "median_margin": "Beam median margin",
    "entropy": "Beam entropy",
    "z-score": "Beam z-score",
    "edit_distance": "Runner-up edit distance",
    "min_token_probability": "Min. token probability",
    "std_token_probability": "Std. token probability",
    "predicted_irt": "Regressor-predicted iRT",
    "irt": "Koina-predicted iRT",
    "retention_time": "Retention time (s)",
}


def _nice_label(col: str) -> str:
    return _NICE_LABELS.get(col, col.replace("_", " ").capitalize())


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _save_fig(fig: plt.Figure, name: str, output_dir: Path) -> None:
    base = output_dir / name
    fig.savefig(f"{base}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"{base}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _style_ax(ax: plt.Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.8)


def _auto_ylim(df: pd.DataFrame, feature: str):
    if feature in SKEWED_FEATURES:
        q99 = df[feature].quantile(0.99)
        q01 = df[feature].quantile(0.01)
        margin = (q99 - q01) * 0.1
        return (q01 - margin, q99 + margin)
    return None


def plot_feature_vs_confidence(
    df: pd.DataFrame,
    feature: str,
    title: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter plot of a feature against model confidence, coloured by class."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in HUE_ORDER:
        subset = df[df["hue"] == label]
        ax.scatter(
            subset["confidence"],
            subset[feature],
            c=HUE_PALETTE[label],
            label=label,
            alpha=0.3,
            s=10,
            rasterized=True,
        )
    ax.set_xlabel(_nice_label("confidence"))
    ax.set_ylabel(_nice_label(feature))
    if ylim is not None:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title)
    ax.legend()
    _style_ax(ax)
    fig.tight_layout()
    return fig, ax


def _plot_peak_normalised_kde(
    ax: plt.Axes,
    subset: pd.Series,
    colour: str,
    label: str,
    fill: bool,
    clip: tuple[float, float] | None,
) -> None:
    """Plot a peak-normalised KDE curve on *ax*."""
    kde = gaussian_kde(subset)
    lo = subset.min() if clip is None else clip[0]
    hi = subset.max() if clip is None else clip[1]
    xs = np.linspace(lo, hi, 500)
    ys = kde(xs)
    ys /= ys.max()
    ax.plot(xs, ys, color=colour, label=label, linewidth=1.5)
    if fill:
        ax.fill_between(xs, ys, alpha=0.3, color=colour)


def plot_kde_by_class(
    df: pd.DataFrame,
    feature: str,
    title: str | None = None,
    fill: bool = True,
    clip: tuple[float, float] | None = None,
    peak_normalise: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """KDE density plot of a feature split by correct/incorrect class."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in HUE_ORDER:
        subset = df[df["hue"] == label][feature].dropna()
        if len(subset) < 2:
            continue
        if peak_normalise:
            _plot_peak_normalised_kde(ax, subset, HUE_PALETTE[label], label, fill, clip)
        else:
            kw: dict = {}
            if clip is not None:
                kw["clip"] = clip
            sns.kdeplot(
                subset,
                ax=ax,
                color=HUE_PALETTE[label],
                label=label,
                fill=fill,
                alpha=0.3,
                linewidth=1.5,
                **kw,
            )
    ax.set_xlabel(_nice_label(feature))
    ax.set_ylabel("Peak-normalised density" if peak_normalise else "Density")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper center")
    _style_ax(ax)
    fig.tight_layout()
    return fig, ax


def plot_mirror_spectrum(
    obs_mz,
    obs_int,
    theo_mz,
    theo_int,
    annotations,
    title: str,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Mirror plot comparing observed vs predicted spectra."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        assert ax is not None
        fig = ax.get_figure()

    obs_int_norm = np.array(obs_int) / max(obs_int) * 100
    theo_int_norm = np.array(theo_int) / max(theo_int) * 100

    ax.vlines(obs_mz, 0, obs_int_norm, color=_OBS_COLOUR, linewidth=1.8)
    ax.vlines(theo_mz, 0, -theo_int_norm, color=_THEO_COLOUR, linewidth=1.8)

    if annotations is not None:
        for mz_val, intensity_val, ann in zip(theo_mz, theo_int_norm, annotations):
            if intensity_val > 10:
                label_text = ann.decode() if isinstance(ann, bytes) else str(ann)
                ax.annotate(
                    label_text,
                    (mz_val, -intensity_val),
                    fontsize=8,
                    ha="center",
                    va="top",
                    rotation=90,
                    color=_THEO_COLOUR,
                )

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("m/z")
    ax.set_ylabel("Relative intensity (%)")
    ax.set_title(title)

    ax.text(
        0.99,
        0.95,
        "Observed",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=_OBS_COLOUR,
        fontweight="bold",
    )
    ax.text(
        0.99,
        0.05,
        "Predicted",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=_THEO_COLOUR,
        fontweight="bold",
    )

    _style_ax(ax)
    if own_fig:
        fig.tight_layout()
    return fig, ax


def compute_discriminative_stats(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Compute AUROC, KS statistic, and Cohen's d for each feature."""
    results = []
    labels = df["correct"].astype(int)
    for feat in features:
        vals = df[feat].dropna()
        valid_mask = df[feat].notna()
        valid_labels = labels[valid_mask]
        valid_vals = vals
        if len(valid_vals) < 10 or valid_labels.nunique() < 2:
            results.append(
                {
                    "feature": feat,
                    "auroc": np.nan,
                    "ks_stat": np.nan,
                    "cohens_d": np.nan,
                }
            )
            continue
        try:
            auroc = roc_auc_score(valid_labels, valid_vals)
            auroc = max(auroc, 1 - auroc)
        except ValueError:
            auroc = np.nan
        correct_vals = valid_vals[valid_labels == 1]
        incorrect_vals = valid_vals[valid_labels == 0]
        ks_stat, _ = sp_stats.ks_2samp(correct_vals, incorrect_vals)
        pooled_std = np.sqrt(
            (
                (len(correct_vals) - 1) * correct_vals.std() ** 2
                + (len(incorrect_vals) - 1) * incorrect_vals.std() ** 2
            )
            / (len(correct_vals) + len(incorrect_vals) - 2)
        )
        cohens_d = (
            abs(correct_vals.mean() - incorrect_vals.mean()) / pooled_std
            if pooled_std > 0
            else np.nan
        )
        results.append(
            {"feature": feat, "auroc": auroc, "ks_stat": ks_stat, "cohens_d": cohens_d}
        )
    return (
        pd.DataFrame(results)
        .sort_values("auroc", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Token / beam stem helpers
# ---------------------------------------------------------------------------
_STEM_FIGSIZE = (10, 4.5)


def _parse_token_probs(row):
    """Extract token probabilities and residue labels from a row."""
    try:
        token_probs = np.exp(np.array(ast.literal_eval(row["token_log_probs"])))
    except (ValueError, SyntaxError):
        return np.array([]), []
    seq_str = row["prediction"]
    residues: list[str] = []
    j = 0
    while j < len(seq_str):
        if j + 1 < len(seq_str) and seq_str[j + 1] == "[":
            end = seq_str.index("]", j + 1) + 1
            residues.append(seq_str[j:end])
            j = end
        else:
            residues.append(seq_str[j])
            j += 1
    n_tokens = min(len(token_probs), len(residues))
    return token_probs[:n_tokens], residues[:n_tokens]


def _plot_token_stem(row, beam_colour: str, title_suffix: str = ""):
    """Stem plot of per-residue token probabilities for one PSM."""
    token_probs, residues = _parse_token_probs(row)
    if len(token_probs) == 0:
        return None

    n = len(token_probs)
    fig, ax = plt.subplots(figsize=_STEM_FIGSIZE)
    markerline, stemlines, baseline = ax.stem(
        range(n),
        token_probs,
        linefmt="-",
        markerfmt="o",
        basefmt="k-",
    )
    plt.setp(stemlines, color=beam_colour, linewidth=2.5)
    plt.setp(markerline, color=beam_colour, markersize=7, zorder=5)

    ax.set_xticks(range(n))
    ax.set_xticklabels(
        residues,
        fontsize=11,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("Token probability")
    ax.set_xlabel("Residue")

    charge = int(row["precursor_charge"]) if "precursor_charge" in row.index else "?"
    ax.set_title(
        f"Token probabilities for {row['prediction']}, "
        f"+{charge}, confidence={row['confidence']:.3f}{title_suffix}",
    )
    _style_ax(ax)
    fig.tight_layout()
    return fig


def _infer_charge(row) -> str:
    """Best-effort charge extraction from a beam CSV row."""
    for col in ("precursor_charge", "charge"):
        if col in row.index and pd.notna(row[col]):
            return str(int(row[col]))
    return "?"


def _plot_beam_stem(row, beam_log_prob_cols, beam_seq_cols, colour: str):
    """Stem plot of per-beam confidence for one spectrum."""
    probs: list[float] = []
    labels: list[str] = []
    for i, (lp_col, seq_col) in enumerate(zip(beam_log_prob_cols, beam_seq_cols)):
        lp = row[lp_col]
        seq = row[seq_col]
        if pd.isna(lp) or np.isinf(lp):
            continue
        probs.append(np.exp(float(lp)))
        label = str(seq) if pd.notna(seq) else f"beam {i}"
        labels.append(label)

    if len(probs) < 2:
        return None

    n = len(probs)
    fig, ax = plt.subplots(figsize=_STEM_FIGSIZE)
    markerline, stemlines, baseline = ax.stem(
        range(n),
        probs,
        linefmt="-",
        markerfmt="o",
        basefmt="k-",
    )
    plt.setp(stemlines, color=colour, linewidth=2.5)
    plt.setp(markerline, color=colour, markersize=7, zorder=5)

    ax.set_xticks(range(n))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-max(probs) * 0.03, max(probs) * 1.15)
    ax.set_ylabel("Beam confidence")
    ax.set_xlabel("Beam prediction index")
    ax.set_title(f"Beam confidence for {labels[0]}, +{_infer_charge(row)}")
    _style_ax(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Section generators — each mirrors a notebook section
# ---------------------------------------------------------------------------
def plot_confidence(df: pd.DataFrame, output_dir: Path) -> None:
    """Section 1: confidence distribution."""
    fig, ax = plot_kde_by_class(df, "confidence", title="Model confidence distribution")
    _save_fig(fig, "01a_confidence_kde", output_dir)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label in HUE_ORDER:
        subset = df[df["hue"] == label]
        ax.hist(
            subset["confidence"],
            bins=50,
            alpha=0.5,
            color=HUE_PALETTE[label],
            edgecolor="black",
            label=label,
            density=True,
        )
    ax.set_xlabel(_nice_label("confidence"))
    ax.set_ylabel("Density")
    ax.set_title("Model confidence histogram")
    ax.legend(loc="upper center")
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, "01b_confidence_histogram", output_dir)


def _mass_error_log_column(df: pd.DataFrame) -> tuple[str, str]:
    """Return (raw mass error column, log-absolute column) for plotting."""
    if "mass_error_da" in df.columns:
        return "mass_error_da", "log_abs_mass_error_da"
    if "mass_error_ppm" in df.columns:
        return "mass_error_ppm", "log_abs_mass_error_ppm"
    raise KeyError(
        "Feature matrix must contain 'mass_error_da' or 'mass_error_ppm' for mass error plots"
    )


def plot_mass_error(df: pd.DataFrame, output_dir: Path) -> None:
    """Section 2: mass error vs confidence (Da or ppm)."""
    raw_col, log_col = _mass_error_log_column(df)
    work = df.copy()
    work[log_col] = np.log(work[raw_col].abs().clip(lower=1e-12))

    fig, _ = plot_kde_by_class(
        work,
        log_col,
        title="Log-absolute precursor mass error distribution",
    )
    _save_fig(fig, "02a_mass_error_kde", output_dir)

    fig, _ = plot_feature_vs_confidence(
        work,
        log_col,
        title="Log-absolute precursor mass error vs model confidence",
    )
    _save_fig(fig, "02b_mass_error_vs_confidence", output_dir)


def plot_mirror_spectra(df_meta: pd.DataFrame, output_dir: Path) -> None:
    """Section 3: mirror plots of observed vs predicted spectra."""
    required = {
        "theoretical_mz",
        "mz_array",
        "intensity_array",
        "theoretical_intensity",
    }
    if not required.issubset(df_meta.columns):
        print("  Skipping mirror plots — missing spectrum columns in metadata.")
        return

    valid_mirror = df_meta[
        df_meta["theoretical_mz"].apply(lambda x: x is not None and len(x) > 0)
        & df_meta["mz_array"].apply(lambda x: x is not None and len(x) > 0)
    ].copy()

    if len(valid_mirror) == 0:
        print("  Skipping mirror plots — no rows with valid spectrum arrays.")
        return

    has_annotations = "theoretical_annotation" in valid_mirror.columns

    def _add_mirror_margin(ax, y_frac=0.11):
        ymin, ymax = ax.get_ylim()
        y_pad = (ymax - ymin) * y_frac
        ax.set_ylim(ymin - y_pad, ymax + y_pad)

    def _mirror_title(row) -> str:
        pred = row.get("prediction", "?")
        charge = (
            int(row["precursor_charge"]) if "precursor_charge" in row.index else "?"
        )
        return f"Observed vs predicted spectrum for {pred}, +{charge}"

    correct_high = valid_mirror[valid_mirror["correct"]].nlargest(3, "confidence")
    incorrect_low = valid_mirror[~valid_mirror["correct"]].nsmallest(3, "confidence")

    conf_middle_lo, conf_middle_hi = 0.45, 0.55
    middle_mask = valid_mirror["confidence"].between(conf_middle_lo, conf_middle_hi)
    n_correct_mid = (valid_mirror["correct"] & middle_mask).sum()
    n_incorrect_mid = (~valid_mirror["correct"] & middle_mask).sum()
    correct_middle = valid_mirror[valid_mirror["correct"] & middle_mask].sample(
        n=min(3, n_correct_mid), random_state=42
    )
    incorrect_middle = valid_mirror[~valid_mirror["correct"] & middle_mask].sample(
        n=min(3, n_incorrect_mid), random_state=42
    )

    groups = [
        (correct_high, "correct", "03a_mirror_high_conf"),
        (incorrect_low, "incorrect", "03b_mirror_low_conf"),
        (correct_middle, "correct", "03c_mirror_middle_conf_correct"),
        (incorrect_middle, "incorrect", "03d_mirror_middle_conf_incorrect"),
    ]

    for subset, _status, prefix in groups:
        for i, (_, row) in enumerate(subset.iterrows()):
            fig, ax = plt.subplots(figsize=(8, 5))
            annotations = row.get("theoretical_annotation") if has_annotations else None
            plot_mirror_spectrum(
                obs_mz=row["mz_array"],
                obs_int=row["intensity_array"],
                theo_mz=row["theoretical_mz"],
                theo_int=row["theoretical_intensity"],
                annotations=annotations,
                title=_mirror_title(row),
                ax=ax,
            )
            _add_mirror_margin(ax)
            _style_ax(ax)
            fig.tight_layout()
            _save_fig(fig, f"{prefix}_{i}", output_dir)


def plot_fragment_features(df: pd.DataFrame, output_dir: Path) -> None:
    """Section 4: fragment ion match features vs confidence."""
    available = [f for f in FRAGMENT_FEATURES if f in df.columns]
    for feat in available:
        fig, _ = plot_feature_vs_confidence(
            df,
            feat,
            title=f"{_nice_label(feat)} vs model confidence",
            ylim=_auto_ylim(df, feat),
        )
        _save_fig(fig, f"04a_fragment_{feat}_vs_confidence", output_dir)

        fig, _ = plot_kde_by_class(df, feat, title=f"{_nice_label(feat)} distribution")
        _save_fig(fig, f"04b_fragment_{feat}_kde", output_dir)


def plot_irt(df: pd.DataFrame, df_meta: pd.DataFrame | None, output_dir: Path) -> None:
    """Section 7: iRT error plots + RT scatter when metadata is available."""
    if df_meta is not None:
        has_rt = (
            "retention_time" in df_meta.columns and "predicted_irt" in df_meta.columns
        )
        has_koina_irt = "irt" in df_meta.columns

        if has_rt and has_koina_irt:
            fig, ax = plt.subplots(figsize=(8, 6))
            for label in HUE_ORDER:
                subset = df_meta[df_meta["hue"] == label]
                ax.scatter(
                    subset["retention_time"],
                    subset["irt"],
                    c=HUE_PALETTE[label],
                    label=label,
                    alpha=0.3,
                    s=10,
                    rasterized=True,
                )
            ax.set_xlabel(_nice_label("retention_time"))
            ax.set_ylabel(_nice_label("irt"))
            ax.set_title("Retention time vs Koina-predicted iRT")
            ax.legend(markerscale=3, frameon=True)
            _style_ax(ax)
            fig.tight_layout()
            _save_fig(fig, "07a_rt_vs_koina_irt", output_dir)

        if has_koina_irt and has_rt:
            fig, ax = plt.subplots(figsize=(8, 6))
            for label in HUE_ORDER:
                subset = df_meta[df_meta["hue"] == label]
                ax.scatter(
                    subset["predicted_irt"],
                    subset["irt"],
                    c=HUE_PALETTE[label],
                    label=label,
                    alpha=0.3,
                    s=10,
                    rasterized=True,
                )
            ax.set_xlabel(_nice_label("predicted_irt"))
            ax.set_ylabel(_nice_label("irt"))
            ax.set_title("Koina-predicted iRT vs regressor-predicted iRT")
            ax.legend(markerscale=3, frameon=True)
            _style_ax(ax)
            fig.tight_layout()
            _save_fig(fig, "07b_predicted_vs_koina_irt", output_dir)

    if "irt_error" not in df.columns:
        return

    irt_ylim = _auto_ylim(df, "irt_error")
    fig, _ = plot_feature_vs_confidence(
        df, "irt_error", title="iRT prediction error vs model confidence", ylim=irt_ylim
    )
    _save_fig(fig, "07c_irt_error_vs_confidence", output_dir)

    fig, _ = plot_kde_by_class(
        df,
        "irt_error",
        title="iRT prediction error distribution",
        clip=(0, df["irt_error"].quantile(0.99)),
    )
    _save_fig(fig, "07d_irt_error_kde", output_dir)


def plot_token_stems(df_meta: pd.DataFrame, output_dir: Path) -> None:
    """Section 8: token-level probability stem plots from metadata."""
    if "token_log_probs" not in df_meta.columns:
        print("  Skipping token stem plots — no token_log_probs column in metadata.")
        return
    if "prediction" not in df_meta.columns:
        print("  Skipping token stem plots — no prediction column in metadata.")
        return

    # High confidence
    high_conf_pool = df_meta[df_meta["confidence"] >= 0.9]
    high_samples = (
        high_conf_pool.sample(3, random_state=42)
        if len(high_conf_pool) >= 3
        else high_conf_pool
    )
    for i, (_, row) in enumerate(high_samples.iterrows()):
        fig = _plot_token_stem(row, _HIGH_CONF_BEAM_COLOUR)
        if fig is not None:
            _save_fig(fig, f"08a_token_stem_high_conf_{i}", output_dir)

    # Medium confidence
    med_conf_pool = df_meta[
        (df_meta["confidence"] >= 0.4) & (df_meta["confidence"] <= 0.7)
    ]
    med_samples = (
        med_conf_pool.sample(3, random_state=42)
        if len(med_conf_pool) >= 3
        else med_conf_pool
    )
    for i, (_, row) in enumerate(med_samples.iterrows()):
        fig = _plot_token_stem(row, _MED_CONF_BEAM_COLOUR)
        if fig is not None:
            _save_fig(fig, f"08b_token_stem_med_conf_{i}", output_dir)

    # Low confidence
    low_conf_pool = df_meta[df_meta["confidence"] <= 0.2]
    low_samples = (
        low_conf_pool.sample(3, random_state=42)
        if len(low_conf_pool) >= 3
        else low_conf_pool
    )
    for i, (_, row) in enumerate(low_samples.iterrows()):
        fig = _plot_token_stem(row, _LOW_CONF_BEAM_COLOUR)
        if fig is not None:
            _save_fig(fig, f"08c_token_stem_low_conf_{i}", output_dir)


def plot_beam_stems(predictions_csv: Path, output_dir: Path) -> None:
    """Section 8b: beam confidence stem plots from the predictions CSV."""
    beam_csv = pd.read_csv(predictions_csv)

    beam_log_prob_cols = sorted(
        [
            c
            for c in beam_csv.columns
            if c.startswith("predictions_log_probability_beam_")
        ],
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    beam_seq_cols = sorted(
        [
            c
            for c in beam_csv.columns
            if c.startswith("predictions_beam_")
            and "log_probability" not in c
            and "token" not in c
        ],
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )

    if not beam_log_prob_cols or not beam_seq_cols:
        print("  Skipping beam stem plots — no beam columns in predictions CSV.")
        return

    beam_csv["top_confidence"] = np.exp(beam_csv[beam_log_prob_cols[0]].astype(float))

    # Filter rows where all beams are -inf or NaN
    valid_beams = beam_csv.dropna(subset=beam_log_prob_cols, how="all").copy()
    for col in beam_log_prob_cols:
        valid_beams[col] = pd.to_numeric(valid_beams[col], errors="coerce")
    valid_beams = valid_beams[
        valid_beams[beam_log_prob_cols].apply(
            lambda row: not all(np.isinf(row) | row.isna()), axis=1
        )
    ]
    valid_beams = valid_beams[
        valid_beams[beam_log_prob_cols].apply(
            lambda row: any(np.exp(row.dropna()) > 1e-15), axis=1
        )
    ]

    if len(valid_beams) == 0:
        print("  Skipping beam stem plots — no valid beam rows after filtering.")
        return

    # High confidence beams
    high_beam = valid_beams[valid_beams["top_confidence"] >= 0.9]
    high_beam_samples = (
        high_beam.sample(3, random_state=42) if len(high_beam) >= 3 else high_beam
    )
    for i, (_, row) in enumerate(high_beam_samples.iterrows()):
        fig = _plot_beam_stem(
            row, beam_log_prob_cols, beam_seq_cols, _HIGH_CONF_BEAM_COLOUR
        )
        if fig is not None:
            _save_fig(fig, f"08d_beam_conf_high_{i}", output_dir)

    # Low confidence beams
    low_beam = valid_beams[valid_beams["top_confidence"] <= 0.2]
    low_beam_samples = (
        low_beam.sample(3, random_state=42) if len(low_beam) >= 3 else low_beam
    )
    for i, (_, row) in enumerate(low_beam_samples.iterrows()):
        fig = _plot_beam_stem(
            row, beam_log_prob_cols, beam_seq_cols, _LOW_CONF_BEAM_COLOUR
        )
        if fig is not None:
            _save_fig(fig, f"08e_beam_conf_low_{i}", output_dir)


def plot_beam_features(df: pd.DataFrame, output_dir: Path) -> None:
    """Section 9: beam search features vs confidence."""
    available = [f for f in BEAM_FEATURES if f in df.columns]
    for feat in available:
        fig, _ = plot_feature_vs_confidence(
            df, feat, title=f"{_nice_label(feat)} vs model confidence"
        )
        _save_fig(fig, f"09a_beam_{feat}_scatter", output_dir)

        fig, _ = plot_kde_by_class(df, feat, title=f"{_nice_label(feat)} distribution")
        _save_fig(fig, f"09b_beam_{feat}_kde", output_dir)


def plot_token_features(df: pd.DataFrame, output_dir: Path) -> None:
    """Section 11: token-level features."""
    if "min_token_probability" not in df.columns:
        return

    fig, _ = plot_kde_by_class(
        df, "min_token_probability", title="Min. token probability distribution"
    )
    _save_fig(fig, "11a_min_token_prob_kde", output_dir)

    fig, _ = plot_kde_by_class(
        df, "std_token_probability", title="Std. token probability distribution"
    )
    _save_fig(fig, "11b_std_token_prob_kde", output_dir)

    fig, _ = plot_feature_vs_confidence(
        df, "min_token_probability", title="Min. token probability vs confidence"
    )
    _save_fig(fig, "11c_min_token_prob_scatter", output_dir)

    fig, _ = plot_feature_vs_confidence(
        df, "std_token_probability", title="Std. token probability vs confidence"
    )
    _save_fig(fig, "11d_std_token_prob_scatter", output_dir)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label in HUE_ORDER:
        subset = df[df["hue"] == label]
        ax.scatter(
            subset["min_token_probability"],
            subset["std_token_probability"],
            c=HUE_PALETTE[label],
            label=label,
            alpha=0.3,
            s=10,
            rasterized=True,
        )
    ax.set_xlabel(_nice_label("min_token_probability"))
    ax.set_ylabel(_nice_label("std_token_probability"))
    ax.set_title("Token-level feature space")
    ax.legend(markerscale=3, frameon=True)
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, "11e_token_feature_2d", output_dir)


def _plot_pca(df: pd.DataFrame, available: list[str], output_dir: Path) -> None:
    """12f/12g — PCA scatter and loadings for calibrator features."""
    feat_df = df[available].dropna()
    if len(feat_df) < 10:
        return

    hue_pca = df.loc[feat_df.index, "hue"].values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(feat_df.values)
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(x_scaled)

    fig, ax = plt.subplots(figsize=(8, 7))
    for label, colour in zip(reversed(HUE_ORDER), [_INCORRECT_COLOUR, _CORRECT_COLOUR]):
        mask = hue_pca == label
        ax.scatter(
            z_pca[mask, 0],
            z_pca[mask, 1],
            c=colour,
            label=label,
            s=10,
            alpha=0.3,
            rasterized=True,
        )
    ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("PCA of calibrator features")
    ax.legend(loc="upper left")
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, "12f_pca_features", output_dir)

    pc1 = pca.components_[0]
    pc2 = pca.components_[1]
    names = [_nice_label(c) for c in available]
    order = np.argsort(np.abs(pc1))[::-1]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        y,
        pc1[order],
        color=_CORRECT_COLOUR,
        alpha=0.6,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.barh(
        y, pc2[order], color=_PALETTE[5], alpha=0.4, edgecolor="black", linewidth=0.4
    )
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("Loading value")
    ax.set_title("PCA loadings for first two principal components")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.legend(
        handles=[
            Patch(facecolor=_CORRECT_COLOUR, alpha=0.6, label="PC 1 loading"),
            Patch(facecolor=_PALETTE[5], alpha=0.4, label="PC 2 loading"),
        ],
        loc="lower right",
    )
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, "12g_pca_loadings", output_dir)


def plot_discriminative_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Section 12: discriminative stats, correlation, violins, pairplot."""
    available = [f for f in FEATURE_COLUMNS if f in df.columns]
    disc_stats = compute_discriminative_stats(df, available)

    # 12a — AUROC bar chart
    fig, ax = plt.subplots(figsize=(8, 7))
    colours = [
        _PALETTE[0] if v >= 0.7 else _NEUTRAL_COLOUR for v in disc_stats["auroc"]
    ]
    ax.barh(range(len(disc_stats)), disc_stats["auroc"], color=colours)
    ax.set_yticks(range(len(disc_stats)))
    ax.set_yticklabels([_nice_label(f) for f in disc_stats["feature"]], fontsize=9)
    ax.set_xlabel("AUROC")
    ax.set_title("Per-feature AUROC for separating correct vs incorrect")
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.invert_yaxis()
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, "12a_feature_auroc_ranking", output_dir)

    # 12b — correlation matrix
    corr = df[available].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=_diverging_cmap(),
        center=0,
        ax=ax,
        xticklabels=[_nice_label(c) for c in available],
        yticklabels=[_nice_label(c) for c in available],
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6},
        linewidths=0.5,
        square=True,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("Feature correlation matrix")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    fig.tight_layout()
    _save_fig(fig, "12b_correlation_matrix", output_dir)

    # 12c — violin plots per feature group
    feature_groups = {
        "Fragment match": [f for f in FRAGMENT_FEATURES if f in df.columns],
        "Beam search": [f for f in BEAM_FEATURES if f in df.columns],
        "Token-level": [f for f in TOKEN_FEATURES if f in df.columns],
    }
    for _group_name, group_feats in feature_groups.items():
        for feat in group_feats:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.violinplot(
                data=df,
                x="hue",
                y=feat,
                hue="hue",
                ax=ax,
                palette=HUE_PALETTE,
                order=HUE_ORDER,
                hue_order=HUE_ORDER,
                inner="quartile",
                cut=0,
                linewidth=0.8,
                legend=False,
            )
            ax.set_xlabel("")
            ax.set_ylabel(_nice_label(feat))
            ax.set_title(f"{_nice_label(feat)} by identification status")
            _style_ax(ax)
            fig.tight_layout()
            _save_fig(fig, f"12c_violin_{feat}", output_dir)

    # 12e — pairplot of top-5 features
    top5 = disc_stats.head(5)["feature"].tolist()
    sample_size = min(2000, len(df))
    df_sample = df[top5 + ["hue"]].sample(n=sample_size, random_state=42)

    g = sns.pairplot(
        df_sample,
        vars=top5,
        hue="hue",
        palette=HUE_PALETTE,
        hue_order=HUE_ORDER,
        diag_kind="kde",
        plot_kws={"alpha": 0.25, "s": 8, "rasterized": True},
        diag_kws={"fill": True, "alpha": 0.3},
        height=2.2,
    )
    g.figure.suptitle("Pairplot of top-5 discriminative features", y=1.01, fontsize=13)
    g._legend.set_title("Identification")
    for ax_row in g.axes:
        for ax_item in ax_row:
            xl = ax_item.get_xlabel()
            yl = ax_item.get_ylabel()
            if xl:
                ax_item.set_xlabel(_nice_label(xl), fontsize=7)
            if yl:
                ax_item.set_ylabel(_nice_label(yl), fontsize=7)
            _style_ax(ax_item)
    _save_fig(g.figure, "12e_pairplot_top5", output_dir)

    _plot_pca(df, available, output_dir)

    # Save discriminative stats as CSV for reference
    disc_stats.to_csv(output_dir / "discriminative_stats.csv", index=False)


def print_summary(df: pd.DataFrame) -> None:
    """Section 13: summary statistics printed to stdout."""
    available = [f for f in FEATURE_COLUMNS if f in df.columns]
    disc_stats = compute_discriminative_stats(df, available)

    print("=" * 80)
    print("DISCRIMINATIVE STATISTICS SUMMARY")
    print("=" * 80)
    n_correct = df["correct"].sum()
    n_incorrect = (~df["correct"]).sum()
    print(
        f"\nDataset: {len(df):,} spectra | {n_correct:,} correct | {n_incorrect:,} incorrect"
    )
    print(f"Class balance: {df['correct'].mean():.1%} correct\n")

    print("Per-feature discriminative power (sorted by AUROC):")
    print("-" * 80)
    print(disc_stats.to_string(index=False, float_format="%.3f"))

    print("\nTop-5 features by AUROC:")
    for _, row in disc_stats.head(5).iterrows():
        print(
            f"  {_nice_label(row['feature']):40s} AUROC={row['auroc']:.3f}  "
            f"KS={row['ks_stat']:.3f}  d={row['cohens_d']:.3f}"
        )

    print("\nBottom-5 features by AUROC:")
    for _, row in disc_stats.tail(5).iterrows():
        print(
            f"  {_nice_label(row['feature']):40s} AUROC={row['auroc']:.3f}  "
            f"KS={row['ks_stat']:.3f}  d={row['cohens_d']:.3f}"
        )

    print("\nConfidence statistics:")
    print(f"  Overall mean confidence: {df['confidence'].mean():.3f}")
    print(f"  Correct mean confidence: {df[df['correct']]['confidence'].mean():.3f}")
    print(f"  Incorrect mean confidence: {df[~df['correct']]['confidence'].mean():.3f}")
    print(
        f"  Confidence AUROC: "
        f"{roc_auc_score(df['correct'].astype(int), df['confidence']):.3f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for feature investigation plots."""
    parser = argparse.ArgumentParser(
        description="Feature investigation plots from calibrator training matrices.",
    )
    parser.add_argument(
        "--features-train",
        type=Path,
        required=True,
        help="Path to the training features parquet produced by `winnow compute-features`.",
    )
    parser.add_argument(
        "--features-val",
        type=Path,
        default=None,
        help="Optional path to validation features parquet.  When provided the train "
        "and val splits are concatenated for richer plots.",
    )
    parser.add_argument(
        "--metadata-train",
        type=Path,
        default=None,
        help="Optional path to full training metadata parquet (produced by compute-features "
        "with metadata_output_path set).  Enables mirror plots, RT scatter, and token stems.",
    )
    parser.add_argument(
        "--metadata-val",
        type=Path,
        default=None,
        help="Optional path to full validation metadata parquet.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
        help="Optional path to InstaNovo-style predictions CSV with beam columns.  "
        "Enables beam confidence stem plots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output plots.  Defaults to a `feature_investigation_plots` "
        "subdirectory next to --features-train.",
    )
    return parser.parse_args(argv)


def _load_metadata(args: argparse.Namespace) -> pd.DataFrame | None:
    """Load and concatenate metadata parquets when provided."""
    parts: list[pd.DataFrame] = []
    for path in (args.metadata_train, args.metadata_val):
        if path is not None:
            print(f"Loading metadata from {path}")
            parts.append(pl.read_parquet(path).to_pandas())
    if not parts:
        return None
    df_meta = pd.concat(parts, ignore_index=True)
    print(f"Combined metadata: {len(df_meta):,} rows, {len(df_meta.columns)} columns")
    df_meta["hue"] = df_meta["correct"].map(
        {True: HUE_LABEL_CORRECT, False: HUE_LABEL_INCORRECT}
    )
    return df_meta


def main(argv: list[str] | None = None) -> None:
    """Generate all feature investigation plots."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args(argv)

    # -- Load features (always required) --
    print(f"Loading training features from {args.features_train}")
    df = pl.read_parquet(args.features_train).to_pandas()

    if args.features_val is not None:
        print(f"Loading validation features from {args.features_val}")
        df_val = pl.read_parquet(args.features_val).to_pandas()
        df = pd.concat([df, df_val], ignore_index=True)
        print(f"Combined dataset: {len(df):,} spectra")

    df["hue"] = df["correct"].map({True: HUE_LABEL_CORRECT, False: HUE_LABEL_INCORRECT})

    # -- Load metadata (optional) --
    df_meta = _load_metadata(args)

    has_metadata = df_meta is not None
    has_predictions_csv = args.predictions_csv is not None

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.features_train.parent / "feature_investigation_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to {output_dir}/\n")

    print(f"Dataset shape: {df.shape}")
    n_correct = df["correct"].sum()
    n_incorrect = (~df["correct"]).sum()
    print(f"Correct: {n_correct:,} | Incorrect: {n_incorrect:,} | Total: {len(df):,}")
    print(f"Class balance: {df['correct'].mean():.1%} correct\n")

    n_steps = 7 + has_metadata * 2 + has_predictions_csv
    step = 0

    step += 1
    print(f"[{step}/{n_steps}] Confidence distribution...")
    plot_confidence(df, output_dir)

    step += 1
    print(f"[{step}/{n_steps}] Mass error vs confidence...")
    plot_mass_error(df, output_dir)

    if has_metadata:
        step += 1
        print(f"[{step}/{n_steps}] Mirror spectrum plots...")
        plot_mirror_spectra(df_meta, output_dir)

    step += 1
    print(f"[{step}/{n_steps}] Fragment ion match features...")
    plot_fragment_features(df, output_dir)

    step += 1
    print(f"[{step}/{n_steps}] iRT error...")
    plot_irt(df, df_meta, output_dir)

    if has_metadata:
        step += 1
        print(f"[{step}/{n_steps}] Token-level stem plots...")
        plot_token_stems(df_meta, output_dir)

    if has_predictions_csv:
        step += 1
        print(f"[{step}/{n_steps}] Beam confidence stem plots...")
        plot_beam_stems(args.predictions_csv, output_dir)

    step += 1
    print(f"[{step}/{n_steps}] Beam search features...")
    plot_beam_features(df, output_dir)

    step += 1
    print(f"[{step}/{n_steps}] Token-level features...")
    plot_token_features(df, output_dir)

    step += 1
    print(
        f"[{step}/{n_steps}] Discriminative analysis (AUROC, correlation, violins, pairplot)..."
    )
    plot_discriminative_analysis(df, output_dir)

    print()
    print_summary(df)

    print(f"\nDone — plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
