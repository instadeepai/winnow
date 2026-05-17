#!/usr/bin/env python3
"""Analyse Winnow calibrator behaviour on out-of-distribution / novel peptides.

Two analyses demonstrate that the calibrator does not penalise peptides absent
from the standard tryptic database-search training distribution:

1. **GluC (``gluc`` subcommand)** -- The model was trained on tryptic data.
   GluC cleaves after D/E, producing peptides whose C-terminus is typically
   *not* K or R.  We classify high-confidence proteome-hit predictions by
   whether their C-terminal residue is tryptic (K/R) or non-tryptic, and show
   the calibrator assigns comparable scores to both groups.

   *Non-tryptic* is defined solely by the C-terminal residue of the
   mod-stripped, I/L-normalised prediction.  N-terminal context is not
   checked because positional information is lost in the substring proteome
   match.

2. **ProteomeTools PXD004732 (``proteometools`` subcommand)** -- Synthetic
   peptide library.  The *lcfm* set contains database-search-confirmed
   peptides; the *acfm* set contains all candidates.  For each acfm
   prediction we check whether it exactly matches, is a subsequence of, or
   shares no overlap with any lcfm peptide.  Subsequence matches are
   validated novel identifications the search engine missed.

3. **Summary (``summary`` subcommand)** -- Combines tables from both analyses
   into a single grouped bar chart.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Annotated

import ahocorasick
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import typer
from Bio import SeqIO
from scipy.stats import gaussian_kde

from winnow.fdr.nonparametric import NonParametricFDRControl

warnings.filterwarnings("ignore", module="winnow")

# ── Style — Paul Tol "bright" palette (colour-blind safe) ────────────
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
_NOVEL_COLOUR = _PALETTE[2]
_MAIN_LINE_COLOUR = _PALETTE[0]
_RAW_LINE_COLOUR = _PALETTE[5]
_IDEAL_LINE_COLOUR = _PALETTE[6]

sns.set_theme(style="white", palette=_PALETTE, context="paper", font_scale=1.5)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MOD_PLUS = re.compile(r"\(\+\d+\.?\d*\)")
_MOD_UNIMOD = re.compile(r"\[UNIMOD:\d+\]-?")
_PROTEOME_JOIN_SEP = "\x1f"

FDR_THRESHOLDS = [0.01, 0.05, 0.10]

FEATURE_COLUMNS = [
    "spectral_angle",
    "xcorr",
    "ion_matches",
    "ion_match_intensity",
    "irt_error",
    "mass_error_ppm",
]

DATASET_DISPLAY_NAMES: dict[str, str] = {
    "gluc": "HeLa degradome",
    "PXD004732": "ProteomeTools",
}

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


# ── Shared helpers ────────────────────────────────────────────────────


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


def _annotate_grouped_bars(
    ax: plt.Axes,
    bar_groups: list,
    *,
    y_headroom: float = 1.28,
    fontsize: int = 8,
) -> None:
    """Label grouped bars and reserve vertical space above the tallest bar."""
    max_h = 0.0
    for bar_group in bar_groups:
        for bar in bar_group:
            max_h = max(max_h, float(bar.get_height()))

    for bar_group in bar_groups:
        for bar in bar_group:
            h = float(bar.get_height())
            if h <= 0:
                continue
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )

    if max_h > 0:
        ax.set_ylim(0, max(max_h * y_headroom, 0.08))


def _strip_mods(seq: str) -> str:
    """Strip PTM annotations and normalise I -> L."""
    if not seq or not isinstance(seq, str):
        return ""
    s = _MOD_PLUS.sub("", seq)
    s = _MOD_UNIMOD.sub("", s)
    return s.replace("I", "L")


def _load_data(predictions_dir: Path) -> pl.DataFrame:
    """Load and join ``preds_and_fdr_metrics.csv`` + ``metadata.csv``."""
    preds = pl.read_csv(predictions_dir / "preds_and_fdr_metrics.csv")
    meta_path = predictions_dir / "metadata.csv"
    if meta_path.exists():
        meta = pl.read_csv(meta_path)
        preds = preds.join(meta, on="spectrum_id", how="inner")
    return preds


def _add_q_values(
    df: pd.DataFrame,
    conf_col: str = "calibrated_confidence",
) -> pd.DataFrame:
    """Fit non-parametric FDR and append ``psm_q_value`` if missing."""
    if "psm_q_value" in df.columns:
        return df
    fdr = NonParametricFDRControl()
    fdr.fit(dataset=df[conf_col])
    return fdr.add_psm_q_value(df, confidence_col=conf_col)


def _load_proteome_haystack(fasta_file: Path) -> str:
    """Load a FASTA proteome into a single string for substring matching."""
    parts: list[str] = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        s = str(record.seq).replace("I", "L")
        if s:
            parts.append(s)
    return _PROTEOME_JOIN_SEP.join(parts)


def _batch_substring_hits(
    needles: list[str],
    haystack: str,
) -> list[bool]:
    """Aho-Corasick batch substring matching."""
    n = len(needles)
    out = [False] * n
    if not haystack:
        return out

    by_needle: dict[str, list[int]] = {}
    for i, p in enumerate(needles):
        if not p:
            continue
        by_needle.setdefault(p, []).append(i)
    if not by_needle:
        return out

    auto = ahocorasick.Automaton()
    needle_for_pid: list[str] = []
    for pid, needle in enumerate(by_needle):
        auto.add_word(needle, pid)
        needle_for_pid.append(needle)
    auto.make_automaton()

    matched_pids: set[int] = set()
    for _end_idx, pid in auto.iter(haystack):
        matched_pids.add(pid)

    for pid in matched_pids:
        needle = needle_for_pid[pid]
        for row_i in by_needle[needle]:
            out[row_i] = True
    return out


def _nice_feature_label(col: str) -> str:
    labels = {
        "spectral_angle": "Spectral angle",
        "xcorr": "Cross-correlation",
        "ion_matches": "Ion match rate",
        "ion_match_intensity": "Ion match intensity",
        "irt_error": "iRT prediction error",
        "mass_error_ppm": "Mass error (ppm)",
    }
    return labels.get(col, col.replace("_", " ").capitalize())


# ── GluC analysis ─────────────────────────────────────────────────────


def _is_tryptic_cterm(seq: str) -> bool:
    """Return True if the mod-stripped C-terminal residue is K or R."""
    stripped = _strip_mods(seq)
    if not stripped:
        return False
    return stripped[-1] in ("K", "R")


def _gluc_annotate(
    df: pl.DataFrame,
    fasta_path: Path,
) -> pl.DataFrame:
    """Annotate predictions with proteome-hit and tryptic-terminus flags."""
    haystack = _load_proteome_haystack(fasta_path)

    processed = df["prediction"].map_elements(
        lambda x: _strip_mods(x) if isinstance(x, str) else "",
        return_dtype=pl.Utf8,
    )
    hits = _batch_substring_hits(processed.to_list(), haystack)
    tryptic = df["prediction"].map_elements(
        lambda x: _is_tryptic_cterm(x) if isinstance(x, str) else False,
        return_dtype=pl.Boolean,
    )
    return df.with_columns(
        pl.Series("proteome_hit", hits, dtype=pl.Boolean),
        tryptic.alias("tryptic_cterm"),
    )


def _gluc_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the tryptic-summary table at each FDR threshold."""
    df = _add_q_values(df)
    rows: list[dict] = []
    for fdr_t in FDR_THRESHOLDS:
        retained = df[df["psm_q_value"] <= fdr_t]
        n_retained = len(retained)
        hits = retained[retained["proteome_hit"]]
        n_hit = len(hits)
        tryptic_hits = hits[hits["tryptic_cterm"]]
        non_tryptic_hits = hits[~hits["tryptic_cterm"]]
        n_tryp = len(tryptic_hits)
        n_non = len(non_tryptic_hits)
        rows.append(
            {
                "fdr_threshold": fdr_t,
                "n_retained": n_retained,
                "n_proteome_hit": n_hit,
                "n_tryptic_hit": n_tryp,
                "n_non_tryptic_hit": n_non,
                "pct_non_tryptic_among_hits": (
                    round(n_non / n_hit * 100, 2) if n_hit > 0 else 0.0
                ),
                "mean_cal_conf_tryptic": (
                    round(float(tryptic_hits["calibrated_confidence"].mean()), 4)
                    if n_tryp > 0
                    else float("nan")
                ),
                "mean_cal_conf_non_tryptic": (
                    round(float(non_tryptic_hits["calibrated_confidence"].mean()), 4)
                    if n_non > 0
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_gluc_conf_by_terminus(
    df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Violin plot of calibrated confidence split by C-terminal residue."""
    df = _add_q_values(df)
    n_cols = len(FDR_THRESHOLDS)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    palette = {"Tryptic (K/R)": _MAIN_LINE_COLOUR, "Non-tryptic": _NOVEL_COLOUR}

    for ax, fdr_t in zip(axes, FDR_THRESHOLDS):
        retained = df[(df["psm_q_value"] <= fdr_t) & df["proteome_hit"]]
        if len(retained) < 5:
            ax.set_title(
                f"Too few proteome-hit PSMs at {int(fdr_t * 100)}% FDR "
                f"(n={len(retained):,})"
            )
            ax.set_visible(False)
            continue
        retained = retained.copy()
        retained["C-terminus"] = retained["tryptic_cterm"].map(
            {True: "Tryptic (K/R)", False: "Non-tryptic"},
        )
        sns.violinplot(
            data=retained,
            x="C-terminus",
            y="calibrated_confidence",
            palette=palette,
            ax=ax,
            inner="quartile",
            cut=0,
            linewidth=0.8,
        )
        ax.set_xlabel("")
        ax.set_ylabel("Calibrated confidence")
        pct = int(fdr_t * 100)
        ax.set_title(
            f"Proteome-hit identifications at {pct}% FDR (n={len(retained):,})"
        )
        ax.grid(False)
        _spine_fmt(ax)

    fig.suptitle(
        "Calibrated confidence for HeLa degradome proteome-hit PSMs\n"
        "by C-terminal residue",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, out_dir, "gluc_conf_by_terminus")


def _plot_gluc_score_histograms(
    df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Overlapping histograms at 5% FDR: tryptic vs non-tryptic."""
    df = _add_q_values(df)
    retained = df[(df["psm_q_value"] <= 0.05) & df["proteome_hit"]]
    if len(retained) < 10:
        print("  skipping gluc_score_histograms (too few PSMs)")
        return

    tryp = retained[retained["tryptic_cterm"]]["calibrated_confidence"].values
    non_tryp = retained[~retained["tryptic_cterm"]]["calibrated_confidence"].values

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = 50

    ax.hist(
        tryp,
        bins=bins,
        alpha=0.6,
        label="Tryptic (K/R)",
        density=False,
        edgecolor="black",
        color=_MAIN_LINE_COLOUR,
    )
    ax.hist(
        non_tryp,
        bins=bins,
        alpha=0.6,
        label="Non-tryptic",
        density=False,
        edgecolor="black",
        color=_NOVEL_COLOUR,
    )

    all_vals = np.concatenate([tryp, non_tryp])
    x_min, x_max = all_vals.min(), all_vals.max()
    x_grid = np.linspace(x_min, x_max, 300)
    bin_width = (x_max - x_min) / bins if bins > 1 else 1.0

    if len(tryp) > 1:
        y_tryp = gaussian_kde(tryp)(x_grid) * len(tryp) * bin_width
        ax.plot(x_grid, y_tryp, color=_MAIN_LINE_COLOUR, lw=1.5)
    if len(non_tryp) > 1:
        y_non = gaussian_kde(non_tryp)(x_grid) * len(non_tryp) * bin_width
        ax.plot(x_grid, y_non, color=_NOVEL_COLOUR, lw=1.5)

    ax.set_xlabel("Calibrated confidence")
    ax.set_ylabel("Frequency")
    ax.set_title("Calibrated confidence for HeLa degradome proteome hits at 5% FDR")
    ax.legend(loc="upper center")
    ax.grid(False)
    _spine_fmt(ax)
    fig.tight_layout()
    _save(fig, out_dir, "gluc_score_histograms")


def _gluc_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Median feature values for tryptic vs non-tryptic proteome hits at 5% FDR."""
    df = _add_q_values(df)
    retained = df[(df["psm_q_value"] <= 0.05) & df["proteome_hit"]]
    available = [c for c in FEATURE_COLUMNS if c in retained.columns]
    if not available:
        return pd.DataFrame()

    rows: list[dict] = []
    for group_name, mask in [
        ("tryptic", retained["tryptic_cterm"]),
        ("non_tryptic", ~retained["tryptic_cterm"]),
    ]:
        sub = retained[mask]
        row: dict = {"group": group_name, "n": len(sub)}
        for col in available:
            vals = sub[col].dropna()
            row[f"median_{col}"] = (
                round(float(vals.median()), 4) if len(vals) > 0 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_gluc_feature_comparison(
    feat_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Grouped bar chart of median features, tryptic vs non-tryptic."""
    if feat_df.empty:
        print("  skipping gluc_feature_comparison (no feature data)")
        return

    med_cols = [c for c in feat_df.columns if c.startswith("median_")]
    if not med_cols:
        return

    raw = feat_df[["group"] + med_cols].set_index("group")
    col_min = raw.min()
    col_max = raw.max()
    denom = col_max - col_min
    denom[denom == 0] = 1.0
    normed = (raw - col_min) / denom

    feature_labels = [_nice_feature_label(c.replace("median_", "")) for c in med_cols]
    x = np.arange(len(med_cols))
    width, gap = 0.38, 0.08

    fig, ax = plt.subplots(figsize=(9, 6.5))
    tryp_vals = (
        normed.loc["tryptic"].values
        if "tryptic" in normed.index
        else np.zeros(len(med_cols))
    )
    non_vals = (
        normed.loc["non_tryptic"].values
        if "non_tryptic" in normed.index
        else np.zeros(len(med_cols))
    )

    bars_t = ax.bar(
        x - width / 2 - gap / 2,
        tryp_vals,
        width,
        label="Tryptic (K/R)",
        color=_MAIN_LINE_COLOUR,
        edgecolor="black",
        linewidth=1,
    )
    bars_n = ax.bar(
        x + width / 2 + gap / 2,
        non_vals,
        width,
        label="Non-tryptic",
        color=_NOVEL_COLOUR,
        edgecolor="black",
        linewidth=1,
    )

    _annotate_grouped_bars(ax, [bars_t, bars_n])

    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, rotation=30, ha="right")
    ax.set_ylabel("Normalised median value")
    ax.set_title(
        "Median feature values for HeLa degradome tryptic versus "
        "non-tryptic proteome hits at 5% FDR"
    )
    ax.legend(loc="upper left")
    ax.grid(False)
    _spine_fmt(ax)
    fig.tight_layout()
    _save(fig, out_dir, "gluc_feature_comparison")


@app.command()
def gluc(
    predictions_dir: Annotated[
        Path,
        typer.Option(
            "--predictions-dir", help="winnow predict output folder for GluC raw."
        ),
    ],
    fasta: Annotated[
        Path,
        typer.Option("--fasta", help="Human proteome FASTA for substring matching."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for output tables and plots."),
    ],
) -> None:
    """Analyse calibrator behaviour on GluC (non-tryptic) peptides."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions from {predictions_dir}")
    df_pl = _load_data(predictions_dir)
    print(f"  {df_pl.height:,} rows loaded")

    print(f"Annotating with proteome hits from {fasta}")
    df_pl = _gluc_annotate(df_pl, fasta)
    n_hits = df_pl.filter(pl.col("proteome_hit")).height
    print(f"  {n_hits:,} proteome hits")

    df = df_pl.to_pandas()

    print("Building tryptic summary table")
    summary = _gluc_summary_table(df)
    summary.to_csv(output_dir / "gluc_tryptic_summary.csv", index=False)
    print(summary.to_string(index=False))

    print("Building feature comparison table")
    feat_df = _gluc_feature_table(df)
    if not feat_df.empty:
        feat_df.to_csv(output_dir / "gluc_feature_comparison.csv", index=False)

    print("Plotting")
    _plot_gluc_conf_by_terminus(df, output_dir)
    _plot_gluc_score_histograms(df, output_dir)
    _plot_gluc_feature_comparison(feat_df, output_dir)

    print(f"\nGluC analysis complete. Output in {output_dir}")


# ── ProteomeTools analysis ────────────────────────────────────────────


def _classify_predictions(
    predictions: list[str],
    lcfm_peptide_set: set[str],
    lcfm_haystack: str,
) -> list[str]:
    """Classify each prediction as exact_match / subsequence / neither.

    Uses Aho-Corasick to find which predictions are substrings of at least
    one lcfm peptide (the haystack is built by joining all lcfm peptides
    with a separator).
    """
    n = len(predictions)
    categories = ["neither"] * n

    for i, p in enumerate(predictions):
        if p in lcfm_peptide_set:
            categories[i] = "exact_match"

    remaining_indices = [i for i in range(n) if categories[i] == "neither"]
    remaining_peps = [predictions[i] for i in remaining_indices]

    if remaining_peps:
        hits = _batch_substring_hits(remaining_peps, lcfm_haystack)
        for j, idx in enumerate(remaining_indices):
            if hits[j]:
                categories[idx] = "subsequence"

    return categories


def _proteometools_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build novelty summary table at each FDR threshold."""
    df = _add_q_values(df)
    rows: list[dict] = []
    for fdr_t in FDR_THRESHOLDS:
        retained = df[df["psm_q_value"] <= fdr_t]
        n = len(retained)
        if n == 0:
            rows.append(
                {
                    "fdr_threshold": fdr_t,
                    "n_retained": 0,
                    "n_exact_match": 0,
                    "n_subsequence": 0,
                    "n_neither": 0,
                    "pct_novel_among_retained": 0.0,
                    "mean_cal_conf_exact": float("nan"),
                    "mean_cal_conf_novel": float("nan"),
                    "mean_cal_conf_neither": float("nan"),
                }
            )
            continue

        cats = retained["novelty_category"]
        n_exact = int((cats == "exact_match").sum())
        n_sub = int((cats == "subsequence").sum())
        n_neither = int((cats == "neither").sum())

        exact_conf = retained.loc[cats == "exact_match", "calibrated_confidence"]
        sub_conf = retained.loc[cats == "subsequence", "calibrated_confidence"]
        neither_conf = retained.loc[cats == "neither", "calibrated_confidence"]

        rows.append(
            {
                "fdr_threshold": fdr_t,
                "n_retained": n,
                "n_exact_match": n_exact,
                "n_subsequence": n_sub,
                "n_neither": n_neither,
                "pct_novel_among_retained": round(n_sub / n * 100, 2) if n > 0 else 0.0,
                "mean_cal_conf_exact": round(float(exact_conf.mean()), 4)
                if len(exact_conf) > 0
                else float("nan"),
                "mean_cal_conf_novel": round(float(sub_conf.mean()), 4)
                if len(sub_conf) > 0
                else float("nan"),
                "mean_cal_conf_neither": round(float(neither_conf.mean()), 4)
                if len(neither_conf) > 0
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _plot_proteometools_conf_by_category(
    df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Violin plot of calibrated confidence by novelty category."""
    df = _add_q_values(df)
    n_cols = len(FDR_THRESHOLDS)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    cat_order = ["exact_match", "subsequence", "neither"]
    cat_labels = {
        "exact_match": "Exact match",
        "subsequence": "Novel (subsequence)",
        "neither": "Neither",
    }
    palette = {
        "Exact match": _MAIN_LINE_COLOUR,
        "Novel (subsequence)": _NOVEL_COLOUR,
        "Neither": _INCORRECT_COLOUR,
    }

    for ax, fdr_t in zip(axes, FDR_THRESHOLDS):
        retained = df[df["psm_q_value"] <= fdr_t].copy()
        if len(retained) < 5:
            ax.set_title(
                f"Too few retained predictions at {int(fdr_t * 100)}% FDR "
                f"(n={len(retained):,})"
            )
            ax.set_visible(False)
            continue

        retained["Category"] = retained["novelty_category"].map(cat_labels)
        present_cats = [
            cat_labels[c]
            for c in cat_order
            if cat_labels[c] in retained["Category"].values
        ]

        sns.violinplot(
            data=retained,
            x="Category",
            y="calibrated_confidence",
            order=present_cats,
            palette=palette,
            ax=ax,
            inner="quartile",
            cut=0,
            linewidth=0.8,
        )
        ax.set_xlabel("")
        ax.set_ylabel("Calibrated confidence")
        pct = int(fdr_t * 100)
        ax.set_title(
            f"Retained ProteomeTools predictions at {pct}% FDR (n={len(retained):,})"
        )
        ax.tick_params(axis="x", rotation=20)
        ax.grid(False)
        _spine_fmt(ax)

    fig.suptitle(
        "Calibrated confidence for ProteomeTools predictions\nby novelty category",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, out_dir, "proteometools_conf_by_category")


def _plot_proteometools_hit_rate(
    df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Line plot: validated hit rate by calibrated confidence decile."""
    if len(df) < 20:
        print("  skipping proteometools_hit_rate_vs_conf (too few PSMs)")
        return

    df = df.copy()
    df["is_validated"] = df["novelty_category"].isin(["exact_match", "subsequence"])
    df["conf_decile"] = pd.qcut(
        df["calibrated_confidence"],
        q=10,
        duplicates="drop",
    )
    grouped = (
        df.groupby("conf_decile", observed=True)
        .agg(
            hit_rate=("is_validated", "mean"),
            mid=("calibrated_confidence", "mean"),
        )
        .sort_values("mid")
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        grouped["mid"],
        grouped["hit_rate"],
        color=_MAIN_LINE_COLOUR,
        linewidth=1.5,
        marker="o",
        markersize=6,
        label="Validated hit rate",
    )
    overall = float(df["is_validated"].mean())
    ax.axhline(
        overall,
        color=_IDEAL_LINE_COLOUR,
        lw=1,
        linestyle="--",
        label=f"Overall mean ({overall:.2%})",
    )
    ax.set_xlabel("Mean calibrated confidence per decile")
    ax.set_ylabel("Fraction validated (exact or subsequence)")
    ax.set_title("Validated hit rate by calibrated confidence decile for ProteomeTools")
    ax.legend(loc="lower right")
    ax.grid(False)
    _spine_fmt(ax)
    fig.tight_layout()
    _save(fig, out_dir, "proteometools_hit_rate_vs_conf")


def _proteometools_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Median features for exact / novel / neither at 5% FDR."""
    df = _add_q_values(df)
    retained = df[df["psm_q_value"] <= 0.05]
    available = [c for c in FEATURE_COLUMNS if c in retained.columns]
    if not available:
        return pd.DataFrame()

    rows: list[dict] = []
    for cat in ["exact_match", "subsequence", "neither"]:
        sub = retained[retained["novelty_category"] == cat]
        row: dict = {"category": cat, "n": len(sub)}
        for col in available:
            vals = sub[col].dropna()
            row[f"median_{col}"] = (
                round(float(vals.median()), 4) if len(vals) > 0 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_proteometools_feature_comparison(
    feat_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Grouped bar chart of median features by category."""
    if feat_df.empty:
        print("  skipping proteometools_feature_comparison (no feature data)")
        return

    med_cols = [c for c in feat_df.columns if c.startswith("median_")]
    if not med_cols:
        return

    raw = feat_df[["category"] + med_cols].set_index("category")
    col_min = raw.min()
    col_max = raw.max()
    denom = col_max - col_min
    denom[denom == 0] = 1.0
    normed = (raw - col_min) / denom

    feature_labels = [_nice_feature_label(c.replace("median_", "")) for c in med_cols]
    x = np.arange(len(med_cols))
    n_groups = len(normed)
    total_width = 0.7
    bar_w = total_width / n_groups

    colours = {
        "exact_match": _MAIN_LINE_COLOUR,
        "subsequence": _NOVEL_COLOUR,
        "neither": _INCORRECT_COLOUR,
    }
    labels = {
        "exact_match": "Exact match",
        "subsequence": "Novel (subsequence)",
        "neither": "Neither",
    }

    fig, ax = plt.subplots(figsize=(9, 6.5))
    all_bars = []
    for i, cat in enumerate(normed.index):
        offset = (i - (n_groups - 1) / 2) * bar_w
        vals = normed.loc[cat].values
        bars = ax.bar(
            x + offset,
            vals,
            bar_w,
            label=labels.get(cat, cat),
            color=colours.get(cat, _PALETTE[i % len(_PALETTE)]),
            edgecolor="black",
            linewidth=1,
        )
        all_bars.append(bars)

    _annotate_grouped_bars(ax, all_bars)

    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, rotation=30, ha="right")
    ax.set_ylabel("Normalised median value")
    ax.set_title(
        "Median feature values for ProteomeTools predictions by novelty "
        "category at 5% FDR"
    )
    ax.legend(loc="upper left")
    ax.grid(False)
    _spine_fmt(ax)
    fig.tight_layout()
    _save(fig, out_dir, "proteometools_feature_comparison")


@app.command()
def proteometools(
    lcfm_predictions_dir: Annotated[
        Path,
        typer.Option(
            "--lcfm-predictions-dir",
            help="winnow predict output for PXD004732 lcfm (labelled).",
        ),
    ],
    acfm_predictions_dir: Annotated[
        Path,
        typer.Option(
            "--acfm-predictions-dir",
            help="winnow predict output for PXD004732 acfm (unlabelled).",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for output tables and plots."),
    ],
) -> None:
    """Analyse calibrator behaviour on ProteomeTools novel identifications."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading lcfm predictions from {lcfm_predictions_dir}")
    lcfm_pl = _load_data(lcfm_predictions_dir)
    print(f"  {lcfm_pl.height:,} lcfm rows")

    print(f"Loading acfm predictions from {acfm_predictions_dir}")
    acfm_pl = _load_data(acfm_predictions_dir)
    print(f"  {acfm_pl.height:,} acfm rows")

    print("Building lcfm peptide set for subsequence matching")
    lcfm_sequences = lcfm_pl["sequence"].drop_nulls().to_list()
    lcfm_peptide_set: set[str] = set()
    for seq in lcfm_sequences:
        stripped = _strip_mods(seq)
        if stripped:
            lcfm_peptide_set.add(stripped)
    print(f"  {len(lcfm_peptide_set):,} unique lcfm peptides")

    lcfm_haystack = _PROTEOME_JOIN_SEP.join(sorted(lcfm_peptide_set))

    print("Classifying acfm predictions")
    acfm_preds_raw = acfm_pl["prediction"].to_list()
    acfm_preds_stripped = [
        _strip_mods(p) if isinstance(p, str) else "" for p in acfm_preds_raw
    ]

    categories = _classify_predictions(
        acfm_preds_stripped,
        lcfm_peptide_set,
        lcfm_haystack,
    )
    acfm_pl = acfm_pl.with_columns(
        pl.Series("novelty_category", categories, dtype=pl.Utf8),
    )
    cat_counts = Counter(categories)
    print(
        f"  exact_match={cat_counts.get('exact_match', 0):,}  "
        f"subsequence={cat_counts.get('subsequence', 0):,}  "
        f"neither={cat_counts.get('neither', 0):,}"
    )

    df = acfm_pl.to_pandas()

    print("Building novelty summary table")
    summary = _proteometools_summary_table(df)
    summary.to_csv(output_dir / "proteometools_novelty_summary.csv", index=False)
    print(summary.to_string(index=False))

    print("Building feature comparison table")
    feat_df = _proteometools_feature_table(df)
    if not feat_df.empty:
        feat_df.to_csv(output_dir / "proteometools_feature_comparison.csv", index=False)

    print("Plotting")
    _plot_proteometools_conf_by_category(df, output_dir)
    _plot_proteometools_hit_rate(df, output_dir)
    _plot_proteometools_feature_comparison(feat_df, output_dir)

    print(f"\nProteomeTools analysis complete. Output in {output_dir}")


# ── Summary figure ────────────────────────────────────────────────────


@app.command()
def summary(
    gluc_dir: Annotated[
        Path,
        typer.Option("--gluc-dir", help="Output directory from the gluc subcommand."),
    ],
    proteometools_dir: Annotated[
        Path,
        typer.Option(
            "--proteometools-dir",
            help="Output directory from the proteometools subcommand.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for the combined summary figure."),
    ],
) -> None:
    """Produce a combined summary bar chart from both analyses."""
    output_dir.mkdir(parents=True, exist_ok=True)

    gluc_csv = gluc_dir / "gluc_tryptic_summary.csv"
    pt_csv = proteometools_dir / "proteometools_novelty_summary.csv"
    if not gluc_csv.is_file():
        raise typer.BadParameter(f"Missing {gluc_csv}")
    if not pt_csv.is_file():
        raise typer.BadParameter(f"Missing {pt_csv}")

    gluc_df = pd.read_csv(gluc_csv)
    pt_df = pd.read_csv(pt_csv)

    fdr_labels = [f"{int(t * 100)}%" for t in FDR_THRESHOLDS]
    x = np.arange(len(FDR_THRESHOLDS))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(10, 6))

    gluc_total = []
    gluc_novel = []
    pt_total = []
    pt_novel = []
    for fdr_t in FDR_THRESHOLDS:
        g_row = gluc_df[gluc_df["fdr_threshold"] == fdr_t]
        gluc_total.append(int(g_row["n_proteome_hit"].iloc[0]) if len(g_row) > 0 else 0)
        gluc_novel.append(
            int(g_row["n_non_tryptic_hit"].iloc[0]) if len(g_row) > 0 else 0
        )

        p_row = pt_df[pt_df["fdr_threshold"] == fdr_t]
        pt_total.append(int(p_row["n_retained"].iloc[0]) if len(p_row) > 0 else 0)
        pt_novel.append(int(p_row["n_subsequence"].iloc[0]) if len(p_row) > 0 else 0)

    bar_sets = [
        (offsets[0], gluc_total, "HeLa degradome total hits", _MAIN_LINE_COLOUR),
        (offsets[1], gluc_novel, "HeLa degradome non-tryptic hits", _NOVEL_COLOUR),
        (offsets[2], pt_total, "ProteomeTools total retained", _PALETTE[4]),
        (offsets[3], pt_novel, "ProteomeTools novel (subsequence)", _PALETTE[3]),
    ]

    for off, vals, label, colour in bar_sets:
        bars = ax.bar(
            x + off * width,
            vals,
            width,
            label=label,
            color=colour,
            edgecolor="black",
            linewidth=1,
        )
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:,}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(fdr_labels)
    ax.set_xlabel("FDR threshold")
    ax.set_ylabel("Peptide-spectrum matches")
    ax.set_title("Novel identifications retained across FDR thresholds")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(False)
    _spine_fmt(ax)
    fig.tight_layout()
    _save(fig, output_dir, "novelty_summary_bar")

    print(f"\nSummary figure saved to {output_dir}")


if __name__ == "__main__":
    app()
