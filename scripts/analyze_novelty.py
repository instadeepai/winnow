#!/usr/bin/env python3
"""Analyse Winnow calibrator behaviour on out-of-distribution / novel peptides.

Two analyses demonstrate that the calibrator does not penalise peptides absent
from the standard tryptic database-search training distribution:

1. **GluC (``gluc`` subcommand)** -- The model was trained on tryptic data.
   GluC cleaves after D/E, producing peptides whose C-terminus is typically
   *not* K or R.  We classify predictions by whether their C-terminal residue
   is tryptic (K/R) or non-tryptic, report terminus proportions before and
   after FDR, compare raw InstaNovo versus Winnow calibrated scores, and
   quantify calibration shifts (``calibrated_confidence - confidence``).

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
_MAIN_LINE_COLOUR = _PALETTE[3]
_RAW_LINE_COLOUR = _PALETTE[5]
_IDEAL_LINE_COLOUR = _PALETTE[6]

sns.set_theme(style="white", palette=_PALETTE, context="paper", font_scale=1.5)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MOD_PLUS = re.compile(r"\(\+\d+\.?\d*\)")
_MOD_UNIMOD = re.compile(r"\[UNIMOD:\d+\]-?")
_PROTEOME_JOIN_SEP = "\x1f"

FDR_THRESHOLDS = [0.01, 0.05, 0.10]

_GLUC_CALIBRATION_SCATTER_MAX_POINTS = 10_000
_GLUC_CALIBRATION_SCATTER_RANDOM_STATE = 42

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


def _subsample_psms(
    df: pd.DataFrame,
    max_points: int,
    random_state: int = _GLUC_CALIBRATION_SCATTER_RANDOM_STATE,
) -> pd.DataFrame:
    """Return up to ``max_points`` rows without replacement."""
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=random_state)


def _annotate_single_bar(ax: plt.Axes, bar, fontsize: int) -> None:
    h = float(bar.get_height())
    if h == 0 or not np.isfinite(h):
        return
    va, offset = ("bottom", 4) if h >= 0 else ("top", -4)
    ax.annotate(
        f"{h:.2f}",
        xy=(bar.get_x() + bar.get_width() / 2, h),
        xytext=(0, offset),
        textcoords="offset points",
        ha="center",
        va=va,
        fontsize=fontsize,
    )


def _set_grouped_bar_ylim(
    ax: plt.Axes,
    max_h: float,
    min_h: float,
    *,
    y_headroom: float,
    symmetric_around_zero: bool,
) -> None:
    if symmetric_around_zero:
        limit = max(abs(max_h), abs(min_h), 0.08) * y_headroom
        ax.set_ylim(-limit, limit)
        return
    if min_h < 0:
        span = max_h - min_h
        pad = span * (y_headroom - 1) / 2 if span > 0 else 0.08
        ax.set_ylim(min_h - pad, max_h + pad)
        return
    if max_h > 0:
        ax.set_ylim(0, max(max_h * y_headroom, 0.08))


def _annotate_grouped_bars(
    ax: plt.Axes,
    bar_groups: list,
    *,
    y_headroom: float = 1.28,
    fontsize: int = 8,
    symmetric_around_zero: bool = False,
) -> None:
    """Label grouped bars and set y-limits with room for annotations."""
    heights = [float(bar.get_height()) for bar_group in bar_groups for bar in bar_group]
    if not heights:
        return

    for bar_group in bar_groups:
        for bar in bar_group:
            _annotate_single_bar(ax, bar, fontsize)

    _set_grouped_bar_ylim(
        ax,
        max(heights),
        min(heights),
        y_headroom=y_headroom,
        symmetric_around_zero=symmetric_around_zero,
    )


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
        join_cols = ["spectrum_id"] + [
            c for c in meta.columns if c != "spectrum_id" and c not in preds.columns
        ]
        if len(join_cols) > 1:
            preds = preds.join(meta.select(join_cols), on="spectrum_id", how="inner")
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


def _terminus_count_row(
    sub: pd.DataFrame,
    *,
    cohort: str,
    fdr_threshold: float | None,
) -> dict:
    """Return one row of tryptic / non-tryptic counts for *sub*."""
    n = len(sub)
    n_tryp = int(sub["tryptic_cterm"].sum()) if n > 0 else 0
    n_non = n - n_tryp
    return {
        "cohort": cohort,
        "fdr_threshold": fdr_threshold,
        "n": n,
        "n_tryptic": n_tryp,
        "n_non_tryptic": n_non,
        "pct_tryptic": round(n_tryp / n * 100, 2) if n > 0 else 0.0,
        "pct_non_tryptic": round(n_non / n * 100, 2) if n > 0 else 0.0,
    }


def _gluc_terminus_proportions_table(df: pd.DataFrame) -> pd.DataFrame:
    """Tryptic versus non-tryptic counts across cohorts and FDR cutoffs."""
    df = _add_q_values(df)
    rows: list[dict] = [
        _terminus_count_row(df, cohort="all_predictions", fdr_threshold=None),
        _terminus_count_row(
            df[df["proteome_hit"]],
            cohort="proteome_hit",
            fdr_threshold=None,
        ),
    ]
    for fdr_t in FDR_THRESHOLDS:
        retained = df[df["psm_q_value"] <= fdr_t]
        rows.append(
            _terminus_count_row(
                retained,
                cohort="retained_at_fdr",
                fdr_threshold=fdr_t,
            )
        )
        rows.append(
            _terminus_count_row(
                retained[retained["proteome_hit"]],
                cohort="proteome_hit_at_fdr",
                fdr_threshold=fdr_t,
            )
        )
    return pd.DataFrame(rows)


def _gluc_calibration_delta_rows(
    sub: pd.DataFrame,
    *,
    cohort: str,
    fdr_threshold: float | None,
) -> list[dict]:
    """Build calibration-shift summary rows for tryptic and non-tryptic groups."""
    out: list[dict] = []
    for tryptic, label in ((True, "tryptic"), (False, "non_tryptic")):
        grp = sub[sub["tryptic_cterm"] == tryptic]
        n = len(grp)
        out.append(
            {
                "cohort": cohort,
                "fdr_threshold": fdr_threshold,
                "terminus_group": label,
                "n": n,
                "mean_confidence": (
                    round(float(grp["confidence"].mean()), 4) if n > 0 else float("nan")
                ),
                "mean_calibrated_confidence": (
                    round(float(grp["calibrated_confidence"].mean()), 4)
                    if n > 0
                    else float("nan")
                ),
                "mean_delta_confidence": (
                    round(float(grp["delta_confidence"].mean()), 4)
                    if n > 0
                    else float("nan")
                ),
                "median_delta_confidence": (
                    round(float(grp["delta_confidence"].median()), 4)
                    if n > 0
                    else float("nan")
                ),
            }
        )
    return out


def _gluc_calibration_delta_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean calibration shift (calibrated - raw) by terminus and cohort."""
    if "confidence" not in df.columns:
        return pd.DataFrame()

    work = _add_q_values(df.copy())
    work["delta_confidence"] = work["calibrated_confidence"] - work["confidence"]
    work = work.dropna(
        subset=["confidence", "calibrated_confidence", "delta_confidence"]
    )

    rows: list[dict] = []
    rows.extend(
        _gluc_calibration_delta_rows(work, cohort="all_predictions", fdr_threshold=None)
    )
    rows.extend(
        _gluc_calibration_delta_rows(
            work[work["proteome_hit"]],
            cohort="proteome_hit",
            fdr_threshold=None,
        )
    )
    for fdr_t in FDR_THRESHOLDS:
        retained = work[(work["psm_q_value"] <= fdr_t) & work["proteome_hit"]]
        rows.extend(
            _gluc_calibration_delta_rows(
                retained,
                cohort="proteome_hit_at_fdr",
                fdr_threshold=fdr_t,
            )
        )
    return pd.DataFrame(rows)


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


def _gluc_score_label(score_col: str) -> str:
    if score_col == "confidence":
        return "Raw InstaNovo confidence"
    return "Calibrated confidence"


def _plot_gluc_score_by_terminus(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    score_col: str,
    save_name: str,
    suptitle: str,
) -> None:
    """Violin plot of a score column split by C-terminal residue at each FDR."""
    if score_col not in df.columns:
        print(f"  skipping {save_name} (missing {score_col})")
        return

    df = _add_q_values(df)
    n_cols = len(FDR_THRESHOLDS)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    palette = {"Tryptic (K/R)": _MAIN_LINE_COLOUR, "Non-tryptic": _NOVEL_COLOUR}
    y_label = _gluc_score_label(score_col)

    for ax, fdr_t in zip(axes, FDR_THRESHOLDS):
        retained = df[(df["psm_q_value"] <= fdr_t) & df["proteome_hit"]]
        retained = retained.dropna(subset=[score_col])
        if len(retained) < 5:
            ax.set_title(
                f"Too few proteome-hit PSMs at {int(fdr_t * 100)}% FDR\n"
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
            y=score_col,
            palette=palette,
            ax=ax,
            inner="quartile",
            cut=0,
            linewidth=0.8,
        )
        ax.set_xlabel("")
        ax.set_ylabel(y_label)
        pct = int(fdr_t * 100)
        ax.set_title(
            f"Proteome-hit identifications at {pct}% FDR\n(n={len(retained):,})"
        )
        ax.grid(False)
        _spine_fmt(ax)

    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    _save(fig, out_dir, save_name)


def _plot_gluc_conf_by_terminus(df: pd.DataFrame, out_dir: Path) -> None:
    """Violin plot of calibrated confidence split by C-terminal residue."""
    _plot_gluc_score_by_terminus(
        df,
        out_dir,
        score_col="calibrated_confidence",
        save_name="gluc_conf_by_terminus",
        suptitle=(
            "Calibrated confidence for HeLa degradome proteome-hit PSMs\n"
            "by C-terminal residue"
        ),
    )


def _plot_gluc_raw_conf_by_terminus(df: pd.DataFrame, out_dir: Path) -> None:
    """Violin plot of raw InstaNovo confidence split by C-terminal residue."""
    _plot_gluc_score_by_terminus(
        df,
        out_dir,
        score_col="confidence",
        save_name="gluc_raw_conf_by_terminus",
        suptitle=(
            "Raw InstaNovo confidence for HeLa degradome proteome-hit PSMs\n"
            "by C-terminal residue"
        ),
    )


def _plot_gluc_overlapping_score_histogram(
    tryp: np.ndarray,
    non_tryp: np.ndarray,
    *,
    score_col: str,
    title: str,
    out_dir: Path,
    save_name: str,
) -> None:
    """Overlapping tryptic / non-tryptic histogram with KDE overlays."""
    if len(tryp) + len(non_tryp) < 2:
        print(f"  skipping {save_name} (too few PSMs)")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = 50

    ax.hist(
        tryp,
        bins=bins,
        alpha=0.6,
        label=f"Tryptic (K/R, n={len(tryp):,})",
        density=False,
        edgecolor="black",
        color=_MAIN_LINE_COLOUR,
    )
    ax.hist(
        non_tryp,
        bins=bins,
        alpha=0.6,
        label=f"Non-tryptic (n={len(non_tryp):,})",
        density=False,
        edgecolor="black",
        color=_NOVEL_COLOUR,
    )

    all_vals = np.concatenate([tryp, non_tryp]) if len(non_tryp) else tryp
    x_min, x_max = float(all_vals.min()), float(all_vals.max())
    if x_max <= x_min:
        x_max = x_min + 1e-6
    x_grid = np.linspace(x_min, x_max, 300)
    bin_width = (x_max - x_min) / bins if bins > 1 else 1.0

    if len(tryp) > 1:
        y_tryp = gaussian_kde(tryp)(x_grid) * len(tryp) * bin_width
        ax.plot(x_grid, y_tryp, color=_MAIN_LINE_COLOUR, lw=1.5)
    if len(non_tryp) > 1:
        y_non = gaussian_kde(non_tryp)(x_grid) * len(non_tryp) * bin_width
        ax.plot(x_grid, y_non, color=_NOVEL_COLOUR, lw=1.5)

    ax.set_xlabel(_gluc_score_label(score_col))
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend(loc="upper center")
    ax.grid(False)
    _spine_fmt(ax)
    fig.tight_layout()
    _save(fig, out_dir, save_name)


def _plot_gluc_score_histograms(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    score_col: str,
    save_name: str,
    title: str,
    subset: pd.DataFrame | None = None,
    min_psms: int = 10,
) -> None:
    """Overlapping tryptic / non-tryptic histograms for a score column."""
    if score_col not in df.columns:
        print(f"  skipping {save_name} (missing {score_col})")
        return

    retained = df if subset is None else subset
    retained = retained.dropna(subset=[score_col])
    if len(retained) < min_psms:
        print(f"  skipping {save_name} (too few PSMs)")
        return

    tryp = retained.loc[retained["tryptic_cterm"], score_col].to_numpy()
    non_tryp = retained.loc[~retained["tryptic_cterm"], score_col].to_numpy()
    _plot_gluc_overlapping_score_histogram(
        tryp,
        non_tryp,
        score_col=score_col,
        title=title,
        out_dir=out_dir,
        save_name=save_name,
    )


def _plot_gluc_score_histograms_at_fdr(df: pd.DataFrame, out_dir: Path) -> None:
    """Calibrated confidence histogram at 5% FDR for proteome hits."""
    df = _add_q_values(df)
    retained = df[(df["psm_q_value"] <= 0.05) & df["proteome_hit"]]
    _plot_gluc_score_histograms(
        df,
        out_dir,
        score_col="calibrated_confidence",
        save_name="gluc_score_histograms",
        title="Calibrated confidence for HeLa degradome proteome hits at 5% FDR",
        subset=retained,
    )


def _plot_gluc_raw_score_histograms_at_fdr(df: pd.DataFrame, out_dir: Path) -> None:
    """Raw InstaNovo confidence histogram at 5% FDR for proteome hits."""
    df = _add_q_values(df)
    retained = df[(df["psm_q_value"] <= 0.05) & df["proteome_hit"]]
    _plot_gluc_score_histograms(
        df,
        out_dir,
        score_col="confidence",
        save_name="gluc_raw_score_histograms",
        title="Raw InstaNovo confidence for HeLa degradome proteome hits at 5% FDR",
        subset=retained,
    )


def _plot_gluc_full_score_histograms(df: pd.DataFrame, out_dir: Path) -> None:
    """Full-dataset raw and calibrated histograms by C-terminal residue."""
    all_preds = df.dropna(subset=["calibrated_confidence"])
    _plot_gluc_score_histograms(
        df,
        out_dir,
        score_col="calibrated_confidence",
        save_name="gluc_calibrated_score_histogram_full",
        title=(
            "Calibrated confidence for all HeLa degradome predictions\n"
            "by C-terminal residue"
        ),
        subset=all_preds,
        min_psms=2,
    )
    if "confidence" not in df.columns:
        print("  skipping gluc_raw_score_histogram_full (missing confidence)")
        return
    raw_preds = df.dropna(subset=["confidence"])
    _plot_gluc_score_histograms(
        df,
        out_dir,
        score_col="confidence",
        save_name="gluc_raw_score_histogram_full",
        title=(
            "Raw InstaNovo confidence for all HeLa degradome predictions\n"
            "by C-terminal residue"
        ),
        subset=raw_preds,
        min_psms=2,
    )

    has_raw = "confidence" in df.columns
    panels: list[tuple[str, str, pd.DataFrame]] = [
        (
            "calibrated_confidence",
            "Calibrated confidence",
            all_preds,
        ),
    ]
    if has_raw:
        panels.append(("confidence", "Raw InstaNovo confidence", raw_preds))

    n_cols = len(panels)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    bins = 50
    for ax, (score_col, y_label, subset) in zip(axes, panels):
        tryp = subset.loc[subset["tryptic_cterm"], score_col].to_numpy()
        non_tryp = subset.loc[~subset["tryptic_cterm"], score_col].to_numpy()
        ax.hist(
            tryp,
            bins=bins,
            alpha=0.6,
            label=f"Tryptic (K/R, n={len(tryp):,})",
            density=False,
            edgecolor="black",
            color=_MAIN_LINE_COLOUR,
        )
        ax.hist(
            non_tryp,
            bins=bins,
            alpha=0.6,
            label=f"Non-tryptic (n={len(non_tryp):,})",
            density=False,
            edgecolor="black",
            color=_NOVEL_COLOUR,
        )
        ax.set_xlabel(y_label)
        ax.set_ylabel("Frequency")
        ax.set_title(f"All predictions (n={len(subset):,})")
        ax.legend(loc="upper center", fontsize=9)
        ax.grid(False)
        _spine_fmt(ax)

    fig.suptitle(
        "Score distributions for all HeLa degradome predictions by C-terminal residue",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, out_dir, "gluc_score_histogram_full_panel")


def _plot_gluc_calibration_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    """Subsampled scatter of raw versus calibrated confidence by C-terminus."""
    if "confidence" not in df.columns:
        print("  skipping gluc_calibration_scatter (missing confidence)")
        return

    work = df.dropna(subset=["confidence", "calibrated_confidence"])
    n_total = len(work)
    if n_total < 10:
        print("  skipping gluc_calibration_scatter (too few PSMs)")
        return

    plot_df = _subsample_psms(work, _GLUC_CALIBRATION_SCATTER_MAX_POINTS)
    n_show = len(plot_df)
    fig, ax = plt.subplots(figsize=(7.5, 7))
    panels = [
        ("Tryptic (K/R)", _MAIN_LINE_COLOUR, True),
        ("Non-tryptic", _NOVEL_COLOUR, False),
    ]
    for label, colour, tryptic in panels:
        sub = plot_df.loc[plot_df["tryptic_cterm"] == tryptic]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["confidence"],
            sub["calibrated_confidence"],
            c=colour,
            s=12,
            alpha=0.3,
            rasterized=True,
            label=f"{label}",
        )

    ax.plot(
        [-0.01, 1.01],
        [-0.01, 1.01],
        ls="--",
        color="black",
        lw=1,
        label="No recalibration",
        zorder=5,
    )
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("Raw InstaNovo confidence")
    ax.set_ylabel("Calibrated confidence")
    if n_show < n_total:
        ax.set_title("Raw vs calibrated confidence for all HeLa degradome predictions")
    else:
        ax.set_title("All HeLa degradome predictions")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(False)
    _spine_fmt(ax)
    fig.tight_layout()
    _save(fig, out_dir, "gluc_calibration_scatter")


def _plot_gluc_delta_by_terminus(df: pd.DataFrame, out_dir: Path) -> None:
    """Calibration shift (calibrated - raw) by C-terminal residue."""
    if "confidence" not in df.columns:
        print("  skipping gluc_delta_by_terminus (missing confidence)")
        return

    df = _add_q_values(df)
    n_cols = len(FDR_THRESHOLDS)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    palette = {"Tryptic (K/R)": _MAIN_LINE_COLOUR, "Non-tryptic": _NOVEL_COLOUR}
    work = df.copy()
    work["delta_confidence"] = work["calibrated_confidence"] - work["confidence"]

    for ax, fdr_t in zip(axes, FDR_THRESHOLDS):
        retained = work[(work["psm_q_value"] <= fdr_t) & work["proteome_hit"]]
        retained = retained.dropna(subset=["delta_confidence"])
        if len(retained) < 5:
            ax.set_title(
                f"Too few proteome-hit PSMs at {int(fdr_t * 100)}% FDR\n"
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
            y="delta_confidence",
            palette=palette,
            ax=ax,
            inner="quartile",
            cut=0,
            linewidth=0.8,
        )
        ax.axhline(0.0, ls="--", color=_IDEAL_LINE_COLOUR, lw=1)
        ax.set_xlabel("")
        ax.set_ylabel("Calibration shift (calibrated − raw)")
        pct = int(fdr_t * 100)
        ax.set_title(
            f"Proteome-hit identifications at {pct}% FDR\n(n={len(retained):,})"
        )
        ax.grid(False)
        _spine_fmt(ax)

    fig.suptitle(
        "Winnow calibration shift for HeLa degradome proteome-hit PSMs\n"
        "by C-terminal residue",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, out_dir, "gluc_delta_by_terminus")


def _pooled_feature_mean_std(retained: pd.DataFrame, col: str) -> tuple[float, float]:
    vals = retained[col].dropna()
    if len(vals) == 0:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals.iloc[0]), float("nan")
    return float(vals.mean()), float(vals.std())


def _feature_group_median_z_row(
    sub: pd.DataFrame,
    available: list[str],
    pooled: dict[str, tuple[float, float]],
) -> dict[str, float]:
    row: dict[str, float] = {}
    for col in available:
        vals = sub[col].dropna()
        if len(vals) == 0:
            row[f"median_{col}"] = float("nan")
            row[f"z_median_{col}"] = float("nan")
            continue
        med = float(vals.median())
        row[f"median_{col}"] = round(med, 4)
        mu, std = pooled[col]
        if np.isnan(std) or std == 0:
            row[f"z_median_{col}"] = float("nan")
        else:
            row[f"z_median_{col}"] = round((med - mu) / std, 4)
    return row


def _feature_median_z_score_table(
    retained: pd.DataFrame,
    available: list[str],
    groups: list[tuple[str, str, pd.Series]],
    *,
    group_col: str,
) -> pd.DataFrame:
    """Per-group feature medians and z-scores relative to *retained* PSMs."""
    pooled = {col: _pooled_feature_mean_std(retained, col) for col in available}

    rows: list[dict] = []
    for group_key, _label, mask in groups:
        sub = retained[mask]
        row: dict = {group_col: group_key, "n": len(sub)}
        row.update(_feature_group_median_z_row(sub, available, pooled))
        rows.append(row)
    return pd.DataFrame(rows)


def _gluc_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Median feature values for tryptic vs non-tryptic proteome hits at 5% FDR."""
    df = _add_q_values(df)
    retained = df[(df["psm_q_value"] <= 0.05) & df["proteome_hit"]]
    available = [c for c in FEATURE_COLUMNS if c in retained.columns]
    if not available:
        return pd.DataFrame()

    return _feature_median_z_score_table(
        retained,
        available,
        [
            ("tryptic", "Tryptic (K/R)", retained["tryptic_cterm"]),
            ("non_tryptic", "Non-tryptic", ~retained["tryptic_cterm"]),
        ],
        group_col="group",
    )


def _plot_grouped_feature_z_scores(
    feat_df: pd.DataFrame,
    *,
    group_col: str,
    out_dir: Path,
    save_name: str,
    title: str,
    group_style: list[tuple[str, str, str]],
) -> None:
    """Grouped bar chart of pooled z-scored feature medians."""
    if feat_df.empty:
        print(f"  skipping {save_name} (no feature data)")
        return

    z_cols = [c for c in feat_df.columns if c.startswith("z_median_")]
    if not z_cols:
        print(f"  skipping {save_name} (no z-scored feature columns)")
        return

    plot_df = feat_df.set_index(group_col)
    feature_labels = [_nice_feature_label(c.replace("z_median_", "")) for c in z_cols]
    x = np.arange(len(z_cols))

    present = [
        (key, label, colour)
        for key, label, colour in group_style
        if key in plot_df.index
    ]
    n_groups = len(present)
    total_width = 0.7
    bar_w = total_width / max(n_groups, 1)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.axhline(0.0, color=_IDEAL_LINE_COLOUR, lw=0.8, zorder=0)
    bar_groups = []
    for plot_i, (group_key, label, colour) in enumerate(present):
        offset = (plot_i - (n_groups - 1) / 2) * bar_w
        vals = plot_df.loc[group_key, z_cols].to_numpy(dtype=float)
        bars = ax.bar(
            x + offset,
            vals,
            bar_w,
            label=label,
            color=colour,
            edgecolor="black",
            linewidth=1,
        )
        bar_groups.append(bars)

    if not bar_groups:
        print(f"  skipping {save_name} (no groups to plot)")
        plt.close(fig)
        return

    _annotate_grouped_bars(ax, bar_groups, symmetric_around_zero=True)

    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, rotation=30, ha="right")
    ax.set_ylabel("Median z-score (vs pooled PSMs at 5% FDR)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(False)
    _spine_fmt(ax)
    fig.tight_layout()
    _save(fig, out_dir, save_name)


def _plot_gluc_feature_comparison(
    feat_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Grouped bar chart of median features, tryptic vs non-tryptic."""
    _plot_grouped_feature_z_scores(
        feat_df,
        group_col="group",
        out_dir=out_dir,
        save_name="gluc_feature_comparison",
        title=(
            "Median feature values for HeLa degradome tryptic versus "
            "non-tryptic proteome hits at 5% FDR"
        ),
        group_style=[
            ("tryptic", "Tryptic (K/R)", _MAIN_LINE_COLOUR),
            ("non_tryptic", "Non-tryptic", _NOVEL_COLOUR),
        ],
    )


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

    print("Building terminus proportion table")
    prop_df = _gluc_terminus_proportions_table(df)
    prop_df.to_csv(output_dir / "gluc_terminus_proportions.csv", index=False)
    print(prop_df.to_string(index=False))

    if "confidence" in df.columns:
        print("Building calibration shift table")
        delta_df = _gluc_calibration_delta_table(df)
        delta_df.to_csv(output_dir / "gluc_calibration_delta_summary.csv", index=False)
        print(delta_df.to_string(index=False))
    else:
        print("  skipping calibration shift table (missing raw confidence)")

    print("Building feature comparison table")
    feat_df = _gluc_feature_table(df)
    if not feat_df.empty:
        feat_df.to_csv(output_dir / "gluc_feature_comparison.csv", index=False)

    print("Plotting")
    _plot_gluc_conf_by_terminus(df, output_dir)
    _plot_gluc_raw_conf_by_terminus(df, output_dir)
    _plot_gluc_score_histograms_at_fdr(df, output_dir)
    _plot_gluc_raw_score_histograms_at_fdr(df, output_dir)
    _plot_gluc_full_score_histograms(df, output_dir)
    _plot_gluc_calibration_scatter(df, output_dir)
    _plot_gluc_delta_by_terminus(df, output_dir)
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
                f"Too few retained predictions at {int(fdr_t * 100)}% FDR\n"
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
            f"Retained ProteomeTools predictions at\n{pct}% FDR (n={len(retained):,})"
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

    return _feature_median_z_score_table(
        retained,
        available,
        [
            (
                "exact_match",
                "Exact match",
                retained["novelty_category"] == "exact_match",
            ),
            (
                "subsequence",
                "Novel (subsequence)",
                retained["novelty_category"] == "subsequence",
            ),
            ("neither", "Neither", retained["novelty_category"] == "neither"),
        ],
        group_col="category",
    )


def _plot_proteometools_feature_comparison(
    feat_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Grouped bar chart of median features by category."""
    _plot_grouped_feature_z_scores(
        feat_df,
        group_col="category",
        out_dir=out_dir,
        save_name="proteometools_feature_comparison",
        title=(
            "Median feature values for ProteomeTools predictions by novelty "
            "category at 5% FDR"
        ),
        group_style=[
            ("exact_match", "Exact match", _MAIN_LINE_COLOUR),
            ("subsequence", "Novel (subsequence)", _NOVEL_COLOUR),
            ("neither", "Neither", _INCORRECT_COLOUR),
        ],
    )


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


def _summary_count(row: pd.DataFrame, column: str) -> int:
    if len(row) == 0:
        return 0
    return int(row[column].iloc[0])


def _novelty_summary_counts(
    gluc_df: pd.DataFrame, pt_df: pd.DataFrame
) -> tuple[list[int], list[int], list[int], list[int]]:
    gluc_total, gluc_novel, pt_total, pt_novel = [], [], [], []
    for fdr_t in FDR_THRESHOLDS:
        g_row = gluc_df[gluc_df["fdr_threshold"] == fdr_t]
        gluc_total.append(_summary_count(g_row, "n_proteome_hit"))
        gluc_novel.append(_summary_count(g_row, "n_non_tryptic_hit"))

        p_row = pt_df[pt_df["fdr_threshold"] == fdr_t]
        pt_total.append(_summary_count(p_row, "n_retained"))
        pt_novel.append(_summary_count(p_row, "n_subsequence"))
    return gluc_total, gluc_novel, pt_total, pt_novel


def _annotate_positive_bars(ax: plt.Axes, bars) -> None:
    for bar in bars:
        h = bar.get_height()
        if h <= 0:
            continue
        ax.annotate(
            f"{h:,}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def _plot_novelty_summary_bars(
    ax: plt.Axes,
    gluc_total: list[int],
    gluc_novel: list[int],
    pt_total: list[int],
    pt_novel: list[int],
) -> None:
    fdr_labels = [f"{int(t * 100)}%" for t in FDR_THRESHOLDS]
    x = np.arange(len(FDR_THRESHOLDS))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

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
        _annotate_positive_bars(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(fdr_labels)
    ax.set_xlabel("FDR threshold")
    ax.set_ylabel("Peptide-spectrum matches")
    ax.set_title("Novel identifications retained across FDR thresholds")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(False)
    _spine_fmt(ax)


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

    counts = _novelty_summary_counts(pd.read_csv(gluc_csv), pd.read_csv(pt_csv))

    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_novelty_summary_bars(ax, *counts)
    fig.tight_layout()
    _save(fig, output_dir, "novelty_summary_bar")

    print(f"\nSummary figure saved to {output_dir}")


if __name__ == "__main__":
    app()
