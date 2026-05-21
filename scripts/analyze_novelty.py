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

2. **ProteomeTools-1 PXD004732 (``proteometools`` subcommand)** -- Synthetic
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
from pathlib import Path
from typing import Annotated

import ahocorasick
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    "PXD004732": "ProteomeTools-1",
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
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-group feature medians and z-scores relative to *reference* or *retained* PSMs."""
    pool_from = reference if reference is not None else retained
    pooled = {col: _pooled_feature_mean_std(pool_from, col) for col in available}

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
    z_score_ylabel: str = "Median z-score (vs pooled PSMs at 5% FDR)",
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

    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, rotation=30, ha="right")
    ax.set_ylabel(z_score_ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
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


# ── ProteomeTools-1 analysis ────────────────────────────────────────────


def _classify_predictions(
    predictions: list[str],
    fits_precursor: list[bool],
    lcfm_peptide_set: set[str],
    lcfm_haystack: str,
) -> list[str]:
    """Classify each prediction by lcfm overlap and precursor mass fit (<20 ppm).

    Uses Aho-Corasick to find which predictions are substrings of at least
    one lcfm peptide (the haystack is built by joining all lcfm peptides
    with a separator). Unmatched predictions are split by precursor fit.
    """
    n = len(predictions)
    categories = ["neither"] * n

    for i, (p, fit) in enumerate(zip(predictions, fits_precursor)):
        if p in lcfm_peptide_set:
            if fit:
                categories[i] = "exact_match_and_fits_precursor"
            else:
                categories[i] = "exact_match_and_no_precursor_fit"

    remaining_indices = [i for i in range(n) if categories[i] == "neither"]
    remaining_peps = [predictions[i] for i in remaining_indices]
    remaining_fits = [fits_precursor[i] for i in remaining_indices]

    if remaining_peps:
        hits = _batch_substring_hits(remaining_peps, lcfm_haystack)
        for j, (idx, fit) in enumerate(zip(remaining_indices, remaining_fits)):
            if hits[j]:
                if fit:
                    categories[idx] = "subsequence_and_fits_precursor"
                else:
                    categories[idx] = "subsequence_and_no_precursor_fit"

    for i in range(n):
        if categories[i] == "neither":
            if fits_precursor[i]:
                categories[i] = "neither_and_fits_precursor"
            else:
                categories[i] = "neither_and_no_precursor_fit"

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
                    "n_exact_match_and_no_precursor_fit": 0,
                    "n_exact_match_and_fits_precursor": 0,
                    "n_subsequence_and_no_precursor_fit": 0,
                    "n_subsequence_and_fits_precursor": 0,
                    "n_neither_and_no_precursor_fit": 0,
                    "n_neither_and_fits_precursor": 0,
                    "pct_exact_or_sub_and_fit_among_retained": 0.0,
                }
            )
            continue

        cats = retained["novelty_category"]
        n_exact_and_no_fit = int((cats == "exact_match_and_no_precursor_fit").sum())
        n_exact_and_fits_precursor = int(
            (cats == "exact_match_and_fits_precursor").sum()
        )
        n_sub_and_no_fit = int((cats == "subsequence_and_no_precursor_fit").sum())
        n_sub_and_fits_precursor = int((cats == "subsequence_and_fits_precursor").sum())
        n_neither_and_no_fit = int((cats == "neither_and_no_precursor_fit").sum())
        n_neither_and_fits_precursor = int((cats == "neither_and_fits_precursor").sum())

        rows.append(
            {
                "fdr_threshold": fdr_t,
                "n_retained": n,
                "n_exact_match_and_no_precursor_fit": n_exact_and_no_fit,
                "n_exact_match_and_fits_precursor": n_exact_and_fits_precursor,
                "n_subsequence_and_no_precursor_fit": n_sub_and_no_fit,
                "n_subsequence_and_fits_precursor": n_sub_and_fits_precursor,
                "n_neither_and_no_precursor_fit": n_neither_and_no_fit,
                "n_neither_and_fits_precursor": n_neither_and_fits_precursor,
                "pct_exact_or_sub_and_fit_among_retained": round(
                    (n_exact_and_fits_precursor + n_sub_and_fits_precursor) / n * 100, 2
                )
                if n > 0
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


_PROTEOMETOOLS_CONF_PLOT_LABELS: dict[str, str] = {
    "exact_match_and_no_precursor_fit": "ID-",
    "exact_match_and_fits_precursor": "ID+",
    "subsequence_and_no_precursor_fit": "Sub-",
    "subsequence_and_fits_precursor": "Sub+",
    "neither_and_no_precursor_fit": "Novel-",
    "neither_and_fits_precursor": "Novel+",
}

_PROTEOMETOOLS_CONF_CATEGORY_ORDER = list(_PROTEOMETOOLS_CONF_PLOT_LABELS.keys())


def _proteometools_conf_category_legend(ax: plt.Axes) -> None:
    handles = [
        Line2D([], [], color="none", label="ID: Exact sequence match to labelled set."),
        Line2D([], [], color="none", label="Sub: Subsequence of labelled set peptide."),
        Line2D([], [], color="none", label="Novel: No sequence match to labelled set."),
        Line2D(
            [],
            [],
            color="none",
            label="+: Matches precursor mass within 20 ppm.",
        ),
    ]
    (
        Line2D(
            [],
            [],
            color="none",
            label="-: Does not match precursor mass within 20 ppm.",
        ),
    )
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=True,
        fontsize=9,
    )


def _plot_proteometools_conf_by_category(
    df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Violin plot of calibrated confidence by novelty category (all unlabelled PSMs)."""
    plot_df = df.dropna(subset=["calibrated_confidence"]).copy()
    if len(plot_df) < 5:
        print("  skipping proteometools_conf_by_category (too few PSMs)")
        return

    palette = {
        _PROTEOMETOOLS_CONF_PLOT_LABELS[k]: _PALETTE[i]
        for i, k in enumerate(_PROTEOMETOOLS_CONF_CATEGORY_ORDER)
    }

    plot_df["Category"] = plot_df["novelty_category"].map(
        _PROTEOMETOOLS_CONF_PLOT_LABELS
    )
    present_cats = [
        _PROTEOMETOOLS_CONF_PLOT_LABELS[c]
        for c in _PROTEOMETOOLS_CONF_CATEGORY_ORDER
        if _PROTEOMETOOLS_CONF_PLOT_LABELS[c] in plot_df["Category"].values
    ]
    if not present_cats:
        print("  skipping proteometools_conf_by_category (no categories present)")
        return

    conf = plot_df["calibrated_confidence"]
    y_min, y_max = float(conf.min()), float(conf.max())

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(
        data=plot_df,
        x="Category",
        y="calibrated_confidence",
        order=present_cats,
        palette=palette,
        ax=ax,
        inner="quartile",
        cut=0,
        linewidth=0.8,
    )
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("")
    ax.set_ylabel("Calibrated confidence")
    ax.set_title(
        "Calibrated confidence for ProteomeTools-1 predictions\nby novelty category"
    )
    ax.grid(False)
    _spine_fmt(ax)
    _proteometools_conf_category_legend(ax)

    fig.tight_layout()
    _save(fig, out_dir, "proteometools_conf_by_category")


def _plot_proteometools_hit_rate(
    df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Line plot: validated hit rate (exact or subsequence match and fits precursor mass within 20ppm) by calibrated confidence decile."""
    if len(df) < 20:
        print("  skipping proteometools_hit_rate_vs_conf (too few PSMs)")
        return

    df = df.copy()
    df["is_validated"] = df["novelty_category"].isin(
        ["exact_match_and_fits_precursor", "subsequence_and_fits_precursor"]
    )
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
    ax.set_ylabel(
        "Fraction validated\n(exact match or subsequence fitting precursor mass)"
    )
    ax.set_title(
        "Validated hit rate by calibrated confidence decile for ProteomeTools-1"
    )
    ax.legend(loc="lower right")
    ax.grid(False)
    _spine_fmt(ax)
    fig.tight_layout()
    _save(fig, out_dir, "proteometools_hit_rate_vs_conf")


def _proteometools_feature_table(
    df: pd.DataFrame,
    labelled_df: pd.DataFrame,
) -> pd.DataFrame:
    """Median features for exact / novel / neither at 5% FDR."""
    df = _add_q_values(df)
    labelled_df = _add_q_values(labelled_df)
    retained = df[df["psm_q_value"] <= 0.05]
    labelled_ref = labelled_df[labelled_df["psm_q_value"] <= 0.05]
    available = [c for c in FEATURE_COLUMNS if c in retained.columns]
    if not available:
        return pd.DataFrame()

    return _feature_median_z_score_table(
        retained,
        available,
        [
            (
                "exact_match_and_fits_precursor",
                "Exact match, fits precursor mass",
                retained["novelty_category"] == "exact_match_and_fits_precursor",
            ),
            (
                "exact_match_and_no_precursor_fit",
                "Exact match, no precursor mass fit",
                retained["novelty_category"] == "exact_match_and_no_precursor_fit",
            ),
            (
                "subsequence_and_fits_precursor",
                "Subsequence, precursor mass fit",
                retained["novelty_category"] == "subsequence_and_fits_precursor",
            ),
            (
                "subsequence_and_no_precursor_fit",
                "Subsequence, no precursor mass fit",
                retained["novelty_category"] == "subsequence_and_no_precursor_fit",
            ),
            (
                "neither_and_fits_precursor",
                "Novel, precursor mass fit",
                retained["novelty_category"] == "neither_and_fits_precursor",
            ),
            (
                "neither_and_no_precursor_fit",
                "Novel, no precursor mass fit",
                retained["novelty_category"] == "neither_and_no_precursor_fit",
            ),
        ],
        group_col="category",
        reference=labelled_ref,
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
            "Median feature values for ProteomeTools-1 predictions by novelty "
            "category at 5% FDR"
        ),
        group_style=[
            (
                "exact_match_and_fits_precursor",
                "Exact match, precursor mass fit",
                _PALETTE[1],
            ),
            (
                "exact_match_and_no_precursor_fit",
                "Exact match, no precursor mass fit",
                _PALETTE[0],
            ),
            (
                "subsequence_and_fits_precursor",
                "Subsequence, precursor mass fit",
                _PALETTE[3],
            ),
            (
                "subsequence_and_no_precursor_fit",
                "Subsequence, no precursor mass fit",
                _PALETTE[2],
            ),
            ("neither_and_fits_precursor", "Novel, precursor mass fit", _PALETTE[5]),
            (
                "neither_and_no_precursor_fit",
                "Novel, no precursor mass fit",
                _PALETTE[4],
            ),
        ],
        z_score_ylabel="Median z-score (vs labelled PSMs at 5% FDR)",
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
    """Analyse calibrator behaviour on ProteomeTools-1 novel identifications."""
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

    print("Classifying unlabelled predictions")
    unlabelled_pl = acfm_pl.join(
        lcfm_pl.select("spectrum_id"), on="spectrum_id", how="anti"
    )
    unlabelled_pl = unlabelled_pl.with_columns(
        (pl.col("delta_mass_ppm").abs() < 20).alias("fits_precursor")
    )
    unlabelled_preds_raw = unlabelled_pl["prediction"].to_list()
    unlabelled_preds_stripped = [
        _strip_mods(p) if isinstance(p, str) else "" for p in unlabelled_preds_raw
    ]
    fits_precursor = unlabelled_pl["fits_precursor"].to_list()

    categories = _classify_predictions(
        unlabelled_preds_stripped,
        fits_precursor,
        lcfm_peptide_set,
        lcfm_haystack,
    )
    unlabelled_pl = unlabelled_pl.with_columns(
        pl.Series("novelty_category", categories, dtype=pl.Utf8),
    )
    df = unlabelled_pl.to_pandas()
    labelled_df = lcfm_pl.to_pandas()

    print("Building novelty summary table")
    summary = _proteometools_summary_table(df)
    summary.to_csv(output_dir / "proteometools_novelty_summary.csv", index=False)
    print(summary.to_string(index=False))

    print("Building feature comparison table")
    feat_df = _proteometools_feature_table(df, labelled_df)
    if not feat_df.empty:
        feat_df.to_csv(output_dir / "proteometools_feature_comparison.csv", index=False)

    print("Plotting")
    _plot_proteometools_conf_by_category(df, output_dir)
    _plot_proteometools_hit_rate(df, output_dir)
    _plot_proteometools_feature_comparison(feat_df, output_dir)

    print(f"\nProteomeTools-1 analysis complete. Output in {output_dir}")


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


if __name__ == "__main__":
    app()
