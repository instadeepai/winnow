#!/usr/bin/env python3
"""Characterise up-scored false positives from Winnow calibration.

For each labelled evaluation dataset (where both ``sequence`` and ``prediction``
are available), this script quantifies the false positives that calibration
"rescues" into high-confidence regions and compares their feature profiles to
true positives.

Inputs are ``winnow predict`` output folders, each containing
``preds_and_fdr_metrics.csv`` and ``metadata.csv``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
import yaml
from instanovo.utils.metrics import Metrics
from instanovo.utils.residues import ResidueSet
from rich.logging import RichHandler

from winnow.fdr.nonparametric import NonParametricFDRControl

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    logger.addHandler(RichHandler())

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

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

TP_COLOR = _CORRECT_COLOUR
FP_COLOR = _INCORRECT_COLOUR

sns.set_theme(style="white", palette=_PALETTE, context="paper", font_scale=1.5)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MOD_PLUS = re.compile(r"\(\+\d+\.?\d*\)-?")
_MOD_UNIMOD = re.compile(r"\[UNIMOD:\d+\]-?")

FDR_THRESHOLDS = [0.01, 0.05, 0.10]

# Max PSMs per correctness panel in the raw-vs-calibrated confidence scatter.
_CONFIDENCE_SCATTER_MAX_POINTS = 10_000
_CONFIDENCE_SCATTER_RANDOM_STATE = 42

DATASET_DISPLAY_NAMES: dict[str, str] = {
    "gluc": "HeLa degradome",
    "helaqc": "HeLa single shot",
    "herceptin": "Herceptin",
    "immuno": "Immunopeptidomics-1",
    "sbrodae": "S. brodae",
    "snakevenoms": "Snake venomics",
    "tplantibodies": "Therapeutic nanobodies",
    "woundfluids": "Wound exudates",
    "PXD014877": "C. elegans",
    "PXD023064": "Immunopeptidomics-2",
    "PXD009935": "Immunopeptidomics-3",
    "PXD004732": "ProteomeTools",
    "Astral": "Astral E. coli",
}

_FOLDER_SUFFIXES = ("_annotated", "_labelled", "_raw", "_unlabelled")

FEATURE_COLUMNS_OF_INTEREST = [
    "spectral_angle",
    "xcorr",
    "ion_matches",
    "ion_match_intensity",
    "irt_error",
    "mass_error_ppm",
    "margin",
    "entropy",
    "min_token_probability",
]

_NICE_LABELS: dict[str, str] = {
    "ion_matches": "Ion match rate",
    "ion_match_intensity": "Ion match intensity",
    "complementary_ion_count": "Complementary ion count",
    "max_ion_gap": "Max ion gap",
    "spectral_angle": "Spectral angle",
    "xcorr": "Cross-correlation (XCorr)",
    "mass_error_ppm": "Precursor mass error (ppm)",
    "irt_error": "iRT prediction error",
    "confidence": "Model confidence",
    "margin": "Beam margin",
    "median_margin": "Beam median margin",
    "entropy": "Beam entropy",
    "z-score": "Beam z-score",
    "edit_distance": "Runner-up edit distance",
    "min_token_probability": "Min. token probability",
    "std_token_probability": "Std. token probability",
}


def _nice_label(col: str) -> str:
    return _NICE_LABELS.get(col, col.replace("_", " ").capitalize())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_residue_masses() -> dict[str, float]:
    config_path = _REPO_ROOT / "winnow" / "configs" / "residues.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["residue_masses"]


def _save_fig(fig: plt.Figure, base_path: Path, fmt: str = "both") -> None:
    if fmt in ("pdf", "both"):
        fig.savefig(f"{base_path}.pdf", bbox_inches="tight", dpi=300)
    if fmt in ("png", "both"):
        fig.savefig(f"{base_path}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _style_ax(ax: plt.Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.8)


def _folder_display_name(folder_name: str) -> str:
    """Map an evaluation folder name to a publication-ready dataset label."""
    key = folder_name
    for suffix in _FOLDER_SUFFIXES:
        if key.endswith(suffix):
            key = key[: -len(suffix)]
            break
    return DATASET_DISPLAY_NAMES.get(key, key)


def _subsample_psms(
    df: pd.DataFrame,
    max_points: int,
    random_state: int = _CONFIDENCE_SCATTER_RANDOM_STATE,
) -> pd.DataFrame:
    """Return up to ``max_points`` rows without replacement."""
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=random_state)


def _strip_mods(seq: str) -> str:
    if not seq or not isinstance(seq, str):
        return ""
    s = _MOD_PLUS.sub("", seq)
    s = _MOD_UNIMOD.sub("", s)
    return s.replace("I", "L")


def _discover_labelled_folders(root: Path) -> dict[str, Path]:
    """Find subfolders with labelled ``preds_and_fdr_metrics.csv``."""
    results: dict[str, Path] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        preds_csv = child / "preds_and_fdr_metrics.csv"
        if not preds_csv.is_file():
            continue
        header = pd.read_csv(preds_csv, nrows=0).columns.tolist()
        required = {"sequence", "prediction", "calibrated_confidence"}
        if not required.issubset(header):
            continue
        results[child.name] = child
    return results


def _load_dataset(folder: Path) -> pd.DataFrame:
    """Load and merge preds + metadata CSVs for a single evaluation folder."""
    preds = pd.read_csv(folder / "preds_and_fdr_metrics.csv")
    meta = pd.read_csv(folder / "metadata.csv")
    if "confidence" in meta.columns and "confidence" not in preds.columns:
        preds = preds.merge(
            meta[["spectrum_id", "confidence"]], on="spectrum_id", how="left"
        )
    feature_cols = [c for c in FEATURE_COLUMNS_OF_INTEREST if c in meta.columns]
    if feature_cols:
        merge_cols = ["spectrum_id"] + feature_cols
        preds = preds.merge(
            meta[merge_cols].drop_duplicates(subset=["spectrum_id"]),
            on="spectrum_id",
            how="left",
        )
    return preds


def _add_q_values(
    df: pd.DataFrame, conf_col: str = "calibrated_confidence"
) -> pd.DataFrame:
    """Fit non-parametric FDR and append ``psm_q_value`` if missing."""
    if "psm_q_value" in df.columns:
        return df
    fdr = NonParametricFDRControl()
    fdr.fit(dataset=df[conf_col])
    df = fdr.add_psm_q_value(df, confidence_col=conf_col)
    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _upscored_summary_table(
    df: pd.DataFrame,
    delta_threshold: float,
    dataset_name: str,
) -> pd.DataFrame:
    """Build per-FDR-threshold summary of up-scored TP / FP counts."""
    df = _add_q_values(df)
    upscored = df["delta_confidence"] > delta_threshold

    rows = []
    for fdr_t in FDR_THRESHOLDS:
        passing = df["psm_q_value"] <= fdr_t
        for label, mask in [
            ("all", pd.Series(True, index=df.index)),
            ("up-scored", upscored),
            ("not up-scored", ~upscored),
        ]:
            sub = df[mask & passing]
            n = len(sub)
            n_correct = int(sub["correct"].sum()) if "correct" in sub.columns else 0
            n_incorrect = n - n_correct
            rows.append(
                {
                    "dataset": dataset_name,
                    "fdr_threshold": fdr_t,
                    "subset": label,
                    "n_passing": n,
                    "n_correct": n_correct,
                    "n_incorrect": n_incorrect,
                    "pct_correct": round(n_correct / n * 100, 2) if n > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _plot_confidence_scatter(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
    plot_format: str,
) -> None:
    """Subsampled scatter of raw vs calibrated confidence, colored by correctness."""
    display = _folder_display_name(dataset_name)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define panels as before
    panels = [
        ("Correct", TP_COLOR, df["correct"].astype(bool)),
        ("Incorrect", FP_COLOR, ~df["correct"].astype(bool)),
    ]

    handles = []
    for label, colour, mask in panels:
        sub = df.loc[mask, ["confidence", "calibrated_confidence"]].dropna()
        n_total = len(sub)
        if n_total < 2:
            # Only skip plotting, no data for this class
            continue

        plot_df = _subsample_psms(sub, _CONFIDENCE_SCATTER_MAX_POINTS)
        handle = ax.scatter(
            plot_df["confidence"],
            plot_df["calibrated_confidence"],
            c=colour,
            s=10,
            alpha=0.3,
            rasterized=True,
            label=f"{label}",
        )
        handles.append(handle)

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
    ax.set_xlabel("Raw confidence")
    ax.set_ylabel("Calibrated confidence")
    ax.legend(loc="lower right", fontsize=9)
    _style_ax(ax)

    fig.suptitle(
        f"Raw versus calibrated confidence for {display}",
        fontsize=13,
    )
    fig.tight_layout()
    _save_fig(fig, output_dir / f"confidence_scatter_{dataset_name}", plot_format)


def _plot_feature_distributions(
    df: pd.DataFrame,
    delta_threshold: float,
    dataset_name: str,
    output_dir: Path,
    plot_format: str,
) -> None:
    """Violin plots of feature values for up-scored TPs vs up-scored FPs."""
    upscored = df[df["delta_confidence"] > delta_threshold].copy()
    if len(upscored) < 10:
        logger.warning(
            "Too few up-scored PSMs (%d) for feature plots on %s",
            len(upscored),
            dataset_name,
        )
        return

    available = [
        c
        for c in FEATURE_COLUMNS_OF_INTEREST
        if c in upscored.columns and upscored[c].notna().sum() > 10
    ]
    if not available:
        logger.warning("No feature columns available for %s", dataset_name)
        return

    upscored["label"] = upscored["correct"].map({True: "TP", False: "FP"})
    n_features = len(available)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    palette = {"TP": TP_COLOR, "FP": FP_COLOR}

    for i, col in enumerate(available):
        ax = axes[i]
        plot_data = upscored[[col, "label"]].dropna(subset=[col])
        if len(plot_data) < 5:
            ax.set_visible(False)
            continue
        sns.violinplot(
            data=plot_data,
            x="label",
            y=col,
            palette=palette,
            ax=ax,
            inner="quartile",
            cut=0,
            linewidth=0.8,
        )
        ax.set_xlabel("")
        ax.set_ylabel(_nice_label(col))
        ax.set_title(_nice_label(col))
        _style_ax(ax)

    for i in range(len(available), len(axes)):
        axes[i].set_visible(False)

    display = _folder_display_name(dataset_name)
    fig.suptitle(
        f"Feature distributions for up-scored PSMs "
        f"(calibration increase > {delta_threshold:.2f}) on {display}",
        fontsize=13,
    )
    fig.tight_layout()
    _save_fig(fig, output_dir / f"upscored_features_{dataset_name}", plot_format)


def _upscored_fp_detail(
    df: pd.DataFrame,
    delta_threshold: float,
    dataset_name: str,
    metrics: Metrics,
) -> pd.DataFrame:
    """Detailed characterisation of up-scored FPs that pass FDR thresholds."""
    df = _add_q_values(df)
    upscored_fps = df[
        (df["delta_confidence"] > delta_threshold) & (~df["correct"].astype(bool))
    ].copy()

    if len(upscored_fps) == 0:
        return pd.DataFrame()

    def _match_fraction(row: pd.Series) -> float:
        nm = row.get("num_matches", 0)
        seq = row.get("sequence", "")
        if isinstance(seq, str):
            tokens = metrics._split_peptide(seq)
        else:
            tokens = seq if seq else []
        return nm / len(tokens) if tokens else 0.0

    upscored_fps["match_fraction"] = upscored_fps.apply(_match_fraction, axis=1)

    def _edit_dist(row: pd.Series) -> int:
        s = _strip_mods(str(row.get("sequence", "")))
        p = _strip_mods(str(row.get("prediction", "")))
        if not s or not p:
            return -1
        return _levenshtein(s, p)

    upscored_fps["edit_distance_norm"] = upscored_fps.apply(_edit_dist, axis=1)

    rows = []
    for fdr_t in FDR_THRESHOLDS:
        sub = upscored_fps[upscored_fps["psm_q_value"] <= fdr_t]
        if len(sub) == 0:
            rows.append(
                {"dataset": dataset_name, "fdr_threshold": fdr_t, "n_upscored_fps": 0}
            )
            continue
        rows.append(
            {
                "dataset": dataset_name,
                "fdr_threshold": fdr_t,
                "n_upscored_fps": len(sub),
                "mean_match_fraction": round(float(sub["match_fraction"].mean()), 4),
                "median_edit_distance": int(sub["edit_distance_norm"].median()),
                "n_edit_dist_le2": int((sub["edit_distance_norm"] <= 2).sum()),
                "n_partial_match": int((sub["match_fraction"] > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def _levenshtein(s: str, t: str) -> int:
    """Simple Levenshtein distance for short peptide strings."""
    n, m = len(s), len(t)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_DEFAULT_PREDICTIONS_ROOT = Path("predictions/general_model")
_DEFAULT_OUTPUT_DIR = Path("analysis/upscored_fps")


@app.command()
def main(
    predictions_root: Annotated[
        Path,
        typer.Option(
            help="Root directory containing winnow predict output folders.",
        ),
    ] = _DEFAULT_PREDICTIONS_ROOT,
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory for output tables and plots."),
    ] = _DEFAULT_OUTPUT_DIR,
    delta_threshold: Annotated[
        float,
        typer.Option(
            help="Minimum delta (calibrated - raw) to classify a PSM as up-scored. "
            "Default 0.1 (10 percentage-point increase).",
        ),
    ] = 0.1,
    plot_format: Annotated[
        str,
        typer.Option(help="Plot format: 'pdf', 'png', or 'both'."),
    ] = "both",
    residues_config: Annotated[
        Path,
        typer.Option(help="Path to residues.yaml for InstaNovo Metrics."),
    ] = _REPO_ROOT / "winnow" / "configs" / "residues.yaml",
) -> None:
    """Characterise false positives that calibration up-scores into high-confidence regions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    residue_masses = _get_residue_masses()
    metrics = Metrics(
        residue_set=ResidueSet(residue_masses=residue_masses),
        isotope_error_range=(0, 1),
    )

    folders = _discover_labelled_folders(predictions_root)
    if not folders:
        logger.error("No labelled output folders found under %s", predictions_root)
        raise typer.Exit(code=1)

    logger.info("Found %d labelled folder(s): %s", len(folders), list(folders.keys()))

    all_summary: list[pd.DataFrame] = []
    all_detail: list[pd.DataFrame] = []

    for name, folder in folders.items():
        logger.info("Processing %s ...", name)
        df = _load_dataset(folder)

        if "confidence" not in df.columns or "calibrated_confidence" not in df.columns:
            logger.warning("Skipping %s: missing confidence columns", name)
            continue
        if "correct" not in df.columns:
            logger.warning("Skipping %s: missing 'correct' column", name)
            continue

        df["delta_confidence"] = df["calibrated_confidence"] - df["confidence"]

        logger.info(
            "  %s: %d PSMs, %d correct, delta stats: mean=%.3f, q75=%.3f",
            name,
            len(df),
            int(df["correct"].sum()),
            df["delta_confidence"].mean(),
            df["delta_confidence"].quantile(0.75),
        )

        summary = _upscored_summary_table(df, delta_threshold, name)
        all_summary.append(summary)

        _plot_confidence_scatter(df, name, plots_dir, plot_format)
        _plot_feature_distributions(df, delta_threshold, name, plots_dir, plot_format)

        detail = _upscored_fp_detail(df, delta_threshold, name, metrics)
        if len(detail) > 0:
            all_detail.append(detail)

    if all_summary:
        combined = pd.concat(all_summary, ignore_index=True)
        combined.to_csv(output_dir / "upscored_summary.csv", index=False)
        logger.info("Summary table:\n%s", combined.to_string(index=False))

        with open(output_dir / "upscored_summary.json", "w") as f:
            json.dump(combined.to_dict(orient="records"), f, indent=2)

    if all_detail:
        detail_df = pd.concat(all_detail, ignore_index=True)
        detail_df.to_csv(output_dir / "upscored_fp_detail.csv", index=False)
        logger.info("FP detail table:\n%s", detail_df.to_string(index=False))

    logger.info("Up-scored FP analysis complete. Output in %s", output_dir)


if __name__ == "__main__":
    app()
