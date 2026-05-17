#!/usr/bin/env python3
"""Post-FDR overlap analysis: Winnow-filtered identifications vs database search.

For each labelled evaluation dataset, at fixed FDR thresholds (1 %, 5 %, 10 %):
  * Count retained PSMs / unique peptides.
  * Report the fraction matching database-search identifications.
  * Categorise discordant calls (partial match, near-miss edit distance,
    mass-shift / PTM candidate, fully discordant).
  * Produce Venn diagrams at the peptide level.

Optionally (``--include-unlabelled``) extends the analysis to unlabelled /
raw datasets using the ``proteome_hit`` column from
``scripts/annotate_preds_proteome_hits.py``.

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
import pandas as pd
import seaborn as sns
import typer
import yaml
from instanovo.utils.metrics import Metrics
from instanovo.utils.residues import ResidueSet
from matplotlib_venn import venn2
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

sns.set_theme(style="white", palette=_PALETTE, context="paper", font_scale=1.5)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MOD_PLUS = re.compile(r"\(\+\d+\.?\d*\)-?")
_MOD_UNIMOD = re.compile(r"\[UNIMOD:\d+\]-?")

FDR_THRESHOLDS = [0.01, 0.05, 0.10]

# Common PTM mass deltas (Da) for mass-shift categorisation
_PTM_DELTAS = {
    "oxidation": 15.995,
    "phosphorylation": 79.966,
    "deamidation": 0.984,
    "acetylation": 42.011,
    "methylation": 14.016,
    "carbamidomethyl": 57.021,
}
_PTM_TOLERANCE_DA = 0.02


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


def _strip_mods(seq: str) -> str:
    """Strip PTM annotations and normalise I -> L."""
    if not seq or not isinstance(seq, str):
        return ""
    s = _MOD_PLUS.sub("", seq)
    s = _MOD_UNIMOD.sub("", s)
    return s.replace("I", "L")


def _peptide_key(seq: str) -> str:
    """Normalised peptide key for set-level comparisons."""
    return _strip_mods(seq)


def _levenshtein(s: str, t: str) -> int:
    """Levenshtein distance for short strings."""
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


def _mass_from_sequence(seq_str: str, residue_masses: dict[str, float]) -> float | None:
    """Approximate neutral mass from a ProForma-ish string using residue masses.

    Returns None if any token is unrecognised.
    """
    key = _strip_mods(seq_str)
    if not key:
        return None
    total = 18.010565  # water
    for aa in key:
        m = residue_masses.get(aa)
        if m is None:
            return None
        total += m
    return total


# ---------------------------------------------------------------------------
# Discordance classification
# ---------------------------------------------------------------------------
DISCORDANCE_CATEGORIES = [
    "partial_match",
    "near_miss_edit_dist",
    "ptm_candidate",
    "fully_discordant",
]


def classify_discordance(
    seq_str: str,
    pred_str: str,
    num_matches: int,
    metrics: Metrics,
    residue_masses: dict[str, float],
) -> str:
    """Classify a non-exact-match PSM into an interpretable discordance category.

    Categories (checked in order — first match wins):
      1. **partial_match** — ``num_matches > 0`` (shares at least one correctly
         placed residue with the ground truth).
      2. **near_miss_edit_dist** — Levenshtein edit distance <= 2 on the
         mod-stripped, I/L-normalised sequences.
      3. **ptm_candidate** — same base (mod-stripped) sequence, or neutral mass
         difference within a common PTM delta window.
      4. **fully_discordant** — none of the above.
    """
    seq_norm = _strip_mods(seq_str)
    pred_norm = _strip_mods(pred_str)

    if num_matches > 0:
        return "partial_match"

    if seq_norm and pred_norm:
        ed = _levenshtein(seq_norm, pred_norm)
        if ed <= 2:
            return "near_miss_edit_dist"

    if seq_norm == pred_norm and seq_str != pred_str:
        return "ptm_candidate"

    seq_mass = _mass_from_sequence(seq_str, residue_masses)
    pred_mass = _mass_from_sequence(pred_str, residue_masses)
    if seq_mass is not None and pred_mass is not None:
        delta = abs(seq_mass - pred_mass)
        for _ptm_name, ptm_delta in _PTM_DELTAS.items():
            if abs(delta - ptm_delta) < _PTM_TOLERANCE_DA:
                return "ptm_candidate"

    return "fully_discordant"


# ---------------------------------------------------------------------------
# Dataset discovery & loading
# ---------------------------------------------------------------------------
def _has_required_columns(header: set[str], labelled: bool) -> bool:
    """Check whether a CSV header has the columns needed for analysis."""
    if labelled:
        return {"sequence", "prediction", "calibrated_confidence"}.issubset(header)
    required = {"prediction", "calibrated_confidence", "proteome_hit"}
    return required.issubset(header) and "sequence" not in header


def _discover_folders(root: Path, labelled: bool) -> dict[str, Path]:
    """Find subfolders with the required columns."""
    results: dict[str, Path] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        preds_csv = child / "preds_and_fdr_metrics.csv"
        if not preds_csv.is_file():
            continue
        header = set(pd.read_csv(preds_csv, nrows=0).columns.tolist())
        if _has_required_columns(header, labelled):
            results[child.name] = child
    return results


def _load_preds(folder: Path) -> pd.DataFrame:
    preds = pd.read_csv(folder / "preds_and_fdr_metrics.csv")
    meta_path = folder / "metadata.csv"
    if meta_path.is_file():
        meta = pd.read_csv(meta_path)
        if "confidence" in meta.columns and "confidence" not in preds.columns:
            preds = preds.merge(
                meta[["spectrum_id", "confidence"]], on="spectrum_id", how="left"
            )
    return preds


def _add_q_values(
    df: pd.DataFrame, conf_col: str = "calibrated_confidence"
) -> pd.DataFrame:
    if "psm_q_value" in df.columns:
        return df
    fdr = NonParametricFDRControl()
    fdr.fit(dataset=df[conf_col])
    df = fdr.add_psm_q_value(df, confidence_col=conf_col)
    return df


# ---------------------------------------------------------------------------
# Labelled overlap analysis
# ---------------------------------------------------------------------------
def _labelled_overlap(
    df: pd.DataFrame,
    dataset_name: str,
    metrics: Metrics,
    residue_masses: dict[str, float],
) -> pd.DataFrame:
    """Compute overlap summary at each FDR threshold for a labelled dataset."""
    df = _add_q_values(df)

    all_seq_keys = set(df["sequence"].dropna().map(_peptide_key))

    rows: list[dict] = []
    for fdr_t in FDR_THRESHOLDS:
        retained = df[df["psm_q_value"] <= fdr_t].copy()
        n_retained = len(retained)
        if n_retained == 0:
            rows.append(
                {
                    "dataset": dataset_name,
                    "fdr_threshold": fdr_t,
                    "n_psms_retained": 0,
                    "n_unique_peptides": 0,
                    "n_matching": 0,
                    "pct_matching": 0.0,
                    "n_discordant": 0,
                    "pct_discordant": 0.0,
                    "n_partial_match": 0,
                    "n_near_miss_edit_dist": 0,
                    "n_ptm_candidate": 0,
                    "n_fully_discordant": 0,
                }
            )
            continue

        retained["pred_key"] = retained["prediction"].map(_peptide_key)
        n_unique = retained["pred_key"].nunique()

        is_correct = retained["correct"].astype(bool)
        n_matching = int(is_correct.sum())
        n_discordant = n_retained - n_matching

        disc = retained[~is_correct].copy()
        cats = disc.apply(
            lambda r: classify_discordance(
                str(r.get("sequence", "")),
                str(r.get("prediction", "")),
                int(r.get("num_matches", 0)),
                metrics,
                residue_masses,
            ),
            axis=1,
        )
        cat_counts = cats.value_counts().to_dict() if len(cats) > 0 else {}

        rows.append(
            {
                "dataset": dataset_name,
                "fdr_threshold": fdr_t,
                "n_psms_retained": n_retained,
                "n_unique_peptides": n_unique,
                "n_db_search_peptides": len(all_seq_keys),
                "n_matching": n_matching,
                "pct_matching": round(n_matching / n_retained * 100, 2),
                "n_discordant": n_discordant,
                "pct_discordant": round(n_discordant / n_retained * 100, 2),
                "n_partial_match": cat_counts.get("partial_match", 0),
                "n_near_miss_edit_dist": cat_counts.get("near_miss_edit_dist", 0),
                "n_ptm_candidate": cat_counts.get("ptm_candidate", 0),
                "n_fully_discordant": cat_counts.get("fully_discordant", 0),
            }
        )

    return pd.DataFrame(rows)


def _plot_venn(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
    plot_format: str,
) -> None:
    """Venn diagrams at each FDR threshold: DB-search peptides vs Winnow-retained peptides."""
    df = _add_q_values(df)
    db_peptides = set(df["sequence"].dropna().map(_peptide_key))

    n_thresholds = len(FDR_THRESHOLDS)
    fig, axes = plt.subplots(1, n_thresholds, figsize=(5 * n_thresholds, 5))
    if n_thresholds == 1:
        axes = [axes]

    for ax, fdr_t in zip(axes, FDR_THRESHOLDS):
        retained = df[df["psm_q_value"] <= fdr_t]
        winnow_peptides = set(retained["prediction"].dropna().map(_peptide_key))

        if len(winnow_peptides) == 0 and len(db_peptides) == 0:
            ax.set_title(f"FDR {fdr_t:.0%}\n(no peptides)")
            ax.set_visible(False)
            continue

        venn2(
            [db_peptides, winnow_peptides],
            set_labels=("DB search", "Winnow"),
            set_colors=(_CORRECT_COLOUR, _INCORRECT_COLOUR),
            alpha=0.6,
            ax=ax,
        )
        overlap = len(db_peptides & winnow_peptides)
        only_db = len(db_peptides - winnow_peptides)
        only_winnow = len(winnow_peptides - db_peptides)
        ax.set_title(
            f"FDR {fdr_t:.0%}\n"
            f"overlap={overlap:,}  DB-only={only_db:,}  Winnow-only={only_winnow:,}"
        )
        _style_ax(ax)

    fig.suptitle(f"Peptide-level overlap — {dataset_name}", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, output_dir / f"venn_{dataset_name}", plot_format)


# ---------------------------------------------------------------------------
# Unlabelled / proteome-hit analysis
# ---------------------------------------------------------------------------
def _unlabelled_overlap(
    df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Proteome-hit fractions at each FDR threshold for an unlabelled dataset."""
    df = _add_q_values(df)
    rows: list[dict] = []
    for fdr_t in FDR_THRESHOLDS:
        retained = df[df["psm_q_value"] <= fdr_t]
        n = len(retained)
        if n == 0:
            rows.append(
                {
                    "dataset": dataset_name,
                    "fdr_threshold": fdr_t,
                    "n_retained": 0,
                    "n_proteome_hit": 0,
                    "pct_proteome_hit": 0.0,
                }
            )
            continue
        n_hit = int(retained["proteome_hit"].sum())
        rows.append(
            {
                "dataset": dataset_name,
                "fdr_threshold": fdr_t,
                "n_retained": n,
                "n_proteome_hit": n_hit,
                "pct_proteome_hit": round(n_hit / n * 100, 2),
                "n_non_hit": n - n_hit,
                "pct_non_hit": round((n - n_hit) / n * 100, 2),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_DEFAULT_PREDICTIONS_ROOT = Path("predictions/general_model")
_DEFAULT_OUTPUT_DIR = Path("analysis/fdr_overlap")


@app.command()
def main(
    predictions_root: Annotated[
        Path,
        typer.Option(help="Root directory containing winnow predict output folders."),
    ] = _DEFAULT_PREDICTIONS_ROOT,
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory for output tables and plots."),
    ] = _DEFAULT_OUTPUT_DIR,
    include_unlabelled: Annotated[
        bool,
        typer.Option(
            "--include-unlabelled",
            help="Also analyse unlabelled/raw folders with proteome_hit annotation.",
        ),
    ] = False,
    plot_format: Annotated[
        str,
        typer.Option(help="Plot format: 'pdf', 'png', or 'both'."),
    ] = "both",
    residues_config: Annotated[
        Path,
        typer.Option(help="Path to residues.yaml for InstaNovo Metrics."),
    ] = _REPO_ROOT / "winnow" / "configs" / "residues.yaml",
) -> None:
    """Post-FDR overlap analysis: Winnow identifications vs database search."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    residue_masses = _get_residue_masses()
    metrics = Metrics(
        residue_set=ResidueSet(residue_masses=residue_masses),
        isotope_error_range=(0, 1),
    )

    # --- Labelled ---
    labelled_folders = _discover_folders(predictions_root, labelled=True)
    logger.info(
        "Found %d labelled folder(s): %s",
        len(labelled_folders),
        list(labelled_folders.keys()),
    )

    all_overlap: list[pd.DataFrame] = []
    for name, folder in labelled_folders.items():
        logger.info("Labelled: %s", name)
        df = _load_preds(folder)
        overlap = _labelled_overlap(df, name, metrics, residue_masses)
        all_overlap.append(overlap)
        logger.info("\n%s", overlap.to_string(index=False))
        _plot_venn(df, name, plots_dir, plot_format)

    if all_overlap:
        combined = pd.concat(all_overlap, ignore_index=True)
        combined.to_csv(output_dir / "labelled_overlap_summary.csv", index=False)
        with open(output_dir / "labelled_overlap_summary.json", "w") as f:
            json.dump(combined.to_dict(orient="records"), f, indent=2)
        logger.info("Labelled overlap summary:\n%s", combined.to_string(index=False))

    # --- Unlabelled / proteome-hit ---
    if include_unlabelled:
        unlabelled_folders = _discover_folders(predictions_root, labelled=False)
        logger.info(
            "Found %d unlabelled folder(s): %s",
            len(unlabelled_folders),
            list(unlabelled_folders.keys()),
        )
        all_unlabelled: list[pd.DataFrame] = []
        for name, folder in unlabelled_folders.items():
            logger.info("Unlabelled: %s", name)
            df = _load_preds(folder)
            overlap = _unlabelled_overlap(df, name)
            all_unlabelled.append(overlap)
            logger.info("\n%s", overlap.to_string(index=False))

        if all_unlabelled:
            combined_u = pd.concat(all_unlabelled, ignore_index=True)
            combined_u.to_csv(
                output_dir / "unlabelled_overlap_summary.csv", index=False
            )
            with open(output_dir / "unlabelled_overlap_summary.json", "w") as f:
                json.dump(combined_u.to_dict(orient="records"), f, indent=2)
            logger.info(
                "Unlabelled overlap summary:\n%s", combined_u.to_string(index=False)
            )

    if not labelled_folders and not (include_unlabelled and unlabelled_folders):
        logger.error("No eligible folders found under %s", predictions_root)
        raise typer.Exit(code=1)

    logger.info("FDR overlap analysis complete. Output in %s", output_dir)


if __name__ == "__main__":
    app()
