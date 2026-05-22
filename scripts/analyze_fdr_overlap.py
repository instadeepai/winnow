#!/usr/bin/env python3
"""Post-FDR overlap analysis: Winnow-filtered identifications vs database search.

For each project (see ``plot_eval_results.py`` CLI pattern), at 1 %, 5 %, and 10 %
nominal FDR:

  * Count retained PSMs / unique peptides vs database-search reference peptides.
  * Match rule: exact ProForma sequence after I/L equivalence; PTM differences
    are not a match.
  * Categorise discordant calls (partial match, PTM candidate, single-AA variant,
    near-miss edit distance 2-3, fully discordant).
  * Full-search Venns (raw/unlabelled Winnow vs paired annotated/labelled DB set).
  * Violin plots comparing database-matched vs fully novel retained PSMs.

Inputs are ``winnow predict`` output folders arranged as subdirectories under two
roots: an **unlabelled** tree (full-search Winnow predictions) and a **labelled**
tree (database-search reference with ``sequence``). Each subfolder must contain
``preds_and_fdr_metrics.csv``; ``metadata.csv`` is merged when present for violin
plots.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Annotated, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
import yaml
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
_NOVEL_COLOUR = _PALETTE[2]

sns.set_theme(style="white", palette=_PALETTE, context="paper", font_scale=1.5)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MOD_PLUS = re.compile(r"\(\+\d+\.?\d*\)-?")
_MOD_UNIMOD = re.compile(r"\[UNIMOD:\d+\]-?")

FDR_THRESHOLDS = [0.01, 0.05, 0.10]

DATASET_DISPLAY_NAMES: dict[str, str] = {
    "gluc": "HeLa degradome",
    "helaqc": "HeLa single shot",
    "herceptin": "Herceptin",
    "immuno": "Immunopeptidomics-1",
    "celegans": "$\\it{C.\\;elegans}$",
    "sbrodae": "$\\it{Scalindua\\;brodae}$",
    "PXD019483": "HepG2",
    "snakevenoms": "Snake venomics",
    "tplantibodies": "Therapeutic nanobodies",
    "woundfluids": "Wound exudates",
    "PXD014877": "$\\it{C.\\;elegans}$",
    "PXD023064": "Immunopeptidomics-2",
    "astral": "Astral $\\it{E.\\;coli}$",
    "01747_C01_P018218_S00_I00_N03_R1": "$\\it{Arabidopsis\\;thaliana}$",
    "20150708_QE3_UPLC8_DBJ_QC_HELA_39frac_Chymotrypsin": "HeLa chymotrypsin",
    "20151020_QE3_UPLC8_DBJ_SA_A549_Rep2_46": "Human lung",
    "20151020_QE3_UPLC8_DBJ_SA_HCT116_Rep2_46": "Human colon",
    "20170303_QEh1_LC2_FaMa_ChCh_SA_HLApI_JY_R1_exp2": "HLA Class I (JY cells)",
    "20170609_QEh1_LC1_ChCh_FAMA_SA_HLAIIp_JY_all_R1": "HLA Class II (JY cells)",
    "PXD004732": "ProteomeTools-1",
}

_FOLDER_SUFFIXES = ("_annotated", "_labelled", "_raw", "_unlabelled")
_UNLABELLED_FOLDER_SUFFIXES = ("_raw", "_unlabelled")
_LABELLED_FOLDER_SUFFIXES = ("_annotated", "_labelled")

NOVEL_FEATURE_COLUMNS: list[tuple[str, str]] = [
    ("spectral_angle", "Spectral angle"),
    ("ion_matches", "Ion match rate"),
    ("ion_match_intensity", "Ion match intensity"),
    ("precursor_charge", "Precursor charge"),
    ("mass_error_da", "Precursor mass error (Da)"),
    ("irt_error", "iRT error"),
    ("confidence", "Raw confidence"),
    ("margin", "Beam margin"),
]

_DISCORDANCE_COUNT_COLS = [
    "n_partial_match",
    "n_ptm_candidate",
    "n_single_aa_variant",
    "n_near_miss_edit_dist",
    "n_fully_discordant",
]

_PTM_DELTAS = {
    "oxidation": 15.995,
    "phosphorylation": 79.966,
    "deamidation": 0.984,
    "acetylation": 42.011,
    "methylation": 14.016,
    "carbamidomethyl": 57.021,
}
_PTM_TOLERANCE_DA = 0.02

_MIN_VIOLIN_GROUP_SIZE = 5
_MAX_VIOLIN_PSMs_PER_GROUP = 5000
_VIOLIN_SUBSAMPLE_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _display_name(key: str) -> str:
    return DATASET_DISPLAY_NAMES.get(key, key)


def _project_key_from_folder(folder_name: str) -> str:
    """Strip a known eval suffix to get the project key (e.g. ``gluc_raw`` -> ``gluc``)."""
    for suffix in _FOLDER_SUFFIXES:
        if folder_name.endswith(suffix):
            return folder_name[: -len(suffix)]
    return folder_name


def _search_space_tag_from_folder(folder_name: str) -> str:
    """Infer eval-type label for tables/plots from the unlabelled subfolder name."""
    for suffix in _UNLABELLED_FOLDER_SUFFIXES:
        if folder_name.endswith(suffix):
            return suffix[1:]  # raw | unlabelled
    return "full_search"


def _get_residue_masses() -> dict[str, float]:
    config_path = _REPO_ROOT / "winnow" / "configs" / "residues.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["residue_masses"]


def _save_fig(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(f"{base_path}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"{base_path}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _style_ax(ax: plt.Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.8)


def _sequence_match_key(seq: str) -> str:
    """Exact match key: ProForma with mods preserved, I/L equivalent."""
    if not seq or not isinstance(seq, str):
        return ""
    return seq.replace("I", "L")


def _strip_mods(seq: str) -> str:
    """Strip PTM annotations and normalise I -> L (discordance subtyping only)."""
    if not seq or not isinstance(seq, str):
        return ""
    s = _MOD_PLUS.sub("", seq)
    s = _MOD_UNIMOD.sub("", s)
    return s.replace("I", "L")


def _levenshtein(s: str, t: str) -> int:
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
    key = _strip_mods(seq_str)
    if not key:
        return None
    total = 18.010565
    for aa in key:
        m = residue_masses.get(aa)
        if m is None:
            return None
        total += m
    return total


def _db_stripped_by_length(db_stripped_list: list[str]) -> dict[int, list[str]]:
    by_len: dict[int, list[str]] = defaultdict(list)
    for s in db_stripped_list:
        by_len[len(s)].append(s)
    return by_len


def _min_edit_distance_to_db(
    pred_stripped: str, db_stripped_by_len: dict[int, list[str]]
) -> int:
    if not pred_stripped:
        return 999
    plen = len(pred_stripped)
    best = 999
    for length in range(max(0, plen - 3), plen + 4):
        for db in db_stripped_by_len.get(length, []):
            d = _levenshtein(pred_stripped, db)
            if d < best:
                best = d
                if best == 0:
                    return 0
    return best


def _build_db_reference_sets(
    db_df: pd.DataFrame,
) -> tuple[set[str], set[str], list[str], dict[int, list[str]]]:
    sequences = db_df["sequence"].dropna().astype(str)
    db_keys = {_sequence_match_key(s) for s in sequences if _sequence_match_key(s)}
    db_stripped = [_strip_mods(s) for s in sequences if _strip_mods(s)]
    db_stripped_set = set(db_stripped)
    db_stripped_unique = list(dict.fromkeys(db_stripped))
    return (
        db_keys,
        db_stripped_set,
        db_stripped_unique,
        _db_stripped_by_length(db_stripped_unique),
    )


def _is_db_match(pred: str, db_keys: set[str]) -> bool:
    return _sequence_match_key(pred) in db_keys


# ---------------------------------------------------------------------------
# Discordance classification
# ---------------------------------------------------------------------------
LabelledDiscordanceKey = tuple[str, str, int]
LabelledDiscordanceCache = dict[LabelledDiscordanceKey, str]
DbDiscordanceCache = dict[str, str]


def _discordance_from_edit_distance(seq_norm: str, pred_norm: str) -> str | None:
    if not (seq_norm and pred_norm):
        return None
    ed = _levenshtein(seq_norm, pred_norm)
    if ed == 1:
        return "single_aa_variant"
    if ed in (2, 3):
        return "near_miss_edit_dist"
    return None


def _discordance_from_ptm_mass(
    seq_str: str,
    pred_str: str,
    residue_masses: dict[str, float],
) -> str | None:
    seq_mass = _mass_from_sequence(seq_str, residue_masses)
    pred_mass = _mass_from_sequence(pred_str, residue_masses)
    if seq_mass is None or pred_mass is None:
        return None
    delta = abs(seq_mass - pred_mass)
    for ptm_delta in _PTM_DELTAS.values():
        if abs(delta - ptm_delta) < _PTM_TOLERANCE_DA:
            return "ptm_candidate"
    return None


def _labelled_discordance_key(row: pd.Series) -> LabelledDiscordanceKey:
    return (
        str(row.get("sequence", "")),
        str(row.get("prediction", "")),
        int(row.get("num_matches", 0)),
    )


def _lookup_labelled_discordance(
    row: pd.Series, cache: LabelledDiscordanceCache
) -> str:
    return cache[_labelled_discordance_key(row)]


def classify_discordance_labelled(
    seq_str: str,
    pred_str: str,
    num_matches: int,
    residue_masses: dict[str, float],
) -> str:
    """Classify a non-matching PSM on a labelled spectrum."""
    if num_matches > 0:
        return "partial_match"

    seq_norm = _strip_mods(seq_str)
    pred_norm = _strip_mods(pred_str)
    if seq_norm == pred_norm and _sequence_match_key(seq_str) != _sequence_match_key(
        pred_str
    ):
        return "ptm_candidate"

    edit_label = _discordance_from_edit_distance(seq_norm, pred_norm)
    if edit_label is not None:
        return edit_label

    ptm_label = _discordance_from_ptm_mass(seq_str, pred_str, residue_masses)
    if ptm_label is not None:
        return ptm_label

    return "fully_discordant"


def classify_discordance_vs_db(
    pred_str: str,
    db_keys: set[str],
    db_stripped_set: set[str],
    db_stripped_by_len: dict[int, list[str]],
) -> str:
    """Classify a discordant full-search PSM vs the database peptide reference set."""
    pred_key = _sequence_match_key(pred_str)
    if pred_key in db_keys:
        return "db_match"

    pred_stripped = _strip_mods(pred_str)
    if pred_stripped in db_stripped_set:
        return "ptm_candidate"

    ed = _min_edit_distance_to_db(pred_stripped, db_stripped_by_len)
    if ed == 1:
        return "single_aa_variant"
    if ed in (2, 3):
        return "near_miss_edit_dist"
    return "fully_discordant"


def _build_discordance_cache(
    predictions: pd.Series,
    db_keys: set[str],
    db_stripped_set: set[str],
    db_stripped_by_len: dict[int, list[str]],
) -> dict[str, str]:
    """Classify each unique discordant prediction string once."""
    cache: dict[str, str] = {}
    for pred in predictions.dropna().unique():
        key = str(pred)
        if _is_db_match(key, db_keys):
            cache[key] = "db_match"
        elif key not in cache:
            cache[key] = classify_discordance_vs_db(
                key, db_keys, db_stripped_set, db_stripped_by_len
            )
    return cache


def _classify_predictions_vs_db(
    predictions: pd.Series,
    cache: dict[str, str],
) -> pd.Series:
    return predictions.map(lambda p: cache.get(str(p), "fully_discordant"))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _preds_header(folder: Path) -> set[str] | None:
    preds_path = folder / "preds_and_fdr_metrics.csv"
    if not preds_path.is_file():
        return None
    return set(pd.read_csv(preds_path, nrows=0).columns.tolist())


def _load_from_folder(folder: Path) -> pd.DataFrame:
    """Load preds_and_fdr_metrics.csv merged with metadata.csv from a project folder."""
    preds_path = folder / "preds_and_fdr_metrics.csv"
    if not preds_path.is_file():
        raise FileNotFoundError(f"Missing predictions file: {preds_path}")

    preds_df = pd.read_csv(preds_path)
    meta_path = folder / "metadata.csv"
    if meta_path.is_file():
        meta_df = pd.read_csv(meta_path)
        overlap_cols = [
            c for c in meta_df.columns if c in preds_df.columns and c != "spectrum_id"
        ]
        if overlap_cols:
            meta_df = meta_df.drop(columns=overlap_cols)
        return preds_df.merge(meta_df, on="spectrum_id", how="left")
    return preds_df


def _discover_unlabelled_folders(root: Path) -> dict[str, Path]:
    """Map project key -> full-search folder under ``root``."""
    projects: dict[str, Path] = {}
    if not root.is_dir():
        return projects
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        header = _preds_header(child)
        if header is None:
            continue
        if "sequence" in header:
            continue
        key = _project_key_from_folder(child.name)
        if key in projects:
            logger.warning(
                "Duplicate unlabelled project key %r: %s and %s",
                key,
                projects[key],
                child,
            )
            continue
        projects[key] = child
    return projects


def _discover_labelled_folders(root: Path) -> dict[str, Path]:
    """Map project key -> database-reference folder under ``root``."""
    projects: dict[str, Path] = {}
    if not root.is_dir():
        return projects
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        header = _preds_header(child)
        if header is None or "sequence" not in header:
            continue
        key = _project_key_from_folder(child.name)
        if key in projects:
            logger.warning(
                "Duplicate labelled project key %r: %s and %s",
                key,
                projects[key],
                child,
            )
            continue
        projects[key] = child
    return projects


def discover_project_pairs(
    unlabelled_dir: Path,
    labelled_dir: Path,
    *,
    projects_filter: set[str] | None = None,
) -> list[tuple[str, Path, Path, str]]:
    """Return ``(project, unlabelled_folder, labelled_folder, search_space_tag)`` pairs."""
    unlabelled = _discover_unlabelled_folders(unlabelled_dir)
    labelled = _discover_labelled_folders(labelled_dir)

    keys = sorted(unlabelled.keys() & labelled.keys())
    if projects_filter is not None:
        keys = [k for k in keys if k in projects_filter]

    pairs: list[tuple[str, Path, Path, str]] = []
    for key in keys:
        pairs.append(
            (
                key,
                unlabelled[key],
                labelled[key],
                _search_space_tag_from_folder(unlabelled[key].name),
            )
        )

    for key in sorted(unlabelled.keys() - labelled.keys()):
        if projects_filter is None or key in projects_filter:
            logger.warning("No labelled folder for unlabelled project %r", key)
    for key in sorted(labelled.keys() - unlabelled.keys()):
        if projects_filter is None or key in projects_filter:
            logger.warning("No unlabelled folder for labelled project %r", key)

    return pairs


def _add_q_values(
    df: pd.DataFrame, conf_col: str = "calibrated_confidence"
) -> pd.DataFrame:
    if "psm_q_value" in df.columns:
        return df
    fdr = NonParametricFDRControl()
    fdr.fit(dataset=df[conf_col])
    return fdr.add_psm_q_value(df, confidence_col=conf_col)


def _empty_overlap_row(
    project: str,
    eval_type: str,
    fdr_t: float,
    n_db_peptides: int,
    labelled_subset: bool,
) -> dict:
    row: dict = {
        "project": project,
        "eval_type": eval_type,
        "fdr_threshold": fdr_t,
        "n_psms_retained": 0,
        "n_unique_peptides_retained": 0,
        "n_db_search_peptides": n_db_peptides,
        "n_matching": 0,
        "pct_matching": 0.0,
        "n_discordant": 0,
        "pct_discordant": 0.0,
    }
    for col in _DISCORDANCE_COUNT_COLS:
        if col == "n_partial_match" and not labelled_subset:
            continue
        row[col] = 0
    return row


def _discordance_cache_for_fdr_retained(
    df: pd.DataFrame,
    db_keys: set[str],
    db_stripped_set: set[str],
    db_stripped_by_len: dict[int, list[str]],
    *,
    labelled_subset: bool,
    residue_masses: dict[str, float] | None,
) -> LabelledDiscordanceCache | DbDiscordanceCache:
    """Build discordance lookup for all predictions retained at any FDR threshold."""
    df = _add_q_values(df)
    retained = df[df["psm_q_value"] <= max(FDR_THRESHOLDS)]
    if labelled_subset and residue_masses is not None:
        cache: LabelledDiscordanceCache = {}
        disc = retained[
            retained["prediction"].map(_sequence_match_key)
            != retained["sequence"].map(_sequence_match_key)
        ]
        for _, row in disc.drop_duplicates(
            subset=["sequence", "prediction"]
        ).iterrows():
            trip = (
                str(row.get("sequence", "")),
                str(row.get("prediction", "")),
                int(row.get("num_matches", 0)),
            )
            if trip not in cache:
                cache[trip] = classify_discordance_labelled(
                    trip[0], trip[1], trip[2], residue_masses
                )
        return cache

    return _build_discordance_cache(
        retained["prediction"], db_keys, db_stripped_set, db_stripped_by_len
    )


def compute_overlap_table(
    df: pd.DataFrame,
    project: str,
    eval_type: str,
    db_keys: set[str],
    discordance_cache: LabelledDiscordanceCache | DbDiscordanceCache,
    *,
    labelled_subset: bool = False,
) -> pd.DataFrame:
    """Overlap summary at each FDR threshold."""
    df = _add_q_values(df.copy())
    n_db_peptides = len(db_keys)

    rows: list[dict] = []
    for fdr_t in FDR_THRESHOLDS:
        retained = df[df["psm_q_value"] <= fdr_t].copy()
        n_retained = len(retained)
        if n_retained == 0:
            rows.append(
                _empty_overlap_row(
                    project, eval_type, fdr_t, n_db_peptides, labelled_subset
                )
            )
            continue

        if labelled_subset:
            retained["pred_key"] = retained["prediction"].map(_sequence_match_key)
            retained["seq_key"] = retained["sequence"].map(_sequence_match_key)
            is_match = retained["pred_key"] == retained["seq_key"]
        else:
            is_match = retained["prediction"].map(
                lambda p: _is_db_match(str(p), db_keys)
            )

        n_matching = int(is_match.sum())
        n_discordant = n_retained - n_matching
        n_unique_peptides = int(
            retained["prediction"].map(_sequence_match_key).replace("", pd.NA).nunique()
        )

        disc = retained[~is_match]
        cat_counts: dict[str, int] = {}
        if len(disc) > 0:
            if labelled_subset:
                labelled_cache = cast(LabelledDiscordanceCache, discordance_cache)
                cats = disc.apply(
                    _lookup_labelled_discordance, axis=1, cache=labelled_cache
                )
            else:
                cats = _classify_predictions_vs_db(
                    disc["prediction"],
                    cast(DbDiscordanceCache, discordance_cache),
                )
            cat_counts = cats.value_counts().to_dict()

        row: dict = {
            "project": project,
            "eval_type": eval_type,
            "fdr_threshold": fdr_t,
            "n_psms_retained": n_retained,
            "n_unique_peptides_retained": n_unique_peptides,
            "n_db_search_peptides": n_db_peptides,
            "n_matching": n_matching,
            "pct_matching": round(n_matching / n_retained * 100, 2),
            "n_discordant": n_discordant,
            "pct_discordant": round(n_discordant / n_retained * 100, 2),
            "n_ptm_candidate": cat_counts.get("ptm_candidate", 0),
            "n_single_aa_variant": cat_counts.get("single_aa_variant", 0),
            "n_near_miss_edit_dist": cat_counts.get("near_miss_edit_dist", 0),
            "n_fully_discordant": cat_counts.get("fully_discordant", 0),
        }
        if labelled_subset:
            row["n_partial_match"] = cat_counts.get("partial_match", 0)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plot_venn_panels(
    db_peptides: set[str],
    df: pd.DataFrame,
    project: str,
    output_path: Path,
    *,
    winnow_label: str,
    title_suffix: str,
) -> None:
    display = _display_name(project)
    df = _add_q_values(df)

    n_thresholds = len(FDR_THRESHOLDS)
    fig, axes = plt.subplots(1, n_thresholds, figsize=(5 * n_thresholds, 5))
    if n_thresholds == 1:
        axes = [axes]

    for ax, fdr_t in zip(axes, FDR_THRESHOLDS):
        retained = df[df["psm_q_value"] <= fdr_t]
        winnow_peptides = set(
            retained["prediction"].dropna().map(_sequence_match_key)
        ) - {""}

        if not winnow_peptides and not db_peptides:
            ax.set_title(f"No peptides retained at {int(fdr_t * 100)}% FDR")
            ax.axis("off")
            continue

        if not winnow_peptides:
            ax.text(
                0.5,
                0.5,
                f"No PSMs retained at {int(fdr_t * 100)}% FDR",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{int(fdr_t * 100)}% FDR")
            ax.axis("off")
            continue

        venn2(
            [db_peptides, winnow_peptides],
            set_labels=("Database search", winnow_label),
            set_colors=(_CORRECT_COLOUR, _INCORRECT_COLOUR),
            alpha=0.6,
            ax=ax,
        )
        ax.set_title(f"Unique peptides at {int(fdr_t * 100)}% FDR")
        _style_ax(ax)

    fig.suptitle(
        f"Peptide overlap between full search space and labelled subset for {display}",
        fontsize=12,
    )
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_full_search_venn(
    df: pd.DataFrame,
    db_df: pd.DataFrame,
    project: str,
    plots_dir: Path,
) -> None:
    """Venn diagrams of database peptides vs Winnow full-search peptides per FDR."""
    db_peptides = {
        _sequence_match_key(s)
        for s in db_df["sequence"].dropna()
        if _sequence_match_key(s)
    }
    _plot_venn_panels(
        db_peptides,
        df,
        project,
        plots_dir / f"venn_{project}_full_search",
        winnow_label="Winnow",
        title_suffix="full search space",
    )


def plot_labelled_subset_venn(
    df: pd.DataFrame,
    project: str,
    plots_dir: Path,
) -> None:
    """Venn diagrams of labelled reference peptides vs Winnow predictions per FDR."""
    db_peptides = {
        _sequence_match_key(s)
        for s in df["sequence"].dropna()
        if _sequence_match_key(s)
    }
    _plot_venn_panels(
        db_peptides,
        df,
        project,
        plots_dir / f"venn_{project}_labelled_subset",
        winnow_label="Winnow",
        title_suffix="labelled subset",
    )


def _assign_retained_groups(
    retained: pd.DataFrame,
    db_keys: set[str],
    discordance_cache: LabelledDiscordanceCache | DbDiscordanceCache,
    *,
    labelled_subset: bool,
) -> pd.Series:
    if labelled_subset and "sequence" in retained.columns:
        labelled_cache = cast(LabelledDiscordanceCache, discordance_cache)
        is_match = retained["prediction"].map(_sequence_match_key) == retained[
            "sequence"
        ].map(_sequence_match_key)
        groups = pd.Series("Database match", index=retained.index)
        disc_mask = ~is_match
        if disc_mask.any():
            disc = retained.loc[disc_mask]
            groups.loc[disc_mask] = disc.apply(
                _lookup_labelled_discordance, axis=1, cache=labelled_cache
            ).values
        return groups

    db_cache = cast(DbDiscordanceCache, discordance_cache)
    is_match = retained["prediction"].map(lambda p: _is_db_match(str(p), db_keys))
    groups = pd.Series("Database match", index=retained.index)
    disc_mask = ~is_match
    if disc_mask.any():
        groups.loc[disc_mask] = _classify_predictions_vs_db(
            retained.loc[disc_mask, "prediction"],
            db_cache,
        ).values
    return groups


def _subsample_violin_groups(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    """Limit points per category so violin plots stay responsive."""
    parts: list[pd.DataFrame] = []
    for _cat, group in df.groupby(category_col, observed=True):
        if len(group) > _MAX_VIOLIN_PSMs_PER_GROUP:
            group = group.sample(
                n=_MAX_VIOLIN_PSMs_PER_GROUP,
                random_state=_VIOLIN_SUBSAMPLE_SEED,
            )
        parts.append(group)
    return pd.concat(parts, ignore_index=True) if parts else df


def plot_novel_feature_violins(
    df: pd.DataFrame,
    db_keys: set[str],
    discordance_cache: LabelledDiscordanceCache | DbDiscordanceCache,
    project: str,
    eval_type: str,
    plots_dir: Path,
    *,
    labelled_subset: bool = False,
) -> None:
    """Violin plots: database-matched vs fully discordant retained PSMs."""
    available = [
        (col, label) for col, label in NOVEL_FEATURE_COLUMNS if col in df.columns
    ]
    if not available:
        logger.warning(
            "%s: no feature columns for violin plots (metadata merge missing?)",
            project,
        )
        return

    df = _add_q_values(df.copy())
    display = _display_name(project)

    for fdr_t in FDR_THRESHOLDS:
        retained = df[df["psm_q_value"] <= fdr_t].copy()
        if len(retained) < _MIN_VIOLIN_GROUP_SIZE:
            logger.info(
                "%s: skip violins at %d%% FDR (n=%d retained)",
                project,
                int(fdr_t * 100),
                len(retained),
            )
            continue

        groups = _assign_retained_groups(
            retained,
            db_keys,
            discordance_cache,
            labelled_subset=labelled_subset,
        )
        retained = retained.assign(_overlap_group=groups)
        plot_df = retained[
            retained["_overlap_group"].isin(["Database match", "fully_discordant"])
        ].copy()
        plot_df["Category"] = plot_df["_overlap_group"].map(
            {
                "Database match": "Database match",
                "fully_discordant": "Novel",
            }
        )
        plot_df = _subsample_violin_groups(plot_df, "Category")

        n_match = (plot_df["Category"] == "Database match").sum()
        n_novel = (plot_df["Category"] == "Novel").sum()
        if n_match < _MIN_VIOLIN_GROUP_SIZE or n_novel < _MIN_VIOLIN_GROUP_SIZE:
            logger.info(
                "%s: skip violins at %d%% FDR (match=%d, novel=%d)",
                project,
                int(fdr_t * 100),
                n_match,
                n_novel,
            )
            continue

        n_feats = len(available)
        n_cols = 4
        n_rows = int(np.ceil(n_feats / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes_flat = np.atleast_1d(axes).flatten()

        palette = {"Database match": _CORRECT_COLOUR, "Novel": _NOVEL_COLOUR}
        cat_order = ["Database match", "Novel"]

        for ax, (col, label) in zip(axes_flat, available):
            sub = plot_df[[col, "Category"]].dropna()
            if sub["Category"].nunique() < 2:
                ax.set_visible(False)
                continue
            sns.violinplot(
                data=sub,
                x="Category",
                y=col,
                order=cat_order,
                palette=palette,
                ax=ax,
                inner="quartile",
                cut=0,
                linewidth=0.8,
            )
            ax.set_xlabel("")
            ax.set_ylabel(label)
            ax.tick_params(axis="x", rotation=15)
            _style_ax(ax)

        for ax in axes_flat[len(available) :]:
            ax.set_visible(False)

        pct = int(fdr_t * 100)
        fig.suptitle(
            f"{display} ({eval_type}): database-matched vs novel features at {pct}% FDR",
            fontsize=12,
        )
        fig.tight_layout()
        _save_fig(fig, plots_dir / f"novel_feature_violins_{project}_fdr{pct}")


# ---------------------------------------------------------------------------
# Per-project orchestration
# ---------------------------------------------------------------------------
def generate_all_analyses(
    df: pd.DataFrame,
    project: str,
    eval_type: str,
    output_dir: Path,
    db_df: pd.DataFrame | None,
    residue_masses: dict[str, float],
) -> pd.DataFrame:
    """Tables and plots for one project."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if db_df is None:
        raise ValueError(f"Full-search analysis requires DB reference for {project}")
    db_keys, db_stripped_set, _, db_by_len = _build_db_reference_sets(db_df)
    disc_cache = _discordance_cache_for_fdr_retained(
        df,
        db_keys,
        db_stripped_set,
        db_by_len,
        labelled_subset=False,
        residue_masses=None,
    )
    overlap = compute_overlap_table(
        df,
        project,
        eval_type,
        db_keys,
        disc_cache,
        labelled_subset=False,
    )
    plot_full_search_venn(df, db_df, project, plots_dir)
    plot_novel_feature_violins(
        df,
        db_keys,
        disc_cache,
        project,
        eval_type,
        plots_dir,
        labelled_subset=False,
    )

    overlap.to_csv(output_dir / f"{project}_overlap_summary.csv", index=False)
    with open(output_dir / f"{project}_overlap_summary.json", "w") as f:
        json.dump(overlap.to_dict(orient="records"), f, indent=2)

    logger.info("\n%s", overlap.to_string(index=False))
    return overlap


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@app.command()
def main(
    unlabelled_dir: Annotated[
        Path,
        typer.Option(
            "--unlabelled-dir",
            help="Root with per-project full-search folders (e.g. gluc_raw/, PXD014877_unlabelled/).",
        ),
    ],
    labelled_dir: Annotated[
        Path,
        typer.Option(
            "--labelled-dir",
            help="Root with per-project database-search folders (e.g. gluc_annotated/, PXD014877_labelled/).",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for tables and plots."),
    ],
    projects: Annotated[
        str | None,
        typer.Option(
            "--projects",
            help="Optional space- or comma-separated project keys to restrict analysis.",
        ),
    ] = None,
) -> None:
    """Post-FDR overlap: full-search Winnow identifications vs database search."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="%H:%M:%S")

    projects_filter: set[str] | None = None
    if projects is not None:
        project_list = [
            p.strip() for p in projects.replace(",", " ").split() if p.strip()
        ]
        if not project_list:
            raise typer.BadParameter("No projects specified in --projects.")
        projects_filter = set(project_list)

    pairs = discover_project_pairs(
        unlabelled_dir, labelled_dir, projects_filter=projects_filter
    )
    if not pairs:
        logger.error(
            "No paired projects found under unlabelled-dir=%s and labelled-dir=%s",
            unlabelled_dir,
            labelled_dir,
        )
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    residue_masses = _get_residue_masses()

    all_tables: list[pd.DataFrame] = []
    for project, unlabelled_folder, labelled_folder, search_tag in pairs:
        display = _display_name(project)
        logger.info(
            "Processing %s (%s): %s vs %s",
            project,
            display,
            unlabelled_folder.name,
            labelled_folder.name,
        )

        try:
            df = _load_from_folder(unlabelled_folder)
            db_df = _load_from_folder(labelled_folder)
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", project, exc)
            continue

        logger.info(
            "  Full search: %d rows; DB reference: %d rows", len(df), len(db_df)
        )

        try:
            table = generate_all_analyses(
                df,
                project,
                search_tag,
                output_dir,
                db_df,
                residue_masses,
            )
            all_tables.append(table)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", project, exc)

    if not all_tables:
        logger.error("No projects produced overlap output under %s", output_dir)
        raise typer.Exit(code=1)

    combined = pd.concat(all_tables, ignore_index=True)
    combined.to_csv(output_dir / "all_projects_overlap_summary.csv", index=False)
    with open(output_dir / "all_projects_overlap_summary.json", "w") as f:
        json.dump(combined.to_dict(orient="records"), f, indent=2)

    logger.info("FDR overlap analysis complete. Output in %s", output_dir)


if __name__ == "__main__":
    app()
