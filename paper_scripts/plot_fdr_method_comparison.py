#!/usr/bin/env python3
"""Compare PSM-level FDR estimates from Winnow and NovoBoard.

Plots and summaries cover labelled-test Novor correctness and unlabelled
reference-proteome membership at 1 %, 5 %, and 10 % FDR on a shared filtered
spectrum pool (NovoBoard mass-deltas converted to ProForma; unsupported
modifications dropped; NovoBoard target-decoy pairs gated). Unlabelled panels
also drop normalised peptides shorter than 8 residues (proteome-substring
proxy); labelled panels keep short peptides because correctness is Novor
agreement. The shared pool is the twin-valid NovoBoard set; Winnow is trimmed
to match under the invariant that NovoBoard ⊆ Winnow after identical InstaNovo
filters.

A long-form ``fdr_method_comparison_curves.csv`` (per spectrum x method) is
written so plots and summary tables can be regenerated with ``--summarise-only``.

External NovoBoard inputs (``--novoboard-root``) must follow
``{root}/{dataset}/novoboard/`` with target/decoy CSVs such as
``annotated_test.csv``, ``annotated_test_decoy_{rate}.csv``,
``raw_unlabelled.csv`` and ``raw_unlabelled_decoy_{rate}.csv``. Point
``--novoboard-root`` at the ``datasets`` directory of a NovoBoard checkout.
Local results were produced from the fork
``git@github.com:JemmaLDaniel/NovoBoard.git``, branch
``feat/adapt-to-instanovo`` at commit
``a9faab3ef1af06987599c2f01e6ba96072c80172``.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import typer

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
_PAPER_SCRIPTS = Path(__file__).resolve().parent
if str(_PAPER_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_PAPER_SCRIPTS))


from winnow.utils.proteome import load_proteome_haystack  # noqa: E402
from fdr_tool_comparison_preprocess import (  # noqa: E402
    LABELLED_MIN_PEPTIDE_LENGTH,
    MIN_PEPTIDE_LENGTH,
    assert_shared_prediction_keys,
    attach_labels_by_spectrum_id,
    attach_novoboard_pair_keys,
    compute_q_values,
    filter_novoboard_target_decoy_pairs,
    filter_prediction_table,
    label_series_by_spectrum_id,
    load_residue_masses,
    novoboard_psm_tdc,
    novor_correctness_mask,
    proteome_hit_mask,
    restrict_winnow_to_novoboard_spectra,
)
from fdr_tool_comparison_summaries import (  # noqa: E402
    SUMMARY_THRESHOLDS,
    acceptance_rows_from_q,
    error_rows_from_q,
    finalise_error_gain_table,
    write_summary_tables,
)
from plot_eval_results import (  # noqa: E402
    _MAIN_LINE_COLOUR,
    _PALETTE,
    _RAW_LINE_COLOUR,
    _display_name,
    _ground_truth_qualifier,
    _save_fig,
    _style_ax,
)
from winnow.fdr.database_grounded import DatabaseGroundedFDRControl  # noqa: E402
from winnow.fdr.nonparametric import NonParametricFDRControl  # noqa: E402

PRIMARY_METHOD = "Winnow (non-parametric)"
DB_CAL_METHOD = "Database-grounded (calibrated confidence)"
DB_RAW_METHOD = "Database-grounded (raw confidence)"
NOVOBOARD_METHOD = "NovoBoard"
CURVES_CSV_NAME = "fdr_method_comparison_curves.csv"
_WINNOW_METHODS = (PRIMARY_METHOD, DB_CAL_METHOD, DB_RAW_METHOD)
_METHOD_ORDER = (*_WINNOW_METHODS, NOVOBOARD_METHOD)

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

sns.set_theme(style="white", palette=_PALETTE, context="paper", font_scale=1.5)

FDR_THRESHOLDS = [0.01, 0.05, 0.10]
_DB_GROUNDED_DROP = 10

DEFAULT_WINNOW_RESULTS = _REPO_ROOT / "results"
DEFAULT_MODEL_ROOT = _REPO_ROOT / "models"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "results/fdr_method_comparison_psm"
DEFAULT_DATASETS = ["helaqc", "celegans"]
_METHOD_COLOURS = {
    PRIMARY_METHOD: _MAIN_LINE_COLOUR,
    DB_CAL_METHOD: _RAW_LINE_COLOUR,
    DB_RAW_METHOD: _PALETTE[3],
    NOVOBOARD_METHOD: _PALETTE[2],
}

EvalType = Literal["labelled", "unlabelled"]

_DATASET_META = {
    "helaqc": {
        "fasta": "fasta/human.fasta",
        "novoboard_decoy": "0.50",
        "winnow_suffix": "helaqc",
    },
    "celegans": {
        "fasta": "fasta/celegans.fasta",
        "novoboard_decoy": "0.70",
        "winnow_suffix": "celegans",
    },
    "sbrodae": {
        "fasta": "fasta/Sb_proteome.fasta",
        "novoboard_decoy": "0.50",
        "winnow_suffix": "sbrodae",
    },
    "PXD019483": {
        "fasta": "fasta/human.fasta",
        "novoboard_decoy": "0.70",
        "winnow_suffix": "pxd019483",
    },
}


@dataclass(frozen=True)
class DatasetConfig:
    """Paths and metadata for one evaluation dataset."""

    key: str
    fasta: Path
    winnow_unlabelled: Path
    winnow_test: Path
    novoboard_dir: Path
    novoboard_decoy_rate: str
    calibrator_train_metadata: Path


def build_dataset_configs(
    winnow_results: Path = DEFAULT_WINNOW_RESULTS,
    *,
    novoboard_root: Path,
    model_root: Path = DEFAULT_MODEL_ROOT,
) -> dict[str, DatasetConfig]:
    """Build per-dataset path bundles from repo roots."""
    configs: dict[str, DatasetConfig] = {}
    for key, meta in _DATASET_META.items():
        suffix = meta["winnow_suffix"]
        configs[key] = DatasetConfig(
            key=key,
            fasta=_REPO_ROOT / meta["fasta"],
            winnow_unlabelled=winnow_results
            / f"instanovo_{suffix}_predictions_unlabelled",
            winnow_test=winnow_results / f"instanovo_{suffix}_predictions_test",
            novoboard_dir=novoboard_root / f"{key}/novoboard",
            novoboard_decoy_rate=meta["novoboard_decoy"],
            calibrator_train_metadata=model_root
            / f"instanovo_{suffix}/metadata_train.parquet",
        )
    return configs


@dataclass
class MethodCurve:
    """One method's confidence and q-value arrays for curve plotting."""

    label: str
    color: str
    confidence: np.ndarray
    q_value: np.ndarray


@dataclass
class MethodCounts:
    """PSM counts per q-value threshold for one method."""

    label: str
    color: str
    counts: list[int]


@dataclass
class MethodRecovery:
    """Correct labelled identifications recovered at q-value thresholds."""

    label: str
    color: str
    q_value: np.ndarray
    correct: np.ndarray


def _load_residue_masses() -> dict[str, float]:
    return load_residue_masses()


def _fit_database_grounded_fdr(
    df: pd.DataFrame,
    correct_col: str,
    confidence_col: str,
    *,
    drop: int = _DB_GROUNDED_DROP,
) -> DatabaseGroundedFDRControl:
    """Fit ``DatabaseGroundedFDRControl`` from per-row correctness labels."""
    ctrl = DatabaseGroundedFDRControl(
        confidence_feature=confidence_col,
        drop=drop,
    )
    sorted_df = df.sort_values(confidence_col, ascending=False)
    labels = sorted_df[correct_col].astype(float).to_numpy()
    conf = sorted_df[confidence_col].to_numpy()
    precision = np.cumsum(labels) / np.arange(1, len(labels) + 1)
    ctrl._fdr_values = np.array(1.0 - precision)[drop:]
    ctrl._confidence_scores = conf[drop:]
    return ctrl


def load_winnow(
    predictions_dir: Path, fasta: Path, eval_type: EvalType
) -> pd.DataFrame:
    """Load Winnow preds + metadata; annotate proteome hits or use labelled ``correct``.

    Always drops unsupported ``[UNIMOD:n]`` tokens. Unlabelled panels also require
    normalised peptide length ≥ :data:`MIN_PEPTIDE_LENGTH` (proteome-substring
    proxy). Labelled panels only require a non-empty normalised key
    (:data:`LABELLED_MIN_PEPTIDE_LENGTH`), because correctness is Novor agreement.

    Labelled ``correct`` keeps Winnow predict-time Novor labels when present;
    otherwise recomputes Novor from ``sequence`` / ``prediction``.
    """
    preds = pl.read_csv(predictions_dir / "preds_and_fdr_metrics.csv")
    meta_path = predictions_dir / "metadata.csv"
    if meta_path.exists():
        meta = pl.read_csv(meta_path, columns=["spectrum_id", "confidence"])
        preds = preds.join(meta, on="spectrum_id", how="inner")

    df = preds.to_pandas()
    min_length = (
        LABELLED_MIN_PEPTIDE_LENGTH if eval_type == "labelled" else MIN_PEPTIDE_LENGTH
    )
    df = filter_prediction_table(
        df, "prediction", min_length=min_length, key_col="peptide_key"
    )

    if eval_type == "labelled":
        if "correct" in df.columns:
            pass
        elif {"sequence", "prediction"}.issubset(df.columns):
            df["correct"] = novor_correctness_mask(df["sequence"], df["prediction"])
        else:
            raise ValueError(
                f"Missing 'correct' (and sequence/prediction) in "
                f"{predictions_dir}/preds_and_fdr_metrics.csv"
            )
        return df

    haystack = load_proteome_haystack(fasta)
    df["proteome_hit"] = proteome_hit_mask(
        df["prediction"], haystack, min_length=MIN_PEPTIDE_LENGTH
    )
    return df


def _effective_db_grounded_drop(n_rows: int, drop: int = _DB_GROUNDED_DROP) -> int:
    """Cap drop so FDR fit retains at least one score when *n_rows* is small."""
    return min(drop, max(0, n_rows - 1))


def _vectorized_fdr_from_control(
    confidence: np.ndarray, ctrl: DatabaseGroundedFDRControl | NonParametricFDRControl
) -> np.ndarray:
    """Vectorized equivalent of ``FDRControl.compute_fdr`` for an array of scores."""
    if ctrl._confidence_scores is None or ctrl._fdr_values is None:
        raise AttributeError("FDR method not fitted, please call `fit()` first")
    conf = np.asarray(confidence, dtype=float)
    scores = np.asarray(ctrl._confidence_scores, dtype=float)
    fdr_values = np.asarray(ctrl._fdr_values, dtype=float)
    n = len(scores)
    idx = np.searchsorted(-scores, -conf, side="left")
    fdr = np.empty(len(conf), dtype=float)
    below = (idx == n) & (conf < scores[-1])
    above = (idx == 0) & (conf > scores[0])
    normal = ~(below | above)
    fdr[below] = 1.0
    fdr[above] = float(fdr_values[0])
    clipped = np.clip(idx[normal], 0, n - 1)
    fdr[normal] = fdr_values[clipped]
    return fdr


def _assign_q_values_fast(
    df: pd.DataFrame,
    confidence_col: str,
    ctrl: DatabaseGroundedFDRControl | NonParametricFDRControl,
    out_col: str,
) -> pd.DataFrame:
    """Assign q-values without per-row ``compute_fdr`` applies (needed for large tables)."""
    work = df.copy()
    conf = work[confidence_col].to_numpy(dtype=float)
    fdr = _vectorized_fdr_from_control(conf, ctrl)
    order = np.argsort(-conf, kind="mergesort")
    q_sorted = compute_q_values(fdr[order])
    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    work[out_col] = q
    return work


def _add_database_grounded_qvalues(
    df: pd.DataFrame,
    correct_col: str,
    confidence_col: str,
    out_col: str,
    residue_masses: dict[str, float],
    *,
    fit_df: pd.DataFrame | None = None,
    drop: int = _DB_GROUNDED_DROP,
) -> pd.DataFrame:
    """Append database-grounded PSM q-values; fit on *fit_df* (defaults to *df*)."""
    reference = fit_df if fit_df is not None else df
    work = df.drop(columns=[out_col], errors="ignore").copy()
    ctrl = _fit_database_grounded_fdr(
        reference,
        correct_col,
        confidence_col,
        drop=_effective_db_grounded_drop(len(reference), drop),
    )
    return _assign_q_values_fast(work, confidence_col, ctrl, out_col)


def _prepare_winnow_psm_table(
    df: pd.DataFrame,
    correct_col: str,
    residue_masses: dict[str, float],
    *,
    fit_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Append Winnow PSM-level q-value columns while retaining labels."""
    reference = fit_df if fit_df is not None else df
    db_cal = _add_database_grounded_qvalues(
        df,
        correct_col,
        "calibrated_confidence",
        "psm_q_value_db_cal",
        residue_masses,
        fit_df=reference,
    )
    db_raw = _add_database_grounded_qvalues(
        db_cal,
        correct_col,
        "confidence",
        "psm_q_value_db_raw",
        residue_masses,
        fit_df=reference,
    )
    return db_raw


def _curves_df_from_winnow_table(
    table: pd.DataFrame,
    *,
    dataset: str,
    panel: str,
    label_col: str,
) -> pd.DataFrame:
    """Long-form curve rows for the three Winnow PSM q-value methods."""
    if "spectrum_id" not in table.columns:
        raise KeyError("Winnow curve export requires spectrum_id")
    if label_col not in table.columns:
        raise KeyError(f"Missing label column {label_col!r}")
    label = table[label_col].astype(bool).to_numpy()
    spectrum_id = table["spectrum_id"].astype(str)
    specs = (
        (PRIMARY_METHOD, "calibrated_confidence", "psm_q_value"),
        (DB_CAL_METHOD, "calibrated_confidence", "psm_q_value_db_cal"),
        (DB_RAW_METHOD, "confidence", "psm_q_value_db_raw"),
    )
    parts: list[pd.DataFrame] = []
    for method, score_col, q_col in specs:
        if score_col not in table.columns or q_col not in table.columns:
            raise KeyError(f"Missing {score_col!r} / {q_col!r} for {method}")
        parts.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "panel": panel,
                    "method": method,
                    "spectrum_id": spectrum_id,
                    "score": table[score_col].to_numpy(dtype=float),
                    "q_value": table[q_col].to_numpy(dtype=float),
                    "label": label,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _curves_df_from_novoboard(
    df: pd.DataFrame,
    *,
    dataset: str,
    panel: str,
    label_col: str,
) -> pd.DataFrame:
    """Long-form curve rows for NovoBoard PSM TDC targets."""
    if "spectrum_id" not in df.columns:
        raise KeyError("NovoBoard curve export requires spectrum_id")
    if label_col not in df.columns:
        raise KeyError(f"Missing label column {label_col!r}")
    return pd.DataFrame(
        {
            "dataset": dataset,
            "panel": panel,
            "method": NOVOBOARD_METHOD,
            "spectrum_id": df["spectrum_id"].astype(str),
            "score": df["ALC (%)"].to_numpy(dtype=float),
            "q_value": df["estimated_q_value"].to_numpy(dtype=float),
            "label": df[label_col].astype(bool).to_numpy(),
        }
    )


def load_novoboard_target_decoy(
    novoboard_dir: Path, split: Literal["unlabelled", "test"], decoy_rate: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load NovoBoard target/decoy tables and attach twin ``_pair_key`` values."""
    prefix = "raw_unlabelled" if split == "unlabelled" else "annotated_test"
    target_path = novoboard_dir / f"{prefix}.csv"
    decoy_path = novoboard_dir / f"{prefix}_decoy_{decoy_rate}.csv"
    if not target_path.is_file():
        raise FileNotFoundError(target_path)
    if not decoy_path.is_file():
        raise FileNotFoundError(decoy_path)
    target = pd.read_csv(target_path)
    decoy = pd.read_csv(decoy_path)
    return attach_novoboard_pair_keys(
        target, decoy, novoboard_dir=novoboard_dir, split_prefix=prefix
    )


def _restrict_winnow_to_novoboard_spectra(
    winnow: pd.DataFrame, novoboard: pd.DataFrame
) -> pd.DataFrame:
    """Trim Winnow to NovoBoard twin-valid spectra under the subset invariant."""
    return restrict_winnow_to_novoboard_spectra(winnow, novoboard)


def _assert_shared_prediction_keys(
    winnow: pd.DataFrame,
    novoboard: pd.DataFrame,
    *,
    winnow_peptide_col: str = "prediction",
    novoboard_peptide_col: str = "Peptide",
) -> None:
    """Require I/L-normalised prediction identity on the shared spectrum pool."""
    assert_shared_prediction_keys(
        winnow,
        novoboard,
        winnow_peptide_col=winnow_peptide_col,
        novoboard_peptide_col=novoboard_peptide_col,
    )


def _label_series_by_spectrum_id(winnow: pd.DataFrame, label_col: str) -> pd.Series:
    """Map ``spectrum_id`` → boolean label from a Winnow table."""
    return label_series_by_spectrum_id(winnow, label_col)


def _attach_labels_by_spectrum_id(
    novoboard: pd.DataFrame,
    label_by_id: pd.Series,
    *,
    label_col: str,
) -> pd.DataFrame:
    """Attach a shared label column to NovoBoard rows by ``spectrum_id``."""
    return attach_labels_by_spectrum_id(novoboard, label_by_id, label_col=label_col)


def _method_curves_from_panel(panel_df: pd.DataFrame) -> list[MethodCurve]:
    """Rebuild plot curves from long-form curve rows for one panel."""
    curves: list[MethodCurve] = []
    for method in _METHOD_ORDER:
        sub = panel_df.loc[panel_df["method"] == method]
        if sub.empty:
            continue
        colour = _METHOD_COLOURS.get(str(method), _PALETTE[0])
        curves.append(
            MethodCurve(
                str(method),
                colour,
                sub["score"].to_numpy(dtype=float),
                sub["q_value"].to_numpy(dtype=float),
            )
        )
    return curves


def _recovery_series_from_panel(panel_df: pd.DataFrame) -> list[MethodRecovery]:
    """Rebuild labelled recovery series from long-form curve rows."""
    series: list[MethodRecovery] = []
    for method in _METHOD_ORDER:
        sub = panel_df.loc[panel_df["method"] == method]
        if sub.empty:
            continue
        colour = _METHOD_COLOURS.get(str(method), _PALETTE[0])
        series.append(
            MethodRecovery(
                str(method),
                colour,
                sub["q_value"].to_numpy(dtype=float),
                sub["label"].astype(bool).to_numpy(),
            )
        )
    return series


def plot_dataset_from_curves(
    curves: pd.DataFrame, dataset_key: str, output_dir: Path
) -> None:
    """Write PSM comparison plots for one dataset from a curves table."""
    out = output_dir / dataset_key
    out.mkdir(parents=True, exist_ok=True)
    ds = curves.loc[curves["dataset"] == dataset_key]
    if ds.empty:
        raise ValueError(f"No curve rows for dataset {dataset_key!r}")

    panel_specs: tuple[tuple[str, EvalType, str], ...] = (
        ("unlabelled", "unlabelled", "unlabelled"),
        ("labelled_test", "labelled", "test"),
    )
    for panel, eval_type, stem in panel_specs:
        panel_df = ds.loc[ds["panel"] == panel]
        method_curves = _method_curves_from_panel(panel_df)
        if not method_curves:
            continue
        plot_qvalue_by_rank(
            method_curves,
            dataset_key,
            eval_type,
            out / f"psm_qvalue_by_rank_{stem}_{dataset_key}",
        )
        plot_threshold_barplot(
            _bar_series_from_curves(method_curves),
            dataset_key,
            eval_type,
            out / f"psm_counts_{stem}_{dataset_key}",
        )

    labelled = ds.loc[ds["panel"] == "labelled_test"]
    recovery = _recovery_series_from_panel(labelled)
    if recovery:
        plot_recovery_curves(
            recovery,
            dataset_key,
            out / f"psm_recovery_test_{dataset_key}",
        )


def write_curves_csv(curves: pd.DataFrame, output_dir: Path) -> Path:
    """Write the long-form replot curves table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / CURVES_CSV_NAME
    # Preserve float64 q/score values so threshold edge cases survive round-trip.
    curves.to_csv(path, index=False, float_format="%.17g")
    logger.info("Wrote %s (%d rows)", path, len(curves))
    return path


def plot_qvalue_by_rank(
    curves: list[MethodCurve],
    dataset_key: str,
    eval_type: EvalType,
    output_path: Path,
    *,
    title_suffix: str = "",
) -> None:
    """Plot q-value against native-score rank/accepted count."""
    display = _display_name(dataset_key)
    qualifier = _ground_truth_qualifier(
        "labelled" if eval_type == "labelled" else "unlabelled"
    )
    title = f"{display} PSM q-value by rank {qualifier}{title_suffix}"

    fig, ax = plt.subplots(figsize=(8, 6))
    q_max = 0.0
    for curve in curves:
        order = np.argsort(-np.asarray(curve.confidence, dtype=float))
        y = np.asarray(curve.q_value, dtype=float)[order]
        rank = np.arange(1, len(y) + 1)
        valid = ~np.isnan(y)
        if not np.any(valid):
            continue
        q_max = max(q_max, float(np.nanmax(y[valid])))
        ax.plot(
            rank[valid],
            y[valid],
            color=curve.color,
            lw=1.5,
            label=curve.label,
        )

    ax.set_xlabel("Accepted PSMs by native-score rank")
    ax.set_ylabel("PSM q-value")
    ax.set_title(title)
    y_top = min(max(q_max * 1.15, 0.05), 1.0)
    ax.set_ylim(0, y_top)
    ax.legend(loc="upper left")
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, output_path)
    logger.info("Wrote %s", output_path)


def _count_at_thresholds(
    q: np.ndarray, thresholds: list[float] = FDR_THRESHOLDS
) -> list[int]:
    q = np.asarray(q, dtype=float)
    valid = q[~np.isnan(q)]
    return [int((valid <= t).sum()) for t in thresholds]


def _bar_series_from_curves(curves: list[MethodCurve]) -> list[MethodCounts]:
    return [
        MethodCounts(
            label=c.label,
            color=c.color,
            counts=_count_at_thresholds(c.q_value),
        )
        for c in curves
    ]


def _recovery_at_thresholds(
    q: np.ndarray,
    correct: np.ndarray,
    thresholds: list[float] = FDR_THRESHOLDS,
) -> list[float]:
    """Return correct-identification recovery percentage at each q-value threshold."""
    q = np.asarray(q, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    denom = int(correct.sum())
    if denom == 0:
        return [np.nan for _ in thresholds]
    valid = ~np.isnan(q)
    return [100.0 * int((valid & correct & (q <= t)).sum()) / denom for t in thresholds]


def plot_recovery_curves(
    series: list[MethodRecovery],
    dataset_key: str,
    output_path: Path,
) -> None:
    """Plot correct-identification recovery versus q-value threshold."""
    display = _display_name(dataset_key)
    fig, ax = plt.subplots(figsize=(8, 6))
    for item in series:
        y = _recovery_at_thresholds(item.q_value, item.correct)
        ax.plot(
            FDR_THRESHOLDS,
            y,
            marker="o",
            lw=1.5,
            label=item.label,
            color=item.color,
        )

    ax.set_xlim(0, max(FDR_THRESHOLDS))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Estimated q-value threshold")
    ax.set_ylabel("Correct PSM recovery\n(% of labelled correct PSMs)")
    ax.set_title(f"{display} labelled PSM recovery by q-value threshold")
    ax.legend(loc="upper left")
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, output_path)
    logger.info("Wrote %s", output_path)


def plot_threshold_barplot(
    series: list[MethodCounts],
    dataset_key: str,
    eval_type: EvalType,
    output_path: Path,
) -> None:
    """Bar chart of identifications retained at each q-value threshold."""
    display = _display_name(dataset_key)
    if eval_type == "labelled":
        split_label = "labelled test set"
    else:
        split_label = "unlabelled set"
    title = f"{display}: accepted PSMs on the {split_label} at q-value thresholds"
    ylabel = "Peptide-spectrum matches"

    n_methods = len(series)
    n_thresh = len(FDR_THRESHOLDS)
    group_spacing = 0.825
    cluster_width = min(0.75, group_spacing * 0.92)
    width = cluster_width / n_methods
    x = np.arange(n_thresh) * group_spacing

    fig_w = max(10.0, 2.2 * n_thresh * group_spacing)
    fig, ax = plt.subplots(figsize=(fig_w, 7))
    for i, item in enumerate(series):
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            item.counts,
            width,
            label=item.label,
            color=item.color,
            edgecolor="black",
            linewidth=1,
        )
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{int(h):,}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    max_count = max((c for item in series for c in item.counts), default=1)
    y_headroom = (1.55 + 0.06 * n_methods) * (2 / 3)
    ax.set_ylim(0, max_count * y_headroom)

    half_cluster = cluster_width / 2
    ax.set_xlim(
        -half_cluster - 0.25,
        (n_thresh - 1) * group_spacing + half_cluster + 0.25,
    )

    ax.set_xlabel("Q-value threshold")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in FDR_THRESHOLDS])
    ax.legend(loc="upper left")
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, output_path)
    logger.info("Wrote %s", output_path)


def _append_method_summary_rows(
    acceptance_rows: list[dict[str, object]],
    error_rows: list[dict[str, object]],
    *,
    dataset: str,
    panel: str,
    level: str,
    method: str,
    q_value: np.ndarray,
    label_mask: np.ndarray | None = None,
    recovery_denom: int | None = None,
    q_ref: np.ndarray | None = None,
) -> None:
    """Append acceptance and error rows for one method."""
    acceptance_rows.extend(
        acceptance_rows_from_q(
            dataset=dataset,
            panel=panel,
            level=level,
            method=method,
            q_value=q_value,
            thresholds=SUMMARY_THRESHOLDS,
            label_mask=label_mask,
            recovery_denom=recovery_denom,
        )
    )
    error_rows.extend(
        error_rows_from_q(
            dataset=dataset,
            panel=panel,
            level=level,
            method=method,
            q_value=q_value,
            thresholds=SUMMARY_THRESHOLDS,
            label_mask=label_mask,
            q_ref=q_ref,
        )
    )


def summary_rows_from_curves(
    curves: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build acceptance and error summary rows from a long-form curves table."""
    acceptance_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    required = {"dataset", "panel", "method", "spectrum_id", "q_value", "label"}
    missing = required - set(curves.columns)
    if missing:
        raise ValueError(f"Curves table missing columns: {sorted(missing)}")

    for (dataset, panel), group in curves.groupby(["dataset", "panel"], sort=False):
        cal = group.loc[group["method"] == DB_CAL_METHOD, ["spectrum_id", "q_value"]]
        q_ref_cal_by_id = cal.drop_duplicates("spectrum_id").set_index("spectrum_id")[
            "q_value"
        ]
        raw = group.loc[group["method"] == DB_RAW_METHOD, ["spectrum_id", "q_value"]]
        q_ref_raw_by_id = raw.drop_duplicates("spectrum_id").set_index("spectrum_id")[
            "q_value"
        ]
        for method, mdf in group.groupby("method", sort=False):
            method_s = str(method)
            q_value = mdf["q_value"].to_numpy(dtype=float)
            label_mask = mdf["label"].astype(bool).to_numpy()
            recovery_denom = int(label_mask.sum())
            q_ref: np.ndarray | None = None
            # Winnow-family methods: deviation vs calibrated-confidence DBG.
            # NovoBoard / Glissade: deviation vs raw-confidence DBG.
            if method_s in _WINNOW_METHODS:
                q_ref = (
                    mdf["spectrum_id"]
                    .astype(str)
                    .map(q_ref_cal_by_id)
                    .to_numpy(dtype=float)
                )
            elif method_s in (NOVOBOARD_METHOD, "Glissade"):
                q_ref = (
                    mdf["spectrum_id"]
                    .astype(str)
                    .map(q_ref_raw_by_id)
                    .to_numpy(dtype=float)
                )
            _append_method_summary_rows(
                acceptance_rows,
                error_rows,
                dataset=str(dataset),
                panel=str(panel),
                level="psm",
                method=method_s,
                q_value=q_value,
                label_mask=label_mask,
                recovery_denom=recovery_denom,
                q_ref=q_ref,
            )
    return acceptance_rows, error_rows


def _comparators_for_panel(panel: str, level: str) -> list[str]:
    """Comparator method labels present in a given summary panel."""
    del panel, level
    return ["NovoBoard"]


def _finalise_method_comparison_tables(
    acceptance_rows: list[dict[str, object]],
    error_rows: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build acceptance and error/gain DataFrames with per-panel relative columns."""
    acceptance = pd.DataFrame(acceptance_rows)
    error = pd.DataFrame(error_rows)
    if acceptance.empty:
        return acceptance, error

    gain_parts: list[pd.DataFrame] = []
    group_cols = ["dataset", "panel", "level"]
    for keys, acc_group in acceptance.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        _, panel, level = keys
        err_mask = True
        for col, val in zip(group_cols, keys):
            err_mask = err_mask & (error[col] == val)
        err_group = error.loc[err_mask]
        gain_parts.append(
            finalise_error_gain_table(
                acc_group,
                err_group,
                primary_method=PRIMARY_METHOD,
                comparators=_comparators_for_panel(str(panel), str(level)),
            )
        )
    error_gain = pd.concat(gain_parts, ignore_index=True) if gain_parts else error
    return acceptance, error_gain


def process_dataset(cfg: DatasetConfig, plots_dir: Path) -> pd.DataFrame:
    """Generate comparison plots and return long-form curve rows for one dataset."""
    residue_masses = _load_residue_masses()

    winnow_unlabelled = load_winnow(cfg.winnow_unlabelled, cfg.fasta, "unlabelled")
    winnow_test = load_winnow(cfg.winnow_test, cfg.fasta, "labelled")
    nb_u_target, nb_u_decoy = load_novoboard_target_decoy(
        cfg.novoboard_dir, "unlabelled", cfg.novoboard_decoy_rate
    )
    nb_t_target, nb_t_decoy = load_novoboard_target_decoy(
        cfg.novoboard_dir, "test", cfg.novoboard_decoy_rate
    )
    nb_u_target, nb_u_decoy = filter_novoboard_target_decoy_pairs(
        nb_u_target, nb_u_decoy, min_length=MIN_PEPTIDE_LENGTH
    )
    nb_t_target, nb_t_decoy = filter_novoboard_target_decoy_pairs(
        nb_t_target, nb_t_decoy, min_length=LABELLED_MIN_PEPTIDE_LENGTH
    )
    # Pair-gated NovoBoard ⊆ Winnow after identical InstaNovo filters; only trim Winnow.
    winnow_unlabelled = _restrict_winnow_to_novoboard_spectra(
        winnow_unlabelled, nb_u_target
    )
    winnow_test = _restrict_winnow_to_novoboard_spectra(winnow_test, nb_t_target)
    _assert_shared_prediction_keys(winnow_test, nb_t_target)
    _assert_shared_prediction_keys(winnow_unlabelled, nb_u_target)

    # Shared labels once on Winnow; NovoBoard reuses them by spectrum_id.
    winnow_test = winnow_test.copy()
    winnow_test["correct"] = novor_correctness_mask(
        winnow_test["sequence"],
        winnow_test["prediction"],
        residue_masses=residue_masses,
    )
    if "proteome_hit" not in winnow_unlabelled.columns:
        raise KeyError("Expected proteome_hit on unlabelled Winnow table")
    correct_by_id = _label_series_by_spectrum_id(winnow_test, "correct")
    hit_by_id = _label_series_by_spectrum_id(winnow_unlabelled, "proteome_hit")

    novoboard_unlabelled = novoboard_psm_tdc(
        nb_u_target, nb_u_decoy, min_length=MIN_PEPTIDE_LENGTH
    )
    n_u_tgt = int(novoboard_unlabelled["is_target"].sum())
    n_u_dec = int((~novoboard_unlabelled["is_target"]).sum())
    if n_u_tgt != n_u_dec:
        raise AssertionError(
            f"Unlabelled PSM TDC unbalanced: targets={n_u_tgt} decoys={n_u_dec}"
        )
    novoboard_unlabelled = novoboard_unlabelled[
        novoboard_unlabelled["is_target"]
    ].copy()
    novoboard_test = novoboard_psm_tdc(
        nb_t_target, nb_t_decoy, min_length=LABELLED_MIN_PEPTIDE_LENGTH
    )
    n_t_tgt = int(novoboard_test["is_target"].sum())
    n_t_dec = int((~novoboard_test["is_target"]).sum())
    if n_t_tgt != n_t_dec:
        raise AssertionError(
            f"Labelled PSM TDC unbalanced: targets={n_t_tgt} decoys={n_t_dec}"
        )
    novoboard_test = novoboard_test[novoboard_test["is_target"]].copy()
    novoboard_test = _attach_labels_by_spectrum_id(
        novoboard_test, correct_by_id, label_col="correct"
    )
    novoboard_unlabelled = _attach_labels_by_spectrum_id(
        novoboard_unlabelled, hit_by_id, label_col="proteome_hit"
    )

    n_w_correct = int(winnow_test["correct"].sum())
    n_nb_correct = int(novoboard_test["correct"].sum())
    if n_w_correct != n_nb_correct:
        raise AssertionError(
            f"Shared labelled correct counts disagree: Winnow={n_w_correct} "
            f"NovoBoard={n_nb_correct}"
        )
    n_w_hit = int(winnow_unlabelled["proteome_hit"].sum())
    n_nb_hit = int(novoboard_unlabelled["proteome_hit"].sum())
    if n_w_hit != n_nb_hit:
        raise AssertionError(
            f"Shared proteome-hit counts disagree: Winnow={n_w_hit} NovoBoard={n_nb_hit}"
        )

    logger.info(
        "%s shared PSM pools: unlabelled=%d labelled=%d "
        "(NovoBoard twin-valid; Winnow trimmed; shared labels correct=%d hits=%d)",
        cfg.key,
        len(winnow_unlabelled),
        len(winnow_test),
        n_w_correct,
        n_w_hit,
    )

    winnow_u_psm_table = _prepare_winnow_psm_table(
        winnow_unlabelled, "proteome_hit", residue_masses
    )
    winnow_t_psm_table = _prepare_winnow_psm_table(
        winnow_test, "correct", residue_masses
    )
    curves = pd.concat(
        [
            _curves_df_from_winnow_table(
                winnow_t_psm_table,
                dataset=cfg.key,
                panel="labelled_test",
                label_col="correct",
            ),
            _curves_df_from_novoboard(
                novoboard_test,
                dataset=cfg.key,
                panel="labelled_test",
                label_col="correct",
            ),
            _curves_df_from_winnow_table(
                winnow_u_psm_table,
                dataset=cfg.key,
                panel="unlabelled",
                label_col="proteome_hit",
            ),
            _curves_df_from_novoboard(
                novoboard_unlabelled,
                dataset=cfg.key,
                panel="unlabelled",
                label_col="proteome_hit",
            ),
        ],
        ignore_index=True,
    )
    plot_dataset_from_curves(curves, cfg.key, plots_dir)
    return curves


@app.command()
def main(
    novoboard_root: Annotated[
        Optional[Path],
        typer.Option(
            "--novoboard-root",
            help=(
                "Root of NovoBoard per-dataset tables: "
                "{root}/{dataset}/novoboard/ with annotated_test*.csv and "
                "raw_unlabelled*.csv target/decoy pairs (the datasets/ dir of "
                "a NovoBoard checkout). Required unless --summarise-only is set. "
                "Local runs used fork JemmaLDaniel/NovoBoard, branch "
                "feat/adapt-to-instanovo "
                "(commit a9faab3ef1af06987599c2f01e6ba96072c80172)."
            ),
        ),
    ] = None,
    results_dir: Annotated[
        Path,
        typer.Option("--results-dir", help="Directory for curves/summary CSVs."),
    ] = DEFAULT_OUTPUT_DIR,
    plots_dir: Annotated[
        Path,
        typer.Option("--plots-dir", help="Directory for png/pdf figures."),
    ] = DEFAULT_OUTPUT_DIR,
    datasets: Annotated[
        Optional[list[str]],
        typer.Option("--datasets", help="Dataset keys to plot."),
    ] = None,
    winnow_results: Annotated[
        Path,
        typer.Option("--winnow-results", help="Winnow results directory."),
    ] = DEFAULT_WINNOW_RESULTS,
    summarise_only: Annotated[
        Optional[Path],
        typer.Option(
            "--summarise-only",
            help="Only write plots and summary CSVs from an existing curves CSV.",
        ),
    ] = None,
) -> None:
    """Generate FDR method comparison plots and summary CSVs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if summarise_only is not None:
        curves = pd.read_csv(summarise_only, float_precision="round_trip")
        if "label" in curves.columns:
            curves["label"] = curves["label"].astype(bool)
        dataset_keys = (
            datasets
            if datasets is not None
            else sorted(curves["dataset"].astype(str).unique())
        )
        for key in dataset_keys:
            logger.info("Replotting %s from curves CSV", key)
            plot_dataset_from_curves(curves, str(key), plots_dir)
        curves_out = curves.loc[curves["dataset"].astype(str).isin(dataset_keys)]
        acceptance_rows, error_rows = summary_rows_from_curves(curves_out)
        acceptance, error_gain = _finalise_method_comparison_tables(
            acceptance_rows, error_rows
        )
        write_summary_tables(
            acceptance, error_gain, results_dir, "fdr_method_comparison"
        )
        if summarise_only.resolve() != (results_dir / CURVES_CSV_NAME).resolve():
            write_curves_csv(curves_out, results_dir)
        return

    if novoboard_root is None:
        raise typer.BadParameter(
            "--novoboard-root is required unless --summarise-only is set."
        )

    dataset_keys = datasets if datasets is not None else list(DEFAULT_DATASETS)
    configs = build_dataset_configs(winnow_results, novoboard_root=novoboard_root)

    curve_parts: list[pd.DataFrame] = []
    for key in dataset_keys:
        if key not in configs:
            raise typer.BadParameter(f"Unknown dataset {key!r}")
        logger.info("Processing %s", key)
        curve_parts.append(process_dataset(configs[key], plots_dir))

    curves = pd.concat(curve_parts, ignore_index=True)
    write_curves_csv(curves, results_dir)
    acceptance_rows, error_rows = summary_rows_from_curves(curves)
    acceptance, error_gain = _finalise_method_comparison_tables(
        acceptance_rows, error_rows
    )
    write_summary_tables(acceptance, error_gain, results_dir, "fdr_method_comparison")


if __name__ == "__main__":
    app()
