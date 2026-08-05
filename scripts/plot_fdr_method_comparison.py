#!/usr/bin/env python3
"""Compare FDR/q-value estimates across Winnow, NovoBoard, and Glissade.

PSM-level plots (Winnow vs NovoBoard): q-value curves and PSM counts at 1 %, 5 %, 10 %.

Peptide-level plots (all three methods): deduplicate to one row per peptide (max native
score), then estimate FDR on the peptide table (Glissade order). Winnow uses
``NonParametricFDRControl`` + database-grounded on deduped rows; NovoBoard peptide curves
use NovoBoard-style target-decoy competition after target and decoy peptide deduplication.

Styling matches ``scripts/plot_eval_results.py``. Database-grounded FDR uses
``DatabaseGroundedFDRControl`` (proteome-hit shortcut for unlabelled; full fit for labelled).
Proteome hits use ``scripts.annotate_preds_proteome_hits`` (FASTA I/L normalisation,
``(+mass)`` and ``[UNIMOD:n]`` stripping).
"""

from __future__ import annotations

import logging
import re
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

from scripts.annotate_preds_proteome_hits import (  # noqa: E402
    _batch_peptide_substring_hits,
    filter_and_annotate_preds,
    load_proteome_haystack,
)
from scripts.fdr_tool_comparison_summaries import (  # noqa: E402
    SUMMARY_THRESHOLDS,
    acceptance_rows_from_q,
    error_rows_from_q,
    finalise_error_gain_table,
    write_summary_tables,
)
from scripts.plot_eval_results import (  # noqa: E402
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

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

sns.set_theme(style="white", palette=_PALETTE, context="paper", font_scale=1.5)

FDR_THRESHOLDS = [0.01, 0.05, 0.10]
_DB_GROUNDED_DROP = 10

DEFAULT_NOVOBOARD_ROOT = Path("/home/j-daniel/repos/NovoBoard/datasets")
DEFAULT_GLISSADE_ROOT = Path("/home/j-daniel/repos/glissade/build")
DEFAULT_WINNOW_RESULTS = _REPO_ROOT / "results"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "results/fdr_method_comparison"

EvalType = Literal["labelled", "unlabelled"]

_DATASET_META = {
    "helaqc": {
        "fasta": "fasta/human.fasta",
        "novoboard_decoy": "0.50",
        "winnow_suffix": "helaqc",
    },
    "celegans": {
        "fasta": "fasta/Celegans.fasta",
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

NORM_MAP: dict[str, Literal["winnow", "novoboard", "glissade"]] = {
    "Winnow (non-parametric)": "winnow",
    "Database-grounded (calibrated confidence)": "winnow",
    "Database-grounded (raw confidence)": "winnow",
    "NovoBoard": "novoboard",
    "Glissade": "glissade",
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
    glissade_dir: Path


def build_dataset_configs(
    winnow_results: Path = DEFAULT_WINNOW_RESULTS,
    novoboard_root: Path = DEFAULT_NOVOBOARD_ROOT,
    glissade_root: Path = DEFAULT_GLISSADE_ROOT,
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
            glissade_dir=glissade_root / key,
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
    """PSM or peptide counts per q-value threshold for one method."""

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
    import yaml

    path = _REPO_ROOT / "winnow/configs/residues.yaml"
    with open(path) as f:
        return yaml.safe_load(f)["residue_masses"]


def _metrics():
    from instanovo.utils.metrics import Metrics
    from instanovo.utils.residues import ResidueSet

    return Metrics(
        residue_set=ResidueSet(residue_masses=_load_residue_masses()),
        isotope_error_range=(0, 1),
    )


def _fit_database_grounded_fdr(
    df: pd.DataFrame,
    correct_col: str,
    confidence_col: str,
    residue_masses: dict[str, float],
    *,
    drop: int = _DB_GROUNDED_DROP,
) -> DatabaseGroundedFDRControl:
    """Fit ``DatabaseGroundedFDRControl`` (proteome shortcut or labelled sequence fit)."""
    ctrl = DatabaseGroundedFDRControl(
        confidence_feature=confidence_col,
        residue_masses=residue_masses,
        drop=drop,
    )
    if correct_col == "proteome_hit":
        sorted_df = df.sort_values(confidence_col, ascending=False)
        labels = sorted_df[correct_col].astype(float).to_numpy()
        conf = sorted_df[confidence_col].to_numpy()
        precision = np.cumsum(labels) / np.arange(1, len(labels) + 1)
        ctrl._fdr_values = np.array(1.0 - precision)[drop:]
        ctrl._confidence_scores = conf[drop:]
    else:
        fit_df = df.copy()
        if "sequence" not in fit_df.columns or "prediction" not in fit_df.columns:
            raise ValueError(
                "Labelled database-grounded FDR requires 'sequence' and 'prediction' columns"
            )
        ctrl.fit(dataset=fit_df, correct_column=correct_col)
    return ctrl


def normalise_confidence_for_plot(
    scores: np.ndarray, method: Literal["winnow", "novoboard", "glissade"]
) -> np.ndarray:
    """Map native scores to [0, 1] for shared x-axis (plotting only)."""
    s = np.asarray(scores, dtype=float)
    if method == "winnow":
        out = s
    elif method == "novoboard":
        out = s / 100.0
    elif method == "glissade":
        out = np.exp(s)
    else:
        raise ValueError(method)

    if np.any(out < -1e-6) or np.any(out > 1 + 1e-6):
        logger.warning(
            "%s confidence outside [0, 1] after normalisation (min=%.4f, max=%.4f); clipping",
            method,
            float(np.nanmin(out)),
            float(np.nanmax(out)),
        )
    return np.clip(out, 0.0, 1.0)


def load_winnow(
    predictions_dir: Path, fasta: Path, eval_type: EvalType
) -> pd.DataFrame:
    """Load Winnow preds + metadata; annotate proteome hits or use labelled ``correct``."""
    preds = pl.read_csv(predictions_dir / "preds_and_fdr_metrics.csv")
    meta_path = predictions_dir / "metadata.csv"
    if meta_path.exists():
        meta = pl.read_csv(meta_path, columns=["spectrum_id", "confidence"])
        preds = preds.join(meta, on="spectrum_id", how="inner")

    if eval_type == "labelled":
        if "correct" not in preds.columns:
            raise ValueError(
                f"Missing 'correct' in {predictions_dir}/preds_and_fdr_metrics.csv"
            )
        return preds.to_pandas()

    metrics = _metrics()
    haystack = load_proteome_haystack(fasta)
    preds = filter_and_annotate_preds(preds, haystack, metrics, min_residue_length=7)
    return preds.to_pandas()


def _effective_db_grounded_drop(n_rows: int, drop: int = _DB_GROUNDED_DROP) -> int:
    """Cap drop so FDR fit retains at least one score when *n_rows* is small (peptide dedupe)."""
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
    q_sorted = _compute_q_values(fdr[order])
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
        residue_masses,
        drop=_effective_db_grounded_drop(len(reference), drop),
    )
    return _assign_q_values_fast(work, confidence_col, ctrl, out_col)


def normalize_peptide_key(peptide: object) -> str:
    """Normalize sequence-only peptide identity by stripping PTM annotations."""
    if isinstance(peptide, list):
        peptide = "".join(str(token) for token in peptide)
    if pd.isna(peptide) or not isinstance(peptide, str):
        return ""
    if len(peptide) > 4 and peptide[1] == "." and peptide[-2] == ".":
        peptide = peptide[2:-2]
    seq = re.sub(r"\[.*?\]", "", peptide)
    seq = re.sub(r"\(.*?\)", "", seq)
    seq = "".join(c for c in seq if c.isalpha())
    return seq.replace("I", "L")


def sequence_only_correct_prediction(sequence: object, prediction: object) -> bool:
    """Return full sequence equality after PTM stripping and I/L normalization."""
    sequence_key = normalize_peptide_key(sequence)
    prediction_key = normalize_peptide_key(prediction)
    return bool(sequence_key) and sequence_key == prediction_key


def sequence_only_correctness_mask(
    sequences: pd.Series, predictions: pd.Series
) -> np.ndarray:
    """Vectorized external correctness rule for method comparison plots."""
    return np.array(
        [
            sequence_only_correct_prediction(sequence, prediction)
            for sequence, prediction in zip(sequences, predictions)
        ],
        dtype=bool,
    )


def dedupe_best_score_per_peptide(
    df: pd.DataFrame, peptide_col: str, score_col: str
) -> pd.DataFrame:
    """Keep the highest-scoring row per peptide (Glissade ``groupby(...).max()`` rule)."""
    return (
        df.sort_values(score_col, ascending=False)
        .groupby(peptide_col, as_index=False)
        .first()
    )


def _dedupe_winnow_peptides(
    df: pd.DataFrame, fit_df: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Add peptide keys and dedupe to best calibrated score per sequence."""
    work = df.copy()
    work["_peptide_key"] = work["prediction"].map(normalize_peptide_key)
    deduped = dedupe_best_score_per_peptide(
        work, "_peptide_key", "calibrated_confidence"
    )
    ref_deduped: pd.DataFrame | None = None
    if fit_df is not None:
        ref = fit_df.copy()
        ref["_peptide_key"] = ref["prediction"].map(normalize_peptide_key)
        ref_deduped = dedupe_best_score_per_peptide(
            ref, "_peptide_key", "calibrated_confidence"
        )
    return deduped, ref_deduped


def prepare_winnow_psm_curves(
    df: pd.DataFrame,
    correct_col: str,
    residue_masses: dict[str, float],
    *,
    fit_df: pd.DataFrame | None = None,
) -> list[MethodCurve]:
    """PSM-level Winnow curves (non-parametric + database-grounded cal/raw)."""
    return _winnow_curves_from_table(
        _prepare_winnow_psm_table(df, correct_col, residue_masses, fit_df=fit_df)
    )


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


def prepare_winnow_peptide_curves(
    df: pd.DataFrame,
    correct_col: str,
    residue_masses: dict[str, float],
    *,
    fit_df: pd.DataFrame | None = None,
) -> list[MethodCurve]:
    """Peptide-level Winnow curves: dedupe by max score, then estimate FDR on peptides."""
    return _winnow_curves_from_table(
        _prepare_winnow_peptide_table(df, correct_col, residue_masses, fit_df=fit_df),
        monotonize=False,
    )


def _prepare_winnow_peptide_table(
    df: pd.DataFrame,
    correct_col: str,
    residue_masses: dict[str, float],
    *,
    fit_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Dedupe to best calibrated score per peptide and append peptide-level q-values."""
    deduped, ref_deduped = _dedupe_winnow_peptides(df, fit_df)
    fit_reference = ref_deduped if ref_deduped is not None else deduped

    np_ctrl = NonParametricFDRControl()
    np_ctrl.fit(deduped["calibrated_confidence"])
    table = deduped.drop(columns=["psm_q_value"], errors="ignore").copy()
    table = _assign_q_values_fast(
        table, "calibrated_confidence", np_ctrl, "psm_q_value"
    )
    table = _add_database_grounded_qvalues(
        table,
        correct_col,
        "calibrated_confidence",
        "psm_q_value_db_cal",
        residue_masses,
        fit_df=fit_reference,
    )
    table = _add_database_grounded_qvalues(
        table,
        correct_col,
        "confidence",
        "psm_q_value_db_raw",
        residue_masses,
        fit_df=fit_reference,
    )
    cal_conf = table["calibrated_confidence"].to_numpy()
    raw_conf = table["confidence"].to_numpy()
    table = table.copy()
    table["psm_q_value"] = _monotonize_q_by_confidence(
        cal_conf, table["psm_q_value"].to_numpy()
    )
    table["psm_q_value_db_cal"] = _monotonize_q_by_confidence(
        cal_conf, table["psm_q_value_db_cal"].to_numpy()
    )
    table["psm_q_value_db_raw"] = _monotonize_q_by_confidence(
        raw_conf, table["psm_q_value_db_raw"].to_numpy()
    )
    return table


def _winnow_curves_from_table(
    table: pd.DataFrame, *, monotonize: bool = False
) -> list[MethodCurve]:
    cal_conf = table["calibrated_confidence"].to_numpy()
    raw_conf = table["confidence"].to_numpy()
    np_q = table["psm_q_value"].to_numpy()
    db_cal_q = table["psm_q_value_db_cal"].to_numpy()
    db_raw_q = table["psm_q_value_db_raw"].to_numpy()
    if monotonize:
        np_q = _monotonize_q_by_confidence(cal_conf, np_q)
        db_cal_q = _monotonize_q_by_confidence(cal_conf, db_cal_q)
        db_raw_q = _monotonize_q_by_confidence(raw_conf, db_raw_q)
    return [
        MethodCurve("Winnow (non-parametric)", _MAIN_LINE_COLOUR, cal_conf, np_q),
        MethodCurve(
            "Database-grounded (calibrated confidence)",
            _RAW_LINE_COLOUR,
            cal_conf,
            db_cal_q,
        ),
        MethodCurve(
            "Database-grounded (raw confidence)",
            _PALETTE[3],
            raw_conf,
            db_raw_q,
        ),
    ]


def _non_database_curves(curves: list[MethodCurve]) -> list[MethodCurve]:
    """Drop database-grounded curves where reference hits are not meaningful."""
    return [c for c in curves if not c.label.startswith("Database-grounded")]


def load_novoboard(
    novoboard_dir: Path, split: Literal["unlabelled", "test"], decoy_rate: str
) -> pd.DataFrame:
    """Load NovoBoard q-value table for a split and decoy rate."""
    if split == "unlabelled":
        path = (
            novoboard_dir
            / f"raw_unlabelled_fdr_raw_unlabelled_decoy_{decoy_rate}_qvalues.csv"
        )
    else:
        path = (
            novoboard_dir
            / f"annotated_test_fdr_annotated_test_decoy_{decoy_rate}_qvalues.csv"
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_novoboard_target_decoy(
    novoboard_dir: Path, split: Literal["unlabelled", "test"], decoy_rate: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load NovoBoard target and decoy prediction tables for peptide-level TDC."""
    prefix = "raw_unlabelled" if split == "unlabelled" else "annotated_test"
    target_path = novoboard_dir / f"{prefix}.csv"
    decoy_path = novoboard_dir / f"{prefix}_decoy_{decoy_rate}.csv"
    if not target_path.is_file():
        raise FileNotFoundError(target_path)
    if not decoy_path.is_file():
        raise FileNotFoundError(decoy_path)
    return pd.read_csv(target_path), pd.read_csv(decoy_path)


def load_glissade(glissade_dir: Path) -> pd.DataFrame:
    """Load Glissade's native peptide-level q-value table."""
    path = glissade_dir / "peptide.tsv"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t")
    return df.rename(
        columns={"Peptide": "peptide", "Score": "score", "q-value": "q-value"}
    )


def novoboard_psm_curve(df: pd.DataFrame) -> MethodCurve:
    """Build a PSM-level NovoBoard q-value curve."""
    return MethodCurve(
        "NovoBoard",
        _PALETTE[2],
        df["ALC (%)"].to_numpy(),
        df["estimated_q_value"].to_numpy(),
    )


def _compute_q_values(fdr: np.ndarray) -> np.ndarray:
    """Convert ranked FDR estimates to q-values using suffix minima."""
    values = np.asarray(fdr, dtype=float)
    q_values = np.empty_like(values)
    fdr_min = np.inf
    for i in range(len(values) - 1, -1, -1):
        fdr_min = min(fdr_min, values[i])
        q_values[i] = fdr_min
    return q_values


def _dedupe_novoboard_peptides(df: pd.DataFrame) -> pd.DataFrame:
    work = df.dropna(subset=["Peptide", "ALC (%)"]).copy()
    work["_peptide_key"] = work["Peptide"].map(normalize_peptide_key)
    work = work[work["_peptide_key"] != ""]
    return dedupe_best_score_per_peptide(work, "_peptide_key", "ALC (%)")


def _novoboard_peptide_tdc_table(
    target_df: pd.DataFrame,
    decoy_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run NovoBoard-style target-decoy competition after peptide deduplication."""
    target = _dedupe_novoboard_peptides(target_df)
    decoy = _dedupe_novoboard_peptides(decoy_df)
    target = target.assign(is_target=True)
    decoy = decoy.assign(is_target=False)
    combined = pd.concat([target, decoy], ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["ALC (%)", "is_target"], ascending=[False, False]
    ).reset_index(drop=True)

    n_target = combined["is_target"].astype(int).cumsum()
    n_decoy = (~combined["is_target"]).astype(int).cumsum()
    estimated_fdr = np.divide(
        n_decoy,
        n_target,
        out=np.ones(len(combined), dtype=float),
        where=n_target > 0,
    )
    combined["estimated_fdr"] = estimated_fdr
    combined["estimated_q_value"] = np.nan
    target_mask = combined["is_target"]
    combined.loc[target_mask, "estimated_q_value"] = _compute_q_values(
        combined.loc[target_mask, "estimated_fdr"].to_numpy()
    )
    return combined


def novoboard_peptide_curve(
    target_df: pd.DataFrame,
    decoy_df: pd.DataFrame,
    *,
    target_spectrum_ids: set[str] | None = None,
    target_peptide_keys: set[str] | None = None,
) -> MethodCurve:
    """Peptide-level NovoBoard curve from deduped target/decoy peptide TDC."""
    table = _novoboard_peptide_tdc_table(target_df, decoy_df)
    target = table[table["is_target"]].copy()
    if target_spectrum_ids is not None:
        target = target[target["spectrum_id"].astype(str).isin(target_spectrum_ids)]
    if target_peptide_keys is not None:
        target = target[target["_peptide_key"].isin(target_peptide_keys)]
    conf = target["ALC (%)"].to_numpy()
    q = _monotonize_q_by_confidence(conf, target["estimated_q_value"].to_numpy())
    return MethodCurve("NovoBoard", _PALETTE[2], conf, q)


def _monotonize_q_by_confidence(
    confidence: np.ndarray, q_value: np.ndarray
) -> np.ndarray:
    """Enforce non-increasing q-values when confidence increases."""
    order = np.argsort(-np.asarray(confidence, dtype=float))
    q_sorted = np.asarray(q_value, dtype=float)[order]
    q_mono = np.empty_like(q_sorted)
    q_min = np.inf
    for i in range(len(q_sorted) - 1, -1, -1):
        if q_sorted[i] > q_min:
            q_mono[i] = q_min
        else:
            q_mono[i] = q_sorted[i]
            q_min = q_sorted[i]
    out = np.empty_like(q_mono)
    out[order] = q_mono
    return out


def glissade_peptide_curve(df: pd.DataFrame) -> MethodCurve:
    """Peptide-level Glissade curve from Glissade's native peptide output."""
    mask = df["q-value"].notna()
    per = df.loc[mask, ["peptide", "score", "q-value"]]
    conf = per["score"].to_numpy()
    q = _monotonize_q_by_confidence(conf, per["q-value"].to_numpy())
    return MethodCurve("Glissade", _PALETTE[4], conf, q)


def _external_peptide_keys(peptides: pd.Series, reference_haystack: str) -> set[str]:
    """Return normalized peptide keys absent from the reference FASTA."""
    unique_keys = [key for key in peptides.map(normalize_peptide_key).unique() if key]
    hits = _batch_peptide_substring_hits(unique_keys, reference_haystack)
    return {key for key, hit in zip(unique_keys, hits) if not hit}


def plot_qvalue_comparison(
    curves: list[MethodCurve],
    dataset_key: str,
    eval_type: EvalType,
    output_path: Path,
    *,
    level: Literal["psm", "peptide"] = "psm",
    title_suffix: str = "",
) -> None:
    """Plot q-value vs confidence curves for multiple FDR methods."""
    display = _display_name(dataset_key)
    qualifier = _ground_truth_qualifier(
        "labelled" if eval_type == "labelled" else "unlabelled"
    )
    level_tag = "peptide" if level == "peptide" else "PSM"
    title = f"{display} {level_tag} q-value comparison {qualifier}{title_suffix}"

    fig, ax = plt.subplots(figsize=(8, 6))
    q_max = 0.0
    for curve in curves:
        norm_kind = NORM_MAP[curve.label]
        x = normalise_confidence_for_plot(curve.confidence, norm_kind)
        order = np.argsort(x)
        y = curve.q_value[order]
        q_max = max(q_max, float(np.nanmax(y)))
        ax.plot(
            x[order],
            y,
            color=curve.color,
            lw=1.5,
            label=curve.label,
        )

    ax.set_xlabel("Model confidence")
    ax.set_ylabel(f"{level_tag} q-value")
    ax.set_title(title)
    # Scale y-axis to the data with modest headroom for the upper-right legend.
    y_top = min(max(q_max * 1.15, 0.05), 1.0)
    ax.set_ylim(0, y_top)
    ax.legend(loc="upper right")
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, output_path)
    logger.info("Wrote %s", output_path)


def plot_qvalue_by_rank(
    curves: list[MethodCurve],
    dataset_key: str,
    eval_type: EvalType,
    output_path: Path,
    *,
    level: Literal["psm", "peptide"] = "psm",
    title_suffix: str = "",
) -> None:
    """Plot q-value against native-score rank/accepted count."""
    display = _display_name(dataset_key)
    qualifier = _ground_truth_qualifier(
        "labelled" if eval_type == "labelled" else "unlabelled"
    )
    level_tag = "peptide" if level == "peptide" else "PSM"
    title = f"{display} {level_tag} q-value by rank {qualifier}{title_suffix}"

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

    ax.set_xlabel(f"Accepted {level_tag}s by native-score rank")
    ax.set_ylabel(f"{level_tag} q-value")
    ax.set_title(title)
    y_top = min(max(q_max * 1.15, 0.05), 1.0)
    ax.set_ylim(0, y_top)
    ax.legend(loc="upper right")
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


def _winnow_psm_recovery_series(
    df: pd.DataFrame,
    correct_col: str,
    residue_masses: dict[str, float],
) -> list[MethodRecovery]:
    table = _prepare_winnow_psm_table(df, correct_col, residue_masses)
    if {"sequence", "prediction"}.issubset(table.columns):
        correct = sequence_only_correctness_mask(table["sequence"], table["prediction"])
    else:
        correct = table[correct_col].astype(bool).to_numpy()
    return [
        MethodRecovery(
            "Winnow (non-parametric)",
            _MAIN_LINE_COLOUR,
            table["psm_q_value"].to_numpy(),
            correct,
        ),
        MethodRecovery(
            "Database-grounded (calibrated confidence)",
            _RAW_LINE_COLOUR,
            table["psm_q_value_db_cal"].to_numpy(),
            correct,
        ),
        MethodRecovery(
            "Database-grounded (raw confidence)",
            _PALETTE[3],
            table["psm_q_value_db_raw"].to_numpy(),
            correct,
        ),
    ]


def _novoboard_labelled_correct(df: pd.DataFrame, dataset_key: str) -> np.ndarray:
    """Mark NovoBoard rows correct via spectrum-joined annotated sequences."""
    truth_path = _REPO_ROOT / f"{dataset_key}_split_parquet" / "annotated_test.parquet"
    if not truth_path.is_file():
        raise FileNotFoundError(truth_path)
    truth = pd.read_parquet(truth_path, columns=["spectrum_id", "sequence"])
    work = df.merge(truth, on="spectrum_id", how="left")
    return sequence_only_correctness_mask(work["sequence"], work["Peptide"])


def _novoboard_peptide_labelled_correct(
    peptide_target: pd.DataFrame, dataset_key: str
) -> np.ndarray:
    """Peptide-level correctness for NovoBoard TDC winners on the labelled test set.

    Each TDC target row retains its originating ``spectrum_id``; correctness is the
    shared sequence-only match against that spectrum's annotated sequence.
    """
    return _novoboard_labelled_correct(peptide_target, dataset_key)


def novoboard_psm_recovery_series(df: pd.DataFrame, dataset_key: str) -> MethodRecovery:
    """Build NovoBoard PSM recovery series against annotated ground truth."""
    return MethodRecovery(
        "NovoBoard",
        _PALETTE[2],
        df["estimated_q_value"].to_numpy(),
        _novoboard_labelled_correct(df, dataset_key),
    )


def plot_recovery_curves(
    series: list[MethodRecovery],
    dataset_key: str,
    output_path: Path,
    *,
    level: Literal["psm", "peptide"] = "psm",
) -> None:
    """Plot correct-identification recovery versus q-value threshold."""
    display = _display_name(dataset_key)
    level_tag = "PSM" if level == "psm" else "peptide"
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
    ax.set_ylabel(f"Correct {level_tag} recovery\n(% of labelled correct {level_tag}s)")
    ax.set_title(f"{display} labelled {level_tag} recovery by q-value threshold")
    ax.legend(loc="best")
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, output_path)
    logger.info("Wrote %s", output_path)


def plot_threshold_barplot(
    series: list[MethodCounts],
    dataset_key: str,
    eval_type: EvalType,
    output_path: Path,
    *,
    level: Literal["psm", "peptide"] = "psm",
    title_suffix: str = "",
) -> None:
    """Bar chart of identifications retained at each q-value threshold."""
    display = _display_name(dataset_key)
    if eval_type == "labelled":
        split_label = "labelled test set"
    elif "external" in title_suffix.lower():
        split_label = "unlabelled external-peptide subset"
    else:
        split_label = "unlabelled set"

    if level == "peptide":
        count_label = "unique peptides"
        ylabel = "Unique peptides"
    else:
        count_label = "PSMs"
        ylabel = "Peptide-spectrum matches"
    title = (
        f"{display}: accepted {count_label} on the {split_label} at q-value thresholds"
    )

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


def _proteome_hit_mask(peptides: pd.Series, haystack: str) -> np.ndarray:
    """True when the normalised peptide key is a proteome substring hit."""
    keys = [normalize_peptide_key(p) for p in peptides]
    return np.asarray(_batch_peptide_substring_hits(keys, haystack), dtype=bool)


def _label_mask_from_winnow_table(table: pd.DataFrame, correct_col: str) -> np.ndarray:
    """Build a boolean correctness / proteome-hit mask for a Winnow table."""
    if correct_col == "correct" and {"sequence", "prediction"}.issubset(table.columns):
        return sequence_only_correctness_mask(table["sequence"], table["prediction"])
    if correct_col not in table.columns:
        raise KeyError(f"Missing label column {correct_col!r}")
    return table[correct_col].astype(bool).to_numpy()


def _append_winnow_summary_rows(
    acceptance_rows: list[dict[str, object]],
    error_rows: list[dict[str, object]],
    *,
    dataset: str,
    panel: str,
    level: str,
    table: pd.DataFrame,
    correct_col: str,
    include_raw: bool = True,
    include_labels: bool = True,
) -> None:
    """Append acceptance and error rows for Winnow NP / DB-cal / DB-raw curves."""
    label_mask = (
        _label_mask_from_winnow_table(table, correct_col) if include_labels else None
    )
    recovery_denom = int(label_mask.sum()) if label_mask is not None else None
    q_ref = table["psm_q_value_db_cal"].to_numpy()
    method_q = {
        PRIMARY_METHOD: table["psm_q_value"].to_numpy(),
        DB_CAL_METHOD: table["psm_q_value_db_cal"].to_numpy(),
    }
    if include_raw and "psm_q_value_db_raw" in table.columns:
        method_q[DB_RAW_METHOD] = table["psm_q_value_db_raw"].to_numpy()

    for method, q_value in method_q.items():
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
) -> None:
    """Append acceptance and error rows for a non-Winnow method (no DB q-ref)."""
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
            q_ref=None,
        )
    )


def novoboard_peptide_q_values(
    target_df: pd.DataFrame,
    decoy_df: pd.DataFrame,
    *,
    target_peptide_keys: set[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return peptide-level NovoBoard q-values and the filtered target table."""
    table = _novoboard_peptide_tdc_table(target_df, decoy_df)
    target = table[table["is_target"]].copy()
    if target_peptide_keys is not None:
        target = target[target["_peptide_key"].isin(target_peptide_keys)]
    conf = target["ALC (%)"].to_numpy()
    q = _monotonize_q_by_confidence(conf, target["estimated_q_value"].to_numpy())
    return q, target


def _comparators_for_panel(panel: str, level: str) -> list[str]:
    """Comparator method labels present in a given summary panel."""
    if panel == "external" and level == "peptide":
        return ["NovoBoard", "Glissade"]
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


def process_dataset(
    cfg: DatasetConfig, output_dir: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Generate comparison plots and summary rows for one dataset.

    Returns:
        ``(acceptance_rows, error_rows)`` long-form metric dicts for CSV export.
    """
    out = output_dir / cfg.key
    out.mkdir(parents=True, exist_ok=True)
    residue_masses = _load_residue_masses()
    acceptance_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []

    winnow_unlabelled = load_winnow(cfg.winnow_unlabelled, cfg.fasta, "unlabelled")
    winnow_test = load_winnow(cfg.winnow_test, cfg.fasta, "labelled")
    novoboard_unlabelled = load_novoboard(
        cfg.novoboard_dir, "unlabelled", cfg.novoboard_decoy_rate
    )
    novoboard_test = load_novoboard(cfg.novoboard_dir, "test", cfg.novoboard_decoy_rate)
    nb_u_target, nb_u_decoy = load_novoboard_target_decoy(
        cfg.novoboard_dir, "unlabelled", cfg.novoboard_decoy_rate
    )
    nb_t_target, nb_t_decoy = load_novoboard_target_decoy(
        cfg.novoboard_dir, "test", cfg.novoboard_decoy_rate
    )
    glissade_peptides = load_glissade(cfg.glissade_dir)
    reference_haystack = load_proteome_haystack(cfg.fasta)

    # ── PSM level (Winnow + NovoBoard only) ─────────────────────────────
    winnow_u_psm_table = _prepare_winnow_psm_table(
        winnow_unlabelled, "proteome_hit", residue_masses
    )
    winnow_t_psm_table = _prepare_winnow_psm_table(
        winnow_test, "correct", residue_masses
    )
    winnow_u_psm = _winnow_curves_from_table(winnow_u_psm_table)
    winnow_t_psm = _winnow_curves_from_table(winnow_t_psm_table)
    nb_u_psm = novoboard_psm_curve(novoboard_unlabelled)
    nb_t_psm = novoboard_psm_curve(novoboard_test)

    psm_unlabelled = winnow_u_psm + [nb_u_psm]
    psm_test = winnow_t_psm + [nb_t_psm]

    plot_qvalue_by_rank(
        psm_unlabelled,
        cfg.key,
        "unlabelled",
        out / f"psm_qvalue_by_rank_unlabelled_{cfg.key}",
        level="psm",
    )
    plot_threshold_barplot(
        _bar_series_from_curves(psm_unlabelled),
        cfg.key,
        "unlabelled",
        out / f"psm_counts_unlabelled_{cfg.key}",
        level="psm",
    )
    plot_qvalue_by_rank(
        psm_test,
        cfg.key,
        "labelled",
        out / f"psm_qvalue_by_rank_test_{cfg.key}",
        level="psm",
    )
    plot_threshold_barplot(
        _bar_series_from_curves(psm_test),
        cfg.key,
        "labelled",
        out / f"psm_counts_test_{cfg.key}",
        level="psm",
    )
    plot_recovery_curves(
        _winnow_psm_recovery_series(winnow_test, "correct", residue_masses)
        + [novoboard_psm_recovery_series(novoboard_test, cfg.key)],
        cfg.key,
        out / f"psm_recovery_test_{cfg.key}",
        level="psm",
    )

    logger.info("Collecting PSM summary rows for %s", cfg.key)
    _append_winnow_summary_rows(
        acceptance_rows,
        error_rows,
        dataset=cfg.key,
        panel="labelled_test",
        level="psm",
        table=winnow_t_psm_table,
        correct_col="correct",
    )
    nb_t_correct = _novoboard_labelled_correct(novoboard_test, cfg.key)
    _append_method_summary_rows(
        acceptance_rows,
        error_rows,
        dataset=cfg.key,
        panel="labelled_test",
        level="psm",
        method="NovoBoard",
        q_value=novoboard_test["estimated_q_value"].to_numpy(),
        label_mask=nb_t_correct,
        recovery_denom=int(nb_t_correct.sum()),
    )

    _append_winnow_summary_rows(
        acceptance_rows,
        error_rows,
        dataset=cfg.key,
        panel="unlabelled",
        level="psm",
        table=winnow_u_psm_table,
        correct_col="proteome_hit",
    )
    nb_u_hit = _proteome_hit_mask(novoboard_unlabelled["Peptide"], reference_haystack)
    _append_method_summary_rows(
        acceptance_rows,
        error_rows,
        dataset=cfg.key,
        panel="unlabelled",
        level="psm",
        method="NovoBoard",
        q_value=novoboard_unlabelled["estimated_q_value"].to_numpy(),
        label_mask=nb_u_hit,
        recovery_denom=int(nb_u_hit.sum()),
    )

    # ── Peptide level (Winnow + NovoBoard + Glissade) ───────────────────
    logger.info("Preparing peptide-level tables for %s", cfg.key)
    winnow_u_pep_table = _prepare_winnow_peptide_table(
        winnow_unlabelled, "proteome_hit", residue_masses
    )
    logger.info(
        "Winnow unlabelled peptide table ready (%d rows)", len(winnow_u_pep_table)
    )
    winnow_t_pep_table = _prepare_winnow_peptide_table(
        winnow_test, "correct", residue_masses
    )
    winnow_u_pep = _winnow_curves_from_table(winnow_u_pep_table, monotonize=False)
    winnow_t_pep = _winnow_curves_from_table(winnow_t_pep_table, monotonize=False)
    logger.info("Running NovoBoard peptide TDC for %s unlabelled", cfg.key)
    # Compute NovoBoard peptide TDC once per split; reuse for plots and summaries.
    nb_u_pep_q, nb_u_pep_target = novoboard_peptide_q_values(nb_u_target, nb_u_decoy)
    logger.info(
        "NovoBoard unlabelled peptide TDC ready (%d targets)", len(nb_u_pep_target)
    )
    nb_t_pep_q, nb_t_pep_target = novoboard_peptide_q_values(nb_t_target, nb_t_decoy)
    nb_u_pep = MethodCurve(
        "NovoBoard",
        _PALETTE[2],
        nb_u_pep_target["ALC (%)"].to_numpy(),
        nb_u_pep_q,
    )
    nb_t_pep = MethodCurve(
        "NovoBoard",
        _PALETTE[2],
        nb_t_pep_target["ALC (%)"].to_numpy(),
        nb_t_pep_q,
    )

    peptide_unlabelled = winnow_u_pep + [nb_u_pep]
    plot_qvalue_comparison(
        peptide_unlabelled,
        cfg.key,
        "unlabelled",
        out / f"peptide_qvalue_comparison_unlabelled_{cfg.key}",
        level="peptide",
    )
    plot_threshold_barplot(
        _bar_series_from_curves(peptide_unlabelled),
        cfg.key,
        "unlabelled",
        out / f"peptide_counts_unlabelled_{cfg.key}",
        level="peptide",
    )

    w_external_keys = _external_peptide_keys(
        winnow_unlabelled["prediction"], reference_haystack
    )
    nb_external_keys = _external_peptide_keys(
        nb_u_pep_target["Peptide"], reference_haystack
    )
    w_ext = winnow_unlabelled[
        winnow_unlabelled["prediction"].map(normalize_peptide_key).isin(w_external_keys)
    ].copy()
    w_ext_pep_table = _prepare_winnow_peptide_table(
        w_ext, "proteome_hit", residue_masses, fit_df=winnow_unlabelled
    )
    # External plots omit DB-grounded curves (proteome membership is tautological there),
    # but summaries still report NP vs DB-cal deviation on the external subset.
    w_ext_pep = _non_database_curves(
        _winnow_curves_from_table(w_ext_pep_table, monotonize=False)
    )
    glissade_curve = glissade_peptide_curve(glissade_peptides)
    nb_ext_target = nb_u_pep_target[
        nb_u_pep_target["_peptide_key"].isin(nb_external_keys)
    ].copy()
    nb_ext_q = _monotonize_q_by_confidence(
        nb_ext_target["ALC (%)"].to_numpy(),
        nb_ext_target["estimated_q_value"].to_numpy(),
    )
    nb_ext_pep = MethodCurve(
        "NovoBoard",
        _PALETTE[2],
        nb_ext_target["ALC (%)"].to_numpy(),
        nb_ext_q,
    )
    peptide_external = w_ext_pep + [nb_ext_pep, glissade_curve]
    plot_qvalue_comparison(
        peptide_external,
        cfg.key,
        "unlabelled",
        out / f"peptide_qvalue_comparison_external_{cfg.key}",
        level="peptide",
        title_suffix="\non external peptides only",
    )
    plot_threshold_barplot(
        _bar_series_from_curves(peptide_external),
        cfg.key,
        "unlabelled",
        out / f"peptide_counts_external_{cfg.key}",
        level="peptide",
        title_suffix="\non external peptides only",
    )

    peptide_test = winnow_t_pep + [nb_t_pep]
    plot_qvalue_comparison(
        peptide_test,
        cfg.key,
        "labelled",
        out / f"peptide_qvalue_comparison_test_{cfg.key}",
        level="peptide",
    )
    plot_threshold_barplot(
        _bar_series_from_curves(peptide_test),
        cfg.key,
        "labelled",
        out / f"peptide_counts_test_{cfg.key}",
        level="peptide",
    )

    _append_winnow_summary_rows(
        acceptance_rows,
        error_rows,
        dataset=cfg.key,
        panel="labelled_test",
        level="peptide",
        table=winnow_t_pep_table,
        correct_col="correct",
    )
    nb_t_pep_correct = _novoboard_peptide_labelled_correct(nb_t_pep_target, cfg.key)
    _append_method_summary_rows(
        acceptance_rows,
        error_rows,
        dataset=cfg.key,
        panel="labelled_test",
        level="peptide",
        method="NovoBoard",
        q_value=nb_t_pep_q,
        label_mask=nb_t_pep_correct,
        recovery_denom=int(nb_t_pep_correct.sum()),
    )

    _append_winnow_summary_rows(
        acceptance_rows,
        error_rows,
        dataset=cfg.key,
        panel="unlabelled",
        level="peptide",
        table=winnow_u_pep_table,
        correct_col="proteome_hit",
    )
    # Deduped peptide table is small enough for Aho-Corasick proteome hits.
    nb_u_pep_hit = _proteome_hit_mask(nb_u_pep_target["Peptide"], reference_haystack)
    _append_method_summary_rows(
        acceptance_rows,
        error_rows,
        dataset=cfg.key,
        panel="unlabelled",
        level="peptide",
        method="NovoBoard",
        q_value=nb_u_pep_q,
        label_mask=nb_u_pep_hit,
        recovery_denom=int(nb_u_pep_hit.sum()),
    )

    _append_winnow_summary_rows(
        acceptance_rows,
        error_rows,
        dataset=cfg.key,
        panel="external",
        level="peptide",
        table=w_ext_pep_table,
        correct_col="proteome_hit",
        include_raw=False,
        include_labels=False,
    )
    _append_method_summary_rows(
        acceptance_rows,
        error_rows,
        dataset=cfg.key,
        panel="external",
        level="peptide",
        method="NovoBoard",
        q_value=nb_ext_q,
    )
    _append_method_summary_rows(
        acceptance_rows,
        error_rows,
        dataset=cfg.key,
        panel="external",
        level="peptide",
        method="Glissade",
        q_value=glissade_curve.q_value,
    )

    return acceptance_rows, error_rows


@app.command()
def main(
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for PNG/PDF outputs."),
    ] = DEFAULT_OUTPUT_DIR,
    datasets: Annotated[
        Optional[list[str]],
        typer.Option("--datasets", help="Dataset keys to plot."),
    ] = None,
    novoboard_root: Annotated[
        Path,
        typer.Option("--novoboard-root", help="NovoBoard datasets root."),
    ] = DEFAULT_NOVOBOARD_ROOT,
    glissade_root: Annotated[
        Path,
        typer.Option("--glissade-root", help="Glissade build root."),
    ] = DEFAULT_GLISSADE_ROOT,
    winnow_results: Annotated[
        Path,
        typer.Option("--winnow-results", help="Winnow results directory."),
    ] = DEFAULT_WINNOW_RESULTS,
) -> None:
    """Generate FDR method comparison plots and summary CSVs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dataset_keys = datasets if datasets is not None else list(_DATASET_META.keys())
    configs = build_dataset_configs(winnow_results, novoboard_root, glissade_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    acceptance_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    for key in dataset_keys:
        if key not in configs:
            raise typer.BadParameter(f"Unknown dataset {key!r}")
        logger.info("Processing %s", key)
        acc, err = process_dataset(configs[key], output_dir)
        acceptance_rows.extend(acc)
        error_rows.extend(err)

    acceptance, error_gain = _finalise_method_comparison_tables(
        acceptance_rows, error_rows
    )
    write_summary_tables(acceptance, error_gain, output_dir, "fdr_method_comparison")


if __name__ == "__main__":
    app()
