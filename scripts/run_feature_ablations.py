"""Feature ablation study for Winnow calibrator.

Trains MLP calibrators on subsets of pre-computed training feature matrices,
computes features from raw spectra for evaluation datasets, and produces
publication-quality plots of calibration, discrimination, and FDR behavior.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import torch
import typer
from rich.logging import RichHandler

from winnow.calibration.calibrator import ProbabilityCalibrator
from winnow.datasets.feature_dataset import FeatureDataset
from winnow.fdr.database_grounded import DatabaseGroundedFDRControl
from winnow.fdr.nonparametric import NonParametricFDRControl

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    logger.addHandler(RichHandler())

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

# ---------------------------------------------------------------------------
# Plot theme and colour assignments — Paul Tol "bright" palette (colour-blind safe)
# ---------------------------------------------------------------------------
_PALETTE = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]

sns.set_theme(style="white", palette=_PALETTE, context="paper", font_scale=1.5)

DATASET_DISPLAY_NAMES: dict[str, str] = {
    "gluc": "HeLa degradome",
    "helaqc": "HeLa single shot",
    "herceptin": "Herceptin",
    "immuno": "Immunopeptidomics-1",
    "sbrodae": "Scalindua brodae",
    "snakevenoms": "Snake venomics",
    "tplantibodies": "Therapeutic nanobodies",
    "woundfluids": "Wound exudates",
    "PXD014877": "C. elegans",
    "PXD023064": "Immunopeptidomics-2",
    "PXD009935": "Immunopeptidomics-3",
    "Astral": "Astral E. coli",
}

# ---------------------------------------------------------------------------
# Feature group definitions
# ---------------------------------------------------------------------------
BEAM_COLUMNS = ["margin", "median_margin", "entropy", "z-score", "edit_distance"]
TOKEN_COLUMNS = ["min_token_probability", "std_token_probability"]
FRAGMENT_MATCH_COLUMNS = [
    "ion_matches",
    "ion_match_intensity",
    "complementary_ion_count",
    "max_ion_gap",
    "spectral_angle",
    "xcorr",
]
RETENTION_TIME_COLUMNS = ["irt_error"]
MASS_ERROR_COLUMNS = ["mass_error_ppm"]

ALL_FEATURE_COLUMNS = (
    ["confidence"]
    + MASS_ERROR_COLUMNS
    + FRAGMENT_MATCH_COLUMNS
    + RETENTION_TIME_COLUMNS
    + BEAM_COLUMNS
    + TOKEN_COLUMNS
)

ABLATION_CONFIGS: dict[str, list[str]] = {
    "Confidence only": ["confidence"],
    "Beam + Token": ["confidence"] + BEAM_COLUMNS + TOKEN_COLUMNS,
    "Prosit": (
        ["confidence"]
        + FRAGMENT_MATCH_COLUMNS
        + RETENTION_TIME_COLUMNS
        + MASS_ERROR_COLUMNS
    ),
    "Full model": ALL_FEATURE_COLUMNS,
}

ABLATION_COLORS: dict[str, str] = {
    name: _PALETTE[i] for i, name in enumerate(ABLATION_CONFIGS)
}
ORIGINAL_COLOR = _PALETTE[len(ABLATION_CONFIGS)]

# Training hyperparameters (matching train_general_model Makefile target)
TRAIN_HYPERPARAMS = {
    "hidden_dims": [128, 64],
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "batch_size": 4096,
    "max_epochs": 200,
    "n_iter_no_change": 10,
    "tol": 1e-4,
}

EVAL_DATASETS = {
    "PXD014877": {
        "label": "C. elegans",
        "spectra": "held_out_projects/lcfm/PXD014877/",
        "predictions": "held_out_projects/lcfm/PXD014877_predictions/PXD014877.csv",
        "koina_mode": "columns",
    },
    "PXD023064": {
        "label": "Immunopeptidomics-2",
        "spectra": "held_out_projects/lcfm/PXD023064/",
        "predictions": "held_out_projects/lcfm/PXD023064_predictions/PXD023064.csv",
        "koina_mode": "columns",
    },
    "helaqc": {
        "label": "HeLa single shot",
        "spectra": "held_out_projects/biological_validation/annotated/dataset-helaqc-annotated-0000-0001.parquet",
        "predictions": "held_out_projects/biological_validation/annotated_predictions/dataset-helaqc-annotated-0000-0001.csv",
        "koina_mode": "constants",
    },
}

# Residue masses for DatabaseGroundedFDRControl (loaded from config at runtime)
_RESIDUE_MASSES: dict[str, float] | None = None


def _get_residue_masses() -> dict[str, float]:
    """Load residue masses from the winnow residues config."""
    global _RESIDUE_MASSES
    if _RESIDUE_MASSES is None:
        import yaml

        config_path = (
            Path(__file__).resolve().parent.parent
            / "winnow"
            / "configs"
            / "residues.yaml"
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        _RESIDUE_MASSES = cfg["residue_masses"]
    return _RESIDUE_MASSES


# ---------------------------------------------------------------------------
# Eval feature computation
# ---------------------------------------------------------------------------
def _compute_eval_features_for_dataset(
    name: str,
    spectra_path: str,
    predictions_path: str,
    cache_dir: Path,
    koina_url: str,
    koina_ssl: bool,
    koina_mode: str = "columns",
) -> Path:
    """Compute the full feature matrix for an eval dataset and cache as Parquet.

    Args:
        koina_mode: ``"columns"`` to read collision_energy / frag_type from
            per-row metadata columns, or ``"constants"`` to use fixed values
            (CE=27, HCD).
    """
    cache_path = cache_dir / f"{name}.parquet"
    if cache_path.exists():
        logger.info("Using cached eval features for %s at %s", name, cache_path)
        return cache_path

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from winnow.utils.config_path import get_primary_config_dir

    primary_config_dir = get_primary_config_dir(None)

    logger.info("Computing features for eval dataset %s ...", name)

    if koina_mode == "columns":
        koina_overrides = [
            "+koina.input_columns.collision_energies=collision_energy",
            "+koina.input_columns.fragmentation_types=frag_type",
            "+calibrator.features.fragment_match_features.model_input_columns.collision_energies=collision_energy",
            "+calibrator.features.fragment_match_features.model_input_columns.fragmentation_types=frag_type",
        ]
    else:
        koina_overrides = [
            "+koina.input_constants.collision_energies=27",
            "+koina.input_constants.fragmentation_types=HCD",
            "+calibrator.features.fragment_match_features.model_input_constants.collision_energies=27",
            "+calibrator.features.fragment_match_features.model_input_constants.fragmentation_types=HCD",
        ]

    with initialize_config_dir(
        config_dir=str(primary_config_dir),
        version_base="1.3",
        job_name=f"winnow_ablation_features_{name}",
    ):
        cfg = compose(
            config_name="compute_features",
            overrides=[
                f"dataset.spectrum_path_or_directory={spectra_path}",
                f"dataset.predictions_path={predictions_path}",
                f"koina.server_url={koina_url}",
                f"koina.ssl={koina_ssl}",
                *koina_overrides,
                "labelled=true",
                "filter_empty_predictions=true",
            ],
        )

    data_loader = instantiate(cfg.data_loader)
    calibrator = instantiate(cfg.calibrator)

    from winnow.scripts.main import (
        _compute_features_batched_metadata,
    )

    spectrum_path = Path(spectra_path)
    preds_path = cfg.dataset.get("predictions_path", predictions_path)

    all_metadata = _compute_features_batched_metadata(
        spectrum_path,
        preds_path,
        data_loader,
        calibrator,
        labelled=True,
        filter_empty=True,
    )

    combined_metadata = pd.concat(all_metadata, ignore_index=True)
    logger.info(
        "  %s: %d spectra after feature computation", name, len(combined_metadata)
    )

    # Write the training matrix parquet with all feature columns + correct + extra cols for FDR
    feature_columns = ["confidence"] + calibrator.columns
    keep_cols = list(feature_columns)
    if "correct" in combined_metadata.columns:
        keep_cols.append("correct")
    if "sequence" in combined_metadata.columns:
        keep_cols.append("sequence")
    if "prediction" in combined_metadata.columns:
        keep_cols.append("prediction")
    if "precursor_mz" in combined_metadata.columns:
        keep_cols.append("precursor_mz")
    if "precursor_charge" in combined_metadata.columns:
        keep_cols.append("precursor_charge")

    # Deduplicate while preserving order
    seen = set()
    unique_cols = []
    for c in keep_cols:
        if c not in seen and c in combined_metadata.columns:
            seen.add(c)
            unique_cols.append(c)

    training_df = pl.from_pandas(combined_metadata[unique_cols])
    cache_dir.mkdir(parents=True, exist_ok=True)
    training_df.write_parquet(cache_path)
    logger.info(
        "  Cached eval features to %s (%d rows, %d cols)",
        cache_path,
        len(training_df),
        len(training_df.columns),
    )
    return cache_path


def compute_all_eval_features(
    output_dir: Path,
    koina_url: str,
    koina_ssl: bool,
    astral_spectra: str | None,
    astral_predictions: str | None,
    skip_feature_compute: bool,
) -> dict[str, Path]:
    """Compute (or locate cached) eval feature Parquets for all datasets."""
    cache_dir = output_dir / "eval_feature_cache"
    result: dict[str, Path] = {}

    for name, info in EVAL_DATASETS.items():
        if skip_feature_compute:
            cache_path = cache_dir / f"{name}.parquet"
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"--skip-feature-compute set but cache not found: {cache_path}"
                )
            result[name] = cache_path
        else:
            result[name] = _compute_eval_features_for_dataset(
                name,
                info["spectra"],
                info["predictions"],
                cache_dir,
                koina_url,
                koina_ssl,
                koina_mode=info.get("koina_mode", "columns"),
            )

    if astral_spectra and astral_predictions:
        name = "Astral"
        if skip_feature_compute:
            cache_path = cache_dir / f"{name}.parquet"
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"--skip-feature-compute set but cache not found: {cache_path}"
                )
            result[name] = cache_path
        else:
            result[name] = _compute_eval_features_for_dataset(
                name,
                astral_spectra,
                astral_predictions,
                cache_dir,
                koina_url,
                koina_ssl,
            )

    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _load_parquet_as_polars(path: str | Path) -> pl.DataFrame:
    """Load a Parquet file or directory of Parquets into a single Polars DataFrame."""
    path = Path(path)
    if path.is_dir():
        parquet_files = sorted(path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files in {path}")
        return pl.concat([pl.read_parquet(f) for f in parquet_files])
    return pl.read_parquet(path)


def _column_slice_to_feature_dataset(
    df: pl.DataFrame, columns: list[str]
) -> FeatureDataset:
    """Select columns from a Polars DataFrame and build a FeatureDataset."""
    if "correct" not in df.columns:
        raise ValueError("Parquet must contain a 'correct' column")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in Parquet: {missing}. Available: {df.columns}"
        )
    features = df.select(columns).to_numpy().astype(np.float32)
    labels = df["correct"].to_numpy().astype(np.float32)
    return FeatureDataset(features=features, labels=labels)


def train_ablation_models(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    output_dir: Path,
    seed: int,
) -> dict[str, ProbabilityCalibrator]:
    """Train one calibrator per ablation config, return dict of fitted calibrators."""
    models: dict[str, ProbabilityCalibrator] = {}

    for config_name, columns in ABLATION_CONFIGS.items():
        logger.info(
            "Training ablation config: %s (%d features)", config_name, len(columns)
        )

        train_ds = _column_slice_to_feature_dataset(train_df, columns)
        val_ds = _column_slice_to_feature_dataset(val_df, columns)

        calibrator = ProbabilityCalibrator(
            seed=seed,
            **TRAIN_HYPERPARAMS,  # type: ignore[arg-type]
        )
        history = calibrator.fit_from_features(train_ds, val_ds)

        model_dir = (
            output_dir
            / "models"
            / config_name.lower().replace(" ", "_").replace("+", "and")
        )
        ProbabilityCalibrator.save(calibrator, model_dir)
        logger.info(
            "  Trained %s: %d epochs, best_epoch=%d",
            config_name,
            history.epochs_trained,
            history.best_epoch,
        )

        models[config_name] = calibrator

    return models


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def _predict_calibrated_scores(
    calibrator: ProbabilityCalibrator,
    features: np.ndarray,
) -> np.ndarray:
    """Run forward pass through a fitted calibrator and return calibrated probabilities."""
    assert calibrator.network is not None
    assert calibrator.feature_mean is not None
    assert calibrator.feature_std is not None

    device = next(calibrator.network.parameters()).device
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    x = (x - calibrator.feature_mean) / calibrator.feature_std

    calibrator.network.eval()
    with torch.no_grad():
        logits = calibrator.network(x)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    return probs


def compute_precision_recall_curve(
    dataset: pd.DataFrame,
    confidence_column: str,
    label_column: str,
    name: str,
) -> pd.DataFrame:
    """Non-standard cumulative PR curve matching the casanovo notebook."""
    original = dataset[[confidence_column, label_column]]
    original = original.sort_values(by=confidence_column, ascending=False)
    cum_correct = np.cumsum(original[label_column].values)
    precision = cum_correct / np.arange(1, len(original) + 1)
    recall = cum_correct / len(original)
    metrics = pd.DataFrame({"precision": precision, "recall": recall}).reset_index(
        drop=True
    )
    metrics["name"] = name
    return metrics


def compute_calibration_curve(
    df: pd.DataFrame,
    pred_col: str,
    label_col: str,
    name: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Fixed-width bin calibration curve matching the casanovo notebook."""
    data = df[[pred_col, label_col]].dropna().copy(deep=True)
    data[pred_col] = data[pred_col].clip(0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_cats = pd.cut(data[pred_col], bins=bins, include_lowest=True)
    bin_cats.name = "bin"
    grouped = (
        data.groupby(bin_cats, observed=True)
        .agg(
            pred_mean=(pred_col, "mean"),
            empirical=(label_col, "mean"),
            count=(label_col, "size"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["count"] > 0]
    grouped["bin_center"] = grouped["bin"].apply(lambda iv: (iv.left + iv.right) / 2)
    grouped["name"] = name
    return grouped[["pred_mean", "empirical", "count", "bin_center", "name"]]


def compute_ece(pred: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(pred, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        avg_conf = pred[mask].mean()
        avg_acc = labels[mask].mean()
        ece += mask.sum() / len(pred) * abs(avg_conf - avg_acc)
    return float(ece)


def compute_tail_ece(
    pred: np.ndarray, labels: np.ndarray, tail_fraction: float = 0.1, n_bins: int = 10
) -> float:
    """ECE in the high-confidence tail (top `tail_fraction` by predicted probability)."""
    threshold_idx = max(1, int(len(pred) * (1 - tail_fraction)))
    sorted_indices = np.argsort(pred)
    tail_mask = np.zeros(len(pred), dtype=bool)
    tail_mask[sorted_indices[threshold_idx:]] = True
    if tail_mask.sum() == 0:
        return 0.0
    return compute_ece(pred[tail_mask], labels[tail_mask], n_bins=n_bins)


def compute_brier_score(pred: np.ndarray, labels: np.ndarray) -> float:
    """Brier score."""
    return float(np.mean((pred - labels) ** 2))


def compute_ids_at_fdr(
    calibrated_scores: np.ndarray,
    labels: np.ndarray,
    fdr_threshold: float,
) -> int:
    """Count PSMs accepted at a given FDR threshold using NonParametricFDRControl."""
    fdr_ctrl = NonParametricFDRControl()
    scores_series = pd.Series(calibrated_scores, name="score")
    fdr_ctrl.fit(dataset=scores_series)
    cutoff = fdr_ctrl.get_confidence_cutoff(threshold=fdr_threshold)
    if np.isnan(cutoff):
        return 0
    return int((calibrated_scores >= cutoff).sum())


@dataclass
class EvalResult:
    """Metrics and curves for a single ablation config evaluated on one dataset."""

    config_name: str
    dataset_name: str
    ece: float
    tail_ece: float
    brier: float
    ids_at_1pct: int
    ids_at_5pct: int
    ids_at_10pct: int
    pr_curve: pd.DataFrame = field(repr=False)
    calibration_curve: pd.DataFrame = field(repr=False)
    calibrated_scores: np.ndarray = field(repr=False)
    labels: np.ndarray = field(repr=False)
    raw_confidence: np.ndarray = field(repr=False)
    eval_df: pd.DataFrame = field(repr=False)


def evaluate_single(
    config_name: str,
    calibrator: ProbabilityCalibrator,
    columns: list[str],
    eval_df: pl.DataFrame,
    dataset_name: str,
) -> EvalResult:
    """Evaluate a single ablation config on a single eval dataset."""
    features = eval_df.select(columns).to_numpy().astype(np.float32)
    labels = eval_df["correct"].to_numpy().astype(np.float32)
    raw_confidence = eval_df["confidence"].to_numpy().astype(np.float64)

    calibrated = _predict_calibrated_scores(calibrator, features)

    # Build a pandas DataFrame for PR / calibration / FDR computations
    meta = pd.DataFrame(
        {
            "confidence": raw_confidence,
            "calibrated_confidence": calibrated,
            "correct": labels,
        }
    )

    # Carry over sequence and prediction for database-grounded FDR if available
    if "sequence" in eval_df.columns:
        meta["sequence"] = eval_df["sequence"].to_pandas()
    if "prediction" in eval_df.columns:
        meta["prediction"] = eval_df["prediction"].to_pandas()

    pr = compute_precision_recall_curve(
        meta, "calibrated_confidence", "correct", config_name
    )

    cal = compute_calibration_curve(
        meta, "calibrated_confidence", "correct", config_name
    )

    ece = compute_ece(calibrated, labels)
    tail_ece = compute_tail_ece(calibrated, labels)
    brier = compute_brier_score(calibrated, labels)

    ids_1 = compute_ids_at_fdr(calibrated, labels, 0.01)
    ids_5 = compute_ids_at_fdr(calibrated, labels, 0.05)
    ids_10 = compute_ids_at_fdr(calibrated, labels, 0.10)

    return EvalResult(
        config_name=config_name,
        dataset_name=dataset_name,
        ece=ece,
        tail_ece=tail_ece,
        brier=brier,
        ids_at_1pct=ids_1,
        ids_at_5pct=ids_5,
        ids_at_10pct=ids_10,
        pr_curve=pr,
        calibration_curve=cal,
        calibrated_scores=calibrated,
        labels=labels,
        raw_confidence=raw_confidence,
        eval_df=meta,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _style_axes(ax: plt.Axes) -> None:
    """Apply standard axes formatting: no grid, black spines."""
    ax.set_axisbelow(True)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.8)


def _save_fig(fig: plt.Figure, base_path: Path, plot_format: str) -> None:
    """Save figure in the requested format(s)."""
    if plot_format in ("pdf", "both"):
        fig.savefig(f"{base_path}.pdf", bbox_inches="tight", dpi=300)
    if plot_format in ("png", "both"):
        fig.savefig(f"{base_path}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_precision_recall(
    results: list[EvalResult],
    dataset_name: str,
    output_dir: Path,
    plot_format: str,
) -> None:
    """PR curve: one line per ablation + raw confidence baseline."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Raw confidence baseline (shared across all ablations -- use first result)
    raw_pr = compute_precision_recall_curve(
        results[0].eval_df, "confidence", "correct", "Original"
    )
    sns.lineplot(
        data=raw_pr,
        x="recall",
        y="precision",
        label="Original",
        color=ORIGINAL_COLOR,
        ax=ax,
    )

    for r in results:
        sns.lineplot(
            data=r.pr_curve,
            x="recall",
            y="precision",
            label=r.config_name,
            color=ABLATION_COLORS[r.config_name],
            ax=ax,
        )

    display = DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title=f"Precision\u2013recall curve \u2014 {display}",
    )
    ax.legend(fontsize=9)
    _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, output_dir / f"pr_curve_{dataset_name}", plot_format)


def plot_calibration(
    results: list[EvalResult],
    dataset_name: str,
    output_dir: Path,
    plot_format: str,
) -> None:
    """Calibration diagram: reliability curves + diagonal."""
    fig, ax = plt.subplots(figsize=(6, 4))

    raw_cal = compute_calibration_curve(
        results[0].eval_df, "confidence", "correct", "Original"
    )
    sns.lineplot(
        data=raw_cal,
        x="pred_mean",
        y="empirical",
        label="Original",
        color=ORIGINAL_COLOR,
        marker="o",
        ax=ax,
    )

    for r in results:
        sns.lineplot(
            data=r.calibration_curve,
            x="pred_mean",
            y="empirical",
            label=r.config_name,
            color=ABLATION_COLORS[r.config_name],
            marker="o",
            ax=ax,
        )

    display = DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Empirical accuracy (database label)",
        title=f"Probability calibration \u2014 {display}",
    )
    ax.legend(fontsize=9)
    _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, output_dir / f"calibration_{dataset_name}", plot_format)


def plot_fdr_vs_confidence(
    results: list[EvalResult],
    dataset_name: str,
    output_dir: Path,
    plot_format: str,
) -> None:
    """PSM FDR vs calibrated confidence: non-parametric vs database-grounded per config."""
    n_configs = len(results)
    fig, axes = plt.subplots(1, n_configs, figsize=(5 * n_configs, 4), squeeze=False)

    residue_masses = _get_residue_masses()

    for i, r in enumerate(results):
        ax = axes[0, i]

        np_fdr = NonParametricFDRControl()
        np_fdr.fit(dataset=r.eval_df["calibrated_confidence"])
        winnow_metrics = np_fdr.add_psm_fdr(
            r.eval_df.copy(), confidence_col="calibrated_confidence"
        )

        has_sequence = (
            "sequence" in r.eval_df.columns and "prediction" in r.eval_df.columns
        )

        if has_sequence:
            dbg_fdr = DatabaseGroundedFDRControl(
                confidence_feature="calibrated_confidence",
                residue_masses=residue_masses,
            )
            try:
                dbg_fdr.fit(dataset=r.eval_df.copy())
                dbg_metrics = dbg_fdr.add_psm_fdr(
                    r.eval_df.copy(), confidence_col="calibrated_confidence"
                )

                sns.lineplot(
                    x=np.asarray(dbg_metrics["calibrated_confidence"], dtype=float),
                    y=np.asarray(dbg_metrics["psm_fdr"], dtype=float),
                    label="Database-grounded",
                    ax=ax,
                    color=_PALETTE[3],
                )
            except Exception as e:
                logger.warning(
                    "Database-grounded FDR failed for %s/%s: %s",
                    r.config_name,
                    dataset_name,
                    e,
                )

        sns.lineplot(
            x=np.asarray(winnow_metrics["calibrated_confidence"], dtype=float),
            y=np.asarray(winnow_metrics["psm_fdr"], dtype=float),
            label="Winnow (non-parametric)",
            ax=ax,
            color=_PALETTE[0],
        )

        ax.set_xlabel("Calibrated confidence")
        ax.set_ylabel("PSM FDR")
        ax.set_title(r.config_name)
        ax.legend(fontsize=8)
        _style_axes(ax)

    display = DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)
    fig.suptitle(f"PSM FDR vs calibrated confidence \u2014 {display}", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, output_dir / f"fdr_vs_confidence_{dataset_name}", plot_format)


def plot_fdr_accepted_psms(
    results: list[EvalResult],
    dataset_name: str,
    output_dir: Path,
    plot_format: str,
) -> None:
    """Number of accepted PSMs vs q-value threshold."""
    fig, ax = plt.subplots(figsize=(6, 4))

    thresholds = np.linspace(0.001, 0.10, 200)

    for r in results:
        np_fdr = NonParametricFDRControl()
        scores_series = pd.Series(r.calibrated_scores, name="score")
        np_fdr.fit(dataset=scores_series)

        meta_with_q = np_fdr.add_psm_q_value(
            pd.DataFrame({"calibrated_confidence": r.calibrated_scores}),
            confidence_col="calibrated_confidence",
        )

        q_values = meta_with_q["psm_q_value"].values
        counts = []
        for t in thresholds:
            counts.append(int((q_values <= t).sum()))

        ax.plot(
            thresholds,
            counts,
            label=r.config_name,
            color=ABLATION_COLORS[r.config_name],
        )

    for fdr_line in [0.01, 0.05, 0.10]:
        ax.axvline(fdr_line, ls="--", color="gray", lw=0.8, alpha=0.7)
        ax.text(
            fdr_line,
            ax.get_ylim()[1] * 0.02,
            f"{fdr_line:.0%}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="gray",
        )

    display = DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)
    ax.set_xlabel("Q-value threshold")
    ax.set_ylabel("Accepted PSMs")
    ax.set_title(f"Accepted PSMs vs q-value \u2014 {display}")
    ax.legend(fontsize=9)
    _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, output_dir / f"fdr_accepted_psms_{dataset_name}", plot_format)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def build_summary_table(all_results: list[EvalResult]) -> pd.DataFrame:
    """Aggregate all EvalResults into a single summary DataFrame."""
    rows = []
    for r in all_results:
        rows.append(
            {
                "config": r.config_name,
                "dataset": r.dataset_name,
                "ECE": round(r.ece, 5),
                "tail_ECE": round(r.tail_ece, 5),
                "Brier": round(r.brier, 5),
                "IDs@1%FDR": r.ids_at_1pct,
                "IDs@5%FDR": r.ids_at_5pct,
                "IDs@10%FDR": r.ids_at_10pct,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------
@app.command()
def main(
    train_features: Annotated[
        Path,
        typer.Option(help="Path to pre-computed training Parquet file or directory."),
    ],
    val_features: Annotated[
        Path,
        typer.Option(help="Path to pre-computed validation Parquet."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory for cached features, models, metrics, and plots."),
    ],
    astral_spectra: Annotated[
        Optional[str],
        typer.Option(help="Optional: path to Astral spectra directory."),
    ] = None,
    astral_predictions: Annotated[
        Optional[str],
        typer.Option(help="Optional: path to Astral predictions CSV."),
    ] = None,
    plot_format: Annotated[
        str,
        typer.Option(help="Plot format: 'pdf', 'png', or 'both'."),
    ] = "both",
    seed: Annotated[
        int,
        typer.Option(help="Random seed."),
    ] = 42,
    koina_url: Annotated[
        str,
        typer.Option(help="Koina server URL for eval feature computation."),
    ] = "koina.wilhelmlab.org:443",
    koina_ssl: Annotated[
        bool,
        typer.Option(help="Use SSL for Koina server."),
    ] = True,
    skip_feature_compute: Annotated[
        bool,
        typer.Option(
            "--skip-feature-compute",
            help="Skip eval feature computation; assume cache exists.",
        ),
    ] = False,
) -> None:
    """Run feature ablation study for the Winnow calibrator."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Compute eval features
    logger.info("Step 1: Computing eval features...")
    eval_parquets = compute_all_eval_features(
        output_dir,
        koina_url,
        koina_ssl,
        astral_spectra,
        astral_predictions,
        skip_feature_compute,
    )

    # 2. Load all Parquets
    logger.info("Step 2: Loading training/validation Parquets...")
    train_df = _load_parquet_as_polars(train_features)
    val_df = _load_parquet_as_polars(val_features)

    eval_dfs: dict[str, pl.DataFrame] = {}
    for name, path in eval_parquets.items():
        eval_dfs[name] = _load_parquet_as_polars(path)
        logger.info("  Loaded eval %s: %d rows", name, len(eval_dfs[name]))

    # 3. Train ablation models
    logger.info("Step 3: Training ablation models...")
    models = train_ablation_models(train_df, val_df, output_dir, seed)

    # 4. Evaluate
    logger.info("Step 4: Evaluating ablation models...")
    all_results: list[EvalResult] = []

    for ds_name, ds_df in eval_dfs.items():
        ds_results: list[EvalResult] = []
        for config_name, columns in ABLATION_CONFIGS.items():
            result = evaluate_single(
                config_name, models[config_name], columns, ds_df, ds_name
            )
            ds_results.append(result)
            all_results.append(result)
            logger.info(
                "  %s / %s: ECE=%.4f, Brier=%.4f, IDs@1%%=%d, IDs@5%%=%d, IDs@10%%=%d",
                ds_name,
                config_name,
                result.ece,
                result.brier,
                result.ids_at_1pct,
                result.ids_at_5pct,
                result.ids_at_10pct,
            )

        # 5. Generate plots per dataset
        logger.info("Step 5: Generating plots for %s...", ds_name)
        plot_precision_recall(ds_results, ds_name, plots_dir, plot_format)
        plot_calibration(ds_results, ds_name, plots_dir, plot_format)
        plot_fdr_vs_confidence(ds_results, ds_name, plots_dir, plot_format)
        plot_fdr_accepted_psms(ds_results, ds_name, plots_dir, plot_format)

    # 6. Summary table
    logger.info("Step 6: Writing summary...")
    summary = build_summary_table(all_results)
    summary.to_csv(output_dir / "ablation_summary.csv", index=False)

    summary_json = summary.to_dict(orient="records")
    with open(output_dir / "ablation_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    logger.info("Summary table:\n%s", summary.to_string(index=False))
    logger.info("Results saved to %s", output_dir)
    logger.info("Feature ablation study complete.")


if __name__ == "__main__":
    app()
