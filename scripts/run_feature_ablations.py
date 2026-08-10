"""Feature ablation study for Winnow calibrator.

Trains MLP calibrators on subsets of pre-computed training feature matrices,
computes features from raw spectra for evaluation datasets, and produces
publication-quality plots of calibration, discrimination, and FDR behavior.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import torch
import typer
from rich.logging import RichHandler

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.feature_subsets import FEATURE_SUBSETS  # noqa: E402
from scripts.plot_ablation_summary import (  # noqa: E402
    FDR_BIAS_COLUMN_BY_THRESHOLD,
    Q_DEV_COLUMN_BY_THRESHOLD,
    TAIL_ECE_COLUMN_BY_THRESHOLD,
    assign_ablation_colors,
    compute_ece,
    compute_fdr_bias_at_fdr_thresholds,
    compute_pr_auc,
    compute_q_value_deviations,
    compute_tail_ece_at_fdr,
    ordered_ablation_configs,
)

from winnow.calibration.calibrator import ProbabilityCalibrator  # noqa: E402
from winnow.datasets.feature_dataset import FeatureDataset  # noqa: E402
from winnow.fdr.database_grounded import DatabaseGroundedFDRControl  # noqa: E402
from winnow.fdr.nonparametric import NonParametricFDRControl  # noqa: E402

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    logger.addHandler(RichHandler())

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

# ---------------------------------------------------------------------------
# Plot theme — Paul Tol qualitative (colour-blind safe)
# ---------------------------------------------------------------------------
_PALETTE = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]

sns.set_theme(style="white", palette=_PALETTE, context="paper", font_scale=1.5)

DATASET_DISPLAY_NAMES: dict[str, str] = {
    "HCT116": "Human colon",
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
    "PXD004732": "ProteomeTools-1",
    "PXD014877": "$\\it{C.\\;elegans}$",
    "PXD023064": "Immunopeptidomics-2",
    "astral": "Astral $\\it{E.\\;coli}$",
    "01747_C01_P018218_S00_I00_N03_R1": "$\\it{Arabidopsis\\;thaliana}$",
    "Arabidopsis": "$\\it{Arabidopsis\\;thaliana}$",
    "20150708_QE3_UPLC8_DBJ_QC_HELA_39frac_Chymotrypsin": "HeLa chymotrypsin",
    "20151020_QE3_UPLC8_DBJ_SA_A549_Rep2_46": "Human lung",
    "20151020_QE3_UPLC8_DBJ_SA_HCT116_Rep2_46": "Human colon",
    "20170303_QEh1_LC2_FaMa_ChCh_SA_HLApI_JY_R1_exp2": "HLA Class I (JY cells)",
    "20170609_QEh1_LC1_ChCh_FAMA_SA_HLAIIp_JY_all_R1": "HLA Class II (JY cells)",
}

# ---------------------------------------------------------------------------
# Feature group definitions (reduced set: no xcorr, spectral_angle, gap/similarity, edit_distance)
# ---------------------------------------------------------------------------
_EXCLUDED_REDUCED = frozenset(
    {
        "xcorr",
        "spectral_angle",
        "complementary_ion_count",
        "max_ion_gap",
        "edit_distance",
    }
)

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
MASS_ERROR_PPM = "mass_error_ppm"
MASS_ERROR_DA = "mass_error_da"

REDUCED_BEAM_COLUMNS = [c for c in BEAM_COLUMNS if c not in _EXCLUDED_REDUCED]
REDUCED_FRAGMENT_COLUMNS = [
    c for c in FRAGMENT_MATCH_COLUMNS if c not in _EXCLUDED_REDUCED
]

# Default training matrix columns (train_extra_small_matrix.parquet).
REDUCED_TRAIN_COLUMNS: list[str] = FEATURE_SUBSETS["no_fragment_similarity"]["columns"]

# Hydra overrides aligned with Makefile ANALYSIS_REDUCED_FEATURE_OVERRIDES (mass_error_da model).
REDUCED_FEATURE_COMPUTE_OVERRIDES: list[str] = [
    "~calibrator.features.mass_error",
    "+calibrator.features.mass_error_da._target_=winnow.calibration.calibration_features.MassErrorDaFeature",
    "+calibrator.features.mass_error_da.residue_masses=${residue_masses}",
    "+calibrator.features.fragment_match_features.excluded_columns=[spectral_angle,xcorr,complementary_ion_count,max_ion_gap]",
    "+calibrator.features.beam_features.excluded_columns=[edit_distance]",
]


def _reference_model_columns(model_dir: Path | None) -> list[str] | None:
    """Return ``feature_columns`` from a saved calibrator, if present."""
    if model_dir is None:
        return None
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return None
    with open(config_path) as f:
        config = json.load(f)
    cols = config.get("feature_columns")
    return list(cols) if cols else None


def _resolve_mass_error_column(
    df: pl.DataFrame,
    reference_model_dir: Path | None,
) -> str:
    """Pick mass-error column present in *df*, preferring the reference model."""
    ref_cols = _reference_model_columns(reference_model_dir)
    if MASS_ERROR_DA in df.columns:
        return MASS_ERROR_DA
    if MASS_ERROR_PPM in df.columns:
        if ref_cols and MASS_ERROR_DA in ref_cols:
            logger.warning(
                "Reference model uses %s but data has %s; using %s for ablations.",
                MASS_ERROR_DA,
                MASS_ERROR_PPM,
                MASS_ERROR_PPM,
            )
        return MASS_ERROR_PPM
    raise ValueError(
        f"No mass error column in data (tried {MASS_ERROR_DA}, {MASS_ERROR_PPM})"
    )


def _columns_available(df: pl.DataFrame, columns: list[str]) -> list[str]:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {df.columns}")
    return columns


def resolve_all_feature_columns(
    df: pl.DataFrame,
    reference_model_dir: Path | None,
) -> list[str]:
    """Full reduced feature set for the 'All features' ablation config."""
    ref_cols = _reference_model_columns(reference_model_dir)
    if ref_cols:
        cols = ["confidence"]
        for col in ref_cols:
            if (
                col == MASS_ERROR_DA
                and col not in df.columns
                and MASS_ERROR_PPM in df.columns
            ):
                cols.append(MASS_ERROR_PPM)
            elif col in df.columns:
                cols.append(col)
    else:
        cols = [c for c in REDUCED_TRAIN_COLUMNS if c in df.columns]
    return _columns_available(df, cols)


def build_ablation_configs(
    df: pl.DataFrame,
    reference_model_dir: Path | None,
) -> dict[str, list[str]]:
    """Build ablation configs using columns available in *df*."""
    mass_col = _resolve_mass_error_column(df, reference_model_dir)
    all_features = resolve_all_feature_columns(df, reference_model_dir)
    return {
        "Confidence only": ["confidence"],
        "Confidence + mass error": ["confidence", mass_col],
        "Confidence + iRT error": ["confidence", *RETENTION_TIME_COLUMNS],
        "Confidence + token-level": ["confidence", *TOKEN_COLUMNS],
        "Confidence + beam search": ["confidence", *REDUCED_BEAM_COLUMNS],
        "Confidence + fragment matching": ["confidence", *REDUCED_FRAGMENT_COLUMNS],
        "All features": all_features,
    }


ABLATION_CONFIGS: dict[str, list[str]] = {}

ABLATION_COLORS: dict[str, str] = {}


def _dataset_display_name(key: str) -> str:
    """Publication-ready dataset label for plot titles."""
    if key in EVAL_DATASETS:
        return str(EVAL_DATASETS[key]["label"])
    return DATASET_DISPLAY_NAMES.get(key, key)


def _configure_ablation_colors(config_names: Iterable[str]) -> None:
    global ABLATION_COLORS
    ABLATION_COLORS = assign_ablation_colors(
        ordered_ablation_configs(set(config_names))
    )


# Default training hyperparameters (overridden by --hyperparams-from-model).
TRAIN_HYPERPARAMS = {
    "hidden_dims": [128, 64],
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "batch_size": 4096,
    "max_epochs": 200,
    "n_iter_no_change": 10,
    "tol": 1e-4,
}


def train_hyperparams_from_model(model_dir: Path) -> dict[str, object]:
    """Load MLP training hyperparameters from a saved calibrator ``config.json``."""
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"No config.json at {model_dir}")
    with open(config_path) as f:
        config = json.load(f)
    return {
        "hidden_dims": tuple(config["hidden_dims"]),
        "dropout": config["dropout"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "batch_size": config["batch_size"],
        "max_epochs": config["max_epochs"],
        "n_iter_no_change": config["n_iter_no_change"],
        "tol": config["tol"],
    }


EVAL_DATASETS = {
    "HCT116": {
        "label": "Human colon",
        "spectra": "new_eval_data/lcfm/PXD004452/20151020_QE3_UPLC8_DBJ_SA_HCT116_Rep2_46.parquet",
        "predictions": "new_eval_data/lcfm/PXD004452/20151020_QE3_UPLC8_DBJ_SA_HCT116_Rep2_46.csv",
        "koina_mode": "columns",
    },
    "Arabidopsis": {
        "label": "Arabidopsis",
        "spectra": "new_eval_data/lcfm/PXD013868/01747_C01_P018218_S00_I00_N03_R1.parquet",
        "predictions": "new_eval_data/lcfm/PXD013868/01747_C01_P018218_S00_I00_N03_R1.csv",
        "koina_mode": "columns",
    },
    "PXD023064": {
        "label": "Immunopeptidomics-2",
        "spectra": "held_out_projects/lcfm/PXD023064/",
        "predictions": "held_out_projects/lcfm/PXD023064_predictions/PXD023064.csv",
        "koina_mode": "columns",
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
def _feature_compute_overrides(reference_model_dir: Path | None) -> list[str]:
    """Hydra overrides so eval features match a mass_error_da / reduced-feature model."""
    ref_cols = _reference_model_columns(reference_model_dir)
    if ref_cols and MASS_ERROR_DA in ref_cols:
        return list(REDUCED_FEATURE_COMPUTE_OVERRIDES)
    return [
        "+calibrator.features.fragment_match_features.excluded_columns=[spectral_angle,xcorr,complementary_ion_count,max_ion_gap]",
        "+calibrator.features.beam_features.excluded_columns=[edit_distance]",
    ]


def _compute_eval_features_for_dataset(
    name: str,
    spectra_path: str,
    predictions_path: str,
    cache_dir: Path,
    koina_url: str,
    koina_ssl: bool,
    koina_mode: str = "columns",
    feature_overrides: list[str] | None = None,
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
                *(feature_overrides or []),
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
    reference_model_dir: Path | None = None,
) -> dict[str, Path]:
    """Compute (or locate cached) eval feature Parquets for all datasets."""
    cache_dir = output_dir / "eval_feature_cache"
    result: dict[str, Path] = {}
    feature_overrides = _feature_compute_overrides(reference_model_dir)

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
                feature_overrides=feature_overrides,
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
                feature_overrides=feature_overrides,
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


def split_train_val_frames(
    df: pl.DataFrame,
    validation_fraction: float,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Random train/validation split with a fixed index permutation.

    Uses the same scheme as ``winnow.scripts.main._maybe_split_calibration_dataset``:
    shuffle all row indices with *seed*, then assign the last ``validation_fraction``
    fraction to validation. The same split is reused for every ablation config.
    """
    if "correct" not in df.columns:
        raise ValueError("Training Parquet must contain a 'correct' column")
    if not 0 < validation_fraction < 1:
        raise ValueError(
            f"validation_fraction must be in (0, 1), got {validation_fraction}"
        )

    n = len(df)
    n_val = max(1, int(n * validation_fraction))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    train_df = df[perm[: n - n_val].tolist()]
    val_df = df[perm[n - n_val :].tolist()]
    logger.info(
        "Train/val split: %d train, %d val (fraction=%.2f, seed=%d)",
        len(train_df),
        len(val_df),
        validation_fraction,
        seed,
    )
    return train_df, val_df


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


def _config_dir_name(config_name: str) -> str:
    """Derive the on-disk directory name for an ablation config."""
    return config_name.lower().replace(" ", "_").replace("+", "and")


_LEGACY_DIR_NAMES: dict[str, list[str]] = {
    "Confidence only": ["confidence_only"],
    "Confidence + mass error": [
        "confidence_and_mass_error",
        "confidence_and_mass_error_and_rt",
    ],
    "Confidence + iRT error": [
        "confidence_and_irt_error",
        "confidence_and_mass_error_and_rt",
        "confidence_and_fragment_matching",
        "prosit",
    ],
    "Confidence + token-level": [
        "confidence_and_token_level",
        "confidence_and_beam_search",
        "beam_and_token",
    ],
    "Confidence + beam search": ["confidence_and_beam_search", "beam_and_token"],
    "Confidence + fragment matching": [
        "confidence_and_fragment_matching",
        "prosit",
    ],
    "All features": ["all_features", "full_model"],
}


def _resolve_model_dir(output_dir: Path, config_name: str) -> Path:
    """Find the model directory for a config, falling back to legacy names."""
    candidates = _LEGACY_DIR_NAMES.get(config_name, [_config_dir_name(config_name)])
    for candidate in candidates:
        model_dir = output_dir / "models" / candidate
        if model_dir.exists():
            return model_dir
    raise FileNotFoundError(
        f"No saved model found for '{config_name}'. "
        f"Searched: {[str(output_dir / 'models' / c) for c in candidates]}"
    )


def train_ablation_models(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    output_dir: Path,
    seed: int,
    train_hyperparams: dict[str, object] | None = None,
) -> dict[str, ProbabilityCalibrator]:
    """Train one calibrator per ablation config, return dict of fitted calibrators."""
    models: dict[str, ProbabilityCalibrator] = {}
    hp = {**TRAIN_HYPERPARAMS, **(train_hyperparams or {})}

    for config_name, columns in ABLATION_CONFIGS.items():
        logger.info(
            "Training ablation config: %s (%d features)", config_name, len(columns)
        )

        train_ds = _column_slice_to_feature_dataset(train_df, columns)
        val_ds = _column_slice_to_feature_dataset(val_df, columns)

        calibrator = ProbabilityCalibrator(
            seed=seed,
            **hp,  # type: ignore[arg-type]
        )
        history = calibrator.fit_from_features(train_ds, val_ds)

        model_dir = output_dir / "models" / _config_dir_name(config_name)
        ProbabilityCalibrator.save(calibrator, model_dir)
        logger.info(
            "  Trained %s: %d epochs, best_epoch=%d",
            config_name,
            history.epochs_trained,
            history.best_epoch,
        )

        models[config_name] = calibrator

    return models


def load_ablation_models(
    output_dir: Path,
) -> dict[str, ProbabilityCalibrator]:
    """Load pre-trained ablation calibrators from ``{output_dir}/models/``."""
    models: dict[str, ProbabilityCalibrator] = {}

    for config_name in ABLATION_CONFIGS:
        model_dir = _resolve_model_dir(output_dir, config_name)
        calibrator = ProbabilityCalibrator.load(model_dir)
        logger.info("  Loaded %s from %s", config_name, model_dir)
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
    tail_ece_at_5pct: float
    tail_ece_at_10pct: float
    brier: float
    ids_at_1pct: int
    ids_at_5pct: int
    ids_at_10pct: int
    pr_auc: float
    fdr_bias_at_5pct: float
    fdr_bias_at_10pct: float
    q_dev_at_5pct: float
    q_dev_at_10pct: float
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
    fdr_ctrl = NonParametricFDRControl()
    fdr_ctrl.fit(dataset=pd.Series(calibrated, name="score"))
    tail_ece_5 = compute_tail_ece_at_fdr(calibrated, labels, 0.05, fdr_ctrl=fdr_ctrl)
    tail_ece_10 = compute_tail_ece_at_fdr(calibrated, labels, 0.10, fdr_ctrl=fdr_ctrl)
    brier = compute_brier_score(calibrated, labels)

    ids_1 = compute_ids_at_fdr(calibrated, labels, 0.01)
    ids_5 = compute_ids_at_fdr(calibrated, labels, 0.05)
    ids_10 = compute_ids_at_fdr(calibrated, labels, 0.10)

    pr_auc = compute_pr_auc(meta)
    fdr_bias = compute_fdr_bias_at_fdr_thresholds(meta)
    q_dev = compute_q_value_deviations(meta)

    return EvalResult(
        config_name=config_name,
        dataset_name=dataset_name,
        ece=ece,
        tail_ece_at_5pct=tail_ece_5,
        tail_ece_at_10pct=tail_ece_10,
        brier=brier,
        ids_at_1pct=ids_1,
        ids_at_5pct=ids_5,
        ids_at_10pct=ids_10,
        pr_auc=pr_auc,
        fdr_bias_at_5pct=fdr_bias[0.05],
        fdr_bias_at_10pct=fdr_bias[0.10],
        q_dev_at_5pct=q_dev[0.05],
        q_dev_at_10pct=q_dev[0.10],
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


def _lineplot(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    label: str,
    color: str,
    linestyle: str = "-",
    linewidth: float = 0.5,
    marker: str | None = None,
) -> None:
    """Line plot with consistent linewidth (seaborn, no auto legend)."""
    kwargs: dict = {
        "data": data,
        "x": x,
        "y": y,
        "label": label,
        "color": color,
        "linestyle": linestyle,
        "linewidth": linewidth,
        "ax": ax,
        "legend": False,
    }
    if marker is not None:
        kwargs["marker"] = marker
    sns.lineplot(**kwargs)


def _generate_plots_for_dataset(
    ds_results: list[EvalResult],
    ds_name: str,
    plots_dir: Path,
    plot_format: str,
) -> None:
    """Generate all ablation figures for one dataset."""
    plot_precision_recall(ds_results, ds_name, plots_dir, plot_format)
    plot_calibration(ds_results, ds_name, plots_dir, plot_format)
    plot_fdr_vs_confidence(ds_results, ds_name, plots_dir, plot_format)
    plot_fdr_accepted_psms(ds_results, ds_name, plots_dir, plot_format)


def plot_precision_recall(
    results: list[EvalResult],
    dataset_name: str,
    output_dir: Path,
    plot_format: str,
) -> None:
    """PR curve: one line per ablation config."""
    fig, ax = plt.subplots(figsize=(6, 4))

    for r in results:
        _lineplot(
            ax,
            r.pr_curve,
            x="recall",
            y="precision",
            label=r.config_name,
            color=ABLATION_COLORS[r.config_name],
        )

    display = _dataset_display_name(dataset_name)
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title=f"{display} precision-recall by feature set",
    )
    ax.legend(loc="lower left", fontsize=7)
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

    for r in results:
        _lineplot(
            ax,
            r.calibration_curve,
            x="pred_mean",
            y="empirical",
            label=r.config_name,
            color=ABLATION_COLORS[r.config_name],
            marker="o",
        )

    display = _dataset_display_name(dataset_name)
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=0.5)
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Empirical accuracy\n(database label)",
        title=f"{display} probability calibration by feature set",
    )
    ax.legend(loc="lower right", fontsize=7)
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
                    linewidth=0.5,
                    legend=False,
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
            linewidth=0.5,
            legend=False,
        )

        ax.set_xlabel("Calibrated confidence")
        ax.set_ylabel("PSM FDR")
        ax.set_title(r.config_name)
        ax.legend(fontsize=7)
        _style_axes(ax)

    display = _dataset_display_name(dataset_name)
    fig.suptitle(
        f"{display} PSM FDR vs calibrated confidence by feature set", fontsize=12
    )
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
            linewidth=0.5,
        )

    for fdr_line in [0.01, 0.05, 0.10]:
        ax.axvline(fdr_line, ls="--", color="gray", lw=0.5, alpha=0.7)

    ax.relim()
    ax.autoscale_view()
    y_text = ax.get_ylim()[1] * 0.02
    for fdr_line in [0.01, 0.05, 0.10]:
        ax.text(
            fdr_line - 0.002,
            y_text,
            f"{fdr_line:.0%}",
            ha="right",
            va="bottom",
            fontsize=7,
            color="gray",
        )

    display = _dataset_display_name(dataset_name)
    ax.set_xlabel("Non-parametric q-value threshold")
    ax.set_ylabel("Accepted PSMs")
    ax.set_title(f"{display} accepted PSMs at non-parametric q-value threshold")
    ax.legend(loc="upper left", fontsize=7)
    _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, output_dir / f"fdr_accepted_psms_{dataset_name}", plot_format)


# ---------------------------------------------------------------------------
# Saving eval results
# ---------------------------------------------------------------------------
def save_eval_results(all_results: list[EvalResult], output_dir: Path) -> None:
    """Persist per-PSM eval DataFrames so plots can be reproduced without re-inference."""
    results_dir = output_dir / "eval_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for r in all_results:
        safe_config = r.config_name.lower().replace(" ", "_").replace("+", "and")
        path = results_dir / f"{r.dataset_name}_{safe_config}.parquet"
        df = r.eval_df.copy()
        df["config_name"] = r.config_name
        df["dataset_name"] = r.dataset_name
        df.to_parquet(path, index=False)

    logger.info("Saved %d eval result Parquets to %s", len(all_results), results_dir)


def load_eval_results_for_plotting(
    output_dir: Path,
) -> dict[str, list[EvalResult]]:
    """Load saved eval Parquets and rebuild curve data for plotting."""
    results_dir = output_dir / "eval_results"
    if not results_dir.is_dir():
        raise FileNotFoundError(f"No eval_results directory at {results_dir}")

    paths = sorted(results_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No eval result Parquets in {results_dir}")

    grouped: dict[str, list[EvalResult]] = defaultdict(list)
    for path in paths:
        df = pd.read_parquet(path)
        config_name = str(df["config_name"].iloc[0])
        dataset_name = str(df["dataset_name"].iloc[0])
        meta = df.drop(columns=["config_name", "dataset_name"], errors="ignore")
        calibrated = meta["calibrated_confidence"].to_numpy(dtype=np.float64)
        labels = meta["correct"].to_numpy(dtype=np.float32)
        pr = compute_precision_recall_curve(
            meta, "calibrated_confidence", "correct", config_name
        )
        cal = compute_calibration_curve(
            meta, "calibrated_confidence", "correct", config_name
        )
        grouped[dataset_name].append(
            EvalResult(
                config_name=config_name,
                dataset_name=dataset_name,
                ece=0.0,
                tail_ece_at_5pct=float("nan"),
                tail_ece_at_10pct=float("nan"),
                brier=0.0,
                ids_at_1pct=0,
                ids_at_5pct=0,
                ids_at_10pct=0,
                pr_auc=0.0,
                fdr_bias_at_5pct=float("nan"),
                fdr_bias_at_10pct=float("nan"),
                q_dev_at_5pct=float("nan"),
                q_dev_at_10pct=float("nan"),
                pr_curve=pr,
                calibration_curve=cal,
                calibrated_scores=calibrated,
                labels=labels,
                raw_confidence=meta["confidence"].to_numpy(dtype=np.float64),
                eval_df=meta,
            )
        )

    for ds_name in grouped:
        grouped[ds_name].sort(key=lambda r: r.config_name)

    return dict(grouped)


def _run_plots_only(output_dir: Path, plot_format: str) -> None:
    """Regenerate plots from ``{output_dir}/eval_results`` without inference."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    grouped = load_eval_results_for_plotting(output_dir)
    config_names = [r.config_name for results in grouped.values() for r in results]
    _configure_ablation_colors(config_names)

    for ds_name in sorted(grouped):
        ds_results = grouped[ds_name]
        logger.info("Generating plots for %s (%d configs)...", ds_name, len(ds_results))
        _generate_plots_for_dataset(ds_results, ds_name, plots_dir, plot_format)

    logger.info("Plots saved to %s", plots_dir)


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
                TAIL_ECE_COLUMN_BY_THRESHOLD[0.05]: round(r.tail_ece_at_5pct, 5),
                TAIL_ECE_COLUMN_BY_THRESHOLD[0.10]: round(r.tail_ece_at_10pct, 5),
                "Brier": round(r.brier, 5),
                "PR_AUC": round(r.pr_auc, 5),
                FDR_BIAS_COLUMN_BY_THRESHOLD[0.05]: round(r.fdr_bias_at_5pct, 5),
                FDR_BIAS_COLUMN_BY_THRESHOLD[0.10]: round(r.fdr_bias_at_10pct, 5),
                Q_DEV_COLUMN_BY_THRESHOLD[0.05]: round(r.q_dev_at_5pct, 5),
                Q_DEV_COLUMN_BY_THRESHOLD[0.10]: round(r.q_dev_at_10pct, 5),
                "IDs@1%FDR": r.ids_at_1pct,
                "IDs@5%FDR": r.ids_at_5pct,
                "IDs@10%FDR": r.ids_at_10pct,
            }
        )
    return pd.DataFrame(rows)


_DEFAULT_OUTPUT_DIR = Path("analysis/hpo_ablation")


def _validate_training_inputs(
    *,
    skip_training: bool,
    train_features: Path | None,
    val_features: Path | None,
    validation_fraction: float | None,
) -> None:
    if skip_training:
        return
    if train_features is None:
        raise typer.BadParameter(
            "--train-features is required unless --skip-training is set."
        )
    if val_features is None and validation_fraction is None:
        raise typer.BadParameter(
            "Provide --val-features or --validation-fraction when training."
        )
    if val_features is not None and validation_fraction is not None:
        logger.warning(
            "Both --val-features and --validation-fraction set; using --val-features."
        )


def _configure_ablation_configs(
    *,
    skip_training: bool,
    train_features: Path | None,
    eval_dfs: dict[str, pl.DataFrame],
    hyperparams_from_model: Path | None,
) -> None:
    global ABLATION_CONFIGS

    if not skip_training:
        assert train_features is not None
        train_schema_df = _load_parquet_as_polars(train_features)
        ABLATION_CONFIGS = build_ablation_configs(
            train_schema_df, hyperparams_from_model
        )
    else:
        first_eval = next(iter(eval_dfs.values()))
        ABLATION_CONFIGS = build_ablation_configs(first_eval, hyperparams_from_model)
    _configure_ablation_colors(ABLATION_CONFIGS.keys())
    logger.info("Ablation configs: %s", list(ABLATION_CONFIGS.keys()))


def _train_or_load_ablation_models(
    *,
    skip_training: bool,
    train_features: Path | None,
    val_features: Path | None,
    validation_fraction: float | None,
    output_dir: Path,
    seed: int,
    hyperparams_from_model: Path | None,
) -> dict[str, ProbabilityCalibrator]:
    if skip_training:
        logger.info("Step 3: Loading pre-trained ablation models...")
        return load_ablation_models(output_dir)

    logger.info("Step 3: Training ablation models...")
    assert train_features is not None
    full_train_df = _load_parquet_as_polars(train_features)
    if val_features is not None:
        train_df = full_train_df
        val_df = _load_parquet_as_polars(val_features)
    else:
        assert validation_fraction is not None
        train_df, val_df = split_train_val_frames(
            full_train_df, validation_fraction, seed
        )
    train_hp = None
    if hyperparams_from_model is not None:
        train_hp = train_hyperparams_from_model(hyperparams_from_model)
        logger.info(
            "Using training hyperparameters from %s: %s",
            hyperparams_from_model,
            train_hp,
        )
    return train_ablation_models(
        train_df, val_df, output_dir, seed, train_hyperparams=train_hp
    )


def _evaluate_ablations(
    *,
    eval_dfs: dict[str, pl.DataFrame],
    models: dict[str, ProbabilityCalibrator],
    plots_dir: Path,
    plot_format: str,
) -> list[EvalResult]:
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

        logger.info("Step 5: Generating plots for %s...", ds_name)
        _generate_plots_for_dataset(ds_results, ds_name, plots_dir, plot_format)

    return all_results


def _write_ablation_summary(output_dir: Path, all_results: list[EvalResult]) -> None:
    logger.info("Step 7: Writing summary...")
    summary = build_summary_table(all_results)
    summary.to_csv(output_dir / "ablation_summary.csv", index=False)

    summary_json = summary.to_dict(orient="records")
    with open(output_dir / "ablation_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    logger.info("Summary table:\n%s", summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------
@app.command()
def main(
    train_features: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to pre-computed training Parquet file or directory. "
            "Required unless --skip-training is set.",
        ),
    ] = None,
    val_features: Annotated[
        Optional[Path],
        typer.Option(
            help="Pre-computed validation Parquet. Omit if using --validation-fraction.",
        ),
    ] = None,
    validation_fraction: Annotated[
        Optional[float],
        typer.Option(
            "--validation-fraction",
            min=0.0,
            max=1.0,
            help=(
                "Hold out this fraction of --train-features for validation "
                "(same row split for every ablation model). Alternative to --val-features."
            ),
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory for cached features, models, metrics, and plots."),
    ] = _DEFAULT_OUTPUT_DIR,
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
    skip_training: Annotated[
        bool,
        typer.Option(
            "--skip-training",
            help="Load pre-trained ablation models from {output-dir}/models/ "
            "instead of training from scratch.",
        ),
    ] = False,
    hyperparams_from_model: Annotated[
        Optional[Path],
        typer.Option(
            help="Use training hyperparameters from this saved calibrator directory "
            "(e.g. HPO best model). Reads config.json.",
        ),
    ] = None,
    plots_only: Annotated[
        bool,
        typer.Option(
            "--plots-only",
            help="Regenerate plots from {output-dir}/eval_results only "
            "(no feature compute, training, or evaluation).",
        ),
    ] = False,
) -> None:
    """Run feature ablation study for the Winnow calibrator."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if plots_only:
        _run_plots_only(output_dir, plot_format)
        logger.info("Feature ablation plots complete.")
        return

    _validate_training_inputs(
        skip_training=skip_training,
        train_features=train_features,
        val_features=val_features,
        validation_fraction=validation_fraction,
    )

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1: Computing eval features...")
    eval_parquets = compute_all_eval_features(
        output_dir,
        koina_url,
        koina_ssl,
        astral_spectra,
        astral_predictions,
        skip_feature_compute,
        reference_model_dir=hyperparams_from_model,
    )

    logger.info("Step 2: Loading Parquets...")
    eval_dfs: dict[str, pl.DataFrame] = {}
    for name, path in eval_parquets.items():
        eval_dfs[name] = _load_parquet_as_polars(path)
        logger.info("  Loaded eval %s: %d rows", name, len(eval_dfs[name]))

    _configure_ablation_configs(
        skip_training=skip_training,
        train_features=train_features,
        eval_dfs=eval_dfs,
        hyperparams_from_model=hyperparams_from_model,
    )
    models = _train_or_load_ablation_models(
        skip_training=skip_training,
        train_features=train_features,
        val_features=val_features,
        validation_fraction=validation_fraction,
        output_dir=output_dir,
        seed=seed,
        hyperparams_from_model=hyperparams_from_model,
    )
    all_results = _evaluate_ablations(
        eval_dfs=eval_dfs,
        models=models,
        plots_dir=plots_dir,
        plot_format=plot_format,
    )

    logger.info("Step 6: Saving eval results...")
    save_eval_results(all_results, output_dir)
    _write_ablation_summary(output_dir, all_results)
    logger.info("Results saved to %s", output_dir)
    logger.info("Feature ablation study complete.")


if __name__ == "__main__":
    app()
