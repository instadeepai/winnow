"""Evaluate calibrator generalisation by training on one source dataset and testing on all others.

Uses a labelled training matrix (HF ``general_model_training_set`` or equivalent)
whose parquet metadata already has a ``source`` column. For each source, trains a
fresh calibrator, evaluates it in-distribution (held-out 20 %) and
out-of-distribution (every other source), then saves a combined results CSV.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Annotated, Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml
from rich.logging import RichHandler
import typer

from winnow.calibration.calibrator import ProbabilityCalibrator, TrainingHistory
from winnow.calibration.features import (
    BeamFeatures,
    FragmentMatchFeatures,
    MassErrorDaFeature,
    RetentionTimeFeature,
    TokenScoreFeatures,
)
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.datasets.data_loaders import InstaNovoDatasetLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
logger.addHandler(RichHandler())

# ---------------------------------------------------------------------------
# Constants — loaded from the canonical Winnow YAML configs
# ---------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.2

_SLIM_RESULTS_FILENAME = "calibrator_generalisation_results_slim.parquet"

_GENERALISATION_SLIM_COLUMNS: tuple[str, ...] = (
    "spectrum_id",
    "prediction_id",
    "confidence",
    "calibrated_confidence",
    "correct",
    "trained_on_dataset",
    "test_dataset",
    "evaluation_type",
)

_MOD_RE = re.compile(r"\[UNIMOD:\d+\]")

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "winnow" / "configs"

with open(_CONFIGS_DIR / "residues.yaml") as _f:
    RESIDUE_MASSES: dict[str, float] = yaml.safe_load(_f)["residue_masses"]

with open(_CONFIGS_DIR / "data_loader" / "instanovo.yaml") as _f:
    _instanovo_cfg = yaml.safe_load(_f)
    RESIDUE_REMAPPING: dict[str, str] = _instanovo_cfg.get("residue_remapping", {})
    BEAM_COLUMNS: dict[str, str] | None = _instanovo_cfg.get("beam_columns")

with open(_CONFIGS_DIR / "calibrator.yaml") as _f:
    _calibrator_cfg = yaml.safe_load(_f)
_CALIBRATOR_TRAIN_CFG: dict = _calibrator_cfg["calibrator"]
with open(_CONFIGS_DIR / "koina.yaml") as _f:
    _KOINA_CFG = yaml.safe_load(_f)["koina"]
_KOINA_CONSTRAINTS = _KOINA_CFG["constraints"]
_KOINA_INPUT_CONSTANTS = _KOINA_CFG.get("input_constants") or {
    "collision_energies": 27,
    "fragmentation_types": "HCD",
}
_UNSUPPORTED_RESIDUES: list[str] = _KOINA_CONSTRAINTS.get("unsupported_residues") or []
_MAX_PRECURSOR_CHARGE: int = _KOINA_CONSTRAINTS["max_precursor_charge"]
_MAX_PEPTIDE_LENGTH: int = _KOINA_CONSTRAINTS["max_peptide_length"]
_INTENSITY_MODEL: str = _KOINA_CFG["intensity_model"]
_IRT_MODEL: str = _KOINA_CFG["irt_model"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IRT_TRAIN_FRACTION_OVERRIDES: Dict[str, float] = {
    "herceptin": 0.15,
}

# Paper general-model feature set: reduced fragment/beam columns.
_REDUCED_FRAGMENT_EXCLUDE = [
    "spectral_angle",
    "xcorr",
    "complementary_ion_count",
    "max_ion_gap",
]
_REDUCED_BEAM_EXCLUDE = ["edit_distance"]


def initialise_calibrator(
    *,
    train_project: Optional[str] = None,
) -> ProbabilityCalibrator:
    """Create a fresh calibrator with the paper general-model feature set."""
    irt_train_fraction = _IRT_TRAIN_FRACTION_OVERRIDES.get(train_project or "", 0.1)

    calibrator = ProbabilityCalibrator(
        hidden_dims=(50, 50),
        dropout=0.3,
        learning_rate=0.0001,
        weight_decay=0.001,
        # Cap matches shipped general / HeLa checkpoints (early stopping usually stops earlier).
        max_epochs=1000,
        batch_size=1024,
        n_iter_no_change=int(_CALIBRATOR_TRAIN_CFG.get("n_iter_no_change", 10)),
        tol=float(_CALIBRATOR_TRAIN_CFG.get("tol", 0.0001)),
        seed=SEED,
        val_early_stopping_max_psms=_CALIBRATOR_TRAIN_CFG.get(
            "val_early_stopping_max_psms"
        ),
        val_subsample_seed=_CALIBRATOR_TRAIN_CFG.get("val_subsample_seed"),
    )
    calibrator.add_feature(MassErrorDaFeature(residue_masses=RESIDUE_MASSES))
    calibrator.add_feature(
        FragmentMatchFeatures(
            mz_tolerance=20,
            mz_tolerance_unit="ppm",
            learn_from_missing=False,
            intensity_model_name=_INTENSITY_MODEL,
            max_precursor_charge=_MAX_PRECURSOR_CHARGE,
            max_peptide_length=_MAX_PEPTIDE_LENGTH,
            unsupported_residues=_UNSUPPORTED_RESIDUES,
            model_input_constants=_KOINA_INPUT_CONSTANTS,
        )
    )
    calibrator.add_feature(
        RetentionTimeFeature(
            train_fraction=irt_train_fraction,
            min_train_points=3,
            learn_from_missing=False,
            irt_model_name=_IRT_MODEL,
            max_peptide_length=_MAX_PEPTIDE_LENGTH,
            unsupported_residues=_UNSUPPORTED_RESIDUES,
        )
    )
    calibrator.add_feature(BeamFeatures())
    calibrator.add_feature(TokenScoreFeatures())
    # Train on a reduced feature subset (exclude some fragment/beam columns).
    training_columns = [
        col
        for col in calibrator.columns
        if col not in _REDUCED_FRAGMENT_EXCLUDE and col not in _REDUCED_BEAM_EXCLUDE
    ]
    calibrator.set_training_feature_columns(training_columns)
    return calibrator


def load_dataset(data_path: Path, predictions_path: Path) -> CalibrationDataset:
    """Load the HF general_model_training_set (or equivalent) dataset."""
    logger.info("Loading dataset from %s and %s", data_path, predictions_path)
    loader = InstaNovoDatasetLoader(
        residue_masses=RESIDUE_MASSES,
        residue_remapping=RESIDUE_REMAPPING,
        beam_columns=BEAM_COLUMNS,
    )
    return loader.load(data_path=data_path, predictions_path=predictions_path)


def subset_dataset(dataset: CalibrationDataset, idx: np.ndarray) -> CalibrationDataset:
    """Return a row subset of *dataset* with aligned beam predictions."""
    meta = dataset.metadata.iloc[idx].reset_index(drop=True)
    preds = (
        [dataset.predictions[i] for i in idx.tolist()]
        if dataset.predictions is not None
        else None
    )
    return CalibrationDataset(metadata=meta, predictions=preds)


def split_dataset_by_source(
    dataset: CalibrationDataset,
) -> Dict[str, CalibrationDataset]:
    """Split a combined dataset into one CalibrationDataset per ``source`` label."""
    if "source" not in dataset.metadata.columns:
        raise ValueError(
            "Expected a 'source' column in the train parquet metadata. "
            "Use HF general_model_training_set or another pre-labelled matrix."
        )

    datasets: Dict[str, CalibrationDataset] = {}
    for source in sorted(dataset.metadata["source"].unique()):
        idx = np.where(dataset.metadata["source"].values == source)[0]
        datasets[source] = subset_dataset(dataset, idx)
    return datasets


def _peptide_key(tokens: object) -> str:
    """Normalise a tokenised peptide to a modification-free, I/L-collapsed key.

    Matches the strategy in ``scripts/split_annotated_raw_parquets.py``:
    strip UNIMOD modifications, normalise I→L.
    """
    if not isinstance(tokens, list):
        return "__MISSING__"
    stripped = [_MOD_RE.sub("", tok).replace("I", "L") for tok in tokens]
    return "".join(stripped)


def create_train_test_split(
    dataset: CalibrationDataset,
) -> tuple[CalibrationDataset, CalibrationDataset]:
    """Split a dataset 80/20 by peptide so no peptide appears in both folds."""
    meta = dataset.metadata
    n = len(meta)
    if n <= 1:
        return dataset, dataset

    pep_keys = meta["sequence"].apply(_peptide_key)
    unique_peptides = pep_keys.unique()

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(unique_peptides))
    n_train = int(len(unique_peptides) * (1 - TEST_SIZE))

    train_peptides = set(unique_peptides[perm[:n_train]])
    train_mask = pep_keys.isin(train_peptides).values

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(~train_mask)[0]

    return subset_dataset(dataset, train_idx), subset_dataset(dataset, test_idx)


def in_distribution_rows_from_ood_cache(
    raw_source: CalibrationDataset,
    ood_cache: CalibrationDataset,
) -> CalibrationDataset:
    """Return peptide-holdout rows from a full-source OOD feature cache.

    The holdout is taken from the loaded source, matching training. Rows dropped
    during featurisation are missing from the cache and are left out.
    """
    _, holdout = create_train_test_split(raw_source)
    holdout_ids = set(holdout.metadata["spectrum_id"].astype(str))
    return subset_dataset_by_spectrum_ids(ood_cache, holdout_ids)


def subset_dataset_by_spectrum_ids(
    dataset: CalibrationDataset,
    spectrum_ids: set[str],
) -> CalibrationDataset:
    """Return rows whose ``spectrum_id`` is in *spectrum_ids* (order preserved)."""
    mask = dataset.metadata["spectrum_id"].astype(str).isin(spectrum_ids).values
    return subset_dataset(dataset, np.where(mask)[0])


def fit_calibrator_on_peptide_split(
    calibrator: ProbabilityCalibrator,
    full_source: CalibrationDataset,
    train_ds: CalibrationDataset,
    val_ds: CalibrationDataset,
) -> tuple[CalibrationDataset, CalibrationDataset, TrainingHistory]:
    """Featurise the full source once, then train with the peptide holdout as val.

    Matches ``winnow train`` single-phase behaviour (features before split) so
    Koina filtering does not empty small validation folds.
    """
    featurized = clone_calibration_dataset(full_source)
    calibrator.compute_features(featurized)

    train_ids = set(train_ds.metadata["spectrum_id"].astype(str))
    val_ids = set(val_ds.metadata["spectrum_id"].astype(str))
    train_featurized = subset_dataset_by_spectrum_ids(featurized, train_ids)
    val_featurized = subset_dataset_by_spectrum_ids(featurized, val_ids)

    train_fd = calibrator.to_feature_dataset(train_featurized)
    val_fd = calibrator.to_feature_dataset(val_featurized)
    val_for_fit = val_fd if len(val_fd) > 0 else None
    if val_for_fit is None:
        logger.warning(
            "Validation fold has no labelled rows after featurisation; "
            "training without early stopping."
        )

    history = calibrator.fit_from_features(
        train_fd,
        val_for_fit,
        progress_bar=True,
    )
    return train_featurized, val_featurized, history


def clone_calibration_dataset(dataset: CalibrationDataset) -> CalibrationDataset:
    """Shallow copy of metadata and predictions for independent predict passes."""
    preds = dataset.predictions
    return CalibrationDataset(
        metadata=dataset.metadata.copy(),
        predictions=list(preds) if preds is not None else None,
    )


def _align_predictions_to_metadata(
    metadata: pd.DataFrame,
    raw_dataset: CalibrationDataset,
) -> Optional[List[Any]]:
    """Return beam predictions aligned to metadata row order by ``spectrum_id``."""
    if raw_dataset.predictions is None:
        return None
    id_to_idx = dict(
        zip(
            raw_dataset.metadata["spectrum_id"].astype(str),
            range(len(raw_dataset.metadata)),
        )
    )
    indices = [id_to_idx[str(sid)] for sid in metadata["spectrum_id"].astype(str)]
    return [raw_dataset.predictions[i] for i in indices]


def _ood_feature_cache_path(cache_dir: Path, source: str) -> Path:
    return cache_dir / f"{source}_ood_featurized.parquet"


def featurize_full_source_for_ood(
    source: str,
    raw_dataset: CalibrationDataset,
    cache_dir: Path | None,
) -> CalibrationDataset:
    """Compute OOD features once per source (full source, RT fit on all rows).

    Out-of-distribution evaluation uses the same featurised rows for every trainer,
    so results are cached in memory and optionally on disk under cache_dir.
    """
    cache_file = _ood_feature_cache_path(cache_dir, source) if cache_dir else None
    if cache_file is not None and cache_file.is_file():
        logger.info("Loading OOD feature cache for %s from %s", source, cache_file)
        meta = pd.read_parquet(cache_file)
        preds = _align_predictions_to_metadata(meta, raw_dataset)
        return CalibrationDataset(metadata=meta, predictions=preds)

    logger.info(
        "Featurising full source %s for OOD cache (%d rows)", source, len(raw_dataset)
    )
    featurized = clone_calibration_dataset(raw_dataset)
    calibrator = initialise_calibrator(train_project=source)
    calibrator.compute_features(featurized)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        featurized.metadata.to_parquet(cache_file, index=False)
        logger.info("Wrote OOD feature cache for %s (%d rows)", source, len(featurized))
    return featurized


def build_ood_feature_caches(
    datasets: Dict[str, CalibrationDataset],
    cache_dir: Path | None,
) -> Dict[str, CalibrationDataset]:
    """Featurise each source once for reuse across all trainers."""
    caches: Dict[str, CalibrationDataset] = {}
    for source in sorted(datasets):
        caches[source] = featurize_full_source_for_ood(
            source, datasets[source], cache_dir
        )
    return caches


def evaluate_model(
    model: ProbabilityCalibrator,
    test_dataset: CalibrationDataset,
    train_project: str,
    test_project: str,
    evaluation_type: str,
    *,
    features_precomputed: bool = False,
) -> pd.DataFrame:
    """Run prediction and tag the results."""
    if not features_precomputed:
        model.compute_features(test_dataset)
    model.predict(test_dataset)

    results = test_dataset.metadata.copy()
    results["trained_on_dataset"] = train_project
    results["test_dataset"] = test_project
    results["evaluation_type"] = evaluation_type
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_OUTPUT_DIR = Path("paper_results/generalisation/models")
_DEFAULT_RESULTS_OUTPUT_DIR = Path("paper_results/generalisation")
_DEFAULT_TRAIN_PARQUET = Path(
    "paper_data/winnow-ms-datasets/general_model_training_set/train.parquet"
)
_DEFAULT_TRAIN_PREDS = Path(
    "paper_data/winnow-ms-datasets/general_model_training_set/train_preds.csv"
)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def _resolve_disk_cache_dir(
    results_output_dir: Path,
    feature_cache_dir: Path | None,
    no_disk_feature_cache: bool,
) -> Path | None:
    if no_disk_feature_cache:
        return None
    if feature_cache_dir is not None:
        return feature_cache_dir
    return results_output_dir / "ood_feature_cache"


def _load_source_datasets(
    train_parquet: Path,
    train_predictions: Path,
) -> Dict[str, CalibrationDataset]:
    if not train_parquet.exists():
        logger.error("Train parquet not found: %s", train_parquet)
        raise typer.Exit(1)
    if not train_predictions.exists():
        logger.error("Train predictions CSV not found: %s", train_predictions)
        raise typer.Exit(1)

    full_dataset = load_dataset(train_parquet, train_predictions)
    if "source" not in full_dataset.metadata.columns:
        logger.error(
            "Train parquet metadata is missing a required 'source' column: %s",
            train_parquet,
        )
        raise typer.Exit(1)

    datasets = split_dataset_by_source(full_dataset)
    logger.info("Found %d source datasets: %s", len(datasets), list(datasets.keys()))
    for source, dataset in datasets.items():
        logger.info("  %s: %d samples", source, len(dataset.metadata))
    return datasets


def _train_and_collect_results(
    datasets: Dict[str, CalibrationDataset],
    ood_feature_caches: Dict[str, CalibrationDataset],
    model_output_dir: Path,
) -> List[pd.DataFrame]:
    all_results: List[pd.DataFrame] = []
    for train_project in datasets:
        logger.info("=== Training on %s ===", train_project)

        train_ds, in_dist_test_ds = create_train_test_split(datasets[train_project])
        logger.info(
            "  train: %d, in-dist test: %d",
            len(train_ds.metadata),
            len(in_dist_test_ds.metadata),
        )

        calibrator = initialise_calibrator(train_project=train_project)
        _, in_dist_test_ds, history = fit_calibrator_on_peptide_split(
            calibrator,
            datasets[train_project],
            train_ds,
            in_dist_test_ds,
        )
        logger.info(
            "  Training finished: %d epochs (best epoch %d); in-dist eval rows %d",
            history.epochs_trained,
            history.best_epoch,
            len(in_dist_test_ds.metadata),
        )

        model_path = model_output_dir / f"trained_on_{train_project}"
        ProbabilityCalibrator.save(calibrator, model_path)

        logger.info(
            "  Evaluating in-distribution on %s (%d samples)",
            train_project,
            len(in_dist_test_ds.metadata),
        )
        in_dist_eval = clone_calibration_dataset(in_dist_test_ds)
        all_results.append(
            evaluate_model(
                calibrator,
                in_dist_eval,
                train_project,
                train_project,
                "in_distribution",
                features_precomputed=True,
            )
        )

        for test_project in datasets:
            if test_project == train_project:
                continue
            test_ds = datasets[test_project]
            logger.info(
                "  Evaluating out-of-distribution on %s (%d samples)",
                test_project,
                len(test_ds.metadata),
            )
            ood_ds = clone_calibration_dataset(ood_feature_caches[test_project])
            all_results.append(
                evaluate_model(
                    calibrator,
                    ood_ds,
                    train_project,
                    test_project,
                    "out_of_distribution",
                    features_precomputed=True,
                )
            )
    return all_results


def _require_ood_caches_on_disk(cache_dir: Path, sources: Iterable[str]) -> None:
    missing = [
        source
        for source in sources
        if not _ood_feature_cache_path(cache_dir, source).is_file()
    ]
    if missing:
        logger.error(
            "Missing OOD feature caches under %s: %s",
            cache_dir,
            ", ".join(sorted(missing)),
        )
        raise typer.Exit(1)


def _repredict_and_collect_results(
    datasets: Dict[str, CalibrationDataset],
    ood_feature_caches: Dict[str, CalibrationDataset],
    model_input_dir: Path,
) -> List[pd.DataFrame]:
    all_results: List[pd.DataFrame] = []
    for train_project in sorted(datasets):
        model_path = model_input_dir / f"trained_on_{train_project}"
        if not model_path.is_dir():
            logger.error("Missing calibrator checkpoint: %s", model_path)
            raise typer.Exit(1)

        logger.info("=== Repredict with model trained on %s ===", train_project)
        calibrator = ProbabilityCalibrator.load(model_path)

        in_dist = in_distribution_rows_from_ood_cache(
            datasets[train_project],
            ood_feature_caches[train_project],
        )
        logger.info(
            "  Evaluating in-distribution on %s (%d samples)",
            train_project,
            len(in_dist.metadata),
        )
        all_results.append(
            evaluate_model(
                calibrator,
                clone_calibration_dataset(in_dist),
                train_project,
                train_project,
                "in_distribution",
                features_precomputed=True,
            )
        )

        for test_project in sorted(datasets):
            if test_project == train_project:
                continue
            logger.info(
                "  Evaluating out-of-distribution on %s (%d samples)",
                test_project,
                len(ood_feature_caches[test_project].metadata),
            )
            ood_ds = clone_calibration_dataset(ood_feature_caches[test_project])
            all_results.append(
                evaluate_model(
                    calibrator,
                    ood_ds,
                    train_project,
                    test_project,
                    "out_of_distribution",
                    features_precomputed=True,
                )
            )
    return all_results


def _persist_generalisation_results(
    combined: pd.DataFrame,
    results_output_dir: Path,
    *,
    write_full_results_csv: bool,
) -> None:
    slim = combined[list(_GENERALISATION_SLIM_COLUMNS)]
    slim_path = results_output_dir / _SLIM_RESULTS_FILENAME
    slim.to_parquet(slim_path, index=False)
    logger.info("Slim results saved to %s (%d rows)", slim_path, len(slim))

    if write_full_results_csv:
        wide = combined
        array_cols = [c for c in ["mz_array", "intensity_array"] if c in wide.columns]
        if array_cols:
            wide = wide.drop(columns=array_cols)
        results_path = results_output_dir / "calibrator_generalisation_results.csv"
        wide.to_csv(results_path, index=False)
        logger.info("Full results saved to %s", results_path)

    logger.info("Evaluation summary:")
    summary = (
        combined.groupby(["trained_on_dataset", "test_dataset", "evaluation_type"])
        .size()
        .reset_index(name="num_samples")
    )
    for _, row in summary.iterrows():
        logger.info(
            "  Trained on %s, tested on %s (%s): %d samples",
            row["trained_on_dataset"],
            row["test_dataset"],
            row["evaluation_type"],
            row["num_samples"],
        )


@app.command()
def main(
    train_parquet: Annotated[
        Path, typer.Option(help="Combined train parquet with a source column.")
    ] = _DEFAULT_TRAIN_PARQUET,
    train_predictions: Annotated[
        Path, typer.Option(help="Combined train predictions CSV.")
    ] = _DEFAULT_TRAIN_PREDS,
    model_output_dir: Annotated[
        Path, typer.Option(help="Directory to save trained models.")
    ] = _DEFAULT_MODEL_OUTPUT_DIR,
    results_output_dir: Annotated[
        Path, typer.Option(help="Directory to save evaluation results.")
    ] = _DEFAULT_RESULTS_OUTPUT_DIR,
    feature_cache_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Directory for per-source OOD feature Parquet caches.",
        ),
    ] = None,
    no_disk_feature_cache: Annotated[
        bool,
        typer.Option(
            "--no-disk-feature-cache",
            help="Keep OOD caches in memory only (no Parquet under results).",
        ),
    ] = False,
    write_full_results_csv: Annotated[
        bool,
        typer.Option(
            "--write-full-results-csv/--no-write-full-results-csv",
            help="Write the wide per-PSM CSV (~6 GB); slim Parquet is always written.",
        ),
    ] = False,
    repredict_only: Annotated[
        bool,
        typer.Option(
            "--repredict-only",
            help="Predict with deposited checkpoints and OOD caches (no training or Koina).",
        ),
    ] = False,
    model_input_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Directory with trained_on_* calibrators (--repredict-only)."
        ),
    ] = None,
) -> None:
    """Evaluate calibrator generalisation across source-labelled training datasets."""
    results_output_dir.mkdir(parents=True, exist_ok=True)
    datasets = _load_source_datasets(train_parquet, train_predictions)

    if repredict_only:
        if model_input_dir is None:
            logger.error("--repredict-only requires --model-input-dir")
            raise typer.Exit(1)
        if feature_cache_dir is None:
            logger.error("--repredict-only requires --feature-cache-dir")
            raise typer.Exit(1)
        if no_disk_feature_cache:
            logger.error("--no-disk-feature-cache cannot be used with --repredict-only")
            raise typer.Exit(1)
        _require_ood_caches_on_disk(feature_cache_dir, datasets)
        logger.info("Loading OOD feature caches from %s", feature_cache_dir)
        ood_feature_caches = build_ood_feature_caches(datasets, feature_cache_dir)
        all_results = _repredict_and_collect_results(
            datasets, ood_feature_caches, model_input_dir
        )
    else:
        model_output_dir.mkdir(parents=True, exist_ok=True)
        disk_cache_dir = _resolve_disk_cache_dir(
            results_output_dir, feature_cache_dir, no_disk_feature_cache
        )
        logger.info("Building OOD feature caches (one featurisation pass per source)")
        ood_feature_caches = build_ood_feature_caches(datasets, disk_cache_dir)
        all_results = _train_and_collect_results(
            datasets, ood_feature_caches, model_output_dir
        )

    combined = pd.concat(all_results, ignore_index=True)
    _persist_generalisation_results(
        combined,
        results_output_dir,
        write_full_results_csv=write_full_results_csv,
    )


if __name__ == "__main__":
    app()
