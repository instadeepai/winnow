"""Evaluate calibrator generalisation by training on one biological validation dataset and testing on all others.

For each project in the data directory, trains a fresh calibrator, evaluates it
in-distribution (held-out 20 %) and out-of-distribution (every other project),
then saves a combined results CSV for downstream plotting.
"""

import logging
import re
from pathlib import Path
from typing import Annotated, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from rich.logging import RichHandler
import typer

from winnow.calibration.calibrator import ProbabilityCalibrator
from winnow.calibration.features import (
    BeamFeatures,
    FragmentMatchFeatures,
    MassErrorDaFeature,
    RetentionTimeFeature,
    TokenScoreFeatures,
)
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.datasets.data_loaders import InstaNovoDatasetLoader

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("winnow.evaluate_generalization")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.addHandler(RichHandler())

# ---------------------------------------------------------------------------
# Constants — loaded from the canonical Winnow YAML configs
# ---------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.2

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "winnow" / "configs"

with open(_CONFIGS_DIR / "residues.yaml") as _f:
    RESIDUE_MASSES: dict[str, float] = yaml.safe_load(_f)["residue_masses"]

with open(_CONFIGS_DIR / "data_loader" / "instanovo.yaml") as _f:
    _instanovo_cfg = yaml.safe_load(_f)
    RESIDUE_REMAPPING: dict[str, str] = _instanovo_cfg.get("residue_remapping", {})
    BEAM_COLUMNS: dict[str, str] | None = _instanovo_cfg.get("beam_columns")

with open(_CONFIGS_DIR / "calibrator.yaml") as _f:
    _calibrator_cfg = yaml.safe_load(_f)
    _KOINA_CFG = _calibrator_cfg["koina"]
    _KOINA_CONSTRAINTS = _KOINA_CFG["constraints"]
    _KOINA_INPUT_CONSTANTS = _KOINA_CFG.get("input_constants") or {
        "collision_energies": 27,
        "fragmentation_types": "HCD",
    }
    _UNSUPPORTED_RESIDUES: list[str] = (
        _KOINA_CONSTRAINTS.get("unsupported_residues") or []
    )
    _MAX_PRECURSOR_CHARGE: int = _KOINA_CONSTRAINTS["max_precursor_charge"]
    _MAX_PEPTIDE_LENGTH: int = _KOINA_CONSTRAINTS["max_peptide_length"]
    _INTENSITY_MODEL: str = _KOINA_CFG["intensity_model"]
    _IRT_MODEL: str = _KOINA_CFG["irt_model"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_project_name(parquet_path: Path) -> str:
    """Extract project name from a filename like ``dataset-helaqc-annotated-0000-0001.parquet``."""
    match = re.match(r"dataset-(.+?)-annotated", parquet_path.stem)
    if match:
        return match.group(1)
    return parquet_path.stem


def _find_predictions_path(predictions_dir: Path, project: str) -> Optional[Path]:
    """Locate the CSV predictions file for *project* inside *predictions_dir*."""
    candidates = sorted(predictions_dir.glob(f"dataset-{project}-annotated*.csv"))
    if candidates:
        return candidates[0]
    return None


_IRT_TRAIN_FRACTION_OVERRIDES: Dict[str, float] = {
    "herceptin": 0.15,
}

# Mirrors Makefile train-extra-small-mass-error-da / EXTRA_SMALL_* overrides.
_EXTRA_SMALL_FRAGMENT_EXCLUDE = [
    "spectral_angle",
    "xcorr",
    "complementary_ion_count",
    "max_ion_gap",
]
_EXTRA_SMALL_BEAM_EXCLUDE = ["edit_distance"]


def initialise_calibrator(
    *,
    koina_server_url: Optional[str] = None,
    koina_ssl: bool = True,
    train_project: Optional[str] = None,
) -> ProbabilityCalibrator:
    """Create a fresh calibrator matching train-extra-small-mass-error-da."""
    koina_kwargs: Dict = {}
    if koina_server_url is not None:
        koina_kwargs["koina_server_url"] = koina_server_url
        koina_kwargs["koina_ssl"] = koina_ssl

    irt_train_fraction = _IRT_TRAIN_FRACTION_OVERRIDES.get(train_project or "", 0.1)

    calibrator = ProbabilityCalibrator(
        hidden_dims=(50, 50),
        dropout=0.3,
        learning_rate=0.0001,
        weight_decay=0.001,
        max_epochs=1000,
        batch_size=1024,
        n_iter_no_change=10,
        tol=0.0001,
        seed=SEED,
        val_early_stopping_max_psms=None,
        val_subsample_seed=None,
    )
    calibrator.add_feature(MassErrorDaFeature(residue_masses=RESIDUE_MASSES))
    calibrator.add_feature(
        FragmentMatchFeatures(
            mz_tolerance_ppm=20,
            learn_from_missing=False,
            intensity_model_name=_INTENSITY_MODEL,
            max_precursor_charge=_MAX_PRECURSOR_CHARGE,
            max_peptide_length=_MAX_PEPTIDE_LENGTH,
            unsupported_residues=_UNSUPPORTED_RESIDUES,
            model_input_constants=_KOINA_INPUT_CONSTANTS,
            excluded_columns=_EXTRA_SMALL_FRAGMENT_EXCLUDE,
            **koina_kwargs,
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
            **koina_kwargs,
        )
    )
    calibrator.add_feature(BeamFeatures(excluded_columns=_EXTRA_SMALL_BEAM_EXCLUDE))
    calibrator.add_feature(TokenScoreFeatures())
    return calibrator


def load_dataset(data_path: Path, predictions_path: Path) -> CalibrationDataset:
    """Load a single annotated dataset."""
    logger.info("Loading dataset from %s and %s", data_path, predictions_path)
    loader = InstaNovoDatasetLoader(
        residue_masses=RESIDUE_MASSES,
        residue_remapping=RESIDUE_REMAPPING,
        beam_columns=BEAM_COLUMNS,
    )
    return loader.load(data_path=data_path, predictions_path=predictions_path)


_MOD_RE = re.compile(r"\[UNIMOD:\d+\]")


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

    def _subset(idx: np.ndarray) -> CalibrationDataset:
        m = meta.iloc[idx].reset_index(drop=True)
        preds = (
            [dataset.predictions[i] for i in idx.tolist()]
            if dataset.predictions is not None
            else None
        )
        return CalibrationDataset(metadata=m, predictions=preds)

    return _subset(train_idx), _subset(test_idx)


def evaluate_model(
    model: ProbabilityCalibrator,
    test_dataset: CalibrationDataset,
    train_project: str,
    test_project: str,
    evaluation_type: str,
) -> pd.DataFrame:
    """Run prediction and tag the results."""
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
_DEFAULT_MODEL_OUTPUT_DIR = Path("models/generalisation")
_DEFAULT_RESULTS_OUTPUT_DIR = Path("results/generalisation")

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def main(
    data_dir: Annotated[
        Path, typer.Option(help="Directory containing annotated parquet files.")
    ],
    predictions_dir: Annotated[
        Path, typer.Option(help="Directory containing prediction CSV files.")
    ],
    model_output_dir: Annotated[
        Path, typer.Option(help="Directory to save trained models.")
    ] = _DEFAULT_MODEL_OUTPUT_DIR,
    results_output_dir: Annotated[
        Path, typer.Option(help="Directory to save evaluation results.")
    ] = _DEFAULT_RESULTS_OUTPUT_DIR,
    koina_server_url: Annotated[
        Optional[str], typer.Option(help="Koina server URL override.")
    ] = None,
    koina_ssl: Annotated[bool, typer.Option(help="Use SSL for Koina server.")] = True,
) -> None:
    """Evaluate calibrator generalisation across biological validation datasets."""
    model_output_dir.mkdir(parents=True, exist_ok=True)
    results_output_dir.mkdir(parents=True, exist_ok=True)

    # Discover datasets
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        logger.error("No parquet files found in %s", data_dir)
        raise typer.Exit(1)

    projects: Dict[str, tuple[Path, Path]] = {}
    for pf in parquet_files:
        project = _extract_project_name(pf)
        pred_path = _find_predictions_path(predictions_dir, project)
        if pred_path is None:
            logger.warning(
                "No predictions file found for project %s, skipping.", project
            )
            continue
        projects[project] = (pf, pred_path)

    logger.info("Found %d projects: %s", len(projects), list(projects.keys()))

    # Load all datasets
    datasets: Dict[str, CalibrationDataset] = {}
    for project, (data_path, pred_path) in projects.items():
        datasets[project] = load_dataset(data_path, pred_path)
        logger.info("  %s: %d samples", project, len(datasets[project].metadata))

    # Train-on-each, evaluate-on-all
    all_results: List[pd.DataFrame] = []
    for train_project in projects:
        logger.info("=== Training on %s ===", train_project)

        train_ds, in_dist_test_ds = create_train_test_split(datasets[train_project])
        logger.info(
            "  train: %d, in-dist test: %d",
            len(train_ds.metadata),
            len(in_dist_test_ds.metadata),
        )

        calibrator = initialise_calibrator(
            koina_server_url=koina_server_url,
            koina_ssl=koina_ssl,
            train_project=train_project,
        )
        calibrator.fit(train_ds)

        model_path = model_output_dir / f"trained_on_{train_project}"
        ProbabilityCalibrator.save(calibrator, model_path)

        # In-distribution evaluation
        logger.info(
            "  Evaluating in-distribution on %s (%d samples)",
            train_project,
            len(in_dist_test_ds.metadata),
        )
        all_results.append(
            evaluate_model(
                calibrator,
                in_dist_test_ds,
                train_project,
                train_project,
                "in_distribution",
            )
        )

        # Out-of-distribution evaluation
        for test_project in projects:
            if test_project == train_project:
                continue
            test_ds = datasets[test_project]
            logger.info(
                "  Evaluating out-of-distribution on %s (%d samples)",
                test_project,
                len(test_ds.metadata),
            )
            all_results.append(
                evaluate_model(
                    calibrator,
                    test_ds,
                    train_project,
                    test_project,
                    "out_of_distribution",
                )
            )

    # Combine and save
    combined = pd.concat(all_results, ignore_index=True)

    # Drop large array columns to save space
    array_cols = [c for c in ["mz_array", "intensity_array"] if c in combined.columns]
    if array_cols:
        combined = combined.drop(columns=array_cols)

    results_path = results_output_dir / "calibrator_generalisation_results.csv"
    combined.to_csv(results_path, index=False)
    logger.info("Results saved to %s", results_path)

    # Summary
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


if __name__ == "__main__":
    app()
