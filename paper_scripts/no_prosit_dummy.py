"""Shared no-Prosit dummy calibrator for paper timing benchmarks.

Used by ``benchmark_scaling.py`` and ``benchmark_runtime.py`` so either can
reuse a checkpoint under the same directory or train and save one if missing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from winnow.calibration.calibrator import ProbabilityCalibrator
from winnow.calibration.features import (
    BeamFeatures,
    MassErrorDaFeature,
    TokenScoreFeatures,
)
from winnow.datasets.calibration_dataset import CalibrationDataset

logger = logging.getLogger(__name__)

_SEED = 42
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _REPO_ROOT / "winnow" / "configs"


def _load_residue_masses() -> dict[str, float]:
    with open(_CONFIGS_DIR / "residues.yaml") as handle:
        return yaml.safe_load(handle)["residue_masses"]


def load_dataset(
    spectrum_path: str,
    predictions_path: str,
    data_loader_name: str,
) -> CalibrationDataset:
    """Load a labelled dataset through the package data loader and filter rows."""
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from winnow.scripts.main import _filter_dataset
    from winnow.utils.config_path import get_primary_config_dir

    primary_config_dir = get_primary_config_dir(None)
    overrides = [f"data_loader={data_loader_name}"]

    with initialize_config_dir(
        config_dir=str(primary_config_dir),
        version_base="1.3",
        job_name="no_prosit_dummy",
    ):
        cfg = compose(config_name="predict", overrides=overrides)

    data_loader = instantiate(cfg.data_loader)
    dataset = data_loader.load(
        data_path=spectrum_path,
        predictions_path=predictions_path,
    )
    return _filter_dataset(dataset)


def build_no_prosit_calibrator() -> ProbabilityCalibrator:
    """Build a small calibrator with only non-Koina features."""
    calibrator = ProbabilityCalibrator(
        hidden_dims=(50, 50),
        dropout=0.3,
        learning_rate=0.0001,
        weight_decay=0.001,
        max_epochs=50,
        batch_size=1024,
        n_iter_no_change=5,
        tol=0.0001,
        seed=_SEED,
        val_early_stopping_max_psms=None,
        val_subsample_seed=None,
    )
    calibrator.add_feature(MassErrorDaFeature(residue_masses=_load_residue_masses()))
    calibrator.add_feature(BeamFeatures())
    calibrator.add_feature(TokenScoreFeatures())
    return calibrator


def checkpoint_exists(model_dir: Path) -> bool:
    """Return True if ``model_dir`` has a loadable calibrator checkpoint."""
    return (model_dir / "config.json").is_file() and (
        model_dir / "model.safetensors"
    ).is_file()


def train_or_load_dummy_calibrator(
    *,
    train_spectrum_path: Path,
    train_predictions_path: Path,
    val_spectrum_path: Path,
    val_predictions_path: Path,
    data_loader_name: str,
    model_output_dir: Path,
    force_retrain: bool = False,
) -> ProbabilityCalibrator:
    """Train a no-Prosit dummy on labelled data, or reuse a checkpoint."""
    model_output_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_exists(model_output_dir) and not force_retrain:
        logger.info("Reusing dummy calibrator at %s", model_output_dir)
        return ProbabilityCalibrator.load(
            pretrained_model_name_or_path=str(model_output_dir)
        )

    logger.info(
        "Training no-Prosit dummy calibrator (Beam + Token Score + Mass Error Da)"
    )
    calibrator = build_no_prosit_calibrator()
    train_ds = load_dataset(
        str(train_spectrum_path), str(train_predictions_path), data_loader_name
    )
    val_ds = load_dataset(
        str(val_spectrum_path), str(val_predictions_path), data_loader_name
    )
    logger.info(
        "  train=%d rows, val=%d rows", len(train_ds.metadata), len(val_ds.metadata)
    )
    calibrator.fit(train_ds, val_ds, progress_bar=True)
    ProbabilityCalibrator.save(calibrator, model_output_dir)
    logger.info("Saved dummy calibrator to %s", model_output_dir)
    return calibrator
