#!/usr/bin/env python
r"""Train and save a calibrator from fixed HPO best-trial hyperparameters.

Uses the same feature matrices and calibrator feature set as ``run_hpo.py``,
without running Optuna. Intended to recover from a completed study whose final
retrain step failed, or to reproduce a saved trial configuration.

Usage
-----
    python scripts/train_hpo_model.py \\
        --train-features-path small_train_shuffled/ \\
        --val-features-path new_val_feature_matrices/PXD010154/ \\
        --config scripts/hpo_best_trial.yaml \\
        --output-dir models/hpo_20260518T230942Z
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Sequence


from scripts.run_hpo import _instantiate_reference_calibrator, _load_config
from winnow.datasets.feature_dataset import FeatureDataset

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "hpo_best_trial.yaml"


def _training_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    hp = cfg["hyperparameters"]
    tr = cfg["training"]
    hidden_dims = tuple(hp["hidden_dims"])
    return {
        "hidden_dims": hidden_dims,
        "dropout": hp["dropout"],
        "learning_rate": hp["learning_rate"],
        "weight_decay": hp["weight_decay"],
        "max_epochs": tr["max_epochs"],
        "batch_size": tr["batch_size"],
        "n_iter_no_change": tr["n_iter_no_change"],
        "tol": tr["tol"],
        "seed": tr["seed"],
        "val_early_stopping_max_psms": tr.get("val_early_stopping_max_psms", 10000),
    }


def train_and_save(
    train_dataset: FeatureDataset,
    val_dataset: FeatureDataset,
    cfg: Dict[str, Any],
    features: Dict,
    output_dir: Path,
    *,
    verbose: bool = False,
) -> None:
    """Fit a calibrator from *cfg* and write it to *output_dir*."""
    from winnow.calibration.calibrator import ProbabilityCalibrator

    params = _training_params(cfg)
    trial_no = cfg.get("trial", "?")
    logger.info(
        "Training calibrator (trial #%s): hidden_dims=%s lr=%s dropout=%s",
        trial_no,
        params["hidden_dims"],
        params["learning_rate"],
        params["dropout"],
    )

    calibrator = ProbabilityCalibrator(features=features, **params)
    calibrator.fit_from_features(
        train_dataset,
        val_dataset,
        progress_bar=verbose,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ProbabilityCalibrator.save(calibrator, output_dir)
    logger.info("Model saved to %s", output_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the train_hpo_model script."""
    p = argparse.ArgumentParser(
        description="Train a calibrator from fixed HPO best-trial hyperparameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--train-features-path",
        required=True,
        type=Path,
        help="Path to training feature Parquet(s).",
    )
    p.add_argument(
        "--val-features-path",
        required=True,
        type=Path,
        help="Path to validation feature Parquet(s).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"YAML with hyperparameters and training settings (default: {_DEFAULT_CONFIG}).",
    )
    p.add_argument(
        "--calibrator-config-dir",
        type=str,
        default=None,
        help="Custom config directory for calibrator.yaml (feature definitions).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save the trained model.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show tqdm bars during training.",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Main entry point for the train_hpo_model script."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not args.verbose:
        logging.getLogger("winnow").setLevel(logging.WARNING)

    cfg = _load_config(args.config)
    ref_calibrator = _instantiate_reference_calibrator(args.calibrator_config_dir)
    feature_columns = ["confidence"] + ref_calibrator.columns

    logger.info("Loading training features from %s ...", args.train_features_path)
    train_dataset = FeatureDataset.from_parquet(
        args.train_features_path, feature_columns
    )
    logger.info(
        "Training set: %d samples, %d features",
        len(train_dataset),
        train_dataset.features.shape[1],
    )

    logger.info("Loading validation features from %s ...", args.val_features_path)
    val_dataset = FeatureDataset.from_parquet(args.val_features_path, feature_columns)
    logger.info(
        "Validation set: %d samples, %d features",
        len(val_dataset),
        val_dataset.features.shape[1],
    )

    train_and_save(
        train_dataset,
        val_dataset,
        cfg,
        features=ref_calibrator.feature_dict,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
