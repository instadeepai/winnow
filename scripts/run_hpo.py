#!/usr/bin/env python
"""Bayesian hyperparameter optimisation for the Winnow calibrator using Optuna.

Wraps ``ProbabilityCalibrator.fit_from_features`` in an Optuna study that
tunes learning rate, architecture, regularisation and batch size while
minimising the Brier score on a held-out validation set.

The calibrator's feature set is loaded from the same ``calibrator.yaml``
used by ``winnow train``, ensuring that saved models always record their
feature columns correctly.

Usage
-----
    python scripts/run_hpo.py \
        --train-features-path /data/train_features/ \
        --val-features-path /data/val_features/ \
        --n-trials 100 --pruning

See ``scripts/hpo_config.yaml`` for the search-space definition; individual
values can be overridden from the CLI with ``--set key=value``.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import optuna
import torch
import yaml

from winnow.datasets.feature_dataset import FeatureDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _SCRIPT_DIR / "hpo_config.yaml"


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_overrides(cfg: Dict[str, Any], overrides: Sequence[str]) -> None:
    """Apply dotted ``key=value`` overrides to *cfg* in place."""
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override!r}")
        key, raw_value = override.split("=", 1)
        parts = key.split(".")
        target = cfg
        for part in parts[:-1]:
            target = target[part]
        value: Any = yaml.safe_load(raw_value)
        target[parts[-1]] = value


def _instantiate_reference_calibrator(
    config_dir: str | None = None,
):
    """Instantiate a calibrator from calibrator.yaml to obtain the feature set.

    This mirrors how ``winnow train`` builds its calibrator, ensuring that
    the feature definitions (and therefore ``calibrator.columns``) are
    identical.

    Returns:
        A ``ProbabilityCalibrator`` with features registered but no
        trained network.
    """
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from winnow.utils.config_path import get_primary_config_dir

    primary_config_dir = get_primary_config_dir(config_dir)

    with initialize_config_dir(
        config_dir=str(primary_config_dir),
        version_base="1.3",
        job_name="winnow_hpo",
    ):
        cfg = compose(config_name="train")

    return instantiate(cfg.calibrator)


# ---------------------------------------------------------------------------
# Search-space sampling
# ---------------------------------------------------------------------------


def suggest_hyperparameters(trial: optuna.Trial, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Sample hyperparameters from the config-driven search space."""
    ss = cfg["search_space"]
    fixed = cfg["fixed"]

    lr = trial.suggest_float(
        "learning_rate",
        ss["learning_rate"]["low"],
        ss["learning_rate"]["high"],
        log=True,
    )
    wd = trial.suggest_float(
        "weight_decay",
        ss["weight_decay"]["low"],
        ss["weight_decay"]["high"],
        log=True,
    )
    dropout = trial.suggest_float(
        "dropout",
        ss["dropout"]["low"],
        ss["dropout"]["high"],
    )
    batch_size = trial.suggest_categorical(
        "batch_size",
        ss["batch_size"]["choices"],
    )

    n_layers = trial.suggest_int(
        "n_layers",
        ss["n_layers"]["low"],
        ss["n_layers"]["high"],
    )
    unit_choices = sorted(ss["n_units"]["choices"])
    dims: list[int] = []
    for i in range(n_layers):
        dims.append(trial.suggest_categorical(f"n_units_l{i}", unit_choices))
    hidden_dims = tuple(dims)

    return {
        "learning_rate": lr,
        "weight_decay": wd,
        "dropout": dropout,
        "batch_size": batch_size,
        "hidden_dims": hidden_dims,
        "max_epochs": fixed["max_epochs"],
        "n_iter_no_change": fixed["n_iter_no_change"],
        "tol": fixed["tol"],
        "seed": fixed["seed"],
    }


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_calibration(
    network: torch.nn.Module,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    val_dataset: FeatureDataset,
    batch_size: int = 4096,
) -> Dict[str, float]:
    """Compute calibration and discrimination metrics on a validation set.

    Returns a dict with keys ``brier_score``, ``pr_auc``, ``ece``, and
    ``val_bce_loss``.
    """
    from sklearn.metrics import average_precision_score

    device = next(network.parameters()).device
    network.eval()

    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_bce = 0.0
    n_total = 0
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")

    loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    for features, labels in loader:
        features = (features.to(device) - feature_mean) / feature_std
        labels = labels.to(device)
        logits = network(features)
        total_bce += criterion(logits, labels).item()
        n_total += len(labels)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    probs_arr = np.concatenate(all_probs)
    labels_arr = np.concatenate(all_labels)

    brier = float(np.mean((probs_arr - labels_arr) ** 2))
    pr_auc = float(average_precision_score(labels_arr, probs_arr))
    ece = _expected_calibration_error(probs_arr, labels_arr)
    val_bce = total_bce / max(n_total, 1)

    return {
        "brier_score": brier,
        "pr_auc": pr_auc,
        "ece": ece,
        "val_bce_loss": val_bce,
    }


def _expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Compute Expected Calibration Error with equal-width bins."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs > lo) & (probs <= hi)
        if not mask.any():
            continue
        bin_probs = probs[mask]
        bin_labels = labels[mask]
        avg_confidence = bin_probs.mean()
        avg_accuracy = bin_labels.mean()
        ece += mask.sum() * abs(avg_confidence - avg_accuracy)
    return float(ece / max(len(probs), 1))


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def make_objective(
    train_dataset: FeatureDataset,
    val_dataset: FeatureDataset,
    cfg: Dict[str, Any],
    features: Dict,
    verbose: bool = False,
):
    """Return a closure suitable for ``study.optimize``.

    Args:
        train_dataset: Training features and labels.
        val_dataset: Validation features and labels.
        cfg: HPO search-space configuration.
        features: Feature dict from the reference calibrator, passed to
            each trial's calibrator so ``columns`` is always populated.
        verbose: Whether to show tqdm bars during training.
    """

    def objective(trial: optuna.Trial) -> float:
        from winnow.calibration.calibrator import ProbabilityCalibrator

        hp = suggest_hyperparameters(trial, cfg)

        calibrator = ProbabilityCalibrator(
            features=features,
            hidden_dims=hp["hidden_dims"],
            dropout=hp["dropout"],
            learning_rate=hp["learning_rate"],
            weight_decay=hp["weight_decay"],
            max_epochs=hp["max_epochs"],
            batch_size=hp["batch_size"],
            n_iter_no_change=hp["n_iter_no_change"],
            tol=hp["tol"],
            seed=hp["seed"],
        )

        def epoch_callback(epoch: int, val_loss: float) -> None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        t0 = time.perf_counter()
        history = calibrator.fit_from_features(
            train_dataset,
            val_dataset,
            progress_bar=verbose,
            epoch_callback=epoch_callback,
        )
        elapsed = time.perf_counter() - t0

        assert calibrator.network is not None
        assert calibrator.feature_mean is not None
        assert calibrator.feature_std is not None
        metrics = evaluate_calibration(
            calibrator.network,
            calibrator.feature_mean,
            calibrator.feature_std,
            val_dataset,
            batch_size=hp["batch_size"],
        )

        trial.set_user_attr("pr_auc", metrics["pr_auc"])
        trial.set_user_attr("ece", metrics["ece"])
        trial.set_user_attr("val_bce_loss", metrics["val_bce_loss"])
        trial.set_user_attr("epochs_trained", history.epochs_trained)
        trial.set_user_attr("best_epoch", history.best_epoch)
        trial.set_user_attr("hidden_dims", str(hp["hidden_dims"]))
        trial.set_user_attr("elapsed_s", round(elapsed, 1))

        logger.info(
            "Trial %d: brier=%.6f  pr_auc=%.4f  ece=%.4f  bce=%.6f  "
            "epochs=%d  best_epoch=%d  elapsed=%.1fs  dims=%s",
            trial.number,
            metrics["brier_score"],
            metrics["pr_auc"],
            metrics["ece"],
            metrics["val_bce_loss"],
            history.epochs_trained,
            history.best_epoch,
            elapsed,
            hp["hidden_dims"],
        )

        return metrics["brier_score"]

    return objective


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_study_summary(study: optuna.Study) -> None:
    best = study.best_trial
    print("\n" + "=" * 60)
    print("HPO COMPLETE")
    print("=" * 60)
    print(f"Best trial:        #{best.number}")
    print(f"  Brier score:     {best.value:.6f}")
    print(f"  PR-AUC:          {best.user_attrs.get('pr_auc', 'n/a')}")
    print(f"  ECE:             {best.user_attrs.get('ece', 'n/a')}")
    print(f"  Val BCE loss:    {best.user_attrs.get('val_bce_loss', 'n/a')}")
    print(f"  Epochs trained:  {best.user_attrs.get('epochs_trained', 'n/a')}")
    print(f"  Best epoch:      {best.user_attrs.get('best_epoch', 'n/a')}")
    print(f"  Elapsed:         {best.user_attrs.get('elapsed_s', 'n/a')}s")
    print()
    print("Best hyperparameters:")
    for k, v in best.params.items():
        print(f"  {k:20s} {v}")
    print()

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    pruned = [t for t in study.trials if t.state.name == "PRUNED"]
    print(
        f"Trials: {len(study.trials)} total, "
        f"{len(completed)} completed, {len(pruned)} pruned"
    )
    print("=" * 60)


def _retrain_and_save_best(
    study: optuna.Study,
    train_dataset: FeatureDataset,
    val_dataset: FeatureDataset,
    cfg: Dict[str, Any],
    features: Dict,
    output_dir: Path,
) -> None:
    """Retrain the best trial's configuration and save the resulting model."""
    from winnow.calibration.calibrator import ProbabilityCalibrator

    best = study.best_trial
    fixed = cfg["fixed"]

    n_layers = best.params["n_layers"]
    hidden_dims = tuple(best.params[f"n_units_l{i}"] for i in range(n_layers))

    logger.info(
        "Retraining best trial #%d (brier=%.6f) for saving ...",
        best.number,
        best.value,
    )

    calibrator = ProbabilityCalibrator(
        features=features,
        hidden_dims=hidden_dims,
        dropout=best.params["dropout"],
        learning_rate=best.params["learning_rate"],
        weight_decay=best.params["weight_decay"],
        max_epochs=fixed["max_epochs"],
        batch_size=best.params["batch_size"],
        n_iter_no_change=fixed["n_iter_no_change"],
        tol=fixed["tol"],
        seed=fixed["seed"],
    )
    calibrator.fit_from_features(train_dataset, val_dataset)

    ProbabilityCalibrator.save(calibrator, output_dir)
    logger.info("Best model saved to %s", output_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the HPO script."""
    p = argparse.ArgumentParser(
        description="Optuna HPO for the Winnow calibrator.",
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
        help="YAML config with search space and fixed params "
        f"(default: {_DEFAULT_CONFIG}).",
    )
    p.add_argument(
        "--calibrator-config-dir",
        type=str,
        default=None,
        help="Custom config directory for calibrator.yaml. "
        "Defaults to the standard winnow config directory.",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials (default: 100).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=43200,
        help="Hard wall-clock timeout in seconds (default: 43200 = 12h).",
    )
    p.add_argument(
        "--pruning",
        action="store_true",
        help="Enable MedianPruner for early trial termination.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show tqdm bars and winnow debug logging.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hpo_best_model"),
        help="Directory to save the best model to (default: hpo_best_model/).",
    )
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. --set search_space.dropout.high=0.7",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the Optuna hyperparameter optimisation study."""
    args = parse_args(argv)

    # -- Logging --------------------------------------------------------
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not args.verbose:
        logging.getLogger("winnow").setLevel(logging.WARNING)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # -- Config ---------------------------------------------------------
    cfg = _load_config(args.config)
    if args.overrides:
        _apply_overrides(cfg, args.overrides)

    logger.info("Search space config:\n%s", yaml.dump(cfg, default_flow_style=False))

    # -- Reference calibrator (for feature definitions) -----------------
    ref_calibrator = _instantiate_reference_calibrator(args.calibrator_config_dir)
    feature_columns = ["confidence"] + ref_calibrator.columns
    logger.info(
        "Feature set from calibrator.yaml: %d columns %s",
        len(feature_columns),
        feature_columns,
    )

    # -- Data -----------------------------------------------------------
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

    # -- Study ----------------------------------------------------------
    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        if args.pruning
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        study_name="winnow-calibrator-hpo",
    )

    objective = make_objective(
        train_dataset,
        val_dataset,
        cfg,
        features=ref_calibrator.feature_dict,
        verbose=args.verbose,
    )

    logger.info(
        "Starting Optuna study: %d trials, timeout=%ds, pruning=%s",
        args.n_trials,
        args.timeout,
        args.pruning,
    )
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    _print_study_summary(study)

    # -- Save best model ------------------------------------------------
    _retrain_and_save_best(
        study,
        train_dataset,
        val_dataset,
        cfg,
        features=ref_calibrator.feature_dict,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
