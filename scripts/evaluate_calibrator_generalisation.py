#!/usr/bin/env python3

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from rich.logging import RichHandler
import typer
from typing import Dict, Tuple

from winnow.calibration.calibration_features import (
    PrositFeatures,
    MassErrorFeature,
    RetentionTimeFeature,
    ChimericFeatures,
    BeamFeatures,
)
from winnow.calibration.calibrator import ProbabilityCalibrator
from winnow.datasets.calibration_dataset import RESIDUE_MASSES, CalibrationDataset
from winnow.scripts.main import load_dataset, filter_dataset, DataSource

# --- Logging Setup ---
logger = logging.getLogger("winnow.evaluate_generalization")  # Use a unique logger name
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent propagation to parent loggers
logger.addHandler(RichHandler())

# --- Constants ---
SEED = 42
MZ_TOLERANCE = 0.02
HIDDEN_DIM = 10
TRAIN_FRACTION = 0.1
TEST_SIZE = 0.2

# --- Typer Options ---
DATA_SOURCE_OPTION = typer.Option(help="The type of PSM dataset to be calibrated.")
CONFIG_DIR_OPTION = typer.Option(help="Directory containing dataset config files.")
MODEL_OUTPUT_DIR_OPTION = typer.Option(help="Directory to save trained models.")
RESULTS_OUTPUT_DIR_OPTION = typer.Option(help="Directory to save evaluation results.")


def initialise_calibrator() -> ProbabilityCalibrator:
    """Set up the probability calibrator with features."""
    calibrator = ProbabilityCalibrator(SEED)
    calibrator.add_feature(MassErrorFeature(residue_masses=RESIDUE_MASSES))
    calibrator.add_feature(PrositFeatures(mz_tolerance=MZ_TOLERANCE))
    calibrator.add_feature(
        RetentionTimeFeature(hidden_dim=HIDDEN_DIM, train_fraction=TRAIN_FRACTION)
    )
    calibrator.add_feature(ChimericFeatures(mz_tolerance=MZ_TOLERANCE))
    calibrator.add_feature(BeamFeatures())
    return calibrator


def load_and_split_dataset(
    data_source: DataSource,
    dataset_config_path: Path,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> Tuple[CalibrationDataset, CalibrationDataset]:
    """Load a dataset and split it into train and test sets."""
    dataset = load_dataset(data_source, dataset_config_path)
    dataset = filter_dataset(dataset)

    # Split indices
    train_idx, test_idx = train_test_split(
        np.arange(len(dataset.metadata)), test_size=test_size, random_state=seed
    )

    # Create train and test datasets
    train_dataset = CalibrationDataset(
        metadata=dataset.metadata.iloc[train_idx].reset_index(drop=True),
        predictions=[dataset.predictions[i] for i in train_idx.tolist()],
    )
    test_dataset = CalibrationDataset(
        metadata=dataset.metadata.iloc[test_idx].reset_index(drop=True),
        predictions=[dataset.predictions[i] for i in test_idx.tolist()],
    )

    return train_dataset, test_dataset


def evaluate_model(
    model: ProbabilityCalibrator, test_dataset: CalibrationDataset, dataset_name: str
) -> pd.DataFrame:
    """Evaluate a model on a test dataset and return results."""
    # Make predictions
    model.predict(test_dataset)

    # Add dataset name to results
    results = test_dataset.metadata.copy()
    results["trained_on_dataset"] = dataset_name

    return results


def main(
    data_source: DataSource = DATA_SOURCE_OPTION,
    config_dir: Path = CONFIG_DIR_OPTION,
    model_output_dir: Path = MODEL_OUTPUT_DIR_OPTION,
    results_output_dir: Path = RESULTS_OUTPUT_DIR_OPTION,
):
    """Evaluate calibrator generalisation across datasets."""
    # Create output directories
    model_output_dir.mkdir(parents=True, exist_ok=True)
    results_output_dir.mkdir(parents=True, exist_ok=True)

    # Get all config files
    config_files = list(config_dir.glob("*.yaml"))
    if not config_files:
        raise ValueError(f"No config files found in {config_dir}")

    # Load and split all datasets
    logger.info("Loading and splitting datasets...")
    datasets: Dict[str, Tuple[CalibrationDataset, CalibrationDataset]] = {}
    for config_path in config_files:
        dataset_name = config_path.stem
        logger.info(f"Processing {dataset_name}...")
        train_dataset, test_dataset = load_and_split_dataset(
            data_source=data_source, dataset_config_path=config_path
        )
        datasets[dataset_name] = (train_dataset, test_dataset)

    # Train models and evaluate
    all_results = []
    for train_name, (train_dataset, _) in datasets.items():
        logger.info(f"Training model on {train_name}...")

        # Train model
        calibrator = initialise_calibrator()
        calibrator.fit(train_dataset)

        # Save model
        model_path = model_output_dir / train_name
        ProbabilityCalibrator.save(calibrator, model_path)

        # Evaluate on all test sets
        logger.info(f"Evaluating model trained on {train_name}...")
        for test_name, (_, test_dataset) in datasets.items():
            results = evaluate_model(calibrator, test_dataset, train_name)
            results["test_dataset"] = test_name
            all_results.append(results)

    # Combine and save all results
    combined_results = pd.concat(all_results, ignore_index=True)
    results_path = results_output_dir / "calibrator_generalisation_results.csv"
    combined_results.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    typer.run(main)
