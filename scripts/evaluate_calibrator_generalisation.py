"""Evaluate the generalisation of the calibrator by training on one source dataset and testing on another."""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from huggingface_hub import snapshot_download
from rich.logging import RichHandler
import typer
from typing import Dict

from winnow.calibration.calibration_features import (
    PrositFeatures,
    MassErrorFeature,
    RetentionTimeFeature,
    ChimericFeatures,
    BeamFeatures,
)
from winnow.calibration.calibrator import ProbabilityCalibrator
from winnow.datasets.calibration_dataset import RESIDUE_MASSES, CalibrationDataset
from winnow.datasets.data_loaders import InstaNovoDatasetLoader

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
DATA_DIR_OPTION = typer.Option(help="Path to the dataset directory.")
MODEL_OUTPUT_DIR_OPTION = typer.Option(help="Directory to save trained models.")
RESULTS_OUTPUT_DIR_OPTION = typer.Option(help="Directory to save evaluation results.")


def download_dataset(repo_id: str, local_dir: str, pattern: str) -> None:
    """Download the dataset from the Hugging Face Hub."""
    logger.info(f"Downloading dataset {repo_id} to {local_dir}.")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=pattern,
        repo_type="dataset",
    )


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


def load_combined_dataset(
    spectrum_data_path: Path, beam_data_path: Path
) -> CalibrationDataset:
    """Load the combined dataset using InstaNovoDatasetLoader."""
    logger.info(
        f"Loading combined dataset from {spectrum_data_path} and {beam_data_path}"
    )
    loader = InstaNovoDatasetLoader()
    dataset = loader.load(spectrum_data_path, beam_data_path)

    dataset = filter_dataset(dataset)

    return dataset


def filter_dataset(dataset: CalibrationDataset) -> CalibrationDataset:
    """Filter out rows whose predictions are empty or contain unsupported PSMs.

    Args:
        dataset (CalibrationDataset): The dataset to be filtered

    Returns:
        CalibrationDataset: The filtered dataset
    """
    logger.info("Filtering dataset.")
    filtered_dataset = (
        dataset.filter_entries(
            metadata_predicate=lambda row: not isinstance(row["prediction"], list)
        )
        .filter_entries(metadata_predicate=lambda row: not row["prediction"])
        .filter_entries(
            metadata_predicate=lambda row: row["precursor_charge"] > 6
        )  # Prosit-specific filtering, see https://github.com/Nesvilab/FragPipe/issues/1775
        .filter_entries(
            predictions_predicate=lambda row: len(row[0].sequence) > 30
        )  # Prosit-specific filtering
        .filter_entries(
            predictions_predicate=lambda row: len(row[1].sequence) > 30
        )  # Prosit-specific filtering
    )
    return filtered_dataset


def create_train_test_splits(
    dataset: CalibrationDataset,
) -> Dict[str, Dict[str, CalibrationDataset]]:
    """Create train/test splits for each source dataset with 80/20 split for training datasets."""
    # Get unique source datasets
    source_datasets = dataset.metadata["source_dataset"].unique()
    logger.info(f"Found source datasets: {source_datasets.tolist()}")

    # First, separate each source dataset
    source_datasets_data = {}
    for source_dataset in source_datasets:
        mask = dataset.metadata["source_dataset"] == source_dataset
        indices = dataset.metadata[mask].index.tolist()

        metadata = dataset.metadata.iloc[indices].reset_index(drop=True)
        predictions = [dataset.predictions[i] for i in indices]

        source_datasets_data[source_dataset] = CalibrationDataset(
            metadata=metadata,
            predictions=predictions,
        )
        logger.info(f"{source_dataset}: {len(metadata)} samples")

    # Now create training scenarios for each source dataset
    training_scenarios = {}

    for train_source in source_datasets:
        logger.info(f"Creating training scenario for {train_source}...")
        scenario_name = train_source
        scenario_data = {}

        # Split the training source dataset 80/20
        train_dataset = source_datasets_data[train_source]
        if len(train_dataset.metadata) > 1:
            train_indices, test_indices = train_test_split(
                np.arange(len(train_dataset.metadata)),
                test_size=TEST_SIZE,
                random_state=SEED,
            )

            # Create training split (80%)
            train_metadata = train_dataset.metadata.iloc[train_indices].reset_index(
                drop=True
            )
            train_predictions = [
                train_dataset.predictions[i] for i in train_indices.tolist()
            ]
            scenario_data[f"{train_source}_train"] = CalibrationDataset(
                metadata=train_metadata,
                predictions=train_predictions,
            )

            # Create in-distribution test split (20%)
            test_metadata = train_dataset.metadata.iloc[test_indices].reset_index(
                drop=True
            )
            test_predictions = [
                train_dataset.predictions[i] for i in test_indices.tolist()
            ]
            scenario_data[f"{train_source}_test"] = CalibrationDataset(
                metadata=test_metadata,
                predictions=test_predictions,
            )

            logger.info(f"  {train_source} train: {len(train_metadata)} samples")
            logger.info(f"  {train_source} test: {len(test_metadata)} samples")
        else:
            # If only one sample, use it for training and testing
            scenario_data[f"{train_source}_train"] = train_dataset
            scenario_data[f"{train_source}_test"] = train_dataset
            logger.info(
                f"  {train_source}: Only 1 sample, using for both train and test"
            )

        # Add all other source datasets as full test sets (out-of-distribution)
        for other_source in source_datasets:
            if other_source != train_source:
                scenario_data[f"{other_source}_full"] = source_datasets_data[
                    other_source
                ]
                logger.info(
                    f"  {other_source} full: {len(source_datasets_data[other_source].metadata)} samples"
                )

        training_scenarios[scenario_name] = scenario_data

    return training_scenarios


def evaluate_model(
    model: ProbabilityCalibrator,
    test_dataset: CalibrationDataset,
    train_dataset_name: str,
    test_dataset_name: str,
    evaluation_type: str,
) -> pd.DataFrame:
    """Evaluate a model on a test dataset and return results."""
    # Make predictions
    model.predict(test_dataset)

    # Add dataset names and evaluation type to results
    results = test_dataset.metadata.copy()
    results["trained_on_dataset"] = train_dataset_name
    results["test_dataset"] = test_dataset_name
    results["evaluation_type"] = evaluation_type

    return results


def main(
    data_dir: Path = DATA_DIR_OPTION,
    model_output_dir: Path = MODEL_OUTPUT_DIR_OPTION,
    results_output_dir: Path = RESULTS_OUTPUT_DIR_OPTION,
):
    """Evaluate calibrator generalisation with in-distribution and out-of-distribution evaluation."""
    # Create output directories
    model_output_dir.mkdir(parents=True, exist_ok=True)
    results_output_dir.mkdir(parents=True, exist_ok=True)

    # Download the dataset from the Hugging Face Hub
    logger.info("Downloading dataset from the Hugging Face Hub...")
    download_dataset(
        repo_id="instadeepai/winnow-ms-datasets",
        local_dir=str(data_dir),
        pattern="general_train*",
    )

    # Load the combined dataset
    logger.info("Loading combined dataset...")
    full_dataset = load_combined_dataset(
        data_dir / "general_train.parquet",
        data_dir / "general_train_beams.csv",
    )

    # Create train/test splits for each source dataset
    logger.info("Creating train/test splits...")
    training_scenarios = create_train_test_splits(full_dataset)

    # Train models and evaluate
    all_results = []
    for train_source, scenario_data in training_scenarios.items():
        logger.info(f"Training model on {train_source}...")

        # Get the training dataset (80% of the source dataset)
        train_dataset = scenario_data[f"{train_source}_train"]
        logger.info(
            f"Training on {len(train_dataset.metadata)} samples from {train_source}"
        )

        # Train model
        calibrator = initialise_calibrator()
        calibrator.fit(train_dataset)

        # Save model
        model_path = model_output_dir / f"trained_on_{train_source}"
        ProbabilityCalibrator.save(calibrator, model_path)

        # Evaluate on all test sets
        logger.info(f"Evaluating model trained on {train_source}...")

        for test_dataset_name, test_dataset in scenario_data.items():
            if test_dataset_name.endswith("_train"):
                continue  # Skip training sets

            # Determine evaluation type
            if test_dataset_name == f"{train_source}_test":
                evaluation_type = "in_distribution"
                display_name = f"{train_source} (in-dist)"
            else:
                evaluation_type = "out_of_distribution"
                display_name = test_dataset_name.replace("_full", " (out-of-dist)")

            logger.info(
                f"  Evaluating on {display_name}: {len(test_dataset.metadata)} samples"
            )

            results = evaluate_model(
                calibrator,
                test_dataset,
                train_source,
                test_dataset_name.replace("_test", "").replace("_full", ""),
                evaluation_type,
            )
            all_results.append(results)

    # Combine and save all results
    combined_results = pd.concat(all_results, ignore_index=True)

    # Save full results
    # results_path = results_output_dir / "calibrator_generalisation_results.csv"
    # combined_results.to_csv(results_path, index=False)

    # Save results without arrays to save storage space, this saves about 10GB of space
    combined_results_no_arrays = combined_results.drop(["mz_array", "intensity_array"])
    results_path_no_arrays = (
        results_output_dir / "calibrator_generalisation_results_no_arrays.csv"
    )
    combined_results_no_arrays.to_csv(results_path_no_arrays, index=False)

    logger.info(f"Results saved to {results_path_no_arrays}")

    # Print summary
    logger.info("Evaluation summary:")
    summary = (
        combined_results.groupby(
            ["trained_on_dataset", "test_dataset", "evaluation_type"]
        )
        .size()
        .reset_index(name="num_samples")
    )
    for _, row in summary.iterrows():
        logger.info(
            f"  Trained on {row['trained_on_dataset']}, tested on {row['test_dataset']} ({row['evaluation_type']}): {row['num_samples']} samples"
        )


if __name__ == "__main__":
    typer.run(main)
