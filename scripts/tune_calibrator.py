"""Script to perform hyperparameter tuning for MLPClassifier.

This script implements a comprehensive hyperparameter tuning experiment for
MLPClassifier using a fixed train/validation split. It evaluates different
combinations of hyperparameters and generates visualizations of the results.

The script supports:
- Fixed train/validation split for proper evaluation
- Grid search over multiple hyperparameters
- Multiple evaluation metrics (AUC, Brier score, calibration error, etc.)
- Publication-quality visualizations
- Comprehensive logging and documentation
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
import logging
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.calibration import calibration_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import typer
from typing_extensions import Annotated

from winnow.datasets.calibration_dataset import CalibrationDataset, RESIDUE_MASSES
from winnow.calibration.calibrator import ProbabilityCalibrator
from winnow.calibration.calibration_features import (
    PrositFeatures,
    MassErrorFeature,
    RetentionTimeFeature,
    ChimericFeatures,
    BeamFeatures,
)

# Set up logging
logger = logging.getLogger("winnow")
logger.setLevel(logging.INFO)

# Constants
SEED = 42


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    auc: float
    brier_score: float
    average_precision: float
    calibration_error: float
    precision_at_90: float
    recall_at_90: float
    f1_at_90: float
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]

    def __post_init__(self) -> None:
        """Validate the metrics after initialization."""
        if not isinstance(self.confusion_matrix, np.ndarray):
            raise TypeError("confusion_matrix must be a numpy array")
        if not isinstance(self.classification_report, dict):
            raise TypeError("classification_report must be a dictionary")


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.9,
) -> EvaluationMetrics:
    """Compute comprehensive evaluation metrics for a classifier.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Probability threshold for precision/recall metrics

    Returns:
        EvaluationMetrics object containing all computed metrics
    """
    # Basic metrics
    auc = roc_auc_score(y_true, y_pred_proba)
    brier_score = brier_score_loss(y_true, y_pred_proba)
    average_precision = average_precision_score(y_true, y_pred_proba)

    # Calibration error
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
    calibration_error = np.mean(np.abs(prob_true - prob_pred))

    # Precision/Recall at threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    precision = precision_recall_curve(y_true, y_pred_proba)[0]
    recall = precision_recall_curve(y_true, y_pred_proba)[1]
    thresholds = precision_recall_curve(y_true, y_pred_proba)[2]

    # Find closest threshold to desired value
    idx = np.argmin(np.abs(thresholds - threshold))
    precision_at_90 = precision[idx]
    recall_at_90 = recall[idx]
    f1_at_90 = (
        2 * (precision_at_90 * recall_at_90) / (precision_at_90 + recall_at_90)
        if (precision_at_90 + recall_at_90) > 0
        else 0.0
    )

    # Confusion matrix and classification report
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,  # Set precision to 0 when there are no predicted samples
    )

    return EvaluationMetrics(
        auc=auc,
        brier_score=brier_score,
        average_precision=average_precision,
        calibration_error=calibration_error,
        precision_at_90=precision_at_90,
        recall_at_90=recall_at_90,
        f1_at_90=f1_at_90,
        confusion_matrix=cm,
        classification_report=report,
    )


def initialize_calibrator(
    dataset: CalibrationDataset,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize calibrator and compute features.

    Args:
        dataset: The calibration dataset
        seed: Random seed for reproducibility

    Returns:
        Tuple of (features, labels)
    """
    # Initialize calibrator with features
    logger.info("Initializing calibrator.")
    calibrator = ProbabilityCalibrator(seed=seed)
    calibrator.add_feature(MassErrorFeature(residue_masses=RESIDUE_MASSES))
    calibrator.add_feature(PrositFeatures(mz_tolerance=0.02))
    calibrator.add_feature(RetentionTimeFeature(hidden_dim=10, train_fraction=0.1))
    calibrator.add_feature(ChimericFeatures(mz_tolerance=0.02))
    calibrator.add_feature(BeamFeatures())

    # Fit calibrator and compute features
    logger.info("Computing features.")
    features, labels = calibrator.compute_features(dataset=dataset, labelled=True)  # type: ignore[misc]

    return features, labels


def filter_dataset(dataset: CalibrationDataset) -> CalibrationDataset:
    """Filter out rows whose predictions are empty or contain unsupported PSMs.

    Args:
        dataset: The dataset to be filtered

    Returns:
        Filtered dataset
    """
    logger.info("Filtering dataset.")
    filtered_dataset = (
        dataset.filter_entries(
            metadata_predicate=lambda row: not isinstance(row["prediction"], list),
        )
        .filter_entries(metadata_predicate=lambda row: not row["prediction"])
        .filter_entries(
            metadata_predicate=lambda row: row["precursor_charge"] > 6
        )  # Prosit-specific filtering
        .filter_entries(
            predictions_predicate=lambda row: len(row[1].sequence) > 30
        )  # Prosit-specific filtering
        .filter_entries(
            predictions_predicate=lambda row: len(row[0].sequence) > 30
        )  # Prosit-specific filtering
    )
    return filtered_dataset


def plot_parameter_importance(
    results: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot importance of different hyperparameters.

    Args:
        results: DataFrame with validation results
        output_dir: Directory to save plots
    """
    # Plot validation AUC for each parameter value
    param_names = [col for col in results.columns if col.startswith("param_")]

    for param in param_names:
        plt.figure(figsize=(10, 6))
        # Convert parameter values to appropriate type for plotting
        param_values = results[param].astype(
            float if param != "param_max_depth" else object
        )
        # Replace None with "None" for max_depth
        if param == "param_max_depth":
            param_values = param_values.replace({np.nan: "None"})

        sns.barplot(data=results, x=param_values, y="val_auc")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Impact of {param.replace('param_', '')} on Validation AUC")
        plt.xlabel(param.replace("param_", ""))
        plt.ylabel("Validation AUC")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{param}_importance.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_parameter_interactions(
    results: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot interactions between important hyperparameters.

    Args:
        results: DataFrame with validation results
        output_dir: Directory to save plots
    """
    # Get top parameters by variance in scores
    param_names = [col for col in results.columns if col.startswith("param_")]
    param_importance = []

    for param in param_names:
        scores_by_param = results.groupby(param)["val_auc"].std()
        param_importance.append((param, scores_by_param.mean()))

    # Sort by importance and take top 2
    top_params = sorted(param_importance, key=lambda x: x[1], reverse=True)[:2]
    if len(top_params) >= 2:
        param1, param2 = top_params[0][0], top_params[1][0]

        # Convert parameter values to appropriate type for plotting
        param1_values = results[param1].astype(
            float if param1 != "param_max_depth" else object
        )
        param2_values = results[param2].astype(
            float if param2 != "param_max_depth" else object
        )

        # Replace None with "None" for max_depth
        if param1 == "param_max_depth":
            param1_values = param1_values.replace({np.nan: "None"})
        if param2 == "param_max_depth":
            param2_values = param2_values.replace({np.nan: "None"})

        plt.figure(figsize=(12, 8))
        pivot_table = results.pivot_table(
            values="val_auc",
            index=param1_values,
            columns=param2_values,
            aggfunc="mean",
        )
        sns.heatmap(pivot_table, annot=True, cmap="YlOrRd", fmt=".3f")
        plt.title(
            f"Interaction between {param1.replace('param_', '')} and {param2.replace('param_', '')}"
        )
        plt.xlabel(param2.replace("param_", ""))
        plt.ylabel(param1.replace("param_", ""))
        plt.tight_layout()
        plt.savefig(
            output_dir / "parameter_interactions.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def format_parameter_value(value: Any) -> str:
    """Format parameter value for display.

    Args:
        value: Parameter value to format

    Returns:
        Formatted string representation of the value
    """
    if pd.isna(value):
        return "None"
    if isinstance(value, (int, float)):
        # Remove decimal point for integers
        return str(int(value)) if value.is_integer() else str(value)
    return str(value)


def load_split_datasets(
    train_spectrum_path: Path,
    train_predictions_path: Path,
    val_spectrum_path: Path,
    val_predictions_path: Path,
) -> Tuple[CalibrationDataset, np.ndarray, np.ndarray]:
    """Load pre-split train/val datasets and compute features on combined dataset.

    Args:
        train_spectrum_path: Path to training spectrum data file
        train_predictions_path: Path to training beam predictions file
        val_spectrum_path: Path to validation spectrum data file
        val_predictions_path: Path to validation beam predictions file

    Returns:
        Tuple of (combined dataset, train indices, val indices)
    """
    logger.info("Loading pre-split datasets...")

    # Load each split
    train_dataset = CalibrationDataset.from_predictions_csv(
        spectrum_path=train_spectrum_path,
        beam_predictions_path=train_predictions_path,
    )
    train_dataset = filter_dataset(train_dataset)

    val_dataset = CalibrationDataset.from_predictions_csv(
        spectrum_path=val_spectrum_path,
        beam_predictions_path=val_predictions_path,
    )
    val_dataset = filter_dataset(val_dataset)

    # Combine datasets
    combined_dataset = CalibrationDataset(
        metadata=pd.concat(
            [train_dataset.metadata, val_dataset.metadata], ignore_index=True
        ),
        predictions=train_dataset.predictions + val_dataset.predictions,  # type: ignore
    )

    # Create indices for train/val split
    train_indices = np.arange(len(train_dataset))
    val_indices = np.arange(len(train_dataset), len(combined_dataset))

    logger.info(
        f"Loaded {len(train_dataset)} train and {len(val_dataset)} validation samples"
    )

    return combined_dataset, train_indices, val_indices


def evaluate_parameters(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    param_grid: Dict[str, List[Any]],
) -> pd.DataFrame:
    """Evaluate different parameter combinations using train/val split.

    Args:
        train_features: Training feature matrix
        train_labels: Training labels
        val_features: Validation feature matrix
        val_labels: Validation labels
        param_grid: Dictionary of parameters to evaluate

    Returns:
        DataFrame with validation results for each parameter combination
    """
    results = []

    # Scale features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)

    # Generate all parameter combinations
    param_combinations = [
        dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())
    ]

    for params in param_combinations:
        # Initialize and train classifier
        classifier = MLPClassifier(random_state=SEED, **params)
        classifier.fit(train_features_scaled, train_labels)

        # Get predictions on validation set
        val_pred_proba = classifier.predict_proba(val_features_scaled)[:, 1]

        # Compute metrics
        metrics = compute_metrics(val_labels, val_pred_proba)

        # Store results
        result = {
            "val_auc": metrics.auc,
            "val_brier_score": metrics.brier_score,
            "val_average_precision": metrics.average_precision,
            "val_calibration_error": metrics.calibration_error,
            "val_precision_at_90": metrics.precision_at_90,
            "val_recall_at_90": metrics.recall_at_90,
            "val_f1_at_90": metrics.f1_at_90,
        }
        # Add parameters
        for param, value in params.items():
            result[f"param_{param}"] = value

        results.append(result)

    return pd.DataFrame(results)


def main(
    train_spectrum_path: Annotated[
        Path,
        typer.Option(help="Path to training spectrum data file."),
    ],
    train_predictions_path: Annotated[
        Path,
        typer.Option(help="Path to training beam predictions file."),
    ],
    val_spectrum_path: Annotated[
        Path,
        typer.Option(help="Path to validation spectrum data file."),
    ],
    val_predictions_path: Annotated[
        Path,
        typer.Option(help="Path to validation beam predictions file."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory to save tuning results and plots."),
    ],
) -> None:
    """Perform hyperparameter tuning for MLPClassifier.

    This script performs a comprehensive hyperparameter tuning experiment for
    MLPClassifier using a fixed train/validation split. It evaluates different
    combinations of hyperparameters and generates visualizations of the results.

    Args:
        train_spectrum_path: Path to training spectrum data file
        train_predictions_path: Path to training beam predictions file
        val_spectrum_path: Path to validation spectrum data file
        val_predictions_path: Path to validation beam predictions file
        output_dir: Directory to save results and plots
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets and compute features on combined data
    logger.info("Loading datasets and computing features...")
    combined_dataset, train_indices, val_indices = load_split_datasets(
        train_spectrum_path=train_spectrum_path,
        train_predictions_path=train_predictions_path,
        val_spectrum_path=val_spectrum_path,
        val_predictions_path=val_predictions_path,
    )

    # Compute features on combined dataset
    features, labels = initialize_calibrator(combined_dataset)

    # Split features and labels
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    val_features = features[val_indices]
    val_labels = labels[val_indices]

    # Define parameter grid with proper typing
    param_grid: Dict[str, List[Any]] = {
        "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50), (100, 100)],
        "alpha": [1e-5, 1e-4, 1e-3],  # L2 penalty
        "learning_rate_init": [0.001, 0.01, 0.1],
        "max_iter": [200, 500, 1000, 2000, 5000],
    }

    # Evaluate parameter combinations
    logger.info("Evaluating parameter combinations...")
    results = evaluate_parameters(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        param_grid=param_grid,
    )

    # Save results
    results.to_csv(output_dir / "parameter_tuning_results.csv", index=False)

    # Plot results
    logger.info("Generating plots...")
    plot_parameter_importance(results, output_dir)
    plot_parameter_interactions(results, output_dir)

    # Print best parameters and score
    best_idx = results["val_auc"].idxmax()
    best_params = {
        k.replace("param_", ""): format_parameter_value(v)
        for k, v in results.iloc[best_idx].items()
        if k.startswith("param_")
    }
    best_score = results.iloc[best_idx]["val_auc"]

    print("\nBest parameters found:")
    print("-" * 60)
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"\nBest validation AUC: {best_score:.4f}")
    print("-" * 60 + "\n")

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    typer.run(main)
