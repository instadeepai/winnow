"""Script to compare different classifiers for probability rescoring.

This script implements a rigorous comparison of different classifiers for probability
rescoring in de novo peptide sequencing. It uses a fixed train/validation split
to evaluate model architectures and hyperparameter tuning.

The script supports:
- Fixed train/validation splits for proper evaluation
- Hyperparameter tuning on validation set
- Multiple evaluation metrics (AUC, Brier score, calibration error, etc.)
- Publication-quality visualizations
- Comprehensive logging and documentation
"""

from pathlib import Path
from typing import Dict, Any, Tuple
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats
import typer
from typing_extensions import Annotated
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

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
ALPHA = 0.05  # Significance level for statistical tests


@dataclass
class DatasetConfig:
    """Configuration for a dataset to be used in the comparison."""

    name: str
    spectrum_path: Path
    beam_predictions_path: Path
    description: str = ""


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


@dataclass
class DatasetSplit:
    """Container for dataset splits."""

    train: CalibrationDataset
    val: CalibrationDataset


# Define classifiers
CLASSIFIERS = {
    "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED),
    "XGBoost": XGBClassifier(random_state=SEED),
    "RandomForest": RandomForestClassifier(random_state=SEED),
    "ExtraTrees": ExtraTreesClassifier(random_state=SEED),
    "SVC": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(probability=True, random_state=SEED)),
        ]
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        random_state=SEED,
    ),
    "KNeighbors": KNeighborsClassifier(
        weights="distance",
        metric="euclidean",
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=1000,
        random_state=SEED,
    ),
}


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
    f1_at_90 = 2 * (precision_at_90 * recall_at_90) / (precision_at_90 + recall_at_90)

    # Confusion matrix and classification report
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

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


def perform_statistical_testing(
    results: Dict[str, Dict[str, Any]],
    metric: str,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Perform statistical testing between classifiers.

    Args:
        results: Dictionary of evaluation results
        metric: Metric to test
        alpha: Significance level

    Returns:
        DataFrame with p-values and significance indicators
    """
    classifiers = list(results.keys())
    n_classifiers = len(classifiers)

    # Initialize results matrix
    p_values = np.zeros((n_classifiers, n_classifiers))
    significant = np.zeros((n_classifiers, n_classifiers), dtype=bool)

    # Perform pairwise tests
    for i, clf1 in enumerate(classifiers):
        for j, clf2 in enumerate(classifiers):
            if i != j:
                # Get metric values
                val1 = getattr(results[clf1]["val_metrics"], metric)
                val2 = getattr(results[clf2]["val_metrics"], metric)

                # Perform t-test
                _, p_val = stats.ttest_ind(
                    [val1],  # Single value for each classifier
                    [val2],
                )
                p_values[i, j] = p_val
                significant[i, j] = p_val < alpha

    # Create DataFrame
    df = pd.DataFrame(
        p_values,
        index=classifiers,
        columns=classifiers,
    )

    # Add significance indicators
    df_significant = pd.DataFrame(
        significant,
        index=classifiers,
        columns=classifiers,
    ).astype(str)
    df_significant = df_significant.replace({"True": "*", "False": ""})

    # Combine p-values and significance
    df = df.round(4).astype(str) + df_significant

    return df


def plot_roc_curves(
    results: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Plot ROC curves for all classifiers.

    Args:
        results: Dictionary of evaluation results
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(10, 8))

    for name, result in results.items():
        # Get predictions and true labels
        y_true = result["val_labels"]
        y_pred_proba = result["val_predictions"]

        # Compute ROC curve
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)

        # Plot ROC curve
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{name} (AUC = {auc:.4f})",
        )

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Different Classifiers")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_calibration_curves(
    results: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Plot calibration curves for all classifiers.

    Args:
        results: Dictionary of evaluation results
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(10, 8))

    for name, result in results.items():
        # Get predictions and true labels
        y_true = result["val_labels"]
        y_pred_proba = result["val_predictions"]

        # Compute calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)

        # Plot calibration curve
        plt.plot(
            prob_pred,
            prob_true,
            label=f"{name}",
        )

    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("True Probability")
    plt.title("Calibration Curves for Different Classifiers")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        output_dir / "calibration_curves.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_precision_recall_curves(
    results: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Plot precision-recall curves for all classifiers.

    Args:
        results: Dictionary of evaluation results
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(10, 8))

    for name, result in results.items():
        # Get predictions and true labels
        y_true = result["val_labels"]
        y_pred_proba = result["val_predictions"]

        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

        # Plot precision-recall curve
        plt.plot(
            recall,
            precision,
            label=f"{name}",
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves for Different Classifiers")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        output_dir / "precision_recall_curves.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_metric_comparison(
    results: Dict[str, Dict[str, Any]],
    metric: str,
    output_dir: Path,
) -> None:
    """Plot comparison of a specific metric across classifiers.

    Args:
        results: Dictionary of evaluation results
        metric: Metric to plot
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(12, 6))

    # Prepare data
    data = []
    for name, result in results.items():
        value = getattr(result["val_metrics"], metric)
        data.append((name, value))

    df = pd.DataFrame(data, columns=["Classifier", "Value"])

    # Plot
    sns.barplot(data=df, x="Classifier", y="Value")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{metric.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{metric}_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


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
        dataset (CalibrationDataset): The dataset to be filtered

    Returns:
        CalibrationDataset: The filtered dataset
    """
    logger.info("Filtering dataset.")
    filtered_dataset = (
        dataset.filter_entries(
            metadata_predicate=lambda row: not isinstance(row["prediction"], list),
        )
        .filter_entries(metadata_predicate=lambda row: not row["prediction"])
        .filter_entries(
            metadata_predicate=lambda row: row["precursor_charge"] > 6
        )  # Prosit-specific filtering, see https://github.com/Nesvilab/FragPipe/issues/1775
        .filter_entries(
            predictions_predicate=lambda row: len(row[1].sequence) > 30
        )  # Prosit-specific filtering
        .filter_entries(
            predictions_predicate=lambda row: len(row[0].sequence) > 30
        )  # Prosit-specific filtering
    )
    return filtered_dataset


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


def evaluate_classifier(
    classifier: BaseEstimator,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate a classifier using train/val splits.

    Args:
        classifier: The classifier to evaluate
        train_features: Training feature matrix
        train_labels: Training labels
        val_features: Validation feature matrix
        val_labels: Validation labels

    Returns:
        Dictionary containing evaluation metrics and predictions
    """
    # Train classifier
    classifier.fit(train_features, train_labels)

    # Get predictions on validation set
    val_pred_proba = classifier.predict_proba(val_features)[:, 1]

    # Compute metrics
    val_metrics = compute_metrics(val_labels, val_pred_proba)

    return {
        "val_metrics": val_metrics,
        "val_predictions": val_pred_proba,
        "val_labels": val_labels,
    }


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
        typer.Option(help="Directory to save comparison results and plots."),
    ],
) -> None:
    """Compare different classifiers for probability rescoring.

    This script performs a comprehensive comparison of different classifiers for
    probability rescoring in de novo peptide sequencing. It uses pre-split
    train/validation datasets to evaluate model architectures and hyperparameter tuning.

    Args:
        train_spectrum_path: Path to training spectrum data file
        train_predictions_path: Path to training beam predictions file
        val_spectrum_path: Path to validation spectrum data file
        val_predictions_path: Path to validation beam predictions file
        output_dir: Directory to save results and plots
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

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

    # Save validation labels
    pd.DataFrame({"true_label": val_labels}).to_csv(
        predictions_dir / "validation_labels.csv", index=False
    )

    # Evaluate each classifier
    logger.info("Evaluating classifiers...")
    results: Dict[str, Dict[str, Any]] = {}
    for name, classifier in CLASSIFIERS.items():
        logger.info(f"Evaluating {name}...")

        # Evaluate classifier
        result = evaluate_classifier(
            classifier=classifier,
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            val_labels=val_labels,
        )

        # Save predictions
        pd.DataFrame({"predicted_probability": result["val_predictions"]}).to_csv(
            predictions_dir / f"{name}_predictions.csv", index=False
        )

        # Store results
        results[name] = result
        logger.info(
            f"{name} - Validation AUC: {result['val_metrics'].auc:.4f}, "
            f"Brier: {result['val_metrics'].brier_score:.4f}"
        )

    # Generate plots
    plot_roc_curves(results, output_dir)
    plot_calibration_curves(results, output_dir)
    plot_precision_recall_curves(results, output_dir)
    for metric in ["auc", "brier_score", "average_precision", "calibration_error"]:
        plot_metric_comparison(results, metric, output_dir)

    # Save summary to CSV
    summary_df = pd.DataFrame(
        {
            name: {
                "val_auc": result["val_metrics"].auc,
                "val_brier": result["val_metrics"].brier_score,
                "val_calibration_error": result["val_metrics"].calibration_error,
                "val_precision_at_90": result["val_metrics"].precision_at_90,
                "val_recall_at_90": result["val_metrics"].recall_at_90,
                "val_f1_at_90": result["val_metrics"].f1_at_90,
            }
            for name, result in results.items()
        }
    ).T
    summary_df.to_csv(output_dir / "summary.csv")

    # Print top classifiers based on validation performance
    print("\nTop three classifiers by validation AUC:")
    print("-" * 60)
    sorted_classifiers = sorted(
        results.items(),
        key=lambda x: x[1]["val_metrics"].auc,
        reverse=True,
    )
    for i, (name, result) in enumerate(sorted_classifiers[:3], 1):
        print(
            f"{i}. {name}: AUC = {result['val_metrics'].auc:.4f}, "
            f"Brier = {result['val_metrics'].brier_score:.4f}"
        )
    print("-" * 60 + "\n")

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    typer.run(main)
