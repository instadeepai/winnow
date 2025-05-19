"""Script to compare different classifiers for probability rescoring."""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging
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
from nets import SteepSigmoidClassifier, HingeClassifier

# Set up logging
logger = logging.getLogger("winnow")
logger.setLevel(logging.INFO)


SEED = 42


# Define classifiers to test
CLASSIFIERS = {
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=SEED
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.1, max_depth=3, random_state=SEED
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=SEED
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=SEED,
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=SEED,
    ),
    "SVC": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    probability=True,
                    kernel="rbf",
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=SEED
    ),
    "KNeighbors": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=SEED),
    "SteepSigmoidNet": SteepSigmoidClassifier(
        hidden_size=100,
        learning_rate=0.001,
        batch_size=128,
        n_epochs=100,
        random_state=SEED,
        calibrate_probs=False,
        alpha=2.0,
    ),
    "HingeNet": HingeClassifier(
        hidden_size=100,
        learning_rate=0.001,
        batch_size=128,
        n_epochs=100,
        random_state=SEED,
        calibrate_probs=False,
        alpha=1.0,
    ),
}


def plot_roc_curves(results: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
    """Plot ROC curves for all classifiers."""
    plt.figure(figsize=(10, 8))

    for name, result in results.items():
        false_positive_rate, true_positive_rate, _ = result["roc_curve"]
        auc = result["auc"]
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{name} (AUC = {auc:.3f})",
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


def plot_probability_distributions(
    results: Dict[str, Dict[str, Any]], output_dir: Path
) -> None:
    """Plot probability distributions for each classifier separately."""
    # Create a DataFrame for seaborn
    data = []
    for name, result in results.items():
        probs = result["probabilities"].flatten()  # Convert to 1D array
        labels = result["test_labels"].flatten()  # Convert to 1D array
        for prob, label in zip(probs, labels):
            data.append(
                {
                    "Classifier": name,
                    "Probability": float(prob),  # Convert numpy float to Python float
                    "Label": "True" if label else "False",
                }
            )

    df = pd.DataFrame(data)

    # Plot each classifier separately
    for name in results.keys():
        plt.figure(figsize=(10, 6))
        classifier_data = df[df["Classifier"] == name]

        # Plot using seaborn
        sns.histplot(
            data=classifier_data,
            x="Probability",
            hue="Label",
            multiple="layer",
            alpha=0.5,
            bins=50,
        )
        plt.title(f"Probability Distribution for {name}")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            output_dir / f"probability_distribution_{name.lower()}.png", dpi=300
        )
        plt.close()


def evaluate_classifier(
    classifier: BaseEstimator,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    x_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Evaluate a classifier using pre-split train/val/test data.

    Args:
        classifier: The classifier to evaluate
        x_train: Training features
        y_train: Training labels
        x_test: Test features
        y_test: Test labels
        x_val: Optional validation features to use for calibration
        y_val: Optional validation labels to use for calibration

    Returns:
        Dictionary containing evaluation metrics
    """
    # Set validation data if classifier accepts it
    if hasattr(classifier, "calibrate_probs") and classifier.calibrate_probs:
        classifier.fit(x_train, y_train, x_val, y_val)
    else:
        # For other classifiers, do not calibrate probabilities
        classifier.fit(x_train, y_train)

    # Get probabilities on test set
    probas = classifier.predict_proba(x_test)

    # Handle both single-column and two-column probability outputs
    if probas.ndim == 1 or probas.shape[1] == 1:
        # Single column case - use as is
        probas = probas.reshape(-1, 1)
    else:
        # Two column case - use probability of positive class
        probas = probas[:, 1].reshape(-1, 1)

    # Calculate AUC
    auc = roc_auc_score(y_test, probas)

    # Calculate ROC curve
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probas)

    result = {
        "auc": auc,
        "roc_curve": (false_positive_rate, true_positive_rate, _),
        "probabilities": probas,
        "test_labels": y_test,
    }

    return result


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


def save_features(features: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    """Save computed features and labels to disk."""
    features_path = output_dir / "cached_features.npy"
    labels_path = output_dir / "cached_labels.npy"
    np.save(features_path, features)
    np.save(labels_path, labels)
    logger.info(f"Saved features and labels to {output_dir}")


def load_features(output_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load cached features and labels from disk."""
    features_path = output_dir / "cached_features.npy"
    labels_path = output_dir / "cached_labels.npy"
    features = np.load(features_path)
    labels = np.load(labels_path)
    logger.info(f"Loaded cached features and labels from {output_dir}")
    return features, labels


def split_data(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train, validation, and test sets.

    Args:
        features: Feature matrix
        labels: Target labels
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        random_state: Random state for reproducibility

    Returns:
        Tuple of (x_train, y_train, x_val, y_val, x_test, y_test)
    """
    # First split off test set
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    # Then split remaining data into train and validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_val,
    )
    return x_train, y_train, x_val, y_val, x_test, y_test


def initialize_calibrator(
    dataset: CalibrationDataset,
    output_dir: Path,
    seed: int,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize calibrator, compute features, and handle caching.

    Args:
        dataset: The calibration dataset
        output_dir: Directory to save cached features
        seed: Random seed for reproducibility
        force_recompute: Whether to force recomputation of features

    Returns:
        Tuple of (features, labels)
    """
    # Check for cached features
    features_path = output_dir / "cached_features.npy"
    labels_path = output_dir / "cached_labels.npy"

    if not force_recompute and features_path.exists() and labels_path.exists():
        logger.info("Found cached features, loading...")
        return load_features(output_dir)

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
    calibrator.fit(dataset)
    features, labels = calibrator.compute_features(dataset=dataset, labelled=True)  # type: ignore[misc]

    # Save computed features
    logger.info("Saving computed features for future use.")
    save_features(features, labels, output_dir)

    return features, labels


def main(
    spectrum_path: Annotated[
        Path, typer.Option(help="Path to the spectrum data file.")
    ],
    beam_predictions_path: Annotated[
        Path, typer.Option(help="Path to the beam predictions CSV file.")
    ],
    output_dir: Annotated[
        Path, typer.Option(help="Directory to save comparison plots.")
    ],
    force_recompute: Annotated[
        bool,
        typer.Option(
            help="Force recomputation of features even if cached version exists."
        ),
    ] = False,
) -> None:
    """Compare different classifiers for probability rescoring."""
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and filter dataset
    logger.info("Loading and filtering dataset.")
    dataset = CalibrationDataset.from_predictions_csv(
        spectrum_path=spectrum_path, beam_predictions_path=beam_predictions_path
    )
    dataset = filter_dataset(dataset)

    # Initialize calibrator and get features
    features, labels = initialize_calibrator(
        dataset=dataset,
        output_dir=output_dir,
        seed=SEED,
        force_recompute=force_recompute,
    )

    # Split data into train/val/test once
    logger.info("Splitting data into train/val/test sets...")
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(
        features, labels, test_size=0.2, val_size=0.2, random_state=SEED
    )
    logger.info(
        f"Data split sizes - Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}"
    )

    # Evaluate each classifier
    results = {}
    for name, classifier in CLASSIFIERS.items():
        logger.info(f"Evaluating {name}...")
        results[name] = evaluate_classifier(
            classifier,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
        )
        logger.info(f"{name} - AUC: {results[name]['auc']:.3f}")

    # Print top three classifiers
    print("\nTop three classifiers by AUC:")
    print("-" * 40)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["auc"], reverse=True)
    for i, (name, result) in enumerate(sorted_results[:3], 1):
        print(f"{i}. {name}: AUC = {result['auc']:.3f}")
    print("-" * 40 + "\n")

    # Generate plots
    logger.info("Generating comparison plots...")
    plot_roc_curves(results, output_dir)
    plot_probability_distributions(results, output_dir)

    # Save results to CSV - create DataFrame from AUC values only
    results_df = pd.DataFrame(
        {
            "Classifier": list(results.keys()),
            "AUC": [result["auc"] for result in results.values()],
        }
    ).set_index("Classifier")
    results_df.to_csv(output_dir / "classifier_comparison.csv")

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    typer.run(main)
