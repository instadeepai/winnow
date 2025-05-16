"""Script to compare different classifiers for probability rescoring."""

from pathlib import Path
from typing import Dict, Any, Tuple

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
from nets import HardSigmoidClassifier, HingeClassifier

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
    "HardSigmoidNet": HardSigmoidClassifier(
        hidden_size=100, learning_rate=0.001, batch_size=128, n_epochs=100
    ),
    "HingeNet": HingeClassifier(
        hidden_size=100, learning_rate=0.001, batch_size=128, n_epochs=100
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

    # Also create a combined plot for comparison
    plt.figure(figsize=(12, 8))
    sns.histplot(
        data=df, x="Probability", hue="Classifier", multiple="layer", alpha=0.5, bins=50
    )
    plt.title("Probability Distributions Across All Classifiers")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "probability_distributions_combined.png", dpi=300)
    plt.close()


def evaluate_classifier(
    classifier: BaseEstimator, features: np.ndarray, labels: np.ndarray
) -> Dict[str, Any]:
    """Evaluate a classifier using a single train/test split.

    Args:
        classifier: The classifier to evaluate
        features: The feature matrix
        labels: The target labels

    Returns:
        Dictionary containing evaluation metrics
    """
    # Split data into train/test
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    # Fit classifier on training data
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

    # Add training history for neural networks
    if hasattr(classifier, "get_training_history"):
        result["training_history"] = classifier.get_training_history()

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

    # Check for cached features
    features_path = output_dir / "cached_features.npy"
    labels_path = output_dir / "cached_labels.npy"

    if not force_recompute and features_path.exists() and labels_path.exists():
        logger.info("Found cached features, loading...")
        features, labels = load_features(output_dir)
    else:
        # Load and filter dataset
        logger.info("Loading and filtering dataset.")
        dataset = CalibrationDataset.from_predictions_csv(
            spectrum_path=spectrum_path, beam_predictions_path=beam_predictions_path
        )
        dataset = filter_dataset(dataset)

        # Initialize calibrator with features
        logger.info("Initializing calibrator.")
        calibrator = ProbabilityCalibrator(seed=SEED)
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

    # Evaluate each classifier
    results = {}
    for name, classifier in CLASSIFIERS.items():
        logger.info(f"Evaluating {name}...")
        results[name] = evaluate_classifier(classifier, features, labels)
        logger.info(f"{name} - AUC: {results[name]['auc']:.3f}")

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
