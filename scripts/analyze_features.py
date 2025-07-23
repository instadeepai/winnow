"""Script to analyze feature importance and correlations for the calibrator.

This script provides comprehensive analysis of feature importance for the calibrator,
including:
- Permutation importance (robust to correlations)
- SHAP values for detailed feature impact analysis
- Feature correlation analysis
- Visualization of results
"""

from pathlib import Path
from typing import Dict, Tuple, Annotated
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import shap
import typer
from typer import Typer
from rich.console import Console
from rich.theme import Theme

from winnow.datasets.calibration_dataset import CalibrationDataset, RESIDUE_MASSES
from winnow.datasets.data_loaders import InstaNovoDatasetLoader
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

# Suppress SHAP's internal logging
logging.getLogger("shap").setLevel(logging.WARNING)

# Constants
SEED = 42
DEFAULT_TRAIN_SPECTRUM_PATH = Path("")
DEFAULT_TRAIN_PREDICTIONS_PATH = Path("")
DEFAULT_OUTPUT_DIR = Path("")

COLUMN_MAPPING = {
    "confidence": "Raw Confidence",
    "Mass Error": "Mass Error",
    "ion_matches": "Ion Matches",
    "ion_match_intensity": "Ion Match Intensity",
    "chimeric_ion_matches": "Chimeric Ion Matches",
    "chimeric_ion_match_intensity": "Chimeric Ion Match Intensity",
    "iRT error": "iRT Error",
    "margin": "Margin",
    "median_margin": "Median Margin",
    "entropy": "Entropy",
    "z-score": "Z-Score",
}

# Example hyperparameters - you can modify these
HYPERPARAMETERS = {
    "hidden_layer_sizes": (50, 50),
    "learning_rate_init": 0.001,
    "alpha": 0.0001,
    "max_iter": 1000,
    "random_state": SEED,
}

# Set up custom error theme
error_theme = Theme(
    {
        "error": "red bold",
        "error_highlight": "red bold underline",
    }
)

# Create console with custom theme
console = Console(theme=error_theme)

# Create app with custom error handler
app = Typer(
    add_completion=False,
    pretty_exceptions_show_locals=False,  # Don't show local variables in traceback
)


def custom_exception_handler(exc: Exception) -> None:
    """Custom exception handler for Typer that prints concise error messages."""
    if isinstance(exc, typer.Exit):
        # Don't print anything for normal exits
        return
    elif isinstance(exc, typer.BadParameter):
        # For parameter errors, just show the error message
        console.print(f"[error]Error:[/error] {exc.message}")
    else:
        # For other errors, show a concise message
        console.print(f"[error]Error:[/error] {str(exc)}")
    raise typer.Exit(1)


# Set the custom exception handler
app.exception_handler = custom_exception_handler


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


def plot_feature_importance(
    importance_scores: Dict[str, float],
    title: str,
    output_path: Path,
) -> None:
    """Plot feature importance scores.

    Args:
        importance_scores: Dictionary mapping feature names to importance scores
        title: Plot title
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    features = list(importance_scores.keys())
    scores = list(importance_scores.values())

    # Sort by importance
    sorted_idx = np.argsort(scores)
    features = [features[i] for i in sorted_idx]
    scores = [scores[i] for i in sorted_idx]

    plt.barh(range(len(features)), scores)
    plt.yticks(range(len(features)), features)
    plt.xlabel("Importance Score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_correlations(
    features: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot feature correlation matrix.

    Args:
        features: DataFrame containing feature values
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    corr_matrix = features.corr()

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Plot correlation matrix
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.5},
    )
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


@app.command()  # Register main as a command
def main(
    train_spectrum_path: Annotated[
        Path,
        typer.Option(help="Path to training spectrum data file."),
    ],
    train_predictions_path: Annotated[
        Path,
        typer.Option(help="Path to training beam predictions file."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory to save analysis results and plots."),
    ],
    n_background_samples: Annotated[
        int,
        typer.Option(
            help="Number of background samples to use for SHAP value computation.",
            min=1,
            max=10000,
        ),
    ] = 1000,
) -> None:
    """Analyze feature importance and correlations for the calibrator.

    This script performs a comprehensive analysis of feature importance for the calibrator,
    including permutation importance, SHAP values, and feature correlations.

    Args:
        train_spectrum_path: Path to training spectrum data file
        train_predictions_path: Path to training beam predictions file
        output_dir: Directory to save results and plots
        n_background_samples: Number of background samples for SHAP computation
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info("Loading dataset...")
    dataset = InstaNovoDatasetLoader().load(
        Path(train_spectrum_path),
        Path(train_predictions_path),
    )

    dataset = filter_dataset(dataset)

    # Compute features
    logger.info("Computing features...")
    features, labels = initialize_calibrator(dataset)

    # Convert features to DataFrame for easier analysis
    feature_names = [
        "confidence",
        "Mass Error",
        "ion_matches",
        "ion_match_intensity",
        "chimeric_ion_matches",
        "chimeric_ion_match_intensity",
        "iRT error",
        "margin",
        "median_margin",
        "entropy",
        "z-score",
    ]

    # Apply column mapping for display names
    display_feature_names = [COLUMN_MAPPING.get(name, name) for name in feature_names]

    # Scale features for MLP
    logger.info("Scaling features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train classifier
    logger.info("Training classifier...")
    classifier = MLPClassifier(**HYPERPARAMETERS)
    classifier.fit(features_scaled, labels)

    # Log class mapping
    logger.info(
        f"Class mapping: {dict(zip(classifier.classes_, range(len(classifier.classes_))))}"
    )
    correct_class_idx = np.where(classifier.classes_ == 1)[0][
        0
    ]  # Get index of class 1 (correct)
    logger.info(
        f"Using SHAP values for class index {correct_class_idx} (correct predictions)"
    )

    # 1. Permutation importance
    logger.info("Computing permutation importance...")
    perm_importance = permutation_importance(
        classifier,
        features_scaled,
        labels,
        n_repeats=10,
        random_state=SEED,
        n_jobs=-1,
    )
    # Use display names for plotting
    perm_importance_dict = dict(
        zip(display_feature_names, perm_importance.importances_mean)
    )
    plot_feature_importance(
        perm_importance_dict,
        "Permutation Feature Importance",
        output_dir / "permutation_importance.png",
    )

    # 2. SHAP values
    logger.info("Computing SHAP values...")
    # Sample background data
    background = shap.sample(
        features_scaled,
        min(n_background_samples, len(features_scaled)),
        random_state=SEED,
    )

    # Use KernelExplainer for MLP
    explainer = shap.KernelExplainer(
        model=classifier.predict_proba,
        data=background,
        seed=SEED,
        link="identity",  # Explain probabilities directly
    )

    # Randomly sample 10,000 training spectra from the dataset
    np.random.seed(SEED)
    indices = np.random.choice(features_scaled.shape[0], size=10000, replace=False)

    # Get SHAP values from explainer for a random subset of the training data
    shap_values = explainer(features_scaled[indices])

    # Switch to original feature space for visualization
    # This is safe because standardization is a univariate transformation
    shap_values.data = features[indices]
    shap_values.feature_names = display_feature_names

    # Plot SHAP summary using built-in function
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(
        shap_values[:, :, correct_class_idx], show=False, max_display=12
    )  # for correct class
    plt.title("SHAP Feature Impact on P(correct) (colored by feature value)")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot SHAP bar plot
    plt.figure(figsize=(10, 8))
    clustering = shap.utils.hclust(features_scaled, labels)
    shap.plots.bar(
        shap_values[:, :, correct_class_idx],
        clustering=clustering,
        show=False,
        clustering_cutoff=0.5,
        max_display=12,
    )  # for correct class
    plt.title("SHAP Feature Importance for P(correct)")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot SHAP dependence plots for top 3 features
    mean_abs_shap = np.abs(shap_values.values[:, :, correct_class_idx]).mean(
        axis=0
    )  # for correct class
    top_features_idx = np.argsort(mean_abs_shap)[-3:][::-1]
    for idx in top_features_idx:
        feature_name = display_feature_names[idx]
        original_feature_name = feature_names[idx]  # For filename
        plt.figure(figsize=(10, 6))
        shap.plots.scatter(
            shap_values[:, idx, correct_class_idx],
            show=False,
            color=plt.cm.viridis(0.3),
        )
        plt.title(f"SHAP Dependence Plot for {feature_name} (impact on P(correct))")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"shap_dependence_{original_feature_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Plot SHAP interaction plots for all combinations of top 3 features
    for i, idx1 in enumerate(top_features_idx):
        feature1_display = display_feature_names[idx1]
        feature1_original = feature_names[idx1]
        for j, idx2 in enumerate(top_features_idx):
            if i != j:  # Skip self-interactions
                feature2_display = display_feature_names[idx2]
                feature2_original = feature_names[idx2]
                plt.figure(figsize=(12, 8))
                shap.plots.scatter(
                    shap_values[:, feature1_display, correct_class_idx],
                    color=shap_values[
                        :, feature2_display, correct_class_idx
                    ],  # Use second feature for coloring
                    show=False,
                )
                plt.title(
                    f"SHAP Interaction Plot for {feature1_display} vs {feature2_display} (impact on P(correct))"
                )
                plt.tight_layout()
                plt.savefig(
                    output_dir
                    / f"shap_interaction_{feature1_original}_vs_{feature2_original}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

    # Plot SHAP heatmap for top 10 features
    plt.figure(figsize=(12, 8))
    shap.plots.heatmap(
        shap_values[:, :, correct_class_idx],  # for correct class
        max_display=12,
        show=False,
    )
    plt.title("SHAP Feature Impact Heatmap (impact on P(correct))")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Feature correlations
    logger.info("Computing feature correlations...")
    # Create a DataFrame with display names for correlation plotting
    features_scaled_display_df = pd.DataFrame(
        features_scaled, columns=display_feature_names
    )
    plot_feature_correlations(
        features_scaled_display_df,
        output_dir / "feature_correlations.png",
    )

    # Save numerical results
    results = pd.DataFrame(
        {
            "Feature": display_feature_names,
            "Permutation Importance": [
                perm_importance_dict[f] for f in display_feature_names
            ],
            "Mean |SHAP| (impact on P(correct))": np.abs(
                shap_values.values[:, :, correct_class_idx]
            ).mean(axis=0),  # for correct class
        }
    )
    results.to_csv(output_dir / "feature_importance_results.csv", index=False)

    # Print summary
    print("\nFeature Importance Analysis Results:")
    print("-" * 60)
    print("\nTop 5 features by permutation importance:")
    for feature, importance in sorted(
        perm_importance_dict.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"{feature}: {importance:.4f}")

    print("\nTop 5 features by mean |SHAP| (impact on P(correct)):")
    mean_abs_shap_dict = dict(
        zip(
            display_feature_names,
            np.abs(shap_values.values[:, :, correct_class_idx]).mean(axis=0),
        )
    )  # for correct class
    for feature, importance in sorted(
        mean_abs_shap_dict.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"{feature}: {importance:.4f}")

    print("\nStrong feature correlations (|r| > 0.7):")
    corr_matrix = features_scaled_display_df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                print(
                    f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}"
                )

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    app()
