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

# Suppress SHAP's internal logging
logging.getLogger("shap").setLevel(logging.WARNING)

# Constants
SEED = 42
DEFAULT_TRAIN_SPECTRUM_PATH = Path("")
DEFAULT_TRAIN_PREDICTIONS_PATH = Path("")
DEFAULT_OUTPUT_DIR = Path("")

# Example hyperparameters - you can modify these
HYPERPARAMETERS = {
    "hidden_layer_sizes": (50, 50),
    "learning_rate_init": 0.001,
    "alpha": 0.0001,
    "max_iter": 1000,
    "random_state": SEED,
}


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


def plot_shap_summary(
    features: pd.DataFrame,
    shap_values: np.ndarray,
    output_path: Path,
) -> None:
    """Plot SHAP summary plot showing feature importance and impact.

    Args:
        features: DataFrame containing feature values
        shap_values: SHAP values for each feature
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        features,
        plot_type="bar",
        show=False,
    )
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_shap_dependence_with_interactions(
    features: pd.DataFrame,
    shap_values: np.ndarray,
    feature_name: str,
    output_path: Path,
) -> None:
    """Plot SHAP dependence plot for a specific feature.

    Args:
        features: DataFrame containing feature values
        shap_values: SHAP values for each feature
        feature_name: Name of the feature to plot
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature_name,
        shap_values,
        features,
        show=False,
    )
    plt.title(f"SHAP Dependence Plot for {feature_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_shap_summary_with_values(
    features: pd.DataFrame,
    shap_values: np.ndarray,
    output_path: Path,
) -> None:
    """Plot SHAP summary plot showing feature values and their impact.

    This plot shows:
    - Each point is a sample
    - X-axis is the SHAP value (impact on prediction)
    - Y-axis shows features ordered by importance
    - Color represents the actual feature value
    - The width of the distribution shows how many samples have that SHAP value

    Args:
        features: DataFrame containing feature values
        shap_values: SHAP values for each feature
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))

    # Create the summary plot
    shap.summary_plot(
        shap_values,
        features,
        plot_type="dot",  # Use dots instead of violin plot
        show=False,
        plot_size=(12, 8),
        color_bar_label="Feature value",
        title="SHAP Feature Impact (colored by feature value)",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_shap_dependence_no_interaction(
    features: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: list,
    output_dir: Path,
) -> None:
    """Plot SHAP dependence plots for each feature without interaction coloring.

    Args:
        features: DataFrame containing feature values
        shap_values: SHAP values for each feature
        feature_names: List of feature names
        output_dir: Path to save the plots
    """
    for feature_name in feature_names:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_name,
            shap_values,
            features,
            interaction_index=None,
            show=False,
        )
        plt.title(f"SHAP Dependence Plot for {feature_name}")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"shap_dependence_{feature_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


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
            min=100,
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
    dataset = CalibrationDataset.from_predictions_csv(
        spectrum_path=train_spectrum_path,
        beam_predictions_path=train_predictions_path,
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
    features_df = pd.DataFrame(features, columns=feature_names)

    # Scale features for MLP
    logger.info("Scaling features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled_df = pd.DataFrame(features_scaled, columns=feature_names)

    # Train classifier
    logger.info("Training classifier...")
    classifier = MLPClassifier(**HYPERPARAMETERS)
    classifier.fit(features_scaled, labels)

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
    perm_importance_dict = dict(zip(feature_names, perm_importance.importances_mean))
    plot_feature_importance(
        perm_importance_dict,
        "Permutation Feature Importance",
        output_dir / "permutation_importance.png",
    )

    # 2. SHAP values
    logger.info("Computing SHAP values...")
    # Use kmeans to summarize background data
    background = shap.kmeans(
        features_scaled,
        min(n_background_samples, len(features_scaled)),
    )

    # Use KernelExplainer for MLP
    explainer = shap.KernelExplainer(
        model=classifier.predict_proba,
        data=background,
    )

    # Compute SHAP values in batches to manage memory
    batch_size = 1000
    n_samples = len(features_scaled)
    n_batches = (n_samples + batch_size - 1) // batch_size

    all_shap_values = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        logger.info(f"Computing SHAP values for batch {i + 1}/{n_batches}")
        batch_shap = explainer.shap_values(features_scaled[start_idx:end_idx])[1]
        all_shap_values.append(batch_shap)

    shap_values = np.vstack(all_shap_values)

    # Plot SHAP summary
    plot_shap_summary(
        features_scaled_df,
        shap_values,
        output_dir / "shap_summary.png",
    )

    # Add new SHAP summary plot with feature value coloring
    plot_shap_summary_with_values(
        features_scaled_df,
        shap_values,
        output_dir / "shap_summary_with_values.png",
    )

    # Plot SHAP dependence plots for top 3 features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[-3:][::-1]
    for idx in top_features_idx:
        feature_name = feature_names[idx]
        plot_shap_dependence_with_interactions(
            features_scaled_df,
            shap_values,
            feature_name,
            output_dir / f"shap_dependence_{feature_name}_interactions.png",
        )

    # Plot SHAP dependence plots for all features without interaction coloring
    plot_shap_dependence_no_interaction(
        features_scaled_df,
        shap_values,
        feature_names,
        output_dir,
    )

    # 3. Feature correlations
    logger.info("Computing feature correlations...")
    plot_feature_correlations(
        features_scaled_df,
        output_dir / "feature_correlations.png",
    )

    # Save numerical results
    results = pd.DataFrame(
        {
            "Feature": feature_names,
            "Permutation Importance": [perm_importance_dict[f] for f in feature_names],
            "Mean |SHAP| (logits)": np.abs(shap_values).mean(axis=0),
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

    print("\nTop 5 features by mean |SHAP| (logits):")
    mean_abs_shap_dict = dict(zip(feature_names, np.abs(shap_values).mean(axis=0)))
    for feature, importance in sorted(
        mean_abs_shap_dict.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"{feature}: {importance:.4f}")

    print("\nStrong feature correlations (|r| > 0.7):")
    corr_matrix = features_df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                print(
                    f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}"
                )

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    typer.run(main)
