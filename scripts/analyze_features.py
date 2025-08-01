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
from matplotlib.colors import LinearSegmentedColormap
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

# Color scheme from the plotting script
COLORS = {
    "fairy": "#FFCAE9",
    "magenta": "#8E5572",
    "ash": "#BBC5AA",
    "ebony": "#5A6650",
    "sky": "#7FC8F8",
    "navy": "#3C81AE",
}

# Configure matplotlib to match seaborn "paper" context
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        # Figure settings
        "figure.figsize": [12.0, 10.0],
        "figure.facecolor": "white",
        "figure.dpi": 100.0,
        # Axes settings
        "axes.labelcolor": ".15",
        "axes.axisbelow": True,
        "axes.grid": False,
        "axes.facecolor": "white",
        "axes.edgecolor": ".15",
        "axes.linewidth": 1.0,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
        # Tick settings
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": ".15",
        "ytick.color": ".15",
        "xtick.top": False,
        "ytick.right": False,
        "xtick.bottom": False,
        "ytick.left": False,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "xtick.major.size": 4.8,
        "ytick.major.size": 4.8,
        "xtick.minor.size": 3.2,
        "ytick.minor.size": 3.2,
        # Grid settings
        "grid.linestyle": "-",
        "grid.color": ".8",
        "grid.linewidth": 0.8,
        # Text settings
        "text.color": ".15",
        "font.family": ["sans-serif"],
        "font.sans-serif": [
            "Arial",
            "DejaVu Sans",
            "Liberation Sans",
            "Bitstream Vera Sans",
            "sans-serif",
        ],
        "font.size": 9.6,
        "axes.labelsize": 9.6,
        "axes.titlesize": 9.6,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "legend.fontsize": 8.8,
        "legend.title_fontsize": 9.6,
        # Line and patch settings
        "lines.linewidth": 1.2,
        "lines.markersize": 4.8,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0.8,
        "patch.edgecolor": "black",
        "patch.force_edgecolor": True,
        # Image settings
        "image.cmap": "rocket",
    }
)


# Create custom colormaps using the color scheme
def create_custom_colormap():
    """Create a custom colormap using the color scheme."""
    # Create a diverging colormap from navy -> ash -> fairy
    colors = [COLORS["navy"], COLORS["ash"], COLORS["fairy"]]
    return LinearSegmentedColormap.from_list("custom", colors, N=256)


def create_sequential_colormap():
    """Create a sequential colormap for single-direction plots."""
    # Create from sky to navy
    colors = [COLORS["navy"], COLORS["magenta"]]
    return LinearSegmentedColormap.from_list("custom_seq", colors, N=256)


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
    "confidence": "Raw confidence",
    "Mass Error": "Mass error",
    "ion_matches": "Ion matches",
    "ion_match_intensity": "Ion match intensity",
    "chimeric_ion_matches": "Chimeric ion matches",
    "chimeric_ion_match_intensity": "Chimeric ion match intensity",
    "iRT error": "iRT error",
    "margin": "Margin",
    "median_margin": "Median margin",
    "entropy": "Entropy",
    "z-score": "Z-score",
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
    output_path_base: Path,
) -> None:
    """Plot feature importance scores.

    Args:
        importance_scores: Dictionary mapping feature names to importance scores
        title: Plot title
        output_path_base: Base path to save the plot (without extension)
    """
    plt.figure(figsize=(8, 6))
    features = list(importance_scores.keys())
    scores = list(importance_scores.values())

    # Sort by importance
    sorted_idx = np.argsort(scores)
    features = [features[i] for i in sorted_idx]
    scores = [scores[i] for i in sorted_idx]

    # Use ash color for the bars
    plt.barh(range(len(features)), scores, color=COLORS["ash"])
    plt.yticks(range(len(features)), features)
    plt.xlabel("Importance Score")
    plt.title(title)

    # Save in both formats
    plt.savefig(f"{output_path_base}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_path_base}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_correlations(
    features: pd.DataFrame,
    output_path_base: Path,
) -> None:
    """Plot feature correlation matrix.

    Args:
        features: DataFrame containing feature values
        output_path_base: Base path to save the plot (without extension)
    """
    plt.figure(figsize=(12, 10))
    corr_matrix = features.corr()

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Use custom colormap
    custom_cmap = create_custom_colormap()

    # Plot correlation matrix
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=custom_cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.5},
    )
    plt.title("Feature correlation matrix")

    # Save in both formats
    plt.savefig(f"{output_path_base}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_path_base}.png", dpi=300, bbox_inches="tight")
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
        output_dir / "permutation_importance",
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
    plt.figure(figsize=(8, 6))
    # Set the color for SHAP beeswarm plot
    sequential_cmap = create_sequential_colormap()
    shap.plots.beeswarm(
        shap_values[:, :, correct_class_idx],
        show=False,
        max_display=12,
        color=sequential_cmap,
    )  # for correct class
    plt.title(r"SHAP feature impact on $P(\text{correct})$")

    # Save in both formats
    plt.savefig(output_dir / "shap_summary.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot SHAP bar plot
    plt.figure(figsize=(8, 6))
    clustering = shap.utils.hclust(features_scaled, labels)
    shap.plots.bar(
        shap_values[:, :, correct_class_idx],
        clustering=clustering,
        show=False,
        clustering_cutoff=0.5,
        max_display=12,
    )  # for correct class
    # Manually set bar colors after the plot is created
    ax = plt.gca()
    for patch in ax.patches:
        patch.set_facecolor(COLORS["magenta"])
    plt.title(r"SHAP feature importance for $P(\text{correct})$")

    # Save in both formats
    plt.savefig(output_dir / "shap_importance.pdf", dpi=300, bbox_inches="tight")
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
        plt.figure(figsize=(8, 6))
        shap.plots.scatter(
            shap_values[:, idx, correct_class_idx],
            show=False,
            color=COLORS["sky"],  # Use sky blue for scatter points
        )
        plt.title(
            "SHAP dependence plot for "
            + feature_name
            + r" (impact on $P(\text{correct})$)"
        )

        # Save in both formats
        plt.savefig(
            output_dir / f"shap_dependence_{original_feature_name}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
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
                plt.figure(figsize=(8, 6))
                # Create a custom colormap for the interaction plot
                interaction_cmap = create_sequential_colormap()
                shap.plots.scatter(
                    shap_values[:, feature1_display, correct_class_idx],
                    color=shap_values[
                        :, feature2_display, correct_class_idx
                    ],  # Use second feature for coloring
                    show=False,
                    cmap=interaction_cmap,
                )
                plt.title(
                    "SHAP interaction plot for "
                    + feature1_display
                    + " vs "
                    + feature2_display
                    + r" (impact on $P(\text{correct})$)"
                )

                # Save in both formats
                plt.savefig(
                    output_dir
                    / f"shap_interaction_{feature1_original}_vs_{feature2_original}.pdf",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.savefig(
                    output_dir
                    / f"shap_interaction_{feature1_original}_vs_{feature2_original}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

    # Plot SHAP heatmap for top 10 features
    plt.figure(figsize=(12, 10))
    custom_cmap = create_custom_colormap()
    shap.plots.heatmap(
        shap_values[:, :, correct_class_idx],  # for correct class
        max_display=12,
        show=False,
        cmap=custom_cmap,
    )
    plt.title(r"SHAP feature impact heatmap (impact on $P(\text{correct})$)")

    # Save in both formats
    plt.savefig(output_dir / "shap_heatmap.pdf", dpi=300, bbox_inches="tight")
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
        output_dir / "feature_correlations",
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

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    app()
