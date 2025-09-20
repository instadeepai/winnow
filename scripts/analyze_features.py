"""Script to analyze feature importance and correlations for a pretrained calibrator.

This script provides comprehensive analysis of feature importance for a pretrained calibrator:
    - Permutation importance on test set
    - SHAP values with training background on test set
    - Feature correlation analysis on training data
    - Optional visualization of results
"""

from pathlib import Path
from typing import Dict, Annotated
import logging
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from huggingface_hub import snapshot_download
import shap
import typer
from typer import Typer
from rich.console import Console
from rich.theme import Theme
from matplotlib.colors import LinearSegmentedColormap
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.datasets.data_loaders import InstaNovoDatasetLoader
from winnow.calibration.calibrator import ProbabilityCalibrator

# Color scheme from the plotting script
COLORS = {
    "fairy": "#FFCAE9",
    "magenta": "#8E5572",
    "ash": "#BBC5AA",
    "ebony": "#5A6650",
    "sky": "#7FC8F8",
    "navy": "#3C81AE",
}

sns.set_theme(style="white", palette="colorblind", context="paper", font_scale=1.5)


def download_dataset(repo_id: str, local_dir: str, pattern: list[str]) -> None:
    """Download the dataset from the Hugging Face Hub."""
    logger.info(f"Downloading dataset {repo_id} to {local_dir}.")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=pattern,
        repo_type="dataset",
    )


def download_model(repo_id: str, local_dir: str) -> None:
    """Download the model from the Hugging Face Hub."""
    logger.info(f"Downloading model {repo_id} to {local_dir}.")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        repo_type="model",
        allow_patterns="*.pkl",
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


def to_sentence_case(feature_name: str) -> str:
    """Convert feature name to lowercase for use in sentence context (after 'for')."""
    return feature_name.lower()


# Set up logging
logger = logging.getLogger("winnow")
logger.setLevel(logging.INFO)

# Suppress SHAP's internal logging
logging.getLogger("shap").setLevel(logging.WARNING)

# Constants
SEED = 42

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
    plt.xlabel("Importance score")
    plt.title(title)

    # Save in both formats
    plt.savefig(f"{output_path_base}.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{output_path_base}.png", bbox_inches="tight", dpi=300)
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
    plt.savefig(f"{output_path_base}.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{output_path_base}.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_shap_summary(
    shap_values,
    correct_class_idx: int,
    output_dir: Path,
) -> None:
    """Plot SHAP summary (beeswarm) plot.

    Args:
        shap_values: SHAP values object
        correct_class_idx: Index of the correct class
        output_dir: Directory to save plots
    """
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
    plt.savefig(output_dir / "shap_summary.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(output_dir / "shap_summary.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_shap_bar(
    shap_values,
    test_features_scaled,
    test_labels,
    correct_class_idx: int,
    output_dir: Path,
) -> None:
    """Plot SHAP bar plot.

    Args:
        shap_values: SHAP values object
        test_features_scaled: Scaled test features for clustering
        test_labels: Test labels for clustering
        correct_class_idx: Index of the correct class
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(8, 6))
    clustering = shap.utils.hclust(test_features_scaled, test_labels)
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
    plt.savefig(output_dir / "shap_importance.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(output_dir / "shap_importance.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_shap_dependence(
    shap_values,
    feature_names: list,
    display_feature_names: list,
    correct_class_idx: int,
    output_dir: Path,
    top_n: int = 3,
) -> None:
    """Plot SHAP dependence plots for top features.

    Args:
        shap_values: SHAP values object
        feature_names: Original feature names (for filenames)
        display_feature_names: Pretty feature names (for display)
        correct_class_idx: Index of the correct class
        output_dir: Directory to save plots
        top_n: Number of top features to plot
    """
    mean_abs_shap = np.abs(shap_values.values[:, :, correct_class_idx]).mean(
        axis=0
    )  # for correct class
    top_features_idx = np.argsort(mean_abs_shap)[-top_n:][::-1]

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
            + to_sentence_case(feature_name)
            + "\n"
            + r"(impact on $P(\text{correct})$)"
        )

        # Fix y-axis label to use sentence case
        ax = plt.gca()
        current_ylabel = ax.get_ylabel()
        if "SHAP value for" in current_ylabel:
            # Replace the feature name in the y-label with sentence case version
            new_ylabel = current_ylabel.replace(
                feature_name, to_sentence_case(feature_name)
            )
            ax.set_ylabel(new_ylabel)

        # Save in both formats
        plt.savefig(
            output_dir / f"shap_dependence_{original_feature_name}.pdf",
            bbox_inches="tight",
            dpi=300,
        )
        plt.savefig(
            output_dir / f"shap_dependence_{original_feature_name}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()


def plot_shap_interactions(
    shap_values,
    feature_names: list,
    display_feature_names: list,
    correct_class_idx: int,
    output_dir: Path,
    top_n: int = 3,
) -> None:
    """Plot SHAP interaction plots for top features.

    Args:
        shap_values: SHAP values object
        feature_names: Original feature names (for filenames)
        display_feature_names: Pretty feature names (for display)
        correct_class_idx: Index of the correct class
        output_dir: Directory to save plots
        top_n: Number of top features to use for interactions
    """
    mean_abs_shap = np.abs(shap_values.values[:, :, correct_class_idx]).mean(
        axis=0
    )  # for correct class
    top_features_idx = np.argsort(mean_abs_shap)[-top_n:][::-1]

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
                    + to_sentence_case(feature1_display)
                    + " vs "
                    + to_sentence_case(feature2_display)
                    + "\n"
                    + r" (impact on $P(\text{correct})$)"
                )

                # Fix y-axis label to use sentence case
                ax = plt.gca()
                current_ylabel = ax.get_ylabel()
                if "SHAP value for" in current_ylabel:
                    # Replace the feature name in the y-label with sentence case version
                    new_ylabel = current_ylabel.replace(
                        feature1_display, to_sentence_case(feature1_display)
                    )
                    ax.set_ylabel(new_ylabel)

                # Save in both formats
                plt.savefig(
                    output_dir
                    / f"shap_interaction_{feature1_original}_vs_{feature2_original}.pdf",
                    bbox_inches="tight",
                    dpi=300,
                )
                plt.savefig(
                    output_dir
                    / f"shap_interaction_{feature1_original}_vs_{feature2_original}.png",
                    bbox_inches="tight",
                    dpi=300,
                )
                plt.close()


def plot_shap_heatmap(
    shap_values,
    correct_class_idx: int,
    output_dir: Path,
) -> None:
    """Plot SHAP heatmap.

    Args:
        shap_values: SHAP values object
        correct_class_idx: Index of the correct class
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(8, 6))
    custom_cmap = create_custom_colormap()
    shap.plots.heatmap(
        shap_values[:, :, correct_class_idx],  # for correct class
        max_display=12,
        show=False,
        cmap=custom_cmap,
    )
    plt.title("SHAP feature impact heatmap\n" + r"(impact on $P(\text{correct})$)")

    # Save in both formats
    plt.savefig(output_dir / "shap_heatmap.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(output_dir / "shap_heatmap.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_all_plots(
    perm_importance_dict: Dict[str, float],
    shap_values,
    train_features_scaled_display_df: pd.DataFrame,
    test_features_scaled,
    test_labels,
    feature_names: list,
    display_feature_names: list,
    correct_class_idx: int,
    output_dir: Path,
) -> None:
    """Create all plots for the analysis.

    Args:
        perm_importance_dict: Permutation importance scores
        shap_values: SHAP values object
        train_features_scaled_display_df: Training features for correlation
        test_features_scaled: Test features for clustering
        test_labels: Test labels for clustering
        feature_names: Original feature names
        display_feature_names: Pretty feature names
        correct_class_idx: Index of the correct class
        output_dir: Directory to save plots
    """
    logger.info("Creating plots...")

    # 1. Permutation importance plot
    plot_feature_importance(
        perm_importance_dict,
        "Permutation feature importance",
        output_dir / "permutation_importance",
    )

    # 2. SHAP plots
    plot_shap_summary(shap_values, correct_class_idx, output_dir)
    plot_shap_bar(
        shap_values, test_features_scaled, test_labels, correct_class_idx, output_dir
    )
    plot_shap_dependence(
        shap_values, feature_names, display_feature_names, correct_class_idx, output_dir
    )
    plot_shap_interactions(
        shap_values, feature_names, display_feature_names, correct_class_idx, output_dir
    )
    plot_shap_heatmap(shap_values, correct_class_idx, output_dir)

    # 3. Feature correlations
    plot_feature_correlations(
        train_features_scaled_display_df,
        output_dir / "feature_correlations",
    )


@app.command()
def main(
    model_path: Annotated[
        Path,
        typer.Option(help="Path to pretrained calibrator model directory."),
    ],
    data_dir: Annotated[
        Path,
        typer.Option(help="Directory to save training and test datasets."),
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
    ] = 500,
    n_test_samples: Annotated[
        int,
        typer.Option(
            help="Number of test samples to use for SHAP value computation.",
            min=1,
            max=10000,
        ),
    ] = 1000,
    create_plots: Annotated[
        bool,
        typer.Option(
            "--create-plots/--no-plots",
            help="Whether to create plots or just compute and save analysis objects.",
        ),
    ] = True,
) -> None:
    """Analyze feature importance and correlations for a pretrained calibrator.

    Args:
        model_path: Path to pretrained calibrator model directory
        data_dir: Directory to save training and test datasets
        output_dir: Directory to save results and plots
        n_background_samples: Number of background samples for SHAP computation
        n_test_samples: Number of test samples to use for SHAP value computation
        create_plots: Whether to create plots or just compute and save analysis objects
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download the dataset from the Hugging Face Hub
    logger.info("Downloading dataset from the Hugging Face Hub...")
    download_dataset(
        repo_id="instadeepai/winnow-ms-datasets",
        local_dir=str(data_dir),
        pattern=["general_train*", "general_test*"],
    )

    # Download the model from the Hugging Face Hub
    logger.info("Downloading model from the Hugging Face Hub...")
    download_model(
        repo_id="instadeepai/winnow-general-model",
        local_dir=str(model_path),
    )

    # Load pretrained calibrator
    logger.info("Loading pretrained calibrator...")
    calibrator = ProbabilityCalibrator.load(model_path)

    # Load training dataset
    logger.info("Loading training dataset...")
    train_dataset = InstaNovoDatasetLoader().load(
        data_dir / "general_train.parquet",
        data_dir / "general_train_beams.csv",
    )
    train_dataset = filter_dataset(train_dataset)

    # Load test dataset
    logger.info("Loading test dataset...")
    test_dataset = InstaNovoDatasetLoader().load(
        data_dir / "general_test.parquet",
        data_dir / "general_test_beams.csv",
    )
    test_dataset = filter_dataset(test_dataset)

    # Compute features for both datasets
    logger.info("Computing features for training set...")
    train_features, train_labels = calibrator.compute_features(
        dataset=train_dataset, labelled=True
    )

    logger.info("Computing features for test set...")
    # Note that we do not pass labelled=True to avoid training the iRT predictor
    test_features = calibrator.compute_features(dataset=test_dataset, labelled=False)
    test_labels = test_dataset.metadata["correct"]

    # Apply column mapping for display names
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
    display_feature_names = [COLUMN_MAPPING.get(name, name) for name in feature_names]

    # Scale features using the calibrator's fitted scaler
    logger.info("Scaling features with pretrained scaler...")
    train_features_scaled = calibrator.scaler.transform(train_features)
    test_features_scaled = calibrator.scaler.transform(test_features)

    # Use the pretrained classifier
    logger.info("Using pretrained classifier...")
    classifier = calibrator.classifier

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

    # 1. Permutation importance on test set
    logger.info("Computing permutation importance on test set...")
    perm_importance = permutation_importance(
        classifier,
        test_features_scaled,
        test_labels,
        n_repeats=10,
        random_state=SEED,
        n_jobs=-1,
    )
    # Use display names for plotting
    perm_importance_dict = dict(
        zip(display_feature_names, perm_importance.importances_mean)
    )

    # 2. SHAP values with training background on test data
    logger.info("Computing SHAP values with training background on test data...")
    # Sample background data from training set
    background = shap.sample(
        train_features_scaled,
        min(n_background_samples, len(train_features_scaled)),
        random_state=SEED,
    )

    # Use KernelExplainer for MLP
    explainer = shap.KernelExplainer(
        model=classifier.predict_proba,
        data=background,
        seed=SEED,
        link="identity",  # Explain probabilities directly
    )

    # Randomly sample test data for SHAP computation
    np.random.seed(SEED)
    n_samples = min(n_test_samples, test_features_scaled.shape[0])
    indices = np.random.choice(
        test_features_scaled.shape[0], size=n_samples, replace=False
    )

    # Get SHAP values from explainer for a random subset of the test data
    shap_values = explainer(test_features_scaled[indices])

    # Switch to original feature space for visualization
    # This is safe because standardization is a univariate transformation
    shap_values.data = test_features[indices]
    shap_values.feature_names = display_feature_names

    # 3. Feature correlations on training data
    logger.info("Computing feature correlations on training data...")
    # Create a DataFrame with display names for correlation plotting
    train_features_scaled_display_df = pd.DataFrame(
        train_features_scaled, columns=display_feature_names
    )

    # Create plots if requested
    if create_plots:
        create_all_plots(
            perm_importance_dict=perm_importance_dict,
            shap_values=shap_values,
            train_features_scaled_display_df=train_features_scaled_display_df,
            test_features_scaled=test_features_scaled,
            test_labels=test_labels,
            feature_names=feature_names,
            display_feature_names=display_feature_names,
            correct_class_idx=correct_class_idx,
            output_dir=output_dir,
        )

    # Save raw objects for later analysis
    logger.info("Saving raw analysis objects...")

    # Save permutation importance object
    with open(output_dir / "perm_importance.pkl", "wb") as f:
        pickle.dump(perm_importance, f)

    # Save SHAP values object
    with open(output_dir / "shap_values.pkl", "wb") as f:
        pickle.dump(shap_values, f)

    logger.info("Analysis complete!")
    logger.info(f"Results saved to {output_dir}")
    logger.info(
        f"Permutation Feature Importance: computed on {len(test_labels)} test samples"
    )
    logger.info(
        f"SHAP values: computed on {n_samples} test samples with {len(background)} training samples as background"
    )
    logger.info(f"Correlation matrix: computed on {len(train_labels)} training samples")

    # List all saved files
    saved_files = [
        "perm_importance.pkl (raw permutation importance object)",
        "shap_values.pkl (raw SHAP values object)",
    ]
    if create_plots:
        saved_files.append("All plots in PDF and PNG formats")
    else:
        logger.info("Plots were skipped (--no-plots flag used)")

    logger.info(f"Saved files: {', '.join(saved_files)}")
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    app()
