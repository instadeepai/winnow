"""Script to plot the calibrator generalisation heatmaps."""

import matplotlib.pyplot as plt
import polars as pl
import pandas as pd
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import auc
import argparse
from matplotlib.colors import LinearSegmentedColormap
import logging
from rich.logging import RichHandler

COLORS = {
    "fairy": "#FFCAE9",
    "magenta": "#8E5572",
    "ash": "#BBC5AA",
    "ebony": "#5A6650",
    "sky": "#7FC8F8",
    "navy": "#3C81AE",
}

sns.set_theme(style="white", palette="colorblind", context="paper", font_scale=2)


# --- Logging Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.addHandler(RichHandler())


def create_custom_diverging_colormap():
    """Create a custom diverging colormap using the color scheme."""
    colors = [COLORS["navy"], COLORS["ash"], COLORS["fairy"]]
    return LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)


def create_custom_sequential_colormap():
    """Create a custom sequential colormap using the color scheme."""
    colors = [COLORS["navy"], COLORS["magenta"]]
    return LinearSegmentedColormap.from_list("custom_sequential", colors, N=256)


# Species name mapping for nicer plot labels
SPECIES_NAME_MAPPING = {
    "helaqc": "HeLa single shot",
    "gluc": "HeLa degradome",
    "herceptin": "Herceptin",
    "snakevenoms": "Snake venomics",
    "woundfluids": "Wound exudates",
    "sbrodae": "Scalindua brodae",
    "immuno": "Immunopeptidomics-1",
    "PXD019483": "HepG2",
}


def compute_pr_auc(
    input_dataset: pd.DataFrame,
    confidence_column: str,
    label_column: str,
) -> float:
    """Compute Area Under Curve for precision-recall curve.

    Args:
        input_dataset: DataFrame containing confidence scores and labels
        confidence_column: Name of the column containing confidence scores
        label_column: Name of the column containing boolean labels

    Returns:
        AUC value for the precision-recall curve
    """
    if len(input_dataset) == 0:
        return 0.0

    # Sort by confidence in descending order
    sorted_data = input_dataset[[confidence_column, label_column]].sort_values(
        by=confidence_column, ascending=False
    )

    # Compute precision and recall
    cum_correct = np.cumsum(sorted_data[label_column])
    precision = cum_correct / np.arange(1, len(sorted_data) + 1)
    recall = (
        cum_correct / cum_correct.iloc[-1]
        if cum_correct.iloc[-1] > 0
        else np.zeros_like(cum_correct)
    )

    # Compute AUC using trapezoidal rule
    if len(precision) < 2:
        return 0.0

    return auc(recall, precision)


def load_and_process_results(
    results_path: str, confidence_column: str = "calibrated_confidence"
) -> pd.DataFrame:
    """Load calibrator generalisation results and compute PR-AUC matrix.

    Args:
        results_path: Path to the calibrator_generalisation_results.csv file
        confidence_column: Name of the confidence column to use for PR-AUC computation

    Returns:
        DataFrame with PR-AUC values for each trained/test dataset combination
    """
    # Load results
    results = pl.scan_csv(results_path)

    # Check required columns
    required_columns = [
        "trained_on_dataset",
        "test_dataset",
        confidence_column,
        "correct",
    ]
    missing_columns = [col for col in required_columns if col not in results.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Get unique datasets
    trained_datasets = sorted(results.select("trained_on_dataset").unique().collect())
    test_datasets = sorted(results.select("test_dataset").unique().collect())

    # Initialize PR-AUC matrix
    auc_matrix = []

    for trained_dataset in trained_datasets:
        auc_row = []
        for test_dataset in test_datasets:
            # Filter data for this combination
            subset = (
                results.filter(
                    (pl.col("trained_on_dataset") == trained_dataset)
                    & (pl.col("test_dataset") == test_dataset)
                )
                .collect()
                .to_pandas()
            )

            if len(subset) > 0:
                # Compute PR-AUC for specified confidence column
                auc_value = compute_pr_auc(subset, confidence_column, "correct")
                auc_row.append(auc_value)
            else:
                auc_row.append(np.nan)

        auc_matrix.append(auc_row)

    # Map dataset names to nicer labels
    trained_labels = [SPECIES_NAME_MAPPING.get(ds, ds) for ds in trained_datasets]
    test_labels = [SPECIES_NAME_MAPPING.get(ds, ds) for ds in test_datasets]

    # Create DataFrame
    auc_df = pd.DataFrame(auc_matrix, index=trained_labels, columns=test_labels)

    return auc_df


def create_auc_heatmap(
    auc_df: pd.DataFrame,
    output_path: str,
    title: str = "Calibrator generalisation: PR-AUC heatmap",
) -> None:
    """Create and save a heatmap of PR-AUC values.

    Args:
        auc_df: DataFrame with PR-AUC values (trained datasets as rows, test datasets as columns)
        output_path: Path to save the plot
        title: Title for the heatmap
    """
    # Set up the plot
    plt.figure(figsize=(12, 10))

    # Create custom sequential colormap
    custom_cmap = create_custom_sequential_colormap()

    # Create heatmap
    sns.heatmap(
        auc_df,
        annot=True,
        fmt=".3f",
        cmap=custom_cmap,
        cbar_kws={"label": "PR-AUC"},
        square=True,
        linewidths=0.5,
    )

    # Customize the plot
    plt.title(title)
    plt.xlabel("Test dataset")
    plt.ylabel("Train dataset")

    # Rotate labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Save the plot
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", dpi=300)
    print(f"Heatmap saved to {output_path}")


def create_comparison_heatmaps(results_path: str, output_dir: str) -> None:
    """Create heatmaps comparing raw vs calibrated confidence PR-AUC values.

    Args:
        results_path: Path to the calibrator_generalisation_results.csv file
        output_dir: Directory to save the plots
    """
    # Load results
    logger.info("Scanning results")
    results = pl.scan_csv(results_path)

    # Get unique datasets
    logger.info("Collecting unique datasets")
    trained_datasets = sorted(
        results.select(pl.col("trained_on_dataset"))
        .unique()
        .collect()
        .to_series()
        .to_list()
    )
    test_datasets = sorted(
        results.select(pl.col("test_dataset")).unique().collect().to_series().to_list()
    )
    logger.info(f"Trained datasets: {trained_datasets}")
    logger.info(f"Test datasets: {test_datasets}")

    # Map dataset names to nicer labels
    trained_labels = [SPECIES_NAME_MAPPING.get(ds, ds) for ds in trained_datasets]
    test_labels = [SPECIES_NAME_MAPPING.get(ds, ds) for ds in test_datasets]

    # Compute PR-AUC matrices for both confidence types
    logger.info("Computing PR-AUC matrices")
    auc_matrices = {}

    for conf_type in ["confidence", "calibrated_confidence"]:
        auc_matrix = []

        for trained_dataset in trained_datasets:
            auc_row = []
            for test_dataset in test_datasets:
                logger.info(
                    f"Computing PR-AUC for {trained_dataset} and {test_dataset}"
                )
                # Filter data for this combination
                subset = (
                    results.filter(
                        (pl.col("trained_on_dataset") == trained_dataset)
                        & (pl.col("test_dataset") == test_dataset)
                    )
                    .collect()
                    .to_pandas()
                )

                if len(subset) > 0:
                    # Compute PR-AUC
                    auc_value = compute_pr_auc(subset, conf_type, "correct")
                    auc_row.append(auc_value)
                else:
                    auc_row.append(np.nan)

            auc_matrix.append(auc_row)

        auc_matrices[conf_type] = pd.DataFrame(
            auc_matrix, index=trained_labels, columns=test_labels
        )

    # Create individual heatmaps
    for conf_type, auc_df in auc_matrices.items():
        conf_name = conf_type.replace("_", " ").title().lower()
        output_path = os.path.join(
            output_dir, f"calibrator_generalisation_{conf_type}_auc_heatmap.png"
        )
        create_auc_heatmap(
            auc_df, output_path, f"Calibrator generalisation: {conf_name} PR-AUC"
        )

    # Create difference heatmap (calibrated - raw)
    diff_matrix = auc_matrices["calibrated_confidence"] - auc_matrices["confidence"]

    # Create custom diverging colormap
    custom_diverging_cmap = create_custom_diverging_colormap()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        diff_matrix,
        annot=True,
        fmt=".3f",
        cmap=custom_diverging_cmap,
        center=0,
        cbar_kws={"label": r"PR-AUC difference $(\text{calibrated} - \text{raw})$"},
        square=True,
        linewidths=0.5,
    )

    plt.title("Calibrator generalisation: PR-AUC improvement")
    plt.xlabel("Test dataset")
    plt.ylabel("Train dataset")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    diff_output_path = os.path.join(
        output_dir, "calibrator_generalisation_auc_difference_heatmap.png"
    )
    plt.savefig(diff_output_path, bbox_inches="tight", dpi=300)
    plt.savefig(diff_output_path.replace(".png", ".pdf"), bbox_inches="tight", dpi=300)


def main():
    """Main function to create calibrator generalisation heatmaps."""
    parser = argparse.ArgumentParser(
        description="Create PR-AUC heatmaps for calibrator generalisation results"
    )
    parser.add_argument(
        "--results-path",
        type=str,
        required=True,
        help="Path to the calibrator generalisation results CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the plots",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if results file exists
    if not os.path.exists(args.results_path):
        raise FileNotFoundError(f"Results file not found: {args.results_path}")

    logger.info(f"Loading results from: {args.results_path}")
    logger.info(f"Saving plots to: {args.output_dir}")

    # Create comparison heatmaps (raw vs calibrated confidence)
    create_comparison_heatmaps(args.results_path, args.output_dir)


if __name__ == "__main__":
    main()
