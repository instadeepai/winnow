"""This script creates plots for the new general model results."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
import numpy as np
import ast
import os
import glob
from winnow.fdr.database_grounded import DatabaseGroundedFDRControl
from winnow.datasets.calibration_dataset import RESIDUE_MASSES
from winnow.fdr.nonparametric import NonParametricFDRControl

# Color scheme from the attached notebook
COLORS = {
    "fairy": "#FFCAE9",
    "magenta": "#8E5572",
    "ash": "#BBC5AA",
    "ebony": "#5A6650",
    "sky": "#7FC8F8",
    "navy": "#3C81AE",
}

# Species name mapping for nicer plot labels
SPECIES_NAME_MAPPING = {
    "gluc": "HeLa degradome",
    "helaqc": "HeLa single shot",
    "herceptin": "Herceptin",
    "immuno": "Immunopeptidomics-1",
    "sbrodae": "Scalindua brodae",
    "snakevenoms": "Snake venomics",
    "woundfluids": "Wound exudates",
    "PXD014877": r"$\mathit{C.\ elegans}$",
    "PXD019483": "HepG2",
    "PXD023064": "Immunopeptidomics-2",
    "general": "General test set",
}

# Configure matplotlib to match seaborn "paper" context with font_scale=2
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
# paper_params = {
#     # bar edge settings
#     "patch.force_edgecolor": True,   # force edgecolors on histogram/bar patches
#     "patch.edgecolor": "black",      # default edge color for patches
#     "patch.linewidth": 1.0,          # border thickness
# }
# mpl.rcParams.update(paper_params)

# Print style information
print("Matplotlib seaborn-v0_0-paper Style Characteristics:")
print(f"Figure size: {plt.rcParams['figure.figsize']}")
print(f"DPI: {plt.rcParams['figure.dpi']}")
print(f"Font size: {plt.rcParams['font.size']}")
print(f"Axes line width: {plt.rcParams['axes.linewidth']}")

print("\nKey rcParams settings:")
paper_relevant_params = [
    "figure.figsize",
    "figure.dpi",
    "font.size",
    "axes.linewidth",
    "axes.grid",
    "axes.spines.left",
    "axes.spines.bottom",
    "axes.spines.top",
    "axes.spines.right",
    "xtick.bottom",
    "xtick.top",
    "ytick.left",
    "ytick.right",
    "axes.axisbelow",
    "grid.linewidth",
    "lines.linewidth",
    "patch.linewidth",
    "lines.markersize",
    "axes.titlesize",
    "axes.labelsize",
    "xtick.labelsize",
    "ytick.labelsize",
    "legend.fontsize",
]

for param in paper_relevant_params:
    if param in plt.rcParams:
        print(f"  {param}: {plt.rcParams[param]}")

print(f"\nCurrent style: {mpl.get_backend()}")
print(
    f"Available styles containing 'seaborn': {[s for s in plt.style.available if 'seaborn' in s]}"
)


def compute_pr_curve(
    input_dataset: pd.DataFrame,
    confidence_column: str,
    label_column: str,
    name: str,
) -> pd.DataFrame:
    """Compute precision-recall curve for given confidence scores and labels.

    Args:
        input_dataset: DataFrame containing confidence scores and labels
        confidence_column: Name of the column containing confidence scores
        label_column: Name of the column containing boolean labels
        name: Name to assign to the computed curve

    Returns:
        DataFrame with precision, recall, and name columns
    """
    original = input_dataset[[confidence_column, label_column]]
    original = original.sort_values(by=confidence_column, ascending=False)
    cum_correct = np.cumsum(original[label_column])
    precision = cum_correct / np.arange(1, len(original) + 1)
    recall = cum_correct / len(original)
    metrics = pd.DataFrame({"precision": precision, "recall": recall}).reset_index(
        drop=True
    )
    metrics["name"] = name
    return metrics


def plot_pr_curve_on_axes(
    metadata: pd.DataFrame,
    ax: plt.Axes,
    title: str = "Precision-Recall Curve",
    label_column: str = "correct",
) -> None:
    """Plot precision-recall curves for original and calibrated confidence on a given axes."""
    # Compute PR curves
    original = compute_pr_curve(
        input_dataset=metadata,
        confidence_column="confidence",
        label_column=label_column,
        name="Raw confidence",
    )
    calibrated = compute_pr_curve(
        input_dataset=metadata,
        confidence_column="calibrated_confidence",
        label_column=label_column,
        name="Calibrated confidence",
    )
    metrics = pd.concat([original, calibrated], axis=0).reset_index(drop=True)

    # Plot each curve with new color scheme
    for name, group in metrics.groupby("name"):
        if name == "Raw confidence":
            color = COLORS["sky"]  # Sky blue for original
        else:
            color = COLORS["ebony"]  # Ebony for calibrated
        ax.plot(group["recall"], group["precision"], label=name, color=color, zorder=2)

    ax.set_axisbelow(True)
    ax.grid(True, color="lightgray", zorder=0)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def plot_confidence_distribution_on_axes(
    metadata: pd.DataFrame,
    ax: plt.Axes,
    confidence_column: str = "confidence",
    title: str = "Confidence Distribution",
    density: bool = False,
    label_column: str = "correct",
) -> None:
    """Plot confidence distribution on a given axes."""
    plot_df = metadata[[confidence_column, label_column]].copy(deep=True)
    plot_df[label_column] = plot_df[label_column].apply(lambda x: "T" if x else "F")

    true_conf = plot_df[plot_df[label_column] == "T"][confidence_column]
    false_conf = plot_df[plot_df[label_column] == "F"][confidence_column]

    ax.hist(
        false_conf,
        bins=50,
        alpha=0.7,
        label="Incorrect",
        color=COLORS["sky"],
        density=density,
        edgecolor="#333333",
    )
    ax.hist(
        true_conf,
        bins=50,
        alpha=0.7,
        label="Correct",
        color=COLORS["ebony"],
        density=density,
        edgecolor="#333333",
    )
    ax.set_xlabel(confidence_column.replace("_", " ").title())
    if density:
        ax.set_ylabel("Density")
    else:
        ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()


def plot_calibration_curve_on_axes(
    metadata: pd.DataFrame,
    ax: plt.Axes,
    confidence_column: str = "confidence",
    title: str = "Confidence Calibration",
    label_column: str = "correct",
) -> None:
    """Plot probability calibration curve on a given axes."""
    confidence_scores = metadata[confidence_column].values
    true_labels = metadata[label_column].values

    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        true_labels, confidence_scores, n_bins=10, strategy="uniform"
    )

    # Determine color based on confidence column
    if confidence_column == "confidence":
        color = COLORS["sky"]  # Sky for original
        label = "Raw confidence"
    else:
        color = COLORS["ebony"]  # Ebony for calibrated
        label = "Calibrated confidence"

    # Plot calibration curve
    ax.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        label=label,
        color=color,
        zorder=2,
    )
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.5, zorder=2)
    ax.set_axisbelow(True)
    ax.grid(True, color="lightgray", zorder=0)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])


def plot_combined_calibration_curves(
    metadata: pd.DataFrame,
    title: str = "Confidence Calibration Comparison",
    label_column: str = "correct",
) -> plt.Figure:
    """Plot both original and calibrated confidence calibration curves on a single axis."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot original confidence calibration
    confidence_scores = metadata["confidence"].values
    true_labels = metadata[label_column].values
    fraction_of_positives, mean_predicted_value = calibration_curve(
        true_labels, confidence_scores, n_bins=10, strategy="uniform"
    )
    ax.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        label="Raw confidence",
        color=COLORS["sky"],
        zorder=2,
    )

    # Plot calibrated confidence calibration
    calibrated_scores = metadata["calibrated_confidence"].values
    fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(
        true_labels, calibrated_scores, n_bins=10, strategy="uniform"
    )
    ax.plot(
        mean_predicted_value_cal,
        fraction_of_positives_cal,
        "o-",
        label="Calibrated confidence",
        color=COLORS["ebony"],
        zorder=2,
    )

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.5, zorder=2)

    ax.set_axisbelow(True)
    ax.grid(True, color="lightgray", zorder=0)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    return fig


def plot_combined_confidence_distributions(
    metadata: pd.DataFrame,
    title: str = "Confidence Distributions",
    label_column: str = "correct",
) -> plt.Figure:
    """Plot both original and calibrated confidence distributions on a single panel."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Original confidence distribution
    plot_confidence_distribution_on_axes(
        metadata,
        ax1,
        "confidence",
        "Raw confidence distribution",
        label_column=label_column,
    )

    # Calibrated confidence distribution
    plot_confidence_distribution_on_axes(
        metadata,
        ax2,
        "calibrated_confidence",
        "Calibrated confidence distribution",
        label_column=label_column,
    )

    plt.suptitle(title)
    return fig


def get_plot_dataframe(
    features_df: pd.DataFrame,
    winnow_metrics_df: pd.DataFrame,
    decoy_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create a dataframe for FDR plotting with true and estimated FDR values.

    Args:
        features_df: DataFrame containing features
        winnow_metrics_df: DataFrame containing Winnow metrics
        decoy_metrics_df: DataFrame containing decoy metrics

    Returns:
        DataFrame with confidence, FDR, and source columns
    """
    metrics_df = pd.merge(
        decoy_metrics_df,
        winnow_metrics_df,
        on="spectrum_id",
        how="inner",
        suffixes=("_dbg", "_winnow"),
    )
    df = pd.merge(
        features_df[["spectrum_id", "calibrated_confidence"]],
        metrics_df,
        on="spectrum_id",
        how="inner",
    ).sort_values(by="calibrated_confidence")
    return df


def plot_fdr_accuracy_on_axes(
    metadata: pd.DataFrame,
    winnow_metrics_df: pd.DataFrame,
    decoy_metrics_df: pd.DataFrame,
    ax: plt.Axes,
    title: str = "FDR Accuracy",
) -> None:
    """Plot FDR accuracy comparison on a given axes.

    Args:
        metadata: DataFrame containing confidence scores and labels
        ax: Matplotlib axes to plot on
        fdr_function: Function to calculate estimated FDR
        confidence_column: Name of the column containing confidence scores
        title: Title for the plot
        label_column: Name of the column containing boolean labels
    """
    # Get the multi-plot dataframe
    multi_plot_df = get_plot_dataframe(
        features_df=metadata,
        winnow_metrics_df=winnow_metrics_df,
        decoy_metrics_df=decoy_metrics_df,
    )

    # Plot FDR lines for each source
    ax.plot(
        multi_plot_df["calibrated_confidence"],
        multi_plot_df["psm_fdr_winnow"],
        label="Winnow FDR",
        color=COLORS["sky"],
        zorder=2,
    )
    ax.plot(
        multi_plot_df["calibrated_confidence"],
        multi_plot_df["psm_fdr_dbg"],
        label="Decoy FDR",
        color=COLORS["ebony"],
        zorder=2,
    )

    # Add horizontal line at FDR = 0.05
    ax.axhline(y=0.05, color="black", linestyle="--", alpha=0.5, zorder=2)

    # Customize the plot
    ax.set_axisbelow(True)
    ax.grid(True, color="lightgray", zorder=0)
    ax.set_xlabel("Calibrated Confidence")
    ax.set_ylabel("False discovery rate (FDR)")
    ax.set_title(title)
    ax.legend()


def create_fdr_accuracy_plot(
    metadata: pd.DataFrame,
    winnow_metrics_df: pd.DataFrame,
    decoy_metrics_df: pd.DataFrame,
    title: str = "FDR Accuracy",
) -> plt.Figure:
    """Create standalone FDR accuracy plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_fdr_accuracy_on_axes(
        metadata,
        winnow_metrics_df,
        decoy_metrics_df,
        ax,
        title,
    )
    return fig


def create_pr_curve_plot(
    metadata: pd.DataFrame,
    title: str = "Precision-Recall Curve",
    label_column: str = "correct",
) -> plt.Figure:
    """Create standalone precision-recall curve plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_pr_curve_on_axes(metadata, ax, title, label_column)
    return fig


def find_data_files(base_dir: str = "new_model/results") -> dict:
    """Find all relevant data files in the results directory.

    Args:
        base_dir: Base directory to search for files

    Returns:
        Dictionary with file categories and their paths
    """
    files: dict[str, list[str]] = {"labelled": [], "de_novo": [], "raw": []}

    # Find all CSV files in the directory
    csv_files = []
    for pattern in ["*.csv", "*.csv.*"]:  # Include files with suffixes
        csv_files.extend(glob.glob(os.path.join(base_dir, pattern)))

    for file_path in csv_files:
        file_name = os.path.basename(file_path)

        if file_name.startswith("labelled_"):
            files["labelled"].append(file_path)
        elif file_name.startswith("de_novo_"):
            files["de_novo"].append(file_path)
        elif file_name.startswith("raw_"):
            files["raw"].append(file_path)

    return files


def extract_dataset_name(file_path: str) -> str:
    """Extract dataset name from file path.

    Args:
        file_path: Path to the data file

    Returns:
        Dataset name
    """
    file_name = os.path.basename(file_path)

    if file_name.startswith("labelled_"):
        # Remove "labelled_" prefix and ".csv" suffix (and any additional suffixes)
        name = file_name[9:]  # Remove "labelled_"
        name = name.split(".csv")[0]  # Remove .csv and any suffixes
        return name.replace("_results", "")
    elif file_name.startswith("de_novo_"):
        # Remove "de_novo_" prefix
        name = file_name[8:]  # Remove "de_novo_"
        name = name.split(".csv")[0]  # Remove .csv and any suffixes
        return name.replace("_preds", "").replace("_results", "")
    elif file_name.startswith("raw_"):
        # Remove "raw_" prefix
        name = file_name[4:]  # Remove "raw_"
        name = name.split(".csv")[0]  # Remove .csv and any suffixes
        return name.replace("_results", "")

    return file_name


def convert_object_columns(metadata: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns that might contain string representations of Python objects."""

    def try_convert(value):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value  # Return original if conversion fails

    # Apply conversion to object (string) columns
    for col in metadata.select_dtypes(include=["object"]).columns:
        metadata[col] = metadata[col].apply(try_convert)

    return metadata


def main() -> None:
    """Plot new model results."""
    # Create output directory for plots
    output_dir = "new_model/results/plots"
    os.makedirs(output_dir, exist_ok=True)

    # Find all data files
    data_files = find_data_files()

    print(f"Found {len(data_files['labelled'])} labelled files")
    print(f"Found {len(data_files['de_novo'])} de novo files")
    print(f"Found {len(data_files['raw'])} raw files")

    # Process labelled data files
    for file_path in data_files["labelled"]:
        dataset_name = extract_dataset_name(file_path)
        print(f"Processing labelled dataset: {dataset_name}")

        try:
            # Load metadata
            metadata = pd.read_csv(file_path)
            metadata = convert_object_columns(metadata)
            display_name = SPECIES_NAME_MAPPING.get(dataset_name, dataset_name)

            # Extract Winnow and decoy metrics
            non_parametric_fdr_control = NonParametricFDRControl()
            non_parametric_fdr_control.fit(dataset=metadata["calibrated_confidence"])
            winnow_metrics_df = non_parametric_fdr_control.add_psm_fdr(
                metadata, "calibrated_confidence"
            )
            winnow_metrics_df = winnow_metrics_df[["spectrum_id", "psm_fdr"]]

            # -- Compute decoy metrics
            database_grounded_fdr_control = DatabaseGroundedFDRControl(
                confidence_feature="calibrated_confidence"
            )
            database_grounded_fdr_control.fit(
                dataset=metadata,
                residue_masses=RESIDUE_MASSES,
                correct_column="correct",
            )
            decoy_metrics_df = database_grounded_fdr_control.add_psm_fdr(
                metadata, "calibrated_confidence"
            )
            decoy_metrics_df = decoy_metrics_df[["spectrum_id", "psm_fdr"]]

            # Create individual plots using database search ("correct" column)

            # 1. Precision-Recall curve
            pr_fig = create_pr_curve_plot(
                metadata,
                f"{display_name} precision-recall curve for labelled data using database search",
                "correct",
            )
            pr_fig.savefig(
                os.path.join(
                    output_dir, f"{dataset_name}_labelled_precision_recall.png"
                ),
                bbox_inches="tight",
                dpi=300,
            )
            pr_fig.savefig(
                os.path.join(
                    output_dir, f"{dataset_name}_labelled_precision_recall.pdf"
                ),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(pr_fig)

            # 2. Combined confidence distributions
            conf_fig = plot_combined_confidence_distributions(
                metadata,
                f"{display_name} confidence distributions for labelled data using database search",
                "correct",
            )
            conf_fig.savefig(
                os.path.join(
                    output_dir, f"{dataset_name}_labelled_confidence_distributions.png"
                ),
                bbox_inches="tight",
                dpi=300,
            )
            conf_fig.savefig(
                os.path.join(
                    output_dir, f"{dataset_name}_labelled_confidence_distributions.pdf"
                ),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(conf_fig)

            # 3. Combined calibration curves
            cal_fig = plot_combined_calibration_curves(
                metadata,
                f"{display_name} calibration curves for labelled data using database search",
                "correct",
            )
            cal_fig.savefig(
                os.path.join(
                    output_dir, f"{dataset_name}_labelled_calibration_curves.png"
                ),
                bbox_inches="tight",
                dpi=300,
            )
            cal_fig.savefig(
                os.path.join(
                    output_dir, f"{dataset_name}_labelled_calibration_curves.pdf"
                ),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(cal_fig)

            # 4. FDR accuracy plot (PSM correctness)
            fdr_fig = create_fdr_accuracy_plot(
                metadata,
                winnow_metrics_df,
                decoy_metrics_df,
                f"{display_name} FDR accuracy for labelled data using database search",
            )
            fdr_fig.savefig(
                os.path.join(output_dir, f"{dataset_name}_labelled_fdr_accuracy.png"),
                bbox_inches="tight",
                dpi=300,
            )
            fdr_fig.savefig(
                os.path.join(output_dir, f"{dataset_name}_labelled_fdr_accuracy.pdf"),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(fdr_fig)

        except Exception as e:
            print(f"Error processing labelled file {file_path}: {e}")

    # Process de novo data files
    for file_path in data_files["de_novo"]:
        dataset_name = extract_dataset_name(file_path)
        print(f"Processing de novo dataset: {dataset_name}")

        try:
            # Load metadata
            metadata = pd.read_csv(file_path)
            display_name = SPECIES_NAME_MAPPING.get(dataset_name, dataset_name)

            # Extract Winnow and decoy metrics
            non_parametric_fdr_control = NonParametricFDRControl()
            non_parametric_fdr_control.fit(dataset=metadata["calibrated_confidence"])
            winnow_metrics_df = non_parametric_fdr_control.add_psm_fdr(
                metadata, "calibrated_confidence"
            )
            winnow_metrics_df = winnow_metrics_df[["spectrum_id", "psm_fdr"]]

            # -- Compute decoy metrics
            database_grounded_fdr_control = DatabaseGroundedFDRControl(
                confidence_feature="calibrated_confidence"
            )
            database_grounded_fdr_control.fit(
                dataset=metadata,
                residue_masses=RESIDUE_MASSES,
                correct_column="proteome_hit",
            )
            decoy_metrics_df = database_grounded_fdr_control.add_psm_fdr(
                metadata, "calibrated_confidence"
            )
            decoy_metrics_df = decoy_metrics_df[["spectrum_id", "psm_fdr"]]

            # Create individual plots using proteome mapping ("proteome_hit" column)

            # 1. Precision-Recall curve
            pr_fig = create_pr_curve_plot(
                metadata,
                f"{display_name} precision-recall curve for "
                + r"$\mathit{de\ novo}$"
                + " data using proteome mapping",
                "proteome_hit",
            )
            pr_fig.savefig(
                os.path.join(
                    output_dir, f"{dataset_name}_de_novo_precision_recall.png"
                ),
                bbox_inches="tight",
                dpi=300,
            )
            pr_fig.savefig(
                os.path.join(
                    output_dir, f"{dataset_name}_de_novo_precision_recall.pdf"
                ),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(pr_fig)

            # 2. Combined confidence distributions
            conf_fig = plot_combined_confidence_distributions(
                metadata,
                f"{display_name} confidence distributions for "
                + r"$\mathit{de\ novo}$"
                + " data using proteome mapping",
                "proteome_hit",
            )
            conf_fig.savefig(
                os.path.join(
                    output_dir, f"{dataset_name}_de_novo_confidence_distributions.png"
                ),
                bbox_inches="tight",
                dpi=300,
            )
            conf_fig.savefig(
                os.path.join(
                    output_dir, f"{dataset_name}_de_novo_confidence_distributions.pdf"
                ),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(conf_fig)

            # 3. Combined calibration curves
            cal_fig = plot_combined_calibration_curves(
                metadata,
                f"{display_name} calibration curves for "
                + r"$\mathit{de\ novo}$"
                + " data using proteome mapping",
                "proteome_hit",
            )
            cal_fig.savefig(
                os.path.join(
                    output_dir, f"{dataset_name}_de_novo_calibration_curves.png"
                ),
                bbox_inches="tight",
                dpi=300,
            )
            cal_fig.savefig(
                os.path.join(
                    output_dir, f"{dataset_name}_de_novo_calibration_curves.pdf"
                ),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(cal_fig)

            # 4. FDR accuracy plot (Proteome hits)
            fdr_fig = create_fdr_accuracy_plot(
                metadata,
                winnow_metrics_df,
                decoy_metrics_df,
                f"{display_name} FDR accuracy for "
                + r"$\mathit{de\ novo}$"
                + " data using proteome mapping",
            )
            fdr_fig.savefig(
                os.path.join(output_dir, f"{dataset_name}_de_novo_fdr_accuracy.png"),
                bbox_inches="tight",
                dpi=300,
            )
            fdr_fig.savefig(
                os.path.join(output_dir, f"{dataset_name}_de_novo_fdr_accuracy.pdf"),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(fdr_fig)

        except Exception as e:
            print(f"Error processing de novo file {file_path}: {e}")

    # Process raw data files
    for file_path in data_files["raw"]:
        dataset_name = extract_dataset_name(file_path)
        print(f"Processing raw dataset: {dataset_name}")

        try:
            # Load metadata
            metadata = pd.read_csv(file_path)
            display_name = SPECIES_NAME_MAPPING.get(dataset_name, dataset_name)

            # Extract Winnow and decoy metrics
            non_parametric_fdr_control = NonParametricFDRControl()
            non_parametric_fdr_control.fit(dataset=metadata["calibrated_confidence"])
            winnow_metrics_df = non_parametric_fdr_control.add_psm_fdr(
                metadata, "calibrated_confidence"
            )
            winnow_metrics_df = winnow_metrics_df[["spectrum_id", "psm_fdr"]]

            # -- Compute decoy metrics
            database_grounded_fdr_control = DatabaseGroundedFDRControl(
                confidence_feature="calibrated_confidence"
            )
            database_grounded_fdr_control.fit(
                dataset=metadata,
                residue_masses=RESIDUE_MASSES,
                correct_column="proteome_hit",
            )
            decoy_metrics_df = database_grounded_fdr_control.add_psm_fdr(
                metadata, "calibrated_confidence"
            )
            decoy_metrics_df = decoy_metrics_df[["spectrum_id", "psm_fdr"]]

            # Create individual plots using proteome mapping ("proteome_hit" column)

            # 1. Precision-Recall curve
            pr_fig = create_pr_curve_plot(
                metadata,
                f"{display_name} precision-recall curve for full search space using proteome mapping",
                "proteome_hit",
            )
            pr_fig.savefig(
                os.path.join(output_dir, f"{dataset_name}_raw_precision_recall.png"),
                bbox_inches="tight",
                dpi=300,
            )
            pr_fig.savefig(
                os.path.join(output_dir, f"{dataset_name}_raw_precision_recall.pdf"),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(pr_fig)

            # 2. Combined confidence distributions
            conf_fig = plot_combined_confidence_distributions(
                metadata,
                f"{display_name} confidence distributions for full search space using proteome mapping",
                "proteome_hit",
            )
            conf_fig.savefig(
                os.path.join(
                    output_dir, f"{dataset_name}_raw_confidence_distributions.png"
                ),
                bbox_inches="tight",
                dpi=300,
            )
            conf_fig.savefig(
                os.path.join(
                    output_dir, f"{dataset_name}_raw_confidence_distributions.pdf"
                ),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(conf_fig)

            # 3. Combined calibration curves
            cal_fig = plot_combined_calibration_curves(
                metadata,
                f"{display_name} calibration curves for full search space using proteome mapping",
                "proteome_hit",
            )
            cal_fig.savefig(
                os.path.join(output_dir, f"{dataset_name}_raw_calibration_curves.png"),
                bbox_inches="tight",
                dpi=300,
            )
            cal_fig.savefig(
                os.path.join(output_dir, f"{dataset_name}_raw_calibration_curves.pdf"),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(cal_fig)

            # 4. FDR accuracy plot (Proteome hits)
            fdr_fig = create_fdr_accuracy_plot(
                metadata,
                winnow_metrics_df,
                decoy_metrics_df,
                f"{display_name} FDR accuracy for full search space using proteome mapping",
            )
            fdr_fig.savefig(
                os.path.join(output_dir, f"{dataset_name}_raw_fdr_accuracy.png"),
                bbox_inches="tight",
                dpi=300,
            )
            fdr_fig.savefig(
                os.path.join(output_dir, f"{dataset_name}_raw_fdr_accuracy.pdf"),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(fdr_fig)

        except Exception as e:
            print(f"Error processing raw file {file_path}: {e}")

    print(f"All individual plots saved to {output_dir}")


if __name__ == "__main__":
    main()
