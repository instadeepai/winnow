import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from typing import Tuple, List
import numpy as np


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
        name="Original",
    )
    calibrated = compute_pr_curve(
        input_dataset=metadata,
        confidence_column="calibrated_confidence",
        label_column=label_column,
        name="Calibrated",
    )
    metrics = pd.concat([original, calibrated], axis=0).reset_index(drop=True)

    # Plot each curve
    for name, group in metrics.groupby("name"):
        if name == "Original":
            color = "#4A90E2"  # Muted blue
        else:
            color = "#F5A623"  # Muted gold
        ax.plot(
            group["recall"], group["precision"], label=name, linewidth=2, color=color
        )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
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
        true_conf, bins=50, alpha=0.7, label="True", color="#4A90E2", density=density
    )
    ax.hist(
        false_conf, bins=50, alpha=0.7, label="False", color="#F5A623", density=density
    )
    ax.set_xlabel(confidence_column.replace("_", " ").title())
    if density:
        ax.set_ylabel("Density")
    else:
        ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


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
        color = "#4A90E2"  # Muted blue
        label = "Original Confidence"
    else:
        color = "#F5A623"  # Muted gold
        label = "Calibrated Confidence"

    # Plot calibration curve
    ax.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        label=label,
        color=color,
        linewidth=2,
        markersize=8,
    )
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated", alpha=0.5)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])


def plot_pr_curve(
    metadata: pd.DataFrame, title: str, label_column: str = "correct"
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot precision-recall curves for original and calibrated confidence."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    plot_pr_curve_on_axes(metadata, ax, title, label_column)
    plt.tight_layout()
    return fig, ax


def plot_confidence_distributions(
    metadata: pd.DataFrame,
    title: str,
    density: bool = False,
    label_column: str = "correct",
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Plot confidence distributions for both confidence and calibrated confidence."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    plt.suptitle(title, fontsize=14, y=0.97)

    plot_confidence_distribution_on_axes(
        metadata, ax1, "confidence", "Confidence Distribution", density, label_column
    )
    plot_confidence_distribution_on_axes(
        metadata,
        ax2,
        "calibrated_confidence",
        "Calibrated Confidence Distribution",
        density,
        label_column,
    )

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_probability_calibration(
    metadata: pd.DataFrame, title: str, label_column: str = "correct"
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Plot probability calibration curves for both confidence and calibrated confidence."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    plt.suptitle(title, fontsize=14, y=0.97)

    plot_calibration_curve_on_axes(
        metadata, ax1, "confidence", "Confidence Calibration", label_column
    )
    plot_calibration_curve_on_axes(
        metadata,
        ax2,
        "calibrated_confidence",
        "Calibrated Confidence Calibration",
        label_column,
    )

    plt.tight_layout()
    return fig, (ax1, ax2)


def count_num_matches_at_fdr(
    metadata: pd.DataFrame,
    confidence_column: str,
    confidence_cutoff: float,
    label_column: str = "correct",
) -> Tuple[int, int]:
    """Count the number of correct and incorrect PSMs at a given FDR threshold.

    Args:
        metadata: DataFrame containing confidence scores and correct labels
        confidence_column: Name of the column containing confidence scores
        confidence_cutoff: Confidence cutoff at a 5% FDR
        label_column: Name of the column containing boolean labels

    Returns:
        Tuple of (correct_psms, incorrect_psms)
    """
    # Sort by confidence in descending order
    df = metadata.sort_values(confidence_column, ascending=False).copy()
    df = df[df[confidence_column] >= confidence_cutoff]

    # Count correct and incorrect PSMs
    correct_psms = df[df[label_column]].shape[0]
    incorrect_psms = df[~df[label_column]].shape[0]

    return correct_psms, incorrect_psms


def plot_comparison_bar_chart(
    species_list: List[str],
    winnow_confidence_cutoffs: List[float],
    dbg_confidence_cutoffs: List[float],
    config: dict,
    fdr_threshold: float = 0.05,
) -> Tuple[plt.Figure, plt.Axes]:
    """Generic function to create a stacked bar chart comparing two methods.

    Args:
        species_list: List of species names
        winnow_confidence_cutoffs: List of confidence cutoffs for Winnow method
        dbg_confidence_cutoffs: List of confidence cutoffs for database-grounded method
        config: Configuration dictionary with chart-specific settings
        fdr_threshold: FDR threshold (default 0.05 for 5% FDR)

    Returns:
        Tuple of (figure, axes)
    """
    # Prepare data for plotting
    plot_data = []

    for species, winnow_confidence_cutoff, dbg_confidence_cutoff in zip(
        species_list, winnow_confidence_cutoffs, dbg_confidence_cutoffs
    ):
        try:
            # Load metadata
            metadata = pd.read_csv(config["file_path_template"].format(species=species))

            # Count matches for Winnow method
            winnow_positive, winnow_negative = count_num_matches_at_fdr(
                metadata,
                "calibrated_confidence",
                winnow_confidence_cutoff,
                config["label_column"],
            )

            # Count matches for DBG method
            dbg_positive, dbg_negative = count_num_matches_at_fdr(
                metadata,
                "calibrated_confidence",
                dbg_confidence_cutoff,
                config["label_column"],
            )

            # Add data for plotting
            plot_data.append(
                {
                    "species": species,
                    "winnow_positive": winnow_positive,
                    "winnow_negative": winnow_negative,
                    "dbg_positive": dbg_positive,
                    "dbg_negative": dbg_negative,
                }
            )

        except FileNotFoundError:
            print(f"Warning: Data file not found for species {species}")
            plot_data.append(
                {
                    "species": species,
                    "winnow_positive": 0,
                    "winnow_negative": 0,
                    "dbg_positive": 0,
                    "dbg_negative": 0,
                }
            )

    # Create DataFrame for plotting
    plot_df = pd.DataFrame(plot_data)

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Set up bar positions
    x = range(len(species_list))
    width = 0.35

    # Create stacked bars for Winnow
    ax.bar(
        [i - width / 2 for i in x],
        plot_df["winnow_positive"],
        width,
        label=config["winnow_positive_label"],
        color="#4A90E2",  # Muted blue
        alpha=0.8,
    )
    ax.bar(
        [i - width / 2 for i in x],
        plot_df["winnow_negative"],
        width,
        bottom=plot_df["winnow_positive"],
        label=config["winnow_negative_label"],
        color="#F5A623",  # Muted gold
        alpha=0.8,
    )

    # Create stacked bars for DBG
    ax.bar(
        [i + width / 2 for i in x],
        plot_df["dbg_positive"],
        width,
        label=config["dbg_positive_label"],
        color="#2E5BBA",  # Darker blue
        alpha=0.8,
    )
    ax.bar(
        [i + width / 2 for i in x],
        plot_df["dbg_negative"],
        width,
        bottom=plot_df["dbg_positive"],
        label=config["dbg_negative_label"],
        color="#D4AF37",  # Darker gold
        alpha=0.8,
    )

    # Add labels and title
    ax.set_xlabel("Species", fontsize=12)
    ax.set_ylabel(f"Number of PSMs at {fdr_threshold * 100}% FDR", fontsize=12)
    ax.set_title(config["title"].format(fdr_threshold * 100), fontsize=14)

    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(species_list, rotation=45, ha="right")

    # Add legend
    ax.legend(fontsize=10)

    # Add grid
    ax.grid(True, alpha=0.3, axis="y")

    # Create a summary table below the plot
    table_data = []
    for i, species in enumerate(species_list):
        # Calculate rates for each method
        np_total = (
            plot_df.iloc[i]["winnow_positive"] + plot_df.iloc[i]["winnow_negative"]
        )
        dbg_total = plot_df.iloc[i]["dbg_positive"] + plot_df.iloc[i]["dbg_negative"]

        np_rate = plot_df.iloc[i]["winnow_negative"] / np_total if np_total > 0 else 0
        dbg_rate = plot_df.iloc[i]["dbg_negative"] / dbg_total if dbg_total > 0 else 0

        table_data.append(
            [
                species,
                f"{plot_df.iloc[i]['winnow_positive']}",
                f"{plot_df.iloc[i]['winnow_negative']}",
                f"{np_rate * 100:.1f}%",
                f"{plot_df.iloc[i]['dbg_positive']}",
                f"{plot_df.iloc[i]['dbg_negative']}",
                f"{dbg_rate * 100:.1f}%",
            ]
        )

    # Add table below the plot
    table_ax = fig.add_axes([0.1, 0.05, 0.8, 0.25])  # Position table below the plot
    table_ax.axis("tight")
    table_ax.axis("off")

    table = table_ax.table(
        cellText=table_data,
        colLabels=config["table_headers"],
        cellLoc="center",
        loc="center",
        colWidths=[0.15, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Color header row
    for j in range(len(table_data[0])):
        table[(0, j)].set_facecolor("#4A90E2")  # Muted blue to match the chart
        table[(0, j)].set_text_props(weight="bold", color="white")

    # Color alternating rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")

    ax.set_yscale("log")

    plt.tight_layout()

    return fig, ax


def plot_psm_hits_bar_chart(
    species_list: List[str],
    labelled_winnow_confidence_cutoffs: List[float],
    labelled_dbg_confidence_cutoffs: List[float],
    fdr_threshold: float = 0.05,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a stacked bar chart comparing the number of PSMs at given FDR threshold.

    Compares between Winnow and database-grounded FDR methods for each species.

    Args:
        species_list: List of species names
        labelled_winnow_confidence_cutoffs: List of confidence cutoffs at a 5% FDR for Winnow method (labelled)
        labelled_dbg_confidence_cutoffs: List of confidence cutoffs at a 5% FDR for database-grounded method (labelled)
        fdr_threshold: FDR threshold (default 0.05 for 5% FDR)

    Returns:
        Tuple of (figure, axes)
    """
    config = {
        "file_path_template": "holdout_results/all_less_{species}_labelled_test_results_with_db_fdr.csv",
        "label_column": "correct",
        "winnow_positive_label": "Non-Parametric True Positive",
        "winnow_negative_label": "Non-Parametric False Positive",
        "dbg_positive_label": "Database-Grounded True Positive",
        "dbg_negative_label": "Database-Grounded False Positive",
        "title": "Comparison of PSMs Identified at {}% FDR: Non-Parametric vs Database-Grounded",
        "table_headers": [
            "Species",
            "Non-Parametric TP",
            "Non-Parametric FP",
            "Non-Parametric FPR",
            "Database-Grounded TP",
            "Database-Grounded FP",
            "Database-Grounded FPR",
        ],
    }

    return plot_comparison_bar_chart(
        species_list,
        labelled_winnow_confidence_cutoffs,
        labelled_dbg_confidence_cutoffs,
        config,
        fdr_threshold,
    )


def plot_proteome_hits_bar_chart(
    species_list: List[str],
    raw_winnow_confidence_cutoffs: List[float],
    labelled_dbg_confidence_cutoffs: List[float],
    fdr_threshold: float = 0.05,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a stacked bar chart comparing the number of proteome hits at given FDR threshold.

    Compares between Winnow and database-grounded FDR methods for each species.
    Skips over species that don't have raw data files (immuno, woundfluids).

    Args:
        species_list: List of species names
        raw_winnow_confidence_cutoffs: List of confidence cutoffs at a 5% FDR for Winnow method (raw)
        labelled_dbg_confidence_cutoffs: List of confidence cutoffs at a 5% FDR for database-grounded method (labelled)
        fdr_threshold: FDR threshold (default 0.05 for 5% FDR)

    Returns:
        Tuple of (figure, axes)
    """
    # Filter out species that don't have raw data files
    species_with_raw_data = []
    filtered_winnow_cutoffs = []
    filtered_dbg_cutoffs = []

    for i, species in enumerate(species_list):
        if species not in ["immuno", "woundfluids"]:
            species_with_raw_data.append(species)
            filtered_winnow_cutoffs.append(raw_winnow_confidence_cutoffs[i])
            filtered_dbg_cutoffs.append(labelled_dbg_confidence_cutoffs[i])
        else:
            print(f"Skipping {species} - no raw data available")

    config = {
        "file_path_template": "holdout_results/all_less_{species}_raw_test_results.csv",
        "label_column": "proteome_hits",
        "winnow_positive_label": "Non-Parametric Proteome Hits",
        "winnow_negative_label": "Non-Parametric Non-Proteome Hits",
        "dbg_positive_label": "Database-Grounded Proteome Hits",
        "dbg_negative_label": "Database-Grounded Non-Proteome Hits",
        "title": "Comparison of Proteome Hits at {}% FDR: Non-Parametric vs Database-Grounded (Raw Data)",
        "table_headers": [
            "Species",
            "Non-Parametric Proteome Hits",
            "Non-Parametric Non-Proteome Hits",
            "Non-Parametric Non-hit Rate",
            "Database-Grounded Proteome Hits",
            "Database-Grounded Non-Proteome Hits",
            "Database-Grounded Non-hit Rate",
        ],
    }

    return plot_comparison_bar_chart(
        species_with_raw_data,
        filtered_winnow_cutoffs,
        filtered_dbg_cutoffs,
        config,
        fdr_threshold,
    )


def create_species_pie_chart(
    species_list: List[str],
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a pie chart showing the fraction of total data each species represents.

    Includes annotations showing total number of PSMs.

    Args:
        species_list: List of species names

    Returns:
        Tuple of (figure, axes)
    """
    # Prepare data for pie chart
    species_data = []
    species_colors = [
        "#4A90E2",
        "#F5A623",
        "#2E5BBA",
        "#D4AF37",
        "#7B68EE",
        "#FF6B6B",
        "#4ECDC4",
    ]

    for i, species in enumerate(species_list):
        try:
            # Load labelled metadata
            metadata = pd.read_csv(
                f"holdout_results/all_less_{species}_labelled_test_results_with_db_fdr.csv"
            )

            # Count total rows in the dataset
            total_rows = len(metadata)

            species_data.append(
                {
                    "species": species,
                    "total_psms": total_rows,
                    "color": species_colors[i % len(species_colors)],
                }
            )

        except FileNotFoundError:
            print(f"Warning: Could not find data file for species {species}")
            species_data.append(
                {
                    "species": species,
                    "total_psms": 0,
                    "color": species_colors[i % len(species_colors)],
                }
            )

    # Create DataFrame and calculate percentages
    df = pd.DataFrame(species_data)
    total_all_psms = df["total_psms"].sum()
    df["percentage"] = (df["total_psms"] / total_all_psms * 100).round(1)

    # Create the pie chart
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Create pie chart
    wedges, texts = ax.pie(
        df["total_psms"],
        labels=None,  # Remove labels
        colors=df["color"],
        autopct=None,  # Remove percentage text
        startangle=90,
        textprops={"fontsize": 10},
    )

    # Add title
    ax.set_title(
        f"Distribution of PSMs Across Species\n(Total: {int(total_all_psms)} PSMs)",
        fontsize=14,
        pad=20,
    )

    # Add legend with counts
    legend_labels = [
        f"{row['species']} ({int(row['total_psms'])} PSMs)" for _, row in df.iterrows()
    ]
    ax.legend(
        wedges,
        legend_labels,
        title="Species",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
    )

    plt.tight_layout()

    return fig, ax


def create_combined_species_plot(
    metadata: pd.DataFrame, title: str, label_column: str = "correct"
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes, plt.Axes]]:
    """Create a combined plot with PR curve, confidence distributions, and probability calibration for a species."""
    # Create a single figure with 3 rows and 2 columns
    fig = plt.figure(figsize=(15, 18))

    # Set the main title for the entire figure
    plt.suptitle(title, fontsize=16, y=0.98)

    # Plot 1: PR curve (spans full width, top row)
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    plot_pr_curve_on_axes(metadata, ax1, "Precision-Recall Curve", label_column)

    # Plot 2: Confidence distribution (middle left)
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    plot_confidence_distribution_on_axes(
        metadata,
        ax2,
        "confidence",
        "Confidence Distribution",
        label_column=label_column,
    )

    # Plot 3: Calibrated confidence distribution (middle right)
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    plot_confidence_distribution_on_axes(
        metadata,
        ax3,
        "calibrated_confidence",
        "Calibrated Confidence Distribution",
        label_column=label_column,
    )

    # Plot 4: Original confidence calibration (bottom left)
    ax4 = plt.subplot2grid((3, 2), (2, 0))
    plot_calibration_curve_on_axes(
        metadata, ax4, "confidence", "Confidence Calibration", label_column
    )

    # Plot 5: Calibrated confidence calibration (bottom right)
    ax5 = plt.subplot2grid((3, 2), (2, 1))
    plot_calibration_curve_on_axes(
        metadata,
        ax5,
        "calibrated_confidence",
        "Calibrated Confidence Calibration",
        label_column,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Add more space at the top

    return fig, (ax1, ax2, ax3, ax4, ax5)


def load_confidence_cutoffs(
    species_list: List[str],
) -> Tuple[List[Tuple[float, float]], List[float]]:
    """Load confidence cutoffs from files generated during FDR analysis.

    Args:
        species_list: List of species names

    Returns:
        Tuple of (winnow_confidence_cutoffs, dbg_confidence_cutoffs)
    """
    winnow_confidence_cutoffs = []
    dbg_confidence_cutoffs = []

    for species in species_list:
        # Load Winnow confidence cutoff
        winnow_labelled_cutoff_path = (
            f"holdout_results/all_less_{species}_winnow_labelled_confidence_cutoff.txt"
        )
        winnow_raw_cutoff_path = (
            f"holdout_results/all_less_{species}_winnow_raw_confidence_cutoff.txt"
        )
        try:
            with open(winnow_labelled_cutoff_path, "r") as f:
                winnow_labelled_cutoff = float(f.read().strip())
            with open(winnow_raw_cutoff_path, "r") as f:
                winnow_raw_cutoff = float(f.read().strip())
            winnow_confidence_cutoffs.append(
                (winnow_labelled_cutoff, winnow_raw_cutoff)
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find Winnow confidence cutoff file for {species}."
            )

        # Load DBG confidence cutoff
        dbg_cutoff_path = (
            f"holdout_results/all_less_{species}_dbg_labelled_confidence_cutoff.txt"
        )
        try:
            with open(dbg_cutoff_path, "r") as f:
                dbg_cutoff = float(f.read().strip())
            dbg_confidence_cutoffs.append(dbg_cutoff)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find DBG confidence cutoff file for {species}: {dbg_cutoff_path}"
            )

    return winnow_confidence_cutoffs, dbg_confidence_cutoffs


def main():
    """Plot holdout results."""
    species_list = [
        "gluc",
        "helaqc",
        "herceptin",
        "immuno",
        "sbrodae",
        "snakevenoms",
        "woundfluids",
    ]

    # Load confidence cutoffs from files
    labelled_and_raw_winnow_confidence_cutoffs, labelled_dbg_confidence_cutoffs = (
        load_confidence_cutoffs(species_list)
    )
    labelled_winnow_confidence_cutoffs = [
        cutoff[0] for cutoff in labelled_and_raw_winnow_confidence_cutoffs
    ]
    raw_winnow_confidence_cutoffs = [
        cutoff[1] for cutoff in labelled_and_raw_winnow_confidence_cutoffs
    ]

    print(
        f"Loaded Winnow confidence cutoffs: {labelled_and_raw_winnow_confidence_cutoffs}"
    )
    print(f"Loaded DBG confidence cutoffs: {labelled_dbg_confidence_cutoffs}")

    for species in species_list:
        # Load labelled metadata
        labelled_metadata = pd.read_csv(
            f"holdout_results/all_less_{species}_labelled_test_results_with_db_fdr.csv"
        )

        # Create combined labelled data plot for the species (uses "correct" column)
        fig, (ax1, ax2, ax3, ax4, ax5) = create_combined_species_plot(
            labelled_metadata, f"{species} (Labelled)", "correct"
        )
        fig.savefig(
            f"holdout_results/plots/all_less_{species}_labelled_combined_plot.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            f"holdout_results/plots/all_less_{species}_labelled_combined_plot.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

        # Load raw metadata (if available) - skip immuno and woundfluids
        if species not in ["immuno", "woundfluids"]:
            try:
                raw_metadata = pd.read_csv(
                    f"holdout_results/all_less_{species}_raw_test_results.csv"
                )

                # Create combined raw data plot for the species (uses "proteome_hits" column)
                fig, (ax1, ax2, ax3, ax4, ax5) = create_combined_species_plot(
                    raw_metadata, f"{species} (Raw)", "proteome_hits"
                )
                fig.savefig(
                    f"holdout_results/plots/all_less_{species}_raw_combined_plot.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                fig.savefig(
                    f"holdout_results/plots/all_less_{species}_raw_combined_plot.pdf",
                    bbox_inches="tight",
                )
                plt.close(fig)
            except FileNotFoundError:
                print(f"Warning: Raw data file not found for species {species}")
        else:
            print(f"Skipping raw combined plot for {species} - no raw data available")

    # Create FDR PSMs comparison bar chart
    fig, ax = plot_psm_hits_bar_chart(
        species_list=species_list,
        labelled_winnow_confidence_cutoffs=labelled_winnow_confidence_cutoffs,
        labelled_dbg_confidence_cutoffs=labelled_dbg_confidence_cutoffs,
    )
    fig.savefig(
        "holdout_results/plots/fdr_comparison_bar_chart.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        "holdout_results/plots/fdr_comparison_bar_chart.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # Create species pie chart
    fig, ax = create_species_pie_chart(
        species_list=species_list,
    )
    fig.savefig(
        "holdout_results/plots/species_pie_chart.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        "holdout_results/plots/species_pie_chart.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # Create proteome hits bar chart
    fig, ax = plot_proteome_hits_bar_chart(
        species_list=species_list,
        raw_winnow_confidence_cutoffs=raw_winnow_confidence_cutoffs,
        labelled_dbg_confidence_cutoffs=labelled_dbg_confidence_cutoffs,
    )
    fig.savefig(
        "holdout_results/plots/proteome_hits_bar_chart.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        "holdout_results/plots/proteome_hits_bar_chart.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
