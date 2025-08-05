import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc
from typing import Tuple, List, Callable, Union, Dict
import numpy as np
import seaborn as sns

# Species name mapping for nicer plot labels
SPECIES_NAME_MAPPING: Dict[str, str] = {
    "helaqc": "HeLa Single Shot",
    "gluc": "HeLa Degradome",
    "herceptin": "Herceptin",
    "snakevenoms": "Snake Venomics",
    "woundfluids": "Wound Exudates",
    "sbrodae": "Scalindua brodae",
    "immuno": "Immunopeptidomics",
    "PXD023064": "HepG2",
}


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

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
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
        true_conf, bins=50, alpha=0.7, label="Correct", color="#4A90E2", density=density
    )
    ax.hist(
        false_conf,
        bins=50,
        alpha=0.7,
        label="Incorrect",
        color="#F5A623",
        density=density,
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
    plt.suptitle(title, y=0.97)

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
    plt.suptitle(title, y=0.97)

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


def get_plot_dataframe(
    input_df: pd.DataFrame, confidence_column: str, fdr_function: Callable
) -> pd.DataFrame:
    """Create a dataframe for FDR plotting with true and estimated FDR values.

    Args:
        input_df: DataFrame containing confidence scores and correct labels
        confidence_column: Name of the column containing confidence scores
        fdr_function: Function to calculate estimated FDR

    Returns:
        DataFrame with confidence, FDR, and source columns
    """
    sorted_df = input_df.sort_values(ascending=False, by=[confidence_column])
    cum_correct = np.cumsum(np.array(sorted_df["correct"]))
    cum_counts = np.arange(1, len(input_df) + 1)
    true_fdr = (cum_counts - cum_correct) / cum_counts
    estimated_fdr = fdr_function(sorted_df[confidence_column])
    multi_plot_df = pd.DataFrame(
        {
            "confidence": pd.concat(
                [sorted_df[confidence_column], sorted_df[confidence_column]]
            ),
            "fdr": true_fdr.tolist() + estimated_fdr.tolist(),
            "source": true_fdr.shape[0] * ["true"]
            + estimated_fdr.shape[0] * ["estimate"],
        }
    )
    return multi_plot_df


def get_confidence_threshold(dataframe: pd.DataFrame) -> float:
    """Get confidence threshold where FDR crosses 0.05.

    Args:
        dataframe: DataFrame containing confidence and FDR values

    Returns:
        Confidence threshold value
    """
    sorted_df = dataframe.sort_values(by=["confidence"])
    idxs = np.where(np.diff(np.sign(np.array(sorted_df["fdr"]) - 0.05)) != 0)

    # Check if we found any crossings
    if len(idxs[0]) == 0:
        # If no crossing found, return the minimum confidence value
        # This happens when FDR never crosses 0.05
        if sorted_df["fdr"].min() < 0.05:
            return sorted_df["confidence"].min()
        else:
            return 1.0

    return sorted_df["confidence"].values[idxs[0][0] + 1].item()


def plot_fdr_accuracy_on_axes(
    metadata: pd.DataFrame,
    ax: plt.Axes,
    fdr_function: Callable,
    confidence_column: str = "confidence",
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
        input_df=metadata,
        confidence_column=confidence_column,
        fdr_function=fdr_function,
    )

    # Get confidence thresholds
    cutoffs = (
        multi_plot_df.groupby("source")
        .apply(get_confidence_threshold)
        .to_frame(name="value")
        .reset_index()
    )

    # Plot FDR lines for each source
    for source in multi_plot_df["source"].unique():
        source_data = multi_plot_df[multi_plot_df["source"] == source]
        if source == "true":
            color = "#4A90E2"  # Muted blue
            label = "True FDR"
        else:
            color = "#F5A623"  # Muted gold
            label = "Estimated FDR"
        ax.plot(
            source_data["confidence"],
            source_data["fdr"],
            label=label,
            linewidth=2,
            color=color,
        )

    # Add horizontal line at FDR = 0.05
    ax.axhline(y=0.05, color="black", linestyle="--", alpha=0.7, linewidth=1.5)

    # Add vertical lines for confidence thresholds
    for _, row in cutoffs.iterrows():
        ax.axvline(
            x=row["value"], color="gray", linestyle=":", alpha=0.7, linewidth=1.5
        )

    # Customize the plot
    ax.set_xlabel("Confidence")
    ax.set_ylabel("False Discovery Rate (FDR)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set y-axis to start from 0 and go up to a reasonable max
    ax.set_ylim(0, min(1.0, multi_plot_df["fdr"].max() * 1.1))


def plot_fdr_accuracy(
    metadata: pd.DataFrame,
    title: str,
    fdr_function: Callable,
    confidence_column: str = "confidence",
    label_column: str = "correct",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot FDR accuracy comparison.

    Args:
        metadata: DataFrame containing confidence scores and labels
        title: Title for the plot
        fdr_function: Function to calculate estimated FDR
        confidence_column: Name of the column containing confidence scores
        label_column: Name of the column containing boolean labels

    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_fdr_accuracy_on_axes(metadata, ax, fdr_function, confidence_column, title)
    plt.tight_layout()
    return fig, ax


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
                    "species_display": SPECIES_NAME_MAPPING.get(species, species),
                    "winnow_positive": winnow_positive,
                    "winnow_negative": winnow_negative,
                    "dbg_positive": dbg_positive,
                    "dbg_negative": dbg_negative,
                }
            )

        except FileNotFoundError:
            print(
                f"Warning: Data file not found for species {SPECIES_NAME_MAPPING.get(species, species)}"
            )
            plot_data.append(
                {
                    "species": species,
                    "species_display": SPECIES_NAME_MAPPING.get(species, species),
                    "winnow_positive": 0,
                    "winnow_negative": 0,
                    "dbg_positive": 0,
                    "dbg_negative": 0,
                }
            )

    # Create DataFrame for plotting
    plot_df = pd.DataFrame(plot_data)

    # Create the plot with GridSpec for better layout control
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)

    # Main chart subplot
    ax = fig.add_subplot(gs[0])

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
    ax.set_xlabel("Species")
    ax.set_ylabel(f"Number of PSMs at {fdr_threshold * 100}% FDR")
    ax.set_title(config["title"].format(fdr_threshold * 100))

    # Set x-axis labels using mapped species names
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["species_display"], rotation=45, ha="right")

    # Add legend
    ax.legend()

    # Add grid
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("log")

    # Create a summary table in the second subplot
    table_data = []
    for i, _species in enumerate(species_list):
        # Calculate rates for each method
        np_total = (
            plot_df.iloc[i]["winnow_positive"] + plot_df.iloc[i]["winnow_negative"]
        )
        dbg_total = plot_df.iloc[i]["dbg_positive"] + plot_df.iloc[i]["dbg_negative"]

        np_rate = plot_df.iloc[i]["winnow_negative"] / np_total if np_total > 0 else 0
        dbg_rate = plot_df.iloc[i]["dbg_negative"] / dbg_total if dbg_total > 0 else 0

        table_data.append(
            [
                plot_df.iloc[i]["species_display"],
                f"{plot_df.iloc[i]['winnow_positive']}",
                f"{plot_df.iloc[i]['winnow_negative']}",
                f"{np_rate * 100:.1f}%",
                f"{plot_df.iloc[i]['dbg_positive']}",
                f"{plot_df.iloc[i]['dbg_negative']}",
                f"{dbg_rate * 100:.1f}%",
            ]
        )

    # Table subplot
    table_ax = fig.add_subplot(gs[1])
    table_ax.axis("tight")
    table_ax.axis("off")

    table = table_ax.table(
        cellText=table_data,
        colLabels=config["table_headers"],
        cellLoc="center",
        loc="center",
        colWidths=[0.12, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27],
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
        "title": "Comparison of PSMs Identified at {}% FDR: Non-Parametric FDR Estimation vs Database-Grounded FDR Estimation",
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
            print(
                f"Skipping {SPECIES_NAME_MAPPING.get(species, species)} - no raw data available"
            )

    config = {
        "file_path_template": "holdout_results/all_less_{species}_raw_test_results.csv",
        "label_column": "proteome_hit",
        "winnow_positive_label": "Non-Parametric Proteome Hits",
        "winnow_negative_label": "Non-Parametric No Hits",
        "dbg_positive_label": "Database-Grounded Proteome Hits",
        "dbg_negative_label": "Database-Grounded No Hits",
        "title": "Comparison of Proteome Hits at {}% FDR: Non-Parametric FDR Estimation vs Database-Grounded FDR Estimation",
        "table_headers": [
            "Species",
            "Non-Parametric Proteome Hits",
            "Non-Parametric No Hits",
            "Non-Parametric Error Rate",
            "Database-Grounded Proteome Hits",
            "Database-Grounded No Hits",
            "Database-Grounded Error Rate",
        ],
    }

    return plot_comparison_bar_chart(
        species_with_raw_data,
        filtered_winnow_cutoffs,
        filtered_dbg_cutoffs,
        config,
        fdr_threshold,
    )


def plot_raw_confidence_comparison_bar_chart(
    species_list: List[str],
    raw_winnow_confidence_cutoffs: List[float],
    raw_dbg_confidence_cutoffs: List[float],
    fdr_threshold: float = 0.05,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a stacked bar chart comparing the number of proteome hits at given FDR threshold.

    Compares between Winnow calibrated confidence and database-grounded raw confidence methods for each species.
    Skips over species that don't have raw data files (immuno, woundfluids).

    Args:
        species_list: List of species names
        raw_winnow_confidence_cutoffs: List of confidence cutoffs at a 5% FDR for Winnow method (calibrated confidence)
        raw_dbg_confidence_cutoffs: List of confidence cutoffs at a 5% FDR for database-grounded method (raw confidence)
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
            filtered_dbg_cutoffs.append(raw_dbg_confidence_cutoffs[i])
        else:
            print(
                f"Skipping {SPECIES_NAME_MAPPING.get(species, species)} - no raw data available"
            )

    config = {
        "file_path_template": "holdout_results/all_less_{species}_raw_test_results.csv",
        "label_column": "proteome_hit",
        "winnow_positive_label": "Calibrated Non-Parametric Proteome Hits",
        "winnow_negative_label": "Calibrated Non-Parametric No Hits",
        "dbg_positive_label": "Raw Database-Grounded Proteome Hits",
        "dbg_negative_label": "Raw Database-Grounded No Hits",
        "title": "Comparison of Proteome Hits at {}% FDR: Calibrated Confidence and Non-Parametric FDR Estimation vs Raw Confidence and Database-Grounded FDR Estimation",
        "table_headers": [
            "Species",
            "Calibrated Non-Parametric Proteome Hits",
            "Calibrated Non-Parametric No Hits",
            "Calibrated Non-Parametric Error Rate",
            "Raw Database-Grounded Proteome Hits",
            "Raw Database-Grounded No Hits",
            "Raw Database-Grounded Error Rate",
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
                    "species_display": SPECIES_NAME_MAPPING.get(species, species),
                    "total_psms": total_rows,
                    "color": species_colors[i % len(species_colors)],
                }
            )

        except FileNotFoundError:
            print(
                f"Warning: Could not find data file for species {SPECIES_NAME_MAPPING.get(species, species)}"
            )
            species_data.append(
                {
                    "species": species,
                    "species_display": SPECIES_NAME_MAPPING.get(species, species),
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
        pad=20,
    )

    # Add legend with counts using mapped species names
    legend_labels = [
        f"{row['species_display']} ({int(row['total_psms'])} PSMs)"
        for _, row in df.iterrows()
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


def compute_auc_from_pr_curve(
    metadata: pd.DataFrame,
    confidence_column: str,
    label_column: str,
) -> float:
    """Compute Area Under the Curve for precision-recall curve.

    Args:
        metadata: DataFrame containing confidence scores and labels
        confidence_column: Name of the column containing confidence scores
        label_column: Name of the column containing boolean labels

    Returns:
        AUC value for the precision-recall curve
    """
    pr_data = compute_pr_curve(
        input_dataset=metadata,
        confidence_column=confidence_column,
        label_column=label_column,
        name="temp",
    )

    # Calculate AUC using trapezoidal rule
    auc_value = auc(pr_data["recall"], pr_data["precision"])
    return auc_value


def create_generalization_heatmap(
    species_list: List[str], label_column: str = "correct"
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a heatmap showing generalization performance across hold-one-out datasets.

    Args:
        species_list: List of species names
        label_column: Name of the column containing boolean labels

    Returns:
        Tuple of (figure, axes)
    """
    # Initialize results dictionary
    results: Dict[str, List[Union[str, float]]] = {
        "Species": [],
        "Original Confidence AUC-PR": [],
        "Calibrated Confidence AUC-PR": [],
    }

    # For each species (as test set), compute AUC for both confidence types
    for species in species_list:
        try:
            # Load test results for this holdout species
            if label_column == "correct":
                metadata = pd.read_csv(
                    f"holdout_results/all_less_{species}_labelled_test_results_with_db_fdr.csv"
                )
            else:
                # For proteome hits, use raw data
                metadata = pd.read_csv(
                    f"holdout_results/all_less_{species}_raw_test_results.csv"
                )

            # Compute AUC for original confidence
            original_auc = compute_auc_from_pr_curve(
                metadata, "confidence", label_column
            )

            # Compute AUC for calibrated confidence
            calibrated_auc = compute_auc_from_pr_curve(
                metadata, "calibrated_confidence", label_column
            )

            # Store results
            results["Species"].append(SPECIES_NAME_MAPPING.get(species, species))
            results["Original Confidence AUC-PR"].append(original_auc)
            results["Calibrated Confidence AUC-PR"].append(calibrated_auc)

        except FileNotFoundError:
            print(f"Warning: Could not find data file for species {species}")
            results["Species"].append(SPECIES_NAME_MAPPING.get(species, species))
            results["Original Confidence AUC-PR"].append(np.nan)
            results["Calibrated Confidence AUC-PR"].append(np.nan)

    # Create DataFrame for heatmap
    df = pd.DataFrame(results)
    df = df.set_index("Species")

    # Create the heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Create heatmap with custom colormap
    sns.heatmap(
        df.T,  # Transpose so methods are rows and species are columns
        annot=True,  # Show values in cells
        fmt=".3f",  # Format to 3 decimal places
        cmap="RdYlBu_r",  # Red-Yellow-Blue colormap (reversed)
        center=0.5,  # Center colormap at 0.5
        vmin=0,  # Minimum value
        vmax=1,  # Maximum value
        cbar_kws={"label": "AUC-PR"},
        ax=ax,
    )

    # Customize the plot
    ax.set_title(
        f"Generalization Performance: AUC-PR Across Hold-One-Out Datasets\n({label_column.title()} Labels)",
        pad=20,
    )
    ax.set_xlabel("Test Species (Hold-Out)")
    ax.set_ylabel("Confidence Method")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    return fig, ax


def nonparametric_calibrated_estimator(probabilities: pd.Series) -> pd.Series:
    """Non-parametric calibrated estimator for FDR.

    Args:
        probabilities: List of confidence scores

    Returns:
        List of estimated FDR values
    """
    error_probabilities = np.array(1 - probabilities)
    counts = np.arange(1, len(error_probabilities) + 1)
    cum_error_probabilities = np.cumsum(error_probabilities)
    false_discovery_rate = cum_error_probabilities / counts
    return false_discovery_rate


def create_combined_species_plot(
    metadata: pd.DataFrame,
    title: str,
    label_column: str = "correct",
    labelled: bool = False,
) -> Tuple[
    plt.Figure,
    Union[
        Tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes, plt.Axes, plt.Axes],
        Tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes, plt.Axes],
    ],
]:
    """Create a combined plot with PR curve, confidence distributions, probability calibration, and FDR accuracy for a species."""
    # Create a single figure with 4 rows and 2 columns
    fig = plt.figure(figsize=(15, 24))

    # Set the main title for the entire figure
    plt.suptitle(title, y=0.98)

    # Plot 1: PR curve (spans full width, top row)
    ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=2)
    plot_pr_curve_on_axes(metadata, ax1, "Precision-Recall Curve", label_column)

    # Plot 2: Confidence distribution (second row left)
    ax2 = plt.subplot2grid((4, 2), (1, 0))
    plot_confidence_distribution_on_axes(
        metadata,
        ax2,
        "confidence",
        "Confidence Distribution",
        label_column=label_column,
    )

    # Plot 3: Calibrated confidence distribution (second row right)
    ax3 = plt.subplot2grid((4, 2), (1, 1))
    plot_confidence_distribution_on_axes(
        metadata,
        ax3,
        "calibrated_confidence",
        "Calibrated Confidence Distribution",
        label_column=label_column,
    )

    # Plot 4: Original confidence calibration (third row left)
    ax4 = plt.subplot2grid((4, 2), (2, 0))
    plot_calibration_curve_on_axes(
        metadata, ax4, "confidence", "Confidence Calibration", label_column
    )

    # Plot 5: Calibrated confidence calibration (third row right)
    ax5 = plt.subplot2grid((4, 2), (2, 1))
    plot_calibration_curve_on_axes(
        metadata,
        ax5,
        "calibrated_confidence",
        "Calibrated Confidence Calibration",
        label_column,
    )

    if labelled:
        # Plot 6: FDR accuracy comparison (fourth row, spans full width)
        ax6 = plt.subplot2grid((4, 2), (3, 0), colspan=2)
        plot_fdr_accuracy_on_axes(
            metadata,
            ax6,
            nonparametric_calibrated_estimator,
            "calibrated_confidence",
            "FDR Accuracy (Calibrated Confidence)",
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # Add more space at the top

        return fig, (ax1, ax2, ax3, ax4, ax5, ax6)

    return fig, (ax1, ax2, ax3, ax4, ax5)


def load_confidence_cutoffs(
    species_list: List[str],
) -> Tuple[List[Tuple[float, float]], List[float], List[float]]:
    """Load confidence cutoffs from files generated during FDR analysis.

    Args:
        species_list: List of species names

    Returns:
        Tuple of (winnow_confidence_cutoffs, labelled_dbg_confidence_cutoffs, labelled_dbg_raw_confidence_cutoffs)
    """
    winnow_confidence_cutoffs = []
    labelled_dbg_confidence_cutoffs = []
    raw_dbg_confidence_cutoffs = []

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

        # Load DBG labelled confidence cutoff
        dbg_labelled_cutoff_path = (
            f"holdout_results/all_less_{species}_dbg_labelled_confidence_cutoff.txt"
        )
        try:
            with open(dbg_labelled_cutoff_path, "r") as f:
                dbg_labelled_cutoff = float(f.read().strip())
            labelled_dbg_confidence_cutoffs.append(dbg_labelled_cutoff)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find database-grounded labelled confidence cutoff file for {species}: {dbg_labelled_cutoff_path}"
            )

        # Load DBG raw confidence cutoff
        dbg_raw_cutoff_path = f"holdout_results/all_less_{species}_dbg_labelled_confidence_cutoff_raw_conf.txt"
        try:
            with open(dbg_raw_cutoff_path, "r") as f:
                dbg_raw_cutoff = float(f.read().strip())
            raw_dbg_confidence_cutoffs.append(dbg_raw_cutoff)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find database-grounded raw confidence cutoff file for {species}: {dbg_raw_cutoff_path}"
            )

    return (
        winnow_confidence_cutoffs,
        labelled_dbg_confidence_cutoffs,
        raw_dbg_confidence_cutoffs,
    )


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
    (
        labelled_and_raw_winnow_confidence_cutoffs,
        labelled_dbg_confidence_cutoffs,
        raw_dbg_confidence_cutoffs,
    ) = load_confidence_cutoffs(species_list)
    labelled_winnow_confidence_cutoffs = [
        cutoff[0] for cutoff in labelled_and_raw_winnow_confidence_cutoffs
    ]
    raw_winnow_confidence_cutoffs = [
        cutoff[1] for cutoff in labelled_and_raw_winnow_confidence_cutoffs
    ]

    print(
        f"Loaded Winnow confidence cutoffs: {labelled_and_raw_winnow_confidence_cutoffs}"
    )
    print(
        f"Loaded database-grounded confidence cutoffs: {labelled_dbg_confidence_cutoffs}"
    )

    for species in species_list:
        # Load labelled metadata
        labelled_metadata = pd.read_csv(
            f"holdout_results/all_less_{species}_labelled_test_results_with_db_fdr.csv"
        )

        # Create combined labelled data plot for the species (uses "correct" column)
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = create_combined_species_plot(
            labelled_metadata,
            f"{SPECIES_NAME_MAPPING.get(species, species)} (Labelled)",
            "correct",
            labelled=True,
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

                # Create combined raw data plot for the species (uses "proteome_hit" column)
                fig, (ax1, ax2, ax3, ax4, ax5) = create_combined_species_plot(
                    raw_metadata,
                    f"{SPECIES_NAME_MAPPING.get(species, species)} (Raw)",
                    "proteome_hit",
                    labelled=False,
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
                print(
                    f"Warning: Raw data file not found for species {SPECIES_NAME_MAPPING.get(species, species)}"
                )
        else:
            print(
                f"Skipping raw combined plot for {SPECIES_NAME_MAPPING.get(species, species)} - no raw data available"
            )

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

    # Create raw confidence comparison bar chart
    fig, ax = plot_raw_confidence_comparison_bar_chart(
        species_list=species_list,
        raw_winnow_confidence_cutoffs=raw_winnow_confidence_cutoffs,
        raw_dbg_confidence_cutoffs=raw_dbg_confidence_cutoffs,
    )
    fig.savefig(
        "holdout_results/plots/raw_confidence_comparison_bar_chart.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        "holdout_results/plots/raw_confidence_comparison_bar_chart.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # Create generalization heatmap for PSMs
    fig, ax = create_generalization_heatmap(
        species_list=species_list, label_column="correct"
    )
    fig.savefig(
        "holdout_results/plots/generalization_heatmap_psms.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        "holdout_results/plots/generalization_heatmap_psms.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # Create generalization heatmap for Proteome Hits
    fig, ax = create_generalization_heatmap(
        species_list=species_list, label_column="proteome_hit"
    )
    fig.savefig(
        "holdout_results/plots/generalization_heatmap_proteome_hits.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        "holdout_results/plots/generalization_heatmap_proteome_hits.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
