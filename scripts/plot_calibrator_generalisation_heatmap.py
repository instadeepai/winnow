"""Plot PR-AUC heatmaps for calibrator generalisation results.

Reads the combined CSV produced by ``evaluate_calibrator_generalisation.py``
and creates heatmaps comparing raw vs calibrated confidence PR-AUC values.
"""

import logging
import sys
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from rich.logging import RichHandler
import typer

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.calibrator_generalisation_utils import SPECIES_NAME_MAPPING  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("winnow.plot_generalisation_heatmap")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.addHandler(RichHandler())

# ---------------------------------------------------------------------------
# Style — Paul Tol "bright" palette + "sunset" diverging colourmap
# ---------------------------------------------------------------------------
_PALETTE = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]

_SUNSET_COLORS = [
    "#364B9A",
    "#4A7BB7",
    "#6EA6CD",
    "#98CAE1",
    "#C2E4EF",
    "#EAECCC",
    "#FEDA8B",
    "#FDB366",
    "#F67E4B",
    "#DD3D2D",
    "#A50026",
]
_BAD_COLOUR = "#FFFFFF"

sns.set_theme(style="white", palette=_PALETTE, context="paper", font_scale=2)


def _diverging_cmap() -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list("tol_sunset", _SUNSET_COLORS, N=256)
    cmap.set_bad(color=_BAD_COLOUR)
    return cmap


def _sequential_cmap() -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list(
        "tol_sunset_seq", _SUNSET_COLORS[5:], N=256
    )
    cmap.set_bad(color=_BAD_COLOUR)
    return cmap


# ---------------------------------------------------------------------------
# PR-AUC computation
# ---------------------------------------------------------------------------
def compute_pr_auc(
    input_dataset: pd.DataFrame,
    confidence_column: str,
    label_column: str,
) -> float:
    """Compute Area Under Curve for precision-recall curve."""
    if len(input_dataset) == 0:
        return 0.0

    sorted_data = input_dataset[[confidence_column, label_column]].sort_values(
        by=confidence_column, ascending=False
    )

    cum_correct = np.cumsum(sorted_data[label_column])
    precision = cum_correct / np.arange(1, len(sorted_data) + 1)
    recall = (
        cum_correct / cum_correct.iloc[-1]
        if cum_correct.iloc[-1] > 0
        else np.zeros_like(cum_correct)
    )

    if len(precision) < 2:
        return 0.0

    from sklearn.metrics import auc

    return auc(recall, precision)


# ---------------------------------------------------------------------------
# Heatmap creation
# ---------------------------------------------------------------------------
def _save_fig(fig: plt.Figure, base_path: Path) -> None:
    """Save figure as both PNG and PDF."""
    fig.savefig(f"{base_path}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"{base_path}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def create_auc_heatmap(
    auc_df: pd.DataFrame,
    output_path: Path,
    title: str = "Calibrator generalisation PR-AUC heatmap",
) -> None:
    """Create and save a heatmap of PR-AUC values."""
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        auc_df,
        annot=True,
        fmt=".3f",
        cmap=_sequential_cmap(),
        cbar_kws={"label": "PR-AUC"},
        square=True,
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Test dataset")
    ax.set_ylabel("Train dataset")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    base = str(output_path).removesuffix(".png")
    _save_fig(fig, Path(base))
    logger.info("Heatmap saved to %s", output_path)


def create_comparison_heatmaps(results_path: Path, output_dir: Path) -> None:
    """Create heatmaps comparing raw vs calibrated confidence PR-AUC values."""
    logger.info("Scanning results from %s", results_path)
    results = pl.scan_csv(results_path)

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
    logger.info("Trained datasets: %s", trained_datasets)
    logger.info("Test datasets: %s", test_datasets)

    trained_labels = [SPECIES_NAME_MAPPING.get(ds, ds) for ds in trained_datasets]
    test_labels = [SPECIES_NAME_MAPPING.get(ds, ds) for ds in test_datasets]

    # Compute PR-AUC matrices for both confidence types
    auc_matrices = {}
    for conf_type in ["confidence", "calibrated_confidence"]:
        auc_matrix = []
        for trained_dataset in trained_datasets:
            auc_row = []
            for test_dataset in test_datasets:
                logger.info(
                    "Computing PR-AUC (%s) for trained=%s, test=%s",
                    conf_type,
                    trained_dataset,
                    test_dataset,
                )
                subset = (
                    results.filter(
                        (pl.col("trained_on_dataset") == trained_dataset)
                        & (pl.col("test_dataset") == test_dataset)
                    )
                    .collect()
                    .to_pandas()
                )

                if len(subset) > 0:
                    auc_row.append(compute_pr_auc(subset, conf_type, "correct"))
                else:
                    auc_row.append(np.nan)
            auc_matrix.append(auc_row)

        auc_matrices[conf_type] = pd.DataFrame(
            auc_matrix, index=trained_labels, columns=test_labels
        )

    # Individual heatmaps
    for conf_type, auc_df in auc_matrices.items():
        conf_name = conf_type.replace("_", " ")
        output_path = (
            output_dir / f"calibrator_generalisation_{conf_type}_auc_heatmap.png"
        )
        create_auc_heatmap(
            auc_df,
            output_path,
            f"Calibrator generalisation {conf_name} PR-AUC",
        )

    # Difference heatmap (calibrated - raw)
    diff_matrix = auc_matrices["calibrated_confidence"] - auc_matrices["confidence"]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        diff_matrix,
        annot=True,
        fmt=".3f",
        cmap=_diverging_cmap(),
        center=0,
        cbar_kws={"label": r"PR-AUC difference $(\mathrm{calibrated} - \mathrm{raw})$"},
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Calibrator generalisation PR-AUC improvement")
    ax.set_xlabel("Test dataset")
    ax.set_ylabel("Train dataset")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    diff_base = output_dir / "calibrator_generalisation_auc_difference_heatmap"
    _save_fig(fig, diff_base)
    logger.info("Difference heatmap saved to %s", diff_base)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_DEFAULT_OUTPUT_DIR = Path("results/generalisation/plots")

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def main(
    results_path: Annotated[
        Path, typer.Option(help="Path to calibrator generalisation results CSV.")
    ],
    output_dir: Annotated[
        Path, typer.Option(help="Directory to save plots.")
    ] = _DEFAULT_OUTPUT_DIR,
) -> None:
    """Create PR-AUC heatmaps for calibrator generalisation results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        logger.error("Results file not found: %s", results_path)
        raise typer.Exit(1)

    logger.info("Loading results from: %s", results_path)
    logger.info("Saving plots to: %s", output_dir)

    create_comparison_heatmaps(results_path, output_dir)


if __name__ == "__main__":
    app()
