#!/usr/bin/env python3
"""Bar charts of ablation calibration metrics from ``ablation_summary.csv``.

Designed for publication main text: tail-ECE (and optionally Brier) per feature-group
config, with a reference line at the full ``All features`` model.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.plot_eval_results import _display_name, _save_fig, _style_ax  # noqa: E402

# Paul Tol qualitative palette (colour-blind safe) — canonical ablation colours.
_ABLATION_PALETTE = [
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#EE7733",
    "#0077BB",
    "#33BBEE",
    "#CC3311",
]

# Ablation summary keys → ``plot_eval_results.DATASET_DISPLAY_NAMES`` keys.
_ABLATION_DATASET_KEYS: dict[str, str] = {
    "Arabidopsis": "01747_C01_P018218_S00_I00_N03_R1",
    "Astral": "astral",
    "HCT116": "20151020_QE3_UPLC8_DBJ_SA_HCT116_Rep2_46",
}

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

ABLATION_CONFIG_ORDER: list[str] = [
    "Confidence only",
    "Confidence + mass error",
    "Confidence + iRT error",
    "Confidence + token-level",
    "Confidence + beam search",
    "Confidence + fragment matching",
    "All features",
]


def assign_ablation_colors(config_names: list[str]) -> dict[str, str]:
    """Assign a unique colour per ablation config (no palette wrap)."""
    if len(config_names) > len(_ABLATION_PALETTE):
        raise ValueError(
            f"Need {len(config_names)} ablation colours but only "
            f"{len(_ABLATION_PALETTE)} defined."
        )
    return {name: _ABLATION_PALETTE[i] for i, name in enumerate(config_names)}


def ordered_ablation_configs(present: set[str]) -> list[str]:
    """Canonical config order for ablation plots and colour assignment."""
    ordered = [c for c in ABLATION_CONFIG_ORDER if c in present]
    extra = sorted(present - set(ordered))
    return ordered + extra


_CONFIG_SHORT_LABELS: dict[str, str] = {
    "Confidence only": "Confidence",
    "Confidence + mass error": "+ Mass",
    "Confidence + iRT error": "+ iRT",
    "Confidence + token-level": "+ Token",
    "Confidence + beam search": "+ Beam",
    "Confidence + fragment matching": "+ Fragment",
    "All features": "All features",
}

MetricName = Literal["tail_ECE", "ECE", "Brier"]

# Must match ``compute_tail_ece`` in ``run_feature_ablations.py``.
TAIL_ECE_FRACTION = 0.1

_DEFAULT_SUMMARY = (
    Path.home() / "Documents/winnow/new_eval_sets_plots/ablations/ablation_summary.csv"
)


def load_ablation_summary(path: Path) -> pd.DataFrame:
    """Load ``ablation_summary.csv`` or ``.json``."""
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".json":
        with open(path) as f:
            return pd.DataFrame(json.load(f))
    return pd.read_csv(path)


def _ablation_dataset_display(dataset: str) -> str:
    """Publication label via ``plot_eval_results._display_name``."""
    return _display_name(_ABLATION_DATASET_KEYS.get(dataset, dataset))


def _wrap_title_before_dataset(title: str, *, max_line: int = 52) -> str:
    """Break before ``on <dataset>`` when the title would be too wide."""
    marker = " on "
    if marker not in title or len(title) <= max_line:
        return title
    split = title.index(marker)
    return f"{title[:split]}\n{title[split + 1 :]}"


def _metric_axis_label(metric: MetricName) -> str:
    if metric == "tail_ECE":
        return "Tail ECE"
    if metric == "ECE":
        return "ECE"
    return "Brier score"


def _metric_plot_title(metric: MetricName, dataset_display: str) -> str:
    """Publication title: full sentence, ECE capitalised."""
    if metric == "tail_ECE":
        pct = int(TAIL_ECE_FRACTION * 100)
        title = (
            f"Tail ECE in the top {pct}% of PSMs ranked by calibrated score "
            f"on {dataset_display}"
        )
    elif metric == "ECE":
        title = f"Expected calibration error (ECE) on {dataset_display}"
    else:
        title = f"Brier score on {dataset_display}."
    return _wrap_title_before_dataset(title)


def plot_ablation_calibration_bars(
    summary: pd.DataFrame,
    dataset: str,
    *,
    metric: MetricName = "tail_ECE",
    output_path: Path,
    figsize: tuple[float, float] = (7.5, 4),
) -> pd.DataFrame:
    """Bar chart of *metric* for one dataset; returns the plotted slice."""
    ds = summary.loc[summary["dataset"] == dataset].copy()
    if ds.empty:
        available = sorted(summary["dataset"].unique())
        raise ValueError(f"No rows for dataset {dataset!r}. Available: {available}")

    configs = ordered_ablation_configs(set(ds["config"]))
    ds = ds.set_index("config").loc[configs].reset_index()
    if metric not in ds.columns:
        raise ValueError(f"Metric {metric!r} not in summary columns: {ds.columns}")

    values = ds[metric].to_numpy(dtype=float)
    all_features_value = float(ds.loc[ds["config"] == "All features", metric].iloc[0])

    colors = assign_ablation_colors(configs)
    short_labels = [_CONFIG_SHORT_LABELS.get(c, c) for c in configs]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(configs))
    bar_colors = [colors[c] for c in configs]
    ax.bar(x, values, color=bar_colors, edgecolor="black", linewidth=0.6, zorder=2)
    ax.axhline(
        all_features_value,
        color="#333333",
        linestyle="--",
        linewidth=1.2,
        zorder=1,
        label="All features",
    )

    display = _ablation_dataset_display(dataset)
    ax.set_ylabel(_metric_axis_label(metric))
    ax.set_xlabel("Calibrator feature groups")
    ax.set_title(_metric_plot_title(metric, display))
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=35, ha="right")
    ax.legend(loc="upper right")
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, output_path)
    logger.info("Wrote %s.png and %s.pdf", output_path, output_path)
    return ds[["config", metric]]


@app.command()
def main(
    summary: Annotated[
        Path,
        typer.Option("--summary", help="ablation_summary.csv or .json"),
    ] = _DEFAULT_SUMMARY,
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Dataset key in the summary table"),
    ] = "Arabidopsis",
    metric: Annotated[
        MetricName,
        typer.Option("--metric", help="Calibration metric to plot"),
    ] = "tail_ECE",
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for figure outputs"),
    ] = _DEFAULT_SUMMARY.parent / "plots",
) -> None:
    """Plot ablation calibration bars for one dataset."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sns.set_theme(style="white", context="paper", font_scale=1.5)

    summary_df = load_ablation_summary(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = dataset.lower().replace(" ", "_")
    out_base = output_dir / f"ablation_{metric.lower()}_{slug}"
    table = plot_ablation_calibration_bars(
        summary_df, dataset, metric=metric, output_path=out_base
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    app()
