#!/usr/bin/env python3
"""Bar charts of ablation calibration metrics from ``ablation_summary.csv``.

Designed for publication main text: tail ECE at FDR operating points (and optionally
Brier) per feature-group config, with a reference line at the full ``All features`` model.
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

from scripts.plot_eval_results import (  # noqa: E402
    _display_name,
    _fit_database_grounded_fdr,
    _save_fig,
    _style_ax,
)
from winnow.fdr.nonparametric import NonParametricFDRControl  # noqa: E402

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

MetricName = Literal[
    "tail_ECE@5%FDR",
    "tail_ECE@10%FDR",
    "ECE",
    "Brier",
    "PR_AUC",
    "fdr_bias@5%FDR",
    "fdr_bias@10%FDR",
    "q_dev@5%FDR",
    "q_dev@10%FDR",
]

FDR_TAIL_THRESHOLDS: tuple[float, ...] = (0.05, 0.10)
TAIL_ECE_COLUMN_BY_THRESHOLD: dict[float, str] = {
    0.05: "tail_ECE@5%FDR",
    0.10: "tail_ECE@10%FDR",
}
Q_DEV_COLUMN_BY_THRESHOLD: dict[float, str] = {
    0.05: "q_dev@5%FDR",
    0.10: "q_dev@10%FDR",
}
FDR_BIAS_COLUMN_BY_THRESHOLD: dict[float, str] = {
    0.05: "fdr_bias@5%FDR",
    0.10: "fdr_bias@10%FDR",
}

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


def compute_ece(
    pred: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected calibration error."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(pred, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        avg_conf = pred[mask].mean()
        avg_acc = labels[mask].mean()
        ece += mask.sum() / len(pred) * abs(avg_conf - avg_acc)
    return float(ece)


def compute_tail_ece_at_fdr(
    pred: np.ndarray,
    labels: np.ndarray,
    fdr_threshold: float,
    *,
    fdr_ctrl: NonParametricFDRControl | None = None,
    n_bins: int = 10,
) -> float:
    """ECE among PSMs accepted at a non-parametric FDR threshold."""
    if len(pred) == 0:
        return float("nan")

    if fdr_ctrl is None:
        fdr_ctrl = NonParametricFDRControl()
        fdr_ctrl.fit(dataset=pd.Series(pred, name="score"))

    cutoff = fdr_ctrl.get_confidence_cutoff(threshold=fdr_threshold)
    if np.isnan(cutoff):
        return float("nan")

    mask = pred >= cutoff
    if not mask.any():
        return float("nan")

    return compute_ece(pred[mask], labels[mask], n_bins=n_bins)


def compute_tail_ece_at_fdr_thresholds(
    df: pd.DataFrame,
    *,
    confidence_col: str = "calibrated_confidence",
    label_col: str = "correct",
    fdr_thresholds: tuple[float, ...] = FDR_TAIL_THRESHOLDS,
) -> dict[float, float]:
    """Tail ECE at each non-parametric FDR operating point."""
    work = df[[confidence_col, label_col]].dropna()
    if work.empty:
        return {threshold: float("nan") for threshold in fdr_thresholds}

    pred = work[confidence_col].to_numpy(dtype=float)
    labels = work[label_col].to_numpy(dtype=float)

    fdr_ctrl = NonParametricFDRControl()
    fdr_ctrl.fit(dataset=work[confidence_col])

    return {
        threshold: compute_tail_ece_at_fdr(
            pred,
            labels,
            threshold,
            fdr_ctrl=fdr_ctrl,
        )
        for threshold in fdr_thresholds
    }


def compute_fdr_bias_at_fdr_thresholds(
    df: pd.DataFrame,
    *,
    confidence_col: str = "calibrated_confidence",
    label_col: str = "correct",
    fdr_thresholds: tuple[float, ...] = FDR_TAIL_THRESHOLDS,
) -> dict[float, float]:
    """Signed FDR bias at each NP-FDR cutoff; equal to empirical sTECE."""
    work = df[[confidence_col, label_col]].dropna()
    if work.empty:
        return {threshold: float("nan") for threshold in fdr_thresholds}

    scores = work[confidence_col].to_numpy(dtype=float)
    labels = work[label_col].to_numpy(dtype=float)

    fdr_ctrl = NonParametricFDRControl()
    fdr_ctrl.fit(dataset=work[confidence_col])

    results: dict[float, float] = {}
    for threshold in fdr_thresholds:
        cutoff = fdr_ctrl.get_confidence_cutoff(threshold=threshold)
        if np.isnan(cutoff):
            results[threshold] = float("nan")
            continue
        mask = scores >= cutoff
        if not mask.any():
            results[threshold] = float("nan")
            continue

        # E[1-S | S>=tau] - E[1-Y | S>=tau] = E[Y-S | S>=tau].
        results[threshold] = float(np.mean(labels[mask] - scores[mask]))
    return results


def compute_pr_auc(
    df: pd.DataFrame,
    confidence_col: str = "calibrated_confidence",
    label_col: str = "correct",
) -> float:
    """Area under the ablation PR curve (matches ``run_feature_ablations`` plots)."""
    work = df[[confidence_col, label_col]].dropna()
    if work.empty:
        return float("nan")

    sorted_data = work.sort_values(by=confidence_col, ascending=False)
    cum_correct = np.cumsum(sorted_data[label_col].values)
    precision = cum_correct / np.arange(1, len(sorted_data) + 1)
    total_correct = cum_correct[-1] if len(cum_correct) else 0
    if total_correct <= 0 or len(precision) < 2:
        return 0.0

    recall = cum_correct / total_correct
    from sklearn.metrics import auc

    return float(auc(recall, precision))


def _vectorized_psm_fdr(
    scores: np.ndarray,
    ctrl: NonParametricFDRControl,
) -> np.ndarray:
    """Map confidence scores to PSM FDR using a fitted controller."""
    conf = np.asarray(ctrl._confidence_scores, dtype=float)
    fdr = np.asarray(ctrl._fdr_values, dtype=float)
    scores = np.asarray(scores, dtype=float)
    idx = np.searchsorted(-conf, -scores, side="left")
    idx = np.clip(idx, 0, max(len(fdr) - 1, 0))
    if len(fdr) == 0:
        return np.ones_like(scores)

    out = fdr[idx]
    below = (idx == len(conf)) & (scores < conf[-1])
    above = (idx == 0) & (scores > conf[0])
    out[below] = 1.0
    out[above] = fdr[0]
    return out


def _vectorized_psm_q_values(
    scores: np.ndarray,
    ctrl: NonParametricFDRControl,
) -> np.ndarray:
    """Assign PSM q-values without per-row ``compute_fdr`` calls."""
    row_fdr = _vectorized_psm_fdr(scores, ctrl)
    order = np.argsort(-scores)
    sorted_fdr = row_fdr[order]
    q_sorted = np.empty_like(sorted_fdr)
    fdr_min = np.inf
    for i in range(len(sorted_fdr) - 1, -1, -1):
        current = sorted_fdr[i]
        if current > fdr_min:
            q_sorted[i] = fdr_min
        else:
            q_sorted[i] = current
            fdr_min = current
    q_values = np.empty_like(q_sorted)
    q_values[order] = q_sorted
    return q_values


def compute_q_value_deviations(
    df: pd.DataFrame,
    *,
    confidence_col: str = "calibrated_confidence",
    label_col: str = "correct",
    fdr_thresholds: tuple[float, ...] = FDR_TAIL_THRESHOLDS,
) -> dict[float, float]:
    """Mean absolute q-value deviation among NP-accepted PSMs at each FDR level."""
    work = df[[confidence_col, label_col]].dropna().copy()
    if work.empty or label_col not in work.columns:
        return {threshold: float("nan") for threshold in fdr_thresholds}

    np_fdr = NonParametricFDRControl()
    np_fdr.fit(dataset=work[confidence_col])

    dbg_ctrl = _fit_database_grounded_fdr(
        work,
        confidence_col=confidence_col,
        correct_col=label_col,
        drop=0 if len(work) <= 10 else 10,
    )

    scores = work[confidence_col].to_numpy(dtype=float)
    est_q = _vectorized_psm_q_values(scores, np_fdr)
    true_q = _vectorized_psm_q_values(scores, dbg_ctrl)
    deviations = np.abs(est_q - true_q)

    results: dict[float, float] = {}
    for threshold in fdr_thresholds:
        mask = est_q <= threshold
        if not mask.any():
            results[threshold] = float("nan")
        else:
            results[threshold] = float(np.mean(deviations[mask]))
    return results


def metrics_from_eval_parquet(path: Path) -> dict[str, float | str]:
    """Compute tail ECE, PR-AUC, and q-value metrics from one eval-results Parquet."""
    df = pd.read_parquet(path)
    config_name = str(df["config_name"].iloc[0])
    dataset_name = str(df["dataset_name"].iloc[0])
    meta = df.drop(columns=["config_name", "dataset_name"], errors="ignore")

    tail_ece = compute_tail_ece_at_fdr_thresholds(meta)
    fdr_bias = compute_fdr_bias_at_fdr_thresholds(meta)
    pr_auc = compute_pr_auc(meta)
    q_dev = compute_q_value_deviations(meta)

    return {
        "config": config_name,
        "dataset": dataset_name,
        TAIL_ECE_COLUMN_BY_THRESHOLD[0.05]: round(tail_ece[0.05], 5),
        TAIL_ECE_COLUMN_BY_THRESHOLD[0.10]: round(tail_ece[0.10], 5),
        FDR_BIAS_COLUMN_BY_THRESHOLD[0.05]: round(fdr_bias[0.05], 5),
        FDR_BIAS_COLUMN_BY_THRESHOLD[0.10]: round(fdr_bias[0.10], 5),
        "PR_AUC": round(pr_auc, 5),
        Q_DEV_COLUMN_BY_THRESHOLD[0.05]: round(q_dev[0.05], 5),
        Q_DEV_COLUMN_BY_THRESHOLD[0.10]: round(q_dev[0.10], 5),
    }


def enrich_summary_from_eval_results(
    summary: pd.DataFrame,
    eval_results_dir: Path,
    *,
    datasets: list[str] | None = None,
) -> pd.DataFrame:
    """Add PR-AUC and q-value deviation columns using saved eval Parquets."""
    if not eval_results_dir.is_dir():
        raise FileNotFoundError(eval_results_dir)

    metric_rows: list[dict[str, float | str]] = []
    for path in sorted(eval_results_dir.glob("*.parquet")):
        dataset_name = path.name.split("_", 1)[0]
        if datasets is not None and dataset_name not in datasets:
            continue
        metric_rows.append(metrics_from_eval_parquet(path))

    if not metric_rows:
        raise FileNotFoundError(
            f"No eval Parquets found under {eval_results_dir}"
            + (f" for datasets {datasets!r}" if datasets else "")
        )

    metrics_df = pd.DataFrame(metric_rows)
    merge_cols = ["config", "dataset"]
    extra_cols = [
        *TAIL_ECE_COLUMN_BY_THRESHOLD.values(),
        *FDR_BIAS_COLUMN_BY_THRESHOLD.values(),
        "PR_AUC",
        *Q_DEV_COLUMN_BY_THRESHOLD.values(),
        "tail_ECE",
    ]
    summary = summary.drop(columns=extra_cols, errors="ignore")
    return summary.merge(metrics_df, on=merge_cols, how="left")


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
    if metric == "tail_ECE@5%FDR":
        return "Tail ECE at 5% FDR"
    if metric == "tail_ECE@10%FDR":
        return "Tail ECE at 10% FDR"
    if metric == "ECE":
        return "ECE"
    if metric == "Brier":
        return "Brier score"
    if metric == "PR_AUC":
        return "PR-AUC"
    if metric == "fdr_bias@5%FDR":
        return "FDR bias (= sTECE) at 5% FDR"
    if metric == "fdr_bias@10%FDR":
        return "FDR bias (= sTECE) at 10% FDR"
    if metric == "q_dev@5%FDR":
        return "Mean |q-value deviation| at 5% FDR"
    return "Mean |q-value deviation| at 10% FDR"


def _metric_plot_title(metric: MetricName, dataset_display: str) -> str:
    """Publication title: full sentence, ECE capitalised."""
    if metric == "tail_ECE@5%FDR":
        title = (
            f"Tail expected calibration error among PSMs accepted at 5% FDR "
            f"on {dataset_display}"
        )
    elif metric == "tail_ECE@10%FDR":
        title = (
            f"Tail expected calibration error among PSMs accepted at 10% FDR "
            f"on {dataset_display}"
        )
    elif metric == "ECE":
        title = f"Expected calibration error (ECE) on {dataset_display}"
    elif metric == "Brier":
        title = f"Brier score on {dataset_display}."
    elif metric == "PR_AUC":
        title = f"Precision-recall AUC on {dataset_display}"
    elif metric == "fdr_bias@5%FDR":
        title = (
            f"FDR bias, equal to signed tail calibration error, "
            f"at 5% FDR on {dataset_display}"
        )
    elif metric == "fdr_bias@10%FDR":
        title = (
            f"FDR bias, equal to signed tail calibration error, "
            f"at 10% FDR on {dataset_display}"
        )
    elif metric == "q_dev@5%FDR":
        title = (
            f"Non-parametric q-value deviation from database-grounded q-values "
            f"at 5% FDR on {dataset_display}"
        )
    else:
        title = (
            f"Non-parametric q-value deviation from database-grounded q-values "
            f"at 10% FDR on {dataset_display}"
        )
    return _wrap_title_before_dataset(title)


def plot_ablation_calibration_bars(
    summary: pd.DataFrame,
    dataset: str,
    *,
    metric: MetricName = "tail_ECE@5%FDR",
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
    ] = "tail_ECE@5%FDR",
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for figure outputs"),
    ] = _DEFAULT_SUMMARY.parent / "plots",
    eval_results_dir: Annotated[
        Path | None,
        typer.Option(
            "--eval-results-dir",
            help="Optional eval_results/ directory to enrich summary before plotting",
        ),
    ] = None,
) -> None:
    """Plot ablation calibration bars for one dataset."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sns.set_theme(style="white", context="paper", font_scale=1.5)

    summary_df = load_ablation_summary(summary)
    if eval_results_dir is not None:
        summary_df = enrich_summary_from_eval_results(
            summary_df,
            eval_results_dir,
            datasets=[dataset],
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = dataset.lower().replace(" ", "_")
    metric_slug = metric.lower().replace("%", "pct").replace("@", "_at_")
    out_base = output_dir / f"ablation_{metric_slug}_{slug}"
    table = plot_ablation_calibration_bars(
        summary_df, dataset, metric=metric, output_path=out_base
    )
    print(table.to_string(index=False))


@app.command("recompute-summary")
def recompute_summary(
    eval_results_dir: Annotated[
        Path,
        typer.Option("--eval-results-dir", help="Directory of eval_results Parquets"),
    ],
    summary: Annotated[
        Path | None,
        typer.Option(
            "--summary",
            help="Existing ablation_summary.csv to merge with (optional)",
        ),
    ] = None,
    datasets: Annotated[
        list[str] | None,
        typer.Option(
            "--datasets",
            help="Restrict to these dataset keys (repeatable)",
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", help="Output CSV path"),
    ] = _DEFAULT_SUMMARY,
) -> None:
    """Recompute PR-AUC and q-value deviation columns from eval Parquets."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if summary is not None:
        base = load_ablation_summary(summary)
        if datasets is not None:
            base = base.loc[base["dataset"].isin(datasets)].copy()
        enriched = enrich_summary_from_eval_results(
            base,
            eval_results_dir,
            datasets=datasets,
        )
    else:
        rows = []
        for path in sorted(eval_results_dir.glob("*.parquet")):
            dataset_name = path.name.split("_", 1)[0]
            if datasets is not None and dataset_name not in datasets:
                continue
            rows.append(metrics_from_eval_parquet(path))
        if not rows:
            raise typer.BadParameter(f"No eval Parquets found under {eval_results_dir}")
        enriched = pd.DataFrame(rows)
        enriched = enriched.sort_values(["dataset", "config"]).reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output, index=False)
    logger.info("Wrote %s", output)
    print(enriched.to_string(index=False))


if __name__ == "__main__":
    app()
