#!/usr/bin/env python3
"""Measure pipeline scaling by running the no-Prosit benchmark at multiple dataset sizes.

Writes subsampled spectrum and prediction files to a temporary directory, then
runs the full pipeline (data loading, feature computation, MLP inference, FDR)
from scratch at each size. Produces a JSON file with the raw measurements and
a matplotlib figure showing per-stage scaling.

Usage:
    python scripts/benchmark_scaling.py \
        --spectrum-path held_out_projects/.../dataset-helaqc-raw-0000-0001.parquet \
        --predictions-path held_out_projects/.../dataset-helaqc-raw-0000-0001.csv \
        --model-path models/benchmark_model_no_prosit \
        --output-dir analysis
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from winnow.calibration.calibrator import ProbabilityCalibrator
from winnow.calibration.features.fragment_match import FragmentMatchFeatures
from winnow.calibration.features.retention_time import RetentionTimeFeature
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.fdr.nonparametric import NonParametricFDRControl

plt.switch_backend("Agg")

# Paul Tol "bright" palette (colorblind-safe) — same as plot_eval_results.py
_PALETTE = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]

sns.set_theme(style="white", palette=_PALETTE, context="paper", font_scale=1.5)

PROSIT_FEATURE_CLASSES = (FragmentMatchFeatures, RetentionTimeFeature)

DEFAULT_FRACTIONS = [0.1, 0.5, 1.0]


@contextmanager
def measure():
    """Context manager that yields a dict populated with wall_time_s and peak_mem_mb on exit."""
    result: Dict[str, float] = {}
    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        result["wall_time_s"] = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result["peak_mem_mb"] = peak / (1024 * 1024)


def write_subsampled_files(
    spectrum_path: str,
    predictions_path: str,
    fraction: float,
    output_dir: Path,
    seed: int = 42,
) -> Tuple[Path, Path, int]:
    """Write subsampled spectrum and prediction files, returning paths and row count.

    Both files are joined on spectrum_id, subsampled together, and written out
    so the data loader sees consistent, smaller files.
    """
    spectra = pl.read_parquet(spectrum_path)
    preds = pl.read_csv(predictions_path)

    n = len(spectra)
    if fraction >= 1.0:
        k = n
        indices = list(range(n))
    else:
        k = max(1, int(n * fraction))
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(n), k))

    spectra_sub = spectra[indices]
    # Match predictions to the subsampled spectra by spectrum_id
    keep_ids = set(spectra_sub["spectrum_id"].to_list())
    preds_sub = preds.filter(pl.col("spectrum_id").is_in(keep_ids))

    spec_path = output_dir / f"spectra_{fraction:.2f}.parquet"
    pred_path = output_dir / f"preds_{fraction:.2f}.csv"
    spectra_sub.write_parquet(spec_path)
    preds_sub.write_csv(pred_path)

    return spec_path, pred_path, len(spectra_sub)


def load_dataset_timed(
    spectrum_path: str,
    predictions_path: str,
    data_loader_name: str,
) -> Tuple[CalibrationDataset, float]:
    """Load a dataset through the full data loader and return it with timing."""
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from winnow.utils.config_path import get_primary_config_dir

    primary_config_dir = get_primary_config_dir(None)
    overrides = [f"data_loader={data_loader_name}"]

    with initialize_config_dir(
        config_dir=str(primary_config_dir),
        version_base="1.3",
        job_name="benchmark_scaling",
    ):
        cfg = compose(config_name="predict", overrides=overrides)

    data_loader = instantiate(cfg.data_loader)

    with measure() as m:
        dataset = data_loader.load(
            data_path=spectrum_path,
            predictions_path=predictions_path,
        )
        from winnow.scripts.main import _filter_dataset

        dataset = _filter_dataset(dataset)

    return dataset, m["wall_time_s"]


@dataclass
class ScalingPoint:
    """Timing measurements for one dataset-size fraction."""

    fraction: float
    n_spectra: int
    stage_times: Dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0


def run_at_size(
    spec_path: Path,
    pred_path: Path,
    expected_n: int,
    calibrator: ProbabilityCalibrator,
    data_loader_name: str,
    fraction: float,
) -> ScalingPoint:
    """Run the full pipeline from disk at a given dataset size."""
    stage_times: Dict[str, float] = {}

    # Data loading (from subsampled files on disk)
    dataset, load_time = load_dataset_timed(
        str(spec_path), str(pred_path), data_loader_name
    )
    n = len(dataset.metadata)
    stage_times["Data loading"] = load_time

    # Feature computation (per feature)
    feature_total = 0.0
    for name, feat in calibrator.feature_dict.items():
        with measure() as m:
            feat.prepare(dataset=dataset)
            feat.compute(dataset=dataset)
        stage_times[f"Feature: {name}"] = m["wall_time_s"]
        feature_total += m["wall_time_s"]
    stage_times["Feature computation (total)"] = feature_total

    # MLP inference
    with measure() as m:
        calibrator.predict(dataset)
    stage_times["MLP inference"] = m["wall_time_s"]

    # FDR / q-value
    fdr = NonParametricFDRControl()
    col = "calibrated_confidence"
    with measure() as m:
        fdr.fit(dataset=dataset.metadata[col])
        dataset.metadata = fdr.add_psm_pep(dataset.metadata, col)
        dataset.metadata = fdr.add_psm_fdr(dataset.metadata, col)
        dataset.metadata = fdr.add_psm_q_value(dataset.metadata, col)
        cutoff = fdr.get_confidence_cutoff(threshold=0.05)
        _ = dataset.metadata[dataset.metadata[col] >= cutoff]
    stage_times["FDR / q-value"] = m["wall_time_s"]

    total = (
        load_time
        + feature_total
        + stage_times["MLP inference"]
        + stage_times["FDR / q-value"]
    )
    stage_times["End-to-end"] = total

    return ScalingPoint(
        fraction=fraction,
        n_spectra=n,
        stage_times=stage_times,
        total_time=total,
    )


def fit_exponent(sizes: List[int], times: List[float]) -> Tuple[float, float]:
    """Fit t = c * n^alpha in log-log space; return (alpha, R²)."""
    log_s = np.log(sizes)
    log_t = np.log(times)
    slope, intercept = np.polyfit(log_s, log_t, 1)
    predicted = slope * log_s + intercept
    ss_res = float(np.sum((log_t - predicted) ** 2))
    ss_tot = float(np.sum((log_t - np.mean(log_t)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, r2


def _save_fig(fig: plt.Figure, base_path: Path) -> None:
    """Save figure as both PNG and PDF."""
    fig.savefig(f"{base_path}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"{base_path}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_scaling(points: List[ScalingPoint], output_path: Path) -> None:
    """Plot per-stage wall time vs. dataset size."""
    sizes = [p.n_spectra for p in points]

    stages_to_plot = [
        ("Data loading", "Data loading", "v", "-", _PALETTE[0]),
        ("Feature computation", "Feature computation (total)", "o", "-", _PALETTE[2]),
        ("MLP inference", "MLP inference", "^", "-", _PALETTE[1]),
        ("FDR / q-value", "FDR / q-value", "D", "-", _PALETTE[3]),
        ("End-to-end", "End-to-end", "s", "--", _PALETTE[5]),
    ]

    fig, ax = plt.subplots(figsize=(6, 4))

    for label, key, marker, linestyle, colour in stages_to_plot:
        times = [p.stage_times[key] for p in points]
        ax.plot(
            sizes,
            times,
            marker=marker,
            linestyle=linestyle,
            label=label,
            color=colour,
            alpha=0.7,
        )

    max_size = max(sizes)
    max_total = max(p.stage_times["End-to-end"] for p in points)
    ref_sizes = np.linspace(0, max_size, 50)
    ref_times = max_total * (ref_sizes / max_size)
    ax.plot(
        ref_sizes,
        ref_times,
        ls=":",
        color=_PALETTE[6],
        linewidth=1,
        label="Linear reference",
    )

    ax.set_xlabel("Number of spectra")
    ax.set_ylabel("Wall time (s)")
    ax.set_ylim(top=300)
    ax.set_title("Pipeline scaling\nexcluding Koina-dependent features")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    base = output_path.with_suffix("")
    _save_fig(fig, base)
    print(f"Scaling plot saved to {base}.png and {base}.pdf")


def load_points_from_json(json_path: Path) -> List[ScalingPoint]:
    """Load previously saved scaling measurements from JSON."""
    with open(json_path) as f:
        data = json.load(f)
    return [
        ScalingPoint(
            fraction=p["fraction"],
            n_spectra=p["n_spectra"],
            stage_times=p["stage_times"],
            total_time=p["total_time"],
        )
        for p in data["points"]
    ]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the scaling benchmark."""
    parser = argparse.ArgumentParser(
        description="Measure pipeline scaling at multiple dataset sizes.",
    )
    parser.add_argument(
        "--replot-json",
        metavar="PATH",
        help="Replot from a saved benchmark_scaling.json without rerunning benchmarks.",
    )
    parser.add_argument(
        "--spectrum-path",
        help="Path to the spectrum data file.",
    )
    parser.add_argument(
        "--predictions-path",
        help="Path to predictions file.",
    )
    parser.add_argument(
        "--model-path",
        help="Path to a calibrator trained without Prosit features.",
    )
    parser.add_argument(
        "--data-loader",
        default="instanovo",
        help="Data loader to use (default: instanovo).",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=DEFAULT_FRACTIONS,
        help="Dataset fractions to benchmark (default: 0.1 0.5 1.0).",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis",
        help="Directory for output files.",
    )
    args = parser.parse_args()
    if args.replot_json is None:
        for name in ("spectrum_path", "predictions_path", "model_path"):
            if getattr(args, name) is None:
                parser.error(
                    f"--{name.replace('_', '-')} is required unless --replot-json is set"
                )
    return args


def main() -> None:
    """Run scaling benchmarks or replot from a saved JSON file."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.replot_json:
        replot_points = load_points_from_json(Path(args.replot_json))
        plot_scaling(replot_points, output_dir / "benchmark_scaling.png")
        return

    # Load calibrator (without Prosit features)
    calibrator = ProbabilityCalibrator.load(
        pretrained_model_name_or_path=args.model_path
    )
    to_remove = [
        name
        for name, feat in calibrator.feature_dict.items()
        if isinstance(feat, PROSIT_FEATURE_CLASSES)
    ]
    for name in to_remove:
        calibrator.remove_feature(name)

    fractions = sorted(args.fractions)
    tmpdir = Path(tempfile.mkdtemp(prefix="winnow_scaling_"))
    points: List[ScalingPoint] = []

    try:
        # Pre-write all subsampled files
        file_info: List[Tuple[float, Path, Path, int]] = []
        print("Preparing subsampled files ...")
        for frac in fractions:
            spec_p, pred_p, n = write_subsampled_files(
                args.spectrum_path, args.predictions_path, frac, tmpdir
            )
            file_info.append((frac, spec_p, pred_p, n))
            print(f"  {frac:.0%}: {n:,} spectra -> {spec_p.name}, {pred_p.name}")

        for frac, spec_p, pred_p, expected_n in file_info:
            print(f"\n>>> Running at {frac:.0%} ({expected_n:,} spectra) ...")
            point = run_at_size(
                spec_p, pred_p, expected_n, calibrator, args.data_loader, frac
            )
            points.append(point)
            print(f"    {point.n_spectra:,} spectra -> {point.total_time:.2f} s total")
            for stage, t in point.stage_times.items():
                if stage not in ("Feature computation (total)", "End-to-end"):
                    print(f"      {stage}: {t:.3f} s")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Save JSON
    json_path = output_dir / "benchmark_scaling.json"
    json_data: Dict[str, Any] = {
        "points": [
            {
                "fraction": p.fraction,
                "n_spectra": p.n_spectra,
                "stage_times": p.stage_times,
                "total_time": p.total_time,
            }
            for p in points
        ],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nScaling data saved to {json_path}")

    # Plot
    plot_path = output_dir / "benchmark_scaling.png"
    plot_scaling(points, plot_path)


if __name__ == "__main__":
    main()
