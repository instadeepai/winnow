#!/usr/bin/env python3
"""Measure pipeline scaling without Koina/Prosit feature stages.

Trains a small dummy calibrator on Beam + Token Score + Mass Error (Da) using
HeLa (or any labelled) train/val inputs, then runs data loading, non-Koina
feature computation, MLP inference, and FDR at multiple subsampled sizes of the
benchmark spectrum set. Produces a JSON file and a matplotlib scaling figure.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import typer

from winnow.calibration.calibrator import ProbabilityCalibrator
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.fdr.nonparametric import NonParametricFDRControl

_PAPER_SCRIPTS = Path(__file__).resolve().parent
if str(_PAPER_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_PAPER_SCRIPTS))

from no_prosit_dummy import train_or_load_dummy_calibrator  # noqa: E402

logger = logging.getLogger(__name__)
app = typer.Typer(
    add_completion=False, pretty_exceptions_show_locals=False, no_args_is_help=True
)

plt.switch_backend("Agg")

# Paul Tol "bright" palette (colorblind-safe) — same as plot_eval_results.py
_PALETTE = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]

sns.set_theme(style="white", palette=_PALETTE, context="paper", font_scale=1.5)

DEFAULT_FRACTIONS = [0.1, 0.5, 1.0]
_SEED = 42
_DEFAULT_MODEL_OUTPUT_DIR = Path("paper_results/scaling/dummy_model")
_DEFAULT_RESULTS_DIR = Path("analysis")
_DEFAULT_PLOTS_DIR = Path("analysis")


@contextmanager
def measure():
    """Context manager that yields a dict with wall_time_s on exit."""
    result: Dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        result["wall_time_s"] = time.perf_counter() - t0


def write_subsampled_files(
    spectrum_paths: Sequence[Path],
    predictions_paths: Sequence[Path],
    fractions: Sequence[float],
    output_dir: Path,
    seed: int = _SEED,
) -> List[Tuple[float, Path, Path, int]]:
    """Write nested subsamples of aligned spectrum and prediction files.

    All input splits are combined, spectrum rows are shuffled once, and each
    fraction takes a prefix of that ordering. Predictions are filtered by
    spectrum ID so every benchmark size has aligned inputs.
    """
    if len(spectrum_paths) != len(predictions_paths):
        raise ValueError(
            "The number of --spectrum-path and --predictions-path values must match"
        )

    spectra = pl.concat(
        [pl.read_parquet(path) for path in spectrum_paths],
        how="diagonal_relaxed",
    )
    preds = pl.concat(
        [pl.read_csv(path) for path in predictions_paths],
        how="diagonal_relaxed",
    )
    n = len(spectra)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    outputs: List[Tuple[float, Path, Path, int]] = []
    for fraction in sorted(fractions):
        k = n if fraction >= 1.0 else max(1, int(n * fraction))
        spectra_sub = spectra[indices[:k]]
        keep_ids = set(spectra_sub["spectrum_id"].to_list())
        preds_sub = preds.filter(pl.col("spectrum_id").is_in(keep_ids))

        spec_path = output_dir / f"spectra_{fraction:.2f}.parquet"
        pred_path = output_dir / f"preds_{fraction:.2f}.csv"
        spectra_sub.write_parquet(spec_path)
        preds_sub.write_csv(pred_path)
        outputs.append((fraction, spec_path, pred_path, len(spectra_sub)))

    return outputs


def load_dataset(
    spectrum_path: str,
    predictions_path: str,
    data_loader_name: str,
) -> CalibrationDataset:
    """Load a dataset through the package data loader and filter invalid rows."""
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from winnow.scripts.main import _filter_dataset
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
    dataset = data_loader.load(
        data_path=spectrum_path,
        predictions_path=predictions_path,
    )
    return _filter_dataset(dataset)


@dataclass
class ScalingPoint:
    """Timing measurements for one dataset-size fraction."""

    fraction: float
    n_spectra: int
    stage_times: Dict[str, float] = field(default_factory=dict)


def run_at_size(
    spec_path: Path,
    pred_path: Path,
    calibrator: ProbabilityCalibrator,
    data_loader_name: str,
    fraction: float,
) -> ScalingPoint:
    """Run the full pipeline from disk at a given dataset size."""
    stage_times: Dict[str, float] = {}

    with measure() as m:
        dataset = load_dataset(str(spec_path), str(pred_path), data_loader_name)
    stage_times["Data loading"] = m["wall_time_s"]
    n = len(dataset.metadata)

    feature_total = 0.0
    for name, feat in calibrator.feature_dict.items():
        with measure() as m:
            feat.prepare(dataset=dataset)
            feat.compute(dataset=dataset)
        stage_times[f"Feature: {name}"] = m["wall_time_s"]
        feature_total += m["wall_time_s"]
    stage_times["Feature computation (total)"] = feature_total

    with measure() as m:
        calibrator.predict(dataset)
    stage_times["MLP inference"] = m["wall_time_s"]

    fdr = NonParametricFDRControl()
    col = "calibrated_confidence"
    with measure() as m:
        fdr.fit(dataset=dataset.metadata[col])
        dataset.metadata = fdr.add_psm_pep(dataset.metadata, col)
        dataset.metadata = fdr.add_psm_fdr(dataset.metadata, col)
        dataset.metadata = fdr.add_psm_q_value(dataset.metadata, col)
    stage_times["FDR / q-value"] = m["wall_time_s"]

    stage_times["End-to-end"] = (
        stage_times["Data loading"]
        + feature_total
        + stage_times["MLP inference"]
        + stage_times["FDR / q-value"]
    )

    return ScalingPoint(fraction=fraction, n_spectra=n, stage_times=stage_times)


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
            color=colour,
            label=label,
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
    ax.set_ylim(top=max_total * 1.05)
    ax.set_title("Pipeline scaling\nexcluding Koina-dependent features")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    base = Path(str(output_path).removesuffix(".png").removesuffix(".pdf"))
    _save_fig(fig, base)
    logger.info("Wrote %s.png/.pdf", base)


def load_points_from_json(json_path: Path) -> List[ScalingPoint]:
    """Load previously saved scaling measurements from JSON."""
    with open(json_path) as f:
        data = json.load(f)
    return [
        ScalingPoint(
            fraction=p["fraction"],
            n_spectra=p["n_spectra"],
            stage_times=p["stage_times"],
        )
        for p in data["points"]
    ]


@app.command()
def main(
    replot_json: Annotated[
        Optional[Path],
        typer.Option(
            "--replot-json",
            help=(
                "Replot from a saved benchmark_scaling.json without rerunning "
                "benchmarks."
            ),
            metavar="PATH",
        ),
    ] = None,
    spectrum_path: Annotated[
        Optional[list[Path]],
        typer.Option(
            "--spectrum-path",
            help="Benchmark spectrum parquet; repeat to combine multiple splits.",
        ),
    ] = None,
    predictions_path: Annotated[
        Optional[list[Path]],
        typer.Option(
            "--predictions-path",
            help="Predictions CSV paired with each --spectrum-path.",
        ),
    ] = None,
    train_spectrum_path: Annotated[
        Optional[Path],
        typer.Option(
            "--train-spectrum-path",
            help="Labelled train spectra for the no-Prosit dummy calibrator.",
        ),
    ] = None,
    train_predictions_path: Annotated[
        Optional[Path],
        typer.Option(
            "--train-predictions-path",
            help="Train predictions CSV for the dummy calibrator.",
        ),
    ] = None,
    val_spectrum_path: Annotated[
        Optional[Path],
        typer.Option(
            "--val-spectrum-path",
            help="Labelled val spectra for the dummy calibrator.",
        ),
    ] = None,
    val_predictions_path: Annotated[
        Optional[Path],
        typer.Option(
            "--val-predictions-path",
            help="Val predictions CSV for the dummy calibrator.",
        ),
    ] = None,
    model_output_dir: Annotated[
        Path,
        typer.Option(
            "--model-output-dir",
            help="Where to save/reuse the no-Prosit dummy calibrator.",
        ),
    ] = _DEFAULT_MODEL_OUTPUT_DIR,
    force_retrain: Annotated[
        bool,
        typer.Option(
            "--force-retrain/--reuse-model",
            help="Retrain the dummy even if model-output-dir already has a checkpoint.",
        ),
    ] = False,
    train_only: Annotated[
        bool,
        typer.Option(
            "--train-only/--run-benchmark",
            help=(
                "Only train or reuse the no-Prosit dummy calibrator; skip the "
                "scaling benchmark."
            ),
        ),
    ] = False,
    data_loader: Annotated[
        str,
        typer.Option("--data-loader", help="Data loader to use (default: instanovo)."),
    ] = "instanovo",
    fractions: Annotated[
        Optional[list[float]],
        typer.Option(
            "--fractions",
            help="Dataset fractions to benchmark (default: 0.1 0.5 1.0).",
        ),
    ] = None,
    results_dir: Annotated[
        Path,
        typer.Option("--results-dir", help="Directory for benchmark_scaling.json."),
    ] = _DEFAULT_RESULTS_DIR,
    plots_dir: Annotated[
        Path,
        typer.Option("--plots-dir", help="Directory for benchmark_scaling.png."),
    ] = _DEFAULT_PLOTS_DIR,
) -> None:
    """Run scaling benchmarks or replot from a saved JSON file."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if replot_json is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_scaling(
            load_points_from_json(replot_json),
            plots_dir / "benchmark_scaling.png",
        )
        return

    required: list[tuple[str, Path | list[Path] | None]] = [
        ("--train-spectrum-path", train_spectrum_path),
        ("--train-predictions-path", train_predictions_path),
        ("--val-spectrum-path", val_spectrum_path),
        ("--val-predictions-path", val_predictions_path),
    ]
    if not train_only:
        required.extend(
            (
                ("--spectrum-path", spectrum_path),
                ("--predictions-path", predictions_path),
            )
        )
    missing = [name for name, value in required if value is None]
    if missing:
        raise typer.BadParameter(
            f"{', '.join(missing)} required unless --replot-json is set"
        )

    assert train_spectrum_path is not None
    assert train_predictions_path is not None
    assert val_spectrum_path is not None
    assert val_predictions_path is not None

    calibrator = train_or_load_dummy_calibrator(
        train_spectrum_path=train_spectrum_path,
        train_predictions_path=train_predictions_path,
        val_spectrum_path=val_spectrum_path,
        val_predictions_path=val_predictions_path,
        data_loader_name=data_loader,
        model_output_dir=model_output_dir,
        force_retrain=force_retrain,
    )
    if train_only:
        logger.info("Train-only complete; dummy at %s", model_output_dir)
        return

    assert spectrum_path is not None
    assert predictions_path is not None

    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    frac_list = sorted(fractions if fractions is not None else DEFAULT_FRACTIONS)
    tmpdir = Path(tempfile.mkdtemp(prefix="winnow_scaling_"))
    points: List[ScalingPoint] = []

    try:
        logger.info("Preparing subsampled files ...")
        file_info = write_subsampled_files(
            spectrum_path,
            predictions_path,
            frac_list,
            tmpdir,
            seed=_SEED,
        )
        for frac, spec_p, pred_p, n in file_info:
            logger.info(
                "  %.0f%%: %s spectra -> %s, %s",
                frac * 100,
                f"{n:,}",
                spec_p.name,
                pred_p.name,
            )
            logger.info(">>> Running at %.0f%% (%s spectra) ...", frac * 100, f"{n:,}")
            point = run_at_size(spec_p, pred_p, calibrator, data_loader, frac)
            points.append(point)
            logger.info(
                "    %s spectra -> %.2f s total",
                f"{point.n_spectra:,}",
                point.stage_times["End-to-end"],
            )
            for stage, t in point.stage_times.items():
                if stage not in ("Feature computation (total)", "End-to-end"):
                    logger.info("      %s: %.3f s", stage, t)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    json_path = results_dir / "benchmark_scaling.json"
    json_data: Dict[str, Any] = {
        "points": [
            {
                "fraction": p.fraction,
                "n_spectra": p.n_spectra,
                "stage_times": p.stage_times,
            }
            for p in points
        ],
        "dummy_model_dir": str(model_output_dir),
    }
    with open(json_path, "w") as handle:
        json.dump(json_data, handle, indent=2)
    logger.info("Wrote %s", json_path)

    plot_scaling(points, plots_dir / "benchmark_scaling.png")


if __name__ == "__main__":
    app()
