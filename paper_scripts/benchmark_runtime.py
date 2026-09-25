#!/usr/bin/env python3
"""Benchmark wall-clock time and memory for the winnow prediction pipeline.

Measures end-to-end processing time and peak memory, broken down by data loading,
per-feature computation, MLP calibration inference, and FDR / q-value computation.
Benchmarks full and/or no-Prosit feature configurations.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import resource
import shutil
import sys
import tempfile
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Sequence, Tuple

import polars as pl
import torch
import typer

from winnow.calibration.calibrator import ProbabilityCalibrator
from winnow.calibration.features.fragment_match import FragmentMatchFeatures
from winnow.calibration.features.retention_time import RetentionTimeFeature
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

PROSIT_FEATURE_CLASSES = (FragmentMatchFeatures, RetentionTimeFeature)


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    """Timing and memory result for a single pipeline stage."""

    name: str
    device: str
    wall_time_s: float
    peak_mem_mb: float
    is_feature: bool = False
    is_prosit: bool = False
    columns: List[str] = field(default_factory=list)


@contextmanager
def measure():
    """Context manager that yields a dict populated with wall_time_s and peak_mem_mb on exit."""
    result: Dict[str, float] = {}
    tracemalloc.start()
    # Reset the peak so we measure only this block
    tracemalloc.reset_peak()
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        result["wall_time_s"] = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result["peak_mem_mb"] = peak / (1024 * 1024)


# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------


def get_hardware_info() -> Dict[str, str]:
    """Collect CPU, RAM, and GPU identifiers for the benchmark report."""
    info: Dict[str, str] = {}

    # CPU
    cpu_name = None
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_name = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    if not cpu_name:
        cpu_name = platform.processor() or "unknown"
    info["cpu"] = cpu_name
    info["cpu_cores"] = str(os.cpu_count() or "unknown")

    # RAM
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    info["ram_gb"] = f"{kb / (1024**2):.0f}"
                    break
    except OSError:
        info["ram_gb"] = "unknown"

    # GPU
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    else:
        info["gpu"] = "none"

    return info


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def write_combined_inputs(
    spectrum_paths: Sequence[Path],
    predictions_paths: Sequence[Path],
    output_dir: Path,
) -> Tuple[Path, Path, str]:
    """Concatenate spectrum/prediction inputs; return paths and a display label.

    A single pair is used as-is. Multiple pairs are joined with
    ``diagonal_relaxed`` so labelled (with ``sequence``) and unlabelled schemas
    can be combined.
    """
    if len(spectrum_paths) != len(predictions_paths):
        raise ValueError(
            "The number of --spectrum-path and --predictions-path values must match"
        )
    if not spectrum_paths:
        raise ValueError(
            "At least one --spectrum-path / --predictions-path pair is required"
        )

    if len(spectrum_paths) == 1:
        return spectrum_paths[0], predictions_paths[0], str(spectrum_paths[0])

    spectra = pl.concat(
        [pl.read_parquet(path) for path in spectrum_paths],
        how="diagonal_relaxed",
    )
    preds = pl.concat(
        [pl.read_csv(path) for path in predictions_paths],
        how="diagonal_relaxed",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "combined_spectra.parquet"
    pred_path = output_dir / "combined_preds.csv"
    spectra.write_parquet(spec_path)
    preds.write_csv(pred_path)
    label = " + ".join(str(path) for path in spectrum_paths)
    logger.info(
        "Combined %s spectra from %d inputs -> %s",
        f"{len(spectra):,}",
        len(spectrum_paths),
        spec_path,
    )
    return spec_path, pred_path, label


def load_dataset(
    spectrum_path: str,
    predictions_path: Optional[str],
    data_loader_name: str,
) -> CalibrationDataset:
    """Load and filter the dataset, returning a CalibrationDataset."""
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from winnow.utils.config_path import get_primary_config_dir

    primary_config_dir = get_primary_config_dir(None)
    overrides = [f"data_loader={data_loader_name}"]

    with initialize_config_dir(
        config_dir=str(primary_config_dir),
        version_base="1.3",
        job_name="benchmark",
    ):
        cfg = compose(config_name="predict", overrides=overrides)

    data_loader = instantiate(cfg.data_loader)
    dataset = data_loader.load(
        data_path=spectrum_path,
        predictions_path=predictions_path,
    )

    from winnow.scripts.main import _filter_dataset

    dataset = _filter_dataset(dataset)
    return dataset


def compute_features_individually(
    calibrator: ProbabilityCalibrator,
    dataset: CalibrationDataset,
) -> List[StageResult]:
    """Run each feature's prepare+compute with individual timing."""
    results: List[StageResult] = []

    # Dependencies (currently all features return [], but measure for completeness)
    for dep in calibrator.dependencies.values():
        with measure() as m:
            dep.compute(dataset=dataset)
        results.append(
            StageResult(
                name=f"Dependency: {dep.name}",
                device="CPU",
                wall_time_s=m["wall_time_s"],
                peak_mem_mb=m["peak_mem_mb"],
            )
        )

    for name, feat in calibrator.feature_dict.items():
        is_prosit = isinstance(feat, PROSIT_FEATURE_CLASSES)
        device = "CPU + network" if is_prosit else "CPU"

        with measure() as m:
            feat.prepare(dataset=dataset)
            feat.compute(dataset=dataset)

        results.append(
            StageResult(
                name=f"Feature: {name}",
                device=device,
                wall_time_s=m["wall_time_s"],
                peak_mem_mb=m["peak_mem_mb"],
                is_feature=True,
                is_prosit=is_prosit,
                columns=list(feat.columns),
            )
        )

    return results


def run_mlp_inference(
    calibrator: ProbabilityCalibrator,
    dataset: CalibrationDataset,
) -> StageResult:
    """Run MLP calibration inference."""
    if calibrator.network is None:
        raise RuntimeError("Calibrator network is not loaded")
    device = str(next(calibrator.network.parameters()).device)
    with measure() as m:
        calibrator.predict(dataset)
    return StageResult(
        name="MLP calibration inference",
        device=device.upper() if device == "cpu" else device,
        wall_time_s=m["wall_time_s"],
        peak_mem_mb=m["peak_mem_mb"],
    )


def run_fdr(
    dataset: CalibrationDataset,
    confidence_column: str = "calibrated_confidence",
    fdr_threshold: float = 0.05,
) -> StageResult:
    """Run FDR / q-value computation."""
    fdr_control = NonParametricFDRControl()

    with measure() as m:
        fdr_control.fit(dataset=dataset.metadata[confidence_column])
        dataset.metadata = fdr_control.add_psm_pep(dataset.metadata, confidence_column)
        dataset.metadata = fdr_control.add_psm_fdr(dataset.metadata, confidence_column)
        dataset.metadata = fdr_control.add_psm_q_value(
            dataset.metadata, confidence_column
        )
        confidence_cutoff = fdr_control.get_confidence_cutoff(threshold=fdr_threshold)
        _ = dataset.metadata[dataset.metadata[confidence_column] >= confidence_cutoff]

    return StageResult(
        name="FDR / q-value computation",
        device="CPU",
        wall_time_s=m["wall_time_s"],
        peak_mem_mb=m["peak_mem_mb"],
    )


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkRun:
    """Results from a single pipeline configuration."""

    config_label: str
    n_spectra: int
    n_features: int
    n_columns: int
    stages: List[StageResult]

    @property
    def total_wall_time_s(self) -> float:
        """Sum of wall times across all recorded stages."""
        return sum(s.wall_time_s for s in self.stages)

    @property
    def feature_wall_time_s(self) -> float:
        """Sum of wall times for feature computation stages only."""
        return sum(s.wall_time_s for s in self.stages if s.is_feature)

    @property
    def peak_mem_mb(self) -> float:
        """Maximum peak memory across stages, in megabytes."""
        return max(s.peak_mem_mb for s in self.stages) if self.stages else 0.0


def _model_matches_features(calibrator: ProbabilityCalibrator) -> bool:
    """Check whether the loaded MLP input dim matches the current feature set."""
    if calibrator.network is None or calibrator.feature_mean is None:
        return False
    expected_dim = calibrator.feature_mean.shape[0]
    actual_dim = 1 + len(calibrator.columns)  # confidence + feature columns
    return expected_dim == actual_dim


def _parse_koina_constants(raw: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    """Parse ``KEY=VALUE`` pairs into a dict, casting numeric strings."""
    if not raw:
        return None
    out: Dict[str, Any] = {}
    for item in raw:
        if "=" not in item:
            raise typer.BadParameter(
                f"Invalid --koina-input-constant format: '{item}'. Expected KEY=VALUE."
            )
        key, value = item.split("=", 1)
        try:
            out[key] = int(value)
        except ValueError:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
    return out


def run_benchmark(
    spectrum_path: str,
    predictions_path: Optional[str],
    model_path: str,
    data_loader_name: str,
    include_prosit: bool,
    koina_input_constants: Optional[Dict[str, Any]] = None,
) -> BenchmarkRun:
    """Execute the full prediction pipeline with per-stage timing."""
    config_label = "Full feature set" if include_prosit else "Without Prosit features"

    # Load calibrator
    calibrator = ProbabilityCalibrator.load(pretrained_model_name_or_path=model_path)

    if koina_input_constants:
        calibrator.apply_koina_model_input_overrides(
            model_input_constants=koina_input_constants,
        )

    # Remove Prosit features if requested
    if not include_prosit:
        to_remove = [
            name
            for name, feat in calibrator.feature_dict.items()
            if isinstance(feat, PROSIT_FEATURE_CLASSES)
        ]
        for name in to_remove:
            calibrator.remove_feature(name)

    stages: List[StageResult] = []

    # Stage 1: Data loading
    with measure() as m:
        dataset = load_dataset(spectrum_path, predictions_path, data_loader_name)
    n_spectra = len(dataset.metadata)
    stages.append(
        StageResult(
            name="Data loading",
            device="CPU",
            wall_time_s=m["wall_time_s"],
            peak_mem_mb=m["peak_mem_mb"],
        )
    )

    # Stage 2: Per-feature computation
    feature_results = compute_features_individually(calibrator, dataset)
    stages.extend(feature_results)

    n_features = len(calibrator.feature_dict)
    n_columns = len(calibrator.columns)

    # Stage 3: MLP calibration inference
    # The MLP input dimension must match the feature set. If features were
    # removed (e.g. Prosit features dropped) but the model was trained with
    # the full set, the dimensions won't match. In that case we skip MLP +
    # FDR and note the mismatch -- these stages are sub-millisecond anyway
    # and their cost is independent of the feature set used.
    can_infer = _model_matches_features(calibrator)
    if can_infer:
        stages.append(run_mlp_inference(calibrator, dataset))
        # Stage 4: FDR / q-value
        stages.append(run_fdr(dataset))
    else:
        mean_dim = (
            calibrator.feature_mean.shape[0]
            if calibrator.feature_mean is not None
            else "unknown"
        )
        print(
            f"  [note] MLP input dim ({mean_dim}) "
            f"does not match current feature count "
            f"({1 + len(calibrator.columns)}). "
            f"Skipping MLP inference and FDR for this configuration.\n"
            f"  To benchmark these stages, supply a model trained with the "
            f"matching feature set via --model-path-no-prosit."
        )

    return BenchmarkRun(
        config_label=config_label,
        n_spectra=n_spectra,
        n_features=n_features,
        n_columns=n_columns,
        stages=stages,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_run(run: BenchmarkRun) -> str:
    """Format a single benchmark run as a human-readable table."""
    lines: List[str] = []
    header = (
        f"=== Configuration: {run.config_label} "
        f"({run.n_features} features, {run.n_columns} columns) ==="
    )
    lines.append("")
    lines.append(header)
    lines.append("")

    col_w = [38, 16, 15, 15]
    hdr = (
        f"{'Stage':<{col_w[0]}}| {'Device':<{col_w[1]}}| "
        f"{'Wall time (s)':>{col_w[2]}}| {'Peak mem (MB)':>{col_w[3]}}"
    )
    sep = (
        "-" * col_w[0]
        + "|"
        + "-" * (col_w[1] + 1)
        + "|"
        + "-" * (col_w[2] + 1)
        + "|"
        + "-" * (col_w[3] + 1)
    )

    lines.append(hdr)
    lines.append(sep)

    feat_time = 0.0
    feat_mem = 0.0
    total_time = 0.0

    def _row(label: str, device: str, t: float, mem: float) -> str:
        return (
            f"{label:<{col_w[0]}}| {device:<{col_w[1]}}| "
            f"{t:>{col_w[2]}.2f}| {mem:>{col_w[3]}.1f}"
        )

    for i, s in enumerate(run.stages):
        lines.append(_row(s.name, s.device, s.wall_time_s, s.peak_mem_mb))
        total_time += s.wall_time_s
        if s.is_feature:
            feat_time += s.wall_time_s
            feat_mem = max(feat_mem, s.peak_mem_mb)

        is_last_feature = s.is_feature and not any(
            st.is_feature for st in run.stages[i + 1 :]
        )
        if is_last_feature:
            lines.append(
                _row(
                    "  Feature computation subtotal",
                    "",
                    feat_time,
                    feat_mem,
                )
            )

    lines.append(sep)
    total_mem = max(s.peak_mem_mb for s in run.stages) if run.stages else 0.0
    lines.append(_row("End-to-end total", "", total_time, total_mem))

    return "\n".join(lines)


def format_full_report(
    hw: Dict[str, str],
    mlp_device: str,
    runs: List[BenchmarkRun],
    n_spectra: int,
    spectrum_path: str,
) -> str:
    """Assemble the complete benchmark report."""
    lines: List[str] = []
    lines.append("=" * 88)
    lines.append("  Winnow Pipeline Runtime Benchmark")
    lines.append("=" * 88)
    lines.append("")
    lines.append(
        f"Hardware: {hw['cpu']} ({hw['cpu_cores']} cores), "
        f"{hw['ram_gb']} GB RAM, GPU: {hw['gpu']}"
    )
    lines.append(f"Dataset: {n_spectra:,} spectra from {spectrum_path}")
    lines.append(f"Calibrator MLP device: {mlp_device}")

    for run in runs:
        lines.append(format_run(run))

    lines.append("")
    lines.append("Notes:")
    lines.append(
        '- "CPU + network" = gRPC calls to a Koina/Triton server for'
        " Prosit-derived spectral predictions."
    )
    lines.append("  No local GPU is used by winnow during prediction.")
    lines.append(
        "- The MLP runs on CPU after loading. GPU is only used during training"
        " (not benchmarked here)."
    )
    lines.append(
        "- Koina predictions are not cached to disk between runs; batching is"
        " handled internally by koinapy."
    )
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    lines.append(f"- Process peak RSS: {peak_rss_mb:.1f} MB")
    lines.append("")
    return "\n".join(lines)


def build_json_report(
    hw: Dict[str, str],
    mlp_device: str,
    runs: List[BenchmarkRun],
    n_spectra: int,
    spectrum_path: str,
) -> Dict[str, Any]:
    """Build a structured dict suitable for JSON serialisation."""
    report: Dict[str, Any] = {
        "hardware": hw,
        "dataset": {
            "spectrum_path": spectrum_path,
            "n_spectra": n_spectra,
        },
        "mlp_device": mlp_device,
        "configurations": [],
    }
    for run in runs:
        cfg: Dict[str, Any] = {
            "label": run.config_label,
            "n_features": run.n_features,
            "n_columns": run.n_columns,
            "total_wall_time_s": run.total_wall_time_s,
            "feature_wall_time_s": run.feature_wall_time_s,
            "stages": [asdict(s) for s in run.stages],
        }
        report["configurations"].append(cfg)
    report["process_peak_rss_mb"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    )
    report["notes"] = {
        "koina_caching": (
            "Koina predictions are not cached to disk; each invocation "
            "re-queries the server."
        ),
        "koina_batching": (
            "Batching is handled internally by koinapy "
            "(gRPC streaming to the Koina/Triton server)."
        ),
        "gpu_usage": (
            "No local GPU is used during prediction. GPU is only used "
            "during training (not benchmarked here)."
        ),
    }
    return report


def _validate_runtime_cli_args(
    *,
    train_spectrum_path: Optional[Path],
    train_predictions_path: Optional[Path],
    val_spectrum_path: Optional[Path],
    val_predictions_path: Optional[Path],
    model_path_no_prosit: Optional[str],
    force_retrain: bool,
) -> bool:
    """Validate train/val CLI combinations. Returns whether all train paths were set."""
    train_args = (
        train_spectrum_path,
        train_predictions_path,
        val_spectrum_path,
        val_predictions_path,
    )
    train_provided = [p is not None for p in train_args]
    if any(train_provided) and not all(train_provided):
        raise typer.BadParameter(
            "Provide all of --train-spectrum-path, --train-predictions-path, "
            "--val-spectrum-path, and --val-predictions-path, or none of them"
        )
    if all(train_provided) and not model_path_no_prosit:
        raise typer.BadParameter(
            "--model-path-no-prosit is required when training or reusing a "
            "no-Prosit dummy (pass the directory to load/save)"
        )
    if force_retrain and not all(train_provided):
        raise typer.BadParameter(
            "--force-retrain requires labelled train/val paths for the dummy"
        )
    return all(train_provided)


def _ensure_no_prosit_model(
    *,
    train_all_provided: bool,
    train_spectrum_path: Optional[Path],
    train_predictions_path: Optional[Path],
    val_spectrum_path: Optional[Path],
    val_predictions_path: Optional[Path],
    model_path_no_prosit: Optional[str],
    model_path: str,
    data_loader: str,
    force_retrain: bool,
) -> str:
    """Train or reuse a no-Prosit calibrator; return the path to load for benchmarking."""
    if train_all_provided:
        assert train_spectrum_path is not None
        assert train_predictions_path is not None
        assert val_spectrum_path is not None
        assert val_predictions_path is not None
        assert model_path_no_prosit is not None
        train_or_load_dummy_calibrator(
            train_spectrum_path=train_spectrum_path,
            train_predictions_path=train_predictions_path,
            val_spectrum_path=val_spectrum_path,
            val_predictions_path=val_predictions_path,
            data_loader_name=data_loader,
            model_output_dir=Path(model_path_no_prosit),
            force_retrain=force_retrain,
        )
        return model_path_no_prosit
    return model_path_no_prosit or model_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    spectrum_path: Annotated[
        Optional[list[Path]],
        typer.Option(
            "--spectrum-path",
            help=(
                "Spectrum parquet; repeat to concatenate multiple splits "
                "(diagonal_relaxed)."
            ),
        ),
    ] = None,
    predictions_path: Annotated[
        Optional[list[Path]],
        typer.Option(
            "--predictions-path",
            help="Predictions CSV paired with each --spectrum-path.",
        ),
    ] = None,
    model_path: Annotated[
        str,
        typer.Option(
            "--model-path",
            help=(
                "Path to a local calibrator directory or HuggingFace model "
                "identifier (default: InstaDeepAI/winnow-general-model)."
            ),
        ),
    ] = "InstaDeepAI/winnow-general-model",
    model_path_no_prosit: Annotated[
        Optional[str],
        typer.Option(
            "--model-path-no-prosit",
            help=(
                "Directory for the no-Prosit calibrator. With labelled train/val "
                "paths, reuse a checkpoint here or train and save one. Without "
                "train/val paths, load an existing calibrator from this path "
                "(falls back to --model-path)."
            ),
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
            help="Train predictions CSV for the no-Prosit dummy calibrator.",
        ),
    ] = None,
    val_spectrum_path: Annotated[
        Optional[Path],
        typer.Option(
            "--val-spectrum-path",
            help="Labelled val spectra for the no-Prosit dummy calibrator.",
        ),
    ] = None,
    val_predictions_path: Annotated[
        Optional[Path],
        typer.Option(
            "--val-predictions-path",
            help="Val predictions CSV for the no-Prosit dummy calibrator.",
        ),
    ] = None,
    force_retrain: Annotated[
        bool,
        typer.Option(
            "--force-retrain/--reuse-model",
            help="Retrain the no-Prosit dummy even if a checkpoint already exists.",
        ),
    ] = False,
    data_loader: Annotated[
        str,
        typer.Option("--data-loader", help="Data loader to use (default: instanovo)."),
    ] = "instanovo",
    no_prosit: Annotated[
        bool,
        typer.Option(
            "--no-prosit",
            help=(
                "Only benchmark without Prosit/Koina features. When omitted, "
                "both configurations (full and no-Prosit) are benchmarked."
            ),
        ),
    ] = False,
    full_only: Annotated[
        bool,
        typer.Option(
            "--full-only",
            help="Only benchmark the full feature set (skip no-Prosit run).",
        ),
    ] = False,
    koina_input_constant: Annotated[
        Optional[List[str]],
        typer.Option(
            "--koina-input-constant",
            help=(
                "Koina model input constant as KEY=VALUE (repeatable). "
                "E.g. --koina-input-constant collision_energies=27 "
                "--koina-input-constant fragmentation_types=HCD"
            ),
        ),
    ] = None,
    output_json: Annotated[
        Optional[Path],
        typer.Option(
            "--output-json",
            help="Save structured results to a JSON file.",
        ),
    ] = None,
    output_text: Annotated[
        Optional[Path],
        typer.Option(
            "--output-text",
            help="Save the human-readable report table to a text file.",
        ),
    ] = None,
) -> None:
    """Run configured pipeline benchmarks and print (optionally save) results."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    hw = get_hardware_info()
    koina_constants = _parse_koina_constants(koina_input_constant)
    run_full = not no_prosit
    run_no_prosit = not full_only
    train_all_provided = _validate_runtime_cli_args(
        train_spectrum_path=train_spectrum_path,
        train_predictions_path=train_predictions_path,
        val_spectrum_path=val_spectrum_path,
        val_predictions_path=val_predictions_path,
        model_path_no_prosit=model_path_no_prosit,
        force_retrain=force_retrain,
    )

    spectrum_paths = spectrum_path or [Path("examples/example_data/spectra.ipc")]
    predictions_paths = predictions_path or [
        Path("examples/example_data/predictions.csv")
    ]

    tmpdir: Optional[Path] = None
    try:
        if len(spectrum_paths) > 1:
            tmpdir = Path(tempfile.mkdtemp(prefix="winnow_runtime_"))
            combined_dir = tmpdir
        else:
            combined_dir = Path(".")
        spectrum_file, predictions_file, spectrum_label = write_combined_inputs(
            spectrum_paths,
            predictions_paths,
            combined_dir,
        )
        spectrum_path_str = str(spectrum_file)
        predictions_path_str = str(predictions_file)

        probe_calibrator = ProbabilityCalibrator.load(
            pretrained_model_name_or_path=model_path
        )
        if probe_calibrator.network is None:
            raise RuntimeError("Calibrator network is not loaded")
        mlp_device = str(next(probe_calibrator.network.parameters()).device)
        del probe_calibrator

        runs: List[BenchmarkRun] = []
        n_spectra = 0

        if run_full:
            logger.info(">>> Benchmarking: Full feature set (including Prosit) ...")
            result = run_benchmark(
                spectrum_path=spectrum_path_str,
                predictions_path=predictions_path_str,
                model_path=model_path,
                data_loader_name=data_loader,
                include_prosit=True,
                koina_input_constants=koina_constants,
            )
            runs.append(result)
            n_spectra = result.n_spectra

        if run_no_prosit:
            logger.info(">>> Benchmarking: Without Prosit features ...")
            no_prosit_model = _ensure_no_prosit_model(
                train_all_provided=train_all_provided,
                train_spectrum_path=train_spectrum_path,
                train_predictions_path=train_predictions_path,
                val_spectrum_path=val_spectrum_path,
                val_predictions_path=val_predictions_path,
                model_path_no_prosit=model_path_no_prosit,
                model_path=model_path,
                data_loader=data_loader,
                force_retrain=force_retrain,
            )
            result = run_benchmark(
                spectrum_path=spectrum_path_str,
                predictions_path=predictions_path_str,
                model_path=no_prosit_model,
                data_loader_name=data_loader,
                include_prosit=False,
                koina_input_constants=koina_constants,
            )
            runs.append(result)
            n_spectra = n_spectra or result.n_spectra

        report = format_full_report(hw, mlp_device, runs, n_spectra, spectrum_label)
        print(report)

        if output_text is not None:
            output_text.parent.mkdir(parents=True, exist_ok=True)
            output_text.write_text(report + "\n")
            logger.info("Text report saved to %s", output_text)

        if output_json is not None:
            json_report = build_json_report(
                hw, mlp_device, runs, n_spectra, spectrum_label
            )
            output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json, "w") as f:
                json.dump(json_report, f, indent=2)
            logger.info("JSON results saved to %s", output_json)
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    app()
