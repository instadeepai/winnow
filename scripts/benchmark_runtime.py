#!/usr/bin/env python3
"""Benchmark wall-clock time and memory for the winnow prediction pipeline.

Measures end-to-end processing time and peak memory, broken down by:
  (i)   data loading,
  (ii)  per-feature computation (individually timed),
  (iii) MLP calibration inference, and
  (iv)  FDR / q-value computation.

Two configurations are benchmarked by default:
  1. Full feature set (including Prosit/Koina-derived features).
  2. Without Prosit features (fragment match + iRT removed).

This directly addresses the modularity claim: Prosit-based features can be
omitted without disrupting the pipeline.

Usage examples:
    # Both configurations on sample data (requires Koina server for full run)
    python scripts/benchmark_runtime.py

    # Only the no-Prosit configuration (no Koina server required)
    python scripts/benchmark_runtime.py --no-prosit

    # Custom dataset and locally-trained model
    python scripts/benchmark_runtime.py \
        --spectrum-path data/spectra.ipc \
        --predictions-path data/predictions.csv \
        --model-path models/my_model \
        --data-loader instanovo

    # Save structured results to JSON
    python scripts/benchmark_runtime.py --output-json results/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from winnow.calibration.calibrator import ProbabilityCalibrator
from winnow.calibration.features.fragment_match import FragmentMatchFeatures
from winnow.calibration.features.retention_time import RetentionTimeFeature
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.fdr.nonparametric import NonParametricFDRControl


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


def run_benchmark(
    spectrum_path: str,
    predictions_path: Optional[str],
    model_path: str,
    data_loader_name: str,
    include_prosit: bool,
    koina_url: Optional[str] = None,
    koina_ssl: Optional[bool] = None,
) -> BenchmarkRun:
    """Execute the full prediction pipeline with per-stage timing."""
    config_label = "Full feature set" if include_prosit else "Without Prosit features"

    # Load calibrator
    calibrator = ProbabilityCalibrator.load(pretrained_model_name_or_path=model_path)

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the runtime benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark winnow prediction pipeline runtime and memory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--spectrum-path",
        default="examples/example_data/spectra.ipc",
        help="Path to the spectrum data file (default: example data).",
    )
    parser.add_argument(
        "--predictions-path",
        default="examples/example_data/predictions.csv",
        help="Path to predictions file (default: example data).",
    )
    parser.add_argument(
        "--model-path",
        default="InstaDeepAI/winnow-general-model",
        help=(
            "Path to a local calibrator directory or HuggingFace model "
            "identifier (default: InstaDeepAI/winnow-general-model)."
        ),
    )
    parser.add_argument(
        "--model-path-no-prosit",
        default=None,
        help=(
            "Path to a calibrator trained without Prosit features, used for "
            "the no-Prosit benchmark. If omitted, --model-path is used and "
            "MLP/FDR stages are skipped when the input dimension mismatches."
        ),
    )
    parser.add_argument(
        "--data-loader",
        default="instanovo",
        help="Data loader to use (default: instanovo).",
    )
    parser.add_argument(
        "--no-prosit",
        action="store_true",
        help=(
            "Only benchmark without Prosit/Koina features. When omitted, "
            "both configurations (full and no-Prosit) are benchmarked."
        ),
    )
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Only benchmark the full feature set (skip no-Prosit run).",
    )
    parser.add_argument(
        "--koina-url",
        default=None,
        help="Koina server URL (e.g. localhost:8500).",
    )
    parser.add_argument(
        "--koina-ssl",
        default=None,
        type=lambda x: x.lower() in ("true", "1", "yes"),
        help="Use SSL for Koina (true/false).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Save structured results to a JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run configured pipeline benchmarks and print (optionally save) results."""
    args = parse_args()

    hw = get_hardware_info()

    # Determine which configs to run
    run_full = not args.no_prosit
    run_no_prosit = not args.full_only

    # Detect MLP device from a probe load
    probe_calibrator = ProbabilityCalibrator.load(
        pretrained_model_name_or_path=args.model_path
    )
    if probe_calibrator.network is None:
        raise RuntimeError("Calibrator network is not loaded")
    mlp_device = str(next(probe_calibrator.network.parameters()).device)
    del probe_calibrator

    runs: List[BenchmarkRun] = []
    n_spectra = 0

    if run_full:
        print("\n>>> Benchmarking: Full feature set (including Prosit) ...")
        result = run_benchmark(
            spectrum_path=args.spectrum_path,
            predictions_path=args.predictions_path,
            model_path=args.model_path,
            data_loader_name=args.data_loader,
            include_prosit=True,
            koina_url=args.koina_url,
            koina_ssl=args.koina_ssl,
        )
        runs.append(result)
        n_spectra = result.n_spectra

    if run_no_prosit:
        print("\n>>> Benchmarking: Without Prosit features ...")
        no_prosit_model = args.model_path_no_prosit or args.model_path
        result = run_benchmark(
            spectrum_path=args.spectrum_path,
            predictions_path=args.predictions_path,
            model_path=no_prosit_model,
            data_loader_name=args.data_loader,
            include_prosit=False,
            koina_url=args.koina_url,
            koina_ssl=args.koina_ssl,
        )
        runs.append(result)
        n_spectra = n_spectra or result.n_spectra

    report = format_full_report(hw, mlp_device, runs, n_spectra, args.spectrum_path)
    print(report)

    if args.output_json:
        json_report = build_json_report(
            hw, mlp_device, runs, n_spectra, args.spectrum_path
        )
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(json_report, f, indent=2)
        print(f"JSON results saved to {out_path}")


if __name__ == "__main__":
    main()
