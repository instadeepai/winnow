"""Manuscript-style validation for the peptide language model feature.

This script runs a bounded ablation benchmark suitable for Aichor:

* train on the HeLa QC labelled training split;
* compare confidence-only calibration against confidence + PLM features;
* evaluate on labelled HeLa test and external labelled datasets;
* report PR-AUC, Brier score, ECE, sTECE/TECE, and accepted counts.

Set ``--validation-mode in_domain_split`` to run an additional diagnostic that
splits each selected labelled dataset into calibration and held-out evaluation
partitions. That mode is useful for separating PLM feature usefulness from
cross-domain transfer from HeLa QC.

The default row caps keep the first Aichor run affordable. Set
``--max-train-rows 0 --max-eval-rows 0`` for full available labelled data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, brier_score_loss

from winnow.calibration.calibrator import ProbabilityCalibrator
from winnow.calibration.diagnostics import (
    DiagnosticArrays,
    run_calibration_diagnostic,
)
from winnow.calibration.features import PeptideLanguageModelFeature
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.datasets.data_loaders.instanovo import InstaNovoDatasetLoader
from winnow.fdr.nonparametric import NonParametricFDRControl


DATASET_FILES = {
    "helaqc_train": ("helaqc/train.parquet", "helaqc/instanovo/train_preds.csv"),
    "helaqc_test": ("helaqc/test.parquet", "helaqc/instanovo/test_preds.csv"),
    "celegans": (
        "general_model_evaluation/celegans/labelled/celegans.parquet",
        "general_model_evaluation/celegans/labelled/celegans_preds.csv",
    ),
    "immuno2": (
        "general_model_evaluation/immuno2/labelled/immuno2.parquet",
        "general_model_evaluation/immuno2/labelled/immuno2_preds.csv",
    ),
}

DATASET_GROUPS = {
    "helaqc": ("helaqc_train", "helaqc_test"),
    "celegans": ("celegans",),
    "immuno2": ("immuno2",),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for manuscript-style validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="InstaDeepAI/winnow-ms-datasets")
    parser.add_argument("--data-dir", default="/runs/winnow-ms-datasets")
    parser.add_argument("--output-dir", default="/runs/plm-validation")
    parser.add_argument("--datasets", default="helaqc,celegans,immuno2")
    parser.add_argument("--plm-backend", default="pepbert")
    parser.add_argument("--plm-feature-mode", default="embedding_summary")
    parser.add_argument("--plm-model-name-or-path", default=None)
    parser.add_argument("--plm-batch-size", type=int, default=32)
    parser.add_argument("--max-train-rows", type=int, default=3000)
    parser.add_argument("--max-eval-rows", type=int, default=3000)
    parser.add_argument(
        "--validation-mode",
        choices=["hela_transfer", "in_domain_split"],
        default="hela_transfer",
        help=(
            "hela_transfer trains on HeLa QC train and evaluates selected datasets. "
            "in_domain_split trains/evaluates on held-out splits within each selected "
            "labelled dataset."
        ),
    )
    parser.add_argument(
        "--split-train-fraction",
        type=float,
        default=0.5,
        help="Training fraction for --validation-mode in_domain_split.",
    )
    parser.add_argument("--fdr-thresholds", default="0.05,0.10")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 < args.split_train_fraction < 1.0:
        raise ValueError("--split-train-fraction must be between 0 and 1.")
    return args


def selected_datasets(value: str) -> list[str]:
    """Parse and validate selected dataset groups."""
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in DATASET_GROUPS]
    if unknown:
        raise ValueError(f"Unknown dataset names: {unknown}")
    return names


def required_dataset_files(datasets: Iterable[str]) -> list[str]:
    """Return dataset files needed for HeLa-transfer validation."""
    required: list[str] = ["helaqc_train"]
    for dataset in datasets:
        for dataset_file in DATASET_GROUPS[dataset]:
            if dataset_file not in required:
                required.append(dataset_file)
    return required


def required_dataset_files_for_mode(
    datasets: Iterable[str],
    validation_mode: str,
) -> list[str]:
    """Return dataset files needed for the selected validation mode."""
    if validation_mode == "hela_transfer":
        return required_dataset_files(datasets)

    required: list[str] = []
    for dataset in datasets:
        for dataset_file in DATASET_GROUPS[dataset]:
            if dataset_file not in required:
                required.append(dataset_file)
    return required


def download_datasets(
    repo_id: str,
    data_dir: Path,
    dataset_files: Iterable[str],
) -> None:
    """Download required validation dataset files from Hugging Face."""
    patterns: list[str] = []
    for dataset in dataset_files:
        patterns.extend(DATASET_FILES[dataset])
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=data_dir,
    )


def slim_predictions_csv(source: Path, destination: Path) -> Path:
    """Write a reduced predictions CSV with only loader-required columns."""
    if destination.exists():
        return destination

    available = pl.scan_csv(source, n_rows=0).collect_schema().names()
    prediction_column = first_available(available, ["predictions", "preds"])
    tokenised_column = first_available(
        available, ["predictions_tokenised", "preds_tokenised"]
    )
    log_probability_column = first_available(
        available, ["log_probs", "log_probability"]
    )
    selected = [
        pl.col("spectrum_id"),
        pl.col(prediction_column).alias("predictions"),
        pl.col(tokenised_column).alias("predictions_tokenised"),
        pl.col(log_probability_column).alias("log_probs"),
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    (
        pl.scan_csv(source)
        .select(selected)
        .collect(engine="streaming")
        .write_csv(destination)
    )
    return destination


def first_available(available: list[str], candidates: list[str]) -> str:
    """Return the first candidate column present in a schema."""
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ValueError(
        f"None of the expected columns {candidates} found. Available columns: {available}"
    )


def make_loader() -> InstaNovoDatasetLoader:
    """Build the InstaNovo dataset loader used by validation datasets."""
    cfg = OmegaConf.load("winnow/configs/residues.yaml")
    return InstaNovoDatasetLoader(
        residue_masses=OmegaConf.to_container(cfg.residue_masses, resolve=True),
        residue_remapping={},
        beam_columns=None,
    )


def load_dataset(data_dir: Path, dataset_name: str) -> CalibrationDataset:
    """Load one validation dataset by logical name."""
    spectrum_file, prediction_file = DATASET_FILES[dataset_name]
    slim_prediction_file = slim_predictions_csv(
        data_dir / prediction_file,
        data_dir / "slim" / prediction_file,
    )
    loader = make_loader()
    return loader.load(
        data_path=data_dir / spectrum_file,
        predictions_path=slim_prediction_file,
    )


def sample_metadata(
    metadata: pd.DataFrame,
    *,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    """Sample metadata rows when a validation cap is configured."""
    if max_rows <= 0 or len(metadata) <= max_rows:
        return metadata.copy(deep=True).reset_index(drop=True)
    return (
        metadata.sample(n=max_rows, random_state=seed)
        .sort_index()
        .reset_index(drop=True)
    )


def split_metadata(
    metadata: pd.DataFrame,
    *,
    train_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split metadata into train and held-out evaluation partitions."""
    train_index = metadata.sample(frac=train_fraction, random_state=seed).index
    train_metadata = metadata.loc[train_index].sort_index().reset_index(drop=True)
    eval_metadata = metadata.drop(index=train_index).sort_index().reset_index(drop=True)
    if len(train_metadata) == 0 or len(eval_metadata) == 0:
        raise ValueError(
            "Split produced an empty training or evaluation partition. "
            "Increase the dataset cap or adjust --split-train-fraction."
        )
    return train_metadata, eval_metadata


def copy_dataset_with_metadata(
    source: CalibrationDataset,
    metadata: pd.DataFrame,
) -> CalibrationDataset:
    """Create a calibration dataset copy with replacement metadata."""
    return CalibrationDataset(metadata=metadata.copy(deep=True), predictions=None)


def build_calibrator(
    variant: str,
    args: argparse.Namespace,
) -> ProbabilityCalibrator:
    """Build a confidence-only or confidence-plus-PLM calibrator."""
    features = []
    if variant == "confidence_plus_plm":
        features.append(
            PeptideLanguageModelFeature(
                backend=args.plm_backend,
                model_name_or_path=args.plm_model_name_or_path,
                feature_mode=args.plm_feature_mode,
                batch_size=args.plm_batch_size,
                learn_from_missing=True,
            )
        )
    elif variant != "confidence_only":
        raise ValueError(f"Unknown calibrator variant: {variant}")

    return ProbabilityCalibrator(
        seed=args.seed,
        features=features,
        hidden_layer_sizes=(32, 16),
        early_stopping=False,
        max_iter=200,
    )


def expected_calibration_error(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute equal-count expected calibration error."""
    order = np.argsort(scores)
    score_bins = np.array_split(scores[order], n_bins)
    label_bins = np.array_split(labels[order], n_bins)
    ece = 0.0
    for score_bin, label_bin in zip(score_bins, label_bins):
        if len(score_bin) == 0:
            continue
        weight = len(score_bin) / len(scores)
        ece += weight * abs(float(label_bin.mean()) - float(score_bin.mean()))
    return float(ece)


def fdr_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float | int | None]:
    """Compute FDR-control metrics at one threshold."""
    fdr = NonParametricFDRControl()
    fdr.fit(pd.Series(scores))
    cutoff = fdr.get_confidence_cutoff(threshold=threshold)
    if not np.isfinite(cutoff):
        return {
            "threshold": threshold,
            "confidence_cutoff": None,
            "accepted_psms": 0,
            "realised_fdr": None,
            "sTECE": None,
            "TECE": None,
            "n_tail": 0,
        }

    accepted = scores >= cutoff
    realised_fdr = float(1.0 - labels[accepted].mean()) if accepted.any() else None

    diagnostics = DiagnosticArrays.from_raw(scores, labels.astype(bool))
    try:
        diagnostic = run_calibration_diagnostic(
            data=diagnostics,
            conf_cutoff=cutoff,
            nominal_fdr=threshold,
            tolerance=0.005,
            label_source="sequence",
            label_column="correct",
            min_tail_psms=20,
        )
        stece = diagnostic.stece
        tece = diagnostic.tece
        n_tail = diagnostic.n_tail
    except ValueError:
        stece = None
        tece = None
        n_tail = int(accepted.sum())

    return {
        "threshold": threshold,
        "confidence_cutoff": float(cutoff),
        "accepted_psms": int(accepted.sum()),
        "realised_fdr": realised_fdr,
        "sTECE": stece,
        "TECE": tece,
        "n_tail": n_tail,
    }


def evaluate(
    calibrator: ProbabilityCalibrator,
    dataset: CalibrationDataset,
    thresholds: list[float],
) -> dict:
    """Evaluate one fitted calibrator on one labelled dataset."""
    scored = copy_dataset_with_metadata(dataset, dataset.metadata)
    calibrator.predict(scored)
    scores = scored.metadata["calibrated_confidence"].to_numpy(dtype=float)
    labels = scored.metadata["correct"].to_numpy(dtype=bool)
    result = {
        "n_psms": int(len(scored.metadata)),
        "n_correct": int(labels.sum()),
        "pr_auc": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels.astype(float), scores)),
        "ece": expected_calibration_error(scores, labels.astype(float)),
        "fdr": [fdr_metrics(scores, labels.astype(float), t) for t in thresholds],
    }
    return result


def run_hela_transfer(
    loaded: dict[str, CalibrationDataset],
    datasets: list[str],
    thresholds: list[float],
    args: argparse.Namespace,
) -> dict:
    """Run the HeLa-trained manuscript-style transfer benchmark."""
    train_source = loaded["helaqc_train"]
    train_metadata = sample_metadata(
        train_source.metadata,
        max_rows=args.max_train_rows,
        seed=args.seed,
    )
    train_dataset = copy_dataset_with_metadata(train_source, train_metadata)

    eval_datasets: dict[str, CalibrationDataset] = {}
    if "helaqc_test" in loaded:
        helaqc_test = loaded["helaqc_test"]
        test_metadata = sample_metadata(
            helaqc_test.metadata,
            max_rows=args.max_eval_rows,
            seed=args.seed,
        )
        eval_datasets["helaqc_test"] = copy_dataset_with_metadata(
            helaqc_test, test_metadata
        )

    for name, dataset in loaded.items():
        if name.startswith("helaqc_"):
            continue
        metadata = sample_metadata(
            dataset.metadata,
            max_rows=args.max_eval_rows,
            seed=args.seed,
        )
        eval_datasets[name] = copy_dataset_with_metadata(dataset, metadata)

    report = base_report(args, datasets, thresholds)
    report["train"] = {
        "dataset": "helaqc_train",
        "n_psms": int(len(train_dataset.metadata)),
        "n_correct": int(train_dataset.metadata["correct"].sum()),
    }
    report["variants"] = {}

    for variant in ["confidence_only", "confidence_plus_plm"]:
        print(
            f"Fitting {variant} on {len(train_dataset.metadata)} HeLa QC train PSMs",
            flush=True,
        )
        calibrator = build_calibrator(variant, args)
        calibrator.fit(
            copy_dataset_with_metadata(train_dataset, train_dataset.metadata)
        )
        print(f"Finished fitting {variant}", flush=True)
        report["variants"][variant] = {}
        for name, dataset in eval_datasets.items():
            print(
                f"Evaluating {variant} on {name} ({len(dataset.metadata)} PSMs)",
                flush=True,
            )
            report["variants"][variant][name] = evaluate(
                calibrator, dataset, thresholds
            )
            print(f"Finished evaluating {variant} on {name}", flush=True)

    return report


def run_in_domain_split(
    loaded: dict[str, CalibrationDataset],
    datasets: list[str],
    thresholds: list[float],
    args: argparse.Namespace,
) -> dict:
    """Run train/evaluation splits within each selected labelled dataset."""
    report = base_report(args, datasets, thresholds)
    report["splits"] = {}

    for dataset_group in datasets:
        for dataset_name in DATASET_GROUPS[dataset_group]:
            if dataset_name not in loaded:
                continue
            source = loaded[dataset_name]
            metadata = sample_metadata(
                source.metadata,
                max_rows=args.max_eval_rows,
                seed=args.seed,
            )
            train_metadata, eval_metadata = split_metadata(
                metadata,
                train_fraction=args.split_train_fraction,
                seed=args.seed,
            )
            train_dataset = copy_dataset_with_metadata(source, train_metadata)
            eval_dataset = copy_dataset_with_metadata(source, eval_metadata)

            split_report = {
                "train": {
                    "dataset": dataset_name,
                    "n_psms": int(len(train_dataset.metadata)),
                    "n_correct": int(train_dataset.metadata["correct"].sum()),
                },
                "eval": {
                    "dataset": dataset_name,
                    "n_psms": int(len(eval_dataset.metadata)),
                    "n_correct": int(eval_dataset.metadata["correct"].sum()),
                },
                "variants": {},
            }

            for variant in ["confidence_only", "confidence_plus_plm"]:
                print(
                    f"Fitting {variant} on {len(train_dataset.metadata)} "
                    f"{dataset_name} split-train PSMs",
                    flush=True,
                )
                calibrator = build_calibrator(variant, args)
                calibrator.fit(
                    copy_dataset_with_metadata(train_dataset, train_dataset.metadata)
                )
                print(f"Finished fitting {variant} on {dataset_name}", flush=True)
                print(
                    f"Evaluating {variant} on {dataset_name} split-eval "
                    f"({len(eval_dataset.metadata)} PSMs)",
                    flush=True,
                )
                split_report["variants"][variant] = evaluate(
                    calibrator,
                    eval_dataset,
                    thresholds,
                )
                print(
                    f"Finished evaluating {variant} on {dataset_name} split-eval",
                    flush=True,
                )

            report["splits"][dataset_name] = split_report

    return report


def base_report(
    args: argparse.Namespace,
    datasets: list[str],
    thresholds: list[float],
) -> dict:
    """Build the common report settings block."""
    return {
        "settings": {
            "datasets": datasets,
            "thresholds": thresholds,
            "max_train_rows": (
                args.max_train_rows if args.validation_mode == "hela_transfer" else None
            ),
            "max_eval_rows": args.max_eval_rows,
            "validation_mode": args.validation_mode,
            "split_train_fraction": args.split_train_fraction,
            "plm_backend": args.plm_backend,
            "plm_feature_mode": args.plm_feature_mode,
            "plm_model_name_or_path": args.plm_model_name_or_path,
            "plm_batch_size": args.plm_batch_size,
        },
    }


def write_report(report: dict, output_dir: Path) -> None:
    """Write JSON and CSV validation reports."""
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    rows = []
    if report["settings"]["validation_mode"] == "hela_transfer":
        for variant, datasets_report in report["variants"].items():
            for dataset_name, metrics in datasets_report.items():
                rows.append(
                    summary_row(
                        report,
                        variant=variant,
                        train_dataset=report["train"]["dataset"],
                        eval_dataset=dataset_name,
                        metrics=metrics,
                    )
                )
    else:
        for split_name, split_report in report["splits"].items():
            for variant, metrics in split_report["variants"].items():
                rows.append(
                    summary_row(
                        report,
                        variant=variant,
                        train_dataset=split_report["train"]["dataset"],
                        eval_dataset=split_name,
                        metrics=metrics,
                    )
                )

    pd.DataFrame(rows).to_csv(output_dir / "summary.csv", index=False)


def summary_row(
    report: dict,
    *,
    variant: str,
    train_dataset: str,
    eval_dataset: str,
    metrics: dict,
) -> dict:
    """Flatten one variant/dataset result into a CSV summary row."""
    row = {
        "validation_mode": report["settings"]["validation_mode"],
        "variant": variant,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "n_psms": metrics["n_psms"],
        "n_correct": metrics["n_correct"],
        "pr_auc": metrics["pr_auc"],
        "brier": metrics["brier"],
        "ece": metrics["ece"],
    }
    for fdr_row in metrics["fdr"]:
        threshold = fdr_row["threshold"]
        row[f"accepted_psms_at_{threshold}"] = fdr_row["accepted_psms"]
        row[f"realised_fdr_at_{threshold}"] = fdr_row["realised_fdr"]
        row[f"sTECE_at_{threshold}"] = fdr_row["sTECE"]
        row[f"TECE_at_{threshold}"] = fdr_row["TECE"]
    return row


def main() -> None:
    """Run manuscript-style PLM validation."""
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = selected_datasets(args.datasets)
    thresholds = [float(value) for value in args.fdr_thresholds.split(",")]
    required_files = required_dataset_files_for_mode(datasets, args.validation_mode)
    print(
        f"Downloading validation datasets: {', '.join(required_files)}",
        flush=True,
    )
    download_datasets(args.repo_id, data_dir, required_files)

    loaded = {}
    for name in required_files:
        print(f"Loading dataset {name}", flush=True)
        loaded[name] = load_dataset(data_dir, name)
        print(f"Loaded {name}: {len(loaded[name].metadata)} PSMs", flush=True)

    if args.validation_mode == "hela_transfer":
        report = run_hela_transfer(loaded, datasets, thresholds, args)
    else:
        report = run_in_domain_split(loaded, datasets, thresholds, args)

    write_report(report, output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote validation outputs to {output_dir}")


if __name__ == "__main__":
    main()
