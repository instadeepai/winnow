"""Shared helpers for calibrator generalisation analysis."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

HEPG2_SOURCE = "PXD019483"

SPECIES_NAME_MAPPING: dict[str, str] = {
    "gluc": "HeLa degradome",
    "helaqc": "HeLa single shot",
    "herceptin": "Herceptin",
    "immuno": "Immunopeptidomics-1",
    "celegans": "$\\it{C.\\;elegans}$",
    "sbrodae": "$\\it{Scalindua\\;brodae}$",
    HEPG2_SOURCE: "HepG2",
    "snakevenoms": "Snake venomics",
    "tplantibodies": "Therapeutic nanobodies",
    "woundfluids": "Wound exudates",
    "PXD014877": "$\\it{C.\\;elegans}$",
}


def extract_project_name(parquet_path: Path) -> str:
    """Extract project name from ``dataset-helaqc-annotated-0000-0001.parquet``."""
    match = re.match(r"dataset-(.+?)-annotated", parquet_path.stem)
    if match:
        return match.group(1)
    return parquet_path.stem


def build_experiment_source_mapping(biological_validation_dir: Path) -> dict[str, str]:
    """Map every experiment in biological validation parquets to its source label."""
    mapping: dict[str, str] = {}
    parquet_files = sorted(biological_validation_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in biological validation directory: "
            f"{biological_validation_dir}"
        )

    for parquet_path in parquet_files:
        project = extract_project_name(parquet_path)
        experiments = (
            pl.scan_parquet(parquet_path)
            .select("experiment_name")
            .unique()
            .collect()["experiment_name"]
            .to_list()
        )
        for experiment_name in experiments:
            mapping[experiment_name] = project

    logger.info(
        "Built experiment->source mapping for %d experiments across %d projects",
        len(mapping),
        len(parquet_files),
    )
    return mapping


def annotate_train_source_labels(
    train_parquet: Path,
    train_predictions: Path,
    biological_validation_dir: Path,
) -> None:
    """Add a ``source`` column to the train parquet and predictions CSV.

    Experiments found in ``biological_validation_dir`` inherit that project name.
    All other experiments are labelled as HepG2 (``PXD019483``).
    """
    experiment_to_source = build_experiment_source_mapping(biological_validation_dir)
    lookup = pl.DataFrame(
        {
            "experiment_name": list(experiment_to_source.keys()),
            "source": list(experiment_to_source.values()),
        }
    )

    spectra = pl.read_parquet(train_parquet)
    if "source" not in spectra.columns:
        spectra = spectra.join(lookup, on="experiment_name", how="left").with_columns(
            pl.col("source").fill_null(HEPG2_SOURCE)
        )
        spectra.write_parquet(train_parquet)
        logger.info("Wrote source labels to %s", train_parquet)
    else:
        logger.info(
            "Parquet already has source column, leaving %s unchanged", train_parquet
        )

    predictions = pl.read_csv(train_predictions)
    if "source" not in predictions.columns:
        source_by_spectrum = spectra.select("spectrum_id", "source")
        predictions = predictions.join(source_by_spectrum, on="spectrum_id", how="left")
        missing = predictions.filter(pl.col("source").is_null())
        if len(missing) > 0:
            raise ValueError(
                f"{len(missing)} prediction rows in {train_predictions} have no matching "
                "spectrum_id in the train parquet"
            )
        predictions.write_csv(train_predictions)
        logger.info("Wrote source labels to %s", train_predictions)
    else:
        logger.info(
            "Predictions CSV already has source column, leaving %s unchanged",
            train_predictions,
        )
