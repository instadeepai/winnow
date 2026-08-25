"""Shared helpers for calibrator generalisation analysis."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

SPECIES_NAME_MAPPING: dict[str, str] = {
    "gluc": "HeLa degradome",
    "helaqc": "HeLa single shot",
    "herceptin": "Herceptin",
    "immuno": "Immunopeptidomics-1",
    "celegans": "$\\it{C.\\;elegans}$",
    "sbrodae": "$\\it{Scalindua\\;brodae}$",
    "PXD019483": "HepG2",
    "hepg2": "HepG2",
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
