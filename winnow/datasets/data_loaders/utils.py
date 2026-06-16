"""Shared spectrum file I/O for dataset loaders (Parquet, IPC, MGF)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Tuple

import polars as pl

if TYPE_CHECKING:
    from matchms import Spectrum


def df_from_matchms(spectra: list[Spectrum]) -> pl.DataFrame:
    """Convert a list of Matchms spectra to a polars DataFrame.

    Includes only metadata columns that matchms exposes for at least one spectrum.
    ``scan_number`` is always a 0-based enumerate index.

    Args:
        spectra: List of Matchms spectrum objects.

    Returns:
        The polars DataFrame.
    """
    metadata_map = {
        "precursor_mz": "precursor_mz",
        "charge": "precursor_charge",
        "retention_time": "retention_time",
    }
    sequence_keys = ("seq", "peptide_sequence")

    all_metadata_keys: set[str] = set()
    for spectrum in spectra:
        all_metadata_keys.update(spectrum.metadata.keys())

    active_columns = {
        mgf_key: col_name
        for mgf_key, col_name in metadata_map.items()
        if mgf_key in all_metadata_keys
    }

    sequence_key = next((k for k in sequence_keys if k in all_metadata_keys), None)

    data: dict[str, list[Any]] = {"scan_number": []}
    for col_name in active_columns.values():
        data[col_name] = []
    if sequence_key:
        data["sequence"] = []
    data["mz_array"] = []
    data["intensity_array"] = []

    for i, spectrum in enumerate(spectra):
        data["scan_number"].append(i)
        for mgf_key, col_name in active_columns.items():
            data[col_name].append(spectrum.metadata.get(mgf_key))
        if sequence_key:
            data["sequence"].append(spectrum.metadata.get(sequence_key))
        data["mz_array"].append(spectrum.peaks.mz)
        data["intensity_array"].append(spectrum.peaks.intensities)

    return pl.DataFrame(data)


def add_index_cols(df: pl.DataFrame, fp: Path | str) -> pl.DataFrame:
    """Add ``experiment_name`` and ``spectrum_id`` columns.

    If ``scan_number`` is present, ``spectrum_id`` is ``experiment_name:scan_number``.
    Otherwise uses a row index, matching InstaNovo's data_handler fallback.
    """
    exp_name = Path(fp).stem
    df = df.with_columns(pl.lit(exp_name).alias("experiment_name").cast(pl.Utf8))
    if "scan_number" in df.columns:
        df = df.with_columns(
            (
                pl.col("experiment_name") + ":" + pl.col("scan_number").cast(pl.Utf8)
            ).alias("spectrum_id")
        )
    else:
        df = df.with_row_index("idx")
        df = df.with_columns(
            (pl.col("experiment_name") + ":" + pl.col("idx").cast(pl.Utf8)).alias(
                "spectrum_id"
            )
        )
        df = df.drop("idx")
    return df


_add_index_cols_fn = add_index_cols


def load_spectrum_data(
    spectrum_path: Path | str, *, add_index_cols: bool = False
) -> Tuple[pl.DataFrame, bool]:
    """Load spectrum data from a Parquet, IPC, or MGF file.

    Args:
        spectrum_path: Path to spectrum data file (.parquet, .ipc, or .mgf).
        add_index_cols: If True, add ``experiment_name`` and ``spectrum_id`` to
            parquet/ipc inputs. MGF inputs always get these columns regardless.

    Returns:
        Tuple of (DataFrame containing spectrum data, whether ground truth labels exist).
    """
    spectrum_path = Path(spectrum_path)

    if spectrum_path.suffix == ".parquet":
        df = pl.read_parquet(spectrum_path)
    elif spectrum_path.suffix == ".ipc":
        df = pl.read_ipc(spectrum_path)
    elif spectrum_path.suffix == ".mgf":
        from matchms.importing import load_from_mgf

        spectra = list(load_from_mgf(str(spectrum_path)))
        df = df_from_matchms(spectra)
    else:
        raise ValueError(
            f"Unsupported file format for spectrum data: {spectrum_path.suffix}. "
            "Supported formats are .parquet, .ipc and .mgf."
        )

    if spectrum_path.suffix == ".mgf" or add_index_cols:
        df = _add_index_cols_fn(df, spectrum_path)

    return df, "sequence" in df.columns
