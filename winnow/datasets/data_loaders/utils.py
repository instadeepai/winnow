"""Shared spectrum file I/O and peptide normalization for dataset loaders."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Tuple, Union

import numpy as np
import polars as pl
import pandas as pd
from instanovo.utils.metrics import Metrics

if TYPE_CHECKING:
    from matchms import Spectrum

DataFrameT = Union[pd.DataFrame, pl.DataFrame]


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


def add_row_order_spectrum_ids(df: pl.DataFrame, experiment_name: str) -> pl.DataFrame:
    """Add ``experiment_name`` and ``spectrum_id`` as ``{experiment_name}:{row_index}``.

    Uses 0-based file row order for the index suffix, not ``scan_number``.
    Replaces any pre-existing ``experiment_name`` or ``spectrum_id`` columns.
    """
    drop_cols = [c for c in ("experiment_name", "spectrum_id") if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols)
    return (
        df.with_columns(pl.lit(experiment_name).alias("experiment_name").cast(pl.Utf8))
        .with_row_index("_row_index")
        .with_columns(
            (
                pl.col("experiment_name") + ":" + pl.col("_row_index").cast(pl.Utf8)
            ).alias("spectrum_id")
        )
        .drop("_row_index")
    )


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


def _is_missing_cell(value: object) -> bool:
    """Return True when a peptide cell is absent (None, NaN, pd.NA)."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return False


def is_usable_peptide_label(value: object) -> bool:
    """Return True when a raw peptide cell contains a label or sequence string."""
    if _is_missing_cell(value):
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    if isinstance(value, pl.Series):
        return len(value) > 0
    if isinstance(value, np.ndarray):
        return value.size > 0
    if hasattr(value, "tolist"):
        tokens = value.tolist()
        return isinstance(tokens, (list, tuple)) and len(tokens) > 0
    return True


def as_token_list(value: object) -> list[str] | None:
    """Coerce a metadata cell to a non-empty AA token list, or ``None``.

    Accepts container types that already hold token strings. Does not parse raw
    peptide strings; use :func:`normalize_peptide_cell` for that.
    """
    if _is_missing_cell(value):
        return None
    if isinstance(value, pl.Series):
        tokens = value.to_list()
        return list(tokens) if tokens else None
    if isinstance(value, (list, tuple)):
        tokens = list(value)
        return tokens if tokens else None
    if isinstance(value, np.ndarray):
        tokens = value.tolist()
        return tokens if isinstance(tokens, list) and tokens else None
    return None


def is_valid_peptide_tokens(value: object) -> bool:
    """Return True when a cell holds a non-empty token list (not a raw string)."""
    return as_token_list(value) is not None


def _normalize_leucine_tokens(tokens: list[str]) -> list[str]:
    """Map leucine to isoleucine at the token level."""
    return ["I" if token == "L" else token for token in tokens]


def _apply_token_postprocessing(
    tokens: list[str],
    *,
    residue_remapping: dict[str, str],
    normalize_leucine: bool = True,
) -> list[str]:
    if normalize_leucine:
        tokens = _normalize_leucine_tokens(tokens)
    return [residue_remapping.get(token, token) for token in tokens]


def normalize_peptide_cell(
    value: object,
    metrics: Metrics,
    *,
    residue_remapping: dict[str, str],
    normalize_leucine: bool = True,
    require_label: bool = False,
) -> list[str] | None:
    """Normalize one peptide cell to ProForma token list, or ``None`` if absent/empty.

    Args:
        value: Raw or tokenized cell from pandas or polars.
        metrics: InstaNovo metrics for string tokenization.
        residue_remapping: Modification token remapping table.
        normalize_leucine: When True, map ``L`` → ``I`` at token level.
        require_label: When True, return ``None`` for cells that fail
            :func:`is_usable_peptide_label` before tokenization (ground truth).
    """
    if require_label and not is_usable_peptide_label(value):
        return None
    if _is_missing_cell(value):
        return None
    if isinstance(value, str):
        if value.strip() == "":
            return None
        peptide = value.replace("L", "I") if normalize_leucine else value
        tokens = metrics._split_peptide(peptide)
    else:
        tokens = as_token_list(value)
        if tokens is None:
            return None
    tokens = _apply_token_postprocessing(
        tokens,
        residue_remapping=residue_remapping,
        normalize_leucine=normalize_leucine,
    )
    return tokens if tokens else None


def _normalize_peptide_column_pandas(
    series: pd.Series,
    metrics: Metrics,
    *,
    residue_remapping: dict[str, str],
    require_label: bool,
) -> pd.Series:
    return series.apply(
        lambda value: normalize_peptide_cell(
            value,
            metrics,
            residue_remapping=residue_remapping,
            require_label=require_label,
        )
    )


def _normalize_peptide_column_polars(
    series: pl.Series,
    metrics: Metrics,
    *,
    residue_remapping: dict[str, str],
    require_label: bool,
) -> pl.Series:
    return series.map_elements(
        lambda value: normalize_peptide_cell(
            value,
            metrics,
            residue_remapping=residue_remapping,
            require_label=require_label,
        ),
        return_dtype=pl.List(pl.Utf8),
    )


def _row_evaluation_pandas(
    row: pd.Series,
    metrics: Metrics,
    *,
    sequence_col: str,
    prediction_col: str,
) -> tuple[int, bool]:
    sequence = as_token_list(row.get(sequence_col)) or []
    prediction = as_token_list(row.get(prediction_col)) or []
    sequence_valid = is_valid_peptide_tokens(row.get(sequence_col))
    prediction_valid = is_valid_peptide_tokens(row.get(prediction_col))
    num_matches = row_num_matches(
        sequence,
        prediction,
        metrics,
        sequence_valid=sequence_valid,
        prediction_valid=prediction_valid,
    )
    correct = row_is_correct(
        num_matches,
        sequence,
        prediction,
        sequence_valid=sequence_valid,
        prediction_valid=prediction_valid,
    )
    return num_matches, correct


def _row_evaluation_polars(
    row: dict[str, Any],
    metrics: Metrics,
    *,
    sequence_col: str,
    prediction_col: str,
) -> tuple[int, bool]:
    sequence = as_token_list(row.get(sequence_col)) or []
    prediction = as_token_list(row.get(prediction_col)) or []
    sequence_valid = is_valid_peptide_tokens(row.get(sequence_col))
    prediction_valid = is_valid_peptide_tokens(row.get(prediction_col))
    num_matches = row_num_matches(
        sequence,
        prediction,
        metrics,
        sequence_valid=sequence_valid,
        prediction_valid=prediction_valid,
    )
    correct = row_is_correct(
        num_matches,
        sequence,
        prediction,
        sequence_valid=sequence_valid,
        prediction_valid=prediction_valid,
    )
    return num_matches, correct


def _finalize_peptide_metadata_pandas(
    metadata: pd.DataFrame,
    metrics: Metrics,
    *,
    has_labels: bool,
    residue_remapping: dict[str, str],
    sequence_col: str,
    prediction_col: str,
) -> pd.DataFrame:
    metadata[prediction_col] = _normalize_peptide_column_pandas(
        metadata[prediction_col],
        metrics,
        residue_remapping=residue_remapping,
        require_label=False,
    )
    metadata["valid_prediction"] = metadata[prediction_col].apply(
        is_valid_peptide_tokens
    )

    if not has_labels:
        return metadata

    metadata[sequence_col] = _normalize_peptide_column_pandas(
        metadata[sequence_col],
        metrics,
        residue_remapping=residue_remapping,
        require_label=True,
    )
    metadata["valid_sequence"] = metadata[sequence_col].apply(is_valid_peptide_tokens)

    num_matches: list[int] = []
    correct: list[bool] = []
    for _, row in metadata.iterrows():
        nm, ok = _row_evaluation_pandas(
            row,
            metrics,
            sequence_col=sequence_col,
            prediction_col=prediction_col,
        )
        num_matches.append(nm)
        correct.append(ok)
    metadata["num_matches"] = num_matches
    metadata["correct"] = correct
    return metadata


def _finalize_peptide_metadata_polars(
    metadata: pl.DataFrame,
    metrics: Metrics,
    *,
    has_labels: bool,
    residue_remapping: dict[str, str],
    sequence_col: str,
    prediction_col: str,
) -> pl.DataFrame:
    metadata = metadata.with_columns(
        _normalize_peptide_column_polars(
            metadata.get_column(prediction_col),
            metrics,
            residue_remapping=residue_remapping,
            require_label=False,
        ).alias(prediction_col),
    ).with_columns(
        pl.col(prediction_col)
        .map_elements(is_valid_peptide_tokens, return_dtype=pl.Boolean)
        .alias("valid_prediction"),
    )

    if not has_labels:
        return metadata

    metadata = metadata.with_columns(
        _normalize_peptide_column_polars(
            metadata.get_column(sequence_col),
            metrics,
            residue_remapping=residue_remapping,
            require_label=True,
        ).alias(sequence_col),
    ).with_columns(
        pl.col(sequence_col)
        .map_elements(is_valid_peptide_tokens, return_dtype=pl.Boolean)
        .fill_null(False)
        .alias("valid_sequence"),
    )

    metadata = metadata.with_columns(
        pl.struct([sequence_col, prediction_col])
        .map_elements(
            lambda row: _row_evaluation_polars(
                row,
                metrics,
                sequence_col=sequence_col,
                prediction_col=prediction_col,
            )[0],
            return_dtype=pl.Int64,
        )
        .alias("num_matches"),
    ).with_columns(
        pl.struct([sequence_col, prediction_col])
        .map_elements(
            lambda row: _row_evaluation_polars(
                row,
                metrics,
                sequence_col=sequence_col,
                prediction_col=prediction_col,
            )[1],
            return_dtype=pl.Boolean,
        )
        .alias("correct"),
    )
    return metadata


def finalize_peptide_metadata(
    metadata: DataFrameT,
    metrics: Metrics,
    *,
    has_labels: bool,
    residue_remapping: dict[str, str] | None = None,
    sequence_col: str = "sequence",
    prediction_col: str = "prediction",
) -> DataFrameT:
    """Normalize peptide columns, set validity flags, and compute match columns.

    Accepts pandas or polars frames and returns the same frame type. All logic
    delegates to cell-level helpers so behaviour is identical regardless of
    whether cells arrive as ``list``, ``np.ndarray``, ``pl.Series``, or ``str``.
    """
    if residue_remapping is None:
        residue_remapping = metrics.residue_set.residue_remapping

    if isinstance(metadata, pl.DataFrame):
        return _finalize_peptide_metadata_polars(
            metadata,
            metrics,
            has_labels=has_labels,
            residue_remapping=residue_remapping,
            sequence_col=sequence_col,
            prediction_col=prediction_col,
        )
    return _finalize_peptide_metadata_pandas(
        metadata,
        metrics,
        has_labels=has_labels,
        residue_remapping=residue_remapping,
        sequence_col=sequence_col,
        prediction_col=prediction_col,
    )


def _ensure_valid_sequence_column(metadata: pd.DataFrame) -> None:
    """Set ``valid_sequence`` from ``sequence`` when the column is absent."""
    if "sequence" in metadata.columns and "valid_sequence" not in metadata.columns:
        metadata["valid_sequence"] = metadata["sequence"].apply(is_valid_peptide_tokens)


def labelled_training_mask(metadata: pd.DataFrame) -> np.ndarray:
    """Return a boolean mask of rows eligible for supervised training or FDR fit.

    When a ``sequence`` column is present, ``valid_sequence`` is derived from it
    if not already supplied. When neither column exists (e.g. precomputed labels
    only), all rows are included.
    """
    if "sequence" in metadata.columns:
        _ensure_valid_sequence_column(metadata)
        return metadata["valid_sequence"].fillna(False).to_numpy(dtype=bool)
    return np.ones(len(metadata), dtype=bool)


def require_labelled_rows(metadata: pd.DataFrame, *, context: str) -> np.ndarray:
    """Return ``labelled_training_mask`` or raise when no rows are eligible."""
    mask = labelled_training_mask(metadata)
    if not mask.any():
        raise ValueError(
            f"{context} requires at least one row with valid_sequence=True."
        )
    return mask


def row_num_matches(
    sequence: list[str],
    prediction: list[str],
    metrics: Metrics,
    *,
    sequence_valid: bool,
    prediction_valid: bool,
) -> int:
    """Count residue matches between tokenized sequence and prediction."""
    if not sequence_valid or not prediction_valid or not sequence or not prediction:
        return 0
    return metrics._novor_match(sequence, prediction)


def row_is_correct(
    num_matches: int,
    sequence: list[str],
    prediction: list[str],
    *,
    sequence_valid: bool,
    prediction_valid: bool,
) -> bool:
    """Return True when prediction is a full-length exact match to sequence."""
    if not sequence_valid or not prediction_valid:
        return False
    return num_matches == len(sequence) == len(prediction)


def has_ground_truth_sequence_labels(df: pl.DataFrame) -> bool:
    """Return True when ``sequence`` contains at least one non-empty label."""
    if "sequence" not in df.columns:
        return False
    return any(is_usable_peptide_label(value) for value in df["sequence"])


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

    has_labels = has_ground_truth_sequence_labels(df)
    if "sequence" in df.columns and not has_labels:
        df = df.drop("sequence")

    return df, has_labels
