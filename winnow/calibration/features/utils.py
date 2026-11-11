"""Utility functions for input validation and spectrum match quality feature computation.

This module provides functions for evaluating how well theoretical peptide
fragmentation spectra match observed experimental spectra. These features
are used by the calibrator to distinguish high-quality PSMs from low-quality ones.
"""

from math import isnan
from typing import (
    Dict,
    List,
    NamedTuple,
    Optional,
    Any,
    Sequence,
    Set,
    Tuple,
    Union,
    Literal,
)
import numpy as np
import pandas as pd
import polars as pl

from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.calibration.features import _kernels
from winnow.calibration.features.constants import (
    CARBON_ISOTOPE_MASS_SHIFT,
    VALID_INTENSITY_MODEL_PROVIDERS,
    XCORR_BIN_SIZE,
    XCORR_BIN_OFFSET,
    XCORR_NUM_WINDOWS,
    XCORR_MAX_OFFSET,
    XCORR_WINDOW_NORM_VALUE,
)

########################################################
# Helper functions
########################################################


class IonMatchResult(NamedTuple):
    """Result of matching theoretical fragment ions to an observed spectrum."""

    match_rate: float
    match_intensity: float
    matched_ion_annotations: List[str]
    matched_ion_mz: List[float]
    matched_ion_intensities_incl_isotopic_env: List[float]
    aligned_m0_intensities: List[float]

    def __iter__(self):
        raise TypeError(
            "Positional unpacking of IonMatchResult is no longer compatible with "
            "this version of winnow. Upgrade to the latest API by accessing result "
            "fields by name, e.g. result.match_rate, result.match_intensity, "
            "result.matched_ion_annotations, result.matched_ion_mz, and "
            "result.matched_ion_intensities_incl_isotopic_env, and result.aligned_m0_intensities."
        )


class IonIdentificationResult(NamedTuple):
    """Per-spectrum ion identification feature columns."""

    ion_match_rate: Tuple[float, ...]
    ion_match_intensity: Tuple[float, ...]
    longest_b_series: Tuple[int, ...]
    longest_y_series: Tuple[int, ...]
    complementary_ion_count: Tuple[int, ...]
    max_ion_gap: Tuple[float, ...]
    b_y_intensity_ratio: Tuple[float, ...]
    spectral_angle: Tuple[float, ...]
    xcorr: Tuple[float, ...]

    def __iter__(self):
        raise TypeError(
            "Positional unpacking of IonIdentificationResult is no longer "
            "compatible with this version of winnow. Upgrade to the latest API "
            "by accessing result fields by name, e.g. result.ion_match_rate, "
            "result.ion_match_intensity, result.longest_b_series, "
            "result.longest_y_series, result.complementary_ion_count, "
            "result.max_ion_gap, result.b_y_intensity_ratio, "
            "result.spectral_angle, and result.xcorr."
        )


def validate_intensity_model_name(intensity_model_name: str) -> None:
    """Validate the intensity model name.

    Supported model providers are Prosit, MS2PIP and AlphaPeptDeep.

    Args:
        intensity_model_name: Name of the intensity model.
    """
    name_lower = intensity_model_name.lower()
    if not any(provider in name_lower for provider in VALID_INTENSITY_MODEL_PROVIDERS):
        raise ValueError(
            f"Invalid intensity model name: {intensity_model_name}. Supported model providers are {VALID_INTENSITY_MODEL_PROVIDERS}."
        )


def require_beam_predictions(dataset: CalibrationDataset, feature_name: str) -> None:
    """Raise a ValueError if the dataset has no beam predictions.

    Args:
        dataset: The calibration dataset to check.
        feature_name: Name of the feature requiring beams (for error message).

    Raises:
        ValueError: If dataset.predictions is None.
    """
    if dataset.predictions is None:
        raise ValueError(
            f"{feature_name} requires beam predictions, but dataset.predictions is None. "
            "This dataset was loaded without beam predictions. "
            "To use this feature, ensure your data loader is configured to load beams: "
            "for InstaNovo, set beam_columns in instanovo.yaml; "
            "for MZTab, set load_beams: true in mztab.yaml."
        )


def validate_model_input_params(
    model_input_constants: Optional[Dict[str, Any]],
    model_input_columns: Optional[Dict[str, str]],
) -> None:
    """Raise ValueError if the same key appears in both model_input_constants and model_input_columns.

    Args:
        model_input_constants: Mapping of Koina input name to a constant value tiled across all rows.
        model_input_columns: Mapping of Koina input name to a metadata column name providing per-row values.

    Raises:
        ValueError: If any key is present in both dicts, since this is an unresolvable conflict.
    """
    if model_input_constants is None or model_input_columns is None:
        return
    conflicts = set(model_input_constants) & set(model_input_columns)
    if conflicts:
        raise ValueError(
            f"The following Koina model input(s) are specified in both model_input_constants "
            f"and model_input_columns, which is ambiguous: {sorted(conflicts)}. "
            f"Specify each input in exactly one of model_input_constants or model_input_columns."
        )


def resolve_model_inputs(
    inputs: pd.DataFrame,
    metadata: pd.DataFrame,
    required_model_inputs: List[str],
    auto_populated: Set[str],
    constants: Optional[Dict[str, Any]],
    columns: Optional[Dict[str, str]],
    model_name: str,
) -> pd.DataFrame:
    """Populate additional Koina model inputs beyond those that are auto-populated.

    Determines which model inputs remain after accounting for auto-populated columns,
    validates that all of them are covered by either a constant value or a metadata column
    mapping, and fills in the inputs DataFrame accordingly.

    Args:
        inputs: DataFrame already containing auto-populated columns (e.g. peptide_sequences,
            precursor_charges). This DataFrame is modified in-place and returned.
        metadata: The full metadata DataFrame for the valid subset being predicted, used to
            resolve per-row column values.
        required_model_inputs: Full list of input names required by the Koina model
            (i.e. model.model_inputs).
        auto_populated: Set of input names already present in inputs and therefore excluded
            from the remaining check (e.g. {"peptide_sequences", "precursor_charges"}).
        constants: Mapping of Koina input name to a scalar value that will be tiled across
            all rows. May be None or empty.
        columns: Mapping of Koina input name to a metadata column name that provides
            per-row values. May be None or empty.
        model_name: Name of the Koina model, used in error messages.

    Returns:
        The inputs DataFrame with all required model inputs populated.

    Raises:
        ValueError: If any required model input (beyond auto-populated ones) is not covered
            by constants or columns.
        KeyError: If a column specified in columns does not exist in metadata.
    """
    constants = constants or {}
    columns = columns or {}

    remaining = [col for col in required_model_inputs if col not in auto_populated]
    missing = [col for col in remaining if col not in constants and col not in columns]
    if missing:
        raise ValueError(
            f"Koina model '{model_name}' requires the following input(s) that have not been "
            f"provided: {missing}. Supply them via:\n"
            f"  model_input_constants: {{{', '.join(f'{m}: <value>' for m in missing)}}}\n"
            f"  or model_input_columns: {{{', '.join(f'{m}: <metadata_column>' for m in missing)}}}"
        )

    n_rows = len(inputs)
    for col in remaining:
        if col in constants:
            inputs[col] = np.array([constants[col]] * n_rows)
        else:
            inputs[col] = metadata[columns[col]].to_numpy()

    return inputs


def format_intensity_prediction_outputs(predictions: pd.DataFrame) -> pd.DataFrame:
    """Format intensity prediction outputs to one row per peptide, ordered by m/z to match experimental data.

    Args:
        predictions: DataFrame containing intensity prediction outputs.

    Returns:
        DataFrame with one row per peptide, containing the peptide sequences, precursor charges, collision energies, intensities, m/z values and annotations ordered by m/z.
    """
    # Group predictions by spectrum_id to get one row per peptide
    # We make a temporary column spectrum_id_col to enable grouping by spectrum_id,
    # and we name this spectrum_id_col to avoid naming conflicts with the index
    predictions["spectrum_id_col"] = predictions.index
    grouped_predictions = predictions.groupby(
        by="spectrum_id_col"
    ).agg(
        {
            "peptide_sequences": "first",  # this is the same for all rows in the group, so we can use the first row
            "precursor_charges": "first",
            "collision_energies": "first",
            "intensities": list,  # form a list of intensities for each peptide
            "mz": list,  # form a list of m/z values for each peptide
            "annotation": list,  # form a list of annotations for each peptide
        }
    )

    # Sort intensities by m/z to match experimental data
    grouped_predictions["intensities"] = grouped_predictions.apply(
        lambda row: np.array(row["intensities"])[np.argsort(row["mz"])].tolist(),
        axis=1,
    )
    # Sort annotations by m/z to match experimental data
    grouped_predictions["annotation"] = grouped_predictions.apply(
        lambda row: np.array(row["annotation"])[np.argsort(row["mz"])].tolist(),
        axis=1,
    )
    # Sort m/z values to match experimental data
    grouped_predictions["mz"] = grouped_predictions["mz"].apply(np.sort)

    return grouped_predictions


########################################################
# Annotation parsing and dtype-matched constants
########################################################


class _ParsedIons(NamedTuple):
    """Integer encodings of fragment ion annotations, one entry per ion."""

    labels: list[str]
    kind: np.ndarray
    series: np.ndarray


def _parse_ions(annotations: Sequence[bytes | str]) -> _ParsedIons:
    """Decode annotations and encode ion type (b, y, other) and series index (-1 if not numeric).

    The ion type comes from the first character; the series index is the text
    between it and the first ``+`` (e.g. ``"b12+2"`` -> 12), when that is all digits.
    """
    labels = [a.decode() if isinstance(a, bytes) else str(a) for a in annotations]
    text = pl.Series("annotation", labels, dtype=pl.String)
    base = text.str.split("+").list.first()
    digits = base.str.slice(1)
    kind = (
        pl.when(text.str.starts_with("b"))
        .then(_kernels.ION_B)
        .when(text.str.starts_with("y"))
        .then(_kernels.ION_Y)
        .otherwise(_kernels.ION_OTHER)
    )
    parsed = pl.DataFrame({"text": text, "digits": digits}).select(
        kind.cast(pl.Int8).alias("kind"),
        pl.when(pl.col("digits").str.contains(r"^[0-9]+$"))
        .then(pl.col("digits").cast(pl.Int64))
        .otherwise(-1)
        .alias("series"),
    )
    return _ParsedIons(labels, parsed["kind"].to_numpy(), parsed["series"].to_numpy())


def _parse_charges(labels: list[str]) -> np.ndarray:
    """Return the fragment charge of each annotation: the integer after the first ``+``."""
    text = pl.Series("annotation", labels, dtype=pl.String)
    return text.str.split("+").list.get(1, null_on_oob=False).cast(pl.Int64).to_numpy()


def _cast(values: Sequence[float], like: np.ndarray) -> np.ndarray:
    """Cast Python float constants to the dtype of ``like``, as NumPy does before mixed arithmetic."""
    dtype = like.dtype if np.issubdtype(like.dtype, np.floating) else np.float64
    return np.array(values, dtype=dtype)


def _isotope_shift_table(max_charge: int, mz: np.ndarray) -> np.ndarray:
    """Return isotope offsets ``i * 1.00335 / charge`` cast to the m/z dtype, indexed [charge, i]."""
    table = np.zeros((max_charge + 1, _kernels.N_ISOTOPES + 1), dtype=np.float64)
    for charge in range(1, max_charge + 1):
        for isotope in range(1, _kernels.N_ISOTOPES + 1):
            table[charge, isotope] = isotope * (CARBON_ISOTOPE_MASS_SHIFT / charge)
    return table.astype(_cast([], mz).dtype)


def _as_float_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return values as a float array, keeping float32 and widening anything else to float64."""
    array = np.asarray(values)
    return (
        array if np.issubdtype(array.dtype, np.floating) else array.astype(np.float64)
    )


class _MatchConstants(NamedTuple):
    """Tolerance and isotope constants cast to the theoretical m/z dtype."""

    ppm: bool
    tolerance_cast: np.ndarray
    million_cast: np.ndarray
    tolerance_da: float
    isotope_shift: np.ndarray


def _match_constants(
    mz_tolerance: float,
    unit: Literal["ppm", "da"],
    theo_mz: np.ndarray,
    charges: np.ndarray,
) -> _MatchConstants:
    """Build the constants ``_kernels.match_ions`` needs for one batch of theoretical ions."""
    max_charge = int(charges.max()) if charges.size else 1
    return _MatchConstants(
        unit == "ppm",
        _cast([mz_tolerance], theo_mz),
        _cast([1e6], theo_mz),
        float(mz_tolerance),
        _isotope_shift_table(max_charge, theo_mz),
    )


########################################################
# Peak Matching
########################################################


def parse_mz_tolerance_unit(unit: str) -> Literal["ppm", "da"]:
    """Normalize and validate an m/z tolerance unit string.

    Args:
        unit: Tolerance unit; must be ``"ppm"`` or ``"da"`` (case-insensitive).

    Returns:
        The normalized unit, either ``"ppm"`` or ``"da"``.

    Raises:
        ValueError: If ``unit`` is not a string or is not a recognised tolerance unit.
    """
    if not isinstance(unit, str):
        raise ValueError(
            f"mz_tolerance_unit must be a string, got {type(unit).__name__}. "
            "Use 'ppm' or 'da'."
        )
    normalized = unit.strip().lower()
    if normalized in {"ppm", "da"}:
        return normalized  # type: ignore[return-value]
    raise ValueError(
        f"Invalid mz_tolerance_unit {unit!r}. Must be 'ppm' or 'da' (case-insensitive)."
    )


def _validate_mz_tolerance_value(mz_tolerance: float) -> float:
    """Raise ValueError unless ``mz_tolerance`` is a numeric value."""
    if isinstance(mz_tolerance, bool) or not isinstance(mz_tolerance, (int, float)):
        raise ValueError(
            f"mz_tolerance must be a number, got {type(mz_tolerance).__name__}."
        )
    return float(mz_tolerance)


def _validate_mz_tolerance(
    mz_tolerance: float,
    mz_tolerance_unit: str,
) -> Literal["ppm", "da"]:
    """Validate m/z tolerance configuration.

    Args:
        mz_tolerance: Tolerance magnitude.
        mz_tolerance_unit: Unit for ``mz_tolerance``; ``"ppm"`` or ``"da"``.

    Returns:
        The normalized tolerance unit.

    Raises:
        ValueError: If ``mz_tolerance`` is not numeric or ``mz_tolerance_unit`` is invalid.
    """
    _validate_mz_tolerance_value(mz_tolerance)
    return parse_mz_tolerance_unit(mz_tolerance_unit)


def find_matching_ions(
    source_mz: List[float],
    target_mz: List[float],
    target_intensities: List[float],
    source_annotations: Union[List[bytes], List[str]],
    *,
    mz_tolerance: float,
    mz_tolerance_unit: str,
) -> IonMatchResult:
    """Finds the matching ions between source and target spectra based on m/z.

    Computes:
      1. The number of matched ions over the total number of source (theoretical) ions.
         Only monoisotopic peaks (M0) are counted to avoid inflating matches from noise.
      2. The sum of observed intensities from matched ions, including their isotopic envelope
         (M0 through M+4), over the sum of all observed intensities.
         Isotopic peaks are searched at spacing of 1.00335/charge Da.
      3. The list of matched theoretical ion annotations.
      4. The list of matched theoretical ion m/z values.
      5. The list of matched ion intensities (M0 + isotopic envelope per ion).
      6. The list of aligned M0 intensities (same length as source_mz, 0.0 for unmatched).

    Each observed peak can only be matched once. Once an observed peak is assigned to a
    theoretical ion (either as M0 or as part of its isotopic envelope), it is excluded
    from matching subsequent theoretical ions. ``target_mz`` must be sorted ascending.

    Args:
        source_mz: List of m/z values from the source (theoretical) spectrum.
        target_mz: List of m/z values from the target (observed) spectrum.
        target_intensities: List of intensities corresponding to target m/z values.
        source_annotations: List of ion annotations from Koina (e.g., "b1+3", "y2+2").
        mz_tolerance: Tolerance magnitude.
        mz_tolerance_unit: Unit for ``mz_tolerance``; ``"ppm"`` or ``"da"`` (case-insensitive).

    Returns:
        :class:`IonMatchResult` with match statistics and per-ion match details.
        Access fields by name; positional unpacking is no longer supported.

    Raises:
        ValueError: If ``mz_tolerance`` is not numeric or ``mz_tolerance_unit`` is invalid.
    """
    normalized_unit = _validate_mz_tolerance(mz_tolerance, mz_tolerance_unit)

    if isinstance(source_mz, float) and isnan(source_mz):
        return IonMatchResult(0.0, 0.0, [], [], [], [])

    theo_mz = _as_float_array(source_mz)
    obs_mz = _as_float_array(target_mz)
    obs_int = _as_float_array(target_intensities)
    ions = _parse_ions(source_annotations)
    charges = _parse_charges(ions.labels)
    constants = _match_constants(mz_tolerance, normalized_unit, theo_mz, charges)

    matched = np.zeros(theo_mz.size, dtype=np.bool_)
    m0_int = np.zeros(theo_mz.size, dtype=obs_int.dtype)
    ion_int = np.zeros(theo_mz.size, dtype=obs_int.dtype)
    num_matches, matched_total, total = _kernels.match_ions(
        theo_mz, charges, obs_mz, obs_int, *constants, matched, m0_int, ion_int
    )
    intensity_type = obs_int.dtype.type
    return IonMatchResult(
        num_matches / len(theo_mz),
        intensity_type(matched_total) / intensity_type(total),
        [label for label, hit in zip(ions.labels, matched) if hit],
        [mz for mz, hit in zip(theo_mz, matched) if hit],
        [intensity for intensity, hit in zip(ion_int, matched) if hit],
        [intensity if hit else 0.0 for intensity, hit in zip(m0_int, matched)],
    )


def _flatten(
    cells: list[Sequence[float] | np.ndarray | float], missing: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (offsets, values) for list-like cells, with missing rows empty."""
    present = [
        np.asarray(cell) for cell, is_missing in zip(cells, missing) if not is_missing
    ]
    lengths = np.zeros(len(cells), dtype=np.int64)
    lengths[~missing] = [array.size for array in present]
    offsets = np.zeros(len(cells) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    values = np.concatenate(present) if present else np.empty(0)
    return offsets, _as_float_array(values)


# Rows per kernel call. Each call holds flat copies of its rows' spectra, so
# batching bounds that extra memory without changing any per-spectrum result.
_KERNEL_BATCH_ROWS = 16_384


class _KernelBatch(NamedTuple):
    """Kernel outputs for a batch of rows plus the scalar types of its inputs."""

    floats: np.ndarray
    ints: np.ndarray
    flags: np.ndarray
    intensity_type: type[np.floating]
    mz_type: type[np.floating]


def _ion_identification_batch(
    dataset: pd.DataFrame,
    source_mz_column: str,
    source_annotation_column: str,
    source_intensity_column: str,
    mz_tolerance: float,
    normalized_unit: Literal["ppm", "da"],
    peptides: Sequence[Sequence[str]],
) -> _KernelBatch:
    """Flatten one batch of rows and run the ion identification kernel on it."""
    n_rows = len(dataset)
    theo_mz_cells = dataset[source_mz_column].tolist()
    missing = np.fromiter(
        (isinstance(cell, float) for cell in theo_mz_cells),
        dtype=np.bool_,
        count=n_rows,
    )
    theo_offsets, theo_mz = _flatten(theo_mz_cells, missing)
    int_offsets, theo_int = _flatten(dataset[source_intensity_column].tolist(), missing)
    annotation_cells = dataset[source_annotation_column].tolist()
    annotation_lengths = [
        0 if is_missing else len(cell)
        for cell, is_missing in zip(annotation_cells, missing)
    ]
    if not np.array_equal(int_offsets, theo_offsets):
        raise ValueError(
            "The intensity lists must align by m/z values.\n"
            "This means every theoretical ion must have an observed intensity "
            "or a zero intensity in the aligned observed intensity list."
        )
    if not np.array_equal(np.diff(theo_offsets), annotation_lengths):
        raise ValueError("Each theoretical m/z value must have exactly one annotation.")
    if np.any(~missing & (np.diff(theo_offsets) == 0)):
        raise ZeroDivisionError("division by zero: a spectrum has no theoretical ions")

    obs_offsets, obs_mz = _flatten(
        dataset["mz_array"].tolist(), np.zeros(n_rows, np.bool_)
    )
    _, obs_int = _flatten(
        dataset["intensity_array"].tolist(), np.zeros(n_rows, np.bool_)
    )
    ions = _parse_ions(
        [a for cell, m in zip(annotation_cells, missing) if not m for a in cell]
    )
    charges = _parse_charges(ions.labels)
    peptide_length = np.fromiter(
        (len(p) for p in peptides), dtype=np.int64, count=n_rows
    )
    theo_float64 = theo_int.astype(np.float64, copy=False)

    floats, ints, flags = _kernels.ion_identifications(
        obs_offsets, obs_mz, obs_int, theo_offsets, theo_mz, theo_float64,
        ions.kind, ions.series, charges, missing, peptide_length,
        *_match_constants(mz_tolerance, normalized_unit, theo_mz, charges),
        _cast([1e-8], obs_int),
        _cast([XCORR_BIN_SIZE, XCORR_BIN_OFFSET], obs_mz),
        _cast([XCORR_BIN_SIZE, XCORR_BIN_OFFSET], theo_mz),
        XCORR_NUM_WINDOWS, XCORR_MAX_OFFSET, XCORR_WINDOW_NORM_VALUE,
    )  # fmt: skip

    return _KernelBatch(floats, ints, flags, obs_int.dtype.type, theo_mz.dtype.type)


def compute_ion_identifications(
    dataset: pd.DataFrame,
    source_mz_column: str,
    source_annotation_column: str,
    source_intensity_column: str,
    *,
    mz_tolerance: float,
    mz_tolerance_unit: str,
    predictions: Optional[List[str]] = None,
) -> IonIdentificationResult:
    """Computes the ion match rate, match intensity, longest b series, longest y series, complementary ion count, max ion gap, b/y intensity ratio, normalised spectral angle and fast cross-correlation score for each spectrum in the dataset.

    Spectra are processed in parallel by a compiled kernel; set ``NUMBA_NUM_THREADS``
    to limit the thread count. Rows whose theoretical m/z cell is a float (NaN for
    a missing theoretical spectrum) get zeros.

    Args:
        dataset: DataFrame containing the mass spectrum data.
        source_mz_column: Column name containing the theoretical m/z values.
        source_annotation_column: Column name containing the ion annotations.
        source_intensity_column: Column name containing the theoretical intensities.
        mz_tolerance: Tolerance magnitude.
        mz_tolerance_unit: Unit for ``mz_tolerance``; ``"ppm"`` or ``"da"`` (case-insensitive).
        predictions: Optional list of tokenised predictions for each spectrum. If not provided, the peptide length will be inferred from the column "predictions" in the metadata.

    Returns:
        :class:`IonIdentificationResult` with one tuple per feature column.
        Access fields by name; positional unpacking is no longer supported.

    Raises:
        ValueError: If ``mz_tolerance`` is not numeric or ``mz_tolerance_unit`` is invalid,
            or if a row's theoretical m/z, intensity and annotation lists differ in length.
        ZeroDivisionError: If a row that is not missing has no theoretical ions.
    """
    normalized_unit = _validate_mz_tolerance(mz_tolerance, mz_tolerance_unit)

    n_rows = len(dataset)
    if n_rows == 0:
        return IonIdentificationResult((), (), (), (), (), (), (), (), ())

    peptides = (
        predictions if predictions is not None else dataset["prediction"].tolist()
    )
    batches = [
        _ion_identification_batch(
            dataset.iloc[start : start + _KERNEL_BATCH_ROWS],
            source_mz_column,
            source_annotation_column,
            source_intensity_column,
            mz_tolerance,
            normalized_unit,
            peptides[start : start + _KERNEL_BATCH_ROWS],
        )
        for start in range(0, n_rows, _KERNEL_BATCH_ROWS)
    ]

    # Values computed in the intensity or m/z dtype are returned as scalars of that
    # dtype and the rest as Python floats, so callers that build columns from these
    # tuples infer the same column dtype as before.
    def _floats(column: int) -> tuple[float, ...]:
        return tuple(v for b in batches for v in b.floats[:, column].tolist())

    def _ints(column: int) -> tuple[int, ...]:
        return tuple(v for b in batches for v in b.ints[:, column].tolist())

    def _typed(
        column: int, flag: int, scalar_of: str
    ) -> tuple[float | np.floating, ...]:
        return tuple(
            getattr(b, scalar_of)(value) if is_typed else 0.0
            for b in batches
            for value, is_typed in zip(
                b.floats[:, column].tolist(), b.flags[:, flag].tolist()
            )
        )

    return IonIdentificationResult(
        _floats(0),
        _typed(1, 0, "intensity_type"),
        _ints(0),
        _ints(1),
        _ints(2),
        _typed(2, 1, "mz_type"),
        _typed(3, 2, "intensity_type"),
        _floats(4),
        _floats(5),
    )


########################################################
# Ion Coverage Features
########################################################


def _select_all(n: int) -> np.ndarray:
    """Return a mask selecting every ion."""
    return np.ones(n, dtype=np.bool_)


def compute_longest_ion_series(
    matched_ion_annotations: List[str], ion_type: str
) -> int:
    """Find the longest consecutive run of matched fragment ions.

    Counts the maximum number of consecutive fragment ions of the specified type
    (e.g., b1, b2, b3 = 3 consecutive) among the matched peaks only.

    Args:
        matched_ion_annotations: Ion annotations for matched peaks only (E.g. "b1+1", "y2+2").
        ion_type: The ion type to count, either "b" or "y".

    Returns:
        Length of the longest consecutive run. Returns 0 if no ions matched.
    """
    ions = _parse_ions(matched_ion_annotations)
    of_type = np.array(
        [label.startswith(ion_type) for label in ions.labels], dtype=np.bool_
    )
    return _kernels.longest_ion_series(of_type.astype(np.int8), ions.series, of_type, 1)


def compute_complementary_ion_count(
    matched_ion_annotations: List[str], peptide_length: int
) -> int:
    """Count bond positions where both b and y ions are matched.

    For a peptide of length n, bond position i produces b_i and y_(n-i).
    Finding both ions for the same bond provides stronger evidence.

    Args:
        matched_ion_annotations: Ion annotations for matched peaks only (E.g. "b1+1", "y2+2").
        peptide_length: Number of amino acid residues in the peptide.

    Returns:
        Number of bond positions where both complementary b and y ions matched.
    """
    ions = _parse_ions(matched_ion_annotations)
    return _kernels.complementary_ion_count(
        ions.kind, ions.series, _select_all(len(ions.labels)), peptide_length
    )


def compute_max_ion_gap(matched_mz: List[float]) -> float:
    """Compute the largest gap between consecutive matched fragment ions.

    Uses the m/z values of the matched theoretical fragment ions, without accounting for the isotopic envelope.

    Args:
        matched_mz: List of m/z values for matched fragment ions.

    Returns:
        Maximum gap in Daltons, in the dtype of ``matched_mz``. Returns 0.0 if fewer than 2 ions matched.
    """
    if len(matched_mz) < 2:
        return 0.0
    mz = _as_float_array(matched_mz)
    return mz.dtype.type(_kernels.max_ion_gap(mz, _select_all(mz.size)))


########################################################
# Intensity-based Features
########################################################


def compute_b_y_intensity_ratio(
    matched_ion_annotations: List[str],
    matched_ion_intensities_incl_isotopic_env: List[float],
    epsilon: float = 1e-8,
) -> float:
    """Compute the ratio of b-ion to y-ion intensities.

    Sums intensities for all b-ions and y-ions separately (including their isotopic
    envelopes), then returns b_total / (y_total + epsilon). The epsilon ensures
    numerical stability when no y-ions are matched, while still producing a high
    ratio that indicates the absence of y-ions.

    Args:
        matched_ion_annotations: Ion annotations for matched peaks (e.g. "b1+1", "y2+2").
        matched_ion_intensities_incl_isotopic_env: Intensity for each matched ion (including isotopic envelope).
        epsilon: Small value added to y-ion intensity for numerical stability.
            Defaults to 1e-8.

    Returns:
        Ratio of b-ion to y-ion intensities, in the dtype of the intensities.
        Returns 0.0 if no ions are matched.
    """
    if not matched_ion_annotations:
        return 0.0
    ions = _parse_ions(matched_ion_annotations)
    intensities = _as_float_array(matched_ion_intensities_incl_isotopic_env)
    ratio = _kernels.b_y_intensity_ratio(
        ions.kind,
        intensities,
        _select_all(len(ions.labels)),
        _cast([epsilon], intensities),
    )
    return intensities.dtype.type(ratio)


def compute_spectral_angle(
    theoretical_ion_intensities: Union[List[float], float],
    aligned_observed_ion_intensities: List[float],
) -> float:
    """Compute the normalised spectral angle similarity score between the theoretical and aligned observed spectra.

    The intensity lists must align by m/z values (same length, with zeros for unmatched
    observed ions).

    Args:
        theoretical_ion_intensities: Intensity for each theoretical ion. May be NaN/float
            for missing data rows.
        aligned_observed_ion_intensities: Aligned observed intensities (same length as
            theoretical, with 0.0 for unmatched ions).

    Returns:
        Normalised spectral angle similarity score. 1.0 indicates perfect correlation, 0.0 indicates perfect anti-correlation.
        Returns 0.0 for missing data or no matches.
    """
    # Handle missing data (NaN theoretical intensities)
    if isinstance(theoretical_ion_intensities, float) and isnan(
        theoretical_ion_intensities
    ):
        return 0.0

    # Handle edge cases: empty lists or all zeros in observed
    if not theoretical_ion_intensities or not aligned_observed_ion_intensities:
        return 0.0

    theoretical = np.asarray(theoretical_ion_intensities, dtype=np.float64)
    observed = _as_float_array(aligned_observed_ion_intensities)
    if theoretical.size != observed.size:
        raise ValueError(
            "The intensity lists must align by m/z values.\n"
            "This means every theoretical ion must have an observed intensity "
            "or a zero intensity in the aligned observed intensity list."
        )

    return _kernels.spectral_angle(theoretical, observed)


########################################################
# Cross-correlation Score (xcorr)
########################################################


def compute_xcorr(
    observed_mz: List[float],
    observed_intensities: List[float],
    theoretical_mz: Union[List[float], float],
    bin_size: float = XCORR_BIN_SIZE,
    bin_offset: float = XCORR_BIN_OFFSET,
    num_windows: int = XCORR_NUM_WINDOWS,
    max_xcorr_offset: int = XCORR_MAX_OFFSET,
) -> float:
    """Compute the SEQUEST fast xcorr score between observed and theoretical spectra.

    Implements the fast cross-correlation score from Eng et al. (2008). The observed
    spectrum is preprocessed by binning into near-unit-dalton bins with sqrt intensity
    compression, normalizing peak intensities within equal m/z windows, and subtracting
    the local background (mean of ±max_xcorr_offset shifted spectra). The xcorr score
    is the dot product of the preprocessed observed spectrum with a binary theoretical
    spectrum that has 1.0 at each predicted fragment ion bin.

    Reference:
        Eng JK, Fischer B, Grossmann J, Maccoss MJ. A fast SEQUEST cross correlation
        algorithm. J Proteome Res. 2008 Oct;7(10):4598-602.

    Args:
        observed_mz: m/z values of the observed (acquired) spectrum.
        observed_intensities: Intensities corresponding to observed_mz.
        theoretical_mz: m/z values of theoretical fragment ions from intensity
            prediction. May be NaN (float) for missing predictions.
        bin_size: Width of each m/z bin in Daltons. Defaults to 1.0005079.
        bin_offset: Fractional offset for bin assignment. Defaults to 0.4.
        num_windows: Number of equal-width m/z windows for intensity normalization.
            Defaults to 10.
        max_xcorr_offset: Number of bins in each direction for background estimation.
            The background is divided by 2 * max_xcorr_offset. Defaults to 75.

    Returns:
        The xcorr score. Returns 0.0 for missing/empty inputs.
    """
    if isinstance(theoretical_mz, float):
        return 0.0
    if len(observed_mz) == 0 or len(theoretical_mz) == 0:
        return 0.0
    obs_mz = _as_float_array(observed_mz)
    theo_mz = _as_float_array(theoretical_mz)
    return float(
        _kernels.xcorr(
            obs_mz,
            _as_float_array(observed_intensities),
            theo_mz,
            _cast([bin_size, bin_offset], obs_mz),
            _cast([bin_size, bin_offset], theo_mz),
            num_windows,
            max_xcorr_offset,
            XCORR_WINDOW_NORM_VALUE,
        )
    )
