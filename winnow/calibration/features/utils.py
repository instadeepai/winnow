"""Utility functions for input validation and spectrum match quality feature computation.

This module provides functions for evaluating how well theoretical peptide
fragmentation spectra match observed experimental spectra. These features
are used by the calibrator to distinguish high-quality PSMs from low-quality ones.
"""

from math import isnan
from typing import Dict, List, Optional, Any, Set, Tuple, Iterator, Union
import bisect
import numpy as np
import pandas as pd

from winnow.datasets.calibration_dataset import CalibrationDataset

# Carbon-13 mass shift for isotopic envelope calculation
CARBON_ISOTOPE_MASS_SHIFT = 1.00335


########################################################
# Helper functions
########################################################


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
# Peak Matching
########################################################


def _iter_candidates_by_distance(
    target_mz: List[float], query_mz: float, insertion_point: int
) -> Iterator[Tuple[int, float]]:
    """Yield (index, distance) pairs outward from insertion point, closest first."""
    left = insertion_point - 1
    right = insertion_point
    n = len(target_mz)

    while left >= 0 or right < n:
        left_dist = abs(target_mz[left] - query_mz) if left >= 0 else float("inf")
        right_dist = abs(target_mz[right] - query_mz) if right < n else float("inf")

        if left_dist <= right_dist:
            yield left, left_dist  # left neighbour is closer
            left -= 1
        else:
            yield right, right_dist  # right neighbour is closer
            right += 1


def _find_peak_index(
    target_mz: List[float],
    query_mz: float,
    mz_tolerance: float,
    excluded_indices: set[int] | None = None,
) -> int | None:
    """Find index of nearest unmatched peak in sorted target_mz within tolerance.

    Searches outward from the binary search insertion point to find the closest
    peak within tolerance that is not in the excluded set.

    Args:
        target_mz: Sorted list of m/z values.
        query_mz: The m/z value to search for.
        mz_tolerance: Tolerance for matching (Daltons).
        excluded_indices: Set of indices to skip (already matched peaks).

    Returns:
        Index of matching peak, or None if no match found.
    """
    if excluded_indices is None:
        excluded_indices = set()

    insertion_point = bisect.bisect_left(target_mz, query_mz)

    # Search outward from insertion point, checking candidates by distance
    for idx, dist in _iter_candidates_by_distance(target_mz, query_mz, insertion_point):
        if dist >= mz_tolerance:
            return None  # out of tolerance and next candidates are further
        if idx not in excluded_indices:
            return idx  # valid match (within tolerance and not already matched)

    return None  # no valid match found


def find_matching_ions(
    source_mz: List[float],
    target_mz: List[float],
    target_intensities: List[float],
    source_annotations: Union[List[bytes], List[str]],
    mz_tolerance: float = 0.02,
) -> Tuple[float, float, List[str], List[float], List[float]]:
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

    Each observed peak can only be matched once. Once an observed peak is assigned to a
    theoretical ion (either as M0 or as part of its isotopic envelope), it is excluded
    from matching subsequent theoretical ions.

    Args:
        source_mz: List of m/z values from the source (theoretical) spectrum.
        target_mz: List of m/z values from the target (observed) spectrum.
        target_intensities: List of intensities corresponding to target m/z values.
        source_annotations: List of ion annotations from Koina (e.g., "b1+3", "y2+2").
        mz_tolerance: Tolerance for matching m/z values (Daltons). Defaults to 0.02 Daltons.

    Returns:
        Tuple of (fraction of matched ions, normalised intensity of matched ions,
        list of matched ion annotations, list of matched ion m/z values,
        list of matched ion intensities including isotopic envelope).
    """
    if isinstance(source_mz, float) and isnan(source_mz):
        return 0.0, 0.0, [], [], []

    num_matches, matched_target_intensity = 0, 0.0
    matched_ion_annotations = []
    matched_ion_mz = []
    matched_ion_intensities = []
    total_target_intensity = sum(target_intensities)

    # Track matched observed peak indices
    matched_indices: set[int] = set()

    # Decode the ion annotations to strings if they are bytes
    source_annotations = [
        ion_annotation.decode() if isinstance(ion_annotation, bytes) else ion_annotation
        for ion_annotation in source_annotations
    ]

    for ion_mz, ion_annotation in zip(source_mz, source_annotations):
        # Find monoisotopic peak (M0), excluding already-matched peaks
        source_ion_charge = extract_fragment_ion_charge(ion_annotation)
        isotope_spacing = CARBON_ISOTOPE_MASS_SHIFT / source_ion_charge
        m0_idx = _find_peak_index(target_mz, ion_mz, mz_tolerance, matched_indices)

        if m0_idx is not None:
            # Count match only for M0 (avoids noise inflation)
            num_matches += 1
            matched_indices.add(m0_idx)

            # Add the ion annotation to the list of matched ion annotations
            matched_ion_annotations.append(ion_annotation)
            # Add the ion m/z to the list of matched ion m/z values
            matched_ion_mz.append(ion_mz)

            # Sum M0 intensity
            ion_intensity = target_intensities[m0_idx]
            matched_target_intensity += ion_intensity

            # Sum isotopic envelope intensities (M+1, M+2, M+3, M+4)
            for i in range(1, 5):
                isotope_mz = ion_mz + i * isotope_spacing
                iso_idx = _find_peak_index(
                    target_mz, isotope_mz, mz_tolerance, matched_indices
                )
                if iso_idx is not None:
                    matched_indices.add(iso_idx)
                    ion_intensity += target_intensities[iso_idx]
                    matched_target_intensity += target_intensities[iso_idx]

            # Record total intensity for this ion (M0 + isotopes)
            matched_ion_intensities.append(ion_intensity)

    return (
        num_matches / len(source_mz),
        matched_target_intensity / total_target_intensity,
        matched_ion_annotations,
        matched_ion_mz,
        matched_ion_intensities,
    )


def compute_ion_identifications(
    dataset: pd.DataFrame,
    source_column: str,
    source_annotation_column: str,
    mz_tolerance: float = 0.02,
    predictions: Optional[List[str]] = None,
) -> Iterator[Tuple[List[float], List[float]]]:
    """Computes the ion match rate and match intensity for each spectrum in the dataset.

    Args:
        dataset: DataFrame containing the mass spectrum data.
        source_column: Column name containing the theoretical m/z values.
        source_annotation_column: Column name containing the ion annotations.
        mz_tolerance: Mass tolerance used to match ions (Daltons). Defaults to 0.02 Daltons.
        predictions: Optional list of tokenised predictions for each spectrum. If not provided, the peptide length will be inferred from the column "predictions" in the metadata.

    Returns:
        Iterator of (ion_match_rate, ion_match_intensity, longest_b_series, longest_y_series, complementary_ion_count, max_ion_gap, b_y_intensity_ratio) tuples.
    """
    per_row_match_results: List[Tuple[float, float, int, int, int, float, float]] = []

    for _, row in dataset.iterrows():
        (
            ion_match,
            ion_match_intensity,
            matched_ion_annotations,
            matched_ion_mz,
            matched_ion_intensities,
        ) = find_matching_ions(
            source_mz=row[source_column],
            target_mz=row["mz_array"],
            target_intensities=row["intensity_array"],
            source_annotations=row[source_annotation_column],
            mz_tolerance=mz_tolerance,
        )

        # Compute the longest consecutive run of matched fragment ions
        longest_b_series = compute_longest_ion_series(matched_ion_annotations, "b")
        longest_y_series = compute_longest_ion_series(matched_ion_annotations, "y")
        # Compute the number of bond positions where both b and y ions are matched
        complementary_ion_count = compute_complementary_ion_count(
            matched_ion_annotations,
            len(predictions) if predictions is not None else len(row["prediction"]),
        )
        # Compute the largest gap between consecutive matched fragment ions
        max_ion_gap = compute_max_ion_gap(matched_ion_mz)
        # Compute the ratio of b-ion to y-ion intensities
        b_y_intensity_ratio = compute_b_y_intensity_ratio(
            matched_ion_annotations, matched_ion_intensities
        )

        per_row_match_results.append(
            (
                ion_match,
                ion_match_intensity,
                longest_b_series,
                longest_y_series,
                complementary_ion_count,
                max_ion_gap,
                b_y_intensity_ratio,
            )
        )

    return zip(*per_row_match_results)


def extract_fragment_ion_charge(annotation: Union[bytes, str]) -> int:
    """Extract the charge states from ion annotations.

    Args:
        annotation: Ion annotation from Koina (e.g., "b1+2", "y2+1", "b3+3").

    Returns:
        Charge of the fragment ion. (e.g. 1, 2, 3)
    """
    label = annotation.decode() if isinstance(annotation, bytes) else str(annotation)
    charge = label.split("+")[1]  # e.g. "1"
    return int(charge)


########################################################
# Ion Coverage Features
########################################################


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
    indices = []
    for anno in matched_ion_annotations:
        # Handle annotations like "b2+2", "y3+3"
        # Extract the base ion type and index
        if anno.startswith(ion_type):
            # Remove charge suffix if present (e.g., "b2+2" -> "b2")
            base = anno.split("+")[0]
            if base[1:].isdigit():
                indices.append(int(base[1:]))

    if not indices:
        return 0

    indices = sorted(set(indices))

    max_run = current_run = 1
    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    return max_run


def compute_complementary_ion_count(
    matched_ion_annotations: List[str], peptide_length: int
) -> int:
    """Count bond positions where both b and y ions are matched.

    For a peptide of length n, bond position i produces b_i and y_(n-1-i).
    Finding both ions for the same bond provides stronger evidence.

    Args:
        matched_ion_annotations: Ion annotations for matched peaks only (E.g. "b1+1", "y2+2").
        peptide_length: Number of amino acid residues in the peptide.

    Returns:
        Number of bond positions where both complementary b and y ions matched.
    """
    b_indices = set()
    y_indices = set()

    for ann in matched_ion_annotations:
        # Handle charge states: "b1", "b2+2", "y3+3"
        base = ann.split("+")[0]
        if base.startswith("b") and base[1:].isdigit():
            b_indices.add(int(base[1:]))
        elif base.startswith("y") and base[1:].isdigit():
            y_indices.add(int(base[1:]))

    # For peptide of length n: bond i produces b_i and y_(n-i)
    # e.g., ACDK (length 4): b1 pairs with y3, b2 pairs with y2, b3 pairs with y1
    return sum(1 for i in b_indices if (peptide_length - i) in y_indices)


def compute_max_ion_gap(matched_mz: List[float]) -> float:
    """Compute the largest gap between consecutive matched fragment ions.

    Uses the m/z values of the matched theoretical fragment ions, without accounting for the isotopic envelope.

    Args:
        matched_mz: List of m/z values for matched fragment ions.

    Returns:
        Maximum gap in Daltons. Returns 0.0 if fewer than 2 ions matched.
    """
    if len(matched_mz) < 2:
        return 0.0

    sorted_mz = sorted(matched_mz)
    return max(sorted_mz[i + 1] - sorted_mz[i] for i in range(len(sorted_mz) - 1))


def compute_b_y_intensity_ratio(
    matched_ion_annotations: List[str],
    matched_ion_intensities: List[float],
    epsilon: float = 1e-8,
) -> float:
    """Compute the ratio of b-ion to y-ion intensities.

    Sums intensities for all b-ions and y-ions separately (including their isotopic
    envelopes), then returns b_total / (y_total + epsilon). The epsilon ensures
    numerical stability when no y-ions are matched, while still producing a high
    ratio that indicates the absence of y-ions.

    Args:
        matched_ion_annotations: Ion annotations for matched peaks (e.g. "b1+1", "y2+2").
        matched_ion_intensities: Intensity for each matched ion (including isotopic envelope).
        epsilon: Small value added to y-ion intensity for numerical stability.
            Defaults to 1e-8.

    Returns:
        Ratio of b-ion to y-ion intensities. Returns 0.0 if no ions are matched.
    """
    if not matched_ion_annotations:
        return 0.0

    b_intensity = 0.0
    y_intensity = 0.0

    for annotation, intensity in zip(matched_ion_annotations, matched_ion_intensities):
        if annotation.startswith("b"):
            b_intensity += intensity
        elif annotation.startswith("y"):
            y_intensity += intensity

    return b_intensity / (y_intensity + epsilon)
