"""Compiled per-spectrum kernels for fragment ion match features.

The functions here are the single implementation of ion matching, ion coverage,
spectral angle and xcorr. ``winnow.calibration.features.utils`` wraps them for
single spectra and for whole datasets.

Arithmetic follows the dtypes of the inputs, as the reference NumPy code did:
Koina m/z values are float32, so ppm tolerances, isotope positions and ion gaps
are computed in float32, and float32 observed intensities are summed in float32.
Python float constants are therefore passed in pre-cast to the matching dtype
(NumPy casts a Python float to the array dtype before operating; numba would
otherwise promote to float64). Dot products call BLAS through ``np.dot`` so they
round identically to NumPy.
"""

import numpy as np
from numba import njit, prange

ION_OTHER = 0
ION_B = 1
ION_Y = 2
N_ISOTOPES = 4

_JIT = {"cache": True, "nogil": True, "error_model": "numpy"}


@njit(**_JIT)
def _bisect_left(values: np.ndarray, query: float) -> int:
    """Return the leftmost insertion point of query in sorted values."""
    lo, hi = 0, values.size
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] < query:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(**_JIT)
def _find_peak_index(
    target_mz: np.ndarray, query_mz: float, tolerance: float, matched: np.ndarray
) -> int:
    """Return the nearest unmatched peak strictly within tolerance, or -1; ties go left."""
    n = target_mz.size
    right = _bisect_left(target_mz, query_mz)
    left = right - 1
    while left >= 0 or right < n:
        left_dist = abs(target_mz[left] - query_mz) if left >= 0 else np.inf
        right_dist = abs(target_mz[right] - query_mz) if right < n else np.inf
        if left_dist <= right_dist:
            idx, dist = left, left_dist
            left -= 1
        else:
            idx, dist = right, right_dist
            right += 1
        if dist >= tolerance:
            return -1
        if not matched[idx]:
            return idx
    return -1


@njit(**_JIT)
def _tolerance(
    query_mz: float,
    ppm: bool,
    tolerance_cast: np.ndarray,
    million_cast: np.ndarray,
    tolerance_da: float,
) -> float:
    """Return the absolute tolerance in Da, computed in the dtype of query_mz for ppm."""
    if ppm:
        return np.float64(query_mz * tolerance_cast[0] / million_cast[0])
    return tolerance_da


@njit(**_JIT)
def match_ions(
    theo_mz: np.ndarray,
    ion_charge: np.ndarray,
    obs_mz: np.ndarray,
    obs_int: np.ndarray,
    ppm: bool,
    tolerance_cast: np.ndarray,
    million_cast: np.ndarray,
    tolerance_da: float,
    isotope_shift: np.ndarray,
    matched_ion: np.ndarray,
    m0_int: np.ndarray,
    ion_int: np.ndarray,
) -> tuple[int, float, float]:
    """Greedily match theoretical ions (M0 plus four isotopes) to observed peaks.

    Each observed peak is used at most once. Fills ``matched_ion``, ``m0_int``
    and ``ion_int`` (one entry per theoretical ion) and returns the number of
    matched ions, the summed matched intensity and the total observed intensity,
    the last two in the dtype of ``obs_int``.
    """
    zero = np.zeros(1, obs_int.dtype)[0]
    total = zero
    for k in range(obs_int.size):
        total += obs_int[k]
    matched_obs = np.zeros(obs_mz.size, np.bool_)
    matched_total = zero
    n_matches = 0
    for i in range(theo_mz.size):
        ion_mz = theo_mz[i]
        tol = _tolerance(ion_mz, ppm, tolerance_cast, million_cast, tolerance_da)
        m0 = _find_peak_index(obs_mz, ion_mz, tol, matched_obs)
        if m0 < 0:
            continue
        n_matches += 1
        matched_ion[i] = True
        matched_obs[m0] = True
        intensity = obs_int[m0]
        m0_int[i] = intensity
        matched_total += intensity
        for iso in range(1, N_ISOTOPES + 1):
            iso_mz = ion_mz + isotope_shift[ion_charge[i], iso]
            iso_tol = _tolerance(
                iso_mz, ppm, tolerance_cast, million_cast, tolerance_da
            )
            idx = _find_peak_index(obs_mz, iso_mz, iso_tol, matched_obs)
            if idx >= 0:
                matched_obs[idx] = True
                intensity += obs_int[idx]
                matched_total += obs_int[idx]
        ion_int[i] = intensity
    return n_matches, matched_total, total


@njit(**_JIT)
def _series_presence(
    kind: np.ndarray, series: np.ndarray, selected: np.ndarray, ion_type: int, size: int
) -> np.ndarray:
    """Return a mask over series indices 0..size-1 present among selected ions of one type."""
    present = np.zeros(size, np.bool_)
    for i in range(kind.size):
        if selected[i] and kind[i] == ion_type and 0 <= series[i] < size:
            present[series[i]] = True
    return present


@njit(**_JIT)
def _max_series(series: np.ndarray, selected: np.ndarray) -> int:
    """Return the largest series index among selected ions, or -1."""
    top = -1
    for i in range(series.size):
        if selected[i] and series[i] > top:
            top = series[i]
    return top


@njit(**_JIT)
def longest_ion_series(
    kind: np.ndarray, series: np.ndarray, selected: np.ndarray, ion_type: int
) -> int:
    """Return the longest run of consecutive series indices of one ion type among selected ions."""
    present = _series_presence(
        kind, series, selected, ion_type, _max_series(series, selected) + 1
    )
    best = run = 0
    for is_present in present:
        run = run + 1 if is_present else 0
        best = max(best, run)
    return best


@njit(**_JIT)
def complementary_ion_count(
    kind: np.ndarray, series: np.ndarray, selected: np.ndarray, peptide_length: int
) -> int:
    """Count bond positions i where b_i and y_(peptide_length - i) are both selected."""
    size = _max_series(series, selected) + 1
    b_present = _series_presence(kind, series, selected, ION_B, size)
    y_present = _series_presence(kind, series, selected, ION_Y, size)
    count = 0
    for b_idx in range(size):
        y_idx = peptide_length - b_idx
        if b_present[b_idx] and 0 <= y_idx < size and y_present[y_idx]:
            count += 1
    return count


@njit(**_JIT)
def max_ion_gap(mz: np.ndarray, selected: np.ndarray) -> float:
    """Return the largest gap between consecutive selected m/z values (dtype of mz), or -1 if fewer than two."""
    picked = np.sort(mz[selected])
    if picked.size < 2:
        return -1.0
    best = picked[1] - picked[0]
    for i in range(2, picked.size):
        best = max(best, picked[i] - picked[i - 1])
    return best


@njit(**_JIT)
def b_y_intensity_ratio(
    kind: np.ndarray,
    intensities: np.ndarray,
    selected: np.ndarray,
    epsilon_cast: np.ndarray,
) -> float:
    """Return summed b over summed y intensity plus epsilon, in the dtype of intensities."""
    b_total = np.zeros(1, intensities.dtype)[0]
    y_total = b_total
    for i in range(kind.size):
        if selected[i] and kind[i] == ION_B:
            b_total += intensities[i]
        elif selected[i] and kind[i] == ION_Y:
            y_total += intensities[i]
    return b_total / (y_total + epsilon_cast[0])


@njit(**_JIT)
def spectral_angle(theoretical: np.ndarray, observed: np.ndarray) -> float:
    """Return the normalised spectral angle; the observed norm is taken in its own dtype, as NumPy does."""
    theoretical_norm = np.sqrt(np.dot(theoretical, theoretical))
    observed_norm = np.float64(np.sqrt(np.dot(observed, observed)))
    if theoretical_norm == 0.0 or observed_norm == 0.0:
        return 0.0
    dot = np.dot(theoretical, observed.astype(np.float64)) / (
        theoretical_norm * observed_norm
    )
    dot = min(max(dot, -1.0), 1.0)
    return 1 - (2 * np.arccos(dot) / np.pi)


@njit(**_JIT)
def _bin_index(mz: float, bin_cast: np.ndarray) -> int:
    """Return the xcorr bin of one m/z value, computed in the dtype of bin_cast."""
    return int(mz / bin_cast[0] + bin_cast[1])


@njit(**_JIT)
def _bin_observed(
    obs_mz: np.ndarray, obs_int: np.ndarray, bin_cast: np.ndarray
) -> np.ndarray:
    """Bin observed peaks keeping the largest sqrt intensity per bin; sqrt is taken in the intensity dtype."""
    num_bins = _bin_index(obs_mz.max(), bin_cast) + 1
    binned = np.zeros(num_bins)
    for k in range(obs_mz.size):
        idx = _bin_index(obs_mz[k], bin_cast)
        if 0 <= idx < num_bins and obs_int[k] >= 0:
            binned[idx] = max(binned[idx], np.float64(np.sqrt(obs_int[k])))
    return binned


@njit(**_JIT)
def _normalize_windows(
    binned: np.ndarray, num_windows: int, window_norm_value: float
) -> None:
    """Scale each of num_windows equal regions in place so its maximum is window_norm_value."""
    num_bins = binned.size
    window_size = max(num_bins // num_windows, 1)
    for w in range(num_windows):
        start = w * window_size
        end = min(start + window_size, num_bins) if w < num_windows - 1 else num_bins
        if start >= end:
            raise ValueError(
                "zero-size array to reduction operation maximum which has no identity"
            )
        top = binned[start:end].max()
        if top > 0:
            factor = window_norm_value / top
            for k in range(start, end):
                binned[k] = binned[k] * factor


@njit(**_JIT)
def _subtract_background(binned: np.ndarray, max_offset: int) -> np.ndarray:
    """Subtract the mean of the +-max_offset neighbouring bins from each bin."""
    num_bins = binned.size
    cumsum = np.zeros(num_bins + 1)
    for k in range(num_bins):
        cumsum[k + 1] = cumsum[k] + binned[k]
    divisor = 2 * max_offset
    processed = np.empty(num_bins)
    for k in range(num_bins):
        lo = max(0, k - max_offset)
        hi = min(num_bins, k + max_offset + 1)
        processed[k] = binned[k] - ((cumsum[hi] - cumsum[lo]) - binned[k]) / divisor
    return processed


@njit(**_JIT)
def xcorr(
    obs_mz: np.ndarray,
    obs_int: np.ndarray,
    theo_mz: np.ndarray,
    obs_bin_cast: np.ndarray,
    theo_bin_cast: np.ndarray,
    num_windows: int,
    max_offset: int,
    window_norm_value: float,
) -> float:
    """Return the SEQUEST fast xcorr; ``*_bin_cast`` hold (bin_size, bin_offset) in the dtype of each m/z array."""
    if obs_mz.size == 0 or theo_mz.size == 0:
        return 0.0
    binned = _bin_observed(obs_mz, obs_int, obs_bin_cast)
    _normalize_windows(binned, num_windows, window_norm_value)
    processed = _subtract_background(binned, max_offset)
    theoretical = np.zeros(binned.size)
    for k in range(theo_mz.size):
        idx = _bin_index(theo_mz[k], theo_bin_cast)
        if 0 <= idx < binned.size:
            theoretical[idx] = 1.0
    return np.dot(theoretical, processed)


# Column order of the float, int and flag outputs of ion_identifications.
FLOAT_COLUMNS = (
    "match_rate",
    "match_intensity",
    "max_ion_gap",
    "b_y_ratio",
    "spectral_angle",
    "xcorr",
)
INT_COLUMNS = ("longest_b", "longest_y", "complementary")
FLAG_COLUMNS = ("match_intensity_typed", "max_ion_gap_typed", "b_y_ratio_typed")


@njit(**_JIT)
def _any_by_matched(kind: np.ndarray, matched: np.ndarray) -> bool:
    """Return True if any matched ion is a b or y ion."""
    for i in range(kind.size):
        if matched[i] and kind[i] != ION_OTHER:
            return True
    return False


@njit(**_JIT)
def _spectrum_features(
    s: int,
    obs_offsets: np.ndarray,
    obs_mz: np.ndarray,
    obs_int: np.ndarray,
    theo_offsets: np.ndarray,
    theo_mz: np.ndarray,
    theo_int: np.ndarray,
    kind: np.ndarray,
    series: np.ndarray,
    charge: np.ndarray,
    peptide_length: np.ndarray,
    ppm: bool,
    tolerance_cast: np.ndarray,
    million_cast: np.ndarray,
    tolerance_da: float,
    isotope_shift: np.ndarray,
    epsilon_cast: np.ndarray,
    obs_bin_cast: np.ndarray,
    theo_bin_cast: np.ndarray,
    num_windows: int,
    max_offset: int,
    window_norm_value: float,
    floats: np.ndarray,
    ints: np.ndarray,
    flags: np.ndarray,
) -> None:
    """Fill row s of the output arrays for one non-missing spectrum."""
    o0, o1 = obs_offsets[s], obs_offsets[s + 1]
    t0, t1 = theo_offsets[s], theo_offsets[s + 1]
    s_obs_mz, s_obs_int = obs_mz[o0:o1], obs_int[o0:o1]
    s_theo_mz, s_kind, s_series = theo_mz[t0:t1], kind[t0:t1], series[t0:t1]
    n_ions = t1 - t0

    matched = np.zeros(n_ions, np.bool_)
    m0_int = np.zeros(n_ions, obs_int.dtype)
    ion_int = np.zeros(n_ions, obs_int.dtype)
    n_matches, matched_total, total = match_ions(
        s_theo_mz, charge[t0:t1], s_obs_mz, s_obs_int, ppm, tolerance_cast,
        million_cast, tolerance_da, isotope_shift, matched, m0_int, ion_int,
    )  # fmt: skip
    floats[s, 0] = n_matches / n_ions
    floats[s, 1] = matched_total / total
    flags[s, 0] = True

    ints[s, 0] = longest_ion_series(s_kind, s_series, matched, ION_B)
    ints[s, 1] = longest_ion_series(s_kind, s_series, matched, ION_Y)
    ints[s, 2] = complementary_ion_count(s_kind, s_series, matched, peptide_length[s])

    if n_matches >= 2:
        floats[s, 2] = max_ion_gap(s_theo_mz, matched)
        flags[s, 1] = True
    if n_matches > 0:
        floats[s, 3] = b_y_intensity_ratio(s_kind, ion_int, matched, epsilon_cast)
        flags[s, 2] = _any_by_matched(s_kind, matched)

    # NumPy builds the aligned observed array in float32 only when every ion matched
    # float32 intensities; otherwise the 0.0 placeholders make it float64.
    if n_ions > 0 and n_matches == n_ions:
        floats[s, 4] = spectral_angle(theo_int[t0:t1], m0_int)
    elif n_ions > 0:
        floats[s, 4] = spectral_angle(theo_int[t0:t1], m0_int.astype(np.float64))

    floats[s, 5] = xcorr(
        s_obs_mz, s_obs_int, s_theo_mz, obs_bin_cast, theo_bin_cast,
        num_windows, max_offset, window_norm_value,
    )  # fmt: skip


@njit(parallel=True, **_JIT)
def ion_identifications(
    obs_offsets: np.ndarray,
    obs_mz: np.ndarray,
    obs_int: np.ndarray,
    theo_offsets: np.ndarray,
    theo_mz: np.ndarray,
    theo_int: np.ndarray,
    kind: np.ndarray,
    series: np.ndarray,
    charge: np.ndarray,
    missing: np.ndarray,
    peptide_length: np.ndarray,
    ppm: bool,
    tolerance_cast: np.ndarray,
    million_cast: np.ndarray,
    tolerance_da: float,
    isotope_shift: np.ndarray,
    epsilon_cast: np.ndarray,
    obs_bin_cast: np.ndarray,
    theo_bin_cast: np.ndarray,
    num_windows: int,
    max_offset: int,
    window_norm_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute all fragment ion match features for every spectrum in parallel.

    Spectra are stored flat: spectrum ``s`` owns ``obs_*[obs_offsets[s]:obs_offsets[s + 1]]``
    and ``theo_*[theo_offsets[s]:theo_offsets[s + 1]]``. Missing rows produce zeros.

    Returns:
        (floats, ints, flags): arrays with one row per spectrum and columns in the
        order of ``FLOAT_COLUMNS``, ``INT_COLUMNS`` and ``FLAG_COLUMNS``. A flag is
        set where the value was computed in the intensity or m/z dtype rather than
        being a plain 0.0 default.
    """
    n = missing.size
    floats = np.zeros((n, len(FLOAT_COLUMNS)))
    ints = np.zeros((n, len(INT_COLUMNS)), np.int64)
    flags = np.zeros((n, len(FLAG_COLUMNS)), np.bool_)
    for s in prange(n):
        if not missing[s]:
            _spectrum_features(
                s, obs_offsets, obs_mz, obs_int, theo_offsets, theo_mz, theo_int,
                kind, series, charge, peptide_length, ppm, tolerance_cast,
                million_cast, tolerance_da, isotope_shift, epsilon_cast,
                obs_bin_cast, theo_bin_cast, num_windows, max_offset,
                window_norm_value, floats, ints, flags,
            )  # fmt: skip
    return floats, ints, flags
