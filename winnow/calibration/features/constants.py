"""Constants for calibration feature computation."""

# Carbon-13 mass shift for isotopic envelope calculation
CARBON_ISOTOPE_MASS_SHIFT = 1.00335
# Default bin width matching Comet's fragment_bin_tol for near-unit-dalton bins.
XCORR_BIN_SIZE = 1.0005079
# Default bin offset matching Comet's fragment_bin_offset.
XCORR_BIN_OFFSET = 0.4
# Number of equal-width m/z windows for intensity normalization (SEQUEST default).
XCORR_NUM_WINDOWS = 10
# Maximum m/z offset in bins for background subtraction (SEQUEST default ±75).
XCORR_MAX_OFFSET = 75
# Normalization target for the maximum intensity within each window.
XCORR_WINDOW_NORM_VALUE = 50.0
