# MassErrorFeature

Computes the signed precursor mass error in parts per million (ppm), correcting for possible isotope peak selection by the instrument.

## Purpose

Mass accuracy is one of the most direct measures of PSM quality. A correctly identified peptide should have a precursor m/z very close to its theoretical m/z. Large mass errors often indicate:

- Incorrect peptide identification
- Unexpected modifications
- Instrument calibration issues

When the instrument selects a precursor ion, it may pick the M+1 or M+2 isotope peak instead of the monoisotopic (M0) peak. Without correction, this would introduce a ~1 Da error that could penalise correct PSMs. This feature accounts for this by evaluating multiple isotope offsets and selecting the one that gives the smallest absolute error.

## Implementation

For each isotope offset in the configured `isotope_error_range`, the mass error is computed in m/z space as:

```python
ppm = (mz_theoretical - (mz_measured - isotope × 1.00335 / z)) / mz_measured × 1e6
```

Where:

- **mz_theoretical** = `(neutral_mass + z × proton_mass) / z`
- **neutral_mass** = `sum(residue_masses) + water_mass`
- **1.00335** is the carbon-13 isotope mass shift
- **z** is the precursor charge

The isotope offset producing the smallest absolute ppm error is selected, and its signed value is stored.

## Columns

| Column | Unit | Description |
| -------- | ------ | ------------- |
| `mass_error_ppm` | Parts per million (ppm) | Signed precursor mass error after isotope correction. Negative = observed m/z is heavier than theoretical. |

## Usage

```python
from winnow.calibration.features import MassErrorFeature
from winnow.constants import RESIDUE_MASSES

feature = MassErrorFeature(
    residue_masses=RESIDUE_MASSES,
    isotope_error_range=(0, 1),
)
calibrator.add_feature(feature)
```

### Parameters

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `residue_masses` | `Dict[str, float]` | Required | Mapping of residue tokens to monoisotopic masses in Daltons |
| `isotope_error_range` | `Tuple[int, int]` | `(0, 1)` | Range of isotope offsets to evaluate (inclusive). `(0, 1)` considers M0 and M+1. |

## Requirements

The dataset must contain:

- `precursor_mz`: Observed precursor m/z value
- `precursor_charge`: Precursor charge state (integer)
- `prediction`: List of residue tokens for the predicted peptide

## Notes

- The error is signed: negative values indicate the observed m/z is heavier than theoretical
- Typical mass accuracy for modern instruments is < 10 ppm
- The `isotope_error_range` should match the setting used by your data loader
- For modifications, ensure the `residue_masses` dictionary includes modified residue tokens (e.g., `"M[UNIMOD:35]"` for oxidised methionine)
