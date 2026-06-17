# Mass Error Features

Computes the signed precursor mass error, correcting for possible isotope peak selection by the instrument. Two feature classes are available:

- **`MassErrorPPMFeature`** — error in parts per million (ppm)
- **`MassErrorDaFeature`** — error in Daltons on the neutral-mass scale

Both features share the same isotope-correction logic and differ only in the unit used for the output column.

## Purpose

Mass accuracy is one of the most direct measures of PSM quality. A correctly identified peptide should have a precursor m/z very close to its theoretical m/z. Large mass errors often indicate:

- Incorrect peptide identification
- Unexpected modifications
- Instrument calibration issues

When the instrument selects a precursor ion, it may pick the M+1 or M+2 isotope peak instead of the monoisotopic (M0) peak. Without correction, this would introduce a ~1 Da error that could penalise correct PSMs. Both features account for this by evaluating multiple isotope offsets and selecting the one that gives the smallest absolute ppm error.

## Implementation

For each isotope offset in the configured `isotope_error_range`, the mass error is computed in m/z space. The isotope offset with the smallest absolute ppm error is selected, and its signed value is stored in the requested unit.

**Parts per million (ppm):**

```python
ppm = (mz_theoretical - (mz_measured - isotope × 1.00335 / z)) / mz_measured × 1e6
```

**Daltons (neutral-mass scale):**

```python
da = (mz_theoretical - (mz_measured - isotope × 1.00335 / z)) × z
```

Where:

- **mz_theoretical** = `(neutral_mass + z × proton_mass) / z`
- **neutral_mass** = `sum(residue_masses) + water_mass`
- **1.00335** is the carbon-13 isotope mass shift
- **z** is the precursor charge

The two outputs are related by `da = ppm × mz_measured / 1e6 × z`.

## Columns

| Column | Feature class | Unit | Description |
| -------- | --------------- | ------ | ------------- |
| `mass_error_ppm` | `MassErrorPPMFeature` | Parts per million (ppm) | Signed precursor mass error after isotope correction. Negative = observed m/z is heavier than theoretical. |
| `mass_error_da` | `MassErrorDaFeature` | Daltons (Da) | Same error on the neutral-mass scale. Negative = observed m/z is heavier than theoretical. |

## Usage

```python
from winnow.calibration.features import MassErrorPPMFeature, MassErrorDaFeature
from winnow.constants import RESIDUE_MASSES

ppm_feature = MassErrorPPMFeature(
    residue_masses=RESIDUE_MASSES,
    isotope_error_range=(0, 1),
)
da_feature = MassErrorDaFeature(
    residue_masses=RESIDUE_MASSES,
    isotope_error_range=(0, 1),
)

calibrator.add_feature(ppm_feature)
calibrator.add_feature(da_feature)
```

### Parameters

Both feature classes accept the same constructor arguments:

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
- Isotope selection always uses the smallest absolute **ppm** error, even when computing the Da column
- Typical mass accuracy for modern instruments is < 10 ppm
- The `isotope_error_range` should match the setting used by your data loader
- For modifications, ensure the `residue_masses` dictionary includes modified residue tokens (e.g., `"M[UNIMOD:35]"` for oxidised methionine)
