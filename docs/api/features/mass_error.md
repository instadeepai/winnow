# MassErrorFeature

Computes the difference between observed and theoretical precursor mass as a fundamental quality indicator for peptide-spectrum matches.

## Purpose

Mass accuracy is one of the most direct measures of PSM quality. A correctly identified peptide should have a precursor mass very close to its theoretical mass. Large mass errors often indicate:

- Incorrect peptide identification
- Unexpected modifications
- Isotope peak assignment errors
- Instrument calibration issues

This feature provides the calibrator with mass accuracy information to help distinguish true from false identifications.

## Implementation

The mass error is computed as:

```python
mass_error = observed_MH+ - theoretical_MH+
```

Where:

- **Observed MH+** = `precursor_mz × precursor_charge - (precursor_charge - 1) × proton_mass`
- **Theoretical MH+** = `sum(residue_masses) + water_mass + proton_mass`
- `water_mass` = 18.01528 Da
- `proton_mass` = 1.00727 Da

The theoretical mass is calculated by summing the monoisotopic masses of all residue tokens in the predicted peptide sequence.

## Columns

| Column | Unit | Description |
| -------- | ------ | ------------- |
| `mass_error` | Daltons (Da) | Observed MH+ minus theoretical MH+. Positive = observed heavier than expected. |

## Usage

```python
from winnow.calibration.features import MassErrorFeature
from winnow.constants import RESIDUE_MASSES

feature = MassErrorFeature(residue_masses=RESIDUE_MASSES)
calibrator.add_feature(feature)
```

### Parameters

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `residue_masses` | `Dict[str, float]` | Required | Mapping of residue tokens to monoisotopic masses in Daltons |

## Requirements

The dataset must contain:

- `precursor_mz`: Observed precursor m/z value
- `precursor_charge`: Precursor charge state (integer)
- `prediction`: List of residue tokens for the predicted peptide

## Notes

- Mass error is signed: positive values indicate the observed mass is heavier than theoretical
- Typical mass accuracy for modern instruments is < 10 ppm (< 0.01 Da for a 1000 Da peptide)
- For modifications, ensure the `residue_masses` dictionary includes modified residue tokens (e.g., `"M[UNIMOD:35]"` for oxidised methionine)
