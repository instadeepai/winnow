# SequenceFeatures

Computes basic peptide sequence properties to help the calibrator distinguish between different peptide types, particularly tryptic vs non-tryptic peptides.

## Purpose

Peptide properties like length and cleavage patterns significantly affect:

- Fragmentation behaviour
- Identification confidence
- Model performance across different sample types

This feature is particularly important for non-tryptic domains, such as immunopeptidomics, where:

- Peptides are typically shorter than tryptic peptides
- A higher proportion of peptides are singly-charged
- C-terminal residues are not constrained to K/R (non-tryptic cleavage)
- Length distributions differ from standard proteomics samples

By providing explicit sequence properties, the calibrator can learn different confidence patterns for different peptide classes.

## Implementation

For each predicted peptide sequence:

1. **Sequence length**: Count the number of residue tokens
2. **C-terminal tryptic check**: Test if the last residue is K (lysine) or R (arginine)
3. **Precursor charge**: Pass through from spectrum metadata (not computed)

## Columns

| Column | Unit | Description |
| -------- | ------ | ------------- |
| `sequence_length` | Count (integer) | Number of residue tokens in the predicted peptide |
| `precursor_charge` | Charge state (integer) | Raw precursor charge from the spectrum |
| `is_c_term_tryptic` | Boolean (0/1) | True if the C-terminal residue is K or R (canonical tryptic cleavage sites) |

## Usage

```python
from winnow.calibration.features import SequenceFeatures

feature = SequenceFeatures()
calibrator.add_feature(feature)
```

### Parameters

`SequenceFeatures` has no configuration parameters.

## Requirements

The dataset must contain:

- `prediction`: List of residue tokens for the predicted peptide
- `precursor_charge`: Precursor charge state (passed through to output)

## Notes

- `is_c_term_tryptic` returns `False` for empty sequences
- The charge is passed through unchanged from the input columns
