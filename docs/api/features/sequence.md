# SequenceFeatures

Computes basic peptide sequence properties that capture characteristics affecting fragmentation behaviour and identification confidence.

## Purpose

Peptide properties like length and charge state significantly affect fragmentation patterns and model performance. Providing these explicitly allows the calibrator to learn different confidence patterns for peptides of varying sizes and charge states.

## Implementation

For each predicted peptide sequence:

1. **Sequence length**: Count the number of residue tokens
2. **Precursor charge**: Pass through from spectrum metadata (not computed)

## Columns

| Column | Unit | Description |
| -------- | ------ | ------------- |
| `sequence_length` | Count (integer) | Number of residue tokens in the predicted peptide |
| `precursor_charge` | Charge state (integer) | Raw precursor charge from the spectrum |

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
