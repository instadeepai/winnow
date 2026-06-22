# TokenScoreFeatures

Extracts position-level confidence metrics from the top beam prediction's token log-probabilities, identifying "weak link" residues within otherwise confident predictions.

## Purpose

The overall sequence confidence score averages over all token positions, which can mask problems:

- A single low-confidence residue in an otherwise confident prediction
- High variance suggesting the model is uncertain at multiple positions
- Systematic patterns in where uncertainty occurs

By examining token-level scores, the calibrator can identify predictions that look confident overall but contain problematic positions, often indicating errors at specific residues.

## Implementation

For the top beam prediction, we:

1. Extract `token_log_probabilities` from `dataset.predictions[i][0]`
2. Convert to probabilities via `exp(log_prob)`
3. Compute summary statistics across positions

## Columns

| Column | Unit | Description |
| -------- | ------ | ------------- |
| `min_token_probability` | Probability (0-1) | Minimum token probability across all positions in the top prediction. Identifies the "weakest link" residue; if this is very low, the prediction may have an error at that position. |
| `std_token_probability` | Probability (0-1) | Standard deviation of token probabilities. High variance may indicate uncertain positions within an otherwise confident prediction. |

## Usage

```python
from winnow.calibration.features import TokenScoreFeatures

feature = TokenScoreFeatures()
calibrator.add_feature(feature)
```

### Parameters

`TokenScoreFeatures` has no configuration parameters.

## Requirements

The dataset must have beam predictions where:

- `dataset.predictions` is not `None`
- Each prediction has `token_log_probabilities` available (list of log-probs per residue position)

This feature raises `ValueError` if any top prediction is missing `token_log_probabilities`.

## Notes

- All computations use **probabilities** (converted from stored log-probabilities via `exp()`)
- Single-token sequences have `std_token_probability = 0.0` (no variance possible)
- Empty sequences return `min_token_probability = 0.0`
- Low `min_token_probability` combined with high overall confidence is a strong signal of potential error
- This feature complements `BeamFeatures` which looks at sequence-level confidence; `TokenScoreFeatures` looks at position-level confidence
