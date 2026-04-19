# BeamFeatures

Extracts confidence signals from beam search diversity metrics, measuring how confident the de novo sequencer is in its top prediction relative to alternatives.

## Purpose

Beam search produces multiple candidate peptide sequences ranked by probability. The distribution of these probabilities provides valuable calibration signals:

- **High confidence**: Top prediction has much higher probability than alternatives
- **Low confidence**: Multiple candidates have similar probabilities (uncertainty)

These metrics help the calibrator identify predictions where the sequencer was uncertain, even if the top-1 confidence score appears high.

## Implementation

All computations use **probabilities** converted from stored log-probabilities via `exp()`.

For each spectrum with beam search results `[p₁, p₂, ..., pₙ]` where `p₁` is the top prediction:

1. **Margin**: `p₁ - p₂` (probability difference between top-1 and top-2)
2. **Median Margin**: `p₁ - median(p₂, ..., pₙ)` (difference from median runner-up)
3. **Entropy**: Shannon entropy of normalised runner-up distribution
4. **Z-score**: `(p₁ - mean(all)) / std(all)` (how unusual is the top score)
5. **Edit distance**: Normalised Levenshtein distance between top-1 and top-2 sequences, treating I/L as the same residue.

## Columns

| Column | Unit | Description |
| -------- | ------ | ------------- |
| `margin` | Probability difference (0-1) | Difference between top-1 and top-2 sequence probabilities. Larger margin = more confident. |
| `median_margin` | Probability difference (0-1) | Difference between top-1 probability and median probability of runner-ups. |
| `entropy` | Nats | Shannon entropy of the **normalised** runner-up probability distribution. Higher entropy = more uncertainty among alternatives. |
| `z-score` | Standard deviations | Z-score of the top-1 probability relative to the full beam distribution (mean and std computed over all beam probabilities). |
| `edit_distance` | Fraction (0-1) | Normalised Levenshtein edit distance between the top-1 and top-2 token sequences, divided by the length of the longer sequence. |

## Usage

```python
from winnow.calibration.features import BeamFeatures

feature = BeamFeatures()
calibrator.add_feature(feature)
```

### Parameters

`BeamFeatures` has no configuration parameters.

## Requirements

The dataset must have beam predictions available (`dataset.predictions` must not be `None`). Each prediction should be a list of `ScoredSequence` objects with `sequence_log_probability` attributes.

## Notes

- A warning is emitted if any beam search results have fewer than two sequences
- When beam size is 1, margin and entropy default to 0; normalised edit distance is set to 1.0.
- Entropy is computed on the **normalised** runner-up probabilities (excluding top-1)
- Z-score uses the full beam including top-1 for mean/std calculation
- All probability values are derived from `exp(sequence_log_probability)`
