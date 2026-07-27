# FDR Control API

The `winnow.fdr` module implements false discovery rate (FDR) estimation and control methods for *de novo* peptide sequencing using both database-grounded and non-parametric approaches.

## Base interface

### FDRControl

Abstract base class that defines the interface for all FDR control methods in Winnow.

```python
from winnow.fdr.base import FDRControl

# All FDR methods inherit from this base class
# and implement the required abstract methods
```

**Core Methods:**

- `fit(dataset)`: Train the FDR model on a dataset
- `get_confidence_cutoff(threshold)`: Get confidence cutoff for target FDR
- `compute_fdr(score)`: Compute FDR estimate for a confidence score
- `filter_entries(dataset, threshold)`: Filter PSMs at target FDR threshold
- `add_psm_fdr(dataset, confidence_col)`: Add PSM-specific FDR values

## Implementations

### DatabaseGroundedFDRControl

Implements database-grounded FDR control using database search results as ground truth for FDR estimation.

```python
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.fdr import DatabaseGroundedFDRControl

# Create FDR controller
fdr_control = DatabaseGroundedFDRControl(
    confidence_feature="confidence",
    drop=10,  # Drop top N predictions for stability
)

# Fit using a CalibrationDataset with loader-finalised metadata
# (requires correct, valid_sequence and confidence columns by default)
fdr_control.fit(dataset=calibration_dataset)

# Or fit against a proxy correctness column (SDK only; not exposed in the CLI),
# e.g. proteome_hit from winnow.utils.proteome.annotate_calibration_dataset.
# Custom correct_column values do not require valid_sequence.
fdr_control.fit(dataset=annotated_dataset, correct_column="proteome_hit")

# Get confidence cutoff for 1% FDR
confidence_cutoff = fdr_control.get_confidence_cutoff(threshold=0.01)

# Filter dataset at target FDR
filtered_data = dataset[dataset["confidence"] >= confidence_cutoff]

# Add PSM-specific FDR values
dataset_with_fdr = fdr_control.add_psm_fdr(dataset, "confidence")

# Add PSM-specific q-values
dataset_with_q_values = fdr_control.add_psm_q_value(dataset, "confidence")
```

**Key Features:**

- **Ground Truth Validation**: Uses loader-derived `correct` labels from database-grounded sequences by default, or a custom boolean proxy via `correct_column` (for example `proteome_hit`)
- **Precision-Recall Analysis**: Computes precision-recall curves from finalised predictions
- **Stability Control**: Drop parameter for robust threshold estimation
- **Finalised Metadata Only**: Does not tokenise peptides or compute match labels; prefer loading labelled data through a DatasetLoader first

**Required Data:**

- Confidence scores (configurable column name)
- Boolean PSM correctness column (`correct` by default, or `correct_column`)
- When `correct_column` is `"correct"`: boolean `valid_sequence` used for filtering eligible fit rows
- When `correct_column` is a proxy such as `"proteome_hit"`: `valid_sequence` is not required

**Fit vs apply:**

`fit` builds the FDR curve only from rows with `valid_sequence=True`. Applying a cutoff is score-based only: `get_confidence_cutoff`, `add_psm_fdr`, and `add_psm_q_value` do not require labels, so retained PSMs may include rows that never entered the curve (unlabelled spectra, or `valid_sequence=False`). That is intentional when filtering by confidence alone.

Power users can fit on one labelled dataset and apply the resulting cutoff to another:

```python
fdr_control.fit(labelled_holdout)
cutoff = fdr_control.get_confidence_cutoff(threshold=0.01)

# Apply to any dataset that has the confidence column (labels optional)
target_metadata = fdr_control.add_psm_fdr(
    target_dataset.metadata, confidence_col="calibrated_confidence"
)
retained = target_dataset.filter_entries(
    metadata_predicate=lambda row: row["calibrated_confidence"] < cutoff
)
```

The `winnow predict` CLI still fits and applies on a single labelled input when using database-grounded FDR; use the library API above for transferred cutoffs.

### NonParametricFDRControl

Uses a label-free, non-parametric method for FDR estimation, specifically designed for scenarios where database ground truth is unavailable.

```python
from winnow.fdr import NonParametricFDRControl

# Create non-parametric FDR controller
fdr_control = NonParametricFDRControl()

# Fit estimation method to a Series of confidence scores
fdr_control.fit(dataset=dataset["confidence"])

# Get confidence cutoff for 5% FDR
confidence_cutoff = fdr_control.get_confidence_cutoff(threshold=0.05)

# Compute FDR for specific score
fdr_estimate = fdr_control.compute_fdr(score=0.8)

# Compute posterior error probability (local FDR)
pep = fdr_control.compute_posterior_probability(score=0.8)

# Add PSM-specific FDR values
dataset_with_fdr = fdr_control.add_psm_fdr(dataset, "confidence")

# Add PSM-specific q-values
dataset_with_q_values = fdr_control.add_psm_q_value(dataset, "confidence")
```

**Key Features:**

- **Non-parametric estimation**: Estimates FDR directly by assuming PSM confidences are calibrated
- **Multiple Metrics**: Computes FDR, q-values, posterior error probability
  - **FDR**: `compute_fdr(score)` - False discovery rate at cutoff
  - **PEP**: `compute_posterior_probability(score)` - Posterior error probability
  - **Q-value**: `compute_q_value(score)` - Minimum FDR for significance
- **No Ground Truth Required**: Works with confidence scores alone

## Additional features

### PSM-specific FDR

Both methods support PSM-specific FDR estimation:

```python
# Add FDR values for each PSM
dataset_with_fdr = fdr_control.add_psm_fdr(
    dataset_metadata=dataset,
    confidence_col="confidence"
)

# Access PSM-specific FDR values
psm_fdr_values = dataset_with_fdr["psm_fdr"]
```

### Q-values

Both methods support q-value computation, the minimum FDR threshold at which a given PSM is significant.

```python
# Add q-values for each PSM
dataset_with_q_values = fdr_control.add_psm_q_value(
    dataset_metadata=dataset,
    confidence_col="confidence"
)

# Access PSM-specific FDR values
psm_q_values = dataset_with_q_values["psm_q_value"]
```

### Confidence curves

Generate FDR vs confidence curves for analysis:

```python
# Get confidence curve
fdr_thresholds, confidence_cutoffs = fdr_control.get_confidence_curve(
    resolution=0.01,        # FDR resolution
    min_confidence=0.01,    # Minimum FDR threshold
    max_confidence=0.50     # Maximum FDR threshold
)

# Plot or analyse the curve
import matplotlib.pyplot as plt
plt.plot(fdr_thresholds, confidence_cutoffs)
plt.xlabel("FDR Threshold")
plt.ylabel("Confidence Cutoff")
```

### Dataset filtering

Filter PSM datasets at target FDR levels:

```python
from winnow.datasets.psm_dataset import PSMDataset

# Filter PSMDataset at 1% FDR
filtered_psms = fdr_control.filter_entries(
    dataset=psm_dataset,
    threshold=0.01
)

print(f"Retained {len(filtered_psms)} PSMs at 1% FDR")
```

### FDR estimation method selection

**Use DatabaseGroundedFDRControl when:**

- High-quality database search results available
- Not restricted to *de novo* sequencing outputs

**Use NonParametricFDRControl when:**

- No database ground truth available
- Working with *de novo* sequencing outputs
- Require additional PSM-specific evaluation metrics such as posterior error probabilities

For detailed examples and usage patterns, refer to the [examples notebook](https://github.com/instadeepai/winnow/blob/main/examples/getting_started_with_winnow.ipynb).
