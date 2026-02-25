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
from winnow.fdr import DatabaseGroundedFDRControl
from winnow.constants import RESIDUE_MASSES

# Create FDR controller
fdr_control = DatabaseGroundedFDRControl(confidence_feature="confidence")

# Fit using labelled dataset
fdr_control.fit(
    dataset=labelled_dataframe,
    residue_masses=RESIDUE_MASSES,
    isotope_error_range=(0, 1),
    drop=10  # Drop top N predictions for stability
)

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

- **Ground Truth Validation**: Uses database search results for validation
- **Precision-Recall Analysis**: Computes precision-recall curves from predictions
- **Isotope Error Handling**: Supports configurable isotope error ranges
- **Stability Control**: Drop parameter for robust threshold estimation

**Required Data:**

- Ground truth peptide sequences (`sequence` column)
- Predicted peptide sequences (`prediction` column)
- Confidence scores (configurable column name)

### NonParametricFDRControl

Uses a label-free, non-parametric method for FDR estimation, specifically designed for scenarios where database ground truth is unavailable.

```python
from winnow.fdr import NonParametricFDRControl

# Create non-parametric FDR controller
fdr_control = NonParametricFDRControl()

# Fit estimation method to confidence scores
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
