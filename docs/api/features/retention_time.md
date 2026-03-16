# RetentionTimeFeature

Uses a Koina iRT model to predict indexed retention times and calibrate against observed retention times, providing chromatographic information for confidence calibration.

## Purpose

Retention time is orthogonal to fragmentation-based features; it depends on peptide hydrophobicity and chromatographic conditions rather than fragmentation patterns. A peptide that elutes at an unexpected time may be:

- Incorrectly identified
- A modification variant
- Subject to unusual chromatographic behaviour

By comparing predicted vs observed retention times, the calibrator gains an independent signal for PSM quality assessment.

## Implementation

### Step 1: Koina iRT Prediction

We call a Koina iRT model (e.g., `Prosit_2019_irt`) with the predicted peptide sequences. The model returns predicted indexed retention time (iRT) values on a standardised scale.

### Step 2: Linear Calibration via MLP

Since raw instrument retention times are in arbitrary units (typically minutes or seconds depending on the LC gradient), we train a lightweight MLP regressor to map observed retention times to the iRT scale:

**Training phase** (`prepare` method):

1. Ranked PSMs in descending order of confidence
2. Select the top fraction of the dataset (default: 10%) for RT-to-iRT training
3. Get "gold-standard" Koina iRT predictions for these peptides using their predicted sequence labels
4. Train a single-layer MLP (default: 10 hidden units) where:
   - **Input**: Observed retention time (instrument scale)
   - **Target**: Predicted iRT from Koina

**Prediction phase** (`compute` method):

1. Predict iRT for all peptides using Koina
2. Use the trained MLP to predict what the iRT "should be" given the observed RT
3. Compute the error: `|Koina_iRT - MLP_predicted_iRT|`

This approach learns the linear (or near-linear) relationship between instrument RT and iRT for each dataset, handling different gradient lengths and LC setups automatically.

## Columns

| Column | Unit | Description |
| -------- | ------ | ------------- |
| `irt_error` | iRT units (dimensionless) | Absolute difference between Koina-predicted iRT and MLP-predicted iRT. Large errors suggest the peptide elutes at an unexpected time. |
| `iRT` | iRT units | The raw Koina iRT prediction (stored for reference) |
| `predicted iRT` | iRT units | The MLP-predicted iRT based on observed retention time |

**Note**: The iRT scale is dimensionless but standardised such that the Biognosys iRT kit peptides span approximately -25 to +120 iRT units.

## Usage

```python
from winnow.calibration.features import RetentionTimeFeature

feature = RetentionTimeFeature(
    hidden_dim=10,
    train_fraction=0.1,
    unsupported_residues=["N[UNIMOD:7]", "Q[UNIMOD:7]"],
    max_peptide_length=30,
    irt_model_name="Prosit_2019_irt",
    learn_from_missing=True,
)
calibrator.add_feature(feature)
```

### Parameters

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `hidden_dim` | `int` | `10` | Number of hidden units in the MLP regressor |
| `train_fraction` | `float` | `0.1` | Fraction of dataset used to train the RT-to-iRT mapping |
| `unsupported_residues` | `List[str]` | `[]` | Residue tokens not supported by the Koina model |
| `max_peptide_length` | `int` | `30` | Maximum peptide length supported by the model |
| `irt_model_name` | `str` | `"Prosit_2019_irt"` | Name of the Koina iRT model |
| `learn_from_missing` | `bool` | `True` | Whether to impute missing features or filter invalid rows |

## Requirements

The dataset must contain:

- `retention_time`: Observed retention time values (instrument scale)
- `prediction`: Predicted peptide sequence tokens

## Notes

- The RT-iRT MLP is trained during the ProbabilityCalibrator model training step via `prepare()`, so the same dataset used for calibrator training should be representative of the chromatographic conditions
- iRT error is always positive (absolute value)
- Peptides with unsupported residues or exceeding length limits are handled according to `learn_from_missing` setting
- When `learn_from_missing=True`, invalid rows get imputed error values and an `is_missing_irt_error` indicator column
