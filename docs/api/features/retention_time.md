# Retention Time Feature

Trains a per-experiment linear regressor that maps observed retention time (RT) to indexed retention time (iRT) from a Koina iRT model. The absolute error between the sequence-based iRT prediction and the regressor-predicted iRT is used as a calibration feature.

## Purpose

Retention time is orthogonal to fragmentation-based features; it depends on peptide hydrophobicity and chromatographic conditions rather than fragmentation patterns. A peptide that elutes at an unexpected time may be:

- Incorrectly identified
- A modification variant
- Subject to unusual chromatographic behaviour

By comparing predicted vs observed retention times, the calibrator gains an independent signal for PSM quality assessment.

## Implementation

### Step 1: Koina iRT Prediction

We call a Koina iRT model (e.g., `Prosit_2019_irt`) with the predicted peptide sequences. The model returns predicted indexed retention time (iRT) values on a standardised scale.

### Step 2: Per-experiment linear calibration

The RT-to-iRT mapping is inherently experiment-specific because different LC-MS experiments have different chromatographic conditions (column, gradient, temperature, etc.).

**Training phase** (`prepare` method):

1. **Self-supervised training** — High-confidence de novo predictions (top `train_fraction` by confidence score, descending) serve as pseudo-labels. The Koina iRT model is called on these peptide sequences to obtain iRT values, then a `LinearRegression` is fitted from observed RT to iRT. No database labels are needed.
2. **Per-experiment fitting** — Spectra are grouped by their `experiment_name` column. One regressor is fitted per experiment. If `experiment_name` is absent, a single global regressor is fitted with a warning.
3. **Always re-fitted** — The regressor is fitted at both training and inference time (in `prepare()`). It is not persisted inside the calibrator pickle. Given the same data and random seed, the same regressor is produced.

**Prediction phase** (`compute` method):

1. Predict iRT for all peptides using Koina
2. Use the per-experiment regressor to predict what the iRT "should be" given the observed RT
3. Compute the error: `|Koina_iRT - regressor_predicted_iRT|`

### `experiment_name` column

For multi-experiment data, each spectrum must have an `experiment_name` column:

- **MGF files**: Derived automatically from the file stem (e.g., `data/run1.mgf` produces
  `experiment_name = "run1"`).
- **Parquet / IPC files**: If the column already exists in the file, it is stringified and used as-is. If
  not, no experiment name is inferred.

### Regressor checkpoint workflow

For within-experiment use cases, especially well-characterised species where the unlabelled data has an unreliable confidence distribution, you can save the regressors trained during the training step and load them at inference time:

```bash
# Train: saves calibrator AND per-experiment iRT regressors
winnow train ... irt_regressor_output_path=./irt_regressors.pkl

# Predict: loads regressors from training; skips re-fitting for known experiments
winnow predict ... calibrator.irt_regressor_path=./irt_regressors.pkl
```

When pre-fitted regressors are loaded, `prepare()` skips re-fitting for those experiments. Experiments in the inference data that were not in the training checkpoint are still fitted from scratch.

This is separate from the calibrator model itself and should not be confused with the general pretrained calibrator workflow, where regressors are always re-fitted automatically from the inference data.

Regressors can also be saved and loaded programmatically:

```python
# After fitting (e.g., after calibrator.fit(dataset))
rt_feature = calibrator.feature_dict["iRT Feature"]
rt_feature.save_regressors("irt_regressors.pkl")

# Before prediction on new data
rt_feature.load_regressors("irt_regressors.pkl")
```

## Columns

| Column | Unit | Description |
| -------- | ------ | ------------- |
| `irt_error` | iRT units (dimensionless) | Absolute difference between Koina-predicted iRT and regressor-predicted iRT. Large errors suggest the peptide elutes at an unexpected time. |
| `iRT` | iRT units | The raw Koina iRT prediction (stored for reference) |
| `predicted iRT` | iRT units | The regressor-predicted iRT based on observed retention time |

When `learn_from_missing=True`, an additional indicator column is produced:

| Column | Unit | Description |
| -------- | ------ | ------------- |
| `is_missing_irt_error` | Boolean | `True` when the prediction cannot be passed to the Koina iRT model (e.g., exceeds length limits or contains unsupported residues) |

**Note**: The iRT scale is dimensionless but standardised such that the Biognosys iRT kit peptides span approximately -25 to +120 iRT units.

## Usage

```python
from winnow.calibration.features import RetentionTimeFeature

feature = RetentionTimeFeature(
    train_fraction=0.1,
    min_train_points=10,
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
| `train_fraction` | `float` | `0.1` | Top fraction of spectra by confidence (descending) used to train the regressor. Only assumes higher confidence is better. |
| `min_train_points` | `int` | `10` | Minimum training points needed per experiment after applying `train_fraction`. Raises a `ValueError` if fewer are available. |
| `seed` | `int` | `42` | Random seed for reproducibility |
| `unsupported_residues` | `List[str]` | `[]` | Residue tokens not supported by the Koina model |
| `max_peptide_length` | `int` | `30` | Maximum peptide length supported by the model |
| `irt_model_name` | `str` | `"Prosit_2019_irt"` | Name of the Koina iRT model |
| `learn_from_missing` | `bool` | `True` | Whether to impute missing features or filter invalid rows |

## Requirements

The dataset must contain:

- `retention_time`: Observed retention time values (instrument scale)
- `prediction`: Predicted peptide sequence tokens

For multi-experiment data, each spectrum should also have an `experiment_name` column (see [Implementation](#experiment_name-column) above).

## Notes

- The RT-to-iRT regressor is trained during the `ProbabilityCalibrator` training step via `prepare()`, so the same dataset used for calibrator training should be representative of the chromatographic conditions
- iRT error is always positive (absolute value)
- Peptides with unsupported residues or exceeding length limits may not be computable due to Koina model constraints. The defaults match the Prosit model family; if you use a different Koina model, adjust `max_peptide_length` and `unsupported_residues` accordingly. See the [configuration guide](../../configuration.md#koina-model-input-validation) for details.

### Handling missing data

Winnow provides two strategies controlled by `learn_from_missing`:

**Learn strategy** (`learn_from_missing=True`, default):

- Includes an `is_missing_irt_error` indicator column
- Invalid rows get imputed error values (zero)
- Calibrator learns patterns associated with missing data
- Uses all available data, maximising recall

**Filter strategy** (`learn_from_missing=False`):

- Invalid PSMs are automatically filtered from the dataset before Koina is called
- A warning is emitted reporting how many PSMs were removed and which constraints applied
- Filtered PSMs are gone entirely; no indicator column is added
- Calibrator trains only on the remaining clean data

Use the filter strategy when you want strict data quality requirements. See [Handling Missing Features](index.md#handling-missing-features) for the general pattern across Koina-dependent features.
