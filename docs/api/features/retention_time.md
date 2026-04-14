# RetentionTimeFeature

Trains a per-experiment linear regressor that maps observed retention time (RT) to indexed retention time (iRT) from a Koina iRT model, using the absolute prediction error as a calibration feature.

## Purpose

Retention time depends on peptide hydrophobicity and chromatographic conditions, making it orthogonal to fragmentation-based features. A correctly identified peptide should elute close to its predicted iRT, while a large discrepancy suggests the identification may be incorrect, even when the de novo confidence score is high. Incorporating this chromatographic signal gives the calibrator an independent axis for rescoring PSMs.

## Implementation

### Per-experiment iRT regression

The RT-to-iRT mapping is inherently experiment-specific because different LC-MS experiments
have different chromatographic conditions (column, gradient, temperature, etc.).

**How it works:**

1. **Self-supervised training** -- The regressor uses high-confidence de novo predictions
   (top `train_fraction` by confidence score) as pseudo-labels. It calls the Koina iRT
   model on these peptide sequences to obtain iRT values, then fits a `LinearRegression`
   from observed RT to iRT. No database labels are needed.

2. **Per-experiment fitting** -- Spectra are grouped by their `experiment_name` column.
   One regressor is fitted per experiment. If `experiment_name` is absent, a single global
   regressor is fitted with a warning.

3. **Always re-fitted** -- The regressor is fitted at both training and inference time (in
   the `prepare()` step). It is not persisted inside the calibrator model directory by
   default. Given the same data and random seed, the same regressor is produced.

### `experiment_name` column

For multi-experiment data, each spectrum must have an `experiment_name` column:

- **MGF files**: Derived automatically from the file stem (e.g., `data/run1.mgf` produces
  `experiment_name = "run1"`).
- **Parquet / IPC files**: If the column already exists in the file, it is used as-is. If
  not, the file stem is used as the experiment name.

## Columns

| Column | Unit | Description |
| -------- | ------ | ------------- |
| `irt_error` | iRT units (dimensionless) | Absolute difference between Koina-predicted iRT and regressor-predicted iRT. Large errors suggest the peptide elutes at an unexpected time. |

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
| `train_fraction` | `float` | `0.1` | Top fraction of spectra by confidence (descending) used to train the regressor |
| `min_train_points` | `int` | `10` | Minimum training points needed per experiment after applying `train_fraction`. Raises a `ValueError` if fewer are available. |
| `seed` | `int` | `42` | Random seed for reproducibility |
| `unsupported_residues` | `List[str]` | `[]` | Residue tokens not supported by the Koina model |
| `max_peptide_length` | `int` | `30` | Maximum peptide length supported by the model |
| `irt_model_name` | `str` | `"Prosit_2019_irt"` | Name of the Koina iRT model |
| `learn_from_missing` | `bool` | `True` | Whether to impute missing features or filter invalid rows |

## Regressor checkpoint workflow

For within-experiment use cases -- especially well-characterised species where the
unlabelled data has an unreliable confidence distribution -- you can save the regressors
trained during the training step and load them at inference time:

```bash
# Train: saves calibrator AND per-experiment iRT regressors
winnow train ... irt_regressor_output_path=./irt_regressors.safetensors

# Predict: loads regressors from training; skips re-fitting for known experiments
winnow predict ... calibrator.irt_regressor_path=./irt_regressors.safetensors
```

When pre-fitted regressors are loaded, `prepare()` skips re-fitting for those experiments.
Experiments in the inference data that were not in the training checkpoint are still fitted
from scratch.

This is separate from the calibrator model itself and should not be confused with the
general pretrained calibrator workflow, where regressors are always re-fitted automatically
from the inference data.

Regressors can also be saved and loaded programmatically:

```python
# After fitting (e.g., after calibrator.fit(dataset))
rt_feature = calibrator.feature_dict["iRT Feature"]
rt_feature.save_regressors("irt_regressors.safetensors")

# Before prediction on new data
rt_feature.load_regressors("irt_regressors.safetensors")
```

## Requirements

The dataset must contain:

- `retention_time`: Observed retention time values (instrument scale)
- `prediction`: Predicted peptide sequence tokens

## Notes

- The RetentionTimeFeature defaults to re-fitting per experiment from the data at hand via `prepare()`, at both training and inference time
- iRT error is always positive (absolute value)
- Peptides with unsupported residues or exceeding length limits are handled according to `learn_from_missing` setting
- When `learn_from_missing=True`, invalid rows get imputed error values and an `is_missing_irt_error` indicator column
