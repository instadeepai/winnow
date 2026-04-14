# Calibration API

The `winnow.calibration` module implements confidence calibration for peptide-spectrum matches using machine learning-based feature extraction and neural network classification.

## Classes

### ProbabilityCalibrator

The main calibration model that transforms raw confidence scores into calibrated probabilities using a PyTorch neural network (`CalibratorNetwork`) with various peptide and spectral features.

```python
from winnow.calibration import ProbabilityCalibrator
from winnow.calibration.calibration_features import (
    MassErrorFeature, FragmentMatchFeatures, BeamFeatures
)
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.constants import RESIDUE_MASSES

# Create and configure calibrator
calibrator = ProbabilityCalibrator(seed=42, hidden_dims=[128, 64])

# Add features for calibration
calibrator.add_feature(MassErrorFeature(residue_masses=RESIDUE_MASSES))
calibrator.add_feature(FragmentMatchFeatures(mz_tolerance=0.02))
calibrator.add_feature(BeamFeatures())

# Train directly from a labelled CalibrationDataset
calibrator.fit(train_dataset)

# Make predictions on new data
calibrator.predict(test_dataset)

# Save/load trained models (safetensors + config.json)
ProbabilityCalibrator.save(calibrator, Path("calibrator_checkpoint"))

# Load models - supports multiple sources
# 1. Load default pretrained model from Hugging Face
loaded_calibrator = ProbabilityCalibrator.load()

# 2. Load a custom Hugging Face model
loaded_calibrator = ProbabilityCalibrator.load("my-org/my-custom-model")

# 3. Load from local directory
loaded_calibrator = ProbabilityCalibrator.load("calibrator_checkpoint")
```

**Key Features:**

- **PyTorch Neural Network**: Uses a custom `CalibratorNetwork` (`nn.Module`) with feature normalisation
- **Feature Management**: Add, remove and track multiple calibration features
- **Dependency Handling**: Automatic computation of feature dependencies
- **Model Persistence**: Save/load using `safetensors` (weights) and `config.json` (architecture, normalisation stats, feature definitions)
- **Two-phase Training**: Supports training from pre-computed Parquet feature matrices via `FeatureDataset.from_parquet()` and `fit_from_features()`
- **GPU Support**: Automatic GPU detection with CPU fallback

**Main Methods:**

- `add_feature(feature)`: Add a calibration feature
- `compute_features(dataset)`: Run feature computation on a `CalibrationDataset`, mutating its metadata in place
- `fit(dataset, val_dataset)`: Compute features and train the calibrator from a `CalibrationDataset`
- `fit_from_features(dataset, val_dataset)`: Train from a pre-computed `FeatureDataset` (two-phase workflow)
- `predict(dataset)`: Generate calibrated confidence scores
- `save(calibrator, path)`: Save trained model to disk (`model.safetensors` + `config.json`)
- `load(pretrained_model_name_or_path, cache_dir)`: Load trained model from Hugging Face Hub or local directory

    - Default: Loads `"InstaDeepAI/winnow-general-model"` from Hugging Face
    - Hugging Face: Pass a repository ID string (e.g., `"my-org/my-model"`)
    - Local: Pass a `str` or `Path` object pointing to a model directory
    - Models from Hugging Face are automatically cached in `~/.cache/huggingface/hub`

### CalibrationFeatures

Abstract base class for defining calibration features. All features inherit from this class and implement feature-specific computation logic.

```python
from winnow.calibration.calibration_features import CalibrationFeatures

class CustomFeature(CalibrationFeatures):
    @property
    def name(self) -> str:
        return "My Custom Feature"

    @property
    def columns(self) -> List[str]:
        return ["custom_feature_1", "custom_feature_2"]

    @property
    def dependencies(self) -> List[FeatureDependency]:
        return []  # No dependencies

    def compute(self, dataset: CalibrationDataset) -> None:
        # Implement feature computation
        dataset.metadata["custom_feature_1"] = computed_values
```

**Key Features:**

- **Extensible Interface**: Create custom features by subclassing
- **Dependency Management**: Declare feature dependencies
- **Column Specification**: Define output column names
- **Dataset Integration**: Direct access to CalibrationDataset for computation

## Built-in features

### MassErrorFeature

Calculates the mass error (in Da) between the observed precursor MH⁺ mass and the theoretical MH⁺ mass based on peptide composition.

This feature requires the columns `precursor_mz` and `precursor_charge` to be supplied in the spectrum dataset.

```python
from winnow.calibration.calibration_features import MassErrorFeature
from winnow.constants import RESIDUE_MASSES

feature = MassErrorFeature(residue_masses=RESIDUE_MASSES)
```

**Purpose**: Provides mass accuracy information as a calibration signal.

### FragmentMatchFeatures

Extracts features by calling a Koina intensity model to generate a theoretical fragmentation spectrum for the top-1 de novo predicted sequence and computing how well it matches the observed spectrum (ion match rate and ion match intensity).

This feature requires the column `precursor_charge` to be supplied in the spectrum dataset.

For more information about Koina, please read the [documentation](https://koina.wilhelmlab.org/docs) or the [publication](https://www.nature.com/articles/s41467-025-64870-5).

```python
from winnow.calibration.calibration_features import FragmentMatchFeatures

feature = FragmentMatchFeatures(
    mz_tolerance=0.02,
    unsupported_residues=["N[UNIMOD:7]", "Q[UNIMOD:7]"],  # Residues not supported by the model
    intensity_model_name="Prosit_2025_intensity_22PTM",
    max_precursor_charge=6,   # Upper charge limit of the Koina model
    max_peptide_length=30,    # Upper length limit of the Koina model
    model_input_constants={"collision_energies": 25, "fragmentation_types": "HCD"},
)
```

**Purpose**: Leverages ML-based intensity predictions for spectral quality assessment.

**Note:** Different Koina models support different charge states, peptide lengths and modifications. Consult the documentation for your chosen model and configure `max_precursor_charge`, `max_peptide_length` and `unsupported_residues` accordingly. See the [configuration guide](../configuration.md#koina-model-input-validation) for full details.

### BeamFeatures

Calculates margin, median margin and entropy of beam search runners-up to assess prediction confidence.

```python
from winnow.calibration.calibration_features import BeamFeatures

feature = BeamFeatures()
```

**Purpose**: Uses beam search diversity metrics as confidence indicators.

### ChimericFeatures

Computes chimeric ion matches by predicting intensities for runner-up (second-best) peptide sequences using a Koina intensity model and comparing with observed spectra.

This feature requires the column `precursor_charge` to be supplied in the spectrum dataset.

```python
from winnow.calibration.calibration_features import ChimericFeatures

feature = ChimericFeatures(
    mz_tolerance=0.02,
    unsupported_residues=["N[UNIMOD:7]", "Q[UNIMOD:7]"],
    max_precursor_charge=6,
    max_peptide_length=30,  # Applied to the runner-up sequence
    model_input_constants={"collision_energies": 25, "fragmentation_types": "HCD"},
)
```

**Purpose**: Detects chimeric spectra that may affect confidence estimates.

### RetentionTimeFeature

Trains a per-experiment linear regressor that maps observed retention time (RT) to indexed
retention time (iRT) from a Koina iRT model. The absolute error between the sequence-based
iRT prediction and the regressor-predicted iRT is used as a calibration feature.

This feature requires the column `retention_time` to be supplied in the spectrum dataset.

```python
from winnow.calibration.calibration_features import RetentionTimeFeature

feature = RetentionTimeFeature(
    train_fraction=0.1,
    min_train_points=10,
    unsupported_residues=["N[UNIMOD:7]", "Q[UNIMOD:7]"],
    max_peptide_length=30,
)
```

**Purpose**: Incorporates chromatographic information for confidence calibration.

#### Per-experiment iRT regression

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

#### `experiment_name` column

For multi-experiment data, each spectrum must have an `experiment_name` column:

- **MGF files**: Derived automatically from the file stem (e.g., `data/run1.mgf` produces
  `experiment_name = "run1"`).
- **Parquet / IPC files**: If the column already exists in the file, it is used as-is. If
  not, the file stem is used as the experiment name.

#### Configuration parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `train_fraction` | `0.1` | Top fraction of spectra by confidence (descending) used to train the regressor. Only assumes higher confidence is better. |
| `min_train_points` | `10` | Minimum training points needed per experiment after applying `train_fraction`. Raises a `ValueError` if fewer are available. |
| `seed` | `42` | Random seed for reproducibility. |

#### Regressor checkpoint workflow

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

## Handling missing features

Koina-dependent features (`FragmentMatchFeatures`, `ChimericFeatures`, `RetentionTimeFeature`) may not be computable for all peptides due to model-specific constraints such as:

- Peptides exceeding the model's maximum length
- Precursor charges exceeding the model's maximum
- Unsupported modifications or residue types
- Lack of runner-up sequences for chimeric features

The defaults match the constraints of the Prosit model family. If you use a different Koina model, adjust these parameters accordingly — see the [configuration guide](../configuration.md#koina-model-input-validation) for details.

Winnow provides two strategies for handling such cases:

### Learn strategy (`learn_from_missing=True`)

- Includes `is_missing_*` indicator columns as features
- Calibrator learns patterns associated with missing data
- Uses all available data, maximising recall
- More robust across diverse datasets

### Filter strategy (Default; `learn_from_missing=False`)

**Use when you want strict data quality requirements.**

- Invalid PSMs are automatically filtered from the dataset before Koina is called
- A warning is emitted reporting how many PSMs were removed and which constraints applied
- Filtered PSMs are gone entirely; no indicator column is added
- Calibrator trains only on the remaining clean data

```python
from winnow.calibration.calibration_features import FragmentMatchFeatures, ChimericFeatures, RetentionTimeFeature

# Learn from missingness (default)
fragment_feat = FragmentMatchFeatures(mz_tolerance=0.02, learn_from_missing=True)
chimeric_feat = ChimericFeatures(mz_tolerance=0.02, learn_from_missing=True)
rt_feat = RetentionTimeFeature(train_fraction=0.1, learn_from_missing=True)

# Require clean data (strict mode)
fragment_feat = FragmentMatchFeatures(mz_tolerance=0.02, learn_from_missing=False)
chimeric_feat = ChimericFeatures(mz_tolerance=0.02, learn_from_missing=False)
rt_feat = RetentionTimeFeature(train_fraction=0.1, learn_from_missing=False)
```

## Workflow

### Training workflow

1. **Create Calibrator**: Initialise `ProbabilityCalibrator`
2. **Add Features**: Use `add_feature()` to include desired calibration features
3. **Fit Model**: Call `fit()` with a labelled `CalibrationDataset` — feature computation and training happen in one step
4. **Save Model**: Use `save()` to persist trained calibrator

For the two-phase workflow (compute features once, save to Parquet, train later):

1. Call `compute_features(dataset)` to populate metadata columns
2. Export to Parquet via `dataset.to_parquet()`
3. Reload with `FeatureDataset.from_parquet()`
4. Train with `fit_from_features(train_ds, val_dataset=val_ds)`

### Prediction workflow

1. **Load Calibrator**: Use `load()` to restore trained model from a Hugging Face repository or a local directory
   ```python
   # Option 1: Use default pretrained model
   calibrator = ProbabilityCalibrator.load()

   # Option 2: Use custom Hugging Face model
   calibrator = ProbabilityCalibrator.load("my-org/my-custom-model")

   # Option 3: Use local model
   calibrator = ProbabilityCalibrator.load("./my_calibrator")
   ```
2. **Predict**: Call `predict()` with unlabelled `CalibrationDataset`
3. **Access Results**: Calibrated scores stored in dataset's "calibrated_confidence" column

## Feature dependencies

The system automatically handles feature dependencies:

- **FeatureDependency**: Base class for shared computations
- **Reference Counting**: Tracks dependency usage across features
- **Automatic Computation**: Dependencies computed before features
- **Memory Efficiency**: Shared dependencies computed once

For detailed examples and usage patterns, refer to the [examples notebook](https://github.com/instadeepai/winnow/blob/main/examples/getting_started_with_winnow.ipynb).
