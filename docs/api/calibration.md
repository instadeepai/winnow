# Calibration API

The `winnow.calibration` module implements confidence calibration for peptide-spectrum matches using machine learning-based feature extraction and neural network classification.

## Classes

### ProbabilityCalibrator

The main calibration model that transforms raw confidence scores into calibrated probabilities using a multi-layer perceptron classifier with various peptide and spectral features.

```python
from winnow.calibration import ProbabilityCalibrator
from winnow.calibration.calibration_features import (
    MassErrorFeature, FragmentMatchFeatures, BeamFeatures
)
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.constants import RESIDUE_MASSES

# Create and configure calibrator
calibrator = ProbabilityCalibrator(seed=42)

# Add features for calibration
calibrator.add_feature(MassErrorFeature(residue_masses=RESIDUE_MASSES))
calibrator.add_feature(FragmentMatchFeatures(mz_tolerance=0.02))
calibrator.add_feature(BeamFeatures())

# Train the calibrator
calibrator.fit(training_dataset)

# Make predictions
calibrator.predict(test_dataset)

# Save/load trained models
ProbabilityCalibrator.save(calibrator, Path("calibrator_checkpoint"))

# Load models - supports multiple sources
# 1. Load default pretrained model from HuggingFace
loaded_calibrator = ProbabilityCalibrator.load()

# 2. Load a custom HuggingFace model
loaded_calibrator = ProbabilityCalibrator.load("my-org/my-custom-model")

# 3. Load from local directory
loaded_calibrator = ProbabilityCalibrator.load("calibrator_checkpoint")
```

**Key Features:**

- **Neural Network Classifier**: Uses MLPClassifier with standardised feature scaling
- **Feature Management**: Add, remove and track multiple calibration features
- **Dependency Handling**: Automatic computation of feature dependencies
- **Model Persistence**: Save and load trained calibrators
- **Feature Extraction**: Computes features and handles both labelled and unlabelled data

**Main Methods:**

- `add_feature(feature)`: Add a calibration feature
- `fit(dataset)`: Train the calibrator on a labelled dataset
- `predict(dataset)`: Generate calibrated confidence scores
- `save(calibrator, path)`: Save trained model to disk
- `load(pretrained_model_name_or_path, cache_dir)`: Load trained model from HuggingFace Hub or local directory
  - Default: Loads `"InstaDeepAI/winnow-general-model"` from HuggingFace
  - HuggingFace: Pass a repository ID string (e.g., `"my-org/my-model"`)
  - Local: Pass a `str` or `Path` object pointing to a model directory
  - Models from HuggingFace are automatically cached in `~/.cache/huggingface/hub`

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

**Note:** Different Koina models support different charge states, peptide lengths and modifications. Consult the documentation for your chosen model at [koina.wilhelmlab.org](https://koina.wilhelmlab.org/) and configure `max_precursor_charge`, `max_peptide_length` and `unsupported_residues` accordingly. See the [configuration guide](../configuration.md#koina-model-input-validation) for full details.

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

Uses a Koina iRT model to predict indexed retention times and calibrate against observed retention times.

This feature requires the column `retention_time` to be supplied in the spectrum dataset.

```python
from winnow.calibration.calibration_features import RetentionTimeFeature

feature = RetentionTimeFeature(
    hidden_dim=10,
    train_fraction=0.1,
    unsupported_residues=["N[UNIMOD:7]", "Q[UNIMOD:7]"],
    max_peptide_length=30,
)
```

**Purpose**: Incorporates chromatographic information for confidence calibration.

## Handling missing features

Koina-dependent features (`FragmentMatchFeatures`, `ChimericFeatures`, `RetentionTimeFeature`) may not be computable for all peptides due to model-specific constraints such as:

- Peptides exceeding the model's maximum length
- Precursor charges exceeding the model's maximum
- Unsupported modifications or residue types
- Lack of runner-up sequences for chimeric features

The defaults match the constraints of the Prosit model family. If you use a different Koina model, adjust these parameters accordingly — see the [configuration guide](../configuration.md#koina-model-input-validation) for details.

Winnow provides two strategies for handling such cases:

### Learn strategy (Default, `learn_from_missing=True`)

**Recommended for most use cases.**

- Includes `is_missing_*` indicator columns as features
- Calibrator learns patterns associated with missing data
- Uses all available data, maximising recall
- More robust across diverse datasets

### Filter strategy (`learn_from_missing=False`)

**Use when you want strict data quality requirements.**

- Invalid PSMs are automatically filtered from the dataset before Koina is called
- A warning is emitted reporting how many PSMs were removed and which constraints applied
- Filtered PSMs are gone entirely; no indicator column is added
- Calibrator trains only on the remaining clean data

### Configuration

Configure via config overrides during training:

```bash
# Default: Learn from missingness
winnow train

# Strict: Require clean data
winnow train \
    calibrator.features.fragment_match_features.learn_from_missing=false \
    calibrator.features.chimeric_features.learn_from_missing=false \
    calibrator.features.retention_time_feature.learn_from_missing=false
```

Or configure programmatically:

```python
from winnow.calibration.calibration_features import FragmentMatchFeatures, ChimericFeatures, RetentionTimeFeature

# Learn from missingness (default)
fragment_feat = FragmentMatchFeatures(mz_tolerance=0.02, learn_from_missing=True)
chimeric_feat = ChimericFeatures(mz_tolerance=0.02, learn_from_missing=True)
rt_feat = RetentionTimeFeature(hidden_dim=10, train_fraction=0.1, learn_from_missing=True)

# Require clean data (strict mode)
fragment_feat = FragmentMatchFeatures(mz_tolerance=0.02, learn_from_missing=False)
chimeric_feat = ChimericFeatures(mz_tolerance=0.02, learn_from_missing=False)
rt_feat = RetentionTimeFeature(hidden_dim=10, train_fraction=0.1, learn_from_missing=False)
```

## Workflow

### Training workflow

1. **Create Calibrator**: Initialise `ProbabilityCalibrator`
2. **Add Features**: Use `add_feature()` to include desired calibration features
3. **Fit Model**: Call `fit()` with labelled `CalibrationDataset`
4. **Save Model**: Use `save()` to persist trained calibrator

### Prediction workflow

1. **Load Calibrator**: Use `load()` to restore trained model from a HuggingFace repository or a local directory
   ```python
   # Option 1: Use default pretrained model
   calibrator = ProbabilityCalibrator.load()

   # Option 2: Use custom HuggingFace model
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
