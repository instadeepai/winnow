# Calibration API

The `winnow.calibration` module implements confidence calibration for peptide-spectrum matches using machine learning-based feature extraction and neural network classification.

## Classes

### ProbabilityCalibrator

The main calibration model that transforms raw confidence scores into calibrated probabilities using a multi-layer perceptron classifier with various peptide and spectral features.

```python
from winnow.calibration import ProbabilityCalibrator
from winnow.calibration.calibration_features import (
    MassErrorFeature, PrositFeatures, BeamFeatures
)
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.constants import RESIDUE_MASSES

# Create and configure calibrator
calibrator = ProbabilityCalibrator(seed=42)

# Add features for calibration
calibrator.add_feature(MassErrorFeature(residue_masses=RESIDUE_MASSES))
calibrator.add_feature(PrositFeatures(mz_tolerance=0.02))
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

## Built-in Features

### MassErrorFeature

Calculates the difference between observed precursor mass and theoretical mass based on peptide composition.

```python
from winnow.calibration.calibration_features import MassErrorFeature
from winnow.constants import RESIDUE_MASSES

feature = MassErrorFeature(residue_masses=RESIDUE_MASSES)
```

**Purpose**: Provides mass accuracy information as a calibration signal.

### PrositFeatures

Extracts features using Prosit intensity prediction models to compare predicted vs observed fragment ion intensities.

```python
from winnow.calibration.calibration_features import PrositFeatures

feature = PrositFeatures(mz_tolerance=0.02)
```

**Purpose**: Leverages ML-based intensity predictions for spectral quality assessment.

### BeamFeatures

Calculates margin, median margin and entropy of beam search runners-up to assess prediction confidence.

```python
from winnow.calibration.calibration_features import BeamFeatures

feature = BeamFeatures()
```

**Purpose**: Uses beam search diversity metrics as confidence indicators.

### ChimericFeatures

Computes chimeric ion matches by predicting intensities for runner-up peptide sequences and comparing with observed spectra.

```python
from winnow.calibration.calibration_features import ChimericFeatures

feature = ChimericFeatures(mz_tolerance=0.02)
```

**Purpose**: Detects chimeric spectra that may affect confidence estimates.

### RetentionTimeFeature

Uses Prosit iRT models to predict indexed retention times and calibrate against observed retention times.

```python
from winnow.calibration.calibration_features import RetentionTimeFeature

feature = RetentionTimeFeature(hidden_dim=10, train_fraction=0.1)
```

**Purpose**: Incorporates chromatographic information for confidence calibration.

## Workflow

### Training Workflow

1. **Create Calibrator**: Initialise `ProbabilityCalibrator`
2. **Add Features**: Use `add_feature()` to include desired calibration features
3. **Fit Model**: Call `fit()` with labelled `CalibrationDataset`
4. **Save Model**: Use `save()` to persist trained calibrator

### Prediction Workflow

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

## Feature Dependencies

The system automatically handles feature dependencies:

- **FeatureDependency**: Base class for shared computations
- **Reference Counting**: Tracks dependency usage across features
- **Automatic Computation**: Dependencies computed before features
- **Memory Efficiency**: Shared dependencies computed once

For detailed examples and usage patterns, refer to the [examples notebook](https://github.com/instadeepai/winnow/blob/main/examples/getting_started_with_winnow.ipynb).
