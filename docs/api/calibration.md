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
calibrator.add_feature(FragmentMatchFeatures(mz_tolerance_ppm=20))
calibrator.add_feature(BeamFeatures())

# Train the calibrator
calibrator.fit(training_dataset)

# Make predictions
calibrator.predict(test_dataset)

# Save/load trained models
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
- `load(pretrained_model_name_or_path, cache_dir)`: Load trained model from Hugging Face Hub or local directory

    - Default: Loads `"InstaDeepAI/winnow-general-model"` from Hugging Face
    - Hugging Face: Pass a repository ID string (e.g., `"my-org/my-model"`)
    - Local: Pass a `str` or `Path` object pointing to a model directory
    - Models from Hugging Face are automatically cached in `~/.cache/huggingface/hub`

## Calibration Features

The calibrator uses a feature-based approach where multiple feature extractors compute signals from the peptide-spectrum match data. See the [Calibration Features](features/index.md) documentation for:

- The `CalibrationFeatures` base class for creating custom features
- Built-in features: [MassErrorFeature](features/mass_error.md), [BeamFeatures](features/beam.md), [FragmentMatchFeatures](features/fragment_match.md), [ChimericFeatures](features/chimeric.md), [RetentionTimeFeature](features/retention_time.md), [SequenceFeatures](features/sequence.md), [TokenScoreFeatures](features/token_score.md)
- Feature dependencies and how they work
- Handling missing features (learn vs filter strategies)

## Workflow

### Training workflow

1. **Create Calibrator**: Initialise `ProbabilityCalibrator`
2. **Add Features**: Use `add_feature()` to include desired calibration features
3. **Fit Model**: Call `fit()` with labelled `CalibrationDataset`
4. **Save Model**: Use `save()` to persist trained calibrator

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

For detailed examples and usage patterns, refer to the [examples notebook](https://github.com/instadeepai/winnow/blob/main/examples/getting_started_with_winnow.ipynb).
