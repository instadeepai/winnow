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

# Train the calibrator and get training history
training_history = calibrator.fit(training_dataset)
print(f"Final training loss: {training_history.final_training_loss:.6f}")
if training_history.final_validation_score is not None:
    print(f"Final validation score: {training_history.final_validation_score:.6f}")

# Plot training progress
training_history.plot(output_path="training_progress.png")

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
- `fit(dataset)`: Train the calibrator on a labelled dataset. Returns a `TrainingHistory` object containing training metrics.
- `predict(dataset)`: Generate calibrated confidence scores
- `save(calibrator, path)`: Save trained model to disk
- `load(pretrained_model_name_or_path, cache_dir)`: Load trained model from Hugging Face Hub or local directory

  - Default: Loads `"InstaDeepAI/winnow-general-model"` from Hugging Face
  - Hugging Face: Pass a repository ID string (e.g., `"my-org/my-model"`)
  - Local: Pass a `str` or `Path` object pointing to a model directory
  - Models from Hugging Face are automatically cached in `~/.cache/huggingface/hub`

### TrainingHistory

A dataclass containing training metrics from calibrator fitting. The training history is automatically saved to a JSON file during training (configurable via `training_history_path`).

```python
from winnow.calibration import TrainingHistory

# Returned from calibrator.fit()
training_history = calibrator.fit(training_dataset)

# Access training metrics
print(training_history.loss_curve)            # List of training loss at each iteration
print(training_history.validation_scores)      # List of validation scores (if early_stopping=True)
print(training_history.final_training_loss)    # Final training loss value
print(training_history.final_validation_score) # Final validation score (if early_stopping=True)
print(training_history.n_iter)                 # Number of training iterations

# Save training history to JSON
training_history.save("training_history.json")

# Load training history from JSON (e.g., for later analysis or plotting)
loaded_history = TrainingHistory.load("training_history.json")

# Plot training progress
loaded_history.plot(output_path="training_plot.png")
```

**Attributes:**

- `loss_curve`: List of training loss values at each iteration
- `validation_scores`: List of validation classification accuracy scores at each iteration (only if `early_stopping=True`)
- `final_training_loss`: The final training loss value
- `final_validation_score`: The final validation score (only if `early_stopping=True`)
- `n_iter`: Number of iterations the solver ran

**Methods:**

- `save(path)`: Save training history to a JSON file
- `load(path)`: Load training history from a JSON file (class method)
- `plot(output_path, show)`: Plot training and validation loss curves

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
