# Command line interface

This guide provides practical examples and workflows for using the `winnow` command-line interface.

**Looking for configuration details?** See the **[Configuration guide](configuration.md)** for comprehensive documentation of the configuration system, YAML structure and advanced patterns.

## Installation

After installing winnow, the `winnow` command becomes available:

```bash
pip install winnow-fdr
# or
uv pip install winnow-fdr
```

## Quick Start

Get started immediately with the included sample data:

```bash
# Generate sample data (if not already present)
make sample-data

# Train a calibrator on the sample data
make train-sample

# Run prediction with the trained model
make predict-sample
```

**Note:** The sample data is minimal (100 spectra) and intended for testing only. When using the sample data, it's **recommended to use the `make` commands** (e.g., `make predict-sample`) as they include necessary configuration adjustments. Specifically, `make predict-sample` sets `fdr_control.fdr_threshold=1.0` because the sample data contains artificial PSMs with relatively high error rates, and using the default threshold (0.05) would filter out all predictions, resulting in empty output. In addition, we increase the validation fraction for the retention time feature in the calibrator's configuration, because with such a small training dataset, a higher validation fraction ensures that the validation set will contain enough samples for stable training and early stopping. For use with real datasets, use the standard FDR threshold (default 0.05) and default validation fractions, or adjust as appropriate for your application.

## Commands

### `winnow config`

Display the resolved configuration for any command without executing it. This is useful for inspecting how defaults and overrides are composed.

```bash
# Show training configuration
winnow config train

# Show prediction configuration
winnow config predict

# Show compute-features configuration
winnow config compute-features

# Check configuration with overrides
winnow config train data_loader=mztab model_output_dir=models/my_model
winnow config predict fdr_method=database_grounded fdr_control.fdr_threshold=0.01
```

This command prints the final YAML configuration with colour-coded hierarchical formatting, making it easy to read and verify your settings. Keys are coloured by nesting depth to help visualise the configuration structure. The output shows all defaults, composition and overrides after they have been applied.

**Note:** Some keys appear with quotes (e.g. `'N'`, `'Y'`) because they are reserved words in YAML that would otherwise be interpreted as boolean values. The quotes ensure they are treated as strings.

### `winnow train`

Train a confidence calibration model on labelled data.

```bash
# Use defaults (configured in configs/train.yaml)
winnow train

# Override specific parameters
winnow train data_loader=mztab model_output_dir=models/my_model

# Specify dataset paths
winnow train dataset.spectrum_path_or_directory=data/spectra.parquet dataset.predictions_path=data/preds.csv
```

**Common Parameters:**

- `data_loader`: Type of dataset loader (`instanovo`, `mztab`, `pointnovo`, `winnow`)
- `dataset.spectrum_path_or_directory`: Path to spectrum/metadata file (or directory for Winnow format)
- `dataset.predictions_path`: Path to predictions file (set to `null` for Winnow format)
- `model_output_dir`: Directory to save trained calibrator
- `dataset_output_path`: Path to save training results CSV
- `features_path`: Optional path to pre-computed feature Parquet file or directory (enables two-phase training)
- `val_features_path`: Optional path to validation feature Parquet file or directory
- `validation_fraction`: Automatic validation split fraction (default: 0.1, used when `val_features_path` is null)
- `training_history_path`: Optional path to save epoch-level training history as JSON

**Advanced calibrator configuration:**

You can customise the calibrator architecture and features using nested parameters:

```bash
# Change network architecture
winnow train calibrator.hidden_dims=[128,64,32]

# Adjust training hyperparameters
winnow train calibrator.learning_rate=0.01 calibrator.max_epochs=200 calibrator.patience=15

winnow train calibrator.features.fragment_match_features.mz_tolerance_da=0.01
```

For comprehensive calibrator configuration options, see:

- [Configuration guide](configuration.md) - Complete parameter reference
- [Calibration API](api/features/index.md#handling-missing-features) - Feature implementation details

### `winnow compute-features`

Compute calibration features and write an enriched metadata CSV and optionally a lean numeric Parquet for model training. Uses the same `data_loader`, `dataset` paths and `calibrator.features` stack as training, but does **not** fit the calibrator or save a model.

```bash
# Defaults (configs/compute_features.yaml)
winnow compute-features

# Paths and output files
winnow compute-features \
    dataset.spectrum_path_or_directory=data/spectra.ipc \
    dataset.predictions_path=data/preds.csv \
    metadata_output_path=results/metadata.csv \
    training_matrix_output_path=results/features.parquet

# De novo spectra (no ground truth): labelled=false; remove retention_time_feature if present
winnow compute-features labelled=false '~calibrator.features.retention_time_feature'
```

**Common parameters:**

- `data_loader`, `dataset.*`: Same as `winnow train`
- `metadata_output_path`: Full metadata CSV for EDA (all columns)
- `training_matrix_output_path`: Optional lean numeric Parquet containing only features and labels for model training (used with two-phase `features_path` workflow)
- `filter_empty_predictions`: Drop empty or invalid predictions (default: true)
- `labelled`: If true (default), spectrum data must include a `sequence` column; runs each feature's `prepare()` (e.g. iRT calibrator).

Feature selection matches training: override `calibrator.features` or use `~calibrator.features.<name>` to drop entries. See [Configuration guide](configuration.md#compute-features-configuration).

### `winnow predict`

Apply calibration and FDR control to new data using a trained model. By default, uses a pretrained general model from Hugging Face Hub.

```bash
# Use defaults (pretrained model from Hugging Face)
winnow predict

# Override specific parameters
winnow predict data_loader=mztab fdr_control.fdr_threshold=0.01 fdr_method=database_grounded

# Specify dataset paths
winnow predict dataset.spectrum_path_or_directory=data/spectra.parquet dataset.predictions_path=data/preds.csv

# Use a custom Hugging Face model
winnow predict calibrator.pretrained_model_name_or_path=my-org/my-custom-model

# Use a local model
winnow predict calibrator.pretrained_model_name_or_path=models/my_model
```

**Common Parameters:**

- `data_loader`: Type of dataset loader (`instanovo`, `mztab`, `pointnovo`, `winnow`)
- `dataset.spectrum_path_or_directory`: Path to spectrum/metadata file (or directory for Winnow format)
- `dataset.predictions_path`: Path to predictions file
- `fdr_method`: FDR estimation method (`nonparametric` or `database_grounded`)
- `fdr_control.fdr_threshold`: Target FDR threshold (e.g. 0.01 for 1%)
- `output_folder`: Folder path to write output files

By default, `winnow predict` uses the pretrained model `InstaDeepAI/winnow-general-model` from Hugging Face Hub. To use a different model, override the calibrator settings (see [Configuration guide](configuration.md#prediction-configuration) for details).

## Configuration system

Winnow uses [Hydra](https://hydra.cc/) for configuration management. All parameters can be configured via:

- **YAML config files** in the `configs/` directory (defines defaults)
- **Command-line overrides** using `key=value` syntax
- **Nested parameters** using dot notation (e.g., `calibrator.seed=42`)

For comprehensive configuration documentation, including:

- Full configuration file structure and composition
- Config interpolation and variable references
- Creating custom configurations
- Advanced patterns and debugging

See the **[Configuration guide](configuration.md)**.

## Data requirements

### Training data

For training (`winnow train`), you need:

- **Labelled dataset**: Ground truth peptide sequences for evaluation
- **Predictions**: Model predictions with confidence scores
- **Spectral data**: MS/MS spectra and metadata
- **Unique identifiers**: Each PSM must have a unique `spectrum_id` in both input files

### Feature export (`winnow compute-features`)

Same inputs as training for loading spectra and predictions. With default `labelled=true`, the spectrum file must include a `sequence` column (as for training). With `labelled=false`, pure *de novo* inputs are allowed if the configured feature set does not include `retention_time_feature`, which requires labelled data to fit an iRT predictor model.

### Prediction data

For prediction (`winnow predict`), you need:

- **Unlabelled dataset**: Predictions and spectra (no ground truth required for non-parametric FDR)
- **Trained model**: Pretrained model from Hugging Face or output from `winnow train`
- **Confidence scores**: Raw confidence values to calibrate
- **Unique identifiers**: Each PSM must have a unique `spectrum_id` in both input files

### Data formats

Winnow supports multiple input formats:

- **InstaNovo**: Parquet, IPC, or MGF spectra + CSV predictions (beam search format)
- **MZTab**: Parquet or IPC spectra + MZTab predictions
- **PointNovo**: Similar to InstaNovo format
- **Winnow**: Internal format (directory with metadata.csv and predictions.pkl)

Specify the format using `data_source=<format>` parameter.

## FDR methods

### Non-parametric method (`fdr_method=nonparametric`)

Uses non-parametric FDR estimation procedure:

- **No ground truth required**: Works with confidence scores alone
- **No correct/incorrect distribution modelling**: Process directly estimates FDR using calibrated confidence scores
- **Multiple metrics**: Provides FDR, PEP and q-values

```bash
winnow predict fdr_method=nonparametric fdr_control.fdr_threshold=0.01
```

### Database-grounded method (`fdr_method=database_grounded`)

Uses database search results for validation:

- **Requires ground truth**: Needs labelled data with sequence column
- **Precision-recall based**: Computes FDR from validation results
- **Direct estimates**: FDR calculated from actual correct/incorrect labels

```bash
winnow predict fdr_method=database_grounded fdr_control.fdr_threshold=0.05
```

## Output files

### Training output

Training produces:

1. **Model directory** (`model_output_dir`):
    - `model.safetensors`: Trained network weights in safetensors format
    - `config.json`: Model architecture, feature definitions, normalisation statistics and training history

2. **Training results** (`dataset_output_path`):
    - CSV file with calibrated scores and evaluation metrics

3. **Optional** (`training_history_path`):
    - JSON file with epoch-level training and validation metrics

### Prediction output

Prediction produces two CSV files in the `output-folder` directory:

1. **`metadata.csv`**: Contains all metadata and feature columns from the input dataset

    - Original metadata columns (spectrum information, precursors, etc.)
    - All feature columns used for calibration
    - Filtered to only include PSMs passing the FDR threshold

2. **`preds_and_fdr_metrics.csv`**: Contains predictions and error metrics

    - `spectrum_id`: Unique spectrum identifier
    - `calibrated_confidence` (or specified confidence column): Calibrated confidence scores
    - `prediction`: Predicted peptide sequences
    - `psm_fdr`: PSM-specific FDR estimates
    - `psm_q_value`: Q-values
    - `psm_pep`: Posterior error probabilities (non-parametric method only)
    - `sequence`: Ground truth sequences (if available, for database-ground method)
    - `num_matches`: Number of matched residues between the predicted and ground-truth peptide (if labelled)
    - `correct`: Whether the predicted sequence is correct (if labelled)
    - Filtered to only include PSMs passing the FDR threshold

This separation allows users to work with metadata and features separately from predictions and error metrics, making downstream analysis more convenient.

### Compute-features output

- **Metadata CSV** at `metadata_output_path`: Full metadata after feature computation, suitable for EDA. No calibrated scores, FDR columns or FDR row filtering (unlike `winnow predict`).
- **Training matrix Parquet** at `training_matrix_output_path` (optional): Lean numeric matrix containing only feature columns and labels, suitable for two-phase training via `features_path`.

## Example workflows

### Quick start with defaults

```bash
# Predict using pretrained model, InstaNovo predictions and default settings
winnow predict \
    dataset.spectrum_path_or_directory=data/test_spectra.parquet \
    dataset.predictions_path=data/test_predictions.csv
```

### Complete training and prediction pipeline

```bash
# Step 1: Train calibrator on labelled data
winnow train \
    data_loader=instanovo \
    dataset.spectrum_path_or_directory=data/train_spectra.parquet \
    dataset.predictions_path=data/train_predictions.csv \
    model_output_dir=models/my_calibrator \
    dataset_output_path=results/training_output.csv

# Step 2: Apply to new data with FDR control
winnow predict \
    data_loader=instanovo \
    dataset.spectrum_path_or_directory=data/test_spectra.parquet \
    dataset.predictions_path=data/test_predictions.csv \
    calibrator.pretrained_model_name_or_path=models/my_calibrator \
    fdr_method=nonparametric \
    fdr_control.fdr_threshold=0.01 \
    output_folder=results/predictions
```

### MZTab format

```bash
# Train with MZTab format
winnow train \
    data_loader=mztab \
    dataset.spectrum_path_or_directory=data/spectra.parquet \
    dataset.predictions_path=data/casanovo_results.mztab \
    model_output_dir=models/mztab_model

# Predict with MZTab format
winnow predict \
    data_loader=mztab \
    dataset.spectrum_path_or_directory=data/test_spectra.parquet \
    dataset.predictions_path=data/test_results.mztab \
    calibrator.pretrained_model_name_or_path=models/mztab_model \
    fdr_control.fdr_threshold=0.05
```

### Two-phase training (large-scale)

For large datasets (e.g. millions of spectra across many projects), pre-compute features per project then train from the Parquet files:

```bash
# Phase 1: Compute features per project
winnow compute-features \
    dataset.spectrum_path_or_directory=data/project_01/spectra.parquet \
    dataset.predictions_path=data/project_01/predictions.csv \
    training_matrix_output_path=features/project_01.parquet

winnow compute-features \
    dataset.spectrum_path_or_directory=data/project_02/spectra.parquet \
    dataset.predictions_path=data/project_02/predictions.csv \
    training_matrix_output_path=features/project_02.parquet

# Phase 2: Train from pre-computed features (directory of Parquets)
winnow train features_path=features/train/ val_features_path=features/val/
```

### Advanced configuration

```bash
# Train with custom calibrator settings
winnow train \
    dataset.spectrum_path_or_directory=data/spectra.parquet \
    dataset.predictions_path=data/predictions.csv \
    calibrator.hidden_dims=[128,64,32] \
    calibrator.learning_rate=0.01 \
    calibrator.max_epochs=200 \
    calibrator.features.fragment_match_features.mz_tolerance_da=0.01

# Predict with database-grounded FDR
winnow predict \
    dataset.spectrum_path_or_directory=data/test_spectra.parquet \
    dataset.predictions_path=data/test_predictions.csv \
    fdr_method=database_grounded \
    fdr_control.fdr_threshold=0.01 \
    calibrator.pretrained_model_name_or_path=models/custom_model
```

## Default configuration

Winnow comes with sensible default settings for all parameters:

- **Calibrator**: PyTorch feed-forward neural network
- **Features**: Mass error, fragment match features, retention time deviation, chimeric features, beam features
- **FDR**: Non-parametric method with 5% threshold
- **Model**: Pretrained general model from Hugging Face

All defaults are defined in YAML files under `configs/` and can be overridden via command line. For a complete reference of all default parameters and configuration options, see the **[Configuration guide](configuration.md)**.

### Getting help

View available options:

```bash
winnow --help           # List all commands
winnow train --help     # Command-specific help
winnow compute-features --help
winnow predict --help
winnow config --help    # Config command help

winnow config train     # View resolved training configuration
winnow config compute-features  # View resolved feature computation configuration
winnow config predict   # View resolved prediction configuration
```

## Where to find information

This CLI guide focuses on **practical command-line usage**. For other information, see:

| Topic | Documentation |
| ------- | --------------- |
| Configuration system, YAML structure, advanced patterns | [Configuration guide](configuration.md) |
| Python API, feature implementation, programmatic usage | [API reference](api/calibration.md) |
| Interactive tutorials and examples | [Examples notebook](https://github.com/instadeepai/winnow/blob/main/examples/getting_started_with_winnow.ipynb) |
| Contributing, development setup | [Contributing guide](contributing.md) |
