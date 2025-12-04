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

## Commands

### `winnow config`

Display the resolved configuration for any command without executing it. This is useful for inspecting how defaults and overrides are composed.

```bash
# Show training configuration
winnow config train

# Show prediction configuration
winnow config predict

# Check configuration with overrides
winnow config train data_loader=mztab model_output_dir=models/my_model
winnow config predict fdr_method=database_grounded fdr_control.fdr_threshold=0.01
```

This command prints the final YAML configuration with colour-coded hierarchical formatting, making it easy to read and verify your settings. Keys are coloured by nesting depth to help visualise the configuration structure. The output shows all defaults, composition and overrides after they have been applied.

**Note:** Some keys appear with quotes (e.g. `'N'`, `'Y'`) because they are reserved words in YAML that would otherwise be interpreted as boolean values. The quotes ensure they are treated as strings.

### `winnow train`

Train a confidence calibration model on labelled data.

```bash
# Use defaults (configured in config/train.yaml)
winnow train

# Override specific parameters
winnow train data_loader=mztab model_output_dir=models/my_model

# Specify dataset paths
winnow train dataset.spectrum_path_or_directory=data/spectra.parquet dataset.predictions_path=data/preds.csv
```

**Common Parameters:**

- `data_loader`: Type of dataset loader (`instanovo`, `mztab`, `pointnovo`, `winnow`)
- `dataset.spectrum_path_or_directory`: Path to spectrum/metadata file (or directory for winnow format)
- `dataset.predictions_path`: Path to predictions file (set to `null` for winnow format)
- `model_output_dir`: Directory to save trained calibrator
- `dataset_output_path`: Path to save training results CSV

**Advanced calibrator configuration:**

You can customise the calibrator architecture and features using nested parameters:

```bash
# Change MLP architecture
winnow train calibrator.hidden_layer_sizes=[100,50,25]

# Configure individual features
winnow train calibrator.features.prosit_features.mz_tolerance=0.01
```

For comprehensive calibrator configuration options, see:
- [Configuration guide](configuration.md) - Complete parameter reference
- [Calibration API](api/calibration.md#handling-missing-features) - Feature implementation details

### `winnow predict`

Apply calibration and FDR control to new data using a trained model. By default, uses a pretrained general model from HuggingFace Hub.

```bash
# Use defaults (pretrained model from HuggingFace)
winnow predict

# Override specific parameters
winnow predict data_loader=mztab fdr_control.fdr_threshold=0.01 fdr_method=database_grounded

# Specify dataset paths
winnow predict dataset.spectrum_path_or_directory=data/spectra.parquet dataset.predictions_path=data/preds.csv

# Use a custom HuggingFace model
winnow predict calibrator.pretrained_model_name_or_path=my-org/my-custom-model

# Use a local model
winnow predict calibrator.pretrained_model_name_or_path=models/my_model
```

**Common Parameters:**

- `data_loader`: Type of dataset loader (`instanovo`, `mztab`, `pointnovo`, `winnow`)
- `dataset.spectrum_path_or_directory`: Path to spectrum/metadata file (or directory for winnow format)
- `dataset.predictions_path`: Path to predictions file
- `fdr_method`: FDR estimation method (`nonparametric` or `database_grounded`)
- `fdr_control.fdr_threshold`: Target FDR threshold (e.g. 0.01 for 1%)
- `output_folder`: Folder path to write output files

By default, `winnow predict` uses the pretrained model `InstaDeepAI/winnow-general-model` from HuggingFace Hub. To use a different model, override the calibrator settings (see [Configuration guide](configuration.md#using-a-custom-model) for details).

## Configuration system

Winnow uses [Hydra](https://hydra.cc/) for configuration management. All parameters can be configured via:

- **YAML config files** in the `config/` directory (defines defaults)
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

### Prediction data

For prediction (`winnow predict`), you need:

- **Unlabelled dataset**: Predictions and spectra (no ground truth required for non-parametric FDR)
- **Trained model**: Pretrained model from HuggingFace or output from `winnow train`
- **Confidence scores**: Raw confidence values to calibrate
- **Unique identifiers**: Each PSM must have a unique `spectrum_id` in both input files

### Data formats

Winnow supports multiple input formats:

- **InstaNovo**: Parquet or IPC spectra + CSV predictions (beam search format)
- **MZTab**: Parquet or IPC spectra + MZTab predictions
- **PointNovo**: Similar to InstaNovo format
- **Winnow**: Internal format (directory with metadata.csv and predictions.pkl)

Specify the format using `data_source=<format>` parameter.

**Note on MGF files**: While many users have their input data in `.mgf` format, Winnow currently requires spectrum data to be in `.parquet` or `.ipc` format. To convert `.mgf` files to `.parquet`, you can use InstaNovo's conversion utilities. See the [InstaNovo documentation](https://instadeepai.github.io/InstaNovo/) for instructions on using `instanovo convert` or the `SpectrumDataFrame` class to perform this conversion.

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

1. **Model checkpoints** (`model_output_dir`):
   - `calibrator.pkl`: Complete trained calibrator with all features and parameters

2. **Training results** (`dataset_output_path`):
   - CSV with calibrated scores and evaluation metrics

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
   - `psm_pep`: Posterior error probabilities (winnow method only)
   - `sequence`: Ground truth sequences (if available, for database-ground method)
   - Filtered to only include PSMs passing the FDR threshold

This separation allows users to work with metadata and features separately from predictions and error metrics, making downstream analysis more convenient.

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

### Advanced configuration

```bash
# Train with custom calibrator settings
winnow train \
    dataset.spectrum_path_or_directory=data/spectra.parquet \
    dataset.predictions_path=data/predictions.csv \
    calibrator.hidden_layer_sizes=[100,50,25] \
    calibrator.learning_rate_init=0.01 \
    calibrator.max_iter=500 \
    calibrator.features.prosit_features.mz_tolerance=0.01

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

- **Calibrator**: 2-layer MLP with 50 hidden units per layer
- **Features**: Mass error, Prosit features, retention time, chimeric features, beam features
- **FDR**: Non-parametric method with 5% threshold
- **Model**: Pretrained general model from HuggingFace

All defaults are defined in YAML files under `config/` and can be overridden via command line. For a complete reference of all default parameters and configuration options, see the **[Configuration guide](configuration.md)**.

## Troubleshooting

### Common issues

**Missing columns**: Ensure your data files contain expected columns:

- `preds`: Main prediction
- `confidence`: Confidence scores
- `sequence`: Ground truth (for training/database-grounded FDR)

**File paths**: Use absolute paths in dataset path overrides to avoid path resolution issues:
```bash
winnow predict dataset.spectrum_path_or_directory=/absolute/path/to/spectra.parquet
```

**Configuration errors**: If Hydra reports a missing config file:
```bash
winnow predict fdr_method=typo
# Error: Could not find 'fdr/typo'
# Available options in 'fdr': nonparametric, database_grounded
```

**Memory issues**: Large datasets may require more memory. Consider:
- Filtering data before processing
- Using a machine with more RAM
- Processing in batches (requires custom Python script)

### Dataset filtering

The CLI automatically filters out:

- Empty predictions
- Peptides longer than 30 amino acids (Prosit limitation)
- Precursor charges above 6 (Prosit limitation)
- Invalid modifications and tokens (defined in `config/residues.yaml`)

### Getting help

View available options:

```bash
winnow --help           # List all commands
winnow train --help     # Command-specific help
winnow predict --help
winnow config --help    # Config command help

winnow config train     # View resolved training configuration
winnow config predict   # View resolved prediction configuration
```

## Where to find information

This CLI guide focuses on **practical command-line usage**. For other information, see:

| Topic | Documentation |
|-------|---------------|
| Configuration system, YAML structure, advanced patterns | [Configuration guide](configuration.md) |
| Python API, feature implementation, programmatic usage | [API reference](api/calibration.md) |
| Interactive tutorials and examples | [Examples notebook](https://github.com/instadeepai/winnow/blob/main/examples/getting_started_with_winnow.ipynb) |
| Contributing, development setup | [Contributing guide](contributing.md) |
