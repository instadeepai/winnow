# Configuration guide

This guide provides comprehensive documentation of Winnow's configuration system, including YAML file structure, parameter reference, advanced patterns and customisation.

**Looking for practical CLI usage?** See the **[CLI reference](cli.md)** for command examples and workflows.

## Overview

Winnow uses [Hydra](https://hydra.cc/) for flexible, hierarchical configuration management. This enables:

- **Composable configs**: Build configurations from multiple YAML files
- **Flexibility**: Override any parameter via command line or config files
- **Reproducibility**: Full configuration is automatically logged

## Quick start

Winnow works out of the box with sensible defaults:

```bash
# Train with default settings
winnow train

# Predict with default settings
winnow predict
```

## Configuration files

Winnow's configuration files are organised in the `config/` directory:

```
config/
├── residues.yaml              # Amino acid masses, modifications
├── data_loader/               # Dataset format loaders
│   ├── instanovo.yaml
│   ├── mztab.yaml
│   ├── pointnovo.yaml
│   └── winnow.yaml
├── fdr_method/                # FDR control methods
│   ├── nonparametric.yaml
│   └── database_grounded.yaml
├── train.yaml                 # Main training config
├── calibrator.yaml            # Model architecture and features
└── predict.yaml               # Main prediction config
```

## Overriding configuration

All configuration parameters have default values defined in YAML files. You can override any parameter from the command line.

### Command-line overrides

Override any parameter from the command line:

```bash
# Override dataset paths
winnow train dataset.spectrum_path_or_directory=data/my_spectra.parquet dataset.predictions_path=data/my_preds.csv

# Change data loader
winnow train data_loader=mztab

# Change output directory
winnow train model_output_dir=models/my_model

# Change multiple parameters
winnow predict data_loader=mztab fdr_control.fdr_threshold=0.01 fdr_method=database_grounded
```

### Nested parameters

Access nested configuration values using dot notation:

```bash
# Change calibrator seed
winnow train calibrator.seed=123

# Change MLP hidden layer sizes
winnow train calibrator.hidden_layer_sizes=[100,50,25]

# Change feature parameters
winnow train calibrator.features.prosit_features.mz_tolerance=0.01
```

### Dataset configuration

Specify dataset paths using nested notation:

```bash
# For InstaNovo format
winnow train dataset.spectrum_path_or_directory=data/spectra.parquet dataset.predictions_path=data/preds.csv

# For MZTab format
winnow train data_loader=mztab dataset.spectrum_path_or_directory=data/spectra.parquet dataset.predictions_path=data/results.mztab
```

## Training configuration

### Main training config (`config/train.yaml`)

Controls dataset loading, output paths and composition:

```yaml
defaults:
  - _self_
  - residues
  - calibrator
  - data_loader: instanovo  # Options: instanovo, mztab, pointnovo, winnow

dataset:
  # Path to the spectrum data file or to folder containing saved internal Winnow dataset
  spectrum_path_or_directory: data/spectra.ipc
  # Path to the beam predictions file
  # Leave as null if data source is winnow, or loading will fail
  predictions_path: data/predictions.csv

# Output paths
model_output_dir: models/new_model
dataset_output_path: results/calibrated_dataset.csv
```

**Key parameters:**

- `data_loader`: Format of input data loader to use (via defaults: `instanovo`, `mztab`, `pointnovo`, `winnow`)
- `dataset.spectrum_path_or_directory`: Path to spectrum/metadata file (or directory for winnow format)
- `dataset.predictions_path`: Path to predictions file (set to null for winnow format)
- `model_output_dir`: Where to save trained model
- `dataset_output_path`: Where to save calibrated training results

### Calibrator config (`config/calibrator.yaml`)

Controls model architecture and calibration features:

```yaml
calibrator:
  _target_: winnow.calibration.calibrator.ProbabilityCalibrator

  seed: 42
  hidden_layer_sizes: [50, 50]  # The number of neurons in each hidden layer of the MLP classifier
  learning_rate_init: 0.001  # The initial learning rate for the MLP classifier
  alpha: 0.0001  # L2 regularisation parameter for the MLP classifier
  max_iter: 1000  # Maximum number of training iterations for the MLP classifier
  early_stopping: true  # Whether to use early stopping to terminate training
  validation_fraction: 0.1  # Proportion of training data to use for early stopping validation

  features:
    mass_error:
      _target_: winnow.calibration.calibration_features.MassErrorFeature
      residue_masses: ${residue_masses}

    prosit_features:
      _target_: winnow.calibration.calibration_features.PrositFeatures
      mz_tolerance: 0.02
      learn_from_missing: true
      invalid_prosit_tokens: ${invalid_prosit_tokens}
      prosit_intensity_model_name: Prosit_2020_intensity_HCD

    retention_time_feature:
      _target_: winnow.calibration.calibration_features.RetentionTimeFeature
      hidden_dim: 10
      train_fraction: 0.1
      learn_from_missing: true
      seed: 42
      learning_rate_init: 0.001
      alpha: 0.0001
      max_iter: 200
      early_stopping: false
      validation_fraction: 0.1
      invalid_prosit_tokens: ${invalid_prosit_tokens}
      prosit_irt_model_name: Prosit_2019_irt

    chimeric_features:
      _target_: winnow.calibration.calibration_features.ChimericFeatures
      mz_tolerance: 0.02
      learn_from_missing: true
      invalid_prosit_tokens: ${invalid_prosit_tokens}
      prosit_intensity_model_name: Prosit_2020_intensity_HCD

    beam_features:
      _target_: winnow.calibration.calibration_features.BeamFeatures
```

**Key parameters:**

- `seed`: Random seed for reproducibility
- `hidden_layer_sizes`: Architecture of MLP classifier
- `learning_rate_init`: Initial learning rate
- `alpha`: L2 regularisation parameter
- `max_iter`: Maximum training iterations
- `early_stopping`: Whether to use early stopping
- `validation_fraction`: Proportion of data for validation
- `features.*`: Individual calibration feature configurations

## Prediction configuration

### Main prediction config (`config/predict.yaml`)

Controls dataset loading, FDR estimation and output:

```yaml
defaults:
  - _self_
  - residues
  - data_loader: instanovo  # Options: instanovo, mztab, pointnovo, winnow
  - fdr_method: nonparametric  # Options: nonparametric, database_grounded

dataset:
  # Path to the spectrum data file or to folder containing saved internal Winnow dataset
  spectrum_path_or_directory: data/spectra.ipc
  # Path to the beam predictions file
  # Leave as null if data source is winnow, or loading will fail
  predictions_path: data/predictions.csv

calibrator:
  # Path to the local calibrator directory or the HuggingFace model identifier
  # If the path is a local directory path, it will be used directly
  # If it is a HuggingFace repository identifier, it will be downloaded from HuggingFace
  pretrained_model_name_or_path: InstaDeepAI/winnow-general-model
  # Directory to cache the HuggingFace model
  cache_dir: null  # can be set to null if using local model or for the default cache directory

fdr_control:
  # Target FDR threshold (e.g. 0.01 for 1%, 0.05 for 5% etc.)
  fdr_threshold: 0.05
  # Name of the column with confidence scores to use for FDR estimation
  confidence_column: calibrated_confidence

# Folder path to write the outputs to
output_folder: results/predictions
```

**Key parameters:**

- `data_loader`: Format of input data loader to use (via defaults: `instanovo`, `mztab`, `pointnovo`, `winnow`)
- `dataset.spectrum_path_or_directory`: Path to spectrum/metadata file (or directory for winnow format)
- `dataset.predictions_path`: Path to predictions file
- `calibrator.pretrained_model_name_or_path`: HuggingFace model identifier or local model directory path
- `calibrator.cache_dir`: Directory to cache HuggingFace models (null for default)
- `fdr_method`: FDR estimation method (via defaults: `nonparametric` or `database_grounded`)
- `fdr_control.fdr_threshold`: Target FDR threshold (e.g. 0.01 for 1%, 0.05 for 5%)
- `fdr_control.confidence_column`: Column name with confidence scores
- `output_folder`: Where to save results

### FDR method configs

**Non-parametric FDR** (`config/fdr_method/nonparametric.yaml`):

```yaml
_target_: winnow.fdr.nonparametric.NonParametricFDRControl
```

No additional parameters required.

**Database-grounded FDR** (`config/fdr_method/database_grounded.yaml`):

```yaml
_target_: winnow.fdr.database_grounded.DatabaseGroundedFDRControl
confidence_feature: ${fdr_control.confidence_column}
residue_masses: ${residue_masses}
isotope_error_range: [0, 1]
drop: 10
```

Requires ground truth sequences in the dataset.

**Key parameters:**

- `confidence_feature`: Name of the column with confidence scores (interpolated from fdr_control)
- `residue_masses`: Amino acid and modification masses (interpolated from residues config)
- `isotope_error_range`: Range of isotope errors to consider when matching peptides
- `drop`: Number of top predictions to drop for stability

## Shared configuration

### Residues config (`config/residues.yaml`)

Defines amino acid masses, modifications and invalid tokens:

```yaml
residue_masses:
  "G": 57.021464
  "A": 71.037114
  "S": 87.032028
  # ... other amino acids
  "M[UNIMOD:35]": 147.035400  # Oxidation
  "C[UNIMOD:4]": 160.030649   # Carboxyamidomethylation
  "N[UNIMOD:7]": 115.026943   # Deamidation
  "Q[UNIMOD:7]": 129.042594   # Deamidation
  # ... other modifications
  "[UNIMOD:1]": 42.010565     # Acetylation (terminal)
  "[UNIMOD:5]": 43.005814     # Carbamylation (terminal)
  "[UNIMOD:385]": -17.026549  # NH3 loss (terminal)

invalid_prosit_tokens:
  # InstaNovo
  - "[UNIMOD:7]"
  - "[UNIMOD:21]"
  - "[UNIMOD:1]"
  - "[UNIMOD:5]"
  - "[UNIMOD:385]"
  # Casanovo
  - "+0.984"
  - "+42.011"
  - "+43.006"
  - "-17.027"
  - "[Deamidated]"
  # ... other unsupported modifications
```

This configuration is shared across all pipelines and referenced via `${residue_masses}` and `${invalid_prosit_tokens}` interpolation.

### Data loader configs

Each data format has a dedicated loader configuration in `config/data_loader/`:

**InstaNovo** (`config/data_loader/instanovo.yaml`):
```yaml
_target_: winnow.datasets.data_loaders.InstaNovoDatasetLoader
residue_masses: ${residue_masses}
residue_remapping:
  "M(ox)": "M[UNIMOD:35]"
  "C(+57.02)": "C[UNIMOD:4]"
  # ... maps legacy notations to UNIMOD tokens
```

**MZTab** (`config/data_loader/mztab.yaml`):
```yaml
_target_: winnow.datasets.data_loaders.MZTabDatasetLoader
residue_masses: ${residue_masses}
residue_remapping:
  "M+15.995": "M[UNIMOD:35]"
  "C+57.021": "C[UNIMOD:4]"
  "C[Carbamidomethyl]": "C[UNIMOD:4]"
  # ... maps Casanovo notations to UNIMOD tokens
```

**PointNovo** (`config/data_loader/pointnovo.yaml`):
```yaml
_target_: winnow.datasets.data_loaders.PointNovoDatasetLoader
residue_masses: ${residue_masses}
```

**Winnow** (`config/data_loader/winnow.yaml`):
```yaml
_target_: winnow.datasets.data_loaders.WinnowDatasetLoader
residue_masses: ${residue_masses}
# Internal format uses UNIMOD tokens directly, no remapping needed
```

## Common configuration patterns

### Using a custom model

```bash
# Use a custom HuggingFace model
winnow predict calibrator.pretrained_model_name_or_path=my-org/my-model

# Use a locally trained model
winnow predict calibrator.pretrained_model_name_or_path=models/my_model

# Specify a custom HuggingFace cache directory
winnow predict calibrator.cache_dir=/path/to/cache
```

### Changing FDR method and threshold

```bash
# Use database-grounded FDR at 1%
winnow predict fdr_method=database_grounded fdr_control.fdr_threshold=0.01

# Use non-parametric FDR at 5%
winnow predict fdr_method=nonparametric fdr_control.fdr_threshold=0.05
```

### Training with different features

```bash
# Change Prosit tolerance
winnow train calibrator.features.prosit_features.mz_tolerance=0.01

# Disable missing value handling for a feature
winnow train calibrator.features.prosit_features.learn_from_missing=false

# Change retention time model architecture
winnow train calibrator.features.retention_time_feature.hidden_dim=20
```

### Processing different data formats

```bash
# MZTab format
winnow train data_loader=mztab dataset.spectrum_path_or_directory=data/spectra.parquet dataset.predictions_path=data/results.mztab

# Previously saved Winnow dataset
winnow train data_loader=winnow dataset.spectrum_path_or_directory=data/winnow_dataset/ dataset.predictions_path=null
```

## Config interpolation

Hydra supports variable interpolation using `${...}` syntax:

```yaml
# Reference from residues config (loaded via defaults)
features:
  mass_error:
    residue_masses: ${residue_masses}  # References residue_masses from residues.yaml

# Reference nested values
fdr_control:
  confidence_column: calibrated_confidence

database_grounded:
  confidence_feature: ${fdr_control.confidence_column}  # References nested value

# Use in defaults for dynamic composition
defaults:
  - fdr_method: nonparametric  # Loads fdr_method/nonparametric.yaml
```

Common interpolation patterns in Winnow configs:
- `${residue_masses}` - References amino acid masses from residues.yaml
- `${invalid_prosit_tokens}` - References invalid tokens from residues.yaml
- `${fdr_control.confidence_column}` - References FDR confidence column setting

## Creating custom configurations

### Add a custom data loader

1. Create loader class implementing `DatasetLoader` protocol
2. Add configuration file: `config/data_loader/custom.yaml`
3. Use with: `winnow train data_loader=custom`

Example `config/data_loader/custom.yaml`:
```yaml
_target_: my_module.CustomDatasetLoader
residue_masses: ${residue_masses}
custom_param: value
```

### Add custom calibration features

1. Create feature class inheriting from `CalibrationFeatures`
2. Add to `config/calibrator.yaml`:
   ```yaml
   features:
     custom_feature:
       _target_: my_module.CustomFeature
       param1: value1
       param2: value2
   ```

### Add custom FDR method

1. Create FDR class implementing the FDR interface
2. Add configuration file: `config/fdr_method/custom_method.yaml`
3. Use with: `winnow predict fdr_method=custom_method`

Example `config/fdr_method/custom_method.yaml`:
```yaml
_target_: my_module.CustomFDRControl
confidence_feature: ${fdr_control.confidence_column}
custom_param: value
```

## Debugging configuration

### View resolved configuration

To see the final composed configuration without running the pipeline, use the `winnow config` command:

```bash
# View training configuration
winnow config train

# View prediction configuration
winnow config predict

# View configuration with overrides
winnow config train data_loader=mztab model_output_dir=custom/path
winnow config predict fdr_method=database_grounded fdr_control.fdr_threshold=0.01
```

This prints the complete resolved YAML configuration with colour-coded hierarchical formatting for easy readability. Keys are coloured by nesting depth to help visualise the configuration structure. The output shows all defaults, composition and command-line overrides after they have been applied.

**Note:** Some keys appear with quotes (e.g. `'N'`, `'Y'`) because they are reserved words in YAML that would otherwise be interpreted as boolean values. The quotes ensure they are treated as strings.

### Configuration validation

Hydra will validate that configuration files exist and can be composed. Invalid configurations will fail early with clear error messages:

```bash
winnow predict fdr_method=typo
# Error: Could not find 'fdr_method/typo'
# Available options in 'fdr_method': nonparametric, database_grounded
```

## Additional resources

- [Hydra documentation](https://hydra.cc/docs/intro/)
- [OmegaConf documentation](https://omegaconf.readthedocs.io/)
- [Winnow API documentation](api/calibration.md)
- [Example notebook](https://github.com/instadeepai/winnow/blob/main/examples/getting_started_with_winnow.ipynb)

## Migration from old CLI

If you're migrating from the old argument-based CLI:

| Old CLI Argument | New Hydra Config Parameter |
|------------------|---------------------------|
| `--data-source instanovo` | `data_loader=instanovo` |
| `--model-output-folder models/X` | `model_output_dir=models/X` |
| `--dataset-output-path results/X.csv` | `dataset_output_path=results/X.csv` |
| `--fdr-threshold 0.01` | `fdr_control.fdr_threshold=0.01` |
| `--method winnow` | `fdr_method=nonparametric` |
| `--method database-ground` | `fdr_method=database_grounded` |
| `--local-model-folder models/X` | `calibrator.pretrained_model_name_or_path=models/X` |
| `--huggingface-model-name X` | `calibrator.pretrained_model_name_or_path=X` |
| `--confidence-column X` | `fdr_control.confidence_column=X` |
| `--output-folder results/` | `output_folder=results/` |

**Key changes:**

- `data_source` renamed to `data_loader` (references config/data_loader/*.yaml)
- `fdr_threshold` and `confidence_column` now nested under `fdr_control`
- `local_model_folder` and `huggingface_model_name` merged into `pretrained_model_name_or_path`
- Dataset paths are now specified directly as Hydra parameters instead of via separate YAML files:
  - **Old**: Create `config.yaml` with `spectrum_path` and `predictions_path`, then use `--dataset-config-path config.yaml`
  - **New**: Use `dataset.spectrum_path_or_directory=... dataset.predictions_path=...` directly on command line
