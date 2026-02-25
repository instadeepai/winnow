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

Winnow's configuration files are organised in the `configs/` directory:

```
configs/
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
winnow train calibrator.features.fragment_match_features.mz_tolerance=0.01
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

### Main training config (`configs/train.yaml`)

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
- `dataset.spectrum_path_or_directory`: Path to spectrum/metadata file (or directory for Winnow format)
- `dataset.predictions_path`: Path to predictions file (set to null for Winnow format)
- `model_output_dir`: Where to save trained model
- `dataset_output_path`: Where to save calibrated training results

### Calibrator config (`configs/calibrator.yaml`)

Controls model architecture and calibration features:

```yaml
calibrator:
  _target_: winnow.calibration.calibrator.ProbabilityCalibrator

  seed: 42
  hidden_layer_sizes: [50, 50]  # The number of neurons in each hidden layer of the MLP classifier.
  learning_rate_init: 0.001  # The initial learning rate for the MLP classifier.
  alpha: 0.0001  # L2 regularisation parameter for the MLP classifier.
  max_iter: 1000  # Maximum number of training iterations for the MLP classifier.
  early_stopping: true  # Whether to use early stopping to terminate training.
  validation_fraction: 0.1  # Proportion of training data to use for early stopping validation.

  features:
    mass_error:
      _target_: winnow.calibration.calibration_features.MassErrorFeature
      residue_masses: ${residue_masses}  # The residue masses to use for the mass error feature.

    fragment_match_features:
      _target_: winnow.calibration.calibration_features.FragmentMatchFeatures
      mz_tolerance: 0.02
      learn_from_missing: false  # If True, impute missing features and add an indicator column. If False, filter invalid entries with a warning.
      intensity_model_name: ${koina.intensity_model}  # The name of the Koina intensity model to use.
      max_precursor_charge: ${koina.constraints.max_precursor_charge}  # Maximum precursor charge accepted by the Koina intensity model.
      max_peptide_length: ${koina.constraints.max_peptide_length}      # Maximum peptide length accepted by the Koina intensity model.
      unsupported_residues: ${koina.constraints.unsupported_residues}  # Residues unsupported by the configured Koina intensity model.
      model_input_constants: ${koina.input_constants}  # Shared Koina model inputs (e.g. collision_energies, fragmentation_types).
      # model_input_columns: ${koina.input_columns}  # Uncomment to use per-row metadata columns instead.

    retention_time_feature:
      _target_: winnow.calibration.calibration_features.RetentionTimeFeature
      hidden_dim: 10  # The hidden dimension size for the MLP regressor used to predict iRT from observed retention times.
      train_fraction: 0.1  # The fraction of the data to use for training the iRT predictor.
      learn_from_missing: false  # If True, impute missing features and add an indicator column. If False, filter invalid entries with a warning.
      seed: 42  # Random seed for the MLP regressor.
      learning_rate_init: 0.001  # The initial learning rate for the MLP regressor.
      alpha: 0.0001  # L2 regularisation parameter for the MLP regressor.
      max_iter: 200  # Maximum number of training iterations for the MLP regressor.
      early_stopping: true  # Whether to use early stopping for the MLP regressor.
      validation_fraction: 0.1  # Proportion of training data to use for early stopping validation.
      irt_model_name: ${koina.irt_model}  # The name of the Koina iRT model to use.
      max_peptide_length: ${koina.constraints.max_peptide_length}      # Maximum peptide length accepted by the Koina iRT model.
      unsupported_residues: ${koina.constraints.unsupported_residues}  # Residues unsupported by the configured Koina iRT model.

    chimeric_features:
      _target_: winnow.calibration.calibration_features.ChimericFeatures
      mz_tolerance: 0.02
      learn_from_missing: false  # If True, impute missing features and add an indicator column. If False, filter invalid entries with a warning.
      prosit_intensity_model_name: ${koina.intensity_model}  # The name of the Koina intensity model to use.
      max_precursor_charge: ${koina.constraints.max_precursor_charge}  # Maximum precursor charge accepted by the Koina intensity model. Applied to the runner-up sequence.
      max_peptide_length: ${koina.constraints.max_peptide_length}      # Maximum peptide length accepted by the Koina intensity model. Applied to the runner-up (second-best) sequence.
      unsupported_residues: ${koina.constraints.unsupported_residues}  # Residues unsupported by the configured Koina intensity model.
      model_input_constants: ${koina.input_constants}  # Shared Koina model inputs (e.g. collision_energies, fragmentation_types).
      # model_input_columns: ${koina.input_columns}  # Uncomment to use per-row metadata columns instead.

    beam_features:
      _target_: winnow.calibration.calibration_features.BeamFeatures


# Koina model configuration — shared settings for all intensity- and iRT-based features.
koina:
  # Model names
  intensity_model: Prosit_2025_intensity_22PTM
  irt_model: Prosit_2025_irt_22PTM

  # Model inputs — applied to FragmentMatchFeatures and ChimericFeatures.
  # To use a constant value tiled across all rows, specify it under input_constants.
  # To use per-row values from a metadata column, add the column mapping under input_columns.
  # Each input key must appear in at most one of these two dicts.
  input_constants:
    collision_energies: 27
    fragmentation_types: HCD
  input_columns: {}

  # Model constraints — adjust to match the capabilities of your Koina models.
  # See docs/configuration.md for guidance on choosing these values.
  constraints:
    max_precursor_charge: 6   # Maximum precursor charge accepted by the intensity models.
    max_peptide_length: 30    # Maximum peptide length (residue token count) accepted by the intensity/iRT models.
    # Residues unsupported by the configured Koina models.
    # These residues must be specified using UNIMOD PTM IDs.
    # Check that all PTMs unsupported by your selected Koina models but supported by Winnow are included.
    unsupported_residues:
      # Residue modifications (amino acid + modification)
      - "N[UNIMOD:7]"   # Deamidated asparagine
      - "Q[UNIMOD:7]"   # Deamidated glutamine
      # ...
      # N-terminal modifications (standalone tokens)
      - "[UNIMOD:1]"    # N-terminal acetylation
      # ...
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

### Koina model input validation

`FragmentMatchFeatures`, `ChimericFeatures`, and `RetentionTimeFeature` all call external
[Koina](https://koina.wilhelmlab.org/) models to generate theoretical spectra or iRT values.
Before calling the model, each prediction is checked against a set of configurable validity
filters. Predictions that fail any check are treated as **missing** rather than passed to Koina.

#### Validity filters

| Parameter | Applies to | Description |
|---|---|---|
| `max_precursor_charge` | `FragmentMatchFeatures`, `ChimericFeatures` | Predictions with a precursor charge strictly greater than this value are excluded. |
| `max_peptide_length` | all three features | Predictions with more residue tokens than this limit are excluded. In `ChimericFeatures`, this limit is applied to the **runner-up (second-best) sequence**, not the top-1 prediction. |
| `unsupported_residues` | all three features | Predictions containing any of the listed ProForma tokens are excluded. |

The defaults (charge ≤ 6, length ≤ 30) match the constraints of the Prosit model family.
If you switch to a different Koina model, check its documentation and adjust these parameters
accordingly.

#### Interaction with `learn_from_missing`

Each Koina feature has a `learn_from_missing` flag that controls what happens to invalid
predictions:

- **`learn_from_missing: true`** — Invalid predictions are recorded in a boolean indicator
  column (`is_missing_fragment_match_features`, `is_missing_chimeric_features`, or
  `is_missing_irt_error`). Their feature values are imputed to zero / NaN. The calibrator
  receives this indicator as an additional feature, allowing it to distinguish genuinely
  low-scoring predictions from those that were simply out of range for the Koina model.
  Use this when your dataset is diverse and you expect a manageable proportion of peptides
  to be out of range; the calibrator will learn to account for missingness automatically.

- **`learn_from_missing: false`** — Invalid entries are automatically **filtered from the
  dataset** before Koina is called. A warning is emitted reporting how many PSMs were
  removed and which constraints were applied. The filtered PSMs are gone entirely; no
  indicator column is added and the calibrator trains only on the remaining clean data.
  Use this when you want the strictest possible data quality and are comfortable losing
  some PSMs.

#### User responsibility

The built-in filters cover the most common Koina model constraints, but they cannot
anticipate every model's requirements. **It is the user's responsibility to:**

1. Consult the [documentation](https://koina.wilhelmlab.org/) of the selected Koina model
   and verify which sequence lengths, charge states and modifications it supports.
2. Set `max_precursor_charge`, `max_peptide_length`, and `unsupported_residues` to match
   those constraints, and update `koina_model_constraints` in `calibrator.yaml` so all
   features stay consistent.
3. Be aware that Koina model constraints which are not expressible via these three
   parameters may still cause errors or silently incorrect results at prediction time,
   and may require pre-filtering the dataset before running Winnow.

Predictions that violate undocumented model constraints may cause Koina errors at prediction
time or silently produce predictions.

#### Shared Koina inputs (`koina_model_input_constants` / `koina_model_input_columns`)

Some Koina intensity models require additional inputs beyond peptide sequence and
precursor charge, typically experimental settings such as `collision_energies` or
`fragmentation_types`. These can be supplied in two ways — both defined once at the top
level and interpolated into every intensity feature to ensure they remain consistent:

- **`koina_model_input_constants`**: A single constant value tiled across all rows.
  Use this when the same collision energy applies to your whole dataset.
- **`koina_model_input_columns`**: A metadata column name providing per-row values.
  Use this when collision energy or fragmentation type varies per spectrum and is already
  present in your spectrum metadata.

Specifying the same key in both dicts raises a `ValueError` at construction time.

#### Shared model constraints (`koina_model_constraints`)

`max_precursor_charge` and `max_peptide_length` are defined once in the top-level
`koina_model_constraints` block and interpolated via `${koina_model_constraints.*}` into
every Koina feature. This ensures all features apply the same limits without requiring
manual duplication. To override a constraint for a single feature only, replace the
interpolation with a literal value directly in that feature's config block.

## Prediction configuration

### Main prediction config (`configs/predict.yaml`)

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
- `dataset.spectrum_path_or_directory`: Path to spectrum/metadata file (or directory for Winnow format)
- `dataset.predictions_path`: Path to predictions file
- `calibrator.pretrained_model_name_or_path`: HuggingFace model identifier or local model directory path
- `calibrator.cache_dir`: Directory to cache HuggingFace models (null for default)
- `fdr_method`: FDR estimation method (via defaults: `nonparametric` or `database_grounded`)
- `fdr_control.fdr_threshold`: Target FDR threshold (e.g. 0.01 for 1%, 0.05 for 5%)
- `fdr_control.confidence_column`: Column name with confidence scores
- `output_folder`: Where to save results

### FDR method configs

**Non-parametric FDR** (`configs/fdr_method/nonparametric.yaml`):

```yaml
_target_: winnow.fdr.nonparametric.NonParametricFDRControl
```

No additional parameters required.

**Database-grounded FDR** (`configs/fdr_method/database_grounded.yaml`):

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

### Residues config (`configs/residues.yaml`)

Defines amino acid masses, modifications and residues unsupported by the configured Koina models:

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
```

This configuration is shared across all pipelines and referenced via `${residue_masses}` and `${unsupported_residues}` interpolation.

Winnow represents PTMs using the UNIMOD format internally, so all residue masses and PTMs to be filtered must use this format. Please check that all PTMs unsupported by your selected Koina models are included in `unsupported_residues`.

### Data loader configs

Each data format has a dedicated loader configuration in `configs/data_loader/`:

**InstaNovo** (`configs/data_loader/instanovo.yaml`):
```yaml
_target_: winnow.datasets.data_loaders.InstaNovoDatasetLoader
residue_masses: ${residue_masses}
residue_remapping:
  "M(ox)": "M[UNIMOD:35]"
  "C(+57.02)": "C[UNIMOD:4]"
  # ... maps legacy notations to UNIMOD tokens
beam_columns:
  sequence: "predictions_beam_"
  log_probability: "predictions_log_probability_beam_"
  token_log_probabilities: "predictions_token_log_probabilities_"
```

#### Beam column configuration

The InstaNovo loader reads beam search predictions from CSV columns that follow a naming convention:
each beam column has a **prefix** that is appended with a **beam index** (0, 1, 2, ...).
Beam column configurability means that Winnow can be run on InstaNovo predictions made with or without refinement,
a setting which changes the column names of saved beams.

The `beam_columns` parameter specifies the prefix for each required column type:

| Key | Description | Example columns |
|-----|-------------|-----------------|
| `sequence` | Peptide sequence for each beam | `predictions_beam_0`, `predictions_beam_1`, ... |
| `log_probability` | Log probability score for each beam | `predictions_log_probability_beam_0`, ... |
| `token_log_probabilities` | Per-token log probabilities | `predictions_token_log_probabilities_beam_0`, ... |

**Column naming requirements:**

- Each prefix must match at least one column in the predictions CSV
- Column names must be exactly `<prefix><beam_index>` (e.g., `predictions_beam_0`)
- The beam index must be a non-negative integer (0, 1, 2, ...)
- All three column types are required for each beam

**Disabling beam predictions:**

If your predictions CSV doesn't have beam columns (single-prediction mode) or you only need
metadata features, you can disable beam loading entirely by setting `beam_columns` to `null`:

```bash
winnow train data_loader.beam_columns=null
```

**MZTab** (`configs/data_loader/mztab.yaml`):
```yaml
_target_: winnow.datasets.data_loaders.MZTabDatasetLoader
residue_masses: ${residue_masses}
load_beams: true  # Set to false to disable beam predictions
residue_remapping:
  "M+15.995": "M[UNIMOD:35]"
  "C+57.021": "C[UNIMOD:4]"
  "C[Carbamidomethyl]": "C[UNIMOD:4]"
  # ... maps Casanovo notations to UNIMOD tokens
```

The `load_beams` parameter controls whether beam predictions are created from multiple
predictions per spectrum. Set to `false` if you only need metadata features.

**PointNovo** (`configs/data_loader/pointnovo.yaml`):
```yaml
_target_: winnow.datasets.data_loaders.PointNovoDatasetLoader
residue_masses: ${residue_masses}
```

**Winnow** (`configs/data_loader/winnow.yaml`):
```yaml
_target_: winnow.datasets.data_loaders.WinnowDatasetLoader
residue_masses: ${residue_masses}
# Internal format uses UNIMOD tokens directly, no remapping needed
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
- `${unsupported_residues}` - References Koina-unsupported residue tokens from residues.yaml
- `${koina_model_input_constants}` - References shared Koina model inputs (e.g. collision energy) from calibrator.yaml
- `${koina_model_constraints.max_precursor_charge}` - References shared charge limit from calibrator.yaml
- `${koina_model_constraints.max_peptide_length}` - References shared length limit from calibrator.yaml
- `${fdr_control.confidence_column}` - References FDR confidence column setting

## Creating custom configurations

### Add a custom data loader

1. Create loader class implementing `DatasetLoader` protocol
2. Add configuration file: `configs/data_loader/custom.yaml`
3. Use with: `winnow train data_loader=custom`

Example `configs/data_loader/custom.yaml`:
```yaml
_target_: my_module.CustomDatasetLoader
residue_masses: ${residue_masses}
custom_param: value
```

### Add custom calibration features

1. Create feature class inheriting from `CalibrationFeatures`
2. Add to `configs/calibrator.yaml`:
   ```yaml
   features:
     custom_feature:
       _target_: my_module.CustomFeature
       param1: value1
       param2: value2
   ```

### Add custom FDR method

1. Create FDR class implementing the FDR interface
2. Add configuration file: `configs/fdr_method/custom_method.yaml`
3. Use with: `winnow predict fdr_method=custom_method`

Example `configs/fdr_method/custom_method.yaml`:
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

## Advanced: custom config directories

For advanced users who have installed Winnow as a package and need to customise multiple configuration files, you can use the `--config-dir` flag to specify a custom configuration directory. This is particularly useful when you have complex customisations that would be verbose to specify via command-line overrides.

**When to use custom config directories:**

- **CLI overrides**: For simple parameter changes (1-3 values) - use command-line overrides like `winnow train calibrator.seed=42`
- **Custom config dirs**: For complex configurations with many custom settings (advanced users) - use `--config-dir`
- **Cloning repo**: For extending Winnow (developers) - clone the repository, make changes, and modify configs directly

### Config file structure and maming

Your custom config directory should mirror the structure of the package configs:

```
my_configs/
├── residues.yaml              # Override residue masses/modifications
├── calibrator.yaml            # Override calibrator features
├── train.yaml                 # Override training config (if needed)
├── predict.yaml               # Override prediction config (if needed)
├── data_loader/               # Override data loaders (if needed)
│   └── instanovo.yaml
│   └── mztab.yaml
│   └── winnow.yaml
└── fdr_method/               # Override FDR methods (if needed)
│   └── database_grounded.yaml
│   └── nonparametric.yaml
```

**Important requirements:**

- File names must match package config names **exactly** (case-sensitive)
- Directory structure should mirror package structure (e.g., `data_loader/`, `fdr_method/`)
- Only include files you want to override - you don't need to include everything
- YAML files must be valid and follow the same structure as package configs

### Partial configs

You can use **partial configs at the file level** - only include the files you want to override. For example, if you only want to customise residue masses and calibrator settings:

```bash
my_configs/
├── residues.yaml       # Your custom residues
└── calibrator.yaml    # Your custom calibrator config
```

When you use `--config-dir`, Winnow will:

1. Use your custom files for files present in your directory (these completely replace package versions)
2. Use package defaults for files not in your directory

**Important limitation**: Partial configs work at the **file level**, not the **key level** within a file. If you provide a custom `calibrator.yaml`, it must contain the complete structure - you can't just override `seed` and expect other settings to come from package defaults. See "Behaviour with variables" below for details.

### Behaviour with variables

**How config files work**: When you provide a custom config file (e.g., `calibrator.yaml`), it **completely replaces** the package version of that file. Hydra does not merge keys within the same file - it uses your file exactly as written.

**What this means:**

- ✅ **Partial configs at file level**: You only need to include the files you want to override (e.g., just `residues.yaml` and `calibrator.yaml`). Files not in your custom directory use package defaults.
- ❌ **Partial configs at key level don't work**: If you provide `calibrator.yaml` with only `seed: 999`, the other settings (`hidden_layer_sizes`, `features`, etc.) will be **missing**, not using package defaults. This will cause errors.

**Example - What happens with minimal config:**
```yaml
# custom/calibrator.yaml - TOO MINIMAL
calibrator:
  _target_: winnow.calibration.calibrator.ProbabilityCalibrator
  seed: 99999
```

**Result**: Only `_target_` and `seed` are present. All other keys (`hidden_layer_sizes`, `learning_rate_init`, `features`, etc.) are **missing** from the final config. This will cause errors when running the pipeline in most cases.

**Example - What you need (complete structure):**
```yaml
# custom/calibrator.yaml - COMPLETE STRUCTURE REQUIRED
calibrator:
  _target_: winnow.calibration.calibrator.ProbabilityCalibrator
  seed: 99999  # Your custom value
  hidden_layer_sizes: [50, 50]  # Must include all settings
  learning_rate_init: 0.001
  alpha: 0.0001
  max_iter: 1000
  early_stopping: true
  validation_fraction: 0.1
  features:
    mass_error:
      _target_: winnow.calibration.calibration_features.MassErrorFeature
      residue_masses: ${residue_masses}
    fragment_match_features:
      # ... include all features you want to keep
    # Features you don't include will be missing (not using defaults)
```

**Removing features**: To remove features, simply **don't include them** in your custom `calibrator.yaml`. Since your file completely replaces the package version, any features you omit will be absent from the final config. It is also possible to specify this using a tilde with CLI overrides (e.g., `~calibrator.features.fragment_match_features`).

**New variables**: Adding new keys that don't exist in package configs will cause them to be ignored (Hydra is not strict by default). They won't cause errors, but they also won't be used unless your code explicitly accesses them. **Stick to overriding existing keys** from package configs.

**Recommendation**: Always start by copying the complete package config file, then modify only the values you need. You can get the package config structure by running `winnow config train` and copying the relevant section, or by copying from `winnow/configs/` in the [repository](https://github.com/instadeepai/winnow).

### Getting package config structure

Before creating custom configs, you need to know the structure of package configs. Here are ways to get them:

1. **View resolved config**: Run `winnow config train` or `winnow config predict` to see the complete resolved configuration
2. **Clone the repository**: Visit the Winnow [repository](https://github.com/instadeepai/winnow) and check `winnow/configs/` directory
3. **Inspect installed package**: Find the package location (e.g., `python -c "import winnow; print(winnow.__file__)"`) and navigate to `configs/`

**Recommended workflow**: Start by creating a new config directory, copying in the package config file you want to customise, and then modify only the values you need.

### Troubleshooting

**Common mistakes:**

1. **Wrong file names**: File names must match exactly (case-sensitive)

    - ❌ `Residues.yaml` (wrong case)
    - ✅ `residues.yaml` (correct)

2. **Incorrect structure**: Directory structure must match package structure

    - ❌ `my_configs/data_loaders/` (wrong directory name)
    - ✅ `my_configs/data_loader/` (correct)

3. **Typos in keys**: YAML keys must match package config keys exactly

    - Check package configs for correct key names

4. **Invalid YAML**: Ensure your YAML files are valid

    - Use a YAML validator if unsure

**How to check if custom config is being used:**

1. Use `winnow config train --config-dir my_configs` and search for your custom values
2. Compare output with `winnow config train` (without custom dir) to see differences
3. Check logs - Winnow logs which config directory is being used

## Additional resources

- [Hydra documentation](https://hydra.cc/docs/intro/)
- [OmegaConf documentation](https://omegaconf.readthedocs.io/)
- [Winnow API documentation](api/calibration.md)
- [Example notebook](https://github.com/instadeepai/winnow/blob/main/examples/getting_started_with_winnow.ipynb)
