# Command Line Interface

The winnow CLI provides a simple interface for confidence calibration and FDR control workflows. It supports both training calibration models and applying them for prediction and FDR filtering.

## Installation

After installing winnow, the `winnow` command becomes available:

```bash
pip install winnow_fdr
# or
uv pip install winnow_fdr
```

## Commands

### `winnow train`

Train a confidence calibration model on labelled data.

```bash
winnow train \
    --data-source instanovo \
    --dataset-config-path train_config.yaml \
    --model-output-folder ./calibrator_model \
    --dataset-output-path ./training_results.csv
```

**Arguments:**

- `--data-source`: Type of dataset (`instanovo`, `mztab`, `winnow`)
- `--dataset-config-path`: Path to YAML configuration file
- `--model-output-folder`: Directory to save trained calibrator
- `--dataset-output-path`: Path to save training results CSV

### `winnow predict`

Apply calibration and FDR control to new data using a trained model.

```bash
winnow predict \
    --data-source instanovo \
    --dataset-config-path test_config.yaml \
    --model-folder ./calibrator_model \
    --method winnow \
    --fdr-threshold 0.01 \
    --confidence-column confidence \
    --output-path ./predictions.csv
```

**Arguments:**

- `--data-source`: Type of dataset (`instanovo`, `winnow`, `mztab`)
- `--dataset-config-path`: Path to YAML configuration file
- `--model-folder`: Directory containing trained calibrator
- `--method`: FDR estimation method (`winnow` or `database-ground`)
- `--fdr-threshold`: Target FDR threshold (e.g., 0.01 for 1%)
- `--confidence-column`: Name of confidence score column
- `--output-path`: Path to save filtered results

## Configuration Files

The CLI uses YAML configuration files to specify dataset locations and parameters. The format depends on the data source.

### InstaNovo Configuration

For InstaNovo/CSV datasets:

```yaml
# instanovo_config.yaml
beam_predictions_path: "/path/to/predictions.csv"
spectrum_path: "/path/to/spectra.csv"
```

**Required files:**

- **Spectrum file**: Parquet/IPC file containing spectral metadata and features
- **Predictions CSV**: Contains beam search results with columns like `preds`, `preds_beam_1`, confidence scores

### MZTab Configuration

For MZTab format datasets (traditional search engines and Casanovo outputs):

```yaml
# mztab_config.yaml
spectrum_path: "/path/to/spectra.parquet"
predictions_path: "/path/to/predictions.mztab"
```

**Required files:**

- **Spectrum file**: Parquet/IPC file with spectrum metadata and row indices matching MZTab spectra_ref
- **MZTab file**: Standard MZTab format containing predictions

### Winnow Internal Configuration

For winnow's internal format:

```yaml
# winnow_config.yaml
data_dir: "/path/to/winnow_dataset_directory"
```

**Required structure:**

- Directory containing `metadata.csv` and optionally `predictions.pkl`
- Created by `CalibrationDataset.save()`

## Data Requirements

### Training Data

For training (`winnow train`), you need:

- **Labelled dataset**: Ground truth peptide sequences for evaluation
- **Predictions**: Model predictions with confidence scores
- **Spectral data**: MS/MS spectra and metadata
- **Unique identifiers**: Each PSM must have a unique `spectrum_id` in both input files

### Prediction Data

For prediction (`winnow predict`), you need:

- **Unlabelled dataset**: Predictions and spectra (no ground truth required)
- **Trained model**: Output from `winnow train`
- **Confidence scores**: Raw confidence values to calibrate
- **Unique identifiers**: Each PSM must have a unique `spectrum_id` in both input files

## FDR Methods

### Winnow Method (`--method winnow`)

Uses non-parametric FDR estimation procedure:

- **No ground truth required**: Works with confidence scores alone
- **No correct/incorrect distribution modelling**: Process directly estimates FDR using calibrated confidence scores
- **Multiple metrics**: Provides FDR, PEP and q-values.

```bash
winnow predict \
    --method winnow \
    --fdr-threshold 0.01 \
    --confidence-column calibrated_confidence \
    # ... other args
```

### Database-Grounded Method (`--method database-ground`)

Uses database search results for validation:

- **Requires ground truth**: Needs labelled data with sequence column
- **Precision-recall based**: Computes FDR from validation results
- **Direct estimates**: FDR calculated from actual correct/incorrect labels

```bash
winnow predict \
    --method database-ground \
    --fdr-threshold 0.05 \
    --confidence-column confidence \
    # ... other args
```

## Output Files

### Training Output

Training produces:

1. **Model checkpoints** (in `--model-output-folder`):
   - `calibrator.pkl`: Trained neural network classifier
   - `irt_predictor.pkl`: Retention time predictor
   - `scaler.pkl`: Feature scaler

2. **Training results** (`--dataset-output-path`):
   - CSV with calibrated scores and evaluation metrics

### Prediction Output

Prediction produces a CSV file (`--output-path`) containing:

- **Original columns**: All input data columns
- **`calibrated_confidence`**: Calibrated confidence scores
- **`psm_fdr`**: PSM-specific FDR estimates
- **`psm_q_value`**: Q-values
- **`psm_pep`**: Posterior error probabilities (winnow method only)
- **Filtered rows**: Only PSMs passing the FDR threshold

## Example Workflows

### Complete Training and Prediction Pipeline

```bash
# Step 1: Train calibrator on labelled data
winnow train \
    --data-source instanovo \
    --dataset-config-path configs/train_data.yaml \
    --model-output-folder models/my_calibrator \
    --dataset-output-path results/training_output.csv

# Step 2: Apply to new data with FDR control
winnow predict \
    --data-source instanovo \
    --dataset-config-path `configs/test_data.yaml` \
    --model-folder `models/my_calibrator` \
    --method winnow \
    --fdr-threshold 0.01 \
    --confidence-column confidence \
    --output-path `results/filtered_predictions.csv`
```

### Configuration File Examples

**Training configuration** (`configs/train_data.yaml`):
```yaml
beam_predictions_path: "data/train_predictions.csv"
spectrum_path: "data/train_spectra.csv"
```

**Test configuration** (`configs/test_data.yaml`):
```yaml
beam_predictions_path: "data/test_predictions.csv"
spectrum_path: "data/test_spectra.csv"
```

## Built-in Features

The CLI automatically includes these calibration features:

- **Mass Error**: Difference between observed and theoretical mass
- **Prosit Features**: ML-based intensity predictions (`mz_tolerance=0.02`)
- **Retention Time**: iRT predictions (`hidden_dim=10`, `train_fraction=0.1`)
- **Chimeric Features**: Chimeric spectrum detection (`mz_tolerance=0.02`)
- **Beam Features**: Beam search diversity metrics

## Default Parameters

The CLI uses these default parameters:

- **Seed**: 42 (for reproducibility)
- **MZ Tolerance**: 0.02 Da
- **FDR Learning Rate**: 0.005
- **FDR Training Steps**: 5000
- **Retention Time Hidden Dim**: 10
- **Retention Time Train Fraction**: 0.1

## Troubleshooting

### Common Issues

**Missing columns**: Ensure your CSV files contain expected columns:

- `preds`: Main prediction
- `confidence`: Confidence scores
- `sequence`: Ground truth (for training/database method)

**File paths**: Use absolute paths in configuration files to avoid path resolution issues.

**Memory issues**: Large datasets may require more memory. Consider filtering data or using smaller batch sizes.

### Dataset Filtering

The CLI automatically filters out:

- Empty predictions
- Peptides longer than 30 amino acids (Prosit limitation)
- Precursor charges above 6 (Prosit limitation)
- Invalid modifications and tokens

For more advanced usage and customisation, refer to the [Python API documentation](api/calibration.md) and [examples notebook](https://github.com/instadeepai/winnow/blob/main/examples/getting_started_with_winnow.ipynb).
