# Examples

## Quick Start with Sample Data

Get started with `winnow` in minutes using the included sample data:

```bash
# Generate sample data
make sample-data

# Train a calibrator
make train-sample

# Run prediction
make predict-sample
```

The sample data is automatically configured in the default config files. See the [CLI guide](cli.md#quick-start) for more details.

## Comprehensive Example Notebook

For a comprehensive example demonstrating the full winnow workflow, see our example notebook:

📓 **[FDR plots example notebook](https://github.com/instadeepai/winnow/blob/main/examples/getting_started_with_winnow.ipynb)**

This notebook shows you how to:

- Load and prepare PSM data using `CalibrationDataset`
- Train a confidence calibrator with `ProbabilityCalibrator`
- Apply FDR control using different methods
- Generate visualisation plots for calibration and FDR analysis
