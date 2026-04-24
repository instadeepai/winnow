# Examples

## Quick start with example data

Try `winnow` in minutes using the bundled example subset under `examples/example_data/`: spectra (`spectra.ipc`) and de novo predictions (`predictions.csv`) from a real HeLa single-shot run, reduced for repository size.

```bash
# Train a calibrator
make train-sample

# Run prediction (uses the model from train-sample)
make predict-sample
```

## Comprehensive Example Notebook

For a comprehensive example demonstrating the full Winnow workflow, see our example notebook:

📓 **[FDR plots example notebook](https://github.com/instadeepai/winnow/blob/main/examples/getting_started_with_winnow.ipynb)**

This notebook shows you how to:

- Load and prepare PSM data using `CalibrationDataset`
- Train a confidence calibrator with `ProbabilityCalibrator`
- Apply FDR control using different methods
- Generate visualisation plots for calibration and FDR analysis
