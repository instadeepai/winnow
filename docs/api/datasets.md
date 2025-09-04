# Datasets API

The `winnow.datasets` module handles data loading and saving for peptide-spectrum match (PSM) data and provides structures for calibration workflows.

## Classes

### CalibrationDataset

The main class for storing and processing calibration datasets for peptide sequencing. It integrates metadata, predictions and spectra data to support various operations such as filtering, merging and evaluating peptide sequence predictions against spectra.

```python
from winnow.datasets import CalibrationDataset

# Load from CSV files
dataset = CalibrationDataset.from_predictions_csv(
    spectrum_path="path/to/spectrum_data.csv",
    beam_predictions_path="path/to/predictions.csv"
)

# Save dataset (winnow's internal format)
dataset.save(Path("output_directory"))
```

**Key Features:**

- **CSV Input**: Load predictions and spectra from CSV files
- **Data Integration**: Combines spectral data with prediction metadata
- **Filtering**: Removes invalid tokens and unsupported modifications
- **Evaluation**: Computes correctness labels when ground truth available

### PSMDataset

A dataset containing multiple peptide-spectrum matches (PSMs). Provides a container for managing collections of PSMs with iteration and indexing support.

```python
from winnow.datasets import PSMDataset, PeptideSpectrumMatch

# Create from separate sequences
dataset = PSMDataset.from_dataset(
    spectra=spectrum_list,
    peptides=peptide_list,
    confidence_scores=score_list
)

# Access PSMs
psm = dataset[0]  # Get first PSM
num_psms = len(dataset)  # Get total count

# Iterate over PSMs
for psm in dataset:
    print(f"Confidence: {psm.confidence}")
```

**Key Features:**

- **Collection Container**: Store and manage multiple PSMs
- **Standard Interface**: List-like indexing and iteration
- **Factory Methods**: Create from separate data arrays

### PeptideSpectrumMatch

Represents a single peptide-spectrum match returned by a peptide sequencing method. Associates a mass spectrum with a candidate peptide sequence and confidence score.

```python
from winnow.datasets import PeptideSpectrumMatch
from winnow.data_types import Spectrum, Peptide

# Create a PSM
psm = PeptideSpectrumMatch(
    spectrum=spectrum,
    peptide=peptide,
    confidence=0.95
)

# Compare PSMs by confidence
psm1 >= psm2  # True if psm1 has higher or equal confidence
```

**Key Features:**

- **Core PSM Representation**: Links spectrum, peptide and confidence
- **Confidence Comparison**: Built-in comparison operators
- **Type Safety**: Structured dataclass with defined fields

For detailed examples and usage patterns, refer to the [examples notebook](https://github.com/instadeepai/winnow/blob/main/examples/fdr_plots.ipynb).
