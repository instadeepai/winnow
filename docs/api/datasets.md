# Datasets API

The `winnow.datasets` module handles data loading and saving for peptide-spectrum match (PSM) data and provides structures for calibration workflows.

## Classes

### CalibrationDataset

The main class for storing and processing calibration datasets for peptide sequencing. It integrates metadata, predictions and spectra data to support various operations such as filtering, merging and evaluating peptide sequence predictions against spectra.

```python
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.datasets.data_loaders import InstaNovoDatasetLoader, MZTabDatasetLoader, WinnowDatasetLoader
from pathlib import Path

# Load using InstaNovo data loader
loader = InstaNovoDatasetLoader()
dataset = loader.load(
    data_path=Path("spectrum_data.parquet"),
    predictions_path=Path("predictions.csv")
)

# Load using MZTab data loader
mztab_loader = MZTabDatasetLoader()
dataset = mztab_loader.load(
    data_path=Path("spectrum_data.parquet"),
    predictions_path=Path("predictions.mztab")
)

# Load previously saved dataset
winnow_loader = WinnowDatasetLoader()
dataset = winnow_loader.load(data_path=Path("saved_dataset_directory"))

# Save dataset (winnow's internal format)
dataset.save(Path("output_directory"))
```

**Key Features:**

- **Multiple Format Support**: Load data using specialized loaders for different file formats
- **Data Integration**: Combines spectral data with prediction metadata
- **Filtering**: Removes invalid tokens and unsupported modifications
- **Evaluation**: Computes correctness labels when ground truth available

### Data Loaders

The datasets module provides several data loaders that implement the `DatasetLoader` protocol:

#### InstaNovoDatasetLoader

Loads InstaNovo predictions from CSV format along with spectrum data from Parquet/IPC files.

```python
from winnow.datasets.data_loaders import InstaNovoDatasetLoader

loader = InstaNovoDatasetLoader()
dataset = loader.load(
    data_path=Path("spectrum_data.parquet"),  # Spectrum metadata
    predictions_path=Path("instanovo_predictions.csv")  # InstaNovo beam predictions
)
```

#### MZTabDatasetLoader

Loads predictions from MZTab format, supporting both traditional search engines and Casanovo outputs.

```python
from winnow.datasets.data_loaders import MZTabDatasetLoader

# Default loader (uses Casanovo residue mapping)
loader = MZTabDatasetLoader()
dataset = loader.load(
    data_path=Path("spectrum_data.parquet"),
    predictions_path=Path("search_results.mztab")
)

# Custom residue mapping
custom_mapping = {"M+15.995": "M[UNIMOD:35]"}
loader = MZTabDatasetLoader(residue_remapping=custom_mapping)
```

#### WinnowDatasetLoader

Loads previously saved CalibrationDataset instances from winnow's internal format.

```python
from winnow.datasets.data_loaders import WinnowDatasetLoader

loader = WinnowDatasetLoader()
dataset = loader.load(data_path=Path("saved_dataset_directory"))
```

### PSMDataset

A dataset containing multiple peptide-spectrum matches (PSMs). Provides a container for managing collections of PSMs with iteration and indexing support.

```python
from winnow.datasets.psm_dataset import PSMDataset, PeptideSpectrumMatch

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
from winnow.datasets.psm_dataset import PeptideSpectrumMatch
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
