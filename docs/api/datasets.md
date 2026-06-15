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

# Save dataset (Winnow's internal format)
dataset.save(Path("output_directory"))
```

**Key Features:**

- **Multiple Format Support**: Load data using specialised loaders for different file formats
- **Data Integration**: Combines spectral data with prediction metadata
- **Filtering**: Removes invalid tokens and unsupported modifications
- **Evaluation**: Computes correctness labels when ground truth available

### Data loaders

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

Loads previously saved CalibrationDataset instances from Winnow's internal format.

```python
from winnow.datasets.data_loaders import WinnowDatasetLoader

loader = WinnowDatasetLoader()
dataset = loader.load(data_path=Path("saved_dataset_directory"))
```

#### Spectrum and prediction matching

InstaNovo and MZTab use **different join keys** to pair spectrum rows with prediction rows.

##### InstaNovo: join on `spectrum_id`

The InstaNovo loader inner-joins the predictions CSV and the spectrum file on a shared `spectrum_id` column. Every prediction row must match exactly one spectrum row.

- The predictions CSV must include `spectrum_id`.
- The spectrum file must include the same `spectrum_id` values (unless you use `add_index_cols`; see below).
- After merging, the row count must equal the number of prediction rows. If any prediction has no matching spectrum, or if `spectrum_id` is duplicated in the spectrum file, loading raises a `ValueError`.

InstaNovo predictions from an MGF file use **0-based spectrum position** in the MGF, not the instrument scan number. If your parquet uses `experiment_name:scan_number` IDs but the CSV uses `{stem}:0`, `{stem}:1`, ... you must re-key the parquet (see `scripts/rekey_split_parquet_spectrum_ids.py`) or set `add_index_cols` appropriately.

##### `add_index_cols` (InstaNovo and MZTab)

Controls whether Winnow synthesises `experiment_name` and `spectrum_id` when loading parquet or IPC spectrum files. MGF inputs **always** receive these columns regardless of this setting. Configure via `data_loader.add_index_cols` in the loader YAML (see [configuration guide](../configuration.md#data-loader-configs)).

When `add_index_cols: true`:

- `experiment_name` is set to the spectrum file stem (e.g. `spectra` for `spectra.parquet`).
- If a `scan_number` column exists, `spectrum_id` is `{file_stem}:{scan_number}`.
- Otherwise, `spectrum_id` is `{file_stem}:{row_index}` (0-based row position in the file).

##### MZTab: join on row index from `spectra_ref`

The MZTab loader does **not** use `spectrum_id` to match predictions. Instead:

1. Each PSM row in the `.mztab` file provides `spectra_ref` (e.g. `ms_run[1]:index=123`).
2. Winnow extracts the integer after `index=` as the join key.
3. Spectrum rows are numbered 0, 1, 2, … in **file order** via a row index.
4. Predictions and spectra are inner-joined on that numeric index.

So row 0 in your parquet/IPC/MGF must correspond to `index=0` in the mzTab file, row 1 to `index=1`, and so on. Spectra with no matching prediction (or vice versa) are dropped.

A pre-existing `spectrum_id` column (e.g. `{file_stem}:{scan_number}`) is preserved in the output metadata but **ignored for matching**. With `add_index_cols: false`, Winnow does not add or overwrite identifier columns in parquet/IPC files.

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

For detailed examples and usage patterns, refer to the [examples notebook](https://github.com/instadeepai/winnow/blob/main/examples/getting_started_with_winnow.ipynb).
