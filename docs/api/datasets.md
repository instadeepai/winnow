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

InstaNovo and MZTab both inner-join spectrum rows and prediction rows on `spectrum_id`, but they derive that key differently.

##### Join and validation

Both loaders inner-join on `spectrum_id`:

- **Spectrum with no matching prediction** — dropped from the merged dataset (these are treated as spectra that did not survive filtering from the peptide sequencer).
- **Prediction with no matching spectrum row** — loading raises a `ValueError`.

Loading also raises a `ValueError` if:

- `spectrum_id` is missing from the predictions or spectrum data.
- `spectrum_id` values are not unique in the spectrum data.

MZTab additionally raises if `spectra_ref` cannot be parsed (expected form `ms_run[k]:index=N`).

##### InstaNovo: join on `spectrum_id`

The InstaNovo loader inner-joins the predictions CSV and the spectrum file on a shared `spectrum_id` column. Every prediction row must match exactly one spectrum row.

- The predictions CSV must include `spectrum_id`.
- The spectrum file must include the same `spectrum_id` values (unless you use `add_index_cols`; see below).

##### `add_index_cols` (InstaNovo)

Controls whether Winnow synthesises `experiment_name` and `spectrum_id` when loading InstaNovo parquet or IPC spectrum files. MGF inputs **always** receive these columns regardless of this setting. Configure via `data_loader.add_index_cols` in the loader YAML (see [configuration guide](../configuration.md#data-loader-configs)).

When `add_index_cols: true`:

- `experiment_name` is set to the spectrum file stem (e.g. `spectra` for `spectra.parquet`).
- If a `scan_number` column exists, `spectrum_id` is `{file_stem}:{scan_number}`.
- Otherwise, `spectrum_id` is `{file_stem}:{row_index}` (0-based row position in the file).

These columns are used directly as the join key (see above).

##### MZTab: join on `{experiment_name}:{index}`

Each row in the mzTab PSM table is one peptide-spectrum match (PSM). Two identifier columns matter for linking predictions to spectra:

- `spectra_ref`: which input spectrum the PSM belongs to (e.g. `ms_run[1]:index=123`). The integer after `index=` is the spectrum’s position in the **full** search input file (0-based). Spectra filtered out before mzTab export are absent from the file, so surviving PSMs can show gaps in `index` (e.g. 0, 2, 5). Every PSM for the same spectrum shares the same `spectra_ref`.
- `PSM_ID`: a monotonic, one-based row label within the mzTab file only (one per written PSM). Winnow does not use `PSM_ID` for matching; it groups and ranks candidates by the integer `index` parsed from `spectra_ref`.

The MZTab loader inner-joins spectrum rows and PSM rows on `spectrum_id` in `{experiment_name}:{index}` form, matching InstaNovo’s namespaced ID convention:

1. `experiment_name` is the spectrum file stem (`Path(data_path).stem`).
2. Winnow parses `index` from each PSM’s `spectra_ref` and sets `spectrum_id` to `{experiment_name}:{index}` (e.g. `spectra:123`). The integer `index` column is retained for beam grouping and sorting within each spectrum.
3. Spectrum rows receive `spectrum_id = {experiment_name}:{N}` where *N* is 0-based **file row order** in your spectrum input, not instrument scan number.
4. Any pre-existing `experiment_name` or `spectrum_id` columns in the input file are replaced.
5. Predictions and spectra are inner-joined on `spectrum_id`.

This is always applied for parquet, IPC, and MGF inputs; there is no config flag.

Use the **same** input file in the **same** order Casanovo indexed.

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
