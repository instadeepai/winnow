# FragmentMatchFeatures

Extracts features by comparing the observed fragmentation spectrum against a theoretical spectrum predicted by a Koina intensity model.

## Purpose

The quality of the match between observed and theoretical fragmentation patterns is a strong indicator of identification correctness. True identifications typically show:

- High fraction of predicted ions observed
- Intensity patterns matching theoretical predictions
- Consecutive ion series without gaps
- Low unexplained intensity

False identifications often show poor spectral agreement even when the de novo sequencer reports high confidence.

## Implementation

### Step 1: Theoretical Spectrum Generation

We call a Koina intensity prediction model (e.g., `Prosit_2020_intensity_HCD`) with:

- Predicted peptide sequence
- Precursor charge
- Collision energy (only required by some Koina models)
- Fragmentation type (only required by some Koina models)

The model returns:

- Theoretical m/z values for all possible b- and y-ions
- Predicted relative intensities for each ion
- Ion annotations (e.g., "b1", "y3", "b2+2" for doubly-charged)

### Step 2: Peak Matching

For each theoretical peak, we search for the nearest observed peak using binary search. A match is recorded if the m/z difference is within the configured tolerance (default: 0.02 Da). This produces a set of matched peaks containing:

- Theoretical m/z and intensity
- Observed intensity
- Ion annotation

## Columns

### Basic Match Metrics

| Column | Unit | Description |
| -------- | ------ | ------------- |
| `ion_matches` | Fraction (0-1) | Number of matched theoretical peaks / total theoretical peaks. A high number indicates presence of much of the predictied peptide's ion ladder in the observed spectrum. Low values suggest missing fragment coverage or an incorrect identification. |
| `ion_match_intensity` | Fraction (0-1) | Sum of observed intensities for matched peaks / total observed intensity, accounting for the isotopic envelope for four additional peaks. A high number indicates a prediction that explains most of the spectral evidence. A low number could indicate contamination, co-eluting peptides, or an incorrect identification. |

### Ion Coverage Features

| Column | Unit | Description |
| -------- | ------ | ------------- |
| `longest_b_series` | Count (integer) | Longest consecutive run of matched b-ions (e.g., b1, b2, b3 = 3). |
| `longest_y_series` | Count (integer) | Same as above for y-ions |
| `complementary_ion_count` | Count (integer) | Number of peptide bond positions where **both** the b-ion and complementary y-ion are matched. For a peptide of length n, bond position i produces b_i and y_(n-i). Finding both provides stronger evidence of a correct identification. |
| `max_ion_gap` | Daltons (Da) | Largest m/z difference between two consecutive matched theoretical peaks when sorted by m/z. Large gaps may indicate missing fragmentation coverage. |
| `b_y_intensity_ratio` | Ratio | Ratio of total matched b-ion intensity to total matched y-ion intensity (including isotopic envelopes). |

## Usage

```python
from winnow.calibration.features import FragmentMatchFeatures

feature = FragmentMatchFeatures(
    mz_tolerance=0.02,
    unsupported_residues=["N[UNIMOD:7]", "Q[UNIMOD:7]"],
    intensity_model_name="Prosit_2020_intensity_HCD",
    max_precursor_charge=6,
    max_peptide_length=30,
    model_input_constants={"collision_energies": 25},
    learn_from_missing=True,
)
calibrator.add_feature(feature)
```

### Parameters

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `mz_tolerance` | `float` | Required | Mass tolerance for peak matching in Daltons |
| `unsupported_residues` | `List[str]` | `[]` | Residue tokens not supported by the Koina model |
| `intensity_model_name` | `str` | `"Prosit_2020_intensity_HCD"` | Name of the Koina intensity model |
| `max_precursor_charge` | `int` | `6` | Maximum charge state supported by the model |
| `max_peptide_length` | `int` | `30` | Maximum peptide length supported by the model |
| `model_input_constants` | `Dict` | `{}` | Constant values for model inputs (e.g., collision energy) |
| `model_input_columns` | `Dict` | `{}` | Column names for per-row model inputs |
| `learn_from_missing` | `bool` | `True` | Whether to impute missing features or filter invalid rows |

## Requirements

The dataset must contain:

- `precursor_charge`: Precursor charge state
- `mz_array`: Observed m/z values (list per row)
- `intensity_array`: Observed intensities (list per row)
- `prediction`: Predicted peptide sequence tokens

For some Koina-hosted intensity prediction models, the dataset may also require:

- `collision_energies`: Kinetic energy used to fragment the peptide
- `fragmentation_types`: Method used to break the ions

## Notes

- Different Koina models have different constraints. See [configuration guide](../../configuration.md#koina-model-input-validation) for details.
- When `learn_from_missing=True`, invalid rows get zero feature values and an `is_missing_fragment_match_features` indicator column.
- `b_y_intensity_ratio` is computed as `b_total / (y_total + epsilon)` where epsilon is a small constant providing numerical stability when no y-ions are matched.
