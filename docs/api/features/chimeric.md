# ChimericFeatures

Computes spectrum match quality features for the runner-up (second-best) peptide prediction, helping to detect chimeric spectra where multiple peptides co-elute.

## Purpose

Chimeric spectra occur when two or more peptides are fragmented together in the same MS/MS scan. In these cases:

- The top-1 prediction may explain only part of the observed peaks
- The runner-up prediction may also match unexplained peaks
- Both peptides contribute to the total spectrum intensity

By computing spectrum match features for the second-best prediction, the calibrator can identify cases where:

- The top prediction has high confidence but the runner-up also matches well (possible chimera)
- The top prediction poorly explains the spectrum but the runner-up provides a better match
- Competition between candidates suggests uncertainty

## Implementation

This feature mirrors all computations from [FragmentMatchFeatures](fragment_match.md) but applied to the **runner-up** (second-best) peptide sequence from beam search:

1. Extract the second sequence from beam predictions (`dataset.predictions[i][1]`)
2. Call Koina intensity model for the runner-up sequence
3. Match theoretical peaks to observed spectrum
4. Compute all spectrum match quality features

All column names are prefixed with `chimeric_` to distinguish from top-1 features.

## Columns

All columns from `FragmentMatchFeatures` with `chimeric_` prefix:

### Basic Match Metrics

| Column | Unit | Description |
| -------- | ------ | ------------- |
| `chimeric_ion_matches` | Fraction (0-1) | Fraction of runner-up theoretical ions matched |
| `chimeric_ion_match_intensity` | Fraction (0-1) | Observed intensity explained by runner-up |

### Ion Coverage Features

| Column | Unit | Description |
| -------- | ------ | ------------- |
| `chimeric_longest_b_series` | Count (integer) | Longest consecutive b-ion run for runner-up |
| `chimeric_longest_y_series` | Count (integer) | Longest consecutive y-ion run for runner-up |
| `chimeric_complementary_ion_count` | Count (integer) | Bond positions with both b and y ions for runner-up |
| `chimeric_max_ion_gap` | Daltons (Da) | Largest gap between matched runner-up ions |


```python
from winnow.calibration.features import ChimericFeatures

feature = ChimericFeatures(
    mz_tolerance_ppm=20,
    unsupported_residues=["N[UNIMOD:7]", "Q[UNIMOD:7]"],
    max_precursor_charge=6,
    max_peptide_length=30,
    model_input_constants={"collision_energies": 25, "fragmentation_types": "HCD"},
    learn_from_missing=True,
)
calibrator.add_feature(feature)
```

### Parameters

Exactly one of `mz_tolerance_ppm` or `mz_tolerance_da` must be provided.

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `mz_tolerance_ppm` | `Optional[float]` | `None` | Relative tolerance in parts per million. The absolute tolerance per ion is `query_mz * ppm / 1e6`. |
| `mz_tolerance_da` | `Optional[float]` | `None` | Absolute tolerance in Daltons, applied uniformly to all ions. |
| `unsupported_residues` | `List[str]` | `[]` | Residue tokens not supported by the Koina model |
| `intensity_model_name` | `str` | `"Prosit_2020_intensity_HCD"` | Name of the Koina intensity model |
| `max_precursor_charge` | `int` | `6` | Maximum charge state supported by the model |
| `max_peptide_length` | `int` | `30` | Maximum peptide length (applied to runner-up) |
| `model_input_constants` | `Dict` | `{}` | Constant values for model inputs |
| `model_input_columns` | `Dict` | `{}` | Column names for per-row model inputs |
| `learn_from_missing` | `bool` | `True` | Whether to impute missing features or filter invalid rows |

## Requirements

The dataset must have:

- Beam predictions with at least 2 sequences (`dataset.predictions[i]` must have length ≥ 2)
- `precursor_charge`: Precursor charge state
- `mz_array`: Observed m/z values
- `intensity_array`: Observed intensities

For some Koina-hosted intensity prediction models, the dataset may also require:

- `collision_energies`: Kinetic energy used to fragment the peptide
- `fragmentation_types`: Method used to break the ions

## Notes

- Requires beam predictions; raises `ValueError` if `dataset.predictions` is `None`
- Spectra with only one beam result (no runner-up) are treated as invalid
- When `learn_from_missing=True`, invalid rows get zero feature values and an `is_missing_chimeric_features` indicator column
- The runner-up validation constraints (length, charge, residues) are applied to the second-best sequence, not the top-1
- Consider using both `FragmentMatchFeatures` and `ChimericFeatures` together to give the calibrator information about both top-1 and runner-up matches
