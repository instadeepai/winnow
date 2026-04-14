# Calibration Features

The `winnow.calibration.features` module provides a modular feature extraction system for PSM confidence calibration. Features are computed from peptide-spectrum match data and used by the `ProbabilityCalibrator` to transform raw confidence scores into calibrated probabilities.

## Quick Reference

| Feature | Description | Requires |
| --------- | ------------- | ---------- |
| [MassErrorFeature](mass_error.md) | Precursor mass accuracy | `precursor_mz`, `precursor_charge` |
| [BeamFeatures](beam.md) | Beam search diversity metrics | Beam predictions |
| [FragmentMatchFeatures](fragment_match.md) | Theoretical vs observed spectrum agreement | `precursor_charge`, spectrum data |
| [ChimericFeatures](chimeric.md) | Runner-up peptide spectrum match | Beam predictions, spectrum data |
| [RetentionTimeFeature](retention_time.md) | Chromatographic retention time error | `retention_time` |
| [SequenceFeatures](sequence.md) | Peptide sequence properties | `prediction`, `precursor_charge` |
| [TokenScoreFeatures](token_score.md) | Position-level confidence metrics | Beam predictions with token log-probs |

## CalibrationFeatures Base Class

All features inherit from `CalibrationFeatures` and implement a common interface:

```python
from winnow.calibration.features import CalibrationFeatures, FeatureDependency
from winnow.datasets.calibration_dataset import CalibrationDataset
from typing import List

class CustomFeature(CalibrationFeatures):
    @property
    def name(self) -> str:
        """Human-readable name for the feature."""
        return "My Custom Feature"

    @property
    def columns(self) -> List[str]:
        """Column names that will be added to dataset.metadata."""
        return ["custom_feature_1", "custom_feature_2"]

    @property
    def dependencies(self) -> List[FeatureDependency]:
        """Other features/computations that must run first."""
        return []

    def prepare(self, dataset: CalibrationDataset) -> None:
        """One-time setup before compute (e.g., model training)."""
        pass

    def compute(self, dataset: CalibrationDataset) -> None:
        """Compute and add feature columns to dataset.metadata."""
        dataset.metadata["custom_feature_1"] = computed_values_1
        dataset.metadata["custom_feature_2"] = computed_values_2
```

### Key Methods

| Method | Description |
| -------- | ------------- |
| `name` | Property returning a human-readable identifier |
| `columns` | Property returning list of column names this feature produces |
| `dependencies` | Property returning list of `FeatureDependency` objects |
| `prepare(dataset)` | Called once during ProbabilityCalibrator training, useful for feature-specific model training |
| `compute(dataset)` | Computes features and adds columns to `dataset.metadata` |

## Feature Dependencies

Features can declare dependencies on shared computations using `FeatureDependency`. This enables:

- **Shared computation**: Dependencies are computed once and reused
- **Reference counting**: Automatic cleanup when no longer needed
- **Ordered execution**: Dependencies always run before dependent features

```python
from winnow.calibration.features import FeatureDependency

class TheoreticalSpectrumDependency(FeatureDependency):
    """Example dependency that computes theoretical spectra once."""

    def compute(self, dataset: CalibrationDataset) -> None:
        # Expensive computation done once
        dataset.metadata["theoretical_mz"] = compute_spectra(...)

    def cleanup(self, dataset: CalibrationDataset) -> None:
        # Remove intermediate columns when no longer needed
        del dataset.metadata["theoretical_mz"]
```

The `ProbabilityCalibrator` automatically handles dependency resolution:

1. Collects all dependencies from added features
2. Computes each unique dependency once (reference counted)
3. Executes feature computations in correct order
4. Cleans up dependencies when reference count reaches zero

## Handling Missing Features

Koina-dependent features (`FragmentMatchFeatures`, `ChimericFeatures`, `RetentionTimeFeature`) may not be computable for all peptides due to model-specific constraints:

- Peptides exceeding the model's maximum length
- Precursor charges exceeding the model's maximum
- Unsupported modifications or residue types
- Lack of runner-up sequences for chimeric features

Winnow provides two strategies controlled by the `learn_from_missing` parameter:

### Filter Strategy (`learn_from_missing=False`, default)

Invalid PSMs are removed from the dataset before feature computation.

```python
from winnow.calibration.features import FragmentMatchFeatures

feature = FragmentMatchFeatures(
    mz_tolerance=0.02,
    learn_from_missing=False,  # Default
    max_peptide_length=30,
    max_precursor_charge=6,
)
```

**Behaviour:**

- Invalid rows are automatically filtered before Koina is called
- A `UserWarning` is emitted reporting how many PSMs were removed
- Filtered PSMs are gone entirely; no indicator column is added
- Calibrator trains only on remaining clean data

**Use when:** You want strict data quality and don't mind losing some PSMs.

### Learn Strategy (`learn_from_missing=True`)

Invalid PSMs are retained with imputed feature values and an indicator column.

```python
feature = FragmentMatchFeatures(
    mz_tolerance=0.02,
    learn_from_missing=True,
    max_peptide_length=30,
    max_precursor_charge=6,
)
```

**Behaviour:**

- All rows are retained in the dataset
- Invalid rows get zero/default feature values
- An `is_missing_*` indicator column is added (e.g., `is_missing_fragment_match_features`)
- Calibrator can learn patterns associated with missing data

**Use when:** You want to maximise recall and let the model learn from missingness patterns.

### Configuration

The defaults match Prosit model constraints. Adjust for other Koina models:

```python
# Example for a model with different constraints
feature = FragmentMatchFeatures(
    mz_tolerance=0.02,
    max_peptide_length=50,           # Model supports longer peptides
    max_precursor_charge=8,          # Model supports higher charges
    unsupported_residues=["U", "O"], # Selenocysteine and pyrrolysine
)
```

See the [configuration guide](../../configuration.md#koina-model-input-validation) for details on model-specific constraints.
