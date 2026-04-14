from typing import List, Optional, Dict, Any
import warnings
import pandas as pd
import numpy as np
import koinapy

from winnow.calibration.features.base import CalibrationFeatures, FeatureDependency
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.calibration.features.utils import (
    require_beam_predictions,
    validate_model_input_params,
    resolve_model_inputs,
    format_intensity_prediction_outputs,
    compute_ion_identifications,
)
from winnow.utils.peptide import tokens_to_proforma


class ChimericFeatures(CalibrationFeatures):
    """Computes chimeric features for calibration.

    Computes spectrum match quality features for the runner-up (second-best) peptide prediction.
    Predicts ion intensities for runner-up (second-best) peptide sequences using a Koina intensity model.
    """

    def __init__(
        self,
        mz_tolerance: float,
        learn_from_missing: bool = True,
        intensity_model_name: str = "Prosit_2020_intensity_HCD",
        max_precursor_charge: int = 6,
        max_peptide_length: int = 30,
        unsupported_residues: Optional[List[str]] = None,
        model_input_constants: Optional[Dict[str, Any]] = None,
        model_input_columns: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize ChimericFeatures.

        Args:
            mz_tolerance (float): The mass-to-charge tolerance for ion matching.
            learn_from_missing (bool): When True, invalid runner-up predictions are recorded
                in an ``is_missing_chimeric_features`` indicator column and imputed with
                zeros, allowing the calibrator to learn from missingness. When False,
                invalid entries are silently filtered out with a warning. Defaults to True.
            intensity_model_name (str): The name of the Koina intensity model to use.
                Defaults to "Prosit_2020_intensity_HCD".
            max_precursor_charge (int): Maximum precursor charge accepted by the Koina
                intensity model. Spectra exceeding this are treated as missing. Defaults to 6.
            max_peptide_length (int): Maximum peptide length (residue token count) accepted
                by the Koina intensity model. Applied to the runner-up (second-best) predicted
                sequence, not the top-1 prediction. Runner-up sequences exceeding this are
                treated as missing. Defaults to 30.
            unsupported_residues (List[str]): Residues unsupported by the configured Koina
                intensity model, in ProForma format. Runner-up predictions containing any of
                these residues are excluded from model input and treated as missing. Defaults to an empty list.
            model_input_constants (Optional[Dict[str, Any]]): Mapping of Koina input name to a
                constant value that will be tiled across all rows (e.g. {"collision_energies": 25}).
                Defaults to None (no additional constants).
            model_input_columns (Optional[Dict[str, str]]): Mapping of Koina input name to a
                metadata column name that provides per-row values
                (e.g. {"collision_energies": "nce_col"}). Defaults to None.

        Raises:
            ValueError: If the same key appears in both model_input_constants and model_input_columns.
        """
        validate_model_input_params(model_input_constants, model_input_columns)
        self.mz_tolerance = mz_tolerance
        self.learn_from_missing = learn_from_missing
        self.unsupported_residues = (
            unsupported_residues if unsupported_residues is not None else []
        )
        self.intensity_model_name = intensity_model_name
        self.max_precursor_charge = max_precursor_charge
        self.max_peptide_length = max_peptide_length
        self.model_input_constants = model_input_constants
        self.model_input_columns = model_input_columns

    @property
    def dependencies(self) -> List[FeatureDependency]:
        """Returns a list of dependencies required before computing the feature.

        Since this feature does not depend on other features, it returns an empty list.

        Returns:
            List[FeatureDependency]: An empty list.
        """
        return []

    @property
    def name(self) -> str:
        """Returns the name of the feature.

        This method provides the natural language identifier used as the key for the feature.

        Returns:
            str: The feature name.
        """
        return "Chimeric Features"

    @property
    def columns(self) -> List[str]:
        """Returns the column names for the computed chimeric features.

        The columns mirror those of FragmentMatchFeatures but with "chimeric_" prefix,
        computed for the runner-up (second-best) peptide prediction:
            - Basic match metrics: chimeric_ion_matches, chimeric_ion_match_intensity
            - Ion coverage: chimeric_longest_b_series, chimeric_longest_y_series,
              chimeric_complementary_ion_count, chimeric_max_ion_gap,
              chimeric_b_y_intensity_ratio

        Returns:
            List[str]: A list of column names for all computed chimeric features,
                and optionally "is_missing_chimeric_features" if learn_from_missing is True.
        """
        columns = [
            # Basic match metrics
            "chimeric_ion_matches",
            "chimeric_ion_match_intensity",
            # Ion coverage features
            "chimeric_longest_b_series",
            "chimeric_longest_y_series",
            "chimeric_complementary_ion_count",
            "chimeric_max_ion_gap",
            "chimeric_b_y_intensity_ratio",
            "chimeric_spectral_angle",
        ]
        if self.learn_from_missing:
            columns.append("is_missing_chimeric_features")
        return columns

    def check_valid_chimeric_prediction(self, dataset: CalibrationDataset) -> pd.Series:
        """Check which predictions are valid for chimeric intensity prediction.

        A spectrum is considered invalid if any of the following conditions hold:
        - The beam search result has fewer than two sequences (no runner-up exists).
        - The precursor charge exceeds ``max_precursor_charge``.
        - The runner-up (second-best) peptide sequence has more than ``max_peptide_length``
          residue tokens.
        - The runner-up sequence contains a residue in ``unsupported_residues``.

        Args:
            dataset (CalibrationDataset): The dataset to check.

        Returns:
            pd.Series: A boolean Series aligned to dataset.metadata, where True indicates
                that the runner-up prediction satisfies all validity constraints and can be
                passed to the Koina intensity model.
        """
        filtered_dataset = (
            dataset.filter_entries(
                predictions_predicate=lambda beam: beam is None or len(beam) < 2
            )
            .filter_entries(
                metadata_predicate=lambda row: (
                    row["precursor_charge"] > self.max_precursor_charge
                )
            )
            .filter_entries(
                predictions_predicate=lambda beam: (
                    len(beam) > 1 and len(beam[1].sequence) > self.max_peptide_length
                )
            )
            .filter_entries(
                predictions_predicate=lambda beam: (
                    len(beam) > 1
                    and any(
                        token in beam[1].sequence for token in self.unsupported_residues
                    )
                )
            )
        )

        # Obtain valid indices
        valid_spectrum_ids = filtered_dataset.metadata["spectrum_id"]

        # Create boolean series indicating whether the runner-up prediction is valid
        is_valid_chimeric_prediction = pd.Series(
            dataset.metadata["spectrum_id"].isin(valid_spectrum_ids),
        )

        return is_valid_chimeric_prediction

    def prepare(self, dataset: CalibrationDataset) -> None:
        """Prepares the dataset before feature computation.

        This method is intended to perform any preprocessing required before computing the feature.

        Args:
            dataset (CalibrationDataset): The dataset to prepare.
        """
        return

    def compute(self, dataset: CalibrationDataset) -> None:
        """Computes chimeric features for the given dataset.

        Uses the Koina-hosted model to predict intensities for runner-up peptide sequences. The method processes predictions by sorting and grouping them, aligns ion match intensities and mass-to-charge ratios (m/z), and stores the results in the dataset metadata.

        Args:
            dataset (CalibrationDataset): The dataset containing metadata for predictions.
        """
        if "precursor_charge" not in dataset.metadata.columns:
            raise ValueError(
                "precursor_charge column not found in dataset. This is required for chimeric features computation."
            )

        # Ensure dataset.predictions is not None (beams required for runner-up sequences)
        require_beam_predictions(dataset, "ChimericFeatures")

        # Check which predictions are valid for Koina intensity prediction
        is_valid_chimeric_prediction = self.check_valid_chimeric_prediction(dataset)
        dataset.metadata["is_missing_chimeric_features"] = ~is_valid_chimeric_prediction

        if not self.learn_from_missing:
            # Filter invalid entries from the dataset in place so that they are dropped entirely
            # (not imputed with zeros) and downstream features also do not see them.
            n_invalid = (~is_valid_chimeric_prediction).sum()
            if n_invalid > 0:
                warnings.warn(
                    f"Filtered {n_invalid} spectra that do not satisfy the validity constraints "
                    f"for the Koina intensity model '{self.intensity_model_name}' "
                    f"(learn_from_missing=False). Constraints applied:\n"
                    f"  - Runner-up sequence required (beam search width >= 2)\n"
                    f"  - max_peptide_length={self.max_peptide_length} residue tokens (runner-up sequence)\n"
                    f"  - max_precursor_charge={self.max_precursor_charge}\n"
                    f"  - unsupported_residues: {self.unsupported_residues[:3]}{'...' if len(self.unsupported_residues) > 3 else ''}\n"
                    f"Set learn_from_missing=True to impute missing features instead of filtering.",
                    stacklevel=2,
                )
            _filtered = dataset.filter_entries(
                metadata_predicate=lambda row: row["is_missing_chimeric_features"]
            )
            dataset.metadata = _filtered.metadata
            dataset.predictions = _filtered.predictions
            # All remaining rows are valid — the reindex below will find every spectrum_id
            # in grouped_predictions with no NaN fill needed.

        original_indices = dataset.metadata.index

        # Filter out invalid spectra for Koina intensity prediction
        valid_chimeric_input = dataset.filter_entries(
            metadata_predicate=lambda row: row["is_missing_chimeric_features"]
        )

        # Prepare input data
        assert valid_chimeric_input.predictions is not None
        inputs = pd.DataFrame()
        inputs["peptide_sequences"] = np.array(
            [
                tokens_to_proforma(items[1].sequence)  # type: ignore
                for items in valid_chimeric_input.predictions
            ]
        )
        inputs["precursor_charges"] = np.array(
            valid_chimeric_input.metadata["precursor_charge"]
        )
        inputs.index = valid_chimeric_input.metadata["spectrum_id"]

        model = koinapy.Koina(self.intensity_model_name)
        inputs = resolve_model_inputs(
            inputs=inputs,
            metadata=valid_chimeric_input.metadata,
            required_model_inputs=model.model_inputs,
            auto_populated={"peptide_sequences", "precursor_charges"},
            constants=self.model_input_constants,
            columns=self.model_input_columns,
            model_name=self.intensity_model_name,
        )
        predictions_per_ion: pd.DataFrame = model.predict(inputs)

        predictions_per_peptide = format_intensity_prediction_outputs(
            predictions_per_ion
        )

        # Match computed metadata to valid spectra and impute missing values for invalid spectra
        # Reindex to match dataset.metadata.index and fill missing values with NaN
        dataset.metadata.index = dataset.metadata["spectrum_id"]
        dataset.metadata["runner_up_theoretical_mz"] = predictions_per_peptide[
            "mz"
        ].reindex(dataset.metadata["spectrum_id"], fill_value=np.nan)
        dataset.metadata["runner_up_theoretical_intensity"] = predictions_per_peptide[
            "intensities"
        ].reindex(dataset.metadata["spectrum_id"], fill_value=np.nan)
        dataset.metadata["runner_up_annotation"] = predictions_per_peptide[
            "annotation"
        ].reindex(dataset.metadata["spectrum_id"], fill_value=np.nan)

        # Revert to original indices
        dataset.metadata.index = original_indices

        # Store runner-up peptide sequences for length calculation
        assert dataset.predictions is not None
        runner_up_predictions = [
            beam[1].sequence if beam is not None and len(beam) > 1 else []
            for beam in dataset.predictions
        ]

        # Compute ion matches and match intensity
        # Zeros are returned for rows with missing Koina-predicted spectra
        (
            ion_matches,
            match_intensity,
            longest_b_series,
            longest_y_series,
            complementary_ion_count,
            max_ion_gap,
            b_y_intensity_ratio,
            spectral_angle,
        ) = compute_ion_identifications(
            dataset=dataset.metadata,
            source_mz_column="runner_up_theoretical_mz",
            source_annotation_column="runner_up_annotation",
            source_intensity_column="runner_up_theoretical_intensity",
            mz_tolerance=self.mz_tolerance,
            predictions=runner_up_predictions,
        )

        dataset.metadata["chimeric_ion_matches"] = ion_matches
        dataset.metadata["chimeric_ion_match_intensity"] = match_intensity
        dataset.metadata["chimeric_longest_b_series"] = longest_b_series
        dataset.metadata["chimeric_longest_y_series"] = longest_y_series
        dataset.metadata["chimeric_complementary_ion_count"] = complementary_ion_count
        dataset.metadata["chimeric_max_ion_gap"] = max_ion_gap
        dataset.metadata["chimeric_b_y_intensity_ratio"] = b_y_intensity_ratio
        dataset.metadata["chimeric_spectral_angle"] = spectral_angle
