from typing import List, Optional, Dict, Any
import warnings
import pandas as pd
import numpy as np
import koinapy

from winnow.calibration.features.base import CalibrationFeatures, FeatureDependency
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.calibration.features.utils import (
    validate_model_input_params,
    resolve_model_inputs,
    format_intensity_prediction_outputs,
    compute_ion_identifications,
)
from winnow.utils.peptide import tokens_to_proforma


class FragmentMatchFeatures(CalibrationFeatures):
    """Computes fragment ion match features between the observed spectrum and the theoretical spectrum.

    Uses a Koina intensity model to generate a theoretical fragmentation spectrum for the top-1
    de novo predicted peptide sequence, then computes how well that theoretical spectrum matches
    the observed spectrum.
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
        """Initialize FragmentMatchFeatures.

        Args:
            mz_tolerance (float): The mass-to-charge tolerance for ion matching.
            learn_from_missing (bool): When True, invalid predictions are recorded in an
                ``is_missing_fragment_match_features`` indicator column and imputed with
                zeros, allowing the calibrator to learn from missingness. When False,
                invalid entries are filtered out with a warning. Defaults to True.
            intensity_model_name (str): The name of the Koina intensity model to use.
                Defaults to "Prosit_2020_intensity_HCD".
            max_precursor_charge (int): Maximum precursor charge accepted by the Koina
                intensity model. Predictions exceeding this are treated as missing.
                Defaults to 6.
            max_peptide_length (int): Maximum peptide length (residue token count) accepted
                by the Koina intensity model. Predictions exceeding this are treated as
                missing. Defaults to 30.
            unsupported_residues (List[str]): Residues unsupported by the configured Koina
                intensity model, in ProForma format. Predictions containing any of these
                residues are excluded from model input and treated as missing.
                Defaults to an empty list.
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
        self.unsupported_residues = (
            unsupported_residues if unsupported_residues is not None else []
        )
        self.learn_from_missing = learn_from_missing
        self.intensity_model_name = intensity_model_name
        self.max_precursor_charge = max_precursor_charge
        self.max_peptide_length = max_peptide_length
        self.model_input_constants = model_input_constants
        self.model_input_columns = model_input_columns

    @property
    def dependencies(self) -> List[FeatureDependency]:
        """Returns the list of feature dependencies for FragmentMatchFeatures.

        Since no dependencies are required, this method returns an empty list.

        Returns:
            List[FeatureDependency]: An empty list, as there are no dependencies for this feature.
        """
        return []

    @property
    def name(self) -> str:
        """Returns the name of the feature.

        This method provides the natural language identifier used as the key for the feature.

        Returns:
            str: The feature identifier, "Fragment Match Features".
        """
        return "Fragment Match Features"

    @property
    def columns(self) -> List[str]:
        """Returns the columns associated with the fragment match features.

        The columns include:
            - Basic match metrics: ion_matches, ion_match_intensity
            - Ion coverage: longest_b_series, longest_y_series,
              complementary_ion_count, max_ion_gap, b_y_intensity_ratio

        Returns:
            List[str]: A list of column names for all computed features, and optionally
                "is_missing_fragment_match_features" if learn_from_missing is True.
        """
        columns = [
            # Basic match metrics
            "ion_matches",
            "ion_match_intensity",
            # Ion coverage features
            "longest_b_series",
            "longest_y_series",
            "complementary_ion_count",
            "max_ion_gap",
            "b_y_intensity_ratio",
            "spectral_angle",
        ]
        if self.learn_from_missing:
            columns.append("is_missing_fragment_match_features")
        return columns

    def check_valid_prediction(self, dataset: CalibrationDataset) -> pd.Series:
        """Check which predictions are valid for intensity prediction.

        A prediction is considered invalid if any of the following conditions hold:
        - The precursor charge exceeds ``max_precursor_charge``.
        - The predicted peptide sequence has more than ``max_peptide_length`` residue tokens.
        - The predicted peptide sequence contains a residue in ``unsupported_residues``.

        Args:
            dataset (CalibrationDataset): The dataset to check.

        Returns:
            pd.Series: A boolean Series aligned to dataset.metadata, where True indicates
                that the prediction satisfies all validity constraints and can be passed to
                the Koina intensity model.
        """
        filtered_dataset = (
            dataset.filter_entries(
                metadata_predicate=lambda row: row["precursor_charge"]
                > self.max_precursor_charge
            )
            .filter_entries(
                metadata_predicate=lambda row: len(row["prediction"])
                > self.max_peptide_length
            )
            .filter_entries(
                metadata_predicate=lambda row: any(
                    token in row["prediction"] for token in self.unsupported_residues
                )
            )
        )

        # Obtain valid indices
        valid_spectrum_ids = filtered_dataset.metadata["spectrum_id"]

        # Create boolean series indicating whether the prediction is valid for intensity prediction
        is_valid_prediction = pd.Series(
            dataset.metadata["spectrum_id"].isin(valid_spectrum_ids),
        )

        return is_valid_prediction

    def prepare(self, dataset: CalibrationDataset) -> None:
        """Prepares the dataset for feature computation.

        This method is intended to perform any preprocessing required before computing the feature.

        Args:
            dataset (CalibrationDataset): The dataset to prepare.
        """
        return

    def compute(self, dataset: CalibrationDataset) -> None:
        """Computes fragment match features for the given dataset.

        Uses the configured Koina model to predict theoretical spectra based on peptide sequences, precursor charges and other metadata.
        Processes and sorts predictions, aligns ion match intensities and mass-to-charge ratios (m/z), and stores the results in the dataset metadata.

        Args:
            dataset (CalibrationDataset): The dataset containing metadata required for predictions.
        """
        if "precursor_charge" not in dataset.metadata.columns:
            raise ValueError(
                "precursor_charge column not found in dataset. This is required for fragment match features computation."
            )

        # Check which predictions are valid for intensity prediction
        is_valid_prediction = self.check_valid_prediction(dataset)
        dataset.metadata["is_missing_fragment_match_features"] = ~is_valid_prediction

        if not self.learn_from_missing:
            # Filter invalid entries from the dataset in place so that they are dropped entirely
            # (not imputed with zeros) and downstream features also do not see them.
            n_invalid = (~is_valid_prediction).sum()
            if n_invalid > 0:
                warnings.warn(
                    f"Filtered {n_invalid} spectra that do not satisfy the validity constraints "
                    f"for the Koina intensity model '{self.intensity_model_name}' "
                    f"(learn_from_missing=False). Constraints applied:\n"
                    f"  - max_peptide_length={self.max_peptide_length} residue tokens\n"
                    f"  - max_precursor_charge={self.max_precursor_charge}\n"
                    f"  - unsupported_residues: {self.unsupported_residues[:3]}{'...' if len(self.unsupported_residues) > 3 else ''}\n"
                    f"Set learn_from_missing=True to impute missing features instead of filtering.",
                    stacklevel=2,
                )
            _filtered = dataset.filter_entries(
                metadata_predicate=lambda row: row["is_missing_fragment_match_features"]
            )
            # Mutate in-place so the caller sees the reduced dataset.  A plain
            # ``dataset = …`` rebind would only update the local name, leaving the
            # caller's object unchanged and preventing subsequent features from
            # seeing the filtered-out rows.
            dataset.metadata = _filtered.metadata
            dataset.predictions = _filtered.predictions
            # All remaining rows are valid — the reindex below will find every spectrum_id
            # in grouped_predictions with no NaN fill needed.

        original_indices = dataset.metadata.index

        # Filter out invalid spectra for intensity prediction
        valid_input = dataset.filter_entries(
            metadata_predicate=lambda row: row["is_missing_fragment_match_features"]
        )

        # Prepare input data
        inputs = pd.DataFrame()
        inputs["peptide_sequences"] = np.array(
            [
                tokens_to_proforma(peptide)
                for peptide in valid_input.metadata["prediction"]
            ]
        )
        inputs["precursor_charges"] = np.array(valid_input.metadata["precursor_charge"])
        inputs.index = valid_input.metadata["spectrum_id"]

        model = koinapy.Koina(self.intensity_model_name)

        # Append any additional input columns required by the model
        inputs = resolve_model_inputs(
            inputs=inputs,
            metadata=valid_input.metadata,
            required_model_inputs=model.model_inputs,
            auto_populated={"peptide_sequences", "precursor_charges"},
            constants=self.model_input_constants,
            columns=self.model_input_columns,
            model_name=self.intensity_model_name,
        )
        predictions_per_ion: pd.DataFrame = model.predict(inputs)
        # This output returns one row per predicted ion, so we need to group by spectrum_id to get one row per peptide.

        predictions_per_peptide = format_intensity_prediction_outputs(
            predictions_per_ion
        )

        # Match computed metadata to valid spectra and impute missing values for invalid spectra
        # i.e., if is_missing_fragment_match_features is True, then theoretical_mz and theoretical_intensity are NaN
        dataset.metadata.index = dataset.metadata["spectrum_id"]
        dataset.metadata["theoretical_mz"] = predictions_per_peptide["mz"].reindex(
            dataset.metadata["spectrum_id"],
            fill_value=np.nan,
        )
        dataset.metadata["theoretical_intensity"] = predictions_per_peptide[
            "intensities"
        ].reindex(dataset.metadata["spectrum_id"], fill_value=np.nan)
        dataset.metadata["theoretical_annotation"] = predictions_per_peptide[
            "annotation"
        ].reindex(dataset.metadata["spectrum_id"], fill_value=np.nan)

        # Revert to original indices
        dataset.metadata.index = original_indices

        # Compute ion matches and match intensity
        # Zeros are returned for rows with missing theoretical spectra
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
            source_mz_column="theoretical_mz",
            source_annotation_column="theoretical_annotation",
            source_intensity_column="theoretical_intensity",
            mz_tolerance=self.mz_tolerance,
        )

        dataset.metadata["ion_matches"] = ion_matches
        dataset.metadata["ion_match_intensity"] = match_intensity
        dataset.metadata["longest_b_series"] = longest_b_series
        dataset.metadata["longest_y_series"] = longest_y_series
        dataset.metadata["complementary_ion_count"] = complementary_ion_count
        dataset.metadata["max_ion_gap"] = max_ion_gap
        dataset.metadata["b_y_intensity_ratio"] = b_y_intensity_ratio
        dataset.metadata["spectral_angle"] = spectral_angle
