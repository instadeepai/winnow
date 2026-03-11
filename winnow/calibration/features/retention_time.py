from typing import List, Optional
from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
import warnings
import koinapy

from winnow.calibration.features.base import CalibrationFeatures, FeatureDependency
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.utils.peptide import tokens_to_proforma


class RetentionTimeFeature(CalibrationFeatures):
    """Computes iRT features and calibrates an iRT predictor.

    Uses a Koina iRT model to predict indexed retention times (iRT) for peptides and trains a
    regression model to calibrate predictions based on observed retention times.
    """

    irt_predictor: MLPRegressor

    def __init__(
        self,
        hidden_dim: int,
        train_fraction: float,
        learn_from_missing: bool = True,
        seed: int = 42,
        learning_rate_init: float = 0.001,
        alpha: float = 0.0001,
        max_iter: int = 200,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        irt_model_name: str = "Prosit_2019_irt",
        max_peptide_length: int = 30,
        unsupported_residues: Optional[List[str]] = None,
    ) -> None:
        """Initialize RetentionTimeFeature.

        Args:
            hidden_dim (int): Hidden dimension size for the MLP regressor.
            train_fraction (float): Fraction of data to use for training the iRT calibrator.
            learn_from_missing (bool): When True, invalid predictions are recorded in an
                ``is_missing_irt_error`` indicator column and imputed with zeros, allowing
                the calibrator to learn from missingness. When False, invalid entries are
                silently filtered out with a warning. Defaults to True.
            seed (int): Random seed for the regressor. Defaults to 42.
            learning_rate_init (float): The initial learning rate. Defaults to 0.001.
            alpha (float): L2 regularisation parameter. Defaults to 0.0001.
            max_iter (int): Maximum number of training iterations. Defaults to 200.
            early_stopping (bool): Whether to use early stopping to terminate training. Defaults to False.
            validation_fraction (float): Proportion of training data to use for early stopping validation. Defaults to 0.1.
            irt_model_name (str): The name of the Koina iRT model to use.
                Defaults to "Prosit_2019_irt".
            max_peptide_length (int): Maximum peptide length (residue token count) accepted
                by the Koina iRT model. Predictions exceeding this are treated as missing.
                Defaults to 30.
            unsupported_residues (List[str]): Residues unsupported by the configured Koina iRT
                model, in ProForma format. Predictions containing any of these residues are
                excluded from model input and treated as missing. Defaults to an empty list.
        """
        self.train_fraction = train_fraction
        self.hidden_dim = hidden_dim
        self.learn_from_missing = learn_from_missing
        self.unsupported_residues = (
            unsupported_residues if unsupported_residues is not None else []
        )
        self.irt_model_name = irt_model_name
        self.max_peptide_length = max_peptide_length
        self.irt_predictor = MLPRegressor(
            hidden_layer_sizes=[hidden_dim],
            random_state=seed,
            learning_rate_init=learning_rate_init,
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
        )

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
            str: The feature identifier.
        """
        return "iRT Feature"

    @property
    def columns(self) -> List[str]:
        """Defines the column names for the computed features.

        Returns:
            List[str]: A list containing "irt_error" and optionally "is_missing_irt_error"
                if learn_from_missing is True.
        """
        columns = ["irt_error"]
        if self.learn_from_missing:
            columns.append("is_missing_irt_error")
        return columns

    def check_valid_irt_prediction(self, dataset: CalibrationDataset) -> pd.Series:
        """Check which predictions are valid for iRT prediction.

        A prediction is considered invalid if any of the following conditions hold:
        - The predicted peptide sequence has more than ``max_peptide_length`` residue tokens.
        - The predicted peptide sequence contains a residue in ``unsupported_residues``.

        Args:
            dataset (CalibrationDataset): The dataset to check.

        Returns:
            pd.Series: A boolean Series aligned to dataset.metadata, where True indicates
                that the prediction satisfies all validity constraints and can be passed to
                the Koina iRT model.
        """
        filtered_dataset = dataset.filter_entries(
            metadata_predicate=lambda row: len(row["prediction"])
            > self.max_peptide_length
        ).filter_entries(
            metadata_predicate=lambda row: any(
                token in row["prediction"] for token in self.unsupported_residues
            )
        )

        # Obtain valid indices
        valid_spectrum_ids = filtered_dataset.metadata["spectrum_id"]

        # Create boolean series indicating whether the prediction is valid for iRT prediction
        is_valid_irt_prediction = pd.Series(
            dataset.metadata["spectrum_id"].isin(valid_spectrum_ids),
        )

        return is_valid_irt_prediction

    def prepare(self, dataset: CalibrationDataset) -> None:
        """Prepares the dataset by training an iRT calibration model.

        This method:
        1. Selects high-confidence peptide sequences.
        2. Uses the configured Koina iRT model to predict iRT values for them.
        3. Trains an MLPRegressor to map observed retention times to predicted iRT values.

        Args:
            dataset (CalibrationDataset): The dataset containing peptide sequences and retention times.
        """
        if "retention_time" not in dataset.metadata.columns:
            raise ValueError(
                "retention_time column not found in dataset. This is required for iRT features computation."
            )

        # Create a copy of the dataset to avoid modifying the original
        dataset_copy = CalibrationDataset(
            metadata=dataset.metadata.copy(deep=True),
            predictions=dataset.predictions.copy() if dataset.predictions else None,
        )

        # Check which predictions are valid for iRT prediction
        is_valid_irt_prediction = self.check_valid_irt_prediction(dataset_copy)
        dataset_copy.metadata["is_missing_irt_error"] = ~is_valid_irt_prediction
        valid_irt_input = dataset_copy.filter_entries(
            metadata_predicate=lambda row: row["is_missing_irt_error"]
        )

        # Prepare training data
        # Select the most confident valid peptide identifications to create training labels
        train_data = valid_irt_input.metadata
        train_data = train_data.sort_values(by="confidence", ascending=False)
        train_data = train_data.iloc[: int(self.train_fraction * len(train_data))]

        inputs = pd.DataFrame()
        inputs["peptide_sequences"] = np.array(
            [tokens_to_proforma(peptide) for peptide in train_data["prediction"]]
        )
        inputs = inputs.set_index(train_data.index)

        prosit_model = koinapy.Koina(self.irt_model_name)
        predictions = prosit_model.predict(inputs)
        train_data["iRT"] = predictions["irt"]

        # -- Fit model
        x, y = train_data["retention_time"].values, train_data["iRT"].values
        self.irt_predictor.fit(x.reshape(-1, 1), y)

    def compute(self, dataset: CalibrationDataset) -> None:
        """Computes the iRT error by comparing observed retention times to predicted iRT values.

        This method:
        1. Uses the configured Koina iRT model to predict iRT values for all peptides.
        2. Uses the trained MLPRegressor to predict iRT values based on observed retention times.
        3. Computes the absolute error between predicted and actual iRT values.

        Args:
            dataset (CalibrationDataset): The dataset containing peptide sequences and retention times.
        """
        # Check which predictions are valid for iRT prediction
        is_valid_irt_prediction = self.check_valid_irt_prediction(dataset)
        dataset.metadata["is_missing_irt_error"] = ~is_valid_irt_prediction

        if not self.learn_from_missing:
            # Filter invalid entries from the dataset in place so that they are dropped entirely
            # (not imputed with zeros) and downstream features also do not see them.
            n_invalid = (~is_valid_irt_prediction).sum()
            if n_invalid > 0:
                warnings.warn(
                    f"Filtered {n_invalid} spectra that do not satisfy the validity constraints "
                    f"for the Koina iRT model '{self.irt_model_name}' "
                    f"(learn_from_missing=False). Constraints applied:\n"
                    f"  - Retention time data required\n"
                    f"  - max_peptide_length={self.max_peptide_length} residue tokens\n"
                    f"  - unsupported_residues: {self.unsupported_residues[:3]}{'...' if len(self.unsupported_residues) > 3 else ''}\n"
                    f"Set learn_from_missing=True to impute missing features instead of filtering.",
                    stacklevel=2,
                )
            _filtered = dataset.filter_entries(
                metadata_predicate=lambda row: row["is_missing_irt_error"]
            )
            dataset.metadata = _filtered.metadata
            dataset.predictions = _filtered.predictions
            # All remaining rows are valid — the reindex below will find every spectrum_id
            # in predictions with no NaN fill needed.

        original_indices = dataset.metadata.index

        # Filter out invalid spectra for iRT prediction
        valid_irt_input = dataset.filter_entries(
            metadata_predicate=lambda row: row["is_missing_irt_error"]
        )

        # Prepare input data
        inputs = pd.DataFrame()
        inputs["peptide_sequences"] = np.array(
            [
                tokens_to_proforma(peptide)
                for peptide in valid_irt_input.metadata["prediction"]
            ]
        )
        inputs.index = valid_irt_input.metadata["spectrum_id"]

        prosit_model = koinapy.Koina(self.irt_model_name)
        predictions = prosit_model.predict(inputs)
        predictions["spectrum_id"] = predictions.index

        # Match computed metadata to valid spectra and impute missing values for invalid spectra
        # Reindex to match dataset.metadata.index and fill missing values with NaN
        dataset.metadata.index = dataset.metadata["spectrum_id"]
        dataset.metadata["iRT"] = predictions["irt"].reindex(
            dataset.metadata["spectrum_id"], fill_value=np.nan
        )

        # Predict iRT using the trained MLPRegressor
        # Note that we will always obtain a predicted iRT value for each spectrum here,
        # even if the spectrum is invalid for iRT prediction, because we predict using observed retention time.
        dataset.metadata["predicted iRT"] = self.irt_predictor.predict(
            dataset.metadata["retention_time"].values.reshape(-1, 1)
        )

        # Revert to original indices
        dataset.metadata.index = original_indices

        # Compute iRT error
        # Set zeros for rows where "iRT" is missing
        dataset.metadata["irt_error"] = np.abs(
            dataset.metadata["predicted iRT"] - dataset.metadata["iRT"]
        ).fillna(0.0)
