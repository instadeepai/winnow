from typing import List, Optional, Dict, Union
from sklearn.linear_model import LinearRegression
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file
import pandas as pd
import numpy as np
import warnings
import koinapy

from winnow.calibration.features.base import CalibrationFeatures, FeatureDependency
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.utils.peptide import tokens_to_proforma


class RetentionTimeFeature(CalibrationFeatures):
    """Computes iRT features using per-experiment linear regression.

    Uses a Koina iRT model to predict indexed retention times (iRT) for high-confidence
    peptides and trains a per-experiment linear regressor to map observed retention times
    to predicted iRT values. The regressor is always re-fitted at both training and
    inference time using self-supervised data (no database labels needed).
    """

    def __init__(
        self,
        train_fraction: float = 0.1,
        min_train_points: int = 10,
        learn_from_missing: bool = True,
        seed: int = 42,
        irt_model_name: str = "Prosit_2019_irt",
        max_peptide_length: int = 30,
        unsupported_residues: Optional[List[str]] = None,
    ) -> None:
        """Initialize RetentionTimeFeature.

        Args:
            train_fraction (float): Top fraction of spectra (by confidence, descending) used
                as training data for the RT->iRT regressor.
            min_train_points (int): Minimum number of high-confidence spectra needed per
                experiment to fit a regressor. If fewer are available after applying
                ``train_fraction``, a ``ValueError`` is raised.
            learn_from_missing (bool): When True, invalid predictions are recorded in an
                ``is_missing_irt_error`` indicator column and imputed with zeros, allowing
                the calibrator to learn from missingness. When False, invalid entries are
                silently filtered out with a warning. Defaults to True.
            seed (int): Random seed for reproducibility. Defaults to 42.
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
        self.min_train_points = min_train_points
        self.learn_from_missing = learn_from_missing
        self.seed = seed
        self.unsupported_residues = (
            unsupported_residues if unsupported_residues is not None else []
        )
        self.irt_model_name = irt_model_name
        self.max_peptide_length = max_peptide_length
        self.irt_predictors: Dict[str, LinearRegression] = {}

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
            metadata_predicate=lambda row: (
                len(row["prediction"]) > self.max_peptide_length
            )
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

    def save_regressors(self, path: Union[Path, str]) -> None:
        """Save fitted per-experiment regressors to a safetensors file.

        Each ``LinearRegression`` is stored as two tensors keyed by
        ``{experiment_name}/coef`` and ``{experiment_name}/intercept``.

        Args:
            path: File path for the output ``.safetensors`` file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors: Dict[str, torch.Tensor] = {}
        for exp_name, reg in self.irt_predictors.items():
            tensors[f"{exp_name}/coef"] = torch.as_tensor(
                reg.coef_, dtype=torch.float64
            )
            tensors[f"{exp_name}/intercept"] = torch.as_tensor(
                np.atleast_1d(reg.intercept_), dtype=torch.float64
            )
        save_file(tensors, path)

    def load_regressors(self, path: Union[Path, str]) -> None:
        """Load per-experiment regressors from a safetensors file.

        Loaded regressors are merged into ``self.irt_predictors``. Experiments
        already present are overwritten by the loaded values.

        Args:
            path: File path to the ``.safetensors`` file containing saved
                regressors.
        """
        tensors = load_file(Path(path))
        experiments: Dict[str, Dict[str, torch.Tensor]] = {}
        for key, tensor in tensors.items():
            exp_name, param = key.rsplit("/", 1)
            experiments.setdefault(exp_name, {})[param] = tensor

        for exp_name, params in experiments.items():
            reg = LinearRegression()
            reg.coef_ = params["coef"].numpy()
            reg.intercept_ = params["intercept"].numpy().item()
            self.irt_predictors[exp_name] = reg

    def prepare(self, dataset: CalibrationDataset) -> None:
        """Fit per-experiment RT->iRT linear regressors.

        For each experiment in the dataset (identified by the ``experiment_name`` column),
        fits a ``LinearRegression`` on the top ``train_fraction`` of spectra by confidence.
        Experiments that already have a regressor (e.g., loaded via ``load_regressors``)
        are skipped.

        If ``experiment_name`` is absent, a single global regressor is fitted with a
        warning.

        Args:
            dataset: The dataset containing peptide sequences and retention times.

        Raises:
            ValueError: If ``retention_time`` column is missing, or if any experiment
                has fewer than ``min_train_points`` valid training spectra.
        """
        if "retention_time" not in dataset.metadata.columns:
            raise ValueError(
                "retention_time column not found in dataset. "
                "This is required for iRT features computation."
            )

        if "experiment_name" not in dataset.metadata.columns:
            if "__global__" not in self.irt_predictors:
                warnings.warn(
                    "No 'experiment_name' column found. Fitting a single global "
                    "RT->iRT regressor. For multi-experiment data, ensure each "
                    "spectrum file includes an 'experiment_name' column or use "
                    "MGF format (which derives it from the filename).",
                    stacklevel=2,
                )
                experiments_to_fit = {"__global__": dataset.metadata}
            else:
                return
        else:
            experiments_to_fit = {
                str(exp_name): group
                for exp_name, group in dataset.metadata.groupby("experiment_name")
                if exp_name not in self.irt_predictors
            }
            if not experiments_to_fit:
                return

        # Select training data per experiment, validate counts, collect into
        # a single DataFrame for one batched Koina call.
        per_exp_train: Dict[str, pd.DataFrame] = {}
        for exp_name, group in experiments_to_fit.items():
            train_data = self._select_training_data(group, exp_name)
            per_exp_train[exp_name] = train_data

        all_train = pd.concat(per_exp_train.values(), ignore_index=True)

        inputs = pd.DataFrame()
        inputs["peptide_sequences"] = np.array(
            [tokens_to_proforma(peptide) for peptide in all_train["prediction"]]
        )

        koina_model = koinapy.Koina(self.irt_model_name)
        irt_predictions = koina_model.predict(inputs)
        all_irt = irt_predictions["irt"].values

        # Distribute iRT values back per experiment and fit regressors
        offset = 0
        for exp_name, train_data in per_exp_train.items():
            n = len(train_data)
            train_data = train_data.copy()
            train_data["irt"] = all_irt[offset : offset + n]
            offset += n

            x = train_data["retention_time"].values.reshape(-1, 1)
            y = train_data["irt"].values
            regressor = LinearRegression()
            regressor.fit(x, y)
            self.irt_predictors[exp_name] = regressor

    def compute(self, dataset: CalibrationDataset) -> None:
        """Compute the iRT error feature for each spectrum.

        Uses per-experiment regressors (fitted in ``prepare``) to predict iRT from
        observed retention time, calls the Koina iRT model for sequence-based iRT, and
        computes the absolute error between the two.

        Args:
            dataset: The dataset containing peptide sequences and retention times.
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

        koina_model = koinapy.Koina(self.irt_model_name)
        predictions = koina_model.predict(inputs)
        predictions["spectrum_id"] = predictions.index

        # Match computed metadata to valid spectra and impute missing values for invalid spectra
        # Reindex to match dataset.metadata.index and fill missing values with NaN
        dataset.metadata.index = dataset.metadata["spectrum_id"]
        dataset.metadata["irt"] = predictions["irt"].reindex(
            dataset.metadata["spectrum_id"], fill_value=np.nan
        )

        # Apply per-experiment regressors
        if "experiment_name" in dataset.metadata.columns:
            predicted_irt = pd.Series(np.nan, index=dataset.metadata.index)
            for exp_name, group in dataset.metadata.groupby("experiment_name"):
                regressor = self.irt_predictors[exp_name]
                predicted_irt.loc[group.index] = regressor.predict(
                    group["retention_time"].values.reshape(-1, 1)
                )
            dataset.metadata["predicted_irt"] = predicted_irt
        else:
            dataset.metadata["predicted_irt"] = self.irt_predictors[
                "__global__"
            ].predict(dataset.metadata["retention_time"].values.reshape(-1, 1))

        # Revert to original indices
        dataset.metadata.index = original_indices

        # Compute iRT error
        # Set zeros for rows where "irt" is missing
        dataset.metadata["irt_error"] = np.abs(
            dataset.metadata["predicted_irt"] - dataset.metadata["irt"]
        ).fillna(0.0)

    def _select_training_data(
        self, metadata: pd.DataFrame, experiment_name: str
    ) -> pd.DataFrame:
        """Select the top high-confidence spectra for regressor training.

        Filters to valid iRT predictions, sorts by confidence descending, and
        takes the top ``train_fraction``. Raises early if there are too few points.

        Args:
            metadata: DataFrame subset for one experiment (or the full dataset for global).
            experiment_name: Identifier for this experiment, used in error messages.

        Returns:
            A DataFrame subset of high-confidence training rows.

        Raises:
            ValueError: If the number of training points after applying
                ``train_fraction`` is fewer than ``min_train_points``.
        """
        dataset_copy = CalibrationDataset(
            metadata=metadata.copy(deep=True),
            predictions=None,
        )

        is_valid = self.check_valid_irt_prediction(dataset_copy)
        dataset_copy.metadata["is_missing_irt_error"] = ~is_valid
        valid_input = dataset_copy.filter_entries(
            metadata_predicate=lambda row: row["is_missing_irt_error"]
        )

        train_data = valid_input.metadata.sort_values(by="confidence", ascending=False)
        n_train = max(1, int(self.train_fraction * len(train_data)))
        train_data = train_data.iloc[:n_train]

        if len(train_data) < self.min_train_points:
            raise ValueError(
                f"Experiment '{experiment_name}': insufficient data for iRT "
                f"calibration. After applying train_fraction={self.train_fraction}, "
                f"only {len(train_data)} valid training points remain "
                f"(from {len(metadata)} total spectra), but "
                f"min_train_points={self.min_train_points} are required. "
                f"Adjust train_fraction, min_train_points, or provide more data."
            )

        return train_data
