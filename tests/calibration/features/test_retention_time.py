"""Unit tests for winnow calibration feature RetentionTimeFeature."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from sklearn.linear_model import LinearRegression
import warnings

from winnow.calibration.features.retention_time import RetentionTimeFeature
from winnow.datasets.calibration_dataset import CalibrationDataset


class TestRetentionTimeFeature:
    """Test the RetentionTimeFeature class."""

    @pytest.fixture()
    def retention_time_feature(self):
        """Create a RetentionTimeFeature instance for testing."""
        return RetentionTimeFeature(
            train_fraction=0.8,
            min_train_points=2,
            unsupported_residues=["U", "O", "X"],
        )

    @pytest.fixture()
    def sample_dataset_with_rt(self):
        """Create a sample dataset with retention time data."""
        metadata = pd.DataFrame(
            {
                "confidence": [0.95, 0.90, 0.85, 0.80, 0.75],
                "prediction": [["A", "G"], ["G", "A"], ["S", "P"], ["V"], ["K"]],
                "prediction_untokenised": ["AG", "GA", "SP", "V", "K"],
                "retention_time": [10.5, 15.2, 20.1, 8.7, 12.3],
                "precursor_charge": [2, 2, 3, 1, 2],
                "spectrum_id": [0, 1, 2, 3, 4],
            }
        )
        return CalibrationDataset(metadata=metadata, predictions=None)

    def test_properties(self, retention_time_feature):
        """Test RetentionTimeFeature properties."""
        assert retention_time_feature.name == "iRT Feature"
        assert retention_time_feature.columns == ["irt_error", "is_missing_irt_error"]
        assert retention_time_feature.dependencies == []

    def test_initialization_parameters(self):
        """Test initialization with custom parameters."""
        feature = RetentionTimeFeature(
            train_fraction=0.8,
            min_train_points=5,
            unsupported_residues=["U", "O", "X"],
        )
        assert feature.train_fraction == 0.8
        assert feature.min_train_points == 5
        assert feature.irt_model_name == "Prosit_2019_irt"
        assert isinstance(feature.irt_predictors, dict)
        assert len(feature.irt_predictors) == 0

    @patch("winnow.calibration.features.retention_time.koinapy.Koina")
    def test_prepare_global_fallback(
        self, mock_koina, retention_time_feature, sample_dataset_with_rt
    ):
        """Test prepare fits a global regressor when experiment_name is absent."""
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.model_inputs = ["peptide_sequences"]
        mock_model_instance.predict.return_value = pd.DataFrame(
            {"irt": [35.1, 20.7, 28.1, 25.5]}
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            retention_time_feature.prepare(sample_dataset_with_rt)
            assert any("experiment_name" in str(warning.message) for warning in w)

        assert "__global__" in retention_time_feature.irt_predictors
        assert len(retention_time_feature.irt_predictors) == 1
        mock_model_instance.predict.assert_called_once()

    @patch("winnow.calibration.features.retention_time.koinapy.Koina")
    def test_prepare_per_experiment(self, mock_koina):
        """Test prepare fits separate regressors for each experiment."""
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.model_inputs = ["peptide_sequences"]
        mock_model_instance.predict.return_value = pd.DataFrame(
            {"irt": [35.1, 20.7, 12.4, 5.3]}
        )

        metadata = pd.DataFrame(
            {
                "confidence": [0.95, 0.90, 0.85, 0.80],
                "prediction": [["A", "G"], ["G", "A"], ["S", "P"], ["V"]],
                "retention_time": [10.5, 15.2, 20.1, 8.7],
                "spectrum_id": [0, 1, 2, 3],
                "experiment_name": ["exp_a", "exp_a", "exp_b", "exp_b"],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        feature = RetentionTimeFeature(train_fraction=1.0, min_train_points=2)
        feature.prepare(dataset)

        assert "exp_a" in feature.irt_predictors
        assert "exp_b" in feature.irt_predictors
        assert "__global__" not in feature.irt_predictors
        assert len(feature.irt_predictors) == 2

    @patch("winnow.calibration.features.retention_time.koinapy.Koina")
    def test_prepare_skips_preloaded_experiments(self, mock_koina):
        """Test that prepare skips experiments with pre-loaded regressors."""
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.model_inputs = ["peptide_sequences"]
        mock_model_instance.predict.return_value = pd.DataFrame({"irt": [35.1, 20.7]})

        metadata = pd.DataFrame(
            {
                "confidence": [0.95, 0.90, 0.85, 0.80],
                "prediction": [["A", "G"], ["G", "A"], ["S", "P"], ["V"]],
                "retention_time": [10.5, 15.2, 20.1, 8.7],
                "spectrum_id": [0, 1, 2, 3],
                "experiment_name": ["exp_a", "exp_a", "exp_b", "exp_b"],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        preloaded = LinearRegression()
        preloaded.fit([[1], [2]], [10, 20])

        feature = RetentionTimeFeature(train_fraction=1.0, min_train_points=2)
        feature.irt_predictors["exp_a"] = preloaded
        feature.prepare(dataset)

        # exp_a should still be the preloaded regressor
        assert feature.irt_predictors["exp_a"] is preloaded
        # exp_b should have been fitted fresh
        assert "exp_b" in feature.irt_predictors
        assert feature.irt_predictors["exp_b"] is not preloaded

    @patch("winnow.calibration.features.retention_time.koinapy.Koina")
    def test_prepare_warns_on_insufficient_data(self, mock_koina):
        """Test that prepare warns and skips when min_train_points is not met."""
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.model_inputs = ["peptide_sequences"]

        predictions = [
            ["A", "G"],
            ["G", "A"],
            ["L", "K"],
            ["M", "V"],
            ["P", "S"],
            ["T", "W"],
            ["Y", "H"],
            ["R", "N"],
            ["D", "E"],
            ["F", "I"],
        ]
        metadata = pd.DataFrame(
            {
                "confidence": [0.95 - 0.01 * i for i in range(len(predictions))],
                "prediction": predictions,
                "retention_time": [10.5 + i * 2.1 for i in range(len(predictions))],
                "spectrum_id": list(range(len(predictions))),
                "experiment_name": ["exp_a"] * len(predictions),
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        feature = RetentionTimeFeature(train_fraction=1.0, min_train_points=11)

        with pytest.warns(
            UserWarning, match=r"unique peptide\(s\), below min_train_points"
        ):
            with pytest.warns(UserWarning, match="Skipping experiment 'exp_a'"):
                feature.prepare(dataset)

        assert "exp_a" not in feature.irt_predictors

    @patch("winnow.calibration.features.retention_time.koinapy.Koina")
    def test_compute_skipped_experiment_learn_from_missing_false(self, mock_koina):
        """Test that compute handles all spectra from a skipped experiment when learn_from_missing=False.

        Reproduces the scenario where an experiment has insufficient data for iRT
        calibration and learn_from_missing=False causes all its rows to be dropped,
        leaving no valid input for the Koina model call.
        """
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.model_inputs = ["peptide_sequences"]
        # Simulate koinapy crash on empty input
        mock_model_instance.predict.side_effect = IndexError("list index out of range")

        predictions = [
            ["A", "G"],
            ["G", "A"],
            ["L", "K"],
            ["M", "V"],
            ["P", "S"],
            ["T", "W"],
            ["Y", "H"],
            ["R", "N"],
            ["D", "E"],
            ["F", "I"],
        ]
        n_spectra = len(predictions)
        metadata = pd.DataFrame(
            {
                "confidence": [0.95 - 0.01 * i for i in range(n_spectra)],
                "prediction": predictions,
                "retention_time": [10.5 + i * 2.1 for i in range(n_spectra)],
                "spectrum_id": list(range(n_spectra)),
                "experiment_name": ["exp_a"] * n_spectra,
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        feature = RetentionTimeFeature(
            train_fraction=1.0,
            min_train_points=11,
            learn_from_missing=False,
        )

        with pytest.warns(
            UserWarning, match=r"unique peptide\(s\), below min_train_points"
        ):
            with pytest.warns(UserWarning, match="Skipping experiment 'exp_a'"):
                feature.prepare(dataset)

        assert "exp_a" not in feature.irt_predictors

        # compute should not crash even though all spectra will be dropped
        with pytest.warns(
            UserWarning,
            match=(
                rf"Filtered {n_spectra} spectra that do not satisfy the validity "
                rf"constraints for the Koina iRT model"
            ),
        ):
            feature.compute(dataset)

        # All rows should have been dropped (learn_from_missing=False)
        assert dataset.metadata.empty
        # Koina should never have been called since there's nothing to predict
        mock_model_instance.predict.assert_not_called()

    @patch("winnow.calibration.features.retention_time.koinapy.Koina")
    def test_compute_with_mock(
        self, mock_koina, retention_time_feature, sample_dataset_with_rt
    ):
        """Test compute method with mocked models."""
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.model_inputs = ["peptide_sequences"]

        mock_model_instance.predict.return_value = pd.DataFrame(
            {"irt": [25.5, 30.2, 35.1, 20.7, 28.1]}
        )

        reg = LinearRegression()
        reg.fit([[10], [20]], [25, 50])
        retention_time_feature.irt_predictors["__global__"] = reg

        retention_time_feature.compute(sample_dataset_with_rt)

        assert "irt" in sample_dataset_with_rt.metadata.columns
        assert "predicted_irt" in sample_dataset_with_rt.metadata.columns
        assert "irt_error" in sample_dataset_with_rt.metadata.columns
        assert "is_missing_irt_error" in sample_dataset_with_rt.metadata.columns

        assert len(sample_dataset_with_rt.metadata["irt_error"]) == 5
        max_abs_diff = abs(
            sample_dataset_with_rt.metadata["predicted_irt"]
            - sample_dataset_with_rt.metadata["irt"]
        ).max()
        max_error = sample_dataset_with_rt.metadata["irt_error"].max()
        assert max_abs_diff == pytest.approx(max_error, rel=1e-10, abs=1e-10)

    def test_compute_maps_values_to_correct_rows_and_imputes_missing(
        self,
        retention_time_feature,
    ):
        """Test that computed iRT values are mapped to correct rows and missing values are imputed correctly."""
        metadata = pd.DataFrame(
            {
                "confidence": [0.9, 0.7, 0.6],
                "prediction": [
                    ["A", "G"],
                    ["A"] * 31,
                    ["A", "G"],
                ],
                "prediction_untokenised": [
                    "AG",
                    "A" * 31,
                    "AG",
                ],
                "retention_time": [10.5, 15.2, 8.7],
                "precursor_charge": [2, 2, 2],
                "spectrum_id": [10, 20, 40],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        def mock_koina_predict(inputs_df):
            if len(inputs_df) == 0:
                return pd.DataFrame(columns=["irt"])
            predictions_list = []
            index_list = []
            for spectrum_id in inputs_df.index:
                if spectrum_id == 10:
                    irt_value = 25.5
                elif spectrum_id == 40:
                    irt_value = 30.2
                else:
                    irt_value = 20.0
                predictions_list.append({"irt": irt_value})
                index_list.append(spectrum_id)
            return pd.DataFrame(predictions_list, index=index_list)

        reg = LinearRegression()
        reg.coef_ = np.array([2.0])
        reg.intercept_ = 0.0
        reg.fit([[1], [2]], [2, 4])
        retention_time_feature.irt_predictors["__global__"] = reg

        with patch(
            "winnow.calibration.features.retention_time.koinapy.Koina"
        ) as mock_koina:
            mock_model = mock_koina.return_value
            mock_model.model_inputs = ["peptide_sequences"]
            mock_model.predict = mock_koina_predict

            retention_time_feature.compute(dataset)

        assert "is_missing_irt_error" in dataset.metadata.columns

        valid_flags = ~dataset.metadata["is_missing_irt_error"]
        spectrum_10_mask = dataset.metadata["spectrum_id"] == 10
        spectrum_20_mask = dataset.metadata["spectrum_id"] == 20
        spectrum_40_mask = dataset.metadata["spectrum_id"] == 40

        assert valid_flags[spectrum_10_mask].iloc[0]
        assert not valid_flags[spectrum_20_mask].iloc[0]
        assert valid_flags[spectrum_40_mask].iloc[0]

        assert "irt" in dataset.metadata.columns
        assert "predicted_irt" in dataset.metadata.columns
        assert "irt_error" in dataset.metadata.columns

        irt_10 = dataset.metadata[spectrum_10_mask]["irt"].iloc[0]
        irt_40 = dataset.metadata[spectrum_40_mask]["irt"].iloc[0]
        assert not pd.isna(irt_10)
        assert not pd.isna(irt_40)
        assert irt_10 == 25.5
        assert irt_40 == 30.2

        irt_20 = dataset.metadata[spectrum_20_mask]["irt"].iloc[0]
        assert pd.isna(irt_20)

        predicted_irt_10 = dataset.metadata[spectrum_10_mask]["predicted_irt"].iloc[0]
        predicted_irt_40 = dataset.metadata[spectrum_40_mask]["predicted_irt"].iloc[0]
        assert not pd.isna(predicted_irt_10)
        assert not pd.isna(predicted_irt_40)

        irt_error_10 = dataset.metadata[spectrum_10_mask]["irt_error"].iloc[0]
        irt_error_40 = dataset.metadata[spectrum_40_mask]["irt_error"].iloc[0]
        expected_error_10 = abs(predicted_irt_10 - irt_10)
        expected_error_40 = abs(predicted_irt_40 - irt_40)
        assert irt_error_10 == pytest.approx(expected_error_10, rel=1e-10)
        assert irt_error_40 == pytest.approx(expected_error_40, rel=1e-10)

        irt_error_20 = dataset.metadata[spectrum_20_mask]["irt_error"].iloc[0]
        assert irt_error_20 == 0.0

        assert len(dataset.metadata) == 3
        assert set(dataset.metadata["spectrum_id"].values) == {10, 20, 40}

    def test_select_training_data_errors_on_single_unique_peptide(self):
        """Raise when duplicate peptides yield a single Koina iRT target."""
        feature = RetentionTimeFeature(
            train_fraction=1.0,
            min_train_points=2,
            learn_from_missing=False,
        )
        metadata = pd.DataFrame(
            {
                "confidence": [0.95, 0.90, 0.85, 0.80, 0.75, 0.70],
                "prediction": [["A", "G"]] * 6,
                "retention_time": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                "precursor_charge": [2] * 6,
                "spectrum_id": list(range(6)),
            }
        )

        with pytest.raises(
            ValueError,
            match=(
                r"(?s)training pool has only 1 unique peptide\(s\).*"
                r"Koina iRT prediction models output one iRT per peptide sequence"
            ),
        ):
            feature._select_training_data(metadata, "exp_a")

    def test_select_training_data_warns_on_partial_sequence_diversity(self):
        """Warn when unique sequences are above 2 but below min_train_points."""
        feature = RetentionTimeFeature(
            train_fraction=1.0,
            min_train_points=5,
            learn_from_missing=False,
        )
        metadata = pd.DataFrame(
            {
                "confidence": [0.95, 0.90, 0.85, 0.80, 0.75, 0.70],
                "prediction": [
                    ["A", "G"],
                    ["A", "G"],
                    ["G", "A"],
                    ["G", "A"],
                    ["L", "K"],
                    ["M", "V"],
                ],
                "retention_time": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                "precursor_charge": [2] * 6,
                "spectrum_id": list(range(6)),
            }
        )

        with pytest.warns(
            UserWarning, match=r"unique peptide\(s\), below min_train_points"
        ):
            train_data = feature._select_training_data(metadata, "exp_a")

        assert len(train_data) == 6

    def test_select_training_data_no_warn_when_two_unique_peptides(self):
        """Do not warn when the pool has exactly two distinct peptides."""
        feature = RetentionTimeFeature(
            train_fraction=1.0,
            min_train_points=2,
            learn_from_missing=False,
        )
        metadata = pd.DataFrame(
            {
                "confidence": [0.95, 0.90, 0.85, 0.80],
                "prediction": [["A", "G"], ["A", "G"], ["G", "A"], ["G", "A"]],
                "retention_time": [10.0, 11.0, 12.0, 13.0],
                "precursor_charge": [2] * 4,
                "spectrum_id": list(range(4)),
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            train_data = feature._select_training_data(metadata, "exp_a")

        assert len(train_data) == 4

    def test_select_training_data_errors_on_single_retention_time(self):
        """Raise when the calibration pool has no RT spread for linear regression."""
        feature = RetentionTimeFeature(
            train_fraction=1.0,
            min_train_points=2,
            learn_from_missing=False,
        )
        metadata = pd.DataFrame(
            {
                "confidence": [0.95, 0.90, 0.85],
                "prediction": [["A", "G"], ["G", "A"], ["S", "P"]],
                "retention_time": [10.0, 10.0, 10.0],
                "precursor_charge": [2, 2, 2],
                "spectrum_id": [0, 1, 2],
            }
        )

        with pytest.raises(
            ValueError,
            match=r"training pool has only 1 unique retention time value\(s\)",
        ):
            feature._select_training_data(metadata, "exp_a")

    @patch("winnow.calibration.features.retention_time.koinapy.Koina")
    def test_prepare_skips_experiment_with_single_unique_peptide(self, mock_koina):
        """Prepare skips experiments whose calibration pool has one peptide."""
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.model_inputs = ["peptide_sequences"]

        metadata = pd.DataFrame(
            {
                "confidence": [0.95, 0.90, 0.85, 0.80],
                "prediction": [["A", "G"]] * 4,
                "retention_time": [10.0, 11.0, 12.0, 13.0],
                "spectrum_id": list(range(4)),
                "experiment_name": ["exp_a"] * 4,
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)
        feature = RetentionTimeFeature(train_fraction=1.0, min_train_points=2)

        with pytest.warns(UserWarning, match="Skipping experiment 'exp_a'"):
            feature.prepare(dataset)

        assert "exp_a" not in feature.irt_predictors
        mock_model_instance.predict.assert_not_called()

    def test_save_and_load_regressors_safetensors(self, tmp_path):
        """Test save_regressors and load_regressors round-trip with safetensors."""
        feature = RetentionTimeFeature(train_fraction=0.5, min_train_points=2)
        reg_a = LinearRegression()
        reg_a.fit([[1], [2]], [10, 20])
        reg_b = LinearRegression()
        reg_b.fit([[3], [4]], [30, 40])
        feature.irt_predictors = {"exp_a": reg_a, "exp_b": reg_b}

        path = tmp_path / "regressors.safetensors"
        feature.save_regressors(path)
        assert path.exists()

        new_feature = RetentionTimeFeature(train_fraction=0.5, min_train_points=2)
        assert len(new_feature.irt_predictors) == 0

        new_feature.load_regressors(path)
        assert set(new_feature.irt_predictors.keys()) == {"exp_a", "exp_b"}

        np.testing.assert_array_almost_equal(
            new_feature.irt_predictors["exp_a"].predict([[1.5]]),
            reg_a.predict([[1.5]]),
        )
