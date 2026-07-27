"""Unit tests for winnow calibration feature RetentionTimeFeature."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
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

    def test_check_valid_irt_prediction_treats_none_as_invalid(
        self, retention_time_feature
    ):
        """Empty→None predictions must be iRT-invalid, not raise TypeError."""
        metadata = pd.DataFrame(
            {
                "confidence": [0.95, 0.90, 0.85],
                "prediction": [["A", "G"], None, ["U", "A"]],
                "spectrum_id": [0, 1, 2],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)
        is_valid = retention_time_feature.check_valid_irt_prediction(dataset)
        assert is_valid.tolist() == [True, False, False]

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
    def test_compute_handles_non_string_experiment_name(self, mock_koina):
        """Test per-experiment iRT computation with non-string experiment_name values."""
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.model_inputs = ["peptide_sequences"]

        def predict(inputs):
            return pd.DataFrame({"irt": range(len(inputs))}, index=inputs.index)

        mock_model_instance.predict.side_effect = predict

        metadata = pd.DataFrame(
            {
                "confidence": [0.95, 0.90, 0.85, 0.80],
                "prediction": [["A", "G"], ["G", "A"], ["S", "P"], ["V"]],
                "retention_time": [10.5, 15.2, 20.1, 8.7],
                "spectrum_id": [0, 1, 2, 3],
                "experiment_name": [1, 1, 2, 2],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        feature = RetentionTimeFeature(train_fraction=1.0, min_train_points=2)
        feature.prepare(dataset)

        feature.compute(dataset)

        assert "irt_error" in dataset.metadata.columns
        assert "predicted iRT" in dataset.metadata.columns
        assert dataset.metadata["irt_error"].notna().all()
        assert dataset.metadata["predicted iRT"].notna().all()

    @patch("winnow.calibration.features.retention_time.koinapy.Koina")
    def test_prepare_skips_preloaded_experiments(self, mock_koina, tmp_path):
        """Test that prepare skips experiments loaded via load_regressors."""
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

        from sklearn.linear_model import LinearRegression

        preloaded = LinearRegression()
        preloaded.fit([[1], [2]], [10, 20])

        regressor_path = tmp_path / "regressors.pkl"
        donor = RetentionTimeFeature(train_fraction=1.0, min_train_points=2)
        donor.irt_predictors["exp_a"] = preloaded
        donor.save_regressors(regressor_path)

        feature = RetentionTimeFeature(train_fraction=1.0, min_train_points=2)
        feature.load_regressors(regressor_path)
        feature.prepare(dataset)

        # exp_a should still be the checkpoint regressor (values, not object identity)
        np.testing.assert_array_almost_equal(
            feature.irt_predictors["exp_a"].predict([[1.5]]),
            preloaded.predict([[1.5]]),
        )
        assert "exp_a" in feature._loaded_experiment_names
        # exp_b should have been fitted fresh from the current dataset
        assert "exp_b" not in feature._loaded_experiment_names
        assert "exp_b" in feature.irt_predictors
        np.testing.assert_array_almost_equal(
            feature.irt_predictors["exp_b"].predict([[20.1], [8.7]]),
            [35.1, 20.7],
        )

    @patch("winnow.calibration.features.retention_time.koinapy.Koina")
    def test_prepare_refits_regressors_if_not_loaded_from_checkpoint(self, mock_koina):
        """Test that prepare refits regressors when the same experiment is seen again."""
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.model_inputs = ["peptide_sequences"]
        mock_model_instance.predict.return_value = pd.DataFrame({"irt": [10.0, 20.0]})

        metadata_a = pd.DataFrame(
            {
                "confidence": [0.95, 0.90],
                "prediction": [["A", "G"], ["G", "A"]],
                "retention_time": [1.0, 2.0],
                "spectrum_id": [0, 1],
                "experiment_name": ["exp_a", "exp_a"],
            }
        )
        dataset_a = CalibrationDataset(metadata=metadata_a, predictions=None)

        feature = RetentionTimeFeature(train_fraction=1.0, min_train_points=2)
        feature.prepare(dataset_a)
        reg_after_a = feature.irt_predictors["exp_a"]
        coef_a = reg_after_a.coef_[0]

        mock_model_instance.predict.return_value = pd.DataFrame({"irt": [30.0, 40.0]})
        metadata_b = pd.DataFrame(
            {
                "confidence": [0.95, 0.90],
                "prediction": [["A", "G"], ["G", "A"]],
                "retention_time": [100.0, 200.0],
                "spectrum_id": [0, 1],
                "experiment_name": ["exp_a", "exp_a"],
            }
        )
        dataset_b = CalibrationDataset(metadata=metadata_b, predictions=None)
        feature.prepare(dataset_b)
        reg_after_b = feature.irt_predictors["exp_a"]

        assert reg_after_b is not reg_after_a
        assert reg_after_b.coef_[0] != coef_a

    @patch("winnow.calibration.features.retention_time.koinapy.Koina")
    def test_prepare_skips_on_insufficient_data(self, mock_koina):
        """Test that prepare skips experiments when min_train_points is not met."""
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.model_inputs = ["peptide_sequences"]

        metadata = pd.DataFrame(
            {
                "confidence": [0.95 - 0.01 * i for i in range(20)],
                "prediction": [
                    ["A", "G"] if i % 2 == 0 else ["G", "A"] for i in range(20)
                ],
                "retention_time": [10.5 + i for i in range(20)],
                "spectrum_id": list(range(20)),
                "experiment_name": ["exp_a"] * 20,
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        feature = RetentionTimeFeature(train_fraction=0.1, min_train_points=10)

        with pytest.warns(
            UserWarning, match="Skipping RT->iRT regressor fit for experiment 'exp_a'"
        ):
            feature.prepare(dataset)

        assert "exp_a" not in feature.irt_predictors

    @pytest.mark.parametrize("learn_from_missing", [True, False])
    @patch("winnow.calibration.features.retention_time.koinapy.Koina")
    def test_skipped_integer_experiment_name_is_marked(
        self, mock_koina, learn_from_missing
    ):
        """Skipped experiments match even when experiment_name is non-string.

        Group keys are stored as str; marking must compare on a string view so
        integer IDs are still flagged missing / dropped.
        """
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.model_inputs = ["peptide_sequences"]

        metadata = pd.DataFrame(
            {
                "confidence": [0.95, 0.90],
                "prediction": [["A", "G"], ["G", "A"]],
                "retention_time": [10.5, 15.2],
                "spectrum_id": [0, 1],
                "experiment_name": [101, 101],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)
        feature = RetentionTimeFeature(
            train_fraction=0.1,
            min_train_points=10,
            learn_from_missing=learn_from_missing,
        )

        with pytest.warns(
            UserWarning, match="Skipping RT->iRT regressor fit for experiment '101'"
        ):
            feature.prepare(dataset)

        assert feature._skipped_experiments == ["101"]
        assert "101" not in feature.irt_predictors

        if learn_from_missing:
            feature.compute(dataset)
            assert len(dataset) == 2
            assert dataset.metadata["is_missing_irt_error"].tolist() == [True, True]
            assert dataset.metadata["irt_error"].tolist() == [0.0, 0.0]
        else:
            with pytest.warns(
                UserWarning,
                match=(
                    r"Filtered 2 spectra that do not satisfy the validity "
                    r"constraints for the Koina iRT model"
                ),
            ):
                feature.compute(dataset)
            assert len(dataset) == 0

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
                "spectrum_id": list(range(6)),
            }
        )

        with pytest.raises(
            ValueError,
            match=(
                r"(?s)Experiment 'exp_a': Cannot fit iRT calibration regressor\."
                r".*Training pool has only 1 unique peptide\(s\).*"
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
                "spectrum_id": list(range(6)),
            }
        )

        with pytest.warns(
            UserWarning, match=r"unique peptide\(s\), below min_train_points"
        ):
            train_data = feature._select_training_data(metadata, "exp_a")

        assert len(train_data) == 6

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
                "spectrum_id": [0, 1, 2],
            }
        )

        with pytest.raises(
            ValueError,
            match=(
                r"(?s)Experiment 'exp_a': Cannot fit iRT calibration regressor\."
                r".*unique retention time value\(s\)"
            ),
        ):
            feature._select_training_data(metadata, "exp_a")

    @patch("winnow.calibration.features.retention_time.koinapy.Koina")
    def test_compute_skips_koina_when_no_valid_spectra(self, mock_koina):
        """Do not call Koina when every spectrum is invalid for iRT prediction."""
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.model_inputs = ["peptide_sequences"]

        metadata = pd.DataFrame(
            {
                "confidence": [0.95, 0.90],
                "prediction": [["A"] * 31, ["G"] * 31],
                "retention_time": [10.5, 15.2],
                "spectrum_id": [0, 1],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)
        feature = RetentionTimeFeature(max_peptide_length=30)

        feature.compute(dataset)

        mock_model_instance.predict.assert_not_called()
        assert dataset.metadata["irt_error"].tolist() == [0.0, 0.0]

    @patch("winnow.calibration.features.retention_time.koinapy.Koina")
    def test_skipped_experiments_reset_between_prepare_calls(self, mock_koina):
        """Stale skipped experiments must not affect a later file with the same name."""
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.model_inputs = ["peptide_sequences"]
        mock_model_instance.predict.side_effect = lambda inputs: pd.DataFrame(
            {"irt": range(len(inputs))}
        )

        feature = RetentionTimeFeature(
            train_fraction=1.0,
            min_train_points=10,
            learn_from_missing=True,
        )

        insufficient_metadata = pd.DataFrame(
            {
                "confidence": [0.95, 0.90],
                "prediction": [["A", "G"], ["G", "A"]],
                "retention_time": [10.5, 15.2],
                "spectrum_id": [0, 1],
                "experiment_name": ["exp_a", "exp_a"],
            }
        )
        insufficient_dataset = CalibrationDataset(
            metadata=insufficient_metadata, predictions=None
        )

        with pytest.warns(
            UserWarning, match="Skipping RT->iRT regressor fit for experiment 'exp_a'"
        ):
            feature.prepare(insufficient_dataset)
        assert feature._skipped_experiments == ["exp_a"]
        assert "exp_a" not in feature.irt_predictors

        feature.compute(insufficient_dataset)
        assert insufficient_dataset.metadata["is_missing_irt_error"].tolist() == [
            True,
            True,
        ]

        repeated_peptides = [
            ["P", "E", "P", "T", "I", "D", "E"],
            ["G", "L", "Y", "G", "A", "T"],
            ["K", "V", "L", "V", "A", "P"],
            ["A", "I", "V", "E", "G"],
            ["S", "T", "D", "K"],
        ]
        sufficient_metadata = pd.DataFrame(
            {
                "confidence": [
                    0.95,
                    0.93,
                    0.91,
                    0.89,
                    0.87,
                    0.85,
                    0.83,
                    0.81,
                    0.79,
                    0.77,
                ],
                "prediction": repeated_peptides + repeated_peptides,
                "retention_time": [
                    12.5,
                    15.2,
                    18.1,
                    21.0,
                    24.3,
                    26.8,
                    29.1,
                    31.5,
                    34.0,
                    36.2,
                ],
                "spectrum_id": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "experiment_name": ["exp_a"] * 10,
            }
        )
        sufficient_dataset = CalibrationDataset(
            metadata=sufficient_metadata, predictions=None
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            feature.prepare(sufficient_dataset)
        assert feature._skipped_experiments == []
        assert "exp_a" in feature.irt_predictors

        feature.compute(sufficient_dataset)
        assert not sufficient_dataset.metadata["is_missing_irt_error"].any()
        assert sufficient_dataset.metadata["irt_error"].notna().all()
        assert sufficient_dataset.metadata["predicted iRT"].notna().all()

    @patch("winnow.calibration.features.retention_time.koinapy.Koina")
    def test_compute_imputes_skipped_global_regressor(self, mock_koina):
        """Test global skipped iRT fits are imputed when experiment_name is absent."""
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.model_inputs = ["peptide_sequences"]
        mock_model_instance.predict.return_value = pd.DataFrame({"irt": [10.0, 20.0]})

        metadata = pd.DataFrame(
            {
                "confidence": [0.95, 0.90],
                "prediction": [["A", "G"], ["G", "A"]],
                "retention_time": [10.5, 15.2],
                "spectrum_id": [0, 1],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        feature = RetentionTimeFeature(
            train_fraction=1.0,
            min_train_points=10,
            learn_from_missing=True,
        )

        with pytest.warns(UserWarning, match="Skipping global RT->iRT regressor fit"):
            feature.prepare(dataset)

        assert "__global__" not in feature.irt_predictors
        assert feature._skipped_experiments == ["__global__"]

        feature.compute(dataset)

        assert dataset.metadata["is_missing_irt_error"].tolist() == [True, True]
        assert dataset.metadata["irt_error"].tolist() == [0.0, 0.0]
        assert dataset.metadata["iRT"].isna().all()
        assert dataset.metadata["predicted iRT"].isna().all()

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

        from sklearn.linear_model import LinearRegression

        reg = LinearRegression()
        reg.fit([[10], [20]], [25, 50])
        retention_time_feature.irt_predictors["__global__"] = reg

        retention_time_feature.compute(sample_dataset_with_rt)

        assert "iRT" in sample_dataset_with_rt.metadata.columns
        assert "predicted iRT" in sample_dataset_with_rt.metadata.columns
        assert "irt_error" in sample_dataset_with_rt.metadata.columns
        assert "is_missing_irt_error" in sample_dataset_with_rt.metadata.columns

        assert len(sample_dataset_with_rt.metadata["irt_error"]) == 5
        max_abs_diff = abs(
            sample_dataset_with_rt.metadata["predicted iRT"]
            - sample_dataset_with_rt.metadata["iRT"]
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

        def mock_prosit_predict(inputs_df):
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

        from sklearn.linear_model import LinearRegression

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
            mock_model.predict = mock_prosit_predict

            retention_time_feature.compute(dataset)

        assert "is_missing_irt_error" in dataset.metadata.columns

        valid_flags = ~dataset.metadata["is_missing_irt_error"]
        spectrum_10_mask = dataset.metadata["spectrum_id"] == 10
        spectrum_20_mask = dataset.metadata["spectrum_id"] == 20
        spectrum_40_mask = dataset.metadata["spectrum_id"] == 40

        assert valid_flags[spectrum_10_mask].iloc[0]
        assert not valid_flags[spectrum_20_mask].iloc[0]
        assert valid_flags[spectrum_40_mask].iloc[0]

        assert "iRT" in dataset.metadata.columns
        assert "predicted iRT" in dataset.metadata.columns
        assert "irt_error" in dataset.metadata.columns

        irt_10 = dataset.metadata[spectrum_10_mask]["iRT"].iloc[0]
        irt_40 = dataset.metadata[spectrum_40_mask]["iRT"].iloc[0]
        assert not pd.isna(irt_10)
        assert not pd.isna(irt_40)
        assert irt_10 == 25.5
        assert irt_40 == 30.2

        irt_20 = dataset.metadata[spectrum_20_mask]["iRT"].iloc[0]
        assert pd.isna(irt_20)

        predicted_irt_10 = dataset.metadata[spectrum_10_mask]["predicted iRT"].iloc[0]
        predicted_irt_40 = dataset.metadata[spectrum_40_mask]["predicted iRT"].iloc[0]
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

    def test_pickle_excludes_regressor_state(self):
        """Test that pickle round-trip excludes transient irt_predictors."""
        import pickle
        from sklearn.linear_model import LinearRegression

        feature = RetentionTimeFeature(train_fraction=0.5, min_train_points=2)
        reg = LinearRegression()
        reg.fit([[1], [2]], [10, 20])
        feature.irt_predictors["test_exp"] = reg

        data = pickle.dumps(feature)
        restored = pickle.loads(data)

        assert isinstance(restored.irt_predictors, dict)
        assert len(restored.irt_predictors) == 0
        assert restored._loaded_experiment_names == set()
        assert restored.train_fraction == 0.5
        assert restored.min_train_points == 2

    @patch("winnow.calibration.features.retention_time.koinapy.Koina")
    def test_legacy_pickle_state_backfills_min_train_points(self, mock_koina):
        """Test old pickled RetentionTimeFeature state restores current defaults."""
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.model_inputs = ["peptide_sequences"]
        mock_model_instance.predict.return_value = pd.DataFrame({"irt": range(10)})

        legacy_state = {
            "train_fraction": 1.0,
            "learn_from_missing": True,
            "seed": 42,
            "unsupported_residues": [],
            "irt_model_name": "Prosit_2019_irt",
            "max_peptide_length": 30,
        }

        feature = RetentionTimeFeature.__new__(RetentionTimeFeature)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            feature.__setstate__(legacy_state)

            # There should be a warning about min_train_points being set to 10.
            matched_warning = [
                warn
                for warn in w
                if "min_train_points not found in state, setting to 10"
                in str(warn.message)
            ]
            assert matched_warning, "Expected warning for backfill of min_train_points"

        assert feature.min_train_points == 10

        unique_predictions = [
            ["A", "G"],
            ["G", "A"],
            ["S", "P"],
            ["L", "K"],
            ["M", "V"],
            ["F", "W"],
            ["Y", "H"],
            ["R", "N"],
            ["D", "E"],
            ["Q", "C"],
        ]
        metadata = pd.DataFrame(
            {
                "confidence": [0.95 - (idx * 0.01) for idx in range(10)],
                "prediction": unique_predictions,
                "retention_time": [float(idx) for idx in range(10)],
                "spectrum_id": list(range(10)),
                "experiment_name": ["exp_a"] * 10,
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        feature.prepare(dataset)

        assert "exp_a" in feature.irt_predictors

    @patch("winnow.calibration.features.retention_time.koinapy.Koina")
    def test_prepare_warns_on_low_peptide_diversity(self, mock_koina):
        """Test warning when unique peptides are below min_train_points but PSM count is not."""
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.model_inputs = ["peptide_sequences"]
        mock_model_instance.predict.return_value = pd.DataFrame({"irt": range(10)})

        # Five distinct peptides, each observed twice (10 PSMs total).
        repeated_peptides = [
            ["P", "E", "P", "T", "I", "D", "E"],
            ["G", "L", "Y", "G", "A", "T"],
            ["K", "V", "L", "V", "A", "P"],
            ["A", "I", "V", "E", "G"],
            ["S", "T", "D", "K"],
        ]

        metadata = pd.DataFrame(
            {
                "confidence": [
                    0.95,
                    0.93,
                    0.91,
                    0.89,
                    0.87,
                    0.85,
                    0.83,
                    0.81,
                    0.79,
                    0.77,
                ],
                "prediction": repeated_peptides + repeated_peptides,
                "retention_time": [
                    12.5,
                    15.2,
                    18.1,
                    21.0,
                    24.3,
                    26.8,
                    29.1,
                    31.5,
                    34.0,
                    36.2,
                ],
                "spectrum_id": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "experiment_name": ["exp_a"] * 10,
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        feature = RetentionTimeFeature(train_fraction=1.0, min_train_points=10)

        with pytest.warns(
            UserWarning,
            match=(
                r"Experiment 'exp_a': iRT calibration pool \(top 100%, 10 PSMs\):\n"
                r"  Only 5 unique peptide\(s\), below min_train_points=10\."
            ),
        ):
            feature.prepare(dataset)

        assert "exp_a" in feature.irt_predictors

    def test_save_and_load_regressors(self, tmp_path):
        """Test save_regressors and load_regressors round-trip."""
        from sklearn.linear_model import LinearRegression

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
        assert new_feature._loaded_experiment_names == {"exp_a", "exp_b"}

        np.testing.assert_array_almost_equal(
            new_feature.irt_predictors["exp_a"].predict([[1.5]]),
            reg_a.predict([[1.5]]),
        )
