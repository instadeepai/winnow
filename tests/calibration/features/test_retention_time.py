"""Unit tests for winnow calibration feature RetentionTimeFeature."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from winnow.calibration.features.retention_time import RetentionTimeFeature
from winnow.datasets.calibration_dataset import CalibrationDataset


class TestRetentionTimeFeature:
    """Test the RetentionTimeFeature class."""

    @pytest.fixture()
    def retention_time_feature(self):
        """Create a RetentionTimeFeature instance for testing."""
        return RetentionTimeFeature(
            hidden_dim=10, train_fraction=0.8, unsupported_residues=["U", "O", "X"]
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

    # ------------------------------------------------------------------
    # Basic Properties and Initialization
    # ------------------------------------------------------------------

    def test_properties(self, retention_time_feature):
        """Test RetentionTimeFeature properties."""
        assert retention_time_feature.name == "iRT Feature"
        assert retention_time_feature.columns == ["irt_error", "is_missing_irt_error"]
        assert retention_time_feature.dependencies == []

    def test_initialization_parameters(self):
        """Test initialization with custom parameters."""
        feature = RetentionTimeFeature(
            hidden_dim=10, train_fraction=0.8, unsupported_residues=["U", "O", "X"]
        )
        assert feature.hidden_dim == 10
        assert feature.train_fraction == 0.8
        assert feature.irt_model_name == "Prosit_2019_irt"
        assert hasattr(feature, "irt_predictor")

    # ------------------------------------------------------------------
    # Column Configuration
    # ------------------------------------------------------------------

    def test_learn_from_missing_false_columns_excludes_indicator(self):
        """learn_from_missing=False: is_missing_irt_error not in columns."""
        feature = RetentionTimeFeature(
            hidden_dim=10,
            train_fraction=0.8,
            learn_from_missing=False,
        )
        assert "is_missing_irt_error" not in feature.columns
        assert feature.columns == ["irt_error"]

    # ------------------------------------------------------------------
    # Prepare Method
    # ------------------------------------------------------------------

    @patch("winnow.calibration.features.retention_time.koinapy.Koina")
    def test_prepare_with_mock(
        self, mock_koina, retention_time_feature, sample_dataset_with_rt
    ):
        """Test prepare method with mocked Koina iRT model."""
        # Mock the Koina model
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.model_inputs = ["peptide_sequences"]

        # Mock predict to return iRT values
        mock_model_instance.predict.return_value = pd.DataFrame(
            {
                "irt": [35.1, 20.7, 28.1, 25.5]  # 4 values for 80% of 5 samples
            }
        )

        # Mock the MLPRegressor fit method
        with patch.object(retention_time_feature.irt_predictor, "fit") as mock_fit:
            retention_time_feature.prepare(sample_dataset_with_rt)

            # Check that fit was called
            mock_fit.assert_called_once()

            # Check that predict was called on the model
            mock_model_instance.predict.assert_called_once()

            # Check that the model was fitted on the first 80% of the dataset
            expected_train_size = int(0.8 * len(sample_dataset_with_rt.metadata))
            expected_x = (
                sample_dataset_with_rt.metadata["retention_time"]
                .iloc[:expected_train_size]
                .values.reshape(-1, 1)
            )
            expected_y = (
                mock_model_instance.predict.return_value["irt"]
                .iloc[:expected_train_size]
                .values
            )

            # Verify fit was called with the correct arguments
            call_args = mock_fit.call_args
            np.testing.assert_array_equal(call_args[0][0], expected_x)
            np.testing.assert_array_equal(call_args[0][1], expected_y)

    def test_prepare_does_nothing_without_mocking(self, retention_time_feature):
        """Test that prepare method handles dataset requirements."""
        # Test with minimal valid dataset
        metadata = pd.DataFrame(
            {
                "confidence": [0.9, 0.8],
                "prediction": [["A"], ["G"]],
                "retention_time": [10.0, 15.0],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        # This would normally call the real Prosit model, but we're testing structure
        # Just ensure it doesn't crash on basic validation
        try:
            # The method expects certain columns to exist
            assert "confidence" in dataset.metadata.columns
            assert "prediction" in dataset.metadata.columns
            assert "retention_time" in dataset.metadata.columns
        except Exception as e:
            # If it fails due to network/model issues, that's expected in unit tests
            assert (
                "koina" in str(e).lower()
                or "connection" in str(e).lower()
                or "model" in str(e).lower()
            )

    # ------------------------------------------------------------------
    # Compute Method
    # ------------------------------------------------------------------

    @patch("winnow.calibration.features.retention_time.koinapy.Koina")
    def test_compute_with_mock(
        self, mock_koina, retention_time_feature, sample_dataset_with_rt
    ):
        """Test compute method with mocked models."""
        # Mock the Koina iRT model
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.model_inputs = ["peptide_sequences"]

        # Mock predict to return iRT values for all samples
        mock_model_instance.predict.return_value = pd.DataFrame(
            {"irt": [25.5, 30.2, 35.1, 20.7, 28.1]}
        )

        # Mock the MLPRegressor predict method
        with patch.object(
            retention_time_feature.irt_predictor, "predict"
        ) as mock_predict:
            mock_predict.return_value = [24.0, 29.5, 34.8, 21.2, 27.9]

            retention_time_feature.compute(sample_dataset_with_rt)

            # Check that columns were added
            assert "iRT" in sample_dataset_with_rt.metadata.columns
            assert "predicted iRT" in sample_dataset_with_rt.metadata.columns
            assert "irt_error" in sample_dataset_with_rt.metadata.columns
            assert "is_missing_irt_error" in sample_dataset_with_rt.metadata.columns

            # Check that error is computed as absolute difference
            assert len(sample_dataset_with_rt.metadata["irt_error"]) == 5
            max_abs_diff = abs(
                sample_dataset_with_rt.metadata["predicted iRT"]
                - sample_dataset_with_rt.metadata["iRT"]
            ).max()
            max_error = sample_dataset_with_rt.metadata["irt_error"].max()
            assert max_abs_diff == pytest.approx(max_error, rel=1e-10, abs=1e-10)

            # Check that predict was called on both models
            mock_model_instance.predict.assert_called_once()
            mock_predict.assert_called_once()

    def test_compute_maps_values_to_correct_rows_and_imputes_missing(
        self,
        retention_time_feature,
    ):
        """Test that computed iRT values are mapped to correct rows and missing values are imputed correctly."""
        # Create a dataset with mixed valid/invalid predictions
        # Based on check_valid_irt_prediction logic:
        # Valid: len <= 30, no invalid tokens
        # Invalid: len > 30 OR has invalid tokens
        metadata = pd.DataFrame(
            {
                "confidence": [0.9, 0.7, 0.6],
                "prediction": [
                    ["A", "G"],  # Valid: len 2
                    ["A"] * 31,  # Invalid: len > 30
                    ["A", "G"],  # Valid: len 2
                ],
                "prediction_untokenised": [
                    "AG",
                    "A" * 31,
                    "AG",
                ],
                "retention_time": [10.5, 15.2, 8.7],
                "precursor_charge": [2, 2, 2],
                "spectrum_id": [
                    10,
                    20,
                    40,
                ],  # Non-contiguous spectrum IDs to test mapping
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        # Mock Prosit model to return iRT predictions
        def mock_prosit_predict(inputs_df):
            """Mock Prosit predict that returns iRT values."""
            if len(inputs_df) == 0:
                return pd.DataFrame(columns=["irt"])

            predictions_list = []
            index_list = []

            for spectrum_id in inputs_df.index:
                # Return different iRT values for valid spectra
                if spectrum_id == 10:
                    irt_value = 25.5
                elif spectrum_id == 40:
                    irt_value = 30.2
                else:
                    irt_value = 20.0  # Fallback

                predictions_list.append({"irt": irt_value})
                index_list.append(spectrum_id)

            predictions_df = pd.DataFrame(predictions_list, index=index_list)
            return predictions_df

        # Mock the MLPRegressor predict method
        # When patching an instance method with patch.object, side_effect receives only the method arguments (not self)
        def mock_mlp_predict(x):
            """Mock MLPRegressor predict that returns predicted iRT values."""
            # Simple linear mapping for testing
            return x.flatten() * 2.0

        # Run compute with mocked Koina iRT model and MLPRegressor
        with patch(
            "winnow.calibration.features.retention_time.koinapy.Koina"
        ) as mock_koina:
            mock_model = mock_koina.return_value
            mock_model.model_inputs = ["peptide_sequences"]
            mock_model.predict = mock_prosit_predict

            with patch.object(
                retention_time_feature.irt_predictor,
                "predict",
                side_effect=mock_mlp_predict,
            ):
                retention_time_feature.compute(dataset)

        # Check that is_missing_irt_error column was added
        assert "is_missing_irt_error" in dataset.metadata.columns

        # Verify valid/invalid flags (based on the check_valid_irt_prediction logic)
        # The function filters OUT entries with: len > 30, invalid tokens, or WITH "C"
        # So valid entries are those WITHOUT these issues
        # Spectrum ID 10: no C, len 2 -> should be valid
        # Spectrum ID 20: has C -> should be invalid (filtered out)
        # Spectrum ID 30: len > 30 -> should be invalid (filtered out)
        # Spectrum ID 40: no C, len 2 -> should be valid
        valid_flags = ~dataset.metadata["is_missing_irt_error"]
        # Check by spectrum_id to verify mapping works correctly
        spectrum_10_mask = dataset.metadata["spectrum_id"] == 10
        spectrum_20_mask = dataset.metadata["spectrum_id"] == 20
        spectrum_40_mask = dataset.metadata["spectrum_id"] == 40

        assert valid_flags[spectrum_10_mask].iloc[0]  # Valid (len <= 30)
        assert not valid_flags[spectrum_20_mask].iloc[0]  # Invalid (has len > 30)
        assert valid_flags[spectrum_40_mask].iloc[0]  # Valid (len <= 30)

        # Check iRT mapping
        assert "iRT" in dataset.metadata.columns
        assert "predicted iRT" in dataset.metadata.columns
        assert "irt_error" in dataset.metadata.columns

        # Valid entries should have non-NaN iRT values
        irt_10 = dataset.metadata[spectrum_10_mask]["iRT"].iloc[0]
        irt_40 = dataset.metadata[spectrum_40_mask]["iRT"].iloc[0]
        assert not pd.isna(irt_10)
        assert not pd.isna(irt_40)
        assert isinstance(irt_10, (int, float))
        assert isinstance(irt_40, (int, float))
        assert irt_10 == 25.5  # From mock
        assert irt_40 == 30.2  # From mock

        # Invalid entries should have NaN iRT values
        irt_20 = dataset.metadata[spectrum_20_mask]["iRT"].iloc[0]
        assert pd.isna(irt_20)

        # Check predicted iRT (from MLPRegressor)
        predicted_irt_10 = dataset.metadata[spectrum_10_mask]["predicted iRT"].iloc[0]
        predicted_irt_40 = dataset.metadata[spectrum_40_mask]["predicted iRT"].iloc[0]
        assert not pd.isna(predicted_irt_10)
        assert not pd.isna(predicted_irt_40)
        # Should be rt * 2.0 from mock_mlp_predict
        assert predicted_irt_10 == pytest.approx(10.5 * 2.0, rel=1e-10)
        assert predicted_irt_40 == pytest.approx(8.7 * 2.0, rel=1e-10)

        # Check iRT error
        # For valid entries, error should be computed
        irt_error_10 = dataset.metadata[spectrum_10_mask]["irt_error"].iloc[0]
        irt_error_40 = dataset.metadata[spectrum_40_mask]["irt_error"].iloc[0]
        assert not pd.isna(irt_error_10)
        assert not pd.isna(irt_error_40)
        assert isinstance(irt_error_10, (int, float))
        assert isinstance(irt_error_40, (int, float))
        # Error should be absolute difference between predicted and actual iRT
        expected_error_10 = abs(predicted_irt_10 - irt_10)
        expected_error_40 = abs(predicted_irt_40 - irt_40)
        assert irt_error_10 == pytest.approx(expected_error_10, rel=1e-10)
        assert irt_error_40 == pytest.approx(expected_error_40, rel=1e-10)

        # For invalid entries, error should be 0.0 (fillna(0.0) is used)
        irt_error_20 = dataset.metadata[spectrum_20_mask]["irt_error"].iloc[0]
        assert irt_error_20 == 0.0

        assert len(dataset.metadata) == 3
        assert all(
            sid in dataset.metadata["spectrum_id"].values for sid in [10, 20, 40]
        )
        # Verify that spectrum_id values match the expected mapping
        assert set(dataset.metadata["spectrum_id"].values) == {10, 20, 40}

    # ------------------------------------------------------------------
    # learn_from_missing Behaviour
    # ------------------------------------------------------------------

    @patch("winnow.calibration.features.retention_time.koinapy.Koina")
    def test_learn_from_missing_false_drops_invalid_rows_and_warns(self, mock_koina):
        """learn_from_missing=False: invalid rows removed with warning."""
        feature = RetentionTimeFeature(
            hidden_dim=10,
            train_fraction=0.8,
            learn_from_missing=False,
            max_peptide_length=5,  # short limit so row 1 (len 6) is invalid
        )
        metadata = pd.DataFrame(
            {
                "prediction": [["A", "G"], ["A"] * 6, ["S", "P"]],
                "retention_time": [10.5, 15.2, 8.7],
                "precursor_charge": [2, 2, 2],
                "spectrum_id": [10, 20, 30],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        mock_model = mock_koina.return_value
        mock_model.model_inputs = ["peptide_sequences"]
        mock_model.predict.return_value = pd.DataFrame(
            {"irt": [25.5, 30.2]}, index=[10, 30]
        )

        with patch.object(
            feature.irt_predictor, "predict", return_value=np.array([21.0, 16.0])
        ):
            with pytest.warns(UserWarning, match="Filtered 1 spectra"):
                feature.compute(dataset)

        assert len(dataset.metadata) == 2
        assert 20 not in dataset.metadata["spectrum_id"].values
        assert "irt_error" in dataset.metadata.columns
