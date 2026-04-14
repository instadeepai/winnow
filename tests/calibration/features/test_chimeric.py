"""Unit tests for winnow calibration feature ChimericFeatures."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from winnow.calibration.features.chimeric import ChimericFeatures
from winnow.datasets.calibration_dataset import CalibrationDataset
from tests.calibration.features.conftest import MockScoredSequence, make_intensity_mock


class TestChimericFeatures:
    """Test the ChimericFeatures class."""

    @pytest.fixture()
    def chimeric_features(self):
        """Create a ChimericFeatures instance for testing."""
        return ChimericFeatures(
            mz_tolerance=0.02,
            unsupported_residues=["U", "O", "X"],
            model_input_constants={"collision_energies": 25},
        )

    @pytest.fixture()
    def sample_dataset_with_beam_predictions(self):
        """Create a sample dataset with beam search predictions for chimeric analysis."""
        metadata = pd.DataFrame(
            {
                "confidence": [0.9, 0.8, 0.7],
                "precursor_charge": [2, 2, 3],
                "spectrum_id": [0, 1, 2],
                "mz_array": [
                    [100.0, 200.0, 300.0],
                    [150.0, 250.0, 350.0],
                    [120.0, 220.0, 320.0],
                ],
                "intensity_array": [
                    [1000.0, 2000.0, 3000.0],
                    [1500.0, 2500.0, 3500.0],
                    [1200.0, 2200.0, 3200.0],
                ],
            }
        )

        predictions = [
            [  # First spectrum - 3 sequences
                MockScoredSequence(["A", "G"], np.log(0.8)),
                MockScoredSequence(["G", "A"], np.log(0.6)),
                MockScoredSequence(["S", "P"], np.log(0.4)),
            ],
            [  # Second spectrum - 2 sequences
                MockScoredSequence(["V", "T"], np.log(0.9)),
                MockScoredSequence(["T", "V"], np.log(0.7)),
            ],
            [  # Third spectrum - 1 sequence (will trigger warning)
                MockScoredSequence(["K"], np.log(0.95))
            ],
        ]

        return CalibrationDataset(metadata=metadata, predictions=predictions)

    # ------------------------------------------------------------------
    # Basic Properties and Initialization
    # ------------------------------------------------------------------

    def test_properties(self, chimeric_features):
        """Test ChimericFeatures properties."""
        assert chimeric_features.name == "Chimeric Features"
        assert chimeric_features.columns == [
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
            # Missing indicator (learn_from_missing=True by default)
            "is_missing_chimeric_features",
        ]
        assert chimeric_features.dependencies == []
        assert chimeric_features.mz_tolerance == 0.02

    def test_initialization_with_tolerance(self):
        """Test initialization with custom tolerance."""
        feature = ChimericFeatures(
            mz_tolerance=0.01,
            unsupported_residues=["U", "O", "X"],
            model_input_constants={"collision_energies": 25},
        )
        assert feature.mz_tolerance == 0.01
        assert feature.model_input_constants == {"collision_energies": 25}
        assert feature.model_input_columns is None

    def test_prepare_does_nothing(
        self, chimeric_features, sample_dataset_with_beam_predictions
    ):
        """Test that prepare method does nothing."""
        original_metadata = sample_dataset_with_beam_predictions.metadata.copy()
        chimeric_features.prepare(sample_dataset_with_beam_predictions)
        pd.testing.assert_frame_equal(
            sample_dataset_with_beam_predictions.metadata, original_metadata
        )

    # ------------------------------------------------------------------
    # Column Configuration
    # ------------------------------------------------------------------

    def test_columns_include_ion_coverage_features(self):
        """Verify columns include all ion coverage features with chimeric_ prefix."""
        feature = ChimericFeatures(mz_tolerance=0.02, learn_from_missing=True)
        columns = feature.columns

        ion_coverage_features = [
            "chimeric_longest_b_series",
            "chimeric_longest_y_series",
            "chimeric_complementary_ion_count",
            "chimeric_max_ion_gap",
        ]
        for col in ion_coverage_features:
            assert col in columns, f"Expected {col} in ChimericFeatures.columns"

    def test_learn_from_missing_false_columns_excludes_indicator(self):
        """learn_from_missing=False: is_missing_chimeric_features not in columns."""
        feature = ChimericFeatures(
            mz_tolerance=0.02,
            learn_from_missing=False,
        )
        assert "is_missing_chimeric_features" not in feature.columns
        assert feature.columns == [
            "chimeric_ion_matches",
            "chimeric_ion_match_intensity",
            "chimeric_longest_b_series",
            "chimeric_longest_y_series",
            "chimeric_complementary_ion_count",
            "chimeric_max_ion_gap",
            "chimeric_b_y_intensity_ratio",
            "chimeric_spectral_angle",
        ]

    # ------------------------------------------------------------------
    # Compute Method
    # ------------------------------------------------------------------

    def test_compute_with_none_predictions(self, chimeric_features):
        """Test that compute raises error when predictions is None."""
        metadata = pd.DataFrame({"confidence": [0.9], "precursor_charge": [2]})
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        with pytest.raises(
            ValueError,
            match="requires beam predictions, but dataset.predictions is None",
        ):
            chimeric_features.compute(dataset)

    def test_compute_raises_for_none_predictions(self, chimeric_features):
        """ChimericFeatures.compute should raise ValueError when predictions is None."""
        metadata = pd.DataFrame(
            {
                "confidence": [0.9],
                "precursor_charge": [2],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        with pytest.raises(
            ValueError, match="ChimericFeatures requires beam predictions"
        ):
            chimeric_features.compute(dataset)

    @patch("winnow.calibration.features.chimeric.koinapy.Koina")
    @patch("winnow.calibration.features.chimeric.compute_ion_identifications")
    def test_compute_preprocessing_pipeline(
        self,
        mock_compute_ions,
        mock_koina,
        chimeric_features,
        sample_dataset_with_beam_predictions,
    ):
        """Test the complete data preprocessing pipeline including groupby aggregation and sorting."""
        # Use existing fixture but take only first 2 spectra for testing
        dataset = sample_dataset_with_beam_predictions
        dataset.metadata = dataset.metadata.iloc[:2].copy()  # Keep first 2 rows
        dataset.predictions = dataset.predictions[:2]  # Keep first 2 predictions

        # Create mock predictions with proper pandas index for grouping and multiple rows per peptide
        mock_predictions = pd.DataFrame(
            {
                "peptide_sequences": ["GA", "GA", "GA", "TV", "TV"],
                "precursor_charges": [2, 2, 2, 2, 2],
                "collision_energies": [25, 25, 25, 25, 25],
                # Individual intensity values (one per ion)
                "intensities": [0.3, 0.1, 0.8, 0.4, 0.2],
                # Individual m/z values (one per ion) - unsorted to test sorting logic
                "mz": [300.0, 100.0, 250.0, 400.0, 200.0],
                # Individual annotations (one per ion)
                "annotation": ["b2+1", "b1+1", "y1+1", "b2+1", "b1+1"],
            },
            # Set pandas index so first 3 rows get index 0, last 2 get index 1
            index=[0, 0, 0, 1, 1],
        )
        mock_model = self._setup_koina_mock(mock_koina, mock_predictions)

        # Mock ion computation
        mock_match_rate = [0.5, 0.6]
        mock_match_intensity = [0.4, 0.5]
        mock_longest_b_series = 1
        mock_longest_y_series = 1
        mock_complementary_ion_count = 1
        mock_max_ion_gap = 1
        mock_b_y_intensity_ratio = 0.5
        mock_spectral_angle = 0.8
        mock_compute_ions.return_value = (
            mock_match_rate,
            mock_match_intensity,
            mock_longest_b_series,
            mock_longest_y_series,
            mock_complementary_ion_count,
            mock_max_ion_gap,
            mock_b_y_intensity_ratio,
            mock_spectral_angle,
        )

        # Run the compute method
        chimeric_features.compute(dataset)

        # Verify input preparation (runner-up sequences extracted correctly)
        call_args = mock_model.predict.call_args[0][0]
        expected_sequences = ["GA", "TV"]  # Runner-up sequences from fixture
        assert list(call_args["peptide_sequences"]) == expected_sequences
        assert list(call_args["precursor_charges"]) == [2, 2]
        assert list(call_args["collision_energies"]) == [25, 25]

        # Verify groupby aggregation AND sorting for multiple fragments per peptide
        actual_mz = dataset.metadata["runner_up_theoretical_mz"].tolist()
        actual_intensities = dataset.metadata[
            "runner_up_theoretical_intensity"
        ].tolist()

        # First peptide "GA": 3 fragments with m/z [300.0, 100.0, 250.0] -> sorted [100.0, 250.0, 300.0]
        # Second peptide "TV": 2 fragments with m/z [400.0, 200.0] -> sorted [200.0, 400.0]
        expected_sorted_mz = [
            [100.0, 250.0, 300.0],  # Sorted m/z for "GA" (3 fragments)
            [200.0, 400.0],  # Sorted m/z for "TV" (2 fragments)
        ]
        # Convert numpy arrays to lists for comparison
        actual_mz_lists = [
            arr.tolist() if hasattr(arr, "tolist") else arr for arr in actual_mz
        ]
        assert actual_mz_lists == expected_sorted_mz

        # Verify intensities were reordered to match sorted m/z
        # "GA" intensities [0.3, 0.1, 0.8] -> reordered to match sorted m/z [0.1, 0.8, 0.3]
        # "TV" intensities [0.4, 0.2] -> reordered to match sorted m/z [0.2, 0.4]
        expected_sorted_intensities = [
            [0.1, 0.8, 0.3],  # Intensities reordered for "GA"
            [0.2, 0.4],  # Intensities reordered for "TV"
        ]
        # Convert numpy arrays to lists for comparison
        actual_intensities_lists = [
            arr.tolist() if hasattr(arr, "tolist") else arr
            for arr in actual_intensities
        ]
        assert actual_intensities_lists == expected_sorted_intensities

        # Verify final features were computed correctly
        assert "chimeric_ion_matches" in dataset.metadata.columns
        assert "chimeric_ion_match_intensity" in dataset.metadata.columns
        assert "is_missing_chimeric_features" in dataset.metadata.columns
        assert list(dataset.metadata["chimeric_ion_matches"]) == [0.5, 0.6]
        assert list(dataset.metadata["chimeric_ion_match_intensity"]) == [0.4, 0.5]

    def test_compute_maps_values_to_correct_rows_and_imputes_missing(
        self,
        chimeric_features,
    ):
        """Test that computed chimeric values are mapped to correct rows and missing values are imputed correctly."""
        metadata = pd.DataFrame(
            {
                "confidence": [0.9, 0.7, 0.6],
                "precursor_charge": [2, 2, 7],  # Last one has charge > 6
                "spectrum_id": [10, 30, 40],
                "mz_array": [
                    [100.0, 200.0, 300.0],
                    [150.0, 250.0],
                    [110.0, 210.0, 310.0],
                ],
                "intensity_array": [
                    [1000.0, 2000.0, 3000.0],
                    [1500.0, 2500.0],
                    [1100.0, 2100.0, 3100.0],
                ],
            }
        )

        predictions = [
            [  # Spectrum 10: Valid - has runner-up, charge 2, len 2
                MockScoredSequence(["A", "G"], np.log(0.8)),
                MockScoredSequence(["G", "A"], np.log(0.6)),  # Runner-up
            ],
            [  # Spectrum 30: Invalid - runner-up len > 30
                MockScoredSequence(["K"], np.log(0.95)),
                MockScoredSequence(["A"] * 31, np.log(0.7)),  # Runner-up len > 30
            ],
            [  # Spectrum 40: Invalid - charge > 6
                MockScoredSequence(["S", "P"], np.log(0.9)),
                MockScoredSequence(["P", "S"], np.log(0.7)),  # Runner-up
            ],
        ]

        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)

        # Run compute with mocked Koina model
        with patch("winnow.calibration.features.chimeric.koinapy.Koina") as mock_koina:
            mock_model = mock_koina.return_value
            mock_model.model_inputs = [
                "peptide_sequences",
                "precursor_charges",
                "collision_energies",
            ]
            mock_model.predict = self._create_chimeric_koina_mock_predict()
            chimeric_features.compute(dataset)

        # Verify results
        assert "is_missing_chimeric_features" in dataset.metadata.columns
        spectrum_masks = self._assert_chimeric_valid_invalid_flags(dataset)
        self._assert_chimeric_koina_theoretical_mz_intensity(dataset, spectrum_masks)
        self._assert_chimeric_ion_matches(dataset, spectrum_masks)

        # Verify dataset structure
        assert len(dataset.metadata) == 3
        assert set(dataset.metadata["spectrum_id"].values) == {10, 30, 40}

    # ------------------------------------------------------------------
    # learn_from_missing Behaviour
    # ------------------------------------------------------------------

    @patch("winnow.calibration.features.chimeric.koinapy.Koina")
    def test_learn_from_missing_false_drops_entries_without_runnerup_and_warns(
        self, mock_koina
    ):
        """learn_from_missing=False: spectra without runner-up are removed with warning."""
        feature = ChimericFeatures(
            mz_tolerance=0.02,
            learn_from_missing=False,
            model_input_constants={"collision_energies": 25},
        )
        metadata = pd.DataFrame(
            {
                "precursor_charge": [2, 2, 2],
                "spectrum_id": [10, 20, 30],
                "mz_array": [[100.0, 200.0], [120.0], [110.0, 210.0]],
                "intensity_array": [[1000.0, 2000.0], [1200.0], [1100.0, 2100.0]],
            }
        )
        predictions = [
            [  # spectrum 10: valid (has runner-up)
                MockScoredSequence(["A", "G"], np.log(0.8)),
                MockScoredSequence(["G", "A"], np.log(0.6)),
            ],
            [  # spectrum 20: invalid (only one beam, no runner-up)
                MockScoredSequence(["V", "T"], np.log(0.9)),
            ],
            [  # spectrum 30: valid (has runner-up)
                MockScoredSequence(["S", "P"], np.log(0.7)),
                MockScoredSequence(["P", "S"], np.log(0.5)),
            ],
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)

        mock_model = mock_koina.return_value
        mock_model.model_inputs = [
            "peptide_sequences",
            "precursor_charges",
            "collision_energies",
        ]
        mock_model.predict = make_intensity_mock(
            {
                10: [(100.0, 0.8, "b1+1"), (200.0, 0.6, "y1+1")],
                30: [(110.0, 0.7, "b1+1"), (210.0, 0.5, "y1+1")],
            }
        )

        with pytest.warns(UserWarning, match="Filtered 1 spectra"):
            feature.compute(dataset)

        assert len(dataset.metadata) == 2
        assert 20 not in dataset.metadata["spectrum_id"].values
        assert "chimeric_ion_matches" in dataset.metadata.columns
        assert "chimeric_ion_match_intensity" in dataset.metadata.columns

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _create_chimeric_koina_mock_predict(self):
        """Create a mock Koina predict function for chimeric testing."""

        def mock_koina_predict(inputs_df):
            """Mock Koina predict that returns realistic predictions matching experimental m/z values."""
            if len(inputs_df) == 0:
                return pd.DataFrame(
                    columns=[
                        "peptide_sequences",
                        "precursor_charges",
                        "collision_energies",
                        "intensities",
                        "mz",
                        "annotation",
                    ]
                )

            predictions_list = []
            index_list = []

            for spectrum_id in inputs_df.index:
                peptide = inputs_df.loc[spectrum_id, "peptide_sequences"]
                mz_values, intensities, annotations = (
                    self._get_chimeric_mock_predictions_for_spectrum(spectrum_id)
                )

                for mz, intensity, annotation in zip(
                    mz_values, intensities, annotations
                ):
                    predictions_list.append(
                        {
                            "peptide_sequences": peptide,
                            "precursor_charges": inputs_df.loc[
                                spectrum_id, "precursor_charges"
                            ],
                            "collision_energies": inputs_df.loc[
                                spectrum_id, "collision_energies"
                            ],
                            "intensities": intensity,
                            "mz": mz,
                            "annotation": annotation,
                        }
                    )
                    index_list.append(spectrum_id)

            return pd.DataFrame(predictions_list, index=index_list)

        return mock_koina_predict

    def _get_chimeric_mock_predictions_for_spectrum(self, spectrum_id):
        """Get mock prediction values for a given spectrum ID in chimeric test."""
        if spectrum_id == 10:
            mz_values = [99.99, 199.99, 299.99, 150.0, 250.0, 350.0]
            intensities = [0.5, 0.7, 0.9, 0.3, 0.4, 0.2]
            annotations = ["y1+1", "y2+1", "y3+1", "b1+1", "b2+1", "b3+1"]
        else:
            mz_values = [100.0, 200.0, 300.0]
            intensities = [0.5, 0.6, 0.7]
            annotations = ["y1+1", "b1+1", "y2+1"]
        return mz_values, intensities, annotations

    def _assert_chimeric_valid_invalid_flags(self, dataset):
        """Assert valid/invalid flags for chimeric features."""
        valid_flags = ~dataset.metadata["is_missing_chimeric_features"]
        spectrum_masks = {
            10: dataset.metadata["spectrum_id"] == 10,
            30: dataset.metadata["spectrum_id"] == 30,
            40: dataset.metadata["spectrum_id"] == 40,
        }

        assert valid_flags[spectrum_masks[10]].iloc[0]  # Valid
        assert not valid_flags[spectrum_masks[30]].iloc[
            0
        ]  # Invalid (runner-up len > 30)
        assert not valid_flags[spectrum_masks[40]].iloc[0]  # Invalid (charge > 6)

        return spectrum_masks

    def _assert_chimeric_koina_theoretical_mz_intensity(self, dataset, spectrum_masks):
        """Assert runner_up_theoretical_mz and runner_up_theoretical_intensity values."""
        assert "runner_up_theoretical_mz" in dataset.metadata.columns
        assert "runner_up_theoretical_intensity" in dataset.metadata.columns

        # Valid entry
        theoretical_mz_10 = dataset.metadata[spectrum_masks[10]][
            "runner_up_theoretical_mz"
        ].iloc[0]
        assert theoretical_mz_10 is not None
        assert hasattr(theoretical_mz_10, "__iter__") or isinstance(
            theoretical_mz_10, list
        )
        mz_10_list = (
            list(theoretical_mz_10)
            if hasattr(theoretical_mz_10, "__iter__")
            else [theoretical_mz_10]
        )
        assert mz_10_list == sorted(mz_10_list)
        assert len(mz_10_list) > 0

        theoretical_intensity_10 = dataset.metadata[spectrum_masks[10]][
            "runner_up_theoretical_intensity"
        ].iloc[0]
        assert theoretical_intensity_10 is not None
        assert len(theoretical_intensity_10) == len(mz_10_list)

        # Invalid entries
        for sid in [30, 40]:
            theoretical_mz = dataset.metadata[spectrum_masks[sid]][
                "runner_up_theoretical_mz"
            ].iloc[0]
            theoretical_intensity = dataset.metadata[spectrum_masks[sid]][
                "runner_up_theoretical_intensity"
            ].iloc[0]
            assert pd.isna(theoretical_mz) or theoretical_mz is None
            assert pd.isna(theoretical_intensity) or theoretical_intensity is None

    def _assert_chimeric_ion_matches(self, dataset, spectrum_masks):
        """Assert chimeric_ion_matches and chimeric_ion_match_intensity values."""
        assert "chimeric_ion_matches" in dataset.metadata.columns
        assert "chimeric_ion_match_intensity" in dataset.metadata.columns

        # Valid entry
        ion_matches_10 = dataset.metadata[spectrum_masks[10]][
            "chimeric_ion_matches"
        ].iloc[0]
        ion_match_intensity_10 = dataset.metadata[spectrum_masks[10]][
            "chimeric_ion_match_intensity"
        ].iloc[0]

        assert not pd.isna(ion_matches_10)
        assert not pd.isna(ion_match_intensity_10)
        assert isinstance(ion_matches_10, (int, float))
        assert isinstance(ion_match_intensity_10, (int, float))
        assert ion_matches_10 > 0.0, (
            f"Expected non-zero chimeric_ion_matches for spectrum 10, got {ion_matches_10}"
        )
        assert ion_match_intensity_10 > 0.0, (
            f"Expected non-zero chimeric_ion_match_intensity for spectrum 10, got {ion_match_intensity_10}"
        )

        # Invalid entries
        for sid in [30, 40]:
            ion_matches = dataset.metadata[spectrum_masks[sid]][
                "chimeric_ion_matches"
            ].iloc[0]
            ion_match_intensity = dataset.metadata[spectrum_masks[sid]][
                "chimeric_ion_match_intensity"
            ].iloc[0]
            assert ion_matches == 0.0
            assert ion_match_intensity == 0.0

    def _setup_koina_mock(self, mock_koina, mock_predictions_df):
        """Helper method to set up Koina model mock with given predictions."""
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.model_inputs = [
            "peptide_sequences",
            "precursor_charges",
            "collision_energies",
        ]
        mock_model_instance.predict.return_value = mock_predictions_df
        return mock_model_instance
