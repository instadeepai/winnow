"""Unit tests for winnow calibration feature FragmentMatchFeatures."""

import warnings
import pytest
import pandas as pd
from unittest.mock import Mock, patch

from winnow.calibration.features.fragment_match import FragmentMatchFeatures
from winnow.datasets.calibration_dataset import CalibrationDataset
from tests.calibration.features.conftest import make_intensity_mock


class TestFragmentMatchFeatures:
    """Test the FragmentMatchFeatures class."""

    @pytest.fixture()
    def prosit_features(self):
        """Create a FragmentMatchFeatures instance for testing."""
        return FragmentMatchFeatures(
            mz_tolerance=0.02,
            unsupported_residues=["U", "O", "X"],
            model_input_constants={"collision_energies": 25},
        )

    @pytest.fixture()
    def sample_dataset_with_spectra(self):
        """Create a sample dataset with spectral data."""
        metadata = pd.DataFrame(
            {
                "confidence": [0.9, 0.8, 0.7],
                "prediction": [["A", "G"], ["G", "A"], ["S", "P"]],
                "prediction_untokenised": ["AG", "GA", "SP"],
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
        return CalibrationDataset(metadata=metadata, predictions=None)

    # ------------------------------------------------------------------
    # Basic Properties and Initialization
    # ------------------------------------------------------------------

    def test_properties(self, prosit_features):
        """Test FragmentMatchFeatures properties."""
        assert prosit_features.name == "Fragment Match Features"
        assert prosit_features.columns == [
            # Basic match metrics
            "ion_matches",
            "ion_match_intensity",
            # Ion coverage features
            "longest_b_series",
            "longest_y_series",
            "complementary_ion_count",
            "max_ion_gap",
            # Missing indicator (learn_from_missing=True by default)
            "is_missing_fragment_match_features",
        ]
        assert prosit_features.dependencies == []
        assert prosit_features.mz_tolerance == 0.02

    def test_initialization_with_tolerance(self):
        """Test initialization with custom tolerance."""
        feature = FragmentMatchFeatures(
            mz_tolerance=0.01,
            unsupported_residues=["U", "O", "X"],
            model_input_constants={"collision_energies": 25},
        )
        assert feature.mz_tolerance == 0.01
        assert feature.intensity_model_name == "Prosit_2020_intensity_HCD"
        assert feature.model_input_constants == {"collision_energies": 25}
        assert feature.model_input_columns is None

    def test_prepare_does_nothing(self, prosit_features, sample_dataset_with_spectra):
        """Test that prepare method does nothing."""
        original_metadata = sample_dataset_with_spectra.metadata.copy()
        prosit_features.prepare(sample_dataset_with_spectra)
        pd.testing.assert_frame_equal(
            sample_dataset_with_spectra.metadata, original_metadata
        )

    # ------------------------------------------------------------------
    # Column Configuration
    # ------------------------------------------------------------------

    def test_columns_include_ion_coverage_features(self):
        """Verify columns include all ion coverage features."""
        feature = FragmentMatchFeatures(mz_tolerance=0.02, learn_from_missing=True)
        columns = feature.columns

        ion_coverage_features = [
            "longest_b_series",
            "longest_y_series",
            "complementary_ion_count",
            "max_ion_gap",
        ]
        for col in ion_coverage_features:
            assert col in columns, f"Expected {col} in FragmentMatchFeatures.columns"

    def test_learn_from_missing_false_columns_excludes_indicator(self):
        """learn_from_missing=False: is_missing_fragment_match_features not in columns."""
        feature = FragmentMatchFeatures(
            mz_tolerance=0.02,
            learn_from_missing=False,
        )
        assert "is_missing_fragment_match_features" not in feature.columns
        assert feature.columns == [
            "ion_matches",
            "ion_match_intensity",
            "longest_b_series",
            "longest_y_series",
            "complementary_ion_count",
            "max_ion_gap",
        ]

    def test_learn_from_missing_true_columns_includes_indicator(self):
        """learn_from_missing=True: is_missing_fragment_match_features in columns."""
        feature = FragmentMatchFeatures(
            mz_tolerance=0.02,
            learn_from_missing=True,
        )
        assert "is_missing_fragment_match_features" in feature.columns

    # ------------------------------------------------------------------
    # Compute Method
    # ------------------------------------------------------------------

    @patch("winnow.calibration.features.fragment_match.koinapy.Koina")
    @patch("winnow.calibration.features.fragment_match.compute_ion_identifications")
    def test_compute_with_mock(
        self,
        mock_compute_ions,
        mock_koina,
        prosit_features,
        sample_dataset_with_spectra,
    ):
        """Test compute method with mocked Koina intensity model and ion computation."""
        # Mock the Koina intensity model
        mock_model_instance = Mock()
        mock_koina.return_value = mock_model_instance
        mock_model_instance.model_inputs = [
            "peptide_sequences",
            "precursor_charges",
            "collision_energies",
        ]

        # Mock the prediction result with proper pandas index for grouping and multiple rows per peptide
        mock_predictions = pd.DataFrame(
            {
                "peptide_sequences": [
                    "AG",
                    "AG",
                    "AG",
                    "AG",
                    "GA",
                    "GA",
                    "GA",
                    "GA",
                    "SP",
                    "SP",
                    "SP",
                    "SP",
                ],
                "precursor_charges": [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3],
                "collision_energies": [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25],
                # Individual intensity values (one per ion)
                "intensities": [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.15,
                    0.25,
                    0.35,
                    0.45,
                    0.12,
                    0.22,
                    0.32,
                    0.42,
                ],
                # Individual m/z values (one per ion)
                "mz": [
                    100.0,
                    200.0,
                    300.0,
                    400.0,
                    110.0,
                    210.0,
                    310.0,
                    410.0,
                    105.0,
                    205.0,
                    305.0,
                    405.0,
                ],
                # Individual annotations (one per ion)
                "annotation": [
                    "b1+1",
                    "y1+1",
                    "b2+1",
                    "y2+1",
                    "b1+2",
                    "y1+2",
                    "b2+2",
                    "y2+2",
                    "b1+3",
                    "y1+3",
                    "b2+3",
                    "y2+3",
                ],
            },
            # Set pandas index: 4 rows per peptide (3 peptides total)
            index=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
        )
        mock_model_instance.predict.return_value = mock_predictions

        # Mock ion identification computation
        mock_match_rate = [0.5, 0.6, 0.7]
        mock_match_intensity = [0.4, 0.5, 0.6]
        mock_longest_b_series = 3
        mock_longest_y_series = 1
        mock_complementary_ion_count = 1
        mock_max_ion_gap = 1
        mock_compute_ions.return_value = (
            mock_match_rate,
            mock_match_intensity,
            mock_longest_b_series,
            mock_longest_y_series,
            mock_complementary_ion_count,
            mock_max_ion_gap,
        )

        prosit_features.compute(sample_dataset_with_spectra)

        # Check that new columns were added
        assert "theoretical_mz" in sample_dataset_with_spectra.metadata.columns
        assert "theoretical_intensity" in sample_dataset_with_spectra.metadata.columns
        assert "ion_matches" in sample_dataset_with_spectra.metadata.columns
        assert "ion_match_intensity" in sample_dataset_with_spectra.metadata.columns
        assert (
            "is_missing_fragment_match_features"
            in sample_dataset_with_spectra.metadata.columns
        )

        # Check that the model was called
        mock_model_instance.predict.assert_called_once()

        # Check that ion computation was called
        mock_compute_ions.assert_called_once()

    def test_compute_maps_values_to_correct_rows_and_imputes_missing(
        self,
        prosit_features,
    ):
        """Test that computed values are mapped to correct rows and missing values are imputed correctly."""
        # Create a dataset with mixed valid/invalid predictions
        metadata = pd.DataFrame(
            {
                "confidence": [0.9, 0.7, 0.6],
                "prediction": [
                    ["A", "G"],  # Valid: len 2
                    ["A"] * 31,  # Invalid: len > 30
                    ["A", "G"],  # Valid: len 2
                ],
                "precursor_charge": [2, 2, 2],
                "spectrum_id": [10, 20, 40],
                "prediction_untokenised": ["AG", "A" * 31, "AG"],
                "mz_array": [
                    [100.0, 200.0, 300.0],
                    [120.0, 220.0],
                    [110.0, 210.0, 310.0],
                ],
                "intensity_array": [
                    [1000.0, 2000.0, 3000.0],
                    [1200.0, 2200.0],
                    [1100.0, 2100.0, 3100.0],
                ],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        # Run compute with mocked Prosit model
        with patch(
            "winnow.calibration.features.fragment_match.koinapy.Koina"
        ) as mock_koina:
            mock_model = mock_koina.return_value
            mock_model.model_inputs = [
                "peptide_sequences",
                "precursor_charges",
                "collision_energies",
            ]
            mock_model.predict = self._create_prosit_mock_predict()
            prosit_features.compute(dataset)

        # Verify results
        assert "is_missing_fragment_match_features" in dataset.metadata.columns
        spectrum_masks = self._assert_valid_invalid_flags(dataset)
        self._assert_theoretical_mz_intensity(dataset, spectrum_masks)
        self._assert_ion_matches(dataset, spectrum_masks)

        # Verify dataset structure
        assert len(dataset.metadata) == 3
        assert set(dataset.metadata["spectrum_id"].values) == {10, 20, 40}

    # ------------------------------------------------------------------
    # learn_from_missing Behaviour
    # ------------------------------------------------------------------

    @patch("winnow.calibration.features.fragment_match.koinapy.Koina")
    def test_learn_from_missing_false_drops_invalid_rows_and_warns(self, mock_koina):
        """learn_from_missing=False: invalid rows are removed and a warning is emitted."""
        feature = FragmentMatchFeatures(
            mz_tolerance=0.02,
            learn_from_missing=False,
            max_peptide_length=5,  # short limit so row 1 (len 6) is invalid
            model_input_constants={"collision_energies": 25},
        )
        metadata = pd.DataFrame(
            {
                "prediction": [["A", "G"], ["A"] * 6, ["S", "P"]],
                "precursor_charge": [2, 2, 2],
                "spectrum_id": [10, 20, 30],
                "mz_array": [[100.0, 200.0], [120.0], [110.0, 210.0]],
                "intensity_array": [[1000.0, 2000.0], [1200.0], [1100.0, 2100.0]],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

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
        assert "ion_matches" in dataset.metadata.columns
        assert "ion_match_intensity" in dataset.metadata.columns

    @patch("winnow.calibration.features.fragment_match.koinapy.Koina")
    def test_learn_from_missing_false_no_warning_when_all_valid(self, mock_koina):
        """learn_from_missing=False: no warning when all entries are valid."""
        feature = FragmentMatchFeatures(
            mz_tolerance=0.02,
            learn_from_missing=False,
            model_input_constants={"collision_energies": 25},
        )
        metadata = pd.DataFrame(
            {
                "prediction": [["A", "G"], ["S", "P"]],
                "precursor_charge": [2, 2],
                "spectrum_id": [10, 30],
                "mz_array": [[100.0, 200.0], [110.0, 210.0]],
                "intensity_array": [[1000.0, 2000.0], [1100.0, 2100.0]],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

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

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            feature.compute(dataset)

        assert len(dataset.metadata) == 2
        assert "ion_matches" in dataset.metadata.columns

    @patch("winnow.calibration.features.fragment_match.koinapy.Koina")
    def test_learn_from_missing_true_retains_invalid_rows_with_zero_features(
        self, mock_koina
    ):
        """learn_from_missing=True: invalid rows kept with zero values and indicator set."""
        feature = FragmentMatchFeatures(
            mz_tolerance=0.02,
            learn_from_missing=True,
            max_peptide_length=5,
            model_input_constants={"collision_energies": 25},
        )
        metadata = pd.DataFrame(
            {
                "prediction": [["A", "G"], ["A"] * 6, ["S", "P"]],
                "precursor_charge": [2, 2, 2],
                "spectrum_id": [10, 20, 30],
                "mz_array": [[100.0, 200.0], [120.0], [110.0, 210.0]],
                "intensity_array": [[1000.0, 2000.0], [1200.0], [1100.0, 2100.0]],
            }
        )
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

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

        feature.compute(dataset)

        assert len(dataset.metadata) == 3
        assert set(dataset.metadata["spectrum_id"].values) == {10, 20, 30}
        assert "is_missing_fragment_match_features" in dataset.metadata.columns

        row20 = dataset.metadata[dataset.metadata["spectrum_id"] == 20].iloc[0]
        assert row20["is_missing_fragment_match_features"]
        assert row20["ion_matches"] == 0.0
        assert row20["ion_match_intensity"] == 0.0

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _create_prosit_mock_predict(self):
        """Create a mock Koina intensity predict function for testing."""

        def mock_prosit_predict(inputs_df):
            """Mock Prosit predict that returns realistic predictions matching experimental m/z values."""
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
                    self._get_mock_predictions_for_spectrum(spectrum_id)
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

        return mock_prosit_predict

    def _get_mock_predictions_for_spectrum(self, spectrum_id):
        """Get mock prediction values for a given spectrum ID."""
        if spectrum_id == 10:
            mz_values = [99.99, 199.99, 299.99, 150.0, 250.0, 350.0]
            intensities = [0.5, 0.7, 0.9, 0.3, 0.4, 0.2]
            annotations = ["y1+1", "y2+1", "y3+1", "b1+1", "b2+1", "b3+1"]
        elif spectrum_id == 40:
            mz_values = [109.99, 209.99, 309.99, 160.0, 260.0, 360.0]
            intensities = [0.55, 0.75, 0.95, 0.35, 0.45, 0.25]
            annotations = ["y1+1", "y2+1", "y3+1", "b1+1", "b2+1", "b3+1"]
        else:
            mz_values = [100.0, 200.0, 300.0]
            intensities = [0.5, 0.6, 0.7]
            annotations = ["y1+1", "b1+1", "y2+1"]
        return mz_values, intensities, annotations

    def _assert_valid_invalid_flags(self, dataset):
        """Assert valid/invalid flags for fragment match features."""
        valid_flags = ~dataset.metadata["is_missing_fragment_match_features"]
        spectrum_masks = {
            10: dataset.metadata["spectrum_id"] == 10,
            20: dataset.metadata["spectrum_id"] == 20,
            40: dataset.metadata["spectrum_id"] == 40,
        }

        assert valid_flags[spectrum_masks[10]].iloc[0]  # Valid
        assert not valid_flags[spectrum_masks[20]].iloc[0]  # Invalid (has len > 30)
        assert valid_flags[spectrum_masks[40]].iloc[0]  # Valid

        return spectrum_masks

    def _assert_theoretical_mz_intensity(self, dataset, spectrum_masks):
        """Assert theoretical_mz and theoretical_intensity values."""
        assert "theoretical_mz" in dataset.metadata.columns
        assert "theoretical_intensity" in dataset.metadata.columns

        # Valid entries
        for sid in [10, 40]:
            theoretical_mz = dataset.metadata[spectrum_masks[sid]][
                "theoretical_mz"
            ].iloc[0]
            assert theoretical_mz is not None
            assert hasattr(theoretical_mz, "__iter__") or isinstance(
                theoretical_mz, list
            )
            mz_list = (
                list(theoretical_mz)
                if hasattr(theoretical_mz, "__iter__")
                else [theoretical_mz]
            )
            assert mz_list == sorted(mz_list)
            assert len(mz_list) > 0

            theoretical_intensity = dataset.metadata[spectrum_masks[sid]][
                "theoretical_intensity"
            ].iloc[0]
            assert theoretical_intensity is not None
            assert len(theoretical_intensity) == len(mz_list)

        # Invalid entries
        for sid in [20]:
            theoretical_mz = dataset.metadata[spectrum_masks[sid]][
                "theoretical_mz"
            ].iloc[0]
            theoretical_intensity = dataset.metadata[spectrum_masks[sid]][
                "theoretical_intensity"
            ].iloc[0]
            assert pd.isna(theoretical_mz) or theoretical_mz is None
            assert pd.isna(theoretical_intensity) or theoretical_intensity is None

    def _assert_ion_matches(self, dataset, spectrum_masks):
        """Assert ion_matches and ion_match_intensity values."""
        assert "ion_matches" in dataset.metadata.columns
        assert "ion_match_intensity" in dataset.metadata.columns

        # Valid entries
        for sid in [10, 40]:
            ion_matches = dataset.metadata[spectrum_masks[sid]]["ion_matches"].iloc[0]
            ion_match_intensity = dataset.metadata[spectrum_masks[sid]][
                "ion_match_intensity"
            ].iloc[0]

            assert not pd.isna(ion_matches)
            assert not pd.isna(ion_match_intensity)
            assert isinstance(ion_matches, (int, float))
            assert isinstance(ion_match_intensity, (int, float))
            assert (
                ion_matches > 0.0
            ), f"Expected non-zero ion_matches for spectrum {sid}"
            assert (
                ion_match_intensity > 0.0
            ), f"Expected non-zero ion_match_intensity for spectrum {sid}"

        # Invalid entries
        for sid in [20]:
            ion_matches = dataset.metadata[spectrum_masks[sid]]["ion_matches"].iloc[0]
            ion_match_intensity = dataset.metadata[spectrum_masks[sid]][
                "ion_match_intensity"
            ].iloc[0]
            assert ion_matches == 0.0
            assert ion_match_intensity == 0.0
