"""Unit tests for filter_dataset function in winnow.scripts.main."""

import numpy as np
import pandas as pd

from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.scripts.main import filter_dataset


class MockScoredSequence:
    """Mock class for a scored sequence."""

    def __init__(self, sequence, log_prob=0.0):
        self.sequence = sequence
        self.sequence_log_probability = log_prob


class TestFilterDataset:
    """Test the filter_dataset function."""

    def test_non_list_predictions_filtered(self):
        """Test that non-list predictions are filtered out."""
        metadata = pd.DataFrame(
            [
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },
                {
                    "prediction": "AB",
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },  # Should be filtered
            ]
        )
        predictions = [
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)
        filtered = filter_dataset(dataset)
        assert len(filtered) == 1

    def test_empty_predictions_filtered(self):
        """Test that empty predictions are filtered out."""
        metadata = pd.DataFrame(
            [
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },
                {
                    "prediction": [],
                    "prediction_untokenised": "",
                    "precursor_charge": 2,
                },  # Should be filtered
            ]
        )
        predictions = [
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)
        filtered = filter_dataset(dataset)
        assert len(filtered) == 1

    def test_missing_runner_up_filtered(self):
        """Test that entries without runner-up beam (< 2 beams) are filtered out."""
        metadata = pd.DataFrame(
            [
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },  # Should be filtered
            ]
        )
        predictions = [
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B"], np.log(0.8))
            ],  # Only one beam - should be filtered
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)
        filtered = filter_dataset(dataset)
        assert len(filtered) == 1

    def test_high_charge_filtered(self):
        """Test that spectra with charge > 6 are filtered out."""
        metadata = pd.DataFrame(
            [
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 7,
                },  # Should be filtered
            ]
        )
        predictions = [
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)
        filtered = filter_dataset(dataset)
        assert len(filtered) == 1

    def test_long_prediction_filtered(self):
        """Test that predictions with length > 30 are filtered out."""
        long_seq = ["A"] * 31
        metadata = pd.DataFrame(
            [
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },
                {
                    "prediction": long_seq,
                    "prediction_untokenised": "A" * 31,
                    "precursor_charge": 2,
                },  # Should be filtered
            ]
        )
        predictions = [
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(long_seq, np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)
        filtered = filter_dataset(dataset)
        assert len(filtered) == 1

    def test_long_runner_up_filtered(self):
        """Test that runner-up predictions with length > 30 are filtered out."""
        long_seq = ["A"] * 31
        metadata = pd.DataFrame(
            [
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },  # Should be filtered
            ]
        )
        predictions = [
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(long_seq, np.log(0.5)),
            ],  # Runner-up too long
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)
        filtered = filter_dataset(dataset)
        assert len(filtered) == 1

    def test_invalid_prosit_tokens_in_prediction_unimod(self):
        """Test that invalid Prosit tokens in prediction_untokenised (UNIMOD format) are filtered out."""
        metadata = pd.DataFrame(
            [
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },
                {
                    "prediction": ["A", "C[UNIMOD:4]", "B"],
                    "prediction_untokenised": "AC[UNIMOD:4]B",
                    "precursor_charge": 2,
                },
                {
                    "prediction": ["A", "M[UNIMOD:35]", "B"],
                    "prediction_untokenised": "AM[UNIMOD:35]B",
                    "precursor_charge": 2,
                },
                {
                    "prediction": ["A", "B", "[UNIMOD:1]"],
                    "prediction_untokenised": "AB[UNIMOD:1]",
                    "precursor_charge": 2,
                },  # Should be filtered
                {
                    "prediction": ["A", "B", "[UNIMOD:5]"],
                    "prediction_untokenised": "AB[UNIMOD:5]",
                    "precursor_charge": 2,
                },  # Should be filtered
                {
                    "prediction": ["A", "B", "Q[UNIMOD:7]"],
                    "prediction_untokenised": "ABQ[UNIMOD:7]",
                    "precursor_charge": 2,
                },  # Should be filtered
                {
                    "prediction": ["A", "B", "S[UNIMOD:21]"],
                    "prediction_untokenised": "ABS[UNIMOD:21]",
                    "precursor_charge": 2,
                },  # Should be filtered
            ]
        )
        predictions = [
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "C[UNIMOD:4]", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "M[UNIMOD:35]", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B", "[UNIMOD:1]"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B", "[UNIMOD:5]"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B", "Q[UNIMOD:7]"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B", "S[UNIMOD:21]"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)
        filtered = filter_dataset(dataset)
        assert len(filtered) == 3

    def test_invalid_prosit_tokens_in_prediction_casanovo(self):
        """Test that invalid Prosit tokens in prediction_untokenised (Casanovo format) are filtered out."""
        metadata = pd.DataFrame(
            [
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },
                {
                    "prediction": ["A", "C[Carbamidomethyl]", "B"],
                    "prediction_untokenised": "AC[Carbamidomethyl]B",
                    "precursor_charge": 2,
                },
                {
                    "prediction": ["A", "M[Oxidation]", "B"],
                    "prediction_untokenised": "AM[Oxidation]B",
                    "precursor_charge": 2,
                },
                {
                    "prediction": ["A", "C+57.021", "B"],
                    "prediction_untokenised": "AC+57.021B",
                    "precursor_charge": 2,
                },
                {
                    "prediction": ["A", "M+15.995", "B"],
                    "prediction_untokenised": "AM+15.995B",
                    "precursor_charge": 2,
                },
                {
                    "prediction": ["A", "B", "+43.006"],
                    "prediction_untokenised": "AB+43.006",
                    "precursor_charge": 2,
                },  # Should be filtered
                {
                    "prediction": ["A", "B", "Q+0.984"],
                    "prediction_untokenised": "ABQ+0.984",
                    "precursor_charge": 2,
                },  # Should be filtered
                {
                    "prediction": ["A", "B", "+42.011"],
                    "prediction_untokenised": "AB+42.011",
                    "precursor_charge": 2,
                },  # Should be filtered
                {
                    "prediction": ["A", "B", "-17.027"],
                    "prediction_untokenised": "AB-17.027",
                    "precursor_charge": 2,
                },  # Should be filtered
                {
                    "prediction": ["A", "B", "[Acetyl]-"],
                    "prediction_untokenised": "AB[Acetyl]-",
                    "precursor_charge": 2,
                },  # Should be filtered
                {
                    "prediction": ["A", "B", "[Carbamyl]-"],
                    "prediction_untokenised": "AB[Carbamyl]-",
                    "precursor_charge": 2,
                },  # Should be filtered
                {
                    "prediction": ["A", "B", "[Ammonia-loss]-"],
                    "prediction_untokenised": "AB[Ammonia-loss]-",
                    "precursor_charge": 2,
                },  # Should be filtered
                {
                    "prediction": ["A", "B", "N[Deamidated]"],
                    "prediction_untokenised": "ABN[Deamidated]",
                    "precursor_charge": 2,
                },  # Should be filtered
            ]
        )
        predictions = [
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "C[Carbamidomethyl]", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "M[Oxidation]", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "C+57.021", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "M+15.995", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B", "+43.006"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B", "Q+0.984"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B", "+42.011"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B", "-17.027"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B", "[Acetyl]-"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B", "[Carbamyl]-"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B", "[Ammonia-loss]-"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B", "N[Deamidated]"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)
        filtered = filter_dataset(dataset)
        assert len(filtered) == 5

    def test_invalid_prosit_tokens_in_runner_up_unimod(self):
        """Test that invalid Prosit tokens in runner-up prediction (UNIMOD format) are filtered out."""
        metadata = pd.DataFrame(
            [
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },  # Should be filtered
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },  # Should be filtered
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },  # Should be filtered
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },  # Should be filtered
            ]
        )
        predictions = [
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B", "[UNIMOD:1]"], np.log(0.5)),
            ],  # Runner-up has invalid token
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B", "[UNIMOD:5]"], np.log(0.5)),
            ],  # Runner-up has invalid token
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B", "Q[UNIMOD:7]"], np.log(0.5)),
            ],  # Runner-up has invalid token
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B", "S[UNIMOD:21]"], np.log(0.5)),
            ],  # Runner-up has invalid token
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)
        filtered = filter_dataset(dataset)
        assert len(filtered) == 1

    def test_invalid_prosit_tokens_in_runner_up_casanovo(self):
        """Test that invalid Prosit tokens in runner-up prediction (Casanovo format) are filtered out."""
        metadata = pd.DataFrame(
            [
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },  # Should be filtered
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },  # Should be filtered
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },  # Should be filtered
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },  # Should be filtered
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },  # Should be filtered
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },  # Should be filtered
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },  # Should be filtered
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },  # Should be filtered
            ]
        )
        predictions = [
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B", "+43.006"], np.log(0.5)),
            ],  # Runner-up has invalid token
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B", "Q+0.984"], np.log(0.5)),
            ],  # Runner-up has invalid token
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B", "+42.011"], np.log(0.5)),
            ],  # Runner-up has invalid token
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B", "-17.027"], np.log(0.5)),
            ],  # Runner-up has invalid token
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B", "[Acetyl]-"], np.log(0.5)),
            ],  # Runner-up has invalid token
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B", "[Carbamyl]-"], np.log(0.5)),
            ],  # Runner-up has invalid token
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B", "[Ammonia-loss]-"], np.log(0.5)),
            ],  # Runner-up has invalid token
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B", "N[Deamidated]"], np.log(0.5)),
            ],  # Runner-up has invalid token
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)
        filtered = filter_dataset(dataset)
        assert len(filtered) == 1

    def test_unmodified_cysteine_in_prediction_filtered(self):
        """Test that non-carbamidomethylated cysteine (exactly "C") in prediction is filtered out."""
        metadata = pd.DataFrame(
            [
                {
                    "prediction": ["A", "B", "C[UNIMOD:4]"],
                    "prediction_untokenised": "ABC[UNIMOD:4]",
                    "precursor_charge": 2,
                },
                {
                    "prediction": ["A", "B", "C"],
                    "prediction_untokenised": "ABC",
                    "precursor_charge": 2,
                },  # Should be filtered
                {
                    "prediction": ["A", "B", "C[Carbamidomethyl]"],
                    "prediction_untokenised": "ABC[Carbamidomethyl]",
                    "precursor_charge": 2,
                },
                {
                    "prediction": ["A", "B", "C[UNIMOD:4]", "C"],
                    "prediction_untokenised": "ABC[UNIMOD:4]C",
                    "precursor_charge": 2,
                },  # Should be filtered
            ]
        )
        predictions = [
            [
                MockScoredSequence(["A", "B", "C[UNIMOD:4]"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B", "C"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B", "C[Carbamidomethyl]"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B", "C[UNIMOD:4]", "C"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)
        filtered = filter_dataset(dataset)
        assert len(filtered) == 2  # Only rows with modified C should remain

    def test_unmodified_cysteine_in_runner_up_filtered(self):
        """Test that non-carbamidomethylated cysteine in runner-up prediction is filtered out."""
        metadata = pd.DataFrame(
            [
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },  # Should be filtered
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },  # Should be filtered
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },  # Should be filtered
            ]
        )
        predictions = [
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B", "C[UNIMOD:4]"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B", "C"], np.log(0.5)),
            ],  # Runner-up has unmodified C
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B", "C[Carbamidomethyl]"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B", "C[UNIMOD:4]", "C"], np.log(0.5)),
            ],  # Runner-up has unmodified C
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)
        filtered = filter_dataset(dataset)
        assert len(filtered) == 2

    def test_mixed_annotations_unimod_prediction_casanovo_untokenised(self):
        """Test mixed annotations: UNIMOD in prediction, Casanovo in prediction_untokenised."""
        metadata = pd.DataFrame(
            [
                {
                    "prediction": ["A", "C[UNIMOD:4]", "B"],
                    "prediction_untokenised": "AC[Carbamidomethyl]B",  # Mixed format
                    "precursor_charge": 2,
                },
                {
                    "prediction": ["A", "M[UNIMOD:35]", "B"],
                    "prediction_untokenised": "AM[Oxidation]B",  # Mixed format
                    "precursor_charge": 2,
                },
                {
                    "prediction": ["A", "C[UNIMOD:4]", "B"],
                    "prediction_untokenised": "AC[Carbamidomethyl]B+43.006",  # Mixed format with invalid token in untokenised
                    "precursor_charge": 2,
                },
            ]
        )
        predictions = [
            [
                MockScoredSequence(["A", "C[UNIMOD:4]", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "M[UNIMOD:35]", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "C[UNIMOD:4]", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)
        filtered = filter_dataset(dataset)
        assert (
            len(filtered) == 2
        )  # Third should be filtered due to invalid token in untokenised

    def test_empty_dataset(self):
        """Test that empty dataset is handled correctly."""
        metadata = pd.DataFrame(
            columns=["prediction", "prediction_untokenised", "precursor_charge"]
        )
        predictions = []
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)
        filtered = filter_dataset(dataset)
        assert len(filtered) == 0

    def test_all_filtered_out(self):
        """Test scenario where all entries are filtered out."""
        metadata = pd.DataFrame(
            [
                {
                    "prediction": "AB",
                    "prediction_untokenised": "AB",
                    "precursor_charge": 2,
                },  # Non-list
                {
                    "prediction": [],
                    "prediction_untokenised": "",
                    "precursor_charge": 2,
                },  # Empty
                {
                    "prediction": ["A", "B"],
                    "prediction_untokenised": "AB",
                    "precursor_charge": 7,
                },  # Charge > 6
            ]
        )
        predictions = [
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
            [
                MockScoredSequence(["A", "B"], np.log(0.8)),
                MockScoredSequence(["A", "B"], np.log(0.5)),
            ],
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)
        filtered = filter_dataset(dataset)
        assert len(filtered) == 0
