"""Unit tests for CalibrationDataset peptide structure validation."""

from __future__ import annotations

import pandas as pd
import pytest

from winnow.datasets.calibration_dataset import CalibrationDataset


class TestCalibrationDatasetPeptideContract:
    def test_requires_prediction_column(self) -> None:
        with pytest.raises(ValueError, match="requires a 'prediction' column"):
            CalibrationDataset(metadata=pd.DataFrame({"confidence": [0.9]}))

    def test_rejects_raw_prediction_strings(self) -> None:
        with pytest.raises(ValueError, match="raw strings"):
            CalibrationDataset(
                metadata=pd.DataFrame(
                    {"confidence": [0.9], "prediction": ["AG"]},
                )
            )

    def test_normalises_empty_prediction_containers_to_none(self) -> None:
        dataset = CalibrationDataset(
            metadata=pd.DataFrame(
                {"confidence": [0.9], "prediction": [[]]},
            )
        )
        assert dataset.metadata["prediction"].tolist() == [None]
        assert dataset.metadata["valid_prediction"].tolist() == [False]

    def test_normalises_empty_sequence_containers_to_none(self) -> None:
        dataset = CalibrationDataset(
            metadata=pd.DataFrame(
                {
                    "confidence": [0.9],
                    "prediction": [["A", "G"]],
                    "sequence": [[]],
                }
            )
        )
        assert dataset.metadata["sequence"].tolist() == [None]
        assert dataset.metadata["valid_sequence"].tolist() == [False]

    def test_rejects_unsupported_prediction_values(self) -> None:
        with pytest.raises(ValueError, match="unsupported value"):
            CalibrationDataset(
                metadata=pd.DataFrame(
                    {"confidence": [0.9], "prediction": [42]},
                )
            )

    def test_accepts_token_lists_and_derives_valid_prediction(self) -> None:
        dataset = CalibrationDataset(
            metadata=pd.DataFrame(
                {
                    "confidence": [0.9, 0.8],
                    "prediction": [["A", "G"], None],
                }
            )
        )
        assert dataset.metadata["valid_prediction"].tolist() == [True, False]

    def test_derives_valid_sequence_when_labelled(self) -> None:
        dataset = CalibrationDataset(
            metadata=pd.DataFrame(
                {
                    "confidence": [0.9, 0.8],
                    "prediction": [["A", "G"], ["A", "G"]],
                    "sequence": [["A", "G"], None],
                }
            )
        )
        assert dataset.metadata["valid_sequence"].tolist() == [True, False]
        assert dataset.metadata["valid_prediction"].tolist() == [True, True]

    def test_overwrites_stale_valid_prediction_with_warning(self) -> None:
        with pytest.warns(UserWarning, match="valid_prediction"):
            dataset = CalibrationDataset(
                metadata=pd.DataFrame(
                    {
                        "confidence": [0.9],
                        "prediction": [["A", "G"]],
                        "valid_prediction": [False],
                    }
                )
            )
        assert dataset.metadata["valid_prediction"].tolist() == [True]

    def test_overwrites_stale_valid_sequence_with_warning(self) -> None:
        with pytest.warns(UserWarning, match="valid_sequence"):
            dataset = CalibrationDataset(
                metadata=pd.DataFrame(
                    {
                        "confidence": [0.9],
                        "prediction": [["A", "G"]],
                        "sequence": [["A", "G"]],
                        "valid_sequence": [False],
                    }
                )
            )
        assert dataset.metadata["valid_sequence"].tolist() == [True]
