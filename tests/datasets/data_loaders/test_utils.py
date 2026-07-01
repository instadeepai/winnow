"""Tests for shared dataset loader utilities."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from instanovo.utils.metrics import Metrics
from instanovo.utils.residues import ResidueSet

from winnow.datasets.data_loaders import utils

RESIDUE_MASSES = {
    "A": 71.037114,
    "G": 57.021464,
    "P": 97.052764,
    "E": 129.042593,
    "L": 113.084064,
}
REMAPPING = {"M[Oxidation]": "M[UniMod:35]"}


@pytest.fixture()
def metrics() -> Metrics:
    return Metrics(
        residue_set=ResidueSet(
            residue_masses=RESIDUE_MASSES,
            residue_remapping=REMAPPING,
        )
    )


class TestHasGroundTruthSequenceLabels:
    """Tests for ground-truth sequence detection at load time."""

    def test_true_when_sequence_has_content(self):
        df = pl.DataFrame({"sequence": ["PEPTIDE"], "charge": [2]})
        assert utils.has_ground_truth_sequence_labels(df) is True

    def test_true_when_sequence_has_at_least_one_non_empty_value(self):
        df = pl.DataFrame({"sequence": ["PEPTIDE", "", None], "charge": [2, 3, 4]})
        assert utils.has_ground_truth_sequence_labels(df) is True
        assert len(df) == 3

    def test_false_when_sequence_column_missing(self):
        df = pl.DataFrame({"charge": [2]})
        assert utils.has_ground_truth_sequence_labels(df) is False

    def test_false_when_sequence_column_all_empty(self):
        df = pl.DataFrame({"sequence": ["", "   ", None], "charge": [2, 3, 4]})
        assert utils.has_ground_truth_sequence_labels(df) is False

    def test_load_spectrum_data_drops_empty_sequence_column(self, tmp_path):
        df = pl.DataFrame({"sequence": ["", None], "charge": [2, 3]})
        path = tmp_path / "empty_labels.parquet"
        df.write_parquet(path)

        loaded, has_labels = utils.load_spectrum_data(path)

        assert has_labels is False
        assert "sequence" not in loaded.columns


class TestIsValidPeptideTokens:
    """Validity checks apply identically to sequence and prediction cells."""

    @pytest.mark.parametrize(
        "value",
        [
            ["A", "G"],
            ("A", "G"),
            np.array(["A", "G"], dtype=object),
            pl.Series(["A", "G"]),
        ],
    )
    def test_valid_non_empty_containers(self, value: object) -> None:
        assert utils.is_valid_peptide_tokens(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            None,
            [],
            np.array([]),
            pl.Series([]),
            "",
            "PEPTIDE",
            float("nan"),
            pd.NA,
        ],
    )
    def test_invalid_values(self, value: object) -> None:
        assert utils.is_valid_peptide_tokens(value) is False

    def test_as_token_list_matches_validity(self):
        assert utils.is_valid_peptide_tokens(["A", "G"]) == (
            utils.as_token_list(["A", "G"]) is not None
        )


class TestNormalizePeptideCell:
    def test_string_splits_and_remaps(self, metrics: Metrics) -> None:
        tokens = utils.normalize_peptide_cell(
            "AG",
            metrics,
            residue_remapping=REMAPPING,
        )
        assert tokens == ["A", "G"]

    def test_leucine_mapped_at_token_level(self, metrics: Metrics) -> None:
        tokens = utils.normalize_peptide_cell(
            ["A", "L"],
            metrics,
            residue_remapping=REMAPPING,
        )
        assert tokens == ["A", "I"]

    def test_modification_remapping_on_list_input(self, metrics: Metrics) -> None:
        tokens = utils.normalize_peptide_cell(
            ["M[Oxidation]", "A"],
            metrics,
            residue_remapping=REMAPPING,
        )
        assert tokens == ["M[UniMod:35]", "A"]

    def test_require_label_false_for_absent(self, metrics: Metrics) -> None:
        assert (
            utils.normalize_peptide_cell(
                None,
                metrics,
                residue_remapping=REMAPPING,
                require_label=True,
            )
            is None
        )

    @pytest.mark.parametrize(
        "value",
        [
            np.array(["P", "E"], dtype=object),
            pl.Series(["P", "E"]),
        ],
    )
    def test_accepts_ndarray_and_polars_series(
        self, metrics: Metrics, value: object
    ) -> None:
        tokens = utils.normalize_peptide_cell(
            value,
            metrics,
            residue_remapping=REMAPPING,
        )
        assert tokens == ["P", "E"]


class TestLabelledTrainingMask:
    def test_imputes_valid_sequence_from_sequence(self):
        metadata = pd.DataFrame({"sequence": [["A"], None, ["B"]]})
        assert utils.labelled_training_mask(metadata).tolist() == [True, False, True]
        assert metadata["valid_sequence"].tolist() == [True, False, True]

    def test_uses_existing_valid_sequence_column(self):
        metadata = pd.DataFrame(
            {"valid_sequence": [True, False, True], "sequence": [["A"], None, ["B"]]}
        )
        assert utils.labelled_training_mask(metadata).tolist() == [True, False, True]

    def test_require_labelled_rows_raises_when_empty(self):
        metadata = pd.DataFrame(
            {
                "sequence": [None],
                "valid_sequence": [False],
            }
        )
        with pytest.raises(ValueError, match="valid_sequence=True"):
            utils.require_labelled_rows(metadata, context="Test context")


class TestRowEvaluation:
    def test_row_num_matches_and_correct(self, metrics: Metrics) -> None:
        assert (
            utils.row_num_matches(
                ["A", "G"],
                ["A", "G"],
                metrics,
                sequence_valid=True,
                prediction_valid=True,
            )
            == 2
        )
        assert (
            utils.row_num_matches(
                ["A", "G"],
                ["A", "G"],
                metrics,
                sequence_valid=False,
                prediction_valid=True,
            )
            == 0
        )
        assert utils.row_is_correct(
            2, ["A", "G"], ["A", "G"], sequence_valid=True, prediction_valid=True
        )
        assert not utils.row_is_correct(
            1, ["A", "G"], ["A", "G"], sequence_valid=True, prediction_valid=True
        )


class TestFinalizePeptideMetadata:
    @pytest.fixture()
    def labelled_fixture(self) -> dict[str, list]:
        return {
            "sequence": [["P", "E"], None],
            "prediction": [["P", "E"], ["P", "E"]],
        }

    def test_pandas_frame(self, metrics: Metrics, labelled_fixture: dict) -> None:
        metadata = pd.DataFrame(labelled_fixture)
        utils.finalize_peptide_metadata(
            metadata,
            metrics,
            has_labels=True,
            residue_remapping=REMAPPING,
        )
        assert metadata["valid_sequence"].tolist() == [True, False]
        assert metadata["valid_prediction"].tolist() == [True, True]
        assert metadata["correct"].tolist() == [True, False]
        assert metadata["num_matches"].tolist() == [2, 0]
        assert all(isinstance(v, list) for v in metadata["prediction"])

    def test_polars_frame(self, metrics: Metrics, labelled_fixture: dict) -> None:
        metadata = pl.DataFrame(labelled_fixture)
        result = utils.finalize_peptide_metadata(
            metadata,
            metrics,
            has_labels=True,
            residue_remapping=REMAPPING,
        )
        assert isinstance(result, pl.DataFrame)
        assert result["valid_sequence"].to_list() == [True, False]
        assert result["valid_prediction"].to_list() == [True, True]
        assert result["correct"].to_list() == [True, False]
        assert result["num_matches"].to_list() == [2, 0]

    def test_pandas_with_ndarray_cells(self, metrics: Metrics) -> None:
        metadata = pd.DataFrame(
            {
                "sequence": [np.array(["P", "E"], dtype=object)],
                "prediction": [np.array(["P", "E"], dtype=object)],
            }
        )
        utils.finalize_peptide_metadata(
            metadata,
            metrics,
            has_labels=True,
            residue_remapping=REMAPPING,
        )
        assert metadata["correct"].iloc[0]
        assert isinstance(metadata["prediction"].iloc[0], list)

    def test_empty_prediction_invalid(self, metrics: Metrics) -> None:
        metadata = pd.DataFrame(
            {
                "sequence": [["P", "E", "P"]],
                "prediction": [[]],
            }
        )
        utils.finalize_peptide_metadata(
            metadata,
            metrics,
            has_labels=True,
            residue_remapping=REMAPPING,
        )
        assert not metadata["valid_prediction"].iloc[0]
        assert metadata["num_matches"].iloc[0] == 0

    def test_unlabelled_sets_valid_prediction_only(self, metrics: Metrics) -> None:
        metadata = pd.DataFrame({"prediction": [["A", "G"]]})
        utils.finalize_peptide_metadata(
            metadata,
            metrics,
            has_labels=False,
            residue_remapping=REMAPPING,
        )
        assert metadata["valid_prediction"].iloc[0]
        assert "valid_sequence" not in metadata.columns
