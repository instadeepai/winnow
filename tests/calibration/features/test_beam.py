"""Unit tests for winnow calibration feature BeamFeatures."""

from __future__ import annotations
import pytest
import pandas as pd
import numpy as np

from winnow.calibration.features.beam import (
    BeamFeatures,
    _normalised_levenshtein,
    _normalised_levenshtein_ints,
)
from winnow.datasets.calibration_dataset import CalibrationDataset
from tests.calibration.features.conftest import MockScoredSequence


class TestNormalisedLevenshtein:
    """Unit tests for edit distance denominator and non-trivial alignments."""

    def test_both_empty_returns_zero(self):
        assert _normalised_levenshtein([], []) == pytest.approx(0.0)
        assert _normalised_levenshtein_ints([], []) == pytest.approx(0.0)

    def test_identical_sequences_zero(self):
        seq = ["G", "L", "V", "G", "S", "D", "K"]
        assert _normalised_levenshtein(seq, seq) == pytest.approx(0.0)

    def test_denominator_uses_max_length_short_vs_long(self):
        # One token vs three: 2 insertions, max(len)=3 -> 2/3
        assert _normalised_levenshtein(["A"], ["A", "B", "C"]) == pytest.approx(2 / 3)
        assert _normalised_levenshtein(["A", "B", "C"], ["A"]) == pytest.approx(2 / 3)

    def test_denominator_empty_vs_nonempty(self):
        # All insertions: distance = len(longer), ratio = 1.0
        assert _normalised_levenshtein([], ["X", "Y", "Z"]) == pytest.approx(1.0)
        assert _normalised_levenshtein(["P", "Q"], []) == pytest.approx(1.0)

    def test_denominator_asymmetric_peptide_lengths(self):
        # KE (2) vs KELEV (5): align K,K and E,E then insert L,E,V -> 3 edits / max(2,5)=5
        top = ["K", "E"]
        second = ["K", "E", "L", "E", "V"]
        assert _normalised_levenshtein(top, second) == pytest.approx(3 / 5)

    def test_multi_edit_alignment(self):
        """Several subs/ins/dels: exercise full DP (not only single substitution)."""
        # G L V A vs G L L A: one substitution at position 3 (V vs L)
        assert _normalised_levenshtein(
            ["G", "L", "V", "A"], ["G", "L", "L", "A"]
        ) == pytest.approx(0.25)

        # 7-mer with two substitutions (T↔S, D↔T); raw distance 2 → 2/7
        a = ["P", "E", "P", "T", "I", "D", "E"]
        b = ["P", "E", "P", "S", "I", "T", "E"]
        assert _normalised_levenshtein(a, b) == pytest.approx(2 / 7)

        # Three substitutions on a 7-mer → 3/7 (prefix differs, suffix matches)
        left = ["A", "A", "A", "D", "E", "F", "G"]
        right = ["B", "B", "B", "D", "E", "F", "G"]
        assert _normalised_levenshtein(left, right) == pytest.approx(3 / 7)


class TestBeamFeatures:
    """Test the BeamFeatures class."""

    @pytest.fixture()
    def beam_features(self):
        """Create a BeamFeatures instance for testing."""
        return BeamFeatures()

    @pytest.fixture()
    def sample_dataset_with_predictions(self):
        """Create a sample dataset with beam search predictions."""
        metadata = pd.DataFrame({"confidence": [0.9, 0.8, 0.7]})

        # Mock beam search results with different numbers of sequences
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
            [  # Third spectrum - 1 sequence
                MockScoredSequence(["K"], np.log(0.95))
            ],
        ]

        return CalibrationDataset(metadata=metadata, predictions=predictions)

    def test_properties(self, beam_features):
        """Test BeamFeatures properties."""
        assert beam_features.name == "Beam Features"
        assert beam_features.columns == [
            "margin",
            "median_margin",
            "entropy",
            "z-score",
            "edit_distance",
        ]
        assert beam_features.dependencies == []

    def test_prepare_does_nothing(self, beam_features, sample_dataset_with_predictions):
        """Test that prepare method does nothing."""
        original_metadata = sample_dataset_with_predictions.metadata.copy()
        beam_features.prepare(sample_dataset_with_predictions)
        pd.testing.assert_frame_equal(
            sample_dataset_with_predictions.metadata, original_metadata
        )

    def test_compute_beam_features(
        self, beam_features, sample_dataset_with_predictions
    ):
        """Test beam features computation."""
        with pytest.warns(
            UserWarning,
            match="1 beam search results have fewer than two sequences. This may affect the efficacy of computed beam features.",
        ):
            beam_features.compute(sample_dataset_with_predictions)

        # Check that all expected columns were added
        expected_columns = [
            "margin",
            "median_margin",
            "entropy",
            "z-score",
            "edit_distance",
        ]
        for col in expected_columns:
            assert col in sample_dataset_with_predictions.metadata.columns

        # Check that we have the right number of rows
        assert len(sample_dataset_with_predictions.metadata) == 3

        # Third spectrum has only one beam sequence: edit distance is 1.0
        assert sample_dataset_with_predictions.metadata.iloc[2]["edit_distance"] == 1.0

    def test_compute_with_none_predictions(self, beam_features):
        """Test that compute raises error when predictions is None."""
        metadata = pd.DataFrame({"confidence": [0.9]})
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        with pytest.raises(
            ValueError,
            match="requires beam predictions, but dataset.predictions is None",
        ):
            beam_features.compute(dataset)

    def test_compute_with_insufficient_sequences_warning(self, beam_features):
        """Test that warning is issued for beam results with < 2 sequences."""
        metadata = pd.DataFrame({"confidence": [0.9, 0.8]})
        predictions = [
            [MockScoredSequence(["A"], np.log(0.8))],  # Only 1 sequence
            [
                MockScoredSequence(["G"], np.log(0.9)),
                MockScoredSequence(["A"], np.log(0.7)),
            ],  # Only 2 sequences
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)

        with pytest.warns(
            UserWarning,
            match="1 beam search results have fewer than two sequences. This may affect the efficacy of computed beam features.",
        ):
            beam_features.compute(dataset)

        assert dataset.metadata.iloc[0]["edit_distance"] == 1.0

    def test_margin_calculation(self, beam_features):
        """Test specific margin calculation."""
        metadata = pd.DataFrame({"confidence": [0.9]})
        predictions = [
            [
                MockScoredSequence(["A"], np.log(0.8)),  # top = 0.8
                MockScoredSequence(["G"], np.log(0.6)),  # second = 0.6
            ]
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)

        beam_features.compute(dataset)

        expected_margin = 0.8 - 0.6  # top_prob - second_prob
        assert dataset.metadata.iloc[0]["margin"] == pytest.approx(
            expected_margin, rel=1e-10, abs=1e-10
        )

    def test_beam_features_with_one_sequence(self, beam_features):
        """Test beam feature calculations with single sequence."""
        metadata = pd.DataFrame({"confidence": [0.9]})
        predictions = [
            [MockScoredSequence(["A"], np.log(0.8))]  # Single sequence
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)

        with pytest.warns(
            UserWarning,
            match="1 beam search results have fewer than two sequences. This may affect the efficacy of computed beam features.",
        ):
            beam_features.compute(dataset)

        # Detailed checks for single sequence:
        # top_prob = 0.8, second_prob = 0.0 (no second sequence)
        # margin = top_prob - second_prob = 0.8 - 0.0 = 0.8
        assert dataset.metadata.iloc[0]["margin"] == pytest.approx(
            0.8, rel=1e-10, abs=1e-10
        )

        # runner_up_probs = [0.0] (no runner-ups since we start from index 1)
        # median_margin = top_prob - median([0.0]) = 0.8 - 0.0 = 0.8
        assert dataset.metadata.iloc[0]["median_margin"] == pytest.approx(
            0.8, rel=1e-10, abs=1e-10
        )

        # entropy([0.0]) should be 0
        assert dataset.metadata.iloc[0]["entropy"] == pytest.approx(
            0.0, rel=1e-10, abs=1e-10
        )

        # z-score with single value should be 0 (no variation)
        z_score = dataset.metadata.iloc[0]["z-score"]
        assert z_score == pytest.approx(
            0.0, rel=1e-10, abs=1e-10
        )  # std_prob = 0, so z-score = 0

        # No second sequence: edit distance is undefined
        assert dataset.metadata.iloc[0]["edit_distance"] == 1.0

    def test_beam_features_with_two_sequences(self, beam_features):
        """Test beam feature calculations with two sequences."""
        metadata = pd.DataFrame({"confidence": [0.9]})
        predictions = [
            [
                MockScoredSequence(["A"], np.log(0.8)),  # top
                MockScoredSequence(["G"], np.log(0.6)),  # second/runner-up
            ]
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)

        beam_features.compute(dataset)

        # Detailed checks for two sequences:
        # top_prob = 0.8, second_prob = 0.6
        # margin = 0.8 - 0.6 = 0.2
        assert dataset.metadata.iloc[0]["margin"] == pytest.approx(
            0.2, rel=1e-10, abs=1e-10
        )

        # runner_up_probs = [0.6] (from index 1 onwards)
        # median_margin = top_prob - median([0.6]) = 0.8 - 0.6 = 0.2
        assert dataset.metadata.iloc[0]["median_margin"] == pytest.approx(
            0.2, rel=1e-10, abs=1e-10
        )

        # entropy([1.0]) where runner_up_probs are normalised
        # Since there's only one runner-up, normalised prob = [1.0]
        # entropy([1.0]) = 0 (no uncertainty)
        assert dataset.metadata.iloc[0]["entropy"] == pytest.approx(
            0.0, rel=1e-10, abs=1e-10
        )

        # z-score: mean_prob = (0.8 + 0.6)/2 = 0.7, std_prob = sqrt(((0.8-0.7)^2 + (0.6-0.7)^2)/2)
        # std_prob = sqrt((0.01 + 0.01)/2) = sqrt(0.01) = 0.1
        # z_score = (0.8 - 0.7) / 0.1 = 1.0
        expected_z_score = 1.0
        assert dataset.metadata.iloc[0]["z-score"] == pytest.approx(
            expected_z_score, rel=1e-10, abs=1e-10
        )

        # edit_distance: ["A"] vs ["G"] = 1 substitution / max(1,1) = 1.0
        assert dataset.metadata.iloc[0]["edit_distance"] == pytest.approx(1.0)

    def test_beam_features_with_three_sequences(self, beam_features):
        """Test beam feature calculations with three sequences."""
        metadata = pd.DataFrame({"confidence": [0.9]})
        predictions = [
            [
                MockScoredSequence(["A"], np.log(0.7)),  # top
                MockScoredSequence(["G"], np.log(0.2)),  # second
                MockScoredSequence(["S"], np.log(0.1)),  # third
            ]
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)

        beam_features.compute(dataset)

        # Detailed checks for three sequences:
        # top_prob = 0.7, second_prob = 0.2, third_prob = 0.1
        # margin = 0.7 - 0.2 = 0.5
        assert dataset.metadata.iloc[0]["margin"] == pytest.approx(
            0.5, rel=1e-10, abs=1e-10
        )

        # runner_up_probs = [0.2, 0.1] (from index 1 onwards)
        # median_margin = top_prob - median([0.2, 0.1]) = 0.7 - 0.15 = 0.55
        assert dataset.metadata.iloc[0]["median_margin"] == pytest.approx(
            0.55, rel=1e-10, abs=1e-10
        )

        # runner_up_probs normalised: [0.2, 0.1] -> [0.2/0.3, 0.1/0.3] = [2/3, 1/3]
        # entropy([2/3, 1/3]) = 0.6365141682948128
        assert dataset.metadata.iloc[0]["entropy"] == pytest.approx(
            0.6365141682948128, rel=1e-10, abs=1e-10
        )

        # z-score calculation with three values
        # mean_prob = (0.7 + 0.2 + 0.1)/3 = 1/3
        # std_prob = sqrt(((0.7-1/3)^2 + (0.2-1/3)^2 + (0.1-1/3)^2)/3) = 0.262466929133727
        # z_score = (0.7 - 1/3) / 0.262466929133727 = 1.3970013970020956
        z_score = dataset.metadata.iloc[0]["z-score"]
        assert z_score == pytest.approx(1.3970013970020956, rel=1e-10, abs=1e-10)

        # edit_distance: ["A"] vs ["G"] = 1 substitution / max(1,1) = 1.0
        assert dataset.metadata.iloc[0]["edit_distance"] == pytest.approx(1.0)

    def test_beam_features_edit_distance_ignores_li(self, beam_features):
        """Test that edit distance treats L and I as identical."""
        metadata = pd.DataFrame({"confidence": [0.9]})
        predictions = [
            [
                MockScoredSequence(["L", "A", "G"], np.log(0.8)),
                MockScoredSequence(["I", "A", "G"], np.log(0.6)),
            ]
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)
        beam_features.compute(dataset)

        assert dataset.metadata.iloc[0]["edit_distance"] == pytest.approx(0.0)

    def test_beam_features_edit_distance_mixed(self, beam_features):
        """Test edit distance with L/I equivalence and real differences."""
        metadata = pd.DataFrame({"confidence": [0.9]})
        predictions = [
            [
                MockScoredSequence(["L", "A", "K"], np.log(0.8)),
                MockScoredSequence(["I", "G", "K"], np.log(0.6)),
            ]
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)
        beam_features.compute(dataset)

        # L->I is free (both normalised to L), A->G is 1 substitution / max(3,3) = 1/3
        assert dataset.metadata.iloc[0]["edit_distance"] == pytest.approx(1 / 3)

    def test_beam_features_edge_case_equal_probabilities(self, beam_features):
        """Test beam features when all sequences have equal probabilities."""
        metadata = pd.DataFrame({"confidence": [0.9]})
        predictions = [
            [
                MockScoredSequence(["A"], np.log(1 / 3)),
                MockScoredSequence(["G"], np.log(1 / 3)),
                MockScoredSequence(["S"], np.log(1 / 3)),
            ]
        ]
        dataset = CalibrationDataset(metadata=metadata, predictions=predictions)

        beam_features.compute(dataset)

        # When all probabilities are equal:
        # margin = 1/3 - 1/3 = 0.0
        assert dataset.metadata.iloc[0]["margin"] == 0.0

        # median_margin = 1/3 - median([1/3, 1/3]) = 1/3 - 1/3 = 0.0
        assert dataset.metadata.iloc[0]["median_margin"] == 0.0

        # entropy of [0.5, 0.5] (normalised) should be 0.6931471805599453
        assert dataset.metadata.iloc[0]["entropy"] == pytest.approx(
            0.6931471805599453, rel=1e-10, abs=1e-10
        )

        # z-score should be 0 (all values equal, so std = 0)
        assert dataset.metadata.iloc[0]["z-score"] == pytest.approx(
            0.0, rel=1e-10, abs=1e-10
        )

    def test_beam_features_raises_for_none_predictions(self, beam_features):
        """BeamFeatures.compute should raise ValueError when predictions is None."""
        metadata = pd.DataFrame({"confidence": [0.9]})
        dataset = CalibrationDataset(metadata=metadata, predictions=None)

        with pytest.raises(ValueError, match="BeamFeatures requires beam predictions"):
            beam_features.compute(dataset)
